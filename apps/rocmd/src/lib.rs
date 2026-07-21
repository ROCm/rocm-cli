// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(test)]
use rocm_core::engine_plugin_dirs;
use rocm_core::{
    AppPaths, AuditEventRecord, AutomationEventRecord, AutomationProposalRecord,
    AutomationRuntimeState, AutomationTriggerEvent, CodexBridgeEngine, CodexBridgeGpuSnapshot,
    CodexBridgeSnapshot, DEFAULT_LOCAL_HOST, ExamineSummary, ManagedServiceRecord,
    ModelRecipeArtifactRecord, RocmCliConfig, WatcherMode, WatcherRuntimeSnapshot,
    append_audit_event, append_automation_event, append_automation_proposal, builtin_watcher,
    builtin_watchers, daemon_binary_path, default_engine_for_platform, format_host_port,
    load_recent_automation_events, model_artifact_cache_status, resolve_amd_smi_binary,
    resolve_model_recipe_artifact, unix_time_millis,
};
#[cfg(test)]
use rocm_engine_protocol::EnginePluginDescriptor;
use rocm_engine_protocol::{
    EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest,
    HealthcheckResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const SERVER_TRANSIENT_STALE_MS: u128 = 5 * 60 * 1_000;
const ENDPOINT_HEALTH_TIMEOUT: Duration = Duration::from_millis(250);
const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;
const GPU_METRICS_INTERVAL_MS: u128 = 60 * 1000;
const GPU_THERMAL_HOTSPOT_PRESSURE_C: f64 = 95.0;
const GPU_THERMAL_MEMORY_PRESSURE_C: f64 = 95.0;
const GPU_MEMORY_VRAM_PRESSURE_PERCENT: f64 = 95.0;
const AMD_SMI_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const ARTIFACT_PREFETCH_TIMEOUT: Duration = Duration::from_mins(10);

#[derive(Parser, Debug)]
#[command(name = "rocmd", about = "rocm-cli local supervisor", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Run {
        #[arg(long, help = "Enable the persistent watcher loop.")]
        automations_enabled: bool,
        #[arg(
            long,
            help = "Listen on 127.0.0.1:<PORT> for local JSON POST /automation-events; never binds publicly."
        )]
        local_webhook_port: Option<u16>,
    },
    Supervise {
        service_id: String,
        #[arg(long)]
        engine: String,
        #[arg(long)]
        model_ref: String,
        #[arg(long)]
        canonical_model_id: String,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
        host: String,
        #[arg(long)]
        port: u16,
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
    Status,
    BridgeSnapshot {
        #[arg(long)]
        pretty: bool,
    },
    SandboxRun {
        #[arg(value_enum)]
        tool: SandboxToolArg,
        #[arg(long)]
        service_id: Option<String>,
        #[arg(long)]
        artifact_ref: Option<String>,
        #[arg(
            long,
            help = "Allow prefetch_artifact to perform an approved network download for direct HTTP(S) artifacts with size and sha256 metadata."
        )]
        allow_artifact_download: bool,
        #[arg(
            long,
            help = "Maximum bytes allowed for an approved artifact download."
        )]
        artifact_max_bytes: Option<u64>,
        #[arg(
            long,
            help = "Allow authenticated Hugging Face artifact downloads using ROCM_CLI_HUGGINGFACE_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN. Tokens are sent only to HTTPS Hugging Face URLs."
        )]
        allow_huggingface_download: bool,
        #[arg(long)]
        message: Option<String>,
        #[arg(
            long,
            help = "Run only the restricted internal tool API when bubblewrap isolation is unavailable; required on Windows."
        )]
        allow_native_fallback: bool,
    },
    #[command(hide = true)]
    SandboxTool {
        #[arg(value_enum)]
        tool: SandboxToolArg,
        #[arg(long)]
        service_id: Option<String>,
        #[arg(long)]
        artifact_ref: Option<String>,
        #[arg(long)]
        allow_artifact_download: bool,
        #[arg(long)]
        artifact_max_bytes: Option<u64>,
        #[arg(long)]
        allow_huggingface_download: bool,
        #[arg(long)]
        message: Option<String>,
    },
    McpServer,
    #[command(hide = true)]
    McpToolsJson,
    #[command(hide = true)]
    McpCall {
        name: String,
        #[arg(long, default_value = "{}")]
        arguments_json: String,
        #[arg(
            long,
            help = "Allow this hidden direct MCP helper to run a mutating ROCm tool call."
        )]
        allow_mutation: bool,
    },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum SandboxToolArg {
    CheckUpdates,
    DriverPlan,
    ExamineSnapshot,
    ListServers,
    RestartServer,
    StopServer,
    PrefetchArtifact,
    NotifyUser,
}

impl SandboxToolArg {
    const fn as_cli_value(self) -> &'static str {
        match self {
            Self::CheckUpdates => "check_updates",
            Self::DriverPlan => "driver_plan",
            Self::ExamineSnapshot => "examine_snapshot",
            Self::ListServers => "list_servers",
            Self::RestartServer => "restart_server",
            Self::StopServer => "stop_server",
            Self::PrefetchArtifact => "prefetch_artifact",
            Self::NotifyUser => "notify_user",
        }
    }

    #[cfg(target_os = "linux")]
    const fn writes_data(self) -> bool {
        matches!(
            self,
            Self::RestartServer | Self::StopServer | Self::NotifyUser
        )
    }

    #[cfg(target_os = "linux")]
    const fn writes_cache(self) -> bool {
        matches!(self, Self::CheckUpdates | Self::PrefetchArtifact)
    }
}

#[derive(Debug, Clone, Default)]
struct SandboxToolPolicy {
    allow_artifact_download: bool,
    artifact_max_bytes: Option<u64>,
    allow_huggingface_download: bool,
    huggingface_token: Option<String>,
}

impl SandboxToolPolicy {
    fn from_cli(
        allow_artifact_download: bool,
        artifact_max_bytes: Option<u64>,
        allow_huggingface_download: bool,
    ) -> Self {
        Self {
            allow_artifact_download,
            artifact_max_bytes,
            allow_huggingface_download,
            huggingface_token: allow_huggingface_download
                .then(resolve_huggingface_token)
                .flatten(),
        }
    }
}

fn resolve_huggingface_token() -> Option<String> {
    [
        "ROCM_CLI_HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    .iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

#[tokio::main]
pub async fn run_bin_cli() -> Result<()> {
    let cli = Cli::parse();
    run_cli(cli).await
}

pub fn run_from_args(args: Vec<OsString>) -> Result<()> {
    let cli = Cli::try_parse_from(args)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to create rocmd runtime")?;
    runtime.block_on(run_cli(cli))
}

async fn run_cli(cli: Cli) -> Result<()> {
    let paths = AppPaths::discover()?;

    match cli.command.unwrap_or(Command::Status) {
        Command::Run {
            automations_enabled,
            local_webhook_port,
        } => run_daemon(&paths, automations_enabled, local_webhook_port).await?,
        Command::Supervise {
            service_id,
            engine,
            model_ref,
            canonical_model_id,
            runtime_id,
            env_id,
            host,
            port,
            device_policy,
            gpu,
            engine_recipe_json,
        } => supervise_service(
            &paths,
            service_id,
            engine,
            model_ref,
            canonical_model_id,
            runtime_id,
            env_id,
            host,
            port,
            device_policy,
            gpu,
            engine_recipe_json,
        )?,
        Command::Status => {
            print_status(&paths)?;
        }
        Command::BridgeSnapshot { pretty } => {
            print_bridge_snapshot(&paths, pretty)?;
        }
        Command::SandboxRun {
            tool,
            service_id,
            artifact_ref,
            allow_artifact_download,
            artifact_max_bytes,
            allow_huggingface_download,
            message,
            allow_native_fallback,
        } => {
            let policy = SandboxToolPolicy::from_cli(
                allow_artifact_download,
                artifact_max_bytes,
                allow_huggingface_download,
            );
            let value = run_sandbox_runner(
                &paths,
                tool,
                service_id,
                artifact_ref,
                message,
                allow_native_fallback,
                policy,
            )?;
            print_json(&value)?;
        }
        Command::SandboxTool {
            tool,
            service_id,
            artifact_ref,
            allow_artifact_download,
            artifact_max_bytes,
            allow_huggingface_download,
            message,
        } => {
            let policy = SandboxToolPolicy::from_cli(
                allow_artifact_download,
                artifact_max_bytes,
                allow_huggingface_download,
            );
            let value = run_sandbox_tool(&paths, tool, service_id, artifact_ref, message, policy)?;
            print_json(&value)?;
        }
        Command::McpServer => {
            run_mcp_server(&paths)?;
        }
        Command::McpToolsJson => {
            print_json(&json!({ "tools": rocm_mcp_tools() }))?;
        }
        Command::McpCall {
            name,
            arguments_json,
            allow_mutation,
        } => {
            let arguments = serde_json::from_str::<Value>(&arguments_json).with_context(|| {
                format!("failed to parse --arguments-json for MCP tool `{name}`")
            })?;
            if !arguments.is_object() {
                bail!("--arguments-json for MCP tool `{name}` must be a JSON object");
            }
            ensure_direct_mcp_call_allowed(&name, allow_mutation)?;
            let result = handle_mcp_tool_call(
                &paths,
                &json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )?;
            print_json(&result)?;
        }
    }

    Ok(())
}

fn print_bridge_snapshot(paths: &AppPaths, pretty: bool) -> Result<()> {
    let snapshot = build_bridge_snapshot(paths)?;

    if pretty {
        println!(
            "{}",
            serde_json::to_string_pretty(&snapshot)
                .context("failed to serialize bridge snapshot")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string(&snapshot).context("failed to serialize bridge snapshot")?
        );
    }

    Ok(())
}

fn build_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: unix_time_millis(),
        examine: ExamineSummary::gather()?,
        gpu: gather_gpu_snapshot_for_config(&config),
        config,
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: bridge_engine_inventory(),
        services: load_managed_services(paths)?,
    })
}

fn run_sandbox_runner(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    allow_native_fallback: bool,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    #[cfg(target_os = "linux")]
    {
        if command_available("bwrap") {
            return run_bubblewrap_sandbox(paths, tool, service_id, artifact_ref, message, policy);
        }
    }

    if allow_native_fallback {
        return run_native_restricted_sandbox(
            paths,
            tool,
            service_id,
            artifact_ref,
            message,
            policy,
        );
    }

    bail!(
        "isolated sandbox runner is unavailable on this host; pass --allow-native-fallback to run only the restricted internal tool API"
    )
}

#[cfg(target_os = "linux")]
fn run_bubblewrap_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    paths.ensure()?;

    let current_exe = std::env::current_exe().context("failed to resolve rocmd executable")?;
    let exe_dir = current_exe
        .parent()
        .context("rocmd executable has no parent directory")?
        .to_path_buf();

    let mut command = ProcessCommand::new("bwrap");
    command
        .arg("--die-with-parent")
        .arg("--new-session")
        .arg("--unshare-ipc")
        .arg("--unshare-uts")
        .arg("--proc")
        .arg("/proc")
        .arg("--dev")
        .arg("/dev")
        .arg("--tmpfs")
        .arg("/tmp")
        .arg("--tmpfs")
        .arg("/run");
    if !(matches!(tool, SandboxToolArg::PrefetchArtifact) && policy.allow_artifact_download) {
        command.arg("--unshare-net");
    }

    for path in ["/usr", "/bin", "/lib", "/lib64", "/etc"] {
        let path = std::path::Path::new(path);
        if path.exists() {
            command.arg("--ro-bind").arg(path).arg(path);
        }
    }

    command.arg("--ro-bind").arg(&exe_dir).arg(&exe_dir);
    bind_app_path_for_sandbox(&mut command, &paths.config_dir, false)?;
    bind_app_path_for_sandbox(&mut command, &paths.data_dir, tool.writes_data())?;
    bind_app_path_for_sandbox(&mut command, &paths.cache_dir, tool.writes_cache())?;
    append_sandbox_tool_command_args(
        &mut command,
        &current_exe,
        tool,
        service_id.as_deref(),
        artifact_ref.as_deref(),
        message.as_deref(),
        policy,
    );

    let output = run_process_with_timeout(command, Duration::from_mins(1))?;
    let value = parse_sandbox_child_output(output, "bubblewrap sandbox")?;
    record_sandbox_audit(paths, tool, "bubblewrap", true, service_id.as_deref())?;
    Ok(sandbox_report(tool, "bubblewrap", value))
}

#[cfg(target_os = "linux")]
fn bind_app_path_for_sandbox(
    command: &mut ProcessCommand,
    path: &std::path::Path,
    writable: bool,
) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    if writable {
        command.arg("--bind");
    } else {
        command.arg("--ro-bind");
    }
    command.arg(path).arg(path);
    Ok(())
}

fn run_native_restricted_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let value = run_sandbox_tool(
        paths,
        tool,
        service_id.clone(),
        artifact_ref,
        message,
        policy,
    )?;
    record_sandbox_audit(
        paths,
        tool,
        "native_restricted",
        true,
        service_id.as_deref(),
    )?;
    Ok(sandbox_report(tool, "native_restricted", value))
}

fn run_sandbox_tool(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    match tool {
        SandboxToolArg::CheckUpdates => {
            let output = run_rocm_capture_for_paths(paths, &["update"], Duration::from_mins(1))?;
            Ok(sandbox_check_updates_value(output))
        }
        SandboxToolArg::DriverPlan => {
            let output = run_rocm_capture_for_paths(
                paths,
                &["install", "driver", "--dkms", "--dry-run"],
                Duration::from_mins(1),
            )?;
            Ok(sandbox_driver_plan_value(output))
        }
        SandboxToolArg::ExamineSnapshot => {
            let examine = ExamineSummary::gather()?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "captured",
                "mutating": false,
                "examine": examine,
            }))
        }
        SandboxToolArg::ListServers => {
            let services = load_managed_services(paths)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "listed",
                "mutating": false,
                "count": services.len(),
                "services": services,
            }))
        }
        SandboxToolArg::RestartServer => {
            let service_id = service_id.context("restart_server requires `--service-id`")?;
            let mut record = load_managed_services(paths)?
                .into_iter()
                .find(|record| record.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            restart_managed_service(paths, &mut record)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "restarted",
                "mutating": true,
                "service": record,
            }))
        }
        SandboxToolArg::StopServer => {
            let service_id = service_id.context("stop_server requires `--service-id`")?;
            let stopped = stop_managed_service(paths, &service_id)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "stopped",
                "mutating": true,
                "result": stopped,
            }))
        }
        SandboxToolArg::PrefetchArtifact => {
            let artifact_ref =
                artifact_ref.context("prefetch_artifact requires `--artifact-ref`")?;
            let resolved = resolve_model_recipe_artifact(&artifact_ref)?.with_context(|| {
                format!("artifact_ref `{artifact_ref}` was not found in the model recipe registry")
            })?;
            let (recipe, artifact) = resolved;
            prefetch_artifact_value_with_policy(
                paths,
                &artifact_ref,
                &recipe.canonical_model_id,
                artifact,
                policy,
            )
        }
        SandboxToolArg::NotifyUser => {
            let message = message.unwrap_or_else(|| "sandbox notification".to_owned());
            record_notification_audit(paths, "sandbox:notify_user", "notify_user", None, &message)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "notified",
                "mutating": false,
                "message": message,
                "notification_recorded": true,
            }))
        }
    }
}

fn record_notification_audit(
    paths: &AppPaths,
    actor: &str,
    action: &str,
    watcher_id: Option<&str>,
    message: &str,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "notification".to_owned(),
            actor: actor.to_owned(),
            level: "info".to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            watcher_id: watcher_id.map(ToOwned::to_owned),
            service_id: None,
        },
    )
}

fn prefetch_artifact_value_with_policy(
    paths: &AppPaths,
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let cache = model_artifact_cache_status(paths, model, &artifact);
    if !policy.allow_artifact_download {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "source_policy_required",
            "mutating": false,
            "network_used": false,
            "message": "artifact prefetch requires an approved source policy before network access; no artifact bytes were downloaded",
        }));
    }

    if cache.marker_path.is_file() {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "cached",
            "mutating": false,
            "network_used": false,
            "message": "artifact cache marker already exists; no network request was made",
        }));
    }

    if let Some(message) = declared_source_policy_block_message(&artifact) {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            &message,
        ));
    }
    let source_policy_requires_huggingface_auth =
        artifact_declares_huggingface_authenticated_policy(&artifact);
    let artifact_kind = artifact.kind.to_ascii_lowercase();
    let huggingface_artifact = artifact_is_huggingface(&artifact);
    if artifact.gated.unwrap_or(false) && !huggingface_artifact {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "gated artifacts require a source-specific authentication policy before download",
        ));
    }
    if !matches!(artifact_kind.as_str(), "url" | "http" | "https" | "direct")
        && !huggingface_artifact
    {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch currently supports direct HTTP(S) artifacts only",
        ));
    }
    if !artifact.uri.starts_with("https://") && !artifact.uri.starts_with("http://") {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            if huggingface_artifact {
                "authenticated Hugging Face prefetch requires an HTTP(S) artifact URI in the signed recipe metadata"
            } else {
                "approved live prefetch requires an HTTP(S) artifact URI"
            },
        ));
    }
    let mut request_headers = Vec::<(&str, String)>::new();
    let source_policy = if huggingface_artifact
        && (artifact.gated.unwrap_or(false)
            || policy.allow_huggingface_download
            || source_policy_requires_huggingface_auth)
    {
        if !policy.allow_huggingface_download {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifacts require --allow-huggingface-download before rocm-cli may use an authentication token",
            ));
        }
        if !artifact.uri.starts_with("https://") {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token over plain HTTP; use an HTTPS Hugging Face artifact URI",
            ));
        }
        if !is_huggingface_url(&artifact.uri) {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token to a non-Hugging Face URL",
            ));
        }
        let Some(token) = policy
            .huggingface_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
        else {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifact prefetch needs ROCM_CLI_HUGGINGFACE_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN",
            ));
        };
        request_headers.push(("Authorization", format!("Bearer {token}")));
        "huggingface_authenticated"
    } else {
        "explicit_allow_artifact_download"
    };
    let Some(expected_sha256) = artifact
        .sha256
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()))
    else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires a valid sha256 in the recipe artifact metadata",
        ));
    };
    let Some(size_bytes) = artifact.size_bytes else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires size_bytes in the recipe artifact metadata",
        ));
    };
    let max_bytes = policy.artifact_max_bytes.unwrap_or(size_bytes);
    if size_bytes > max_bytes {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "artifact size exceeds the approved download byte limit",
        ));
    }

    let bytes = download_artifact_bytes(&artifact.uri, max_bytes, &request_headers)?;
    if bytes.len() as u64 != size_bytes {
        bail!(
            "artifact download size mismatch for `{artifact_ref}`: expected {size_bytes} bytes, got {}",
            bytes.len()
        );
    }
    let actual_sha256 = sha256_hex(&bytes);
    if actual_sha256 != expected_sha256 {
        bail!(
            "artifact sha256 mismatch for `{artifact_ref}`: expected {expected_sha256}, got {actual_sha256}"
        );
    }

    let artifact_path = artifact_bytes_path_for_marker(&cache.marker_path);
    write_file_atomically(&artifact_path, &bytes)?;
    write_file_atomically(
        &cache.marker_path,
        &serde_json::to_vec_pretty(&json!({
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "bytes_path": artifact_path,
            "size_bytes": size_bytes,
            "sha256": actual_sha256,
            "prefetched_at_unix_ms": unix_time_millis(),
            "source_policy": source_policy,
        }))
        .context("failed to serialize artifact cache marker")?,
    )?;
    let cache = model_artifact_cache_status(paths, model, &artifact);
    Ok(json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "prefetched",
        "mutating": true,
        "network_used": true,
        "bytes_path": artifact_path,
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
        "source_policy": source_policy,
        "message": "artifact downloaded and verified with recipe sha256",
    }))
}

fn declared_source_policy_block_message(artifact: &ModelRecipeArtifactRecord) -> Option<String> {
    let source_policy = artifact.source_policy.as_ref()?;
    if !source_policy.required_hosts.is_empty() {
        let Some(host) = http_url_host(&artifact.uri) else {
            return Some(
                "recipe source policy declares required hosts but the artifact URI is not HTTP(S)"
                    .to_owned(),
            );
        };
        if !source_policy
            .required_hosts
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&host))
        {
            return Some(format!(
                "artifact host `{host}` is not allowed by the recipe source policy"
            ));
        }
    }

    match source_policy.policy.as_str() {
        "direct_https_sha256" => {
            if artifact.uri.starts_with("https://") {
                None
            } else {
                Some(
                    "recipe source policy direct_https_sha256 requires an HTTPS artifact URI"
                        .to_owned(),
                )
            }
        }
        "huggingface_public" => {
            if artifact.gated.unwrap_or(false) {
                Some(
                    "recipe source policy marks this as public Hugging Face, but the artifact is gated"
                        .to_owned(),
                )
            } else {
                huggingface_policy_uri_block_message(artifact, "huggingface_public")
            }
        }
        "huggingface_authenticated" => {
            huggingface_policy_uri_block_message(artifact, "huggingface_authenticated")
        }
        "manual_only" => Some(
            "recipe source policy marks this artifact as manual-only; rocm-cli will not download it"
                .to_owned(),
        ),
        other => Some(format!(
            "recipe source policy `{other}` is not supported by this rocm-cli build"
        )),
    }
}

fn huggingface_policy_uri_block_message(
    artifact: &ModelRecipeArtifactRecord,
    policy: &str,
) -> Option<String> {
    if !artifact.uri.starts_with("https://") {
        return Some(format!(
            "recipe source policy {policy} requires an HTTPS Hugging Face artifact URI"
        ));
    }
    if !is_huggingface_url(&artifact.uri) {
        return Some(format!(
            "recipe source policy {policy} requires a Hugging Face artifact URI"
        ));
    }
    None
}

fn artifact_declares_huggingface_authenticated_policy(
    artifact: &ModelRecipeArtifactRecord,
) -> bool {
    artifact
        .source_policy
        .as_ref()
        .is_some_and(|policy| policy.policy == "huggingface_authenticated")
}

fn artifact_is_huggingface(artifact: &ModelRecipeArtifactRecord) -> bool {
    let kind = artifact.kind.to_ascii_lowercase();
    matches!(kind.as_str(), "huggingface" | "hf" | "hugging_face")
        || is_huggingface_url(&artifact.uri)
}

fn is_huggingface_url(url: &str) -> bool {
    http_url_host(url).is_some_and(|host| {
        host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
    })
}

fn http_url_host(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = authority
        .strip_prefix('[')
        .and_then(|value| value.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| authority.split(':').next().unwrap_or_default())
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn prefetch_blocked_value(
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    cache: rocm_core::ModelArtifactCacheStatus,
    reason: &str,
) -> Value {
    json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "blocked",
        "mutating": false,
        "network_used": false,
        "message": reason,
    })
}

fn artifact_bytes_path_for_marker(marker_path: &Path) -> PathBuf {
    marker_path.with_extension("bin")
}

fn download_artifact_bytes(
    url: &str,
    max_bytes: u64,
    headers: &[(&str, String)],
) -> Result<Vec<u8>> {
    let agent = ureq::AgentBuilder::new()
        .timeout(ARTIFACT_PREFETCH_TIMEOUT)
        .build();
    let mut request = agent.get(url).set("User-Agent", "rocm-cli");
    for (name, value) in headers {
        request = request.set(name, value);
    }
    let response = request.call().map_err(|error| match error {
        ureq::Error::Status(status, _) => anyhow::anyhow!("HTTP {status} while fetching {url}"),
        other @ ureq::Error::Transport(_) => {
            anyhow::anyhow!("HTTP request failed for {url}: {other}")
        }
    })?;
    let mut reader = response.into_reader().take(max_bytes.saturating_add(1));
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read artifact response for {url}"))?;
    if bytes.len() as u64 > max_bytes {
        bail!("artifact download exceeded approved byte limit of {max_bytes}");
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    digest.iter().fold(String::new(), |mut acc, byte| {
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let tmp = path.with_extension(format!("tmp-{}", unix_time_millis()));
    fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path).or_else(|_| {
        let _ = fs::remove_file(path);
        fs::rename(&tmp, path)
    })?;
    Ok(())
}

fn sandbox_check_updates_value(output: CommandCapture) -> Value {
    let status = update_check_status(&output);
    let update_available =
        output.exit_status == 0 && update_output_reports_update_available(&output.stdout);
    let message = update_check_message(status);
    json!({
        "tool": SandboxToolArg::CheckUpdates.as_cli_value(),
        "status": status,
        "update_available": update_available,
        "mutating": false,
        "message": message,
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn update_check_status(output: &CommandCapture) -> &'static str {
    if output.exit_status != 0 {
        "error"
    } else if update_output_reports_update_available(&output.stdout) {
        "update_available"
    } else {
        "checked"
    }
}

fn update_output_reports_update_available(stdout: &str) -> bool {
    stdout
        .split_whitespace()
        .any(|part| part == "status=update_available" || part == "update_available=true")
}

fn update_check_message(status: &str) -> &'static str {
    match status {
        "update_available" => {
            "ran read-only `rocm update`; a ROCm runtime update is available; no updates were applied"
        }
        "error" => "read-only `rocm update` failed; no updates were applied",
        _ => "ran read-only `rocm update`; no updates were applied",
    }
}

fn sandbox_driver_plan_value(output: CommandCapture) -> Value {
    let status = if output.exit_status == 0 {
        "planned"
    } else {
        "error"
    };
    json!({
        "tool": SandboxToolArg::DriverPlan.as_cli_value(),
        "status": status,
        "mutating": false,
        "message": "ran read-only `rocm install driver --dkms --dry-run`; no driver commands were executed",
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn sandbox_report(tool: SandboxToolArg, isolation: &str, output: Value) -> Value {
    json!({
        "protocol": "rocmd-sandbox-run-v0",
        "tool": tool.as_cli_value(),
        "isolation": isolation,
        "ok": true,
        "ok_meaning": "sandbox wrapper completed; inspect output.status and output.exit_status for the restricted tool result",
        "output": output,
    })
}

fn record_sandbox_audit(
    paths: &AppPaths,
    tool: SandboxToolArg,
    isolation: &str,
    ok: bool,
    service_id: Option<&str>,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "sandbox".to_owned(),
            actor: "sandbox-runner".to_owned(),
            level: if ok { "info" } else { "error" }.to_owned(),
            action: tool.as_cli_value().to_owned(),
            message: format!(
                "sandbox tool `{}` completed with isolation `{isolation}`",
                tool.as_cli_value()
            ),
            watcher_id: None,
            service_id: service_id.map(str::to_owned),
        },
    )
}

#[cfg(target_os = "linux")]
fn append_sandbox_tool_command_args(
    command: &mut ProcessCommand,
    rocmd_binary: &std::path::Path,
    tool: SandboxToolArg,
    service_id: Option<&str>,
    artifact_ref: Option<&str>,
    message: Option<&str>,
    policy: SandboxToolPolicy,
) {
    command
        .arg("--")
        .arg(rocmd_binary)
        .arg("sandbox-tool")
        .arg(tool.as_cli_value());
    if let Some(service_id) = service_id {
        command.arg("--service-id").arg(service_id);
    }
    if let Some(artifact_ref) = artifact_ref {
        command.arg("--artifact-ref").arg(artifact_ref);
    }
    if policy.allow_artifact_download {
        command.arg("--allow-artifact-download");
    }
    if policy.allow_huggingface_download {
        command.arg("--allow-huggingface-download");
    }
    if let Some(max_bytes) = policy.artifact_max_bytes {
        command
            .arg("--artifact-max-bytes")
            .arg(max_bytes.to_string());
    }
    if let Some(message) = message {
        command.arg("--message").arg(message);
    }
}

#[cfg(target_os = "linux")]
fn run_process_with_timeout(
    mut command: ProcessCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn sandbox process")?;
    let started = std::time::Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll sandbox process")?
            .is_some()
        {
            return child
                .wait_with_output()
                .context("failed to collect sandbox process output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .context("failed to collect timed-out sandbox process output")?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "sandbox process exceeded {}s timeout: {}",
                timeout.as_secs(),
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    "no output".to_owned()
                }
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(target_os = "linux")]
fn parse_sandbox_child_output(output: std::process::Output, label: &str) -> Result<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        bail!(
            "{label} failed: {}",
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }
    serde_json::from_str(&stdout).with_context(|| format!("failed to parse {label} json output"))
}

#[cfg(target_os = "linux")]
fn command_available(name: &str) -> bool {
    ProcessCommand::new(name)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn gather_gpu_snapshot() -> CodexBridgeGpuSnapshot {
    let static_snapshot = match capture_amd_smi_json(&["static", "-a", "-g", "all", "--json"]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: false,
                static_snapshot: None,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    let monitor_snapshot = match capture_amd_smi_json(&[
        "monitor", "-p", "-t", "-u", "-m", "-v", "-g", "all", "--json",
    ]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    CodexBridgeGpuSnapshot {
        amd_smi_available: true,
        static_snapshot,
        monitor_snapshot,
        note: None,
    }
}

fn gather_gpu_snapshot_for_config(config: &RocmCliConfig) -> CodexBridgeGpuSnapshot {
    if config.telemetry.local_inspection_enabled() {
        gather_gpu_snapshot()
    } else {
        CodexBridgeGpuSnapshot {
            amd_smi_available: false,
            static_snapshot: None,
            monitor_snapshot: None,
            note: Some(
                "gpu telemetry is disabled by rocm-cli config; no external reporting is implemented"
                    .to_owned(),
            ),
        }
    }
}

fn capture_amd_smi_json(args: &[&str]) -> Result<Value> {
    let amd_smi_binary = resolve_amd_smi_binary();
    let mut command = ProcessCommand::new(&amd_smi_binary);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_command_with_timeout(command, AMD_SMI_PROBE_TIMEOUT)
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        anyhow::bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
}

fn bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    let current_exe = std::env::current_exe().ok();
    rocmd_engine_inventory()
        .iter()
        .map(|(id, summary)| CodexBridgeEngine {
            id: (*id).to_owned(),
            summary: (*summary).to_owned(),
            default_for_platform: *id == default_engine,
            installed_binary: true,
            binary_path: current_exe.as_ref().map(|path| path.display().to_string()),
        })
        .collect()
}

const fn rocmd_engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "lemonade",
            "embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
    ]
}

#[cfg(test)]
fn find_engine_plugin_binary<I, P>(engine: &str, plugin_dirs: I) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
}

fn run_mcp_server(paths: &AppPaths) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let message: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {error}"),
                        }
                    }),
                )?;
                continue;
            }
        };

        let Some(method) = message.get("method").and_then(Value::as_str) else {
            if message.get("id").is_some() {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": message.get("id").cloned().unwrap_or(Value::Null),
                        "error": {
                            "code": -32600,
                            "message": "invalid request: missing method",
                        }
                    }),
                )?;
            }
            continue;
        };

        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "initialize" => {
                let protocol_version = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2025-03-26");
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "protocolVersion": protocol_version,
                                "capabilities": {
                                    "tools": {
                                        "listChanged": true,
                                    }
                                },
                                "serverInfo": {
                                    "name": "rocmd-mcp-server",
                                    "title": "ROCm AI Command Center",
                                    "version": env!("CARGO_PKG_VERSION"),
                                }
                            }
                        }),
                    )?;
                }
            }
            "ping" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {}
                        }),
                    )?;
                }
            }
            "notifications/initialized" => {}
            "tools/list" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "tools": rocm_mcp_tools(),
                                "nextCursor": Value::Null,
                            }
                        }),
                    )?;
                }
            }
            "tools/call" => {
                if let Some(id) = id {
                    let result = match handle_mcp_tool_call(paths, &params) {
                        Ok(result) => result,
                        Err(error) => tool_error(
                            format!("ROCm MCP tool call failed: {error:#}"),
                            json!({
                                "tool": params.get("name").cloned().unwrap_or(Value::Null),
                            }),
                        ),
                    };
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": result,
                        }),
                    )?;
                }
            }
            notification if notification.starts_with("notifications/") => {}
            other => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": format!("method not found: {other}"),
                            }
                        }),
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn write_json_line(writer: &mut impl Write, value: &Value) -> Result<()> {
    writer.write_all(serde_json::to_string(value)?.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).context("failed to serialize json output")?
    );
    Ok(())
}

fn rocm_mcp_tools() -> Vec<Value> {
    vec![
        rocm_mcp_tool(
            "examine",
            "Read the current ROCm AI Command Center host summary.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "bridge_snapshot",
            "Read the full ROCm bridge snapshot including examine data, engines, services, automations, and gpu telemetry.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "gpu_snapshot",
            "Read the current amd-smi GPU telemetry snapshot if available.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "engines",
            "List available ROCm serving engines and whether each one is installed.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "services",
            "List managed model services and their current status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "service_logs",
            "Read the tail of a managed service log file.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "automations",
            "List automation runtime status, watcher events, and local webhook events.",
            json!({
                "type": "object",
                "properties": {
                    "event_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "natural_language_plan",
            "Ask `rocm` to translate a natural-language ROCm request into a visible plan without executing privileged work.",
            json!({
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string"
                    }
                },
                "required": ["request"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "rocm_command",
            "Run a supported read-only ROCm CLI command with argv-style arguments. Commands that change ROCm state are rejected here and must go through the ROCm CLI approval UI.",
            json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "maxItems": 64
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["args"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "update_check",
            "Run `rocm update` and return the current TheRock update status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk_dry_run",
            "Run a dry-run TheRock SDK install plan.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["wheel", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "build_date": {
                        "type": "string"
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk",
            "Install a TheRock SDK into the managed runtime area or an explicitly approved prefix.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["wheel", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "build_date": {
                        "type": "string"
                    },
                    "allow_system_prefix": {
                        "type": "boolean"
                    }
                },
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "install_engine",
            "Install or refresh a managed serving engine environment.",
            json!({
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "python_version": {
                        "type": "string"
                    },
                    "reinstall": {
                        "type": "boolean"
                    }
                },
                "required": ["engine"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "launch_server",
            "Launch a managed local model server through `rocm serve --managed`.",
            json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string"
                    },
                    "engine": {
                        "type": "string"
                    },
                    "device": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "env_id": {
                        "type": "string"
                    },
                    "host": {
                        "type": "string"
                    },
                    "port": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 65535
                    },
                    "allow_public_bind": {
                        "type": "boolean"
                    }
                },
                "required": ["model"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "stop_server",
            "Stop a managed service by service id and update its manifest status.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "watcher_enable",
            "Enable a watcher and optionally set its mode.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["observe", "propose", "contained"]
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "watcher_disable",
            "Disable a watcher.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
    ]
}

fn rocm_mcp_tool(
    name: &str,
    description: &str,
    input_schema: Value,
    read_only: bool,
    destructive: bool,
) -> Value {
    json!({
        "name": name,
        "title": name.replace('_', " "),
        "description": description,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "openWorldHint": false,
        },
        "inputSchema": input_schema,
    })
}

fn mcp_tool_requires_direct_approval(name: &str) -> bool {
    matches!(
        name,
        "install_sdk"
            | "install_engine"
            | "launch_server"
            | "stop_server"
            | "watcher_enable"
            | "watcher_disable"
    )
}

fn ensure_direct_mcp_call_allowed(name: &str, allow_mutation: bool) -> Result<()> {
    if mcp_tool_requires_direct_approval(name) && !allow_mutation {
        bail!(
            "MCP tool `{name}` changes local ROCm state; rerun `rocmd mcp-call {name}` with --allow-mutation only after an explicit user approval"
        );
    }
    Ok(())
}

fn handle_mcp_tool_call(paths: &AppPaths, params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = params
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    match name {
        "examine" => {
            let examine = ExamineSummary::gather()?;
            let output = run_rocm_capture(&["examine"])?;
            let text = command_capture_text(&output);
            if output.exit_status == 0 {
                Ok(tool_success(text, json!(examine)))
            } else {
                Ok(tool_error(
                    text,
                    json!({
                        "examine": examine,
                        "argv": output.argv,
                        "exit_status": output.exit_status,
                        "stderr": output.stderr,
                    }),
                ))
            }
        }
        "bridge_snapshot" => {
            let snapshot = build_bridge_snapshot(paths)?;
            Ok(tool_success(
                format!(
                    "Captured bridge snapshot for {} / {} with default engine `{}`.",
                    snapshot.examine.os, snapshot.examine.arch, snapshot.examine.default_engine
                ),
                json!(snapshot),
            ))
        }
        "gpu_snapshot" => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let gpu = gather_gpu_snapshot_for_config(&config);
            let status = if !config.telemetry.local_inspection_enabled() {
                "GPU telemetry is disabled by rocm-cli config."
            } else if gpu.amd_smi_available {
                "Captured amd-smi GPU snapshot."
            } else {
                "amd-smi is unavailable on this host."
            };
            Ok(tool_success(status.to_owned(), json!(gpu)))
        }
        "engines" => {
            let engines = bridge_engine_inventory();
            Ok(tool_success(
                format!("Found {} engine entries.", engines.len()),
                json!({ "engines": engines }),
            ))
        }
        "services" => {
            let services = load_managed_services(paths)?;
            Ok(tool_success(
                format!("Found {} managed services.", services.len()),
                json!({ "services": services }),
            ))
        }
        "service_logs" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("service_logs requires `service_id`")?;
            let lines = arguments
                .get("lines")
                .and_then(Value::as_u64)
                .unwrap_or(80)
                .clamp(1, 500) as usize;
            let record = load_managed_services(paths)?
                .into_iter()
                .find(|service| service.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            let tail = read_tail_lines(&record.log_path, lines)?;
            Ok(tool_success(
                format!(
                    "Read the last {} line(s) from service `{}`.",
                    lines, record.service_id
                ),
                json!({
                    "service": record,
                    "lines": lines,
                    "tail": tail,
                }),
            ))
        }
        "automations" => {
            let event_limit = arguments
                .get("event_limit")
                .and_then(Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 64) as usize;
            let runtime = AutomationRuntimeState::load(paths)?;
            let events = load_recent_automation_events(paths, event_limit)?;
            Ok(tool_success(
                format!(
                    "Loaded automation runtime and {} recent events.",
                    events.len()
                ),
                json!({
                    "runtime": runtime,
                    "recent_events": events,
                }),
            ))
        }
        "natural_language_plan" => {
            let request = arguments
                .get("request")
                .and_then(Value::as_str)
                .context("natural_language_plan requires `request`")?;
            let output = run_rocm_capture(&[request])?;
            Ok(tool_result_from_command(
                "Ran natural-language planning through `rocm`.",
                output,
                false,
            ))
        }
        "rocm_command" => {
            let argv = normalized_rocm_command_args(&arguments)?;
            ensure_rocm_command_is_read_only(&argv)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran read-only `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output = run_rocm_capture(&["update"])?;
            Ok(tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let argv = build_install_sdk_args(&arguments, true)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "install_sdk" => {
            let argv = build_install_sdk_args(&arguments, false)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk`.",
                output,
                false,
            ))
        }
        "install_engine" => {
            let argv = build_install_engine_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm engines install`.",
                output,
                false,
            ))
        }
        "launch_server" => {
            let argv = build_launch_server_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm serve --managed`.",
                output,
                false,
            ))
        }
        "stop_server" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("stop_server requires `service_id`")?;
            let stopped = stop_managed_service(paths, service_id)?;
            Ok(tool_success(
                format!("Stopped managed service `{service_id}`."),
                stopped,
            ))
        }
        "watcher_enable" => {
            let argv = build_watcher_enable_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations enable`.",
                output,
                false,
            ))
        }
        "watcher_disable" => {
            let watcher = arguments
                .get("watcher")
                .and_then(Value::as_str)
                .context("watcher_disable requires `watcher`")?;
            let argv = [
                "automations".to_owned(),
                "disable".to_owned(),
                watcher.to_owned(),
            ];
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations disable`.",
                output,
                false,
            ))
        }
        other => Ok(tool_error(
            format!("Unknown ROCm MCP tool `{other}`."),
            json!({ "tool": other }),
        )),
    }
}

fn tool_success(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": false,
    })
}

fn tool_error(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": true,
    })
}

fn tool_result_from_command(prefix: &str, output: CommandCapture, is_error: bool) -> Value {
    let text = format!("{prefix}\n\n{}", command_capture_text(&output));
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": {
            "argv": output.argv,
            "exit_status": output.exit_status,
            "stdout": output.stdout,
            "stderr": output.stderr,
        },
        "isError": is_error || output.exit_status != 0,
    })
}

fn command_capture_text(output: &CommandCapture) -> String {
    if output.stderr.trim().is_empty() {
        output.stdout.trim().to_owned()
    } else if output.stdout.trim().is_empty() {
        format!("stderr:\n{}", output.stderr.trim())
    } else {
        format!(
            "stdout:\n{}\n\nstderr:\n{}",
            output.stdout.trim(),
            output.stderr.trim()
        )
    }
}

#[derive(Debug)]
struct CommandCapture {
    argv: Vec<String>,
    exit_status: i32,
    stdout: String,
    stderr: String,
}

fn run_rocm_capture(args: &[&str]) -> Result<CommandCapture> {
    let paths = AppPaths::discover()?;
    run_rocm_capture_for_paths(&paths, args, Duration::from_mins(2))
}

fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    let rocm_binary = rocm_core::daemon_binary_path()?;
    let mut command = ProcessCommand::new(&rocm_binary);
    command
        .args(args)
        .env("ROCM_CLI_CONFIG_DIR", &paths.config_dir)
        .env("ROCM_CLI_DATA_DIR", &paths.data_dir)
        .env("ROCM_CLI_CACHE_DIR", &paths.cache_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_command_with_timeout(command, timeout)
        .with_context(|| format!("failed to run {}", rocm_binary.display()))?;
    Ok(CommandCapture {
        argv: std::iter::once(rocm_binary.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        exit_status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn run_command_with_timeout(
    mut command: ProcessCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    let mut child = command.spawn().context("failed to spawn child process")?;
    let started = std::time::Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll child process")?
            .is_some()
        {
            return child
                .wait_with_output()
                .context("failed to collect child process output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .context("failed to collect timed-out child process output")?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "process exceeded {}s timeout: {}",
                timeout.as_secs(),
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    "no output".to_owned()
                }
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn read_tail_lines(path: &std::path::Path, limit: usize) -> Result<String> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = VecDeque::with_capacity(limit);
    for line in content.lines() {
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line.to_owned());
    }
    Ok(lines.into_iter().collect::<Vec<_>>().join("\n"))
}

fn normalized_rocm_command_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let values = arguments
        .get("args")
        .and_then(Value::as_array)
        .context("rocm_command requires `args`")?;
    if values.is_empty() || values.len() > 64 {
        bail!("rocm_command `args` must contain 1 to 64 strings");
    }
    let mut args = Vec::with_capacity(values.len());
    for value in values {
        let arg = value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context("rocm_command `args` entries must be non-empty strings")?;
        if arg.contains('\0') || arg.contains('\n') || arg.contains('\r') {
            bail!("rocm_command arguments must not contain control characters");
        }
        if arg.len() > 512 {
            bail!("rocm_command argument is too long");
        }
        args.push(arg.to_owned());
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("rocm"))
    {
        args.remove(0);
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfy"))
    {
        args[0] = "comfyui".to_owned();
    }
    if args.is_empty() {
        bail!("rocm_command args should omit the leading `rocm` program name");
    }
    Ok(args)
}

fn ensure_rocm_command_is_read_only(args: &[String]) -> Result<()> {
    let first = args.first().map(|value| value.to_ascii_lowercase());
    let second = args.get(1).map(|value| value.to_ascii_lowercase());
    let read_only = match first.as_deref() {
        Some("examine" | "version" | "model" | "models" | "daemon" | "logs") => true,
        Some("update") => !args.iter().any(|arg| arg == "--apply"),
        Some("runtimes") => second.as_deref().is_none_or(|value| value == "list"),
        Some("engines") => second.as_deref().is_some_and(|value| value == "list"),
        Some("services") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "list" | "logs")),
        Some("automations") => second.as_deref().is_none_or(|value| value == "list"),
        Some("config") => second.as_deref() == Some("show"),
        Some("comfyui") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "status" | "logs" | "log")),
        Some("uninstall") => args.iter().any(|arg| arg == "--dry-run"),
        // `setup status` reports first-time setup state (read-only); `setup reset`
        // re-arms it and is mutating. Mirrors the bin's rocm_command classifier so
        // the read-only allowlist is consistent across binaries.
        Some("setup") => second.as_deref().is_none_or(|value| value == "status"),
        _ => false,
    };
    if read_only {
        return Ok(());
    }
    bail!(
        "rocm_command changes local ROCm state or is unsupported here; request it through the ROCm CLI approval UI instead"
    )
}

fn build_install_sdk_args(
    arguments: &serde_json::Map<String, Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = arguments
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or("release");
    let format = arguments
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wheel");
    let prefix = arguments.get("prefix").and_then(Value::as_str);
    let version = arguments.get("version").and_then(Value::as_str);
    let build_date = arguments.get("build_date").and_then(Value::as_str);
    let allow_system_prefix = arguments
        .get("allow_system_prefix")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if version.is_some() && build_date.is_some() {
        bail!("install_sdk accepts either `version` or `build_date`, not both");
    }

    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel.to_owned(),
        "--format".to_owned(),
        format.to_owned(),
    ];
    if let Some(prefix) = prefix {
        let prefix_path = std::path::Path::new(prefix);
        if system_prefix_requires_ack(prefix_path) && !allow_system_prefix {
            bail!(
                "install_sdk prefix `{}` is outside the user home; require `allow_system_prefix=true` before using system paths",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix.to_owned());
    }
    if let Some(version) = version {
        if version.trim().is_empty() {
            bail!("install_sdk `version` cannot be empty");
        }
        argv.push("--version".to_owned());
        argv.push(version.to_owned());
    }
    if let Some(build_date) = build_date {
        if build_date.trim().is_empty() {
            bail!("install_sdk `build_date` cannot be empty");
        }
        argv.push("--build-date".to_owned());
        argv.push(build_date.to_owned());
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    }
    Ok(argv)
}

fn build_install_engine_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let engine = arguments
        .get("engine")
        .and_then(Value::as_str)
        .context("install_engine requires `engine`")?;
    let runtime_id = arguments
        .get("runtime_id")
        .and_then(Value::as_str)
        .unwrap_or("therock-release");
    let python_version = arguments.get("python_version").and_then(Value::as_str);
    let reinstall = arguments
        .get("reinstall")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut argv = vec![
        "engines".to_owned(),
        "install".to_owned(),
        engine.to_owned(),
        "--runtime-id".to_owned(),
        runtime_id.to_owned(),
    ];
    if let Some(python_version) = python_version {
        argv.push("--python-version".to_owned());
        argv.push(python_version.to_owned());
    }
    if reinstall {
        argv.push("--reinstall".to_owned());
    }
    Ok(argv)
}

fn build_launch_server_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let model = arguments
        .get("model")
        .and_then(Value::as_str)
        .context("launch_server requires `model`")?;
    let host = arguments
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_LOCAL_HOST);
    let allow_public_bind = arguments
        .get("allow_public_bind")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "launch_server host `{host}` is not loopback; require `allow_public_bind=true` before binding a non-local interface"
        );
    }

    let mut argv = vec!["serve".to_owned(), model.to_owned(), "--managed".to_owned()];
    if let Some(engine) = arguments.get("engine").and_then(Value::as_str) {
        argv.push("--engine".to_owned());
        argv.push(engine.to_owned());
    }
    if let Some(device) = arguments.get("device").and_then(Value::as_str) {
        argv.push("--device".to_owned());
        argv.push(device.to_owned());
    }
    if let Some(runtime_id) = arguments.get("runtime_id").and_then(Value::as_str) {
        argv.push("--runtime-id".to_owned());
        argv.push(runtime_id.to_owned());
    }
    if let Some(env_id) = arguments.get("env_id").and_then(Value::as_str) {
        argv.push("--env-id".to_owned());
        argv.push(env_id.to_owned());
    }
    argv.push("--host".to_owned());
    argv.push(host.to_owned());
    if allow_public_bind {
        argv.push("--allow-public-bind".to_owned());
    }
    if let Some(port) = arguments.get("port").and_then(Value::as_u64) {
        argv.push("--port".to_owned());
        argv.push(port.to_string());
    }
    Ok(argv)
}

fn build_watcher_enable_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let watcher = arguments
        .get("watcher")
        .and_then(Value::as_str)
        .context("watcher_enable requires `watcher`")?;
    let mut argv = vec![
        "automations".to_owned(),
        "enable".to_owned(),
        watcher.to_owned(),
    ];
    if let Some(mode) = arguments.get("mode").and_then(Value::as_str) {
        argv.push("--mode".to_owned());
        argv.push(mode.to_owned());
    }
    Ok(argv)
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn system_prefix_requires_ack(prefix: &std::path::Path) -> bool {
    match rocm_core::runtime_home_dir() {
        Some(home) => !rocm_core::runtime_path_is_same_or_inside(prefix, &home),
        None => true,
    }
}

fn stop_managed_service(paths: &AppPaths, service_id: &str) -> Result<Value> {
    let mut record = load_managed_services(paths)?
        .into_iter()
        .find(|record| record.service_id == service_id)
        .with_context(|| format!("managed service `{service_id}` not found"))?;
    let mut signaled_pids = Vec::new();
    let mut skipped_pids = Vec::new();
    let mut root_pids = Vec::new();
    if let Some(engine_pid) = record.engine_pid
        && engine_pid != 0
    {
        if engine_pid == std::process::id() {
            skipped_pids.push(engine_pid);
        } else {
            root_pids.push(engine_pid);
        }
    }
    if record.supervisor_pid != 0
        && record.supervisor_pid != std::process::id()
        && Some(record.supervisor_pid) != record.engine_pid
    {
        root_pids.push(record.supervisor_pid);
    }
    let mut pids_to_signal = descendant_pids_for_roots(&root_pids)?;
    pids_to_signal.extend(root_pids);
    let mut seen_pids = HashSet::new();
    for pid in pids_to_signal {
        if !seen_pids.insert(pid) {
            continue;
        }
        if terminate_process(pid)? {
            signaled_pids.push(pid);
        } else {
            skipped_pids.push(pid);
        }
    }
    let force_signaled_pids = force_terminate_remaining_processes(&signaled_pids)?;
    record.status = "stopped".to_owned();
    record.write()?;
    // Best-effort and idempotent: a missing key file is not an error, so this
    // is safe to call unconditionally on every stop (including loopback
    // services that never had a key, and repeated stops of an already-stopped
    // service). Leaving the 0600 key file behind after stop would strand a
    // plaintext secret on disk for a service that no longer exists.
    let _ = std::fs::remove_file(rocm_engine_protocol::endpoint_key_file_path(
        paths, service_id,
    ));
    Ok(json!({
        "service": record,
        "signaled_pids": signaled_pids,
        "force_signaled_pids": force_signaled_pids,
        "skipped_pids": skipped_pids,
    }))
}

#[cfg(unix)]
fn descendant_pids_for_roots(root_pids: &[u32]) -> Result<Vec<u32>> {
    if root_pids.is_empty() {
        return Ok(Vec::new());
    }
    let output = ProcessCommand::new("ps")
        .args(["-eo", "pid=,ppid="])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to list process tree with ps")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        bail!(
            "failed to list process tree: {}",
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }
    let output = String::from_utf8_lossy(&output.stdout);
    Ok(descendant_pids_from_ps_output(&output, root_pids))
}

#[cfg(not(unix))]
fn descendant_pids_for_roots(_root_pids: &[u32]) -> Result<Vec<u32>> {
    Ok(Vec::new())
}

#[cfg(any(unix, test))]
fn descendant_pids_from_ps_output(output: &str, root_pids: &[u32]) -> Vec<u32> {
    fn append_descendants(
        parent: u32,
        processes: &[(u32, u32)],
        seen: &mut HashSet<u32>,
        output: &mut Vec<u32>,
    ) {
        for (pid, ppid) in processes {
            if *ppid != parent || *pid == parent || !seen.insert(*pid) {
                continue;
            }
            append_descendants(*pid, processes, seen, output);
            output.push(*pid);
        }
    }

    let processes = output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let pid = parts.next()?.parse::<u32>().ok()?;
            let ppid = parts.next()?.parse::<u32>().ok()?;
            Some((pid, ppid))
        })
        .collect::<Vec<_>>();
    let mut seen = root_pids.iter().copied().collect::<HashSet<_>>();
    let mut descendants = Vec::new();
    for root in root_pids {
        append_descendants(*root, &processes, &mut seen, &mut descendants);
    }
    descendants
}

fn terminate_process(pid: u32) -> Result<bool> {
    if pid == std::process::id() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        let output = ProcessCommand::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch kill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if stderr.contains("No such process") || stdout.contains("No such process") {
                return Ok(false);
            }
            bail!(
                "failed to signal pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
    #[cfg(windows)]
    {
        let output = ProcessCommand::new("taskkill")
            .arg("/PID")
            .arg(pid.to_string())
            .arg("/T")
            .arg("/F")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch taskkill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if stderr.contains("not found") || stdout.contains("not found") {
                return Ok(false);
            }
            bail!(
                "failed to stop pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
}

#[cfg(unix)]
fn force_terminate_remaining_processes(pids: &[u32]) -> Result<Vec<u32>> {
    if pids.is_empty() {
        return Ok(Vec::new());
    }
    thread::sleep(Duration::from_millis(750));
    let mut force_signaled = Vec::new();
    for pid in pids {
        if *pid == std::process::id() || !process_is_running(*pid)? {
            continue;
        }
        let output = ProcessCommand::new("kill")
            .arg("-KILL")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch kill -KILL for pid {pid}"))?;
        if output.status.success() {
            force_signaled.push(*pid);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "failed to force stop pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            );
        }
    }
    Ok(force_signaled)
}

#[cfg(not(unix))]
fn force_terminate_remaining_processes(_pids: &[u32]) -> Result<Vec<u32>> {
    Ok(Vec::new())
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> Result<bool> {
    let output = ProcessCommand::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch kill -0 for pid {pid}"))?;
    Ok(output.status.success())
}

async fn run_daemon(
    paths: &AppPaths,
    automations_enabled: bool,
    local_webhook_port: Option<u16>,
) -> Result<()> {
    if local_webhook_port.is_some() && !automations_enabled {
        bail!("local webhook source requires --automations-enabled");
    }

    let config = RocmCliConfig::load(paths)?;
    let mut state = build_runtime_state(&config, automations_enabled);
    let local_webhook = if let Some(port) = local_webhook_port {
        Some(start_local_webhook_source(port).await?)
    } else {
        None
    };
    let (local_webhook_endpoint, mut local_webhook_receiver, local_webhook_task) =
        match local_webhook {
            Some(source) => (
                Some(source.endpoint),
                Some(source.receiver),
                Some(source.task),
            ),
            None => (None, None, None),
        };
    state.local_webhook_endpoint = local_webhook_endpoint.clone();

    println!("rocmd run");
    println!("  automations enabled: {automations_enabled}");
    println!(
        "  lifecycle: {}",
        if automations_enabled {
            "persistent"
        } else {
            "on-demand"
        }
    );
    println!("  config: {}", paths.config_path().display());
    println!("  state: {}", paths.automation_state_path().display());
    println!(
        "  local_webhook_endpoint: {}",
        local_webhook_endpoint.as_deref().unwrap_or("disabled")
    );
    let enabled_count = state
        .active_watchers
        .iter()
        .filter(|watcher| watcher.enabled)
        .count();
    println!("  enabled watchers: {enabled_count}");

    if !automations_enabled {
        println!(
            "  note: rerun with --automations-enabled to keep rocmd alive for watcher execution"
        );
        return Ok(());
    }

    paths.ensure()?;
    state.write(paths)?;
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let mut ticker = time::interval(WATCHER_TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let config = RocmCliConfig::load(paths)?;
                reconcile_watcher_snapshots(&config, &mut state);
                evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            event = receive_local_webhook_event(&mut local_webhook_receiver) => {
                if let Some(event) = event {
                    let config = RocmCliConfig::load(paths)?;
                    reconcile_watcher_snapshots(&config, &mut state);
                    record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "info",
                        "local_webhook_event",
                        &format!(
                            "received local webhook event kind={} watcher_hint={}; dispatching through existing watcher policy; webhook payload grants no new action",
                            event.kind,
                            event.watcher_hint.as_deref().unwrap_or("<none>")
                        ),
                        event.service_id.clone(),
                    )?;
                    if let Err(error) =
                        evaluate_watchers_for_events(paths, &config, &mut state, &[event])
                    {
                        record_event(
                            paths,
                            &mut state,
                            "rocmd",
                            "error",
                            "local_webhook_dispatch_failed",
                            &format!(
                                "local webhook event could not be dispatched through watcher policy: {error}"
                            ),
                            None,
                        )?;
                    }
                    state.last_tick_unix_ms = unix_time_millis();
                    state.write(paths)?;
                } else {
                    local_webhook_receiver = None;
                    state.local_webhook_endpoint = None;
                    record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "warn",
                        "local_webhook_stopped",
                        "local webhook source stopped; automation daemon continues without webhook ingestion",
                        None,
                    )?;
                    state.write(paths)?;
                }
            }
            () = &mut shutdown => {
                break;
            }
        }
    }

    state.running = false;
    state.last_tick_unix_ms = unix_time_millis();
    state.local_webhook_endpoint = None;
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_stop",
        "rocmd automation supervisor stopped",
        None,
    )?;
    state.write(paths)?;
    if let Some(task) = local_webhook_task {
        task.abort();
    }
    Ok(())
}

fn print_status(paths: &AppPaths) -> Result<()> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  policy: on-demand by default, persistent only with background features");
    println!(
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match AutomationRuntimeState::load(paths)? {
        Some(state) => {
            println!(
                "  automations runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
            println!(
                "  local_webhook_endpoint: {}",
                state
                    .local_webhook_endpoint
                    .as_deref()
                    .unwrap_or("disabled")
            );
            for watcher in state
                .active_watchers
                .into_iter()
                .filter(|watcher| watcher.enabled)
            {
                println!(
                    "  watcher {} mode={} last_event={}",
                    watcher.id,
                    watcher.mode.as_str(),
                    watcher.last_event.as_deref().unwrap_or("<none>")
                );
            }
        }
        None => println!("  automations runtime: inactive"),
    }
    println!(
        "  automation events: {}",
        paths.automation_events_path().display()
    );
    println!("  audit events: {}", paths.audit_events_path().display());

    let records = load_managed_services(paths)?;
    if records.is_empty() {
        println!("  services: none");
        return Ok(());
    }

    for record in records {
        println!(
            "  service {} engine={} status={} endpoint={}",
            record.service_id, record.engine, record.status, record.endpoint_url
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn supervise_service(
    paths: &AppPaths,
    service_id: String,
    engine: String,
    model_ref: String,
    canonical_model_id: String,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    device_policy: String,
    gpu: Option<String>,
    engine_recipe_json: Option<String>,
) -> Result<()> {
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(&engine))?;
    fs::create_dir_all(paths.engine_state_dir(&engine))?;
    fs::create_dir_all(paths.services_dir())?;

    let gpu_indices = parse_gpu_indices_arg(gpu.as_deref())?;
    let _ = daemon_binary_path();

    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine.clone(),
        model_ref,
        canonical_model_id.clone(),
        host,
        port,
        "managed",
        std::process::id(),
        runtime_id.clone(),
        env_id.clone(),
        Some(device_policy.clone()),
    );
    record.gpu_indices = gpu_indices;
    record.engine_recipe_json = engine_recipe_json.clone();
    record.write()?;

    let log_file = fs::File::create(&record.log_path)
        .with_context(|| format!("failed to create {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    let rocm_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let mut command = ProcessCommand::new(rocm_binary);
    command
        .args(engine_serve_http_args(
            &engine,
            &record.service_id,
            &canonical_model_id,
            &record.host,
            record.port,
            &device_policy,
            &record.gpu_indices,
            runtime_id.as_deref(),
            env_id.as_deref(),
            engine_recipe_json.as_deref(),
            &record.engine_state_path,
        ))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    // Re-thread the endpoint key file (public bind only) onto the engine child,
    // same as the initial `rocm serve` spawn. This path also runs on daemon
    // recovery (`restart_managed_service` re-execs `rocmd supervise`), so
    // without this a previously-authenticated public service would come back
    // up anonymous after a crash/recover cycle.
    apply_endpoint_key_env(&mut command, paths, &record.service_id);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn engine supervisor child for {engine}"))?;

    record.engine_pid = Some(child.id());
    record.status = "running".to_owned();
    record.write()?;

    // Clone the fields the poller reads so the `on_phase` closure can borrow
    // `record` mutably to persist each startup-phase transition to disk.
    let ready_engine = record.engine.clone();
    let ready_service_id = record.service_id.clone();
    let ready_log_path = record.log_path.clone();
    let became_ready = wait_for_service_ready(
        paths,
        &ready_engine,
        &ready_service_id,
        &ready_log_path,
        Duration::from_mins(3),
        |phase| {
            record.startup_phase = Some(phase.to_owned());
            let _ = record.write();
        },
    );
    if became_ready {
        record.status = "ready".to_owned();
        // The phase only describes the coming-up window; clear it once ready.
        record.startup_phase = None;
        record.write()?;
    }

    let exit_status = child.wait().context("failed waiting for engine child")?;
    record.status = if exit_status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;

    if exit_status.success() {
        Ok(())
    } else {
        std::process::exit(exit_status.code().unwrap_or(1));
    }
}

#[allow(clippy::too_many_arguments)]
fn engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &str,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe_json: Option<&str>,
    state_path: &Path,
) -> Vec<String> {
    let mut args = vec![
        "__engine-serve-http".to_owned(),
        engine.to_owned(),
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--device-policy".to_owned(),
        device_policy.to_owned(),
    ];
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(optional_arg("--runtime-id", runtime_id));
    args.extend(optional_arg("--env-id", env_id));
    args.extend(optional_arg("--engine-recipe-json", engine_recipe_json));
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    args
}

fn build_runtime_state(
    config: &RocmCliConfig,
    automations_enabled: bool,
) -> AutomationRuntimeState {
    let now = unix_time_millis();
    let active_watchers = builtin_watchers()
        .iter()
        .map(|watcher| WatcherRuntimeSnapshot {
            id: watcher.id.to_owned(),
            enabled: config.watcher_enabled(watcher),
            mode: config.effective_watcher_mode(watcher),
            summary: watcher.summary.to_owned(),
            last_event: None,
            last_event_unix_ms: None,
        })
        .collect();
    AutomationRuntimeState {
        running: automations_enabled,
        automations_enabled,
        daemon_pid: std::process::id(),
        started_at_unix_ms: now,
        last_tick_unix_ms: now,
        local_webhook_endpoint: None,
        active_watchers,
    }
}

#[derive(Debug)]
struct LocalWebhookSource {
    endpoint: String,
    receiver: mpsc::Receiver<AutomationTriggerEvent>,
    task: JoinHandle<()>,
}

#[derive(Clone)]
struct LocalWebhookState {
    sender: mpsc::Sender<AutomationTriggerEvent>,
}

#[derive(Debug, Deserialize)]
struct LocalWebhookEventRequest {
    watcher_hint: String,
    kind: String,
    #[serde(default)]
    service_id: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    payload: Value,
}

async fn start_local_webhook_source(port: u16) -> Result<LocalWebhookSource> {
    let listener = TcpListener::bind((DEFAULT_LOCAL_HOST, port))
        .await
        .with_context(|| {
            format!("failed to bind local webhook source to {DEFAULT_LOCAL_HOST}:{port}")
        })?;
    let addr = listener
        .local_addr()
        .context("failed to read local webhook listener address")?;
    if !addr.ip().is_loopback() {
        bail!("local webhook source must bind to loopback; resolved address was {addr}");
    }

    let (sender, receiver) = mpsc::channel(64);
    let app = Router::new()
        .route("/health", get(local_webhook_health))
        .route("/automation-events", post(local_webhook_post_event))
        .with_state(LocalWebhookState { sender });
    let task = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("local webhook source stopped: {error}");
        }
    });

    Ok(LocalWebhookSource {
        endpoint: format!("http://{addr}/automation-events"),
        receiver,
        task,
    })
}

async fn receive_local_webhook_event(
    receiver: &mut Option<mpsc::Receiver<AutomationTriggerEvent>>,
) -> Option<AutomationTriggerEvent> {
    if let Some(receiver) = receiver {
        receiver.recv().await
    } else {
        std::future::pending().await
    }
}

async fn local_webhook_health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "source": "local_webhook",
        "bind": "loopback_only",
    }))
}

async fn local_webhook_post_event(
    State(state): State<LocalWebhookState>,
    Json(request): Json<LocalWebhookEventRequest>,
) -> impl IntoResponse {
    let event = match local_webhook_event_from_request(request) {
        Ok(event) => event,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": "rejected",
                    "error": error.to_string(),
                })),
            );
        }
    };

    if state.sender.send(event.clone()).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "error": "local webhook receiver is not running",
            })),
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "queued",
            "source": event.source,
            "kind": event.kind,
            "watcher_hint": event.watcher_hint,
            "message": "queued automation event; dispatch uses existing watcher policy; no new action is granted by webhook payload",
        })),
    )
}

fn local_webhook_event_from_request(
    request: LocalWebhookEventRequest,
) -> Result<AutomationTriggerEvent> {
    let watcher_hint = request.watcher_hint.trim();
    if watcher_hint.is_empty() {
        bail!("watcher_hint is required");
    }
    if builtin_watcher(watcher_hint).is_none() {
        bail!("unknown watcher_hint: {watcher_hint}");
    }

    let kind = request.kind.trim();
    if kind.is_empty() {
        bail!("kind is required");
    }
    if !local_webhook_kind_allowed(watcher_hint, kind) {
        bail!(
            "kind `{kind}` is not accepted for watcher `{watcher_hint}`; local webhooks can only trigger accepted watcher event kinds"
        );
    }
    let service_id = request.service_id.as_deref().map(str::trim);
    if let Some(service_id) = service_id
        && (service_id.contains('/') || service_id.contains('\\'))
    {
        bail!("service_id must not contain path separators");
    }
    match watcher_hint {
        "server-recover" if service_id.unwrap_or_default().is_empty() => {
            bail!("server-recover webhook events require service_id");
        }
        "cache-warm" if payload_string(&request.payload, "artifact_ref").is_none() => {
            bail!("cache-warm webhook events require payload.artifact_ref");
        }
        "driver-upgrade"
            if payload_string(&request.payload, "component").as_deref() != Some("driver") =>
        {
            bail!("driver-upgrade webhook events require payload.component=driver");
        }
        _ => {}
    }

    Ok(AutomationTriggerEvent {
        at_unix_ms: unix_time_millis(),
        kind: kind.to_owned(),
        source: "local_webhook".to_owned(),
        watcher_hint: Some(watcher_hint.to_owned()),
        service_id: request
            .service_id
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
        reason: request
            .reason
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
        payload: request.payload,
    })
}

fn local_webhook_kind_allowed(watcher_id: &str, kind: &str) -> bool {
    match watcher_id {
        "therock-update" => kind == "schedule.tick",
        "server-recover" => matches!(
            kind,
            "service.manifest_recoverable"
                | "service.endpoint_recoverable"
                | "service.healthcheck_recoverable"
        ),
        "gpu-metrics" => matches!(kind, "gpu.metrics" | "gpu.metrics_unavailable"),
        "cache-warm" => kind == "cache.warm",
        "driver-upgrade" => kind == "update.available",
        "gpu-thermal-protect" => {
            matches!(kind, "gpu.thermal_pressure" | "gpu.memory_pressure")
        }
        _ => false,
    }
}

fn reconcile_watcher_snapshots(config: &RocmCliConfig, state: &mut AutomationRuntimeState) {
    for watcher in builtin_watchers() {
        match state.watcher_mut(watcher.id) {
            Some(snapshot) => {
                snapshot.enabled = config.watcher_enabled(watcher);
                snapshot.mode = config.effective_watcher_mode(watcher);
                snapshot.summary = watcher.summary.to_owned();
            }
            None => state.active_watchers.push(WatcherRuntimeSnapshot {
                id: watcher.id.to_owned(),
                enabled: config.watcher_enabled(watcher),
                mode: config.effective_watcher_mode(watcher),
                summary: watcher.summary.to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }),
        }
    }
}

fn evaluate_watchers(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let events = collect_automation_events(paths, config, state)?;
    evaluate_watchers_for_events(paths, config, state, &events)
}

fn collect_automation_events(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &AutomationRuntimeState,
) -> Result<Vec<AutomationTriggerEvent>> {
    collect_automation_events_with_gpu_snapshot(paths, state, || {
        gather_gpu_snapshot_for_config(config)
    })
}

fn collect_automation_events_with_gpu_snapshot<F>(
    paths: &AppPaths,
    state: &AutomationRuntimeState,
    gpu_snapshot: F,
) -> Result<Vec<AutomationTriggerEvent>>
where
    F: FnMut() -> CodexBridgeGpuSnapshot,
{
    let now = unix_time_millis();
    let mut events = Vec::new();

    if therock_update_due(state, now) {
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({
                "interval_ms": THEROCK_UPDATE_INTERVAL_MS,
            }),
        });
    }

    if server_recover_due(state, now)
        && let Some((record, recovery_reason)) = find_recoverable_service(paths)?
    {
        let kind = service_recovery_event_kind(&recovery_reason);
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: kind.to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some(recovery_reason.clone()),
            payload: json!({
                "engine": record.engine,
                "status": record.status,
                "endpoint": record.endpoint_url,
                "recovery_reason": recovery_reason,
            }),
        });
    }

    let gpu_metrics_due_now = gpu_metrics_due(state, now);
    let gpu_thermal_protect_due_now = gpu_thermal_protect_due(state, now);
    let snapshot = (gpu_metrics_due_now || gpu_thermal_protect_due_now).then(gpu_snapshot);

    if gpu_metrics_due_now {
        let snapshot = snapshot
            .as_ref()
            .expect("GPU snapshot should be collected for due metrics");
        let available = snapshot.amd_smi_available && snapshot.monitor_snapshot.is_some();
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: if available {
                "gpu.metrics".to_owned()
            } else {
                "gpu.metrics_unavailable".to_owned()
            },
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-metrics".to_owned()),
            service_id: None,
            reason: if available {
                Some("amd_smi_snapshot_available".to_owned())
            } else {
                snapshot
                    .note
                    .clone()
                    .or_else(|| Some("amd_smi_snapshot_unavailable".to_owned()))
            },
            payload: json!({
                "amd_smi_available": snapshot.amd_smi_available,
                "static_available": snapshot.static_snapshot.is_some(),
                "monitor_available": snapshot.monitor_snapshot.is_some(),
                "summary": gpu_snapshot_summary(snapshot),
                "interval_ms": GPU_METRICS_INTERVAL_MS,
            }),
        });
    }

    if gpu_thermal_protect_due_now && let Some(snapshot) = snapshot.as_ref() {
        events.extend(gpu_pressure_events(now, snapshot));
    }

    Ok(events)
}

fn evaluate_watchers_for_events(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
    events: &[AutomationTriggerEvent],
) -> Result<()> {
    for watcher in builtin_watchers() {
        if !config.watcher_enabled(watcher) {
            continue;
        }
        let mode = config.effective_watcher_mode(watcher);
        match watcher.id {
            "therock-update" => {
                for event in events_for_watcher(events, watcher.id, "schedule.tick") {
                    handle_therock_update_event(paths, mode, state, event)?;
                }
            }
            "server-recover" => {
                for event in events_for_watcher(events, watcher.id, "service.") {
                    handle_server_recover_event(paths, mode, state, event)?;
                }
            }
            "gpu-metrics" => {
                for event in events_for_watcher(events, watcher.id, "gpu.") {
                    handle_gpu_metrics_event(paths, mode, state, event)?;
                }
            }
            "gpu-thermal-protect" => {
                for event in
                    events_for_watcher_exact(events, watcher.id, "gpu.thermal_pressure").chain(
                        events_for_watcher_exact(events, watcher.id, "gpu.memory_pressure"),
                    )
                {
                    handle_gpu_thermal_protect_event(paths, mode, state, event)?;
                }
            }
            "cache-warm" => {
                for event in events_for_watcher_exact(events, watcher.id, "cache.warm") {
                    handle_cache_warm_event(paths, mode, state, event)?;
                }
            }
            "driver-upgrade" => {
                for event in events_for_watcher_exact(events, watcher.id, "update.available") {
                    handle_driver_upgrade_event(paths, mode, state, event)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn events_for_watcher<'a>(
    events: &'a [AutomationTriggerEvent],
    watcher_id: &str,
    kind_prefix: &str,
) -> impl Iterator<Item = &'a AutomationTriggerEvent> {
    events.iter().filter(move |event| {
        event.watcher_hint.as_deref() == Some(watcher_id) && event.kind.starts_with(kind_prefix)
    })
}

fn events_for_watcher_exact<'a>(
    events: &'a [AutomationTriggerEvent],
    watcher_id: &str,
    kind: &'static str,
) -> impl Iterator<Item = &'a AutomationTriggerEvent> {
    events.iter().filter(move |event| {
        event.watcher_hint.as_deref() == Some(watcher_id) && event.kind == kind
    })
}

fn handle_therock_update_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_therock_update_event_with_runner(paths, mode, state, event, |paths| {
        run_sandbox_tool(
            paths,
            SandboxToolArg::CheckUpdates,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
    })
}

fn handle_therock_update_event_with_runner<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    _event: &AutomationTriggerEvent,
    update_runner: F,
) -> Result<()>
where
    F: FnOnce(&AppPaths) -> Result<Value>,
{
    let policy = watcher_policy_action("therock-update", mode);
    let action = match policy {
        WatcherPolicyAction::Observe => "observe_schedule",
        WatcherPolicyAction::QueueProposal => "queue_update_proposal",
        WatcherPolicyAction::RunContained => "run_update_check",
    };
    let message = match policy {
        WatcherPolicyAction::Observe => {
            "scheduled TheRock update check reminder emitted; run `rocm update` to inspect the selected channel"
        }
        WatcherPolicyAction::QueueProposal => {
            "scheduled TheRock update check reminder emitted; queueing read-only update-check proposal for review"
        }
        WatcherPolicyAction::RunContained => {
            "scheduled TheRock update check is approved for contained read-only execution"
        }
    };
    match policy {
        WatcherPolicyAction::Observe | WatcherPolicyAction::QueueProposal => {
            record_event(
                paths,
                state,
                "therock-update",
                "info",
                action,
                message,
                None,
            )?;
            if policy == WatcherPolicyAction::QueueProposal {
                queue_proposal(
                    paths,
                    "therock-update",
                    action,
                    "Check TheRock updates",
                    "Run `rocm update` to inspect available CLI, runtime, engine, and recipe updates before applying changes.",
                    None,
                )?;
            }
        }
        WatcherPolicyAction::RunContained => match update_runner(paths) {
            Ok(output) => match restricted_check_updates_result(&output) {
                Ok(result) => {
                    record_event(
                        paths,
                        state,
                        "therock-update",
                        if result.exit_status == 0 {
                            "info"
                        } else {
                            "error"
                        },
                        action,
                        &format!(
                            "{message}; restricted check_updates status={}; {}",
                            result.status,
                            update_check_message(result.status)
                        ),
                        None,
                    )?;
                    if result.update_available {
                        record_update_available_notification(paths, state)?;
                    }
                }
                Err(error) => {
                    record_event(
                        paths,
                        state,
                        "therock-update",
                        "error",
                        "update_check_failed",
                        &format!(
                            "scheduled TheRock update check failed during contained restricted execution: {error}; no updates were applied"
                        ),
                        None,
                    )?;
                }
            },
            Err(error) => {
                record_event(
                    paths,
                    state,
                    "therock-update",
                    "error",
                    "update_check_failed",
                    &format!(
                        "scheduled TheRock update check failed during contained read-only execution: {error}; no updates were applied"
                    ),
                    None,
                )?;
            }
        },
    }
    Ok(())
}

struct RestrictedCheckUpdatesResult<'a> {
    status: &'a str,
    update_available: bool,
    exit_status: i64,
}

fn restricted_check_updates_result(value: &Value) -> Result<RestrictedCheckUpdatesResult<'_>> {
    let tool = value
        .get("tool")
        .and_then(Value::as_str)
        .context("restricted update check did not report a tool name")?;
    if tool != SandboxToolArg::CheckUpdates.as_cli_value() {
        bail!("restricted update check returned `{tool}`, expected `check_updates`");
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("checked");
    let update_available = value
        .get("update_available")
        .and_then(Value::as_bool)
        .unwrap_or(status == "update_available");
    let exit_status = value
        .get("exit_status")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| i64::from(status == "error"));
    Ok(RestrictedCheckUpdatesResult {
        status,
        update_available,
        exit_status,
    })
}

fn record_update_available_notification(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let message =
        "A ROCm runtime update is available. Preview it before applying. No updates were applied.";
    record_event(
        paths,
        state,
        "therock-update",
        "info",
        "notify_if_newer",
        message,
        None,
    )?;
    record_notification_audit(
        paths,
        "watcher:therock-update",
        "notify_if_newer",
        Some("therock-update"),
        message,
    )
}

fn therock_update_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "therock-update" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= THEROCK_UPDATE_INTERVAL_MS)
}

fn gpu_metrics_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "gpu-metrics" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= GPU_METRICS_INTERVAL_MS)
}

fn gpu_thermal_protect_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "gpu-thermal-protect" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= GPU_METRICS_INTERVAL_MS)
}

fn handle_gpu_metrics_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let summary = event
        .payload
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("summary unavailable");
    let level = if event.kind == "gpu.metrics" {
        "info"
    } else {
        "warn"
    };
    let mode_note = match mode {
        WatcherMode::Observe => "observe mode records telemetry only",
        WatcherMode::Propose => {
            "propose mode has no GPU mutation policy yet, so telemetry is recorded only"
        }
        WatcherMode::Contained => {
            "contained mode has no GPU mutation policy yet, so telemetry is recorded only"
        }
    };
    let reason = event
        .reason
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("no detail");
    let source = match event.source.as_str() {
        "gpu_telemetry" => "local amd-smi telemetry",
        "local_webhook" => "local webhook",
        other => other,
    };

    record_event(
        paths,
        state,
        "gpu-metrics",
        level,
        "record_gpu_metrics",
        &format!(
            "GPU metrics event from {source}: {summary}; reason={reason}; {mode_note}; no mutating action was taken"
        ),
        None,
    )
}

fn gpu_snapshot_summary(snapshot: &CodexBridgeGpuSnapshot) -> String {
    let mut parts = Vec::new();
    parts.push(format!("amd_smi_available={}", snapshot.amd_smi_available));
    parts.push(format!(
        "static_snapshot={}",
        if snapshot.static_snapshot.is_some() {
            "available"
        } else {
            "missing"
        }
    ));
    parts.push(format!(
        "monitor_snapshot={}",
        if snapshot.monitor_snapshot.is_some() {
            "available"
        } else {
            "missing"
        }
    ));
    if let Some(count) = snapshot.static_snapshot.as_ref().and_then(gpu_data_count) {
        parts.push(format!("gpu_count={count}"));
    }
    if let Some(note) = snapshot.note.as_deref()
        && !note.trim().is_empty()
    {
        parts.push(format!("note={note}"));
    }
    parts.join(" ")
}

fn gpu_data_count(value: &Value) -> Option<usize> {
    value
        .get("gpu_data")
        .and_then(Value::as_array)
        .map(Vec::len)
}

#[derive(Debug, Clone, Copy)]
struct GpuPressureReading {
    gpu_index: Option<u64>,
    hotspot_temperature_c: Option<f64>,
    memory_temperature_c: Option<f64>,
    vram_percent: Option<f64>,
}

fn gpu_pressure_events(
    now: u128,
    snapshot: &CodexBridgeGpuSnapshot,
) -> Vec<AutomationTriggerEvent> {
    let Some(monitor_snapshot) = snapshot.monitor_snapshot.as_ref() else {
        return Vec::new();
    };
    monitor_entries(monitor_snapshot)
        .into_iter()
        .filter_map(gpu_pressure_reading)
        .filter_map(|reading| gpu_pressure_event(now, reading))
        .collect()
}

fn gpu_pressure_event(now: u128, reading: GpuPressureReading) -> Option<AutomationTriggerEvent> {
    let (kind, reason, metric_label, value, threshold) = if let Some(value) =
        reading.hotspot_temperature_c
        && value >= GPU_THERMAL_HOTSPOT_PRESSURE_C
    {
        (
            "gpu.thermal_pressure",
            "hotspot_temperature_threshold",
            "hotspot temperature",
            value,
            GPU_THERMAL_HOTSPOT_PRESSURE_C,
        )
    } else if let Some(value) = reading.memory_temperature_c
        && value >= GPU_THERMAL_MEMORY_PRESSURE_C
    {
        (
            "gpu.thermal_pressure",
            "memory_temperature_threshold",
            "memory temperature",
            value,
            GPU_THERMAL_MEMORY_PRESSURE_C,
        )
    } else if let Some(value) = reading.vram_percent
        && value >= GPU_MEMORY_VRAM_PRESSURE_PERCENT
    {
        (
            "gpu.memory_pressure",
            "vram_pressure_threshold",
            "VRAM use",
            value,
            GPU_MEMORY_VRAM_PRESSURE_PERCENT,
        )
    } else {
        return None;
    };
    let gpu_label = reading
        .gpu_index
        .map_or_else(|| "the GPU".to_owned(), |gpu| format!("GPU {gpu}"));
    let unit = if metric_label == "VRAM use" {
        "%"
    } else {
        " C"
    };
    let summary = format!(
        "{gpu_label} {metric_label} is {}{} (limit {}{})",
        display_metric(value),
        unit,
        display_metric(threshold),
        unit
    );
    Some(AutomationTriggerEvent {
        at_unix_ms: now,
        kind: kind.to_owned(),
        source: "gpu_telemetry".to_owned(),
        watcher_hint: Some("gpu-thermal-protect".to_owned()),
        service_id: None,
        reason: Some(reason.to_owned()),
        payload: json!({
            "gpu": reading.gpu_index,
            "hotspot_temperature_c": reading.hotspot_temperature_c,
            "memory_temperature_c": reading.memory_temperature_c,
            "vram_percent": reading.vram_percent,
            "hotspot_threshold_c": GPU_THERMAL_HOTSPOT_PRESSURE_C,
            "memory_temperature_threshold_c": GPU_THERMAL_MEMORY_PRESSURE_C,
            "vram_threshold_percent": GPU_MEMORY_VRAM_PRESSURE_PERCENT,
            "recommended_action": "stop_serving_load",
            "summary": summary,
        }),
    })
}

fn monitor_entries(value: &Value) -> Vec<&Value> {
    if let Some(entries) = value.as_array() {
        return entries.iter().collect();
    }
    value
        .get("gpu_data")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().collect())
        .unwrap_or_default()
}

fn gpu_pressure_reading(entry: &Value) -> Option<GpuPressureReading> {
    let reading = GpuPressureReading {
        gpu_index: metric_u64(entry, &["gpu", "gpu_id", "gpu_index"]),
        hotspot_temperature_c: metric_f64(
            entry,
            &[
                "hotspot_temperature",
                "hotspot_temperature_c",
                "temperature_hotspot",
            ],
        ),
        memory_temperature_c: metric_f64(
            entry,
            &[
                "memory_temperature",
                "memory_temperature_c",
                "temperature_memory",
            ],
        ),
        vram_percent: metric_f64(
            entry,
            &["vram_percent", "vram_usage_percent", "vram_used_percent"],
        ),
    };
    (reading.hotspot_temperature_c.is_some()
        || reading.memory_temperature_c.is_some()
        || reading.vram_percent.is_some())
    .then_some(reading)
}

fn metric_f64(entry: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| entry.get(*key).and_then(value_as_metric_f64))
}

fn metric_u64(entry: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| entry.get(*key).and_then(value_as_metric_u64))
}

fn value_as_metric_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        Value::Object(map) => map
            .get("value")
            .or_else(|| map.get("val"))
            .and_then(value_as_metric_f64),
        _ => None,
    }
}

fn value_as_metric_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        Value::Object(map) => map
            .get("value")
            .or_else(|| map.get("val"))
            .and_then(value_as_metric_u64),
        _ => None,
    }
}

fn display_metric(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn handle_gpu_thermal_protect_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let summary = payload_string(&event.payload, "summary")
        .unwrap_or_else(|| "GPU pressure is high".to_owned());
    let reason = event.reason.as_deref().unwrap_or("gpu_pressure_threshold");

    if matches!(mode, WatcherMode::Observe) {
        return record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "warn",
            "observe_gpu_pressure",
            &format!(
                "{summary}; observe mode records this only and does not stop any model server"
            ),
            event.service_id.clone(),
        );
    }

    let Some(record) = resolve_gpu_pressure_service_target(paths, event)? else {
        return record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "warn",
            "gpu_pressure_no_clear_target",
            &format!(
                "{summary}; rocm-cli did not choose a model server to stop because there was no single clear running managed server"
            ),
            None,
        );
    };

    if pending_stop_proposal_exists(paths, &record.service_id)? {
        return record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "info",
            "stop_proposal_already_pending",
            &format!(
                "{summary}; a reviewed stop request is already waiting for {}",
                record.service_id
            ),
            Some(record.service_id),
        );
    }

    let action = "queue_stop_server_proposal";
    let mode_note = if matches!(mode, WatcherMode::Contained) {
        "contained mode still asks before stopping anything"
    } else {
        "asking before stopping anything"
    };
    let message = format!(
        "{summary}; {mode_note}; selected managed server {} ({})",
        record.service_id, record.endpoint_url
    );
    record_event(
        paths,
        state,
        "gpu-thermal-protect",
        "warn",
        action,
        &message,
        Some(record.service_id.clone()),
    )?;
    queue_proposal_with_arguments(
        paths,
        "gpu-thermal-protect",
        action,
        "Review GPU pressure stop",
        &message,
        Some(record.service_id.clone()),
        json!({
            "service_id": record.service_id,
            "model_ref": record.model_ref,
            "canonical_model_id": record.canonical_model_id,
            "endpoint_url": record.endpoint_url,
            "engine": record.engine,
            "pressure_kind": event.kind,
            "pressure_reason": reason,
            "pressure_summary": summary,
            "gpu": event.payload.get("gpu").cloned().unwrap_or(Value::Null),
            "hotspot_temperature_c": event.payload.get("hotspot_temperature_c").cloned().unwrap_or(Value::Null),
            "memory_temperature_c": event.payload.get("memory_temperature_c").cloned().unwrap_or(Value::Null),
            "vram_percent": event.payload.get("vram_percent").cloned().unwrap_or(Value::Null),
            "hotspot_threshold_c": GPU_THERMAL_HOTSPOT_PRESSURE_C,
            "memory_temperature_threshold_c": GPU_THERMAL_MEMORY_PRESSURE_C,
            "vram_threshold_percent": GPU_MEMORY_VRAM_PRESSURE_PERCENT,
        }),
    )
}

fn resolve_gpu_pressure_service_target(
    paths: &AppPaths,
    event: &AutomationTriggerEvent,
) -> Result<Option<ManagedServiceRecord>> {
    if let Some(service_id) = event
        .service_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let record = load_service_record(paths, service_id)?;
        return Ok(active_pressure_target(record));
    }

    let active = load_managed_services(paths)?
        .into_iter()
        .filter_map(active_pressure_target)
        .collect::<Vec<_>>();
    if active.len() == 1 {
        Ok(active.into_iter().next())
    } else {
        Ok(None)
    }
}

fn active_pressure_target(record: ManagedServiceRecord) -> Option<ManagedServiceRecord> {
    (record.mode == "managed" && matches!(record.status.as_str(), "ready" | "running"))
        .then_some(record)
}

fn pending_stop_proposal_exists(paths: &AppPaths, service_id: &str) -> Result<bool> {
    Ok(rocm_core::load_recent_automation_proposals(paths, 100)?
        .into_iter()
        .any(|proposal| {
            proposal.status == "pending"
                && proposal.watcher_id == "gpu-thermal-protect"
                && proposal.service_id.as_deref() == Some(service_id)
                && (proposal.action == "queue_stop_server_proposal"
                    || proposal.tool.as_deref() == Some("stop_server"))
        }))
}

fn handle_cache_warm_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_cache_warm_event_with_resolver(paths, mode, state, event, |artifact_ref| {
        resolve_model_recipe_artifact(artifact_ref).map(|resolved| resolved.is_some())
    })
}

fn handle_cache_warm_event_with_resolver<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    mut artifact_exists: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<bool>,
{
    let Some(artifact_ref) = payload_string(&event.payload, "artifact_ref") else {
        return record_event(
            paths,
            state,
            "cache-warm",
            "warn",
            "cache_warm_missing_artifact",
            "cache warm event did not include artifact_ref; no prefetch proposal was queued",
            None,
        );
    };
    match artifact_exists(&artifact_ref) {
        Ok(true) => {}
        Ok(false) => {
            return record_event(
                paths,
                state,
                "cache-warm",
                "warn",
                "cache_warm_unknown_artifact",
                &format!(
                    "cache warm requested unknown registry artifact {artifact_ref}; no prefetch proposal was queued"
                ),
                None,
            );
        }
        Err(error) => {
            return record_event(
                paths,
                state,
                "cache-warm",
                "error",
                "cache_warm_registry_error",
                &format!(
                    "cache warm could not verify registry artifact {artifact_ref}: {error}; no prefetch proposal was queued"
                ),
                None,
            );
        }
    }
    match mode {
        WatcherMode::Observe => record_event(
            paths,
            state,
            "cache-warm",
            "info",
            "observe_cache_warm_request",
            &format!(
                "observed cache warm request for {artifact_ref}; observe mode does not queue or download artifacts"
            ),
            None,
        ),
        WatcherMode::Propose | WatcherMode::Contained => {
            let action = "queue_prefetch_proposal";
            let message = if matches!(mode, WatcherMode::Contained) {
                format!(
                    "cache warm requested for {artifact_ref}; contained mode still queues a review because artifact downloads require explicit source-policy approval"
                )
            } else {
                format!(
                    "cache warm requested for {artifact_ref}; queueing a reviewed prefetch proposal"
                )
            };
            record_event(paths, state, "cache-warm", "info", action, &message, None)?;
            queue_proposal_with_arguments(
                paths,
                "cache-warm",
                action,
                "Prefetch model artifact",
                &message,
                None,
                json!({
                    "artifact_ref": artifact_ref,
                }),
            )
        }
    }
}

fn handle_driver_upgrade_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_driver_upgrade_event_with_runner(paths, mode, state, event, |paths| {
        run_sandbox_tool(
            paths,
            SandboxToolArg::DriverPlan,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
    })
}

fn handle_driver_upgrade_event_with_runner<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    driver_plan_runner: F,
) -> Result<()>
where
    F: FnOnce(&AppPaths) -> Result<Value>,
{
    if payload_string(&event.payload, "component").as_deref() != Some("driver") {
        return record_event(
            paths,
            state,
            "driver-upgrade",
            "warn",
            "driver_upgrade_ignored_component",
            "driver-upgrade event did not include payload.component=driver; no driver plan proposal was queued",
            None,
        );
    }

    match mode {
        WatcherMode::Observe => record_event(
            paths,
            state,
            "driver-upgrade",
            "info",
            "observe_driver_update",
            "observed local driver update signal; observe mode does not queue or run a driver plan",
            None,
        ),
        WatcherMode::Propose => {
            let action = "prepare_driver_plan";
            let message =
                "local driver update signal received; queueing a reviewed read-only driver plan";
            record_event(
                paths,
                state,
                "driver-upgrade",
                "info",
                action,
                message,
                None,
            )?;
            queue_proposal(
                paths,
                "driver-upgrade",
                action,
                "Review driver install plan",
                message,
                None,
            )
        }
        WatcherMode::Contained => match driver_plan_runner(paths) {
            Ok(output) => match restricted_driver_plan_result(&output) {
                Ok(result) => record_event(
                    paths,
                    state,
                    "driver-upgrade",
                    if result.exit_status == 0 {
                        "info"
                    } else {
                        "error"
                    },
                    "run_driver_plan",
                    &format!(
                        "local driver update signal received; contained restricted driver_plan status={}; no driver commands were executed",
                        result.status
                    ),
                    None,
                ),
                Err(error) => record_event(
                    paths,
                    state,
                    "driver-upgrade",
                    "error",
                    "driver_plan_failed",
                    &format!(
                        "local driver update signal received, but contained restricted driver_plan failed: {error}; no driver commands were executed"
                    ),
                    None,
                ),
            },
            Err(error) => record_event(
                paths,
                state,
                "driver-upgrade",
                "error",
                "driver_plan_failed",
                &format!(
                    "local driver update signal received, but contained restricted driver_plan failed: {error}; no driver commands were executed"
                ),
                None,
            ),
        },
    }
}

struct RestrictedDriverPlanResult<'a> {
    status: &'a str,
    exit_status: i64,
}

fn restricted_driver_plan_result(value: &Value) -> Result<RestrictedDriverPlanResult<'_>> {
    let tool = value
        .get("tool")
        .and_then(Value::as_str)
        .context("restricted driver plan did not report a tool name")?;
    if tool != SandboxToolArg::DriverPlan.as_cli_value() {
        bail!("restricted driver plan returned `{tool}`, expected `driver_plan`");
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("planned");
    let exit_status = value
        .get("exit_status")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| i64::from(status == "error"));
    Ok(RestrictedDriverPlanResult {
        status,
        exit_status,
    })
}

fn payload_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
fn evaluate_server_recover(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let now = unix_time_millis();
    if !server_recover_due(state, now) {
        return Ok(());
    }

    let Some((mut record, recovery_reason)) = find_recoverable_service(paths)? else {
        return Ok(());
    };
    let kind = service_recovery_event_kind(&recovery_reason);
    let event = AutomationTriggerEvent {
        at_unix_ms: now,
        kind: kind.to_owned(),
        source: "managed_service".to_owned(),
        watcher_hint: Some("server-recover".to_owned()),
        service_id: Some(record.service_id.clone()),
        reason: Some(recovery_reason),
        payload: json!({
            "engine": record.engine.clone(),
            "status": record.status.clone(),
            "endpoint": record.endpoint_url.clone(),
        }),
    };
    handle_server_recover_event_with_record(paths, mode, state, &event, &mut record)
}

fn handle_server_recover_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let service_id = event
        .service_id
        .as_deref()
        .context("server-recover event is missing service_id")?;
    let mut record = load_service_record(paths, service_id)?;
    if !service_record_matches_recovery_event(paths, &record, event) {
        record_event(
            paths,
            state,
            "server-recover",
            "info",
            "ignore_nonrecoverable_service",
            &format!(
                "managed service {} does not currently need recovery; restart not attempted",
                record.service_id
            ),
            Some(record.service_id.clone()),
        )?;
        return Ok(());
    }
    handle_server_recover_event_with_record(paths, mode, state, event, &mut record)
}

fn service_record_matches_recovery_event(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    event: &AutomationTriggerEvent,
) -> bool {
    match event.kind.as_str() {
        "service.manifest_recoverable" => {
            manifest_service_recovery_reason(record, unix_time_millis()).is_some()
        }
        "service.endpoint_recoverable" => endpoint_service_recovery_reason(record).is_some(),
        "service.healthcheck_recoverable" => {
            engine_healthcheck_response(paths, &record.engine, &record.service_id)
                .is_ok_and(|healthcheck| healthcheck_response_recoverable(&healthcheck))
        }
        _ => false,
    }
}

fn handle_server_recover_event_with_record(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    record: &mut ManagedServiceRecord,
) -> Result<()> {
    let now = unix_time_millis();
    let recovery_reason = event.reason.as_deref().unwrap_or("recoverable_event");
    let recovery_reason_display = display_recovery_reason(recovery_reason);

    match watcher_policy_action("server-recover", mode) {
        WatcherPolicyAction::Observe => record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "observe_failure",
            &format!(
                "observed managed service {} needing recovery ({recovery_reason_display}); restart not attempted in observe mode",
                record.service_id,
            ),
            Some(record.service_id.clone()),
        ),
        WatcherPolicyAction::QueueProposal => {
            let message = format!(
                "managed service {} needs recovery ({recovery_reason_display}); queueing restart proposal",
                record.service_id,
            );
            record_event(
                paths,
                state,
                "server-recover",
                "warn",
                "queue_restart_proposal",
                &message,
                Some(record.service_id.clone()),
            )?;
            queue_proposal(
                paths,
                "server-recover",
                "queue_restart_proposal",
                "Restart managed service",
                &message,
                Some(record.service_id.clone()),
            )
        }
        WatcherPolicyAction::RunContained => {
            if let Some(last_restart) = record.last_restart_unix_ms
                && now.saturating_sub(last_restart) < SERVER_RECOVER_BACKOFF_MS
            {
                return Ok(());
            }
            restart_managed_service(paths, &mut *record)?;
            record_event(
                paths,
                state,
                "server-recover",
                "info",
                "restart_managed_service",
                &format!(
                    "restarted managed service {} on {}:{} after {recovery_reason_display}",
                    record.service_id, record.host, record.port
                ),
                Some(record.service_id.clone()),
            )
        }
    }
}

fn display_recovery_reason(reason: &str) -> String {
    match reason {
        "manifest_status_failed" => "manifest reports failed".to_owned(),
        "manifest_status_exited" => "manifest reports exited".to_owned(),
        "manifest_status_unreachable" => "manifest reports unreachable".to_owned(),
        "manifest_status_starting_stale" => "service has been starting for too long".to_owned(),
        "manifest_status_recovering_stale" => "service has been recovering for too long".to_owned(),
        "endpoint_status_unreachable" => "endpoint port is unreachable".to_owned(),
        other if other.starts_with("healthcheck_status_") => format!(
            "engine healthcheck reports {}",
            other.trim_start_matches("healthcheck_status_")
        ),
        other => other.replace('_', " "),
    }
}

fn service_recovery_event_kind(recovery_reason: &str) -> &'static str {
    if recovery_reason.starts_with("healthcheck_status_") {
        "service.healthcheck_recoverable"
    } else if recovery_reason.starts_with("endpoint_status_") {
        "service.endpoint_recoverable"
    } else {
        "service.manifest_recoverable"
    }
}

fn server_recover_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "server-recover" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= SERVER_RECOVER_BACKOFF_MS)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WatcherPolicyAction {
    Observe,
    QueueProposal,
    RunContained,
}

const fn watcher_policy_action(watcher_id: &str, mode: WatcherMode) -> WatcherPolicyAction {
    match (watcher_id, mode) {
        (_, WatcherMode::Observe) => WatcherPolicyAction::Observe,
        (_, WatcherMode::Propose) => WatcherPolicyAction::QueueProposal,
        (_, WatcherMode::Contained) => WatcherPolicyAction::RunContained,
    }
}

fn find_recoverable_service(paths: &AppPaths) -> Result<Option<(ManagedServiceRecord, String)>> {
    let now = unix_time_millis();
    for record in load_managed_services(paths)? {
        if record.mode != "managed" {
            continue;
        }
        if let Some(reason) = manifest_service_recovery_reason(&record, now) {
            return Ok(Some((record, reason)));
        }
        if matches!(record.status.as_str(), "ready" | "running") {
            let Ok(healthcheck) =
                engine_healthcheck_response(paths, &record.engine, &record.service_id)
            else {
                if let Some(reason) = endpoint_service_recovery_reason(&record) {
                    return Ok(Some((record, reason)));
                }
                continue;
            };
            if healthcheck_response_recoverable(&healthcheck) {
                return Ok(Some((
                    record,
                    format!("healthcheck_status_{}", healthcheck.status),
                )));
            }
            if let Some(reason) = endpoint_service_recovery_reason(&record) {
                return Ok(Some((record, reason)));
            }
        }
    }
    Ok(None)
}

fn endpoint_service_recovery_reason(record: &ManagedServiceRecord) -> Option<String> {
    (!wait_for_port(&record.host, record.port, ENDPOINT_HEALTH_TIMEOUT))
        .then(|| "endpoint_status_unreachable".to_owned())
}

fn load_service_record(paths: &AppPaths, service_id: &str) -> Result<ManagedServiceRecord> {
    if service_id.trim().is_empty() || service_id.contains('/') || service_id.contains('\\') {
        bail!("invalid managed service id `{service_id}`");
    }
    let manifest_path = paths.service_manifest_path(service_id);
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "managed service `{service_id}` not found at {}",
            manifest_path.display()
        )
    })?;
    let record = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if record.service_id != service_id {
        bail!(
            "managed service manifest {} contains service_id `{}`, expected `{service_id}`",
            manifest_path.display(),
            record.service_id
        );
    }
    Ok(record)
}

fn manifest_service_recovery_reason(
    record: &ManagedServiceRecord,
    now_unix_ms: u128,
) -> Option<String> {
    match record.status.as_str() {
        "failed" | "exited" | "unreachable" => Some(format!("manifest_status_{}", record.status)),
        "starting" | "recovering" => {
            let started_at = record
                .last_restart_unix_ms
                .unwrap_or(record.created_at_unix_ms);
            (now_unix_ms.saturating_sub(started_at) >= SERVER_TRANSIENT_STALE_MS)
                .then(|| format!("manifest_status_{}_stale", record.status))
        }
        _ => None,
    }
}

fn restart_managed_service(_paths: &AppPaths, record: &mut ManagedServiceRecord) -> Result<()> {
    let rocmd_binary =
        std::env::current_exe().context("failed to resolve current rocmd executable path")?;
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    record.status = "recovering".to_owned();
    record.restart_count = record.restart_count.saturating_add(1);
    record.last_restart_unix_ms = Some(unix_time_millis());
    record.supervisor_pid = std::process::id();
    record.write()?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .args(recovery_supervise_args(record))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    record.supervisor_pid = child.id();
    record.write()?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        record.status = "failed".to_owned();
        record.write()?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

fn recovery_supervise_args(record: &ManagedServiceRecord) -> Vec<String> {
    let mut args = vec![
        "supervise".to_owned(),
        record.service_id.clone(),
        "--engine".to_owned(),
        record.engine.clone(),
        "--model-ref".to_owned(),
        record.model_ref.clone(),
        "--canonical-model-id".to_owned(),
        record.canonical_model_id.clone(),
        "--host".to_owned(),
        record.host.clone(),
        "--port".to_owned(),
        record.port.to_string(),
        "--device-policy".to_owned(),
        record
            .device_policy
            .as_deref()
            .unwrap_or("gpu_required")
            .to_owned(),
    ];
    args.extend(optional_arg("--runtime-id", record.runtime_id.as_deref()));
    args.extend(optional_arg("--env-id", record.env_id.as_deref()));
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&record.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(optional_arg(
        "--engine-recipe-json",
        record.engine_recipe_json.as_deref(),
    ));
    args
}

fn record_event(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
    watcher_id: &str,
    level: &str,
    action: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    let now = unix_time_millis();
    if let Some(snapshot) = state.watcher_mut(watcher_id) {
        snapshot.last_event = Some(message.to_owned());
        snapshot.last_event_unix_ms = Some(now);
    }
    let event = AutomationEventRecord {
        at_unix_ms: now,
        watcher_id: watcher_id.to_owned(),
        level: level.to_owned(),
        action: action.to_owned(),
        message: message.to_owned(),
        service_id,
    };
    append_automation_event(paths, &event)?;

    let audit_watcher_id = (watcher_id != "rocmd").then(|| watcher_id.to_owned());
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: now,
            source: "rocmd".to_owned(),
            category: "automation".to_owned(),
            actor: audit_watcher_id
                .as_deref()
                .map_or_else(|| "rocmd".to_owned(), |id| format!("watcher:{id}")),
            level: level.to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            watcher_id: audit_watcher_id,
            service_id: event.service_id,
        },
    )
}

fn queue_proposal(
    paths: &AppPaths,
    watcher_id: &str,
    action: &str,
    title: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    queue_proposal_with_arguments(
        paths,
        watcher_id,
        action,
        title,
        message,
        service_id.clone(),
        proposal_arguments_for_action(action, service_id.as_deref()),
    )
}

fn queue_proposal_with_arguments(
    paths: &AppPaths,
    watcher_id: &str,
    action: &str,
    title: &str,
    message: &str,
    service_id: Option<String>,
    arguments: Value,
) -> Result<()> {
    append_automation_proposal(
        paths,
        &AutomationProposalRecord {
            at_unix_ms: unix_time_millis(),
            proposal_id: String::new(),
            watcher_id: watcher_id.to_owned(),
            action: action.to_owned(),
            title: title.to_owned(),
            message: message.to_owned(),
            status: "pending".to_owned(),
            service_id,
            tool: proposal_tool_for_action(action).map(str::to_owned),
            arguments,
            reviewed_at_unix_ms: None,
        },
    )
}

fn proposal_tool_for_action(action: &str) -> Option<&'static str> {
    match action {
        "queue_restart_proposal" => Some("restart_server"),
        "queue_stop_server_proposal" => Some("stop_server"),
        "queue_update_proposal" => Some("check_updates"),
        "queue_prefetch_proposal" => Some("prefetch_artifact"),
        "prepare_driver_plan" => Some("driver_plan"),
        _ => None,
    }
}

fn proposal_arguments_for_action(action: &str, service_id: Option<&str>) -> Value {
    match action {
        "queue_restart_proposal" => json!({
            "service_id": service_id,
        }),
        "queue_stop_server_proposal" => json!({
            "service_id": service_id,
        }),
        "queue_update_proposal" => json!({}),
        "prepare_driver_plan" => json!({}),
        _ => Value::Null,
    }
}

fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) {
            records.push(record);
        }
    }

    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    Ok(records)
}

#[cfg(unix)]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("setsid");
    command.arg(rocmd_binary);
    command
}

#[cfg(not(unix))]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    ProcessCommand::new(rocmd_binary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use rocm_core::ModelRecipeArtifactSourcePolicyRecord;
    use std::path::PathBuf;

    #[test]
    fn last_cr_segment_keeps_final_progress_redraw() {
        // A tqdm/HF-style in-place redraw collapses to its last segment.
        assert_eq!(
            last_cr_segment("Downloading:  10%\rDownloading:  55%\rDownloading: 100%"),
            "Downloading: 100%"
        );
        // A plain line is unchanged.
        assert_eq!(
            last_cr_segment("Loading model weights"),
            "Loading model weights"
        );
    }

    #[test]
    fn classify_startup_phase_maps_engine_vocabulary() {
        assert_eq!(
            classify_startup_phase("Downloading shards: 100%"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("Fetching 12 files"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("INFO: Loading model weights took 4.2s"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("llama_model_loader: loaded meta data"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("Capturing CUDA graph shapes"),
            Some("warmup")
        );
        assert_eq!(
            classify_startup_phase("Warming up the engine"),
            Some("warmup")
        );
        // Ordinary chatter carries no phase signal.
        assert_eq!(
            classify_startup_phase("Uvicorn running on http://..."),
            None
        );
    }

    #[test]
    fn classify_startup_phase_emits_only_dashboard_known_tokens() {
        // These tokens are the wire contract with the dashboard's
        // `StartupPhase::from_token` (rocm-dash-core); emitting anything else
        // would be silently dropped there. rocmd can't link that crate, so the
        // contract is pinned here by literal.
        for line in [
            "Downloading shards",
            "Loading model weights",
            "Capturing CUDA graph",
        ] {
            let token = classify_startup_phase(line).expect("line is a phase signal");
            assert!(
                matches!(token, "downloading" | "loading" | "warmup"),
                "token {token:?} must be one the dashboard understands"
            );
        }
    }

    #[test]
    fn read_new_log_phase_advances_and_tracks_latest() {
        use std::io::Write as _;
        // Workspace-local test root (rooted at CARGO_MANIFEST_DIR, not the
        // ambient temp dir) — same helper the other rocmd tests use.
        let dir = unique_test_root(&format!("rocmd-phase-{}", std::process::id()));
        let log = dir.join("svc.log");
        std::fs::write(&log, "boot\nDownloading shards: 100%\n").unwrap();

        let mut pos = 0_u64;
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("downloading"));
        // No new bytes → no phase, cursor unchanged.
        let after_first = pos;
        assert_eq!(read_new_log_phase(&log, &mut pos), None);
        assert_eq!(pos, after_first);

        // Appending a later stage advances the phase.
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "Loading model weights took 3s").unwrap();
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("loading"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_serve_http_args_forward_engine_recipe_json() {
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            Some("therock-release:gfx120X-all"),
            Some("env-1"),
            Some(engine_recipe_json),
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2).any(|pair| {
                pair[0] == "--runtime-id" && pair[1] == "therock-release:gfx120X-all"
            })
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--state-path" && pair[1] == "state.json")
        );
    }

    #[test]
    fn engine_serve_http_args_emit_gpu_indices_when_pinned() {
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[1],
            None,
            None,
            None,
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--gpu" && pair[1] == "1")
        );

        let auto = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            None,
            None,
            None,
            Path::new("state.json"),
        );
        assert!(!auto.iter().any(|arg| arg == "--gpu"));
    }

    #[test]
    fn recovery_supervise_args_preserve_engine_recipe_json() {
        let (_root, paths) = temp_app_paths("recovery-engine-recipe");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "managed",
            123,
            Some("therock-release:gfx120X-all".to_owned()),
            Some("env-1".to_owned()),
            Some("gpu_required".to_owned()),
        );
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        record.engine_recipe_json = Some(engine_recipe_json.to_owned());

        let args = recovery_supervise_args(&record);

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2)
                .any(|pair| { pair[0] == "--canonical-model-id" && pair[1] == "Qwen/Qwen3.5-4B" })
        );
    }

    #[test]
    fn rocm_mcp_tools_include_bridge_gaps() {
        let tools = rocm_mcp_tools();
        let names = tools
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
            .collect::<Vec<_>>();
        assert!(names.contains(&"gpu_snapshot".to_owned()));
        assert!(names.contains(&"service_logs".to_owned()));
        assert!(names.contains(&"natural_language_plan".to_owned()));
        assert!(names.contains(&"rocm_command".to_owned()));
        assert!(names.contains(&"install_sdk".to_owned()));
        assert!(names.contains(&"install_engine".to_owned()));
        assert!(names.contains(&"launch_server".to_owned()));
        assert!(names.contains(&"stop_server".to_owned()));
        assert!(names.contains(&"watcher_enable".to_owned()));
        assert!(names.contains(&"watcher_disable".to_owned()));
        let automations = tools
            .iter()
            .find(|tool| tool.get("name").and_then(Value::as_str) == Some("automations"))
            .expect("automations tool should be present");
        assert!(
            automations
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|description| description.contains("local webhook events"))
        );
    }

    #[test]
    fn direct_mcp_call_requires_approval_for_every_mutating_tool() {
        for tool in rocm_mcp_tools() {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .expect("tool should have a name");
            let read_only = tool
                .get("annotations")
                .and_then(|annotations| annotations.get("readOnlyHint"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                mcp_tool_requires_direct_approval(name),
                !read_only,
                "hidden direct MCP helper approval classification drifted for `{name}`"
            );
        }
    }

    #[test]
    fn direct_mcp_call_parses_allow_mutation_flag() {
        let cli = Cli::try_parse_from([
            "rocmd",
            "mcp-call",
            "install_sdk",
            "--arguments-json",
            "{}",
            "--allow-mutation",
        ])
        .expect("hidden direct MCP helper args should parse");

        match cli.command {
            Some(Command::McpCall {
                name,
                arguments_json,
                allow_mutation,
            }) => {
                assert_eq!(name, "install_sdk");
                assert_eq!(arguments_json, "{}");
                assert!(allow_mutation);
            }
            _ => panic!("expected mcp-call command"),
        }
    }

    #[test]
    fn direct_mcp_call_guard_blocks_mutation_without_explicit_ack() {
        ensure_direct_mcp_call_allowed("examine", false)
            .expect("read-only direct MCP helper calls should not need mutation approval");

        let error = ensure_direct_mcp_call_allowed("install_sdk", false)
            .expect_err("mutating direct MCP helper calls should require approval");
        assert!(error.to_string().contains("--allow-mutation"), "{error:#}");

        ensure_direct_mcp_call_allowed("install_sdk", true)
            .expect("explicitly approved direct MCP mutation should pass the helper guard");
    }

    #[test]
    fn rocm_command_helper_allows_only_read_only_rocm_commands() -> Result<()> {
        let status_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["rocm", "comfy", "status"]
            })
            .as_object()
            .expect("json object"),
        )?;
        assert_eq!(status_args, vec!["comfyui".to_owned(), "status".to_owned()]);
        ensure_rocm_command_is_read_only(&status_args).expect("ComfyUI status should be read-only");

        let log_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "logs"]
            })
            .as_object()
            .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&log_args).expect("ComfyUI logs should be read-only");

        let install_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "install"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&install_args)
            .expect_err("ComfyUI install must go through approval");
        assert!(error.to_string().contains("approval UI"));

        let shell_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["powershell", "-Command", "whoami"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&shell_args)
            .expect_err("non-rocm shell commands should be rejected");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn rocm_command_helper_treats_setup_status_as_read_only_and_reset_as_mutating() -> Result<()> {
        // Mirrors the bin's rocm_command classifier so `setup status` is read-only
        // on every binary's tool surface while `setup reset` stays approval-gated.
        let bare_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&bare_args).expect("bare setup should be read-only");

        let status_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "status"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&status_args).expect("setup status should be read-only");

        let reset_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "reset"] })
                .as_object()
                .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&reset_args)
            .expect_err("setup reset must go through approval");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn gpu_snapshot_respects_disabled_telemetry_policy() {
        let mut config = RocmCliConfig::default();
        config.telemetry.mode = rocm_core::TELEMETRY_MODE_OFF.to_owned();

        let snapshot = gather_gpu_snapshot_for_config(&config);

        assert!(!snapshot.amd_smi_available);
        assert!(snapshot.static_snapshot.is_none());
        assert!(snapshot.monitor_snapshot.is_none());
        assert!(
            snapshot
                .note
                .as_deref()
                .is_some_and(|note| note.contains("disabled by rocm-cli config"))
        );
    }

    #[test]
    fn read_tail_lines_returns_last_lines_only() -> Result<()> {
        let path = unique_test_path(&format!(
            "rocmd-tail-test-{}-{}.log",
            std::process::id(),
            unix_time_millis()
        ));
        fs::write(&path, "line1\nline2\nline3\nline4\n")?;
        let tail = read_tail_lines(&path, 2)?;
        fs::remove_file(&path)?;
        assert_eq!(tail, "line3\nline4");
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = temp_app_paths("engine-plugin");
        let plugin_dir = paths.data_dir.join("engines").join("plugins");
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        fs::remove_dir_all(root).ok();

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }

    #[test]
    fn launch_server_rejects_public_bind_without_ack() {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
        ]);
        let error = build_launch_server_args(&arguments).unwrap_err();
        assert!(
            error.to_string().contains("allow_public_bind=true"),
            "{error:#}"
        );
    }

    #[test]
    fn launch_server_forwards_public_bind_ack() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
            ("allow_public_bind".to_owned(), Value::Bool(true)),
        ]);
        let args = build_launch_server_args(&arguments)?;
        assert!(args.contains(&"--allow-public-bind".to_owned()));
        Ok(())
    }

    #[test]
    fn supervise_defaults_to_gpu_required_without_cpu_fallback() {
        let cli = Cli::try_parse_from([
            "rocmd",
            "supervise",
            "svc",
            "--engine",
            "vllm",
            "--model-ref",
            "qwen",
            "--canonical-model-id",
            "Qwen/Qwen3.5",
            "--host",
            "127.0.0.1",
            "--port",
            "11435",
        ])
        .expect("supervise args should parse");

        match cli.command {
            Some(Command::Supervise { device_policy, .. }) => {
                assert_eq!(device_policy, "gpu_required");
            }
            _ => panic!("expected supervise command"),
        }
    }

    #[test]
    fn install_sdk_rejects_system_prefix_without_ack() {
        let arguments = serde_json::Map::from_iter([(
            "prefix".to_owned(),
            Value::String("/opt/rocm".to_owned()),
        )]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    #[test]
    fn install_sdk_forwards_requested_build_date_and_rejects_conflict() -> Result<()> {
        let arguments = serde_json::Map::from_iter([(
            "build_date".to_owned(),
            Value::String("2026-06-05".to_owned()),
        )]);
        let argv = build_install_sdk_args(&arguments, true)?;
        assert_eq!(
            argv,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
                "--dry-run".to_owned(),
            ]
        );

        let conflicting = serde_json::Map::from_iter([
            (
                "version".to_owned(),
                Value::String("7.13.0a20260605".to_owned()),
            ),
            (
                "build_date".to_owned(),
                Value::String("2026-06-05".to_owned()),
            ),
        ]);
        let error = build_install_sdk_args(&conflicting, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("either `version` or `build_date`"));
        Ok(())
    }

    #[test]
    fn watcher_enable_builds_mode_args() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            (
                "watcher".to_owned(),
                Value::String("server-recover".to_owned()),
            ),
            ("mode".to_owned(), Value::String("contained".to_owned())),
        ]);
        let argv = build_watcher_enable_args(&arguments)?;
        assert_eq!(
            argv,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "contained".to_owned()
            ]
        );
        Ok(())
    }

    #[test]
    fn watcher_policy_maps_modes_to_decisions() {
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Observe),
            WatcherPolicyAction::Observe
        );
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Propose),
            WatcherPolicyAction::QueueProposal
        );
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Contained),
            WatcherPolicyAction::RunContained
        );
        assert_eq!(
            watcher_policy_action("therock-update", WatcherMode::Contained),
            WatcherPolicyAction::RunContained
        );
    }

    #[test]
    fn local_webhook_help_mentions_loopback_only_binding() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("run")
            .expect("run subcommand should exist")
            .render_long_help()
            .to_string();
        assert!(help.contains("--local-webhook-port"));
        assert!(help.contains("127.0.0.1:<PORT>"));
        assert!(help.contains("never binds publicly"));
    }

    #[test]
    fn local_webhook_port_rejects_out_of_range_values() {
        let error = Cli::try_parse_from([
            "rocmd",
            "run",
            "--automations-enabled",
            "--local-webhook-port",
            "70000",
        ])
        .unwrap_err();

        assert!(error.to_string().contains("70000"));
    }

    #[tokio::test]
    async fn local_webhook_requires_enabled_automation_loop() {
        let (_root, paths) = temp_app_paths("local-webhook-requires-loop");
        let error = run_daemon(&paths, false, Some(0)).await.unwrap_err();

        assert!(error.to_string().contains("requires --automations-enabled"));
    }

    #[test]
    fn local_webhook_event_rejects_unknown_watcher() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "unknown".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("unknown watcher_hint"));
    }

    #[test]
    fn local_webhook_event_rejects_kind_outside_existing_watcher_prefix() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_therock_update_accepts_exact_schedule_tick_only() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "therock-update".to_owned(),
            kind: "schedule.tick".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })?;
        assert_eq!(event.kind, "schedule.tick");

        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "therock-update".to_owned(),
            kind: "schedule.tick.extra".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();
        assert!(error.to_string().contains("accepted watcher event kinds"));
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_metrics_rejects_thermal_action_kinds() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.thermal_pressure".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_server_recover_rejects_nonrecoverable_service_kind() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.started".to_owned(),
            service_id: Some("svc-1".to_owned()),
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_cache_warm_rejects_other_cache_events() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.evict".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_event_requires_service_id_for_server_recover() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("require service_id"));
    }

    #[test]
    fn local_webhook_event_rejects_service_id_path_separators() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: Some("../svc".to_owned()),
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("path separators"));
    }

    #[test]
    fn local_webhook_event_builds_loopback_source_trigger() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: Some("manual test".to_owned()),
            payload: json!({
                "summary": "manual webhook probe",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-metrics"));
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.reason.as_deref(), Some("manual test"));
        Ok(())
    }

    #[test]
    fn local_webhook_cache_warm_requires_artifact_ref() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err();

        assert!(error.to_string().contains("payload.artifact_ref"));
    }

    #[test]
    fn local_webhook_cache_warm_builds_prefetch_event() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("cache-warm"));
        assert_eq!(event.kind, "cache.warm");
        assert_eq!(
            event.payload.get("artifact_ref").and_then(Value::as_str),
            Some("Qwen/Test-1B#hf-main")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_driver_upgrade_requires_driver_component() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "component": "runtime",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("payload.component=driver"));
    }

    #[test]
    fn local_webhook_driver_upgrade_rejects_other_update_events() {
        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.checked".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({
                "component": "driver",
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("accepted watcher event kinds"));
    }

    #[test]
    fn local_webhook_driver_upgrade_builds_update_event() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
                "available_version": "test-driver",
                "tool": "restart_server",
            }),
        })?;

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.watcher_hint.as_deref(), Some("driver-upgrade"));
        assert_eq!(event.kind, "update.available");
        assert_eq!(
            event.payload.get("component").and_then(Value::as_str),
            Some("driver")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_thermal_protect_accepts_only_pressure_events() -> Result<()> {
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-thermal-protect".to_owned(),
            kind: "gpu.thermal_pressure".to_owned(),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        })?;
        assert_eq!(event.kind, "gpu.thermal_pressure");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-thermal-protect"));
        assert_eq!(event.service_id.as_deref(), Some("svc-hot"));

        let error = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-thermal-protect".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: None,
            payload: json!({}),
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("accepted watcher event kinds"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn local_webhook_http_endpoint_queues_valid_event() -> Result<()> {
        use std::io::{Read, Write};

        let mut source = start_local_webhook_source(0).await?;
        let addr = source
            .endpoint
            .strip_prefix("http://")
            .and_then(|value| value.strip_suffix("/automation-events"))
            .expect("endpoint should include host and path")
            .to_owned();
        let body = json!({
            "watcher_hint": "gpu-metrics",
            "kind": "gpu.metrics",
            "reason": "http smoke",
            "payload": {
                "summary": "http local webhook"
            }
        })
        .to_string();
        let mut stream = std::net::TcpStream::connect(&addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        write!(
            stream,
            "POST /automation-events HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;

        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer)?;
        let response = String::from_utf8_lossy(&buffer[..read]).to_string();
        assert!(response.starts_with("HTTP/1.1 202 Accepted"), "{response}");
        assert!(response.contains("no new action is granted"));
        let event = time::timeout(Duration::from_secs(2), source.receiver.recv())
            .await?
            .expect("queued webhook event should be received");
        source.task.abort();

        assert_eq!(event.source, "local_webhook");
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.watcher_hint.as_deref(), Some("gpu-metrics"));
        assert_eq!(event.reason.as_deref(), Some("http smoke"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn local_webhook_http_endpoint_rejects_malformed_json() -> Result<()> {
        use std::io::{Read, Write};

        let source = start_local_webhook_source(0).await?;
        let addr = source
            .endpoint
            .strip_prefix("http://")
            .and_then(|value| value.strip_suffix("/automation-events"))
            .expect("endpoint should include host and path")
            .to_owned();
        let body = "{";
        let mut stream = std::net::TcpStream::connect(&addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        write!(
            stream,
            "POST /automation-events HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;

        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer)?;
        let response = String::from_utf8_lossy(&buffer[..read]).to_string();
        source.task.abort();

        let status_line = response.lines().next().unwrap_or_default();
        assert!(status_line.starts_with("HTTP/1.1 4"), "{response}");
        assert!(response.to_ascii_lowercase().contains("json"), "{response}");
        Ok(())
    }

    #[test]
    fn event_collector_emits_schedule_tick_for_due_update() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-schedule");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "therock-update",
            WatcherMode::Observe,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("therock-update"))
            .expect("schedule tick event should be emitted");
        assert_eq!(event.kind, "schedule.tick");
        assert_eq!(event.source, "scheduler");
        assert_eq!(event.reason.as_deref(), Some("therock_update_interval_due"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_recoverable_service_event() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-service");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("server-recover"))
            .expect("recoverable service event should be emitted");
        assert_eq!(event.kind, "service.manifest_recoverable");
        assert_eq!(event.service_id.as_deref(), Some("svc-failed"));
        assert_eq!(event.reason.as_deref(), Some("manifest_status_failed"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_endpoint_recoverable_service_event() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-endpoint");
        paths.ensure()?;
        let mut service = ManagedServiceRecord::new(
            &paths,
            "svc-endpoint",
            "missing-engine",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            1,
            "managed",
            123,
            None,
            None,
            None,
        );
        service.status = "ready".to_owned();
        service.write()?;
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("server-recover"))
            .expect("endpoint recoverable service event should be emitted");
        assert_eq!(event.kind, "service.endpoint_recoverable");
        assert_eq!(event.service_id.as_deref(), Some("svc-endpoint"));
        assert_eq!(event.reason.as_deref(), Some("endpoint_status_unreachable"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_gpu_metrics_event_when_enabled() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-metrics");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Observe,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: Some(json!({
                    "gpu_data": [
                        { "gpu": 0, "asic": { "market_name": "AMD Radeon Test" } }
                    ]
                })),
                monitor_snapshot: Some(json!({ "gpu_data": [] })),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("gpu-metrics"))
            .expect("gpu metrics event should be emitted");
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.source, "gpu_telemetry");
        assert_eq!(event.reason.as_deref(), Some("amd_smi_snapshot_available"));
        assert_eq!(
            event
                .payload
                .get("monitor_available")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            event
                .payload
                .get("summary")
                .and_then(Value::as_str)
                .is_some_and(|summary| summary.contains("gpu_count=1"))
        );
        Ok(())
    }

    #[test]
    fn event_collector_emits_gpu_thermal_pressure_event_when_enabled() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-thermal-pressure");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: None,
                monitor_snapshot: Some(json!({
                    "gpu_data": [
                        {
                            "gpu": 0,
                            "hotspot_temperature": { "value": 96.0 },
                            "memory_temperature": { "value": 88.0 },
                            "vram_percent": { "value": 72.0 }
                        }
                    ]
                })),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("gpu-thermal-protect"))
            .expect("thermal pressure event should be emitted");
        assert_eq!(event.kind, "gpu.thermal_pressure");
        assert_eq!(event.source, "gpu_telemetry");
        assert_eq!(
            event.reason.as_deref(),
            Some("hotspot_temperature_threshold")
        );
        assert!(
            event
                .payload
                .get("summary")
                .and_then(Value::as_str)
                .is_some_and(|summary| summary.contains("GPU 0 hotspot temperature is 96 C"))
        );
        Ok(())
    }

    #[test]
    fn event_collector_skips_gpu_pressure_below_thresholds() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-pressure-cool");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: None,
                monitor_snapshot: Some(json!([
                    {
                        "gpu": 0,
                        "hotspot_temperature": 80.0,
                        "memory_temperature": 82.0,
                        "vram_percent": 50.0
                    }
                ])),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        assert!(
            !events
                .iter()
                .any(|event| event.watcher_hint.as_deref() == Some("gpu-thermal-protect"))
        );
        Ok(())
    }

    #[test]
    fn gpu_metrics_event_records_read_only_status_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-metrics-record");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Contained,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        let events = vec![AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.metrics_unavailable".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-metrics".to_owned()),
            service_id: None,
            reason: Some("amd-smi missing".to_owned()),
            payload: json!({
                "summary": "amd_smi_available=false static_snapshot=missing monitor_snapshot=missing",
            }),
        }];

        evaluate_watchers_for_events(&paths, &config, &mut state, &events)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-metrics");
        assert_eq!(event.action, "record_gpu_metrics");
        assert!(event.message.contains("telemetry is recorded only"));
        assert!(event.message.contains("amd-smi missing"));
        assert!(event.message.contains("no mutating action was taken"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_propose_queues_reviewed_stop_for_one_running_service() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-propose");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-thermal-protect");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "gpu": 0,
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
                "hotspot_temperature_c": 96.0,
            }),
        };

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let saved = load_service_record(&paths, "svc-hot")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-thermal-protect");
        assert_eq!(event.action, "queue_stop_server_proposal");
        assert_eq!(event.service_id.as_deref(), Some("svc-hot"));
        assert!(event.message.contains("asking before stopping anything"));
        assert_eq!(saved.status, "ready");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-hot"));
        assert_eq!(
            proposals[0]
                .arguments
                .get("pressure_reason")
                .and_then(Value::as_str),
            Some("hotspot_temperature_threshold")
        );
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_contained_still_queues_reviewed_stop() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-contained");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "qwen",
            "Qwen/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.memory_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("vram_pressure_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 VRAM use is 96% (limit 95%)",
                "vram_percent": 96.0,
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Contained, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let saved = load_service_record(&paths, "svc-hot")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "queue_stop_server_proposal");
        assert!(event.message.contains("contained mode still asks"));
        assert_eq!(saved.status, "running");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_observe_records_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-observe");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Observe,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("memory_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU memory temperature is high",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Observe, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "observe_gpu_pressure");
        assert!(event.message.contains("does not stop any model server"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_ambiguous_services_records_no_action() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-ambiguous");
        paths.ensure()?;
        for service_id in ["svc-a", "svc-b"] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Test",
                "127.0.0.1",
                11435,
                "managed",
                123,
                None,
                None,
                None,
            );
            record.status = "ready".to_owned();
            record.write()?;
        }
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "gpu_pressure_no_clear_target");
        assert!(event.message.contains("did not choose a model server"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_does_not_duplicate_pending_stop_proposals() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-dedupe");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let events = rocm_core::load_recent_automation_events(&paths, 2)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 10)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        assert!(
            events
                .iter()
                .any(|event| event.action == "stop_proposal_already_pending")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_metrics_event_uses_existing_read_only_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("local-webhook-gpu-metrics");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Contained,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: Some("manual smoke".to_owned()),
            payload: json!({
                "summary": "manual webhook probe",
                "action": "restart_server",
                "mode": "contained",
            }),
        })?;

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-metrics");
        assert_eq!(event.action, "record_gpu_metrics");
        assert!(event.message.contains("from local webhook"));
        assert!(event.message.contains("manual smoke"));
        assert!(event.message.contains("no mutating action was taken"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn cache_warm_propose_mode_queues_prefetch_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-propose");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Propose,
            None,
        )]);
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
                "tool": "restart_server",
                "allow_artifact_download": true,
                "artifact_max_bytes": 1024,
            }),
        })?;

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            |_| Ok(true),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "cache-warm");
        assert_eq!(event.action, "queue_prefetch_proposal");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "cache-warm");
        assert_eq!(proposals[0].tool.as_deref(), Some("prefetch_artifact"));
        assert_eq!(
            proposals[0]
                .arguments
                .get("artifact_ref")
                .and_then(Value::as_str),
            Some("Qwen/Test-1B#hf-main")
        );
        assert!(
            proposals[0].arguments.get("tool").is_none(),
            "webhook payload must not grant arbitrary tool choice"
        );
        assert!(
            proposals[0]
                .arguments
                .get("allow_artifact_download")
                .is_none(),
            "webhook payload must not grant source-policy approval"
        );
        assert!(
            proposals[0].arguments.get("artifact_max_bytes").is_none(),
            "webhook payload must not grant download byte-limit approval"
        );
        Ok(())
    }

    #[test]
    fn cache_warm_unknown_artifact_does_not_queue_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-unknown");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "cache.warm".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("cache-warm".to_owned()),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "missing#artifact",
            }),
        };

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            |_| Ok(false),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "cache_warm_unknown_artifact");
        assert!(event.message.contains("unknown registry artifact"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn cache_warm_contained_mode_still_requires_reviewed_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-contained");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "cache.warm".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("cache-warm".to_owned()),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        };

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_| Ok(true),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "queue_prefetch_proposal");
        assert!(event.message.contains("explicit source-policy approval"));
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("prefetch_artifact"));
        Ok(())
    }

    #[test]
    fn driver_upgrade_propose_mode_queues_driver_plan_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-propose");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Propose,
            None,
        )]);
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
                "tool": "restart_server",
            }),
        })?;

        handle_driver_upgrade_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "driver-upgrade");
        assert_eq!(event.action, "prepare_driver_plan");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "driver-upgrade");
        assert_eq!(proposals[0].tool.as_deref(), Some("driver_plan"));
        assert!(
            proposals[0].arguments.get("tool").is_none(),
            "webhook payload must not grant arbitrary tool choice"
        );
        Ok(())
    }

    #[test]
    fn driver_upgrade_contained_mode_runs_restricted_driver_plan() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-contained");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox_driver_plan_value(CommandCapture {
                    argv: vec![
                        "rocm".to_owned(),
                        "install".to_owned(),
                        "driver".to_owned(),
                        "--dkms".to_owned(),
                        "--dry-run".to_owned(),
                    ],
                    exit_status: 0,
                    stdout: "driver install plan\n  supported: true\n".to_owned(),
                    stderr: String::new(),
                }))
            },
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "run_driver_plan");
        assert!(
            event
                .message
                .contains("contained restricted driver_plan status=planned")
        );
        assert!(event.message.contains("no driver commands were executed"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_contained_mode_requires_restricted_driver_plan_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-contained-tool");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(json!({
                    "tool": "check_updates",
                    "status": "checked",
                    "mutating": false,
                }))
            },
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "driver_plan_failed");
        assert!(event.message.contains("expected `driver_plan`"));
        assert!(event.message.contains("no driver commands were executed"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_observe_mode_records_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-observe");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Observe,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event(&paths, WatcherMode::Observe, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "observe_driver_update");
        assert!(event.message.contains("does not queue or run"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_ignores_non_driver_component_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-wrong-component");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("runtime version is newer".to_owned()),
            payload: json!({
                "component": "runtime",
            }),
        };

        handle_driver_upgrade_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "driver_upgrade_ignored_component");
        assert!(event.message.contains("payload.component=driver"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn event_dispatcher_preserves_server_recover_proposal_behavior() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-dispatch");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("server-recover");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let events = vec![AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some("svc-failed".to_owned()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        }];

        evaluate_watchers_for_events(&paths, &config, &mut state, &events)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-failed"));
        assert_eq!(proposals[0].tool.as_deref(), Some("restart_server"));
        assert!(proposals[0].message.contains("manifest reports failed"));
        assert!(!proposals[0].message.contains("manifest_status_failed"));
        Ok(())
    }

    #[test]
    fn server_recover_local_webhook_does_not_restart_healthy_service() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-healthy-webhook");
        paths.ensure()?;
        let mut healthy = ManagedServiceRecord::new(
            &paths,
            "svc-healthy",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        healthy.status = "ready".to_owned();
        healthy.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("server-recover");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let event = local_webhook_event_from_request(LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: Some("svc-healthy".to_owned()),
            reason: Some("manual recovery smoke".to_owned()),
            payload: json!({}),
        })?;

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let reloaded = load_service_record(&paths, "svc-healthy")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "ignore_nonrecoverable_service");
        assert!(event.message.contains("does not currently need recovery"));
        assert!(proposals.is_empty());
        assert_eq!(reloaded.status, "ready");
        Ok(())
    }

    #[test]
    fn recovery_reason_display_avoids_raw_status_tokens() {
        assert_eq!(
            display_recovery_reason("manifest_status_starting_stale"),
            "service has been starting for too long"
        );
        assert_eq!(
            display_recovery_reason("healthcheck_status_unreachable"),
            "engine healthcheck reports unreachable"
        );
        assert_eq!(
            display_recovery_reason("endpoint_status_unreachable"),
            "endpoint port is unreachable"
        );
    }

    #[test]
    fn healthcheck_readiness_requires_ready_loaded_model() {
        let ready = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(healthcheck_response_ready(&ready));

        let mut loading = ready.clone();
        loading.status = "loading_model".to_owned();
        assert!(!healthcheck_response_ready(&loading));

        let mut unloaded = ready;
        unloaded.model_loaded = false;
        assert!(!healthcheck_response_ready(&unloaded));
    }

    #[test]
    fn healthcheck_recoverability_tracks_failed_endpoint_state() {
        let mut response = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(!healthcheck_response_recoverable(&response));

        response.status = "unreachable".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "failed".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "loading_model".to_owned();
        assert!(!healthcheck_response_recoverable(&response));
    }

    #[test]
    fn apply_endpoint_key_env_sets_var_only_when_key_file_present() {
        let (_root, paths) = temp_app_paths("apply-endpoint-key-env");
        let service_id = "svc-endpoint-key-env";

        // Loopback service: no key file has ever been written, so the child's
        // environment must be left untouched.
        let mut command = ProcessCommand::new("true");
        apply_endpoint_key_env(&mut command, &paths, service_id);
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
            "loopback service must not receive the endpoint key env var"
        );

        // Public service: once a key file exists at the deterministic path,
        // it must be threaded onto the child so the engine's HTTP probe
        // authenticates.
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir()).unwrap();
        fs::write(&key_path, "secret-key").unwrap();
        let mut command = ProcessCommand::new("true");
        apply_endpoint_key_env(&mut command, &paths, service_id);
        let env_value = command
            .get_envs()
            .find_map(|(key, value)| {
                (key == rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV)
                    .then_some(value)
                    .flatten()
            })
            .expect("endpoint key env var must be set once the key file exists");
        assert_eq!(env_value, key_path.as_os_str());
    }

    #[test]
    fn manifest_recovery_policy_covers_terminal_and_stale_transient_states() {
        let (root, paths) = temp_app_paths("manifest-recovery-policy");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-stale",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.created_at_unix_ms = 1_000;

        record.status = "exited".to_owned();
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_001).as_deref(),
            Some("manifest_status_exited")
        );

        record.status = "unreachable".to_owned();
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_001).as_deref(),
            Some("manifest_status_unreachable")
        );

        record.status = "starting".to_owned();
        assert_eq!(manifest_service_recovery_reason(&record, 2_000), None);
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_000 + SERVER_TRANSIENT_STALE_MS).as_deref(),
            Some("manifest_status_starting_stale")
        );

        record.status = "recovering".to_owned();
        record.last_restart_unix_ms = Some(5_000);
        assert_eq!(manifest_service_recovery_reason(&record, 6_000), None);
        assert_eq!(
            manifest_service_recovery_reason(&record, 5_000 + SERVER_TRANSIENT_STALE_MS).as_deref(),
            Some("manifest_status_recovering_stale")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn find_recoverable_service_prefers_failed_managed_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("recoverable-service");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;

        let found = find_recoverable_service(&paths)?.expect("failed service should be found");
        fs::remove_dir_all(root).ok();
        assert_eq!(found.0.service_id, "svc-failed");
        assert_eq!(found.1, "manifest_status_failed");
        Ok(())
    }

    #[test]
    fn find_recoverable_service_detects_stale_starting_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("recoverable-stale-starting");
        paths.ensure()?;
        let mut stale = ManagedServiceRecord::new(
            &paths,
            "svc-starting",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        stale.status = "starting".to_owned();
        stale.created_at_unix_ms = 0;
        stale.write()?;

        let found =
            find_recoverable_service(&paths)?.expect("stale starting service should recover");
        fs::remove_dir_all(root).ok();
        assert_eq!(found.0.service_id, "svc-starting");
        assert_eq!(found.1, "manifest_status_starting_stale");
        Ok(())
    }

    #[test]
    fn record_event_mirrors_watcher_actions_to_audit_log() -> Result<()> {
        let (root, paths) = temp_app_paths("record-event-audit");
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "server-recover".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "recover failed managed services".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        record_event(
            &paths,
            &mut state,
            "server-recover",
            "info",
            "restart_managed_service",
            "restarted failed managed service svc-1",
            Some("svc-1".to_owned()),
        )?;

        let automation_text = fs::read_to_string(paths.automation_events_path())?;
        let automation_event =
            serde_json::from_str::<AutomationEventRecord>(automation_text.trim())?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit_event = serde_json::from_str::<AuditEventRecord>(audit_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(automation_event.watcher_id, "server-recover");
        assert_eq!(automation_event.action, "restart_managed_service");
        assert_eq!(audit_event.category, "automation");
        assert_eq!(audit_event.actor, "watcher:server-recover");
        assert_eq!(audit_event.watcher_id.as_deref(), Some("server-recover"));
        assert_eq!(audit_event.service_id.as_deref(), Some("svc-1"));
        Ok(())
    }

    #[test]
    fn server_recover_propose_mode_queues_restart_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-proposal");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "server-recover".to_owned(),
                enabled: true,
                mode: WatcherMode::Propose,
                summary: "recover".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        evaluate_server_recover(&paths, WatcherMode::Propose, &mut state)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].action, "queue_restart_proposal");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-1"));
        assert_eq!(proposals[0].status, "pending");
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_runs_read_only_check_without_queueing() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox_check_updates_value(CommandCapture {
                    argv: vec!["rocm".to_owned(), "update".to_owned()],
                    exit_status: 0,
                    stdout: "update\n  managed runtimes: none\n".to_owned(),
                    stderr: String::new(),
                }))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "therock-update");
        assert_eq!(event.action, "run_update_check");
        assert!(event.message.contains("contained read-only execution"));
        assert!(
            event
                .message
                .contains("restricted check_updates status=checked")
        );
        assert!(event.message.contains("no updates were applied"));
        assert!(!event.message.contains("fallback"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_records_update_available_without_applying() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained-available");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox_check_updates_value(CommandCapture {
                    argv: vec!["rocm".to_owned(), "update".to_owned()],
                    exit_status: 0,
                    stdout: "update\n  runtime release-pip-gfx120x-all status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
                    stderr: String::new(),
            }))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let events = event_text
            .lines()
            .map(serde_json::from_str::<AutomationEventRecord>)
            .collect::<Result<Vec<_>, _>>()?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        let update_check = events
            .iter()
            .find(|event| event.action == "run_update_check")
            .expect("update check event should be recorded");
        assert_eq!(update_check.watcher_id, "therock-update");
        assert!(
            update_check
                .message
                .contains("restricted check_updates status=update_available")
        );
        assert!(
            update_check
                .message
                .contains("a ROCm runtime update is available")
        );
        assert!(update_check.message.contains("no updates were applied"));
        assert!(!update_check.message.contains("fallback"));
        let notification = events
            .iter()
            .find(|event| event.action == "notify_if_newer")
            .expect("notify-if-newer event should be recorded");
        assert_eq!(notification.watcher_id, "therock-update");
        assert!(
            notification
                .message
                .contains("ROCm runtime update is available")
        );
        assert!(notification.message.contains("No updates were applied"));
        assert!(audit_text.contains("\"category\":\"notification\""));
        assert!(audit_text.contains("\"action\":\"notify_if_newer\""));
        assert!(audit_text.contains("ROCm runtime update is available"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_uses_restricted_check_updates_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained-tool");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(json!({
                    "tool": "examine_snapshot",
                    "status": "captured",
                    "mutating": false,
                }))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "therock-update");
        assert_eq!(event.action, "update_check_failed");
        assert!(event.message.contains("expected `check_updates`"));
        assert!(event.message.contains("no updates were applied"));
        Ok(())
    }

    #[test]
    fn therock_update_notify_if_newer_uses_restricted_notification_contract() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-notify-contract");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        record_update_available_notification(&paths, &mut state)?;

        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit = audit_text
            .lines()
            .map(serde_json::from_str::<AuditEventRecord>)
            .collect::<Result<Vec<_>, _>>()?;
        fs::remove_dir_all(root).ok();

        let notification = audit
            .iter()
            .find(|event| event.category == "notification" && event.action == "notify_if_newer")
            .expect("notify_if_newer audit should be recorded");
        assert_eq!(notification.category, "notification");
        assert_eq!(notification.actor, "watcher:therock-update");
        assert_eq!(notification.watcher_id.as_deref(), Some("therock-update"));
        assert!(
            notification
                .message
                .contains("ROCm runtime update is available")
        );
        assert!(notification.message.contains("No updates were applied"));
        Ok(())
    }

    #[test]
    fn sandbox_check_updates_value_is_read_only_and_preserves_output() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release status=up_to_date\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("checked"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no updates were applied"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=up_to_date"))
        );
    }

    #[test]
    fn sandbox_check_updates_value_marks_runtime_update_available() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release-pip-gfx120x-all status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("update_available")
        );
        assert_eq!(
            value.get("update_available").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(
                    |message| message.contains("a ROCm runtime update is available")
                        && message.contains("no updates were applied")
                )
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=update_available"))
        );
    }

    #[test]
    fn sandbox_driver_plan_value_is_read_only_and_preserves_output() {
        let value = sandbox_driver_plan_value(CommandCapture {
            argv: vec![
                "rocm".to_owned(),
                "install".to_owned(),
                "driver".to_owned(),
                "--dkms".to_owned(),
                "--dry-run".to_owned(),
            ],
            exit_status: 0,
            stdout: "driver install plan\n  supported: true\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("driver_plan")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("planned"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no driver commands were executed"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("driver install plan"))
        );
    }

    #[test]
    fn sandbox_tool_cli_values_cover_restricted_plan_api() {
        let names = [
            SandboxToolArg::CheckUpdates,
            SandboxToolArg::ExamineSnapshot,
            SandboxToolArg::ListServers,
            SandboxToolArg::RestartServer,
            SandboxToolArg::StopServer,
            SandboxToolArg::PrefetchArtifact,
            SandboxToolArg::NotifyUser,
            SandboxToolArg::DriverPlan,
        ]
        .into_iter()
        .map(SandboxToolArg::as_cli_value)
        .collect::<Vec<_>>();

        for expected in [
            "check_updates",
            "examine_snapshot",
            "list_servers",
            "restart_server",
            "stop_server",
            "prefetch_artifact",
            "notify_user",
            "driver_plan",
        ] {
            assert!(names.contains(&expected), "missing sandbox tool {expected}");
        }
    }

    #[test]
    fn sandbox_tool_examine_snapshot_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-examine-snapshot");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ExamineSnapshot,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("examine_snapshot")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("captured")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(value.get("examine").is_some());
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_returns_records() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers");
        paths.ensure()?;
        let record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.write()?;

        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("listed"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(1));
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_first_run_returns_empty_list() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers-empty");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(0));
        assert_eq!(
            value
                .get("services")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_restart() {
        let (root, paths) = temp_app_paths("sandbox-restart-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("restart_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_restart_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-restart-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_stop() {
        let (root, paths) = temp_app_paths("sandbox-stop-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("stop_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_stop_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-stop-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_stop_server_updates_manifest_and_skips_current_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-current-pid");
        paths.ensure()?;
        let current_pid = std::process::id();
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-current",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            Some("svc-current".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        let reloaded = load_service_record(&paths, "svc-current")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("stopped"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(reloaded.status, "stopped");
        assert!(
            value
                .get("result")
                .and_then(|result| result.get("skipped_pids"))
                .and_then(Value::as_array)
                .is_some_and(|pids| pids
                    .iter()
                    .any(|pid| pid.as_u64() == Some(u64::from(current_pid))))
        );
        Ok(())
    }

    #[test]
    fn stop_managed_service_removes_endpoint_key_file() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-removes-endpoint-key");
        paths.ensure()?;
        let current_pid = std::process::id();
        let service_id = "svc-endpoint-key-stop";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "0.0.0.0",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir())?;
        fs::write(&key_path, "secret-key")?;
        assert!(key_path.exists());

        let result = stop_managed_service(&paths, service_id);
        // Observe the real filesystem state before the blanket temp-dir cleanup,
        // otherwise remove_dir_all would delete the key file and mask a missing
        // production cleanup (the regression this test guards).
        let key_removed = !key_path.exists();
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value
                .get("service")
                .and_then(|service| service.get("status"))
                .and_then(Value::as_str),
            Some("stopped")
        );
        assert!(key_removed, "endpoint key file must be removed after stop");
        Ok(())
    }

    #[test]
    fn stop_managed_service_without_endpoint_key_file_succeeds() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-no-endpoint-key");
        paths.ensure()?;
        let current_pid = std::process::id();
        let service_id = "svc-no-endpoint-key-stop";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        // Loopback service: no endpoint key file was ever written for it.
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        assert!(!key_path.exists());

        let result = stop_managed_service(&paths, service_id);
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value
                .get("service")
                .and_then(|service| service.get("status"))
                .and_then(Value::as_str),
            Some("stopped")
        );
        assert_eq!(reloaded?.status, "stopped");
        assert!(!key_path.exists());
        Ok(())
    }

    #[test]
    fn stop_server_process_tree_discovers_descendants_before_parents() {
        let output = "\
10 1
11 10
12 11
13 10
20 1
21 20
";

        assert_eq!(
            descendant_pids_from_ps_output(output, &[10]),
            vec![12, 11, 13]
        );
        assert_eq!(
            descendant_pids_from_ps_output(output, &[10, 20]),
            vec![12, 11, 13, 21]
        );
    }

    #[test]
    fn sandbox_tool_notify_user_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-notify-user");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("ROCm setup is ready.".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("notify_user")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("notified")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("message").and_then(Value::as_str),
            Some("ROCm setup is ready.")
        );
        assert!(audit_text.contains("\"category\":\"notification\""));
        assert!(audit_text.contains("ROCm setup is ready."));
        Ok(())
    }

    #[test]
    fn sandbox_runner_native_fallback_records_audit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-native-audit");
        paths.ensure()?;

        let value = run_native_restricted_sandbox(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("isolation").and_then(Value::as_str),
            Some("native_restricted")
        );
        assert!(audit_text.contains("\"category\":\"sandbox\""));
        assert!(audit_text.contains("\"action\":\"notify_user\""));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_command_separates_child_args_from_bwrap_options() {
        let mut command = ProcessCommand::new("bwrap");
        append_sandbox_tool_command_args(
            &mut command,
            Path::new("/tmp/rocmd"),
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello"),
            SandboxToolPolicy::default(),
        );
        let args = command
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let separator = args
            .iter()
            .position(|arg| arg == "--")
            .expect("bubblewrap command should separate child command args");

        assert_eq!(
            args.get(separator + 1).map(String::as_str),
            Some("/tmp/rocmd")
        );
        assert_eq!(
            args.get(separator + 4).map(String::as_str),
            Some("--message")
        );
        assert_eq!(args.get(separator + 5).map(String::as_str), Some("hello"));
    }

    #[test]
    fn sandbox_prefetch_requires_artifact_ref() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-requires-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("prefetch_artifact requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_prefetch_reports_policy_required_without_network() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-policy");
        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            ModelRecipeArtifactRecord {
                artifact_id: "hf-main".to_owned(),
                kind: "huggingface".to_owned(),
                uri: "Qwen/Test-1B".to_owned(),
                revision: Some("main".to_owned()),
                sha256: Some("a".repeat(64)),
                size_bytes: Some(1024),
                license: Some("apache-2.0".to_owned()),
                gated: Some(false),
                quantization: Some("bf16".to_owned()),
                engines: vec!["vllm".to_owned()],
                source_policy: None,
            },
            SandboxToolPolicy::default(),
        )
        .expect("policy value should render");
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("source_policy_required")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("cache")
                .and_then(|cache| cache.get("status"))
                .and_then(Value::as_str)
                .is_some_and(|status| status == "missing")
        );
    }

    #[test]
    fn sandbox_prefetch_unknown_artifact_ref_errors() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-unknown-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            Some("Qwen/Missing#hf-main".to_owned()),
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(error.to_string().contains("was not found"), "{error:#}");
    }

    #[test]
    fn sandbox_prefetch_cached_marker_skips_network() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-cached");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/should-not-be-fetched.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };
        let cache = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);
        write_file_atomically(&cache.marker_path, br#"{"cached":true}"#)?;

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("cached"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_artifact_larger_than_approved_limit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-byte-limit");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/too-large.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(2048),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("byte limit"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_non_direct_non_huggingface_source() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-non-direct");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "torrent-bin".to_owned(),
            kind: "torrent".to_owned(),
            uri: "https://example.invalid/artifact.torrent".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#torrent-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("direct HTTP(S)"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_gated_huggingface_without_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_huggingface_source_policy_without_token() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-token");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: None,
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("ROCM_CLI_HUGGINGFACE_TOKEN"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_manual_only_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-manual-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "manual-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/manual.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(ModelRecipeArtifactSourcePolicyRecord {
                policy: "manual_only".to_owned(),
                required_hosts: Vec::new(),
                notes: vec!["requires manual license review".to_owned()],
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#manual-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("manual-only"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_declared_huggingface_auth_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-declared-auth");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(ModelRecipeArtifactSourcePolicyRecord {
                policy: "huggingface_authenticated".to_owned(),
                required_hosts: vec!["huggingface.co".to_owned()],
                notes: Vec::new(),
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_to_non_huggingface_url() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-host");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://example.invalid/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("non-Hugging Face URL"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_over_plain_http() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-https");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "http://huggingface.co/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("plain HTTP"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_downloads_direct_artifact_when_approved() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-download");
        let bytes = b"tiny verified artifact".to_vec();
        let uri = serve_one_http_response(bytes.clone())?;
        let sha256 = sha256_hex(&bytes);
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri,
            revision: None,
            sha256: Some(sha256.clone()),
            size_bytes: Some(bytes.len() as u64),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        let bytes_path = value
            .get("bytes_path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("bytes_path should be present");
        let marker_path = value
            .get("cache")
            .and_then(|cache| cache.get("marker_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("marker path should be present");

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("prefetched")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            value.get("sha256").and_then(Value::as_str),
            Some(sha256.as_str())
        );
        assert_eq!(
            value.get("source_policy").and_then(Value::as_str),
            Some("explicit_allow_artifact_download")
        );
        assert_eq!(fs::read(&bytes_path)?, bytes);
        assert!(marker_path.is_file());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_approved_artifact_without_sha256() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-no-sha");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/artifact.bin".to_owned(),
            revision: None,
            sha256: None,
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("sha256"))
        );
        Ok(())
    }

    #[test]
    fn huggingface_url_detection_is_host_scoped() {
        assert!(is_huggingface_url(
            "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors"
        ));
        assert!(is_huggingface_url(
            "https://cdn-lfs.huggingface.co/repos/example"
        ));
        assert!(is_huggingface_url("https://hf.co/Qwen/Test-1B"));
        assert!(!is_huggingface_url(
            "https://huggingface.co.evil.example/Qwen/Test-1B"
        ));
        assert!(!is_huggingface_url(
            "https://huggingface.co@evil.example/Qwen/Test-1B"
        ));
    }

    fn serve_one_http_response(bytes: Vec<u8>) -> Result<String> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let addr = listener.local_addr()?;
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let header = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    bytes.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&bytes);
            }
        });
        Ok(format!("http://{addr}/artifact.bin"))
    }

    fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = unique_test_root(&format!(
            "rocmd-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    fn unique_test_root(label: &str) -> PathBuf {
        let root = workspace_test_artifact_dir().join(label);
        fs::create_dir_all(&root).expect("create workspace-local test root");
        root
    }

    fn unique_test_path(label: &str) -> PathBuf {
        let root = workspace_test_artifact_dir();
        fs::create_dir_all(&root).expect("create workspace-local test dir");
        root.join(label)
    }

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("rocmd")
    }

    fn test_watcher_snapshot(
        id: &str,
        mode: WatcherMode,
        last_event_unix_ms: Option<u128>,
    ) -> WatcherRuntimeSnapshot {
        WatcherRuntimeSnapshot {
            id: id.to_owned(),
            enabled: true,
            mode,
            summary: "test watcher".to_owned(),
            last_event: None,
            last_event_unix_ms,
        }
    }

    fn test_runtime_state(active_watchers: Vec<WatcherRuntimeSnapshot>) -> AutomationRuntimeState {
        AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers,
        }
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    match rocm_engine_protocol::GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg)? {
        rocm_engine_protocol::GpuSelection::Auto => Ok(Vec::new()),
        rocm_engine_protocol::GpuSelection::Index(index) => Ok(vec![index]),
    }
}

fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

/// Keep only the final visible segment of a `\r`-redrawn progress line.
///
/// Progress tools (pip, tqdm, Hugging Face) redraw a line in place with a bare
/// carriage return and no newline, so the segment after the last `\r` is its
/// final visible state. Lines without `\r` pass through unchanged. (Same
/// collapse rule the dashboard job console applies to streamed job output.)
fn last_cr_segment(line: &str) -> &str {
    line.rsplit('\r').next().unwrap_or(line)
}

/// Classify a single serve-log line into a coarse startup phase token
/// (`downloading`/`loading`/`warmup`), or `None` when the line carries no phase
/// signal. Case-insensitive substring match over the common vLLM / llama.cpp /
/// Hugging Face startup vocabulary. Checked warmup → loading → downloading so
/// the latest lifecycle stage a line mentions wins.
fn classify_startup_phase(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("capturing cuda graph")
        || lower.contains("capturing the model")
        || lower.contains("warming up")
        || lower.contains("warmup")
    {
        Some("warmup")
    } else if lower.contains("loading weights")
        || lower.contains("loading model")
        || lower.contains("load_tensors")
        || lower.contains("llama_model_loader")
        || lower.contains("model loading took")
    {
        Some("loading")
    } else if lower.contains("downloading") || lower.contains("fetching") {
        Some("downloading")
    } else {
        None
    }
}

/// Read log bytes appended since `*pos`, advance `*pos`, and return the most
/// recent recognizable startup phase in that new output (later lines win, so a
/// download → load → warmup progression advances naturally).
///
/// Best-effort: any I/O error (file not created yet, transient read) yields
/// `None`. A shrunk file (rotation/truncation) resets the cursor to the top.
fn read_new_log_phase(log_path: &Path, pos: &mut u64) -> Option<&'static str> {
    let mut file = fs::File::open(log_path).ok()?;
    let len = file.metadata().ok()?.len();
    if len < *pos {
        *pos = 0;
    }
    if len == *pos {
        return None;
    }
    file.seek(SeekFrom::Start(*pos)).ok()?;
    let mut bytes = Vec::new();
    let read = file.read_to_end(&mut bytes).ok()?;
    *pos += read as u64;
    let text = String::from_utf8_lossy(&bytes);
    let mut phase = None;
    for line in text.lines() {
        if let Some(found) = classify_startup_phase(last_cr_segment(line)) {
            phase = Some(found);
        }
    }
    phase
}

/// Poll a freshly-spawned service until its healthcheck reports ready (or the
/// timeout elapses), tailing its log file meanwhile and reporting each coarse
/// startup phase transition via `on_phase`.
fn wait_for_service_ready(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    log_path: &Path,
    timeout: Duration,
    mut on_phase: impl FnMut(&str),
) -> bool {
    let start = std::time::Instant::now();
    let mut log_pos: u64 = 0;
    let mut last_phase: Option<&'static str> = None;
    while start.elapsed() < timeout {
        if let Some(phase) = read_new_log_phase(log_path, &mut log_pos)
            && last_phase != Some(phase)
        {
            last_phase = Some(phase);
            on_phase(phase);
        }
        if engine_healthcheck_ready(paths, engine, service_id).unwrap_or(false) {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

fn engine_healthcheck_ready(paths: &AppPaths, engine: &str, service_id: &str) -> Result<bool> {
    Ok(healthcheck_response_ready(&engine_healthcheck_response(
        paths, engine, service_id,
    )?))
}

fn engine_healthcheck_response(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Result<HealthcheckResponse> {
    engine_request::<_, HealthcheckResponse>(
        paths,
        engine,
        service_id,
        EngineMethod::Healthcheck,
        &HealthcheckRequest {
            service_id: service_id.to_owned(),
        },
    )
}

fn healthcheck_response_ready(response: &HealthcheckResponse) -> bool {
    response.status == "ready" && response.model_loaded
}

fn healthcheck_response_recoverable(response: &HealthcheckResponse) -> bool {
    matches!(
        response.status.as_str(),
        "failed" | "unreachable" | "exited"
    )
}

/// Re-thread the endpoint key file (public bind only) onto an engine child's
/// environment, mirroring the initial `rocm serve` spawn. `None` for a
/// loopback service with no stored key leaves the command's environment
/// untouched, matching the unauthenticated default.
fn apply_endpoint_key_env(command: &mut ProcessCommand, paths: &AppPaths, service_id: &str) {
    if let Some(key_file) = rocm_engine_protocol::endpoint_key_file_if_present(paths, service_id) {
        command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
    }
}

fn engine_request<T, R>(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request).context("failed to serialize engine request")?,
    };
    let engine_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let mut command = ProcessCommand::new(&engine_binary);
    command
        .arg("__engine-stdio")
        .arg(engine)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // A protected public endpoint enforces its key on every request, including
    // this stdio-transported healthcheck — without the carrier the engine
    // adapter's HTTP probe is unauthenticated and the endpoint looks
    // unreachable, which would misclassify a healthy protected service as
    // recoverable.
    apply_endpoint_key_env(&mut command, paths, service_id);
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn engine stdio process {}",
            engine_binary.display()
        )
    })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .context("engine stdio child did not expose stdin")?;
        serde_json::to_writer(&mut *stdin, &envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
    }

    let output = child
        .wait_with_output()
        .context("failed waiting for engine stdio response")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {}", output.status);
        }
        bail!(
            "engine stdio process exited with status {}: {}",
            output.status,
            stderr
        );
    }
    let envelope: EngineResponseEnvelope =
        serde_json::from_slice(&output.stdout).context("failed to parse engine response")?;
    if !envelope.ok {
        let detail = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |error| format!("{}: {}", error.code, error.message),
        );
        bail!(detail);
    }
    let data = envelope
        .data
        .context("engine response did not include data")?;
    serde_json::from_value(data).context("failed to deserialize engine response data")
}

fn wait_for_port(host: &str, port: u16, timeout: Duration) -> bool {
    let address: SocketAddr = match format_host_port(host, port).parse() {
        Ok(value) => value,
        Err(_) => return false,
    };

    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}
