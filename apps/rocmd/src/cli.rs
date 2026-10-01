// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{AppPaths, DEFAULT_LOCAL_HOST};
use serde_json::Value;
use serde_json::json;
use std::ffi::OsString;

#[derive(Parser, Debug)]
#[command(name = "rocmd", about = "rocm-cli local supervisor", version)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
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
pub(crate) enum SandboxToolArg {
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
    pub(crate) const fn as_cli_value(self) -> &'static str {
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
    pub(crate) const fn writes_data(self) -> bool {
        matches!(
            self,
            Self::RestartServer | Self::StopServer | Self::NotifyUser
        )
    }

    #[cfg(target_os = "linux")]
    pub(crate) const fn writes_cache(self) -> bool {
        matches!(self, Self::CheckUpdates | Self::PrefetchArtifact)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SandboxToolPolicy {
    pub(crate) allow_artifact_download: bool,
    pub(crate) artifact_max_bytes: Option<u64>,
    pub(crate) allow_huggingface_download: bool,
    pub(crate) huggingface_token: Option<String>,
}

impl SandboxToolPolicy {
    pub(crate) fn from_cli(
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
        } => crate::run_daemon(&paths, automations_enabled, local_webhook_port).await?,
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
        } => crate::supervise_service(
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
            crate::print_status(&paths)?;
        }
        Command::BridgeSnapshot { pretty } => {
            crate::common::print_bridge_snapshot(&paths, pretty)?;
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
            let value = crate::run_sandbox_runner(
                &paths,
                tool,
                service_id,
                artifact_ref,
                message,
                allow_native_fallback,
                policy,
            )?;
            crate::print_json(&value)?;
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
            let value =
                crate::run_sandbox_tool(&paths, tool, service_id, artifact_ref, message, policy)?;
            crate::print_json(&value)?;
        }
        Command::McpServer => {
            crate::run_mcp_server(&paths)?;
        }
        Command::McpToolsJson => {
            crate::print_json(&json!({ "tools": crate::rocm_mcp_tools() }))?;
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
            crate::ensure_direct_mcp_call_allowed(&name, allow_mutation)?;
            let result = crate::handle_mcp_tool_call(
                &paths,
                &json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )?;
            crate::print_json(&result)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

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
}
