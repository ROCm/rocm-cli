// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

mod automations;
mod bootstrap;
mod comfyui;
mod dash;
mod dash_seam;
mod provider_keys;
mod providers;
mod therock;
mod tui;
mod uninstall;

// Per-command handler fns mechanically relocated into modules.
// Dispatch call sites stay byte-identical via these re-imports (upstream-sync
// mergeability); only the fn definitions moved out of main.rs.
use crate::automations::automations;
use crate::uninstall::uninstall;

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, AuditEventRecord, AutomationEventRecord, AutomationProposalRecord,
    AutomationRuntimeState, CodexBridgeEngine, CodexBridgeGpuSnapshot, CodexBridgeSnapshot,
    DEFAULT_LOCAL_HOST, ExamineSummary, ManagedServiceRecord, ModelRecipeRecord,
    ModelRecipeRegistry, ModelRecipeRegistrySource, PERMISSIONS_MODE_ASK,
    PERMISSIONS_MODE_FULL_ACCESS, RocmCliConfig, TELEMETRY_MODE_LOCAL, TELEMETRY_MODE_OFF,
    WatcherMode, append_audit_event, builtin_model_recipes, builtin_watcher, builtin_watchers,
    connect_tcp_stream, daemon_binary_path, default_engine_for_platform,
    default_interactive_shell_program, engine_binary_path, engine_plugin_dirs, format_host_port,
    format_http_base_url, generate_service_id, interactive_terminal, load_model_recipe_registry,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
    managed_pip_cache_dir, managed_service_endpoint_model_ready, model_artifact_cache_status,
    prepend_runtime_path, process_is_running, read_tcp_stream_to_string,
    resolve_builtin_model_recipe, resolve_model_recipe, runtime_install_root_is_protected,
    runtime_path_is_same_or_inside, runtime_python_activation_hint, runtime_python_env_bin_dir,
    runtime_python_executable_in_env, shell_command_for_host, write_all_tcp_stream,
};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DetectRequest, DetectResponse, DevicePolicy,
    ENGINE_RECIPE_CONTRACT_VERSION, EngineMethod, EnginePluginDescriptor, EngineRecipeEndpointHint,
    EngineRecipeHint, EngineRecipeUnsupportedCombinationHint, EngineRequestEnvelope,
    EngineResponseEnvelope, InstallRequest, InstallResponse, ResolveModelRequest,
    ResolveModelResponse, StopRequest, StopResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

static BUILTIN_ENGINE_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Parser, Debug)]
#[command(name = "rocm", about = "ROCm AI Command Center CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Check this computer's GPU, ROCm install, engines, and setup folders.
    Examine {
        /// Emit the machine-readable Examination JSON (for diagnosis tooling).
        #[arg(long)]
        json: bool,
    },
    /// Diagnose known ROCm/PyTorch/llama.cpp failure modes against a closed catalog.
    Diagnose {
        /// Raw error text from the user; sharpens keyword scoring.
        #[arg(long)]
        symptom: Option<String>,
        /// Show at most this many matches (default 5).
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Emit the machine-readable diagnosis JSON.
        #[arg(long)]
        json: bool,
    },
    /// Apply a known fix by id (see `rocm diagnose`); run with no id to list fixes.
    Fix {
        /// Fix id, e.g. fix-4-render-group. Omit to list available fixes.
        fix_id: Option<String>,
        /// Skip the interactive confirmation (use after approving the plan).
        #[arg(long)]
        yes: bool,
        /// Show the plan without changing anything.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// For fix-9-igpu-dgpu: the discrete GPU index to pin.
        #[arg(long)]
        device_index: Option<i64>,
    },
    /// Print the rocm-cli version.
    Version,
    /// Generate a shell completion script for the given shell.
    Completions {
        /// Shell to generate completions for.
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    #[command(hide = true)]
    Bootstrap {
        #[command(subcommand)]
        command: Option<bootstrap::BootstrapCommand>,
    },
    /// Manage first-time setup state.
    Setup {
        #[command(subcommand)]
        command: Option<SetupCommand>,
    },
    #[command(name = "__engine-serve-http", hide = true)]
    EngineServeHttp {
        engine: String,
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
        host: String,
        #[arg(long, default_value_t = rocm_core::DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        log_path: Option<PathBuf>,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
    #[command(name = "__engine-stdio", hide = true)]
    EngineStdio { engine: String },
    #[command(name = "status", hide = true)]
    InternalStatus,
    #[command(name = "bridge-snapshot", hide = true)]
    InternalBridgeSnapshot {
        #[arg(long)]
        pretty: bool,
    },
    #[command(name = "sandbox-run", hide = true)]
    InternalSandboxRun {
        #[arg(value_enum)]
        tool: SandboxToolArg,
        #[arg(long)]
        service_id: Option<String>,
        #[arg(long)]
        allow_native_fallback: bool,
    },
    #[command(name = "mcp-call", hide = true)]
    McpCall {
        name: String,
        #[arg(long)]
        arguments_json: Option<String>,
        #[arg(long, conflicts_with = "arguments_json")]
        arguments_file: Option<PathBuf>,
        #[arg(long)]
        allow_mutation: bool,
    },
    /// Chat with an assistant provider.
    Chat {
        /// Assistant provider to use.
        #[arg(long)]
        provider: Option<Provider>,
        /// Model name to request from the provider.
        #[arg(long)]
        model: Option<String>,
        /// Prompt to send. If omitted, rocm-cli reads from standard input when possible.
        #[arg(long)]
        prompt: Option<String>,
        #[arg(
            long,
            help = "Allow an OpenAI-compatible provider to request ROCm tool calls."
        )]
        tools: bool,
    },
    /// Install ROCm, drivers, or related local AI components.
    Install {
        #[command(subcommand)]
        target: InstallTarget,
    },
    /// Check for a newer ROCm package and optionally install it.
    Update {
        /// Install the selected update instead of only checking.
        #[arg(long)]
        apply: bool,
        /// Runtime key to update.
        #[arg(long, requires = "apply")]
        runtime: Option<String>,
        /// Use the updated ROCm install as the default after installing it.
        #[arg(long, requires = "apply")]
        activate: bool,
        /// Show what would happen without changing files.
        #[arg(long, requires = "apply")]
        dry_run: bool,
    },
    /// List, choose, add, or remove ROCm installs.
    Runtimes {
        #[command(subcommand)]
        command: Option<RuntimesCommand>,
    },
    /// List, install, or open shells for local model engines.
    Engines {
        #[command(subcommand)]
        command: EnginesCommand,
    },
    /// Show recommended local models and what this machine can run.
    #[command(alias = "models")]
    Model {
        /// Show detailed recipe diagnostics.
        #[arg(long)]
        verbose: bool,
    },
    /// Start a local model server.
    Serve {
        /// Model name, alias, or local model file path.
        model: String,
        /// Engine to use, such as lemonade, pytorch, or llama.cpp.
        #[arg(long)]
        engine: Option<String>,
        /// Device policy. Use gpu_required for ROCm GPU execution.
        #[arg(long)]
        device: Option<String>,
        /// ROCm runtime key to use for this server.
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        /// Engine environment id to use for this server.
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        /// Host address to bind.
        #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
        host: String,
        /// TCP port to bind.
        #[arg(long, default_value_t = rocm_core::DEFAULT_LOCAL_PORT)]
        port: u16,
        /// Run in this terminal instead of as a managed background server.
        #[arg(long, conflicts_with = "managed")]
        foreground: bool,
        /// Keep the server managed by ROCm CLI.
        #[arg(long)]
        managed: bool,
        /// Allow binding to a non-local address.
        #[arg(long)]
        allow_public_bind: bool,
    },
    /// Install, start, stop, or inspect ComfyUI.
    #[command(alias = "comfy")]
    Comfyui {
        #[command(subcommand)]
        command: Option<ComfyuiCommand>,
    },
    /// Show or control local model servers.
    Services {
        #[command(subcommand)]
        command: Option<ServicesCommand>,
    },
    /// Manage optional background checks and review requests.
    Automations {
        #[command(subcommand)]
        command: Option<AutomationsCommand>,
    },
    /// Show or change saved ROCm CLI settings.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Browse recent ROCm CLI logs.
    Logs {
        /// Show logs for a managed service id.
        #[arg(long)]
        service: Option<String>,
        /// Search recent logs for one or more words.
        #[arg(long, num_args = 1.., value_name = "QUERY")]
        search: Vec<String>,
        /// Optional free-text search query.
        #[arg(value_name = "QUERY")]
        query: Vec<String>,
    },
    /// Start the background helper in the foreground.
    Daemon {
        /// Print the automation status panel instead of running the helper loop.
        #[arg(long)]
        status: bool,
    },
    /// Launch the unified telemetry dashboard (TUI) with an embedded daemon.
    Dash {
        /// Replay a recorded session NDJSON instead of connecting to a live daemon.
        #[arg(long, value_name = "FILE")]
        replay: Option<PathBuf>,
        /// Show a deterministic synthetic demo session (no GPU or daemon needed).
        #[arg(long, conflicts_with = "replay")]
        demo: bool,
        /// Use the offline mock chat backend (no live LLM).
        #[arg(long)]
        chat_mock: bool,
    },
    /// Remove ROCm CLI-managed files from this computer.
    Uninstall {
        /// Do not ask for interactive confirmation.
        #[arg(long)]
        yes: bool,
        /// Show what would be removed without deleting anything.
        #[arg(long)]
        dry_run: bool,
        /// Keep installed rocm-cli binaries.
        #[arg(long)]
        keep_binaries: bool,
        /// Keep saved settings.
        #[arg(long)]
        keep_config: bool,
        /// Keep app data such as logs, services, and engines.
        #[arg(long)]
        keep_data: bool,
        /// Keep caches.
        #[arg(long)]
        keep_cache: bool,
        /// Allow removing development binaries inside the current build tree.
        #[arg(long)]
        force_dev_binaries: bool,
    },
}

#[derive(Subcommand, Debug)]
enum InstallTarget {
    /// Install TheRock ROCm wheels into a Python folder managed by ROCm CLI.
    Sdk {
        /// Package channel to install.
        #[arg(long, default_value = "release")]
        channel: String,
        /// Package format to install.
        #[arg(long, default_value = "wheel")]
        format: InstallFormat,
        /// Full folder path where the ROCm Python environment should be created.
        #[arg(long)]
        prefix: Option<std::path::PathBuf>,
        /// Exact TheRock package version to install.
        #[arg(long, conflicts_with = "build_date")]
        version: Option<String>,
        /// Pick the TheRock package built on this date.
        #[arg(long, value_name = "YYYY-MM-DD", conflicts_with = "version")]
        build_date: Option<String>,
        /// TheRock GPU package family to install, such as gfx110X-all.
        #[arg(long)]
        family: Option<String>,
        /// Resolve the install plan without changing files.
        #[arg(long)]
        dry_run: bool,
    },
    /// Preview or install Linux AMD driver support.
    Driver {
        /// Use the DKMS driver wrapper flow.
        #[arg(long)]
        dkms: bool,
        /// Apply without asking.
        #[arg(long)]
        yes: bool,
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Check and repair driver setup where supported.
        #[arg(long)]
        reconcile: bool,
    },
}

#[derive(Subcommand, Debug)]
enum EnginesCommand {
    /// Show local model engines and whether they are ready.
    List,
    /// Install the selected engine into ROCm CLI's managed engine folder.
    Install {
        /// Engine name, such as lemonade, pytorch, or llama.cpp.
        engine: String,
        /// ROCm runtime key to install against.
        #[arg(long)]
        runtime_id: Option<String>,
        /// Python version to use for engine setup.
        #[arg(long)]
        python_version: Option<String>,
        /// Reinstall even if the engine already exists.
        #[arg(long)]
        reinstall: bool,
    },
    /// Open a shell with the selected engine environment activated.
    Shell {
        /// Engine name.
        engine: String,
        /// ROCm runtime key to use.
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        /// Engine environment id to use.
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        /// Shell executable to launch.
        #[arg(long)]
        shell: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum RuntimesCommand {
    /// Show ROCm installs known to ROCm CLI.
    List,
    /// Use the selected ROCm install by default.
    Activate {
        /// Runtime key or friendly runtime selector.
        runtime: String,
    },
    /// Switch back to the previously selected ROCm install.
    Rollback,
    /// Remove a ROCm install from ROCm CLI.
    #[command(alias = "remove")]
    Uninstall {
        /// Runtime key or friendly runtime selector.
        runtime: String,
    },
    /// Add a ROCm install from a saved manifest file.
    Import {
        /// Manifest file path.
        manifest: PathBuf,
        /// Replace an existing record with the same key.
        #[arg(long)]
        replace: bool,
    },
    /// Add an existing Python ROCm folder without modifying it.
    Adopt {
        /// Python executable inside the existing ROCm environment.
        #[arg(long)]
        python: PathBuf,
        /// Optional SDK root path.
        #[arg(long)]
        root: Option<PathBuf>,
        /// Runtime id to assign.
        #[arg(long)]
        runtime_id: Option<String>,
        /// Runtime key to assign.
        #[arg(long)]
        runtime_key: Option<String>,
        /// Channel label to assign.
        #[arg(long)]
        channel: Option<String>,
        /// Replace an existing record with the same key.
        #[arg(long)]
        replace: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ComfyuiCommand {
    /// Show whether ComfyUI is installed or running.
    Status,
    /// Print the ComfyUI models folder path.
    #[command(name = "models-path", alias = "models")]
    ModelsPath,
    /// Show recent ComfyUI logs.
    Logs {
        /// Number of recent log lines to show.
        #[arg(long, default_value_t = 80)]
        lines: usize,
    },
    /// Install ComfyUI into ROCm CLI's app folder.
    Install {
        /// ROCm runtime key to use.
        #[arg(long)]
        runtime_id: Option<String>,
        /// Reinstall even if ComfyUI already exists.
        #[arg(long)]
        reinstall: bool,
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
    },
    /// Start ComfyUI and print its local URL.
    Start {
        /// Host address to bind.
        #[arg(long, default_value = comfyui::default_host())]
        host: String,
        /// TCP port to bind.
        #[arg(long, default_value_t = comfyui::default_port())]
        port: u16,
        /// Do not try to open a browser window.
        #[arg(long)]
        no_open_browser: bool,
    },
    /// Stop a ROCm CLI-managed ComfyUI server.
    Stop,
}

#[derive(Subcommand, Debug)]
enum ServicesCommand {
    /// Show currently running local model servers.
    List {
        /// Include failed, stopped, and old service records.
        #[arg(short, long)]
        all: bool,
    },
    /// Show logs for a local model server.
    Logs {
        /// Service id from `rocm services list`.
        service_id: String,
    },
    /// Stop a local model server.
    Stop {
        /// Service id from `rocm services list`.
        service_id: String,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
    },
    /// Restart a local model server.
    Restart {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AutomationsCommand {
    /// Show background checks and pending review requests.
    List,
    /// Enable a background check.
    Enable {
        /// Background check id.
        watcher: String,
        /// How the check should behave.
        #[arg(long)]
        mode: Option<WatcherModeArg>,
    },
    /// Disable a background check.
    Disable {
        /// Background check id.
        watcher: String,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    /// Show saved settings.
    Show,
    /// Set the preferred ROCm install for one engine.
    SetEngine {
        /// Engine name.
        engine: String,
        /// ROCm runtime key to use.
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        /// Engine environment id to use.
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        /// Clear this engine's preferred ROCm install.
        #[arg(long)]
        clear: bool,
    },
    /// Choose the default local model engine.
    SetDefaultEngine {
        /// Engine name.
        engine: String,
    },
    /// Clear the saved default engine.
    ClearDefaultEngine,
    /// Choose the default ROCm install.
    SetDefaultRuntime {
        /// Runtime key from `rocm runtimes list`.
        runtime_id: String,
    },
    /// Clear the saved default ROCm install.
    ClearDefaultRuntime,
    /// Choose local GPU telemetry mode.
    SetTelemetry {
        /// Telemetry mode.
        mode: TelemetryModeArg,
    },
    /// Choose the assistant permissions mode (ask vs full access).
    SetPermissions {
        /// Permissions mode.
        mode: PermissionsModeArg,
    },
    /// Choose the provider used for ambiguous natural-language plans.
    SetPlannerProvider {
        /// Provider name.
        provider: Provider,
    },
    /// Clear the planner provider.
    ClearPlannerProvider,
    /// Enable an assistant provider.
    EnableProvider {
        /// Provider name.
        provider: Provider,
    },
    /// Disable an assistant provider.
    DisableProvider {
        /// Provider name.
        provider: Provider,
    },
    /// Save an API key for a provider.
    SetProviderKey {
        /// Provider name.
        provider: Provider,
    },
    /// Remove a saved provider API key.
    ClearProviderKey {
        /// Provider name.
        provider: Provider,
    },
}

#[derive(Subcommand, Debug)]
enum SetupCommand {
    /// Show first-time setup status.
    Status,
    /// Reset setup so the next TUI launch shows first-time setup again.
    Reset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InstallFormat {
    Wheel,
    Tarball,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Provider {
    Local,
    Anthropic,
    Openai,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WatcherModeArg {
    Observe,
    Propose,
    Contained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TelemetryModeArg {
    Local,
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum SandboxToolArg {
    ListServers,
    RestartServer,
    StopServer,
}

impl SandboxToolArg {
    const fn as_cli_value(self) -> &'static str {
        match self {
            Self::ListServers => "list_servers",
            Self::RestartServer => "restart_server",
            Self::StopServer => "stop_server",
        }
    }
}

impl TelemetryModeArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Local => TELEMETRY_MODE_LOCAL,
            Self::Off => TELEMETRY_MODE_OFF,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PermissionsModeArg {
    FullAccess,
    Ask,
}

impl PermissionsModeArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::FullAccess => PERMISSIONS_MODE_FULL_ACCESS,
            Self::Ask => PERMISSIONS_MODE_ASK,
        }
    }
}

fn main() -> Result<()> {
    maybe_migrate_legacy_dashboard_config();

    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() {
        return launch_default();
    }

    let freeform_invocation = parse_freeform_invocation(&raw_args);
    if should_treat_as_freeform(&freeform_invocation) {
        return run_freeform(
            freeform_invocation.request_args.join(" "),
            freeform_invocation.approve,
        );
    }
    if freeform_invocation.approve {
        bail!(
            "global --yes is only supported for natural-language plans; run structured commands directly and use their approval flag when they define one"
        );
    }

    dispatch(Cli::parse())
}

/// One-shot, best-effort migration of a legacy rocm-dash `config.toml` into the
/// unified `config.json`. Prints a notice when a migration runs;
/// never fails startup if the legacy file is malformed.
fn maybe_migrate_legacy_dashboard_config() {
    let Ok(paths) = AppPaths::discover() else {
        return;
    };
    match RocmCliConfig::migrate_legacy_dashboard_toml(&paths) {
        Ok(Some(legacy)) => {
            eprintln!(
                "rocm: migrated legacy dashboard config from {} into {} (the original TOML was left untouched)",
                legacy.display(),
                paths.config_path().display()
            );
        }
        Ok(None) => {}
        Err(err) => {
            eprintln!("rocm: skipped legacy dashboard config migration: {err:#}");
        }
    }
}

fn launch_default() -> Result<()> {
    refresh_startup_update_check_quietly();
    if interactive_terminal() {
        return tui::run(None);
    }

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    print!("{}", render_launch_summary(&paths, &config));
    Ok(())
}

fn run_freeform(request: String, approve: bool) -> Result<()> {
    refresh_startup_update_check_quietly();
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let plan = build_freeform_plan_with_context(&request, &paths, &config);
    if !approve
        && let Some(answer) = render_freeform_read_only_answer(&request, &plan, &paths, &config)?
    {
        print!("{answer}");
        return Ok(());
    }
    print!("{}", render_structured_request_plan(&plan, &paths));
    if approve {
        execute_freeform_next_action(&request, &paths, &config)?;
    }
    Ok(())
}

fn setup(command: Option<SetupCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(SetupCommand::Status) {
        SetupCommand::Status => {
            print!("{}", render_setup_status_text(&paths, &config)?);
        }
        SetupCommand::Reset => {
            print!("{}", reset_setup_prompt_state(&paths, &mut config)?);
        }
    }
    Ok(())
}

fn render_setup_status_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let active_manifest = config
        .active_runtime_key
        .as_deref()
        .and_then(|runtime_key| {
            manifests
                .iter()
                .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(runtime_key))
        });
    let active_ready = active_manifest
        .is_some_and(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok());
    let state = if config.setup.completed && active_ready {
        "completed"
    } else if config.setup.completed {
        "completed; active runtime needs attention"
    } else if active_ready {
        "runtime ready; setup not completed"
    } else if config.onboarding_dismissed {
        "setup dismissed"
    } else {
        "first-time setup will show"
    };

    let mut output = String::new();
    let _ = writeln!(output, "ROCm setup");
    let _ = writeln!(output, "  status: {state}");
    if let Some(root) = config.setup.therock_venv.as_ref() {
        let _ = writeln!(output, "  install folder: {}", root.display());
    }
    if let Some(runtime_key) = config.active_runtime_key.as_deref() {
        let _ = writeln!(output, "  active_runtime_key: {runtime_key}");
    }
    match active_manifest {
        Some(manifest) => {
            let _ = writeln!(output, "  active_runtime_id: {}", manifest.runtime_id);
            let status = if active_ready { "ready" } else { "not_ready" };
            let _ = writeln!(output, "  active_runtime_status: {status}");
        }
        None if config.active_runtime_key.is_some() => {
            let _ = writeln!(output, "  active_runtime_status: missing_manifest");
        }
        None => {
            let _ = writeln!(output, "  active_runtime_status: <unset>");
        }
    }
    let _ = writeln!(output, "  help: run `rocm help` to see how to use rocm-cli");
    Ok(output)
}

fn reset_setup_prompt_state(paths: &AppPaths, config: &mut RocmCliConfig) -> Result<String> {
    config.onboarding_dismissed = false;
    config.setup.completed = false;
    config.save(paths)?;
    Ok([
        "Setup will show again the next time you run `rocm`.",
        "ROCm installs were not deleted.",
        "Installed ROCm folders, API keys, and provider settings were kept.",
        "",
    ]
    .join("\n"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeformInvocation {
    approve: bool,
    request_args: Vec<String>,
}

fn parse_freeform_invocation(raw_args: &[String]) -> FreeformInvocation {
    if raw_args.first().is_some_and(|arg| arg == "--yes") {
        return FreeformInvocation {
            approve: true,
            request_args: raw_args[1..].to_vec(),
        };
    }

    FreeformInvocation {
        approve: false,
        request_args: raw_args.to_vec(),
    }
}

fn should_treat_as_freeform(invocation: &FreeformInvocation) -> bool {
    if invocation
        .request_args
        .first()
        .is_some_and(|arg| arg.starts_with('-'))
    {
        return false;
    }
    treat_as_natural_language(&invocation.request_args)
}

fn execute_freeform_next_action(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<()> {
    let action = freeform_plan_next_action_with_context(request, paths, config)
        .context("natural-language plan did not produce a structured tool call")?;
    validate_freeform_execution_action(&action)?;
    print!("{}", render_freeform_execution_header(&action));

    let mut argv = vec!["rocm".to_owned()];
    argv.extend(action.args);
    let cli = Cli::try_parse_from(argv)?;
    dispatch(cli)
}

fn validate_freeform_execution_action(action: &FreeformPlanAction) -> Result<()> {
    if action.has_placeholders {
        bail!(
            "cannot execute natural-language plan because the next tool call still contains placeholder values: {}",
            format_structured_tool_call("rocm", &action.args)
        );
    }
    if action.provider_assisted {
        bail!(
            "provider-assisted plans must be reviewed interactively; run the displayed structured rocm command directly after reviewing it"
        );
    }
    Ok(())
}

fn render_freeform_execution_header(action: &FreeformPlanAction) -> String {
    let mut output = String::new();
    let _ = writeln!(output);
    let _ = writeln!(output, "execution");
    let _ = writeln!(
        output,
        "  approval: {}",
        if action.approval_required {
            "granted by --yes"
        } else {
            "not required; executing because --yes was supplied"
        }
    );
    let _ = writeln!(
        output,
        "  tool_call: {}",
        format_structured_tool_call("rocm", &action.args)
    );
    output
}

fn render_freeform_read_only_answer(
    request: &str,
    plan: &StructuredRequestPlan,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    if plan.actions.len() != 1 || plan.actions[0].approval != "not required" {
        return Ok(None);
    }
    match plan.actions[0].args.as_slice() {
        [command] if command == "examine" => {
            render_freeform_examine_answer(request, paths, config).map(Some)
        }
        [command, subcommand] if command == "comfyui" && subcommand == "status" => {
            render_freeform_comfyui_status_answer(paths, config).map(Some)
        }
        [command, subcommand] if command == "comfyui" && subcommand == "logs" => {
            let logs = comfyui::render_logs(paths, DEFAULT_LOG_TAIL_LINES)?;
            Ok(Some(format!("ComfyUI logs\n\n{logs}")))
        }
        _ => Ok(None),
    }
}

fn render_freeform_examine_answer(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<String> {
    recover_setup_runtime_registration(paths, config)?;
    let examine = ExamineSummary::gather()?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let active = current_runtime_manifest(config, &manifests);
    let lower = request.to_ascii_lowercase();
    let asks_where = any_substring(&lower, &["where", "folder", "path"]);
    let mut output = String::new();
    let _ = writeln!(
        output,
        "{}",
        if asks_where {
            "ROCm install location"
        } else {
            "ROCm status"
        }
    );
    let _ = writeln!(output);

    if let Some(detail) = examine.driver.detail.as_deref() {
        let _ = writeln!(output, "GPU: {detail}");
    } else if let Some(target) = examine.detected_gfx_target.as_deref() {
        let _ = writeln!(output, "GPU: AMD GPU target {target}");
    } else {
        let _ = writeln!(output, "GPU: I could not identify an AMD GPU yet.");
    }
    if let Some(target) = examine.detected_gfx_target.as_deref() {
        let _ = writeln!(output, "Target: {target}");
    }

    if let Some(manifest) = active {
        let status = runtime_usability_status(manifest);
        if status == "ready" {
            let _ = writeln!(output, "ROCm/TheRock: installed and active for ROCm CLI");
        } else {
            let _ = writeln!(output, "ROCm/TheRock: found, but status is {status}");
        }
        let _ = writeln!(output, "Folder: {}", manifest.install_root.display());
        let _ = writeln!(
            output,
            "Version: {}",
            therock::runtime_version_display(&manifest.version)
        );
        let _ = writeln!(output, "GPU package: {}", manifest.family);
    } else {
        let setup_root = config.setup.therock_venv.as_deref();
        if let Some(root) = setup_root {
            let _ = writeln!(
                output,
                "ROCm/TheRock: setup folder saved, but no active runtime is selected"
            );
            let _ = writeln!(output, "Folder: {}", root.display());
        } else if manifests.len() == 1 {
            let manifest = &manifests[0];
            let _ = writeln!(
                output,
                "ROCm/TheRock: installed, but not selected as the active runtime"
            );
            let _ = writeln!(output, "Folder: {}", manifest.install_root.display());
            let _ = writeln!(output, "Runtime: {}", manifest.runtime_key);
            let _ = writeln!(
                output,
                "Next step: rocm runtimes activate {}",
                manifest.runtime_key
            );
        } else if manifests.is_empty() {
            let _ = writeln!(output, "ROCm/TheRock: not installed for ROCm CLI yet");
            let _ = writeln!(
                output,
                "Next step: run `rocm` and choose Install ROCm, or ask to install TheRock into a folder you choose."
            );
        } else {
            let _ = writeln!(
                output,
                "ROCm/TheRock: multiple installs found, but none is active"
            );
            let _ = writeln!(output, "Run `rocm runtimes list` to choose one.");
        }
    }

    if examine.legacy_rocm.status == "not_detected" && active.is_some() {
        let _ = writeln!(
            output,
            "Note: ROCm CLI is using its managed TheRock runtime, not a global ROCm install."
        );
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Nothing was changed.");
    Ok(output)
}

fn render_freeform_comfyui_status_answer(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<String> {
    let status = comfyui::render_status(paths, config)?;
    let installed = status.contains("  installed: yes");
    let running = status.contains("  status: running");
    let starting = status.contains("  status: starting");
    let mut output = String::new();
    let _ = writeln!(output, "ComfyUI status");
    let _ = writeln!(output);
    if installed {
        let _ = writeln!(output, "ComfyUI: installed");
    } else {
        let _ = writeln!(output, "ComfyUI: not installed yet");
    }
    if running {
        let url = chat_tool_value(&status, "url").unwrap_or_else(|| "<unknown>".to_owned());
        let _ = writeln!(output, "Running: yes");
        let _ = writeln!(output, "URL: {url}");
    } else if starting {
        let _ = writeln!(output, "Running: starting");
    } else {
        let _ = writeln!(output, "Running: no");
    }
    if !installed {
        let _ = writeln!(
            output,
            "To install it, ask `can you setup ComfyUI for me` or run `rocm comfyui install`."
        );
    } else if !running {
        let _ = writeln!(
            output,
            "To open it, ask `can you start ComfyUI` or run `rocm comfyui start`."
        );
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Nothing was changed.");
    Ok(output)
}

fn dispatch(cli: Cli) -> Result<()> {
    if !matches!(
        cli.command,
        Some(Command::Update { .. } | Command::Bootstrap { .. } | Command::Completions { .. })
    ) {
        refresh_startup_update_check_quietly();
    }

    match cli.command {
        Some(Command::Examine { json }) => examine(json),
        Some(Command::Diagnose { symptom, top, json }) => diagnose(symptom, top, json),
        Some(Command::Fix {
            fix_id,
            yes,
            dry_run,
            device_index,
        }) => fix(fix_id, yes, dry_run, device_index),
        Some(Command::Version) => {
            println!("rocm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Setup { command }) => setup(command),
        Some(Command::EngineServeHttp {
            engine,
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe_json,
        }) => run_builtin_engine_serve_http(
            &engine,
            service_id,
            model_ref,
            host,
            port,
            &device_policy,
            runtime_id,
            env_id,
            state_path,
            log_path,
            parse_engine_recipe_json_arg(engine_recipe_json)?,
        ),
        Some(Command::EngineStdio { engine }) => run_builtin_engine_stdio(&engine),
        Some(Command::InternalStatus) => {
            let paths = AppPaths::discover()?;
            print!("{}", render_internal_status_text(&paths)?);
            Ok(())
        }
        Some(Command::InternalBridgeSnapshot { pretty }) => {
            let paths = AppPaths::discover()?;
            let snapshot = build_codex_bridge_snapshot(&paths)?;
            if pretty {
                println!("{}", serde_json::to_string_pretty(&snapshot)?);
            } else {
                println!("{}", serde_json::to_string(&snapshot)?);
            }
            Ok(())
        }
        Some(Command::InternalSandboxRun {
            tool,
            service_id,
            allow_native_fallback,
        }) => {
            let paths = AppPaths::discover()?;
            let value = run_internal_sandbox_tool(&paths, tool, service_id, allow_native_fallback)?;
            println!("{}", serde_json::to_string_pretty(&value)?);
            Ok(())
        }
        Some(Command::McpCall {
            name,
            arguments_json,
            arguments_file,
            allow_mutation,
        }) => {
            let paths = AppPaths::discover()?;
            let arguments_json = match arguments_file {
                Some(path) => fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?,
                None => arguments_json.unwrap_or_else(|| "{}".to_owned()),
            };
            let arguments = serde_json::from_str(arguments_json.trim_start_matches('\u{feff}'))
                .with_context(|| format!("failed to parse arguments JSON for `{name}`"))?;
            let value = run_internal_mcp_call(&paths, &name, arguments, allow_mutation)?;
            println!("{}", serde_json::to_string(&value)?);
            Ok(())
        }
        Some(Command::Chat {
            provider,
            model,
            prompt,
            tools,
        }) => {
            let paths = AppPaths::discover()?;
            if interactive_terminal() && prompt.is_none() {
                return tui::run(provider.map(provider_name).map(str::to_owned));
            }
            match prompt {
                Some(prompt) => print!(
                    "{}",
                    render_chat_prompt_text(
                        &paths,
                        provider.map_or("local", provider_name),
                        model.as_deref(),
                        &prompt,
                        tools
                    )?
                ),
                None => print!(
                    "{}",
                    render_chat_text(&paths, provider.map_or("local", provider_name))?
                ),
            }
            Ok(())
        }
        Some(Command::Bootstrap { command }) => bootstrap::run(command),
        Some(Command::Install { target }) => install(target),
        Some(Command::Update {
            apply,
            runtime,
            activate,
            dry_run,
        }) => {
            let paths = AppPaths::discover()?;
            if apply {
                let mut config = RocmCliConfig::load(&paths)?;
                match apply_runtime_update(
                    &paths,
                    &mut config,
                    runtime.as_deref(),
                    activate,
                    dry_run,
                ) {
                    Ok(text) => {
                        print!("{text}");
                        record_cli_audit_event(
                            &paths,
                            "runtime",
                            if dry_run {
                                "runtime_update_dry_run"
                            } else {
                                "runtime_update_apply"
                            },
                            "info",
                            format!(
                                "runtime update completed runtime={} activate={} dry_run={}",
                                runtime.as_deref().unwrap_or("<selected>"),
                                activate,
                                dry_run
                            ),
                            None,
                        );
                    }
                    Err(error) => {
                        record_cli_audit_event(
                            &paths,
                            "runtime",
                            if dry_run {
                                "runtime_update_dry_run"
                            } else {
                                "runtime_update_apply"
                            },
                            "error",
                            format!(
                                "runtime update failed runtime={} activate={} dry_run={}: {error}",
                                runtime.as_deref().unwrap_or("<selected>"),
                                activate,
                                dry_run
                            ),
                            None,
                        );
                        return Err(error);
                    }
                }
                return Ok(());
            }
            match render_update_text(&paths) {
                Ok(text) => {
                    print!("{text}");
                    record_cli_audit_event(
                        &paths,
                        "update",
                        "update_check",
                        "info",
                        "rendered update report",
                        None,
                    );
                }
                Err(error) => {
                    record_cli_audit_event(
                        &paths,
                        "update",
                        "update_check",
                        "error",
                        format!("update report failed: {error}"),
                        None,
                    );
                    return Err(error);
                }
            }
            Ok(())
        }
        Some(Command::Runtimes { command }) => runtimes(command),
        Some(Command::Engines { command }) => engines(command),
        Some(Command::Model { verbose }) => {
            let paths = AppPaths::discover()?;
            let rendered = if verbose {
                render_model_registry_verbose_text_with_context_and_host(Some(&paths), None, None)
            } else {
                render_model_registry_text_with_context_and_host(Some(&paths), None, None)
            };
            print!("{rendered}");
            Ok(())
        }
        Some(Command::Serve {
            model,
            engine,
            device,
            runtime_id,
            env_id,
            host,
            port,
            foreground,
            managed,
            allow_public_bind,
        }) => serve(
            model,
            engine,
            device,
            runtime_id,
            env_id,
            host,
            port,
            foreground,
            managed,
            allow_public_bind,
        ),
        Some(Command::Comfyui { command }) => comfyui(command),
        Some(Command::Services { command }) => services(command),
        Some(Command::Automations { command }) => automations(command),
        Some(Command::Config { command }) => config(command),
        Some(Command::Logs {
            service,
            search,
            query,
        }) => {
            let paths = AppPaths::discover()?;
            if service.is_some() && (!search.is_empty() || !query.is_empty()) {
                bail!(
                    "`rocm logs` accepts either --service <service-id> or a search query, not both"
                );
            }
            if !search.is_empty() && !query.is_empty() {
                bail!(
                    "`rocm logs` accepts either --search <query> or a positional query, not both"
                );
            }
            if let Some(service_id) = service {
                print!("{}", render_service_logs_text(&paths, &service_id)?);
            } else {
                let query = if search.is_empty() {
                    (!query.is_empty()).then(|| query.join(" "))
                } else {
                    Some(search.join(" "))
                };
                match query.as_deref() {
                    Some(query) => print!("{}", render_logs_browser_text(&paths, Some(query))),
                    None => print!("{}", render_logs_text(&paths)),
                }
            }
            Ok(())
        }
        Some(Command::Daemon { status }) => {
            if status {
                let paths = AppPaths::discover()?;
                let config = RocmCliConfig::load(&paths)?;
                print!("{}", render_daemon_text(&paths, &config));
                Ok(())
            } else {
                rocmd::run_from_args(daemon_run_argv())
            }
        }
        Some(Command::Dash {
            replay,
            demo,
            chat_mock,
        }) => dash::run(replay, demo, chat_mock),
        Some(Command::Uninstall {
            yes,
            dry_run,
            keep_binaries,
            keep_config,
            keep_data,
            keep_cache,
            force_dev_binaries,
        }) => uninstall(UninstallOptions {
            yes,
            dry_run,
            keep_binaries,
            keep_config,
            keep_data,
            keep_cache,
            force_dev_binaries,
        }),
        Some(Command::Completions { shell }) => {
            let mut cmd = completion_command();
            clap_complete::generate(shell, &mut cmd, "rocm", &mut std::io::stdout());
            Ok(())
        }
        None => launch_default(),
    }
}

/// Build the command tree handed to `clap_complete` for shell completions.
///
/// clap_complete's AOT generators do not filter `hide = true` subcommands the
/// way `--help` does, so generating directly from `Cli::command()` would leak
/// internal verbs (e.g. `__engine-stdio`, `mcp-call`) into the completion
/// scripts. clap 4.x has no API to remove a subcommand from an existing
/// `Command`, so we rebuild the root from the derived definition while dropping
/// every hidden subcommand. This keeps the generated completions in sync with
/// what `--help` shows, at every nesting level.
fn completion_command() -> clap::Command {
    without_hidden_subcommands(Cli::command())
}

/// Return a copy of `cmd` whose (recursive) subcommand set excludes every
/// `hide = true` subcommand, preserving the command's own settings, args, and
/// visible subcommands intact.
///
/// clap exposes no API to remove a subcommand from a `Command`
/// (`get_subcommands_mut` cannot remove, `mut_subcommands` is map-only, and
/// `subcommands`/`subcommand` only append). So we rebuild the command without
/// subcommands and re-attach only the visible ones, each filtered recursively.
fn without_hidden_subcommands(cmd: clap::Command) -> clap::Command {
    let visible: Vec<clap::Command> = cmd
        .get_subcommands()
        .filter(|sc| !sc.is_hide_set())
        .cloned()
        .map(without_hidden_subcommands)
        .collect();
    strip_subcommands(cmd).subcommands(visible)
}

/// Rebuild a `Command` without any subcommands, preserving the fields that
/// matter for completion generation (name, metadata, args, key settings).
fn strip_subcommands(cmd: clap::Command) -> clap::Command {
    let mut bare = clap::Command::new(cmd.get_name().to_owned());
    if let Some(about) = cmd.get_about() {
        bare = bare.about(about.clone());
    }
    if let Some(long_about) = cmd.get_long_about() {
        bare = bare.long_about(long_about.clone());
    }
    if let Some(version) = cmd.get_version() {
        bare = bare.version(version.to_owned());
    }
    if let Some(long_version) = cmd.get_long_version() {
        bare = bare.long_version(long_version.to_owned());
    }
    for alias in cmd.get_visible_aliases() {
        bare = bare.visible_alias(alias.to_owned());
    }
    for arg in cmd.get_arguments() {
        bare = bare.arg(arg.clone());
    }
    if cmd.is_subcommand_required_set() {
        bare = bare.subcommand_required(true);
    }
    if cmd.is_arg_required_else_help_set() {
        bare = bare.arg_required_else_help(true);
    }
    bare
}

fn refresh_startup_update_check_quietly() {
    let Ok(paths) = AppPaths::discover() else {
        return;
    };
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let _ =
        therock::maybe_refresh_startup_update_check(&paths, config.active_runtime_key.as_deref());
}

fn build_codex_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: rocm_core::unix_time_millis(),
        examine: ExamineSummary::gather()?,
        gpu: build_codex_bridge_gpu_snapshot(&config),
        config,
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: builtin_codex_bridge_engine_inventory(),
        services: load_managed_services(paths)?,
    })
}

fn build_codex_bridge_gpu_snapshot(config: &RocmCliConfig) -> CodexBridgeGpuSnapshot {
    if !config.telemetry.local_inspection_enabled() {
        return CodexBridgeGpuSnapshot {
            amd_smi_available: false,
            static_snapshot: None,
            monitor_snapshot: None,
            note: Some("GPU telemetry is disabled by rocm-cli config.".to_owned()),
        };
    }

    CodexBridgeGpuSnapshot {
        amd_smi_available: false,
        static_snapshot: None,
        monitor_snapshot: None,
        note: Some("Use `rocm examine` for the current local AMD GPU summary.".to_owned()),
    }
}

fn builtin_codex_bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    let binary_path = daemon_binary_path()
        .ok()
        .map(|path| path.display().to_string());
    builtin_engine_inventory()
        .iter()
        .map(|(id, summary)| CodexBridgeEngine {
            id: (*id).to_owned(),
            summary: (*summary).to_owned(),
            default_for_platform: *id == default_engine,
            installed_binary: true,
            binary_path: binary_path.clone(),
        })
        .collect()
}

const fn builtin_engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        ("pytorch", "TheRock PyTorch local serving engine"),
        (
            "llama.cpp",
            "GGUF serving with ROCm GPU required by rocm-cli",
        ),
        (
            "lemonade",
            "default embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
        (
            "sglang",
            "Linux/WSL ROCm GPU serving engine through external SGLang",
        ),
        (
            "atom",
            "Linux/WSL ROCm GPU serving engine through external ATOM Python",
        ),
    ]
}

fn examine(json: bool) -> Result<()> {
    // `rocm examine` is the general system inspector: the exit code reports
    // whether it RAN, not what it found. Any finding (no GPU, WSL, degraded) is
    // surfaced in the output and the `--json` `status` field, and the command
    // exits 0; a genuine inability to examine propagates as an error via `?`.
    if json {
        let examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Auto);
        println!("{}", serde_json::to_string_pretty(&examination)?);
        return Ok(());
    }
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let (text, summary) = examine_human_report(&paths, &config)?;
    print!("{text}");
    if summary.wsl.as_ref().is_some_and(|w| w.is_wsl) {
        // Informational route-out guidance for humans (the verdict is also in
        // the `status` field for `--json` consumers).
        println!("\n{}", rocm_core::WSL_ROUTE_OUT_NOTE);
    }
    Ok(())
}

fn diagnose(symptom: Option<String>, top: usize, json: bool) -> Result<()> {
    // `rocm diagnose` is a query: it exits 0 whether it matched, found nothing,
    // or is out of scope. Callers read `has_match` / `out_of_scope` /
    // `route_when_no_match` from `--json` rather than branching on the exit code.
    let examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Auto);
    let report = rocm_core::run_diagnose(&examination, &symptom.unwrap_or_default());
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", rocm_core::render_diagnose_text(&report, top));
    }
    Ok(())
}

fn fix(fix_id: Option<String>, yes: bool, dry_run: bool, device_index: Option<i64>) -> Result<()> {
    let Some(fix_id) = fix_id else {
        print!("{}", rocm_core::list_fix_recipes());
        return Ok(());
    };
    let opts = rocm_core::FixOptions {
        yes,
        dry_run,
        device_index,
    };
    let code = rocm_core::apply_fix(&fix_id, &opts);
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

fn record_cli_audit_event(
    paths: &AppPaths,
    category: &str,
    action: &str,
    level: &str,
    message: impl Into<String>,
    service_id: Option<&str>,
) {
    let event = AuditEventRecord {
        at_unix_ms: rocm_core::unix_time_millis(),
        source: "rocm".to_owned(),
        category: category.to_owned(),
        actor: "cli".to_owned(),
        level: level.to_owned(),
        action: action.to_owned(),
        message: message.into(),
        watcher_id: None,
        service_id: service_id.map(str::to_owned),
    };
    if let Err(error) = append_audit_event(paths, &event) {
        eprintln!("warning: failed to write audit event: {error}");
    }
    if let Err(error) = append_cli_lifecycle_logs(paths, &event) {
        eprintln!("warning: failed to write CLI lifecycle log: {error}");
    }
}

fn append_cli_lifecycle_logs(paths: &AppPaths, event: &AuditEventRecord) -> Result<()> {
    paths.ensure()?;
    let line = render_cli_lifecycle_log_line(event);
    append_text_log_line(&cli_lifecycle_log_path(paths), &line)?;
    append_text_log_line(
        &cli_action_log_path(paths, &event.category, &event.action),
        &line,
    )?;
    Ok(())
}

fn render_cli_lifecycle_log_line(event: &AuditEventRecord) -> String {
    format!(
        "{} level={} category={} action={} service_id={} message={}\n",
        event.at_unix_ms,
        sanitize_log_value(&event.level),
        sanitize_log_value(&event.category),
        sanitize_log_value(&event.action),
        event.service_id.as_deref().unwrap_or("<none>"),
        sanitize_log_value(&event.message)
    )
}

fn append_text_log_line(path: &Path, line: &str) -> Result<()> {
    let parent = path.parent().context("log path has no parent directory")?;
    fs::create_dir_all(parent)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(line.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn cli_lifecycle_log_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("logs").join("cli-lifecycle.log")
}

fn cli_action_log_path(paths: &AppPaths, category: &str, action: &str) -> PathBuf {
    paths.data_dir.join("logs").join("cli").join(format!(
        "{}-{}.log",
        sanitize_log_component(category),
        sanitize_log_component(action)
    ))
}

fn sanitize_log_component(value: &str) -> String {
    let component = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    if component.is_empty() {
        "unknown".to_owned()
    } else {
        component
    }
}

fn sanitize_log_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '\r' | '\n' | '\t' => ' ',
            _ => ch,
        })
        .collect::<String>()
}

fn therock_install_version_selector(
    version: Option<String>,
    build_date: Option<String>,
) -> Result<Option<therock::RuntimeVersionSelector>> {
    match (version, build_date) {
        (Some(_), Some(_)) => bail!("use either --version or --build-date, not both"),
        (Some(version), None) => Ok(Some(therock::RuntimeVersionSelector::version(version)?)),
        (None, Some(build_date)) => Ok(Some(therock::RuntimeVersionSelector::build_date(
            build_date,
        )?)),
        (None, None) => Ok(None),
    }
}

fn therock_install_version_selector_display(selector: &therock::RuntimeVersionSelector) -> String {
    match selector {
        therock::RuntimeVersionSelector::Version(version) => format!("version:{version}"),
        therock::RuntimeVersionSelector::BuildDate(date) => format!("build-date:{date}"),
    }
}

fn install(target: InstallTarget) -> Result<()> {
    let paths = AppPaths::discover()?;
    match target {
        InstallTarget::Sdk {
            channel,
            format,
            prefix,
            version,
            build_date,
            family,
            dry_run,
        } => {
            let format_name = match format {
                InstallFormat::Wheel => "wheel",
                InstallFormat::Tarball => "tarball",
            };
            let version_selector = therock_install_version_selector(version, build_date)?;
            let version_selector_display = version_selector.as_ref().map_or_else(
                || "latest-compatible".to_owned(),
                therock_install_version_selector_display,
            );
            let prefix_display = prefix
                .as_ref()
                .map_or_else(|| "<managed>".to_owned(), |path| path.display().to_string());
            match therock::install_sdk(
                &paths,
                &channel,
                format_name,
                prefix,
                version_selector,
                family.as_deref(),
                dry_run,
            ) {
                Ok(output) => {
                    let finalized = if dry_run {
                        None
                    } else {
                        finalize_successful_sdk_install(&paths)?
                    };
                    print!("{output}");
                    if let Some(finalized) = finalized {
                        print_sdk_install_success(&finalized);
                    }
                    record_cli_audit_event(
                        &paths,
                        "runtime",
                        if dry_run {
                            "install_sdk_dry_run"
                        } else {
                            "install_sdk"
                        },
                        "info",
                        format!(
                            "sdk install completed channel={channel} format={format_name} prefix={prefix_display} version_selector={version_selector_display} dry_run={dry_run}"
                        ),
                        None,
                    );
                }
                Err(error) => {
                    record_cli_audit_event(
                        &paths,
                        "runtime",
                        if dry_run {
                            "install_sdk_dry_run"
                        } else {
                            "install_sdk"
                        },
                        "error",
                        format!(
                            "sdk install failed channel={channel} format={format_name} prefix={prefix_display} version_selector={version_selector_display} dry_run={dry_run}: {error}"
                        ),
                        None,
                    );
                    return Err(error);
                }
            }
        }
        InstallTarget::Driver {
            dkms,
            yes,
            dry_run,
            reconcile,
        } => {
            if reconcile {
                if dkms || yes || dry_run {
                    bail!(
                        "`rocm install driver --reconcile` cannot be combined with --dkms, --yes, or --dry-run"
                    );
                }
                match reconcile_driver_install(&paths) {
                    Ok(output) => {
                        print!("{output}");
                        record_cli_audit_event(
                            &paths,
                            "driver",
                            "install_driver_reconcile",
                            "info",
                            "driver install state reconciled",
                            None,
                        );
                    }
                    Err(error) => {
                        record_cli_audit_event(
                            &paths,
                            "driver",
                            "install_driver_reconcile",
                            "error",
                            format!("driver install reconciliation failed: {error}"),
                            None,
                        );
                        return Err(error);
                    }
                }
                return Ok(());
            }
            match install_driver(&paths, dkms, yes, dry_run) {
                Ok(result) => {
                    print!("{}", result.output);
                    record_cli_audit_event(
                        &paths,
                        "driver",
                        if result.executed {
                            "install_driver_execute"
                        } else {
                            "install_driver_plan"
                        },
                        "info",
                        format!("driver install handled dkms={dkms} yes={yes} dry_run={dry_run}"),
                        None,
                    );
                }
                Err(error) => {
                    let executed = error.executed;
                    let error = error.source;
                    record_cli_audit_event(
                        &paths,
                        "driver",
                        if executed {
                            "install_driver_execute"
                        } else {
                            "install_driver_plan"
                        },
                        "error",
                        format!(
                            "driver install failed dkms={dkms} yes={yes} dry_run={dry_run}: {error}"
                        ),
                        None,
                    );
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

fn install_driver(
    paths: &AppPaths,
    dkms: bool,
    yes: bool,
    dry_run: bool,
) -> std::result::Result<DriverInstallResult, DriverInstallError> {
    let examine =
        ExamineSummary::gather().map_err(|source| DriverInstallError::new(source, false))?;
    let os_release = read_os_release().unwrap_or_default();
    let plan = build_driver_install_plan(&examine, &os_release, dkms);
    let mut output = render_driver_install_plan(&plan, yes, dry_run);
    if !yes || dry_run || !plan.supported || !plan.mutating {
        return Ok(DriverInstallResult {
            output,
            executed: false,
        });
    }

    let boot_id = current_boot_id();
    let mut state = DriverInstallState {
        approved_at_unix_ms: rocm_core::unix_time_millis(),
        executed_at_unix_ms: None,
        pre_driver: examine.driver,
        post_driver: None,
        boot_id_at_execution: boot_id,
        reboot_required: false,
        reboot_observed: false,
        commands: plan.execution_commands(),
        reconciled_at_unix_ms: None,
        reconciliation: None,
    };
    write_driver_install_state(paths, &state)
        .map_err(|source| DriverInstallError::new(source, false))?;

    for command in &plan.commands {
        if !matches!(
            command.phase,
            DriverCommandPhase::Prepare | DriverCommandPhase::Execute
        ) {
            continue;
        }
        run_driver_shell_command(&command.command)
            .with_context(|| format!("driver command failed: {}", command.command))
            .map_err(|source| DriverInstallError::new(source, true))?;
    }

    let post_driver = ExamineSummary::gather()
        .map_err(|source| DriverInstallError::new(source, true))?
        .driver;
    state.executed_at_unix_ms = Some(rocm_core::unix_time_millis());
    state.post_driver = Some(post_driver);
    state.reboot_required = true;
    state.reboot_observed = driver_reboot_observed(state.boot_id_at_execution.as_deref());
    write_driver_install_state(paths, &state)
        .map_err(|source| DriverInstallError::new(source, true))?;

    let _ = writeln!(output, "execution:");
    let _ = writeln!(output, "  status: completed");
    let _ = writeln!(output, "  reboot_required: true");
    let _ = writeln!(
        output,
        "  state: {}",
        driver_install_state_path(paths).display()
    );
    Ok(DriverInstallResult {
        output,
        executed: true,
    })
}

fn reconcile_driver_install(paths: &AppPaths) -> Result<String> {
    let Some(mut state) = read_driver_install_state(paths)? else {
        let mut output = String::new();
        let _ = writeln!(output, "driver install reconciliation");
        let _ = writeln!(
            output,
            "  state: {}",
            driver_install_state_path(paths).display()
        );
        let _ = writeln!(output, "  approval: not required");
        let _ = writeln!(output, "  privileged_commands: <none>");
        let _ = writeln!(output, "  status: no prior driver execution state found");
        let _ = writeln!(
            output,
            "  action: run `rocm install driver --dkms` to review the native driver plan"
        );
        return Ok(output);
    };
    let examine = ExamineSummary::gather()?;
    let checks = passive_driver_checks();
    reconcile_driver_install_state(paths, &mut state, examine.driver, current_boot_id(), checks)
}

fn reconcile_driver_install_state(
    paths: &AppPaths,
    state: &mut DriverInstallState,
    driver: rocm_core::DriverSummary,
    current_boot_id: Option<String>,
    checks: Vec<DriverPassiveCheck>,
) -> Result<String> {
    let reboot_observed = state
        .boot_id_at_execution
        .as_deref()
        .zip(current_boot_id.as_deref())
        .is_some_and(|(executed, current)| executed != current);
    state.reboot_observed = reboot_observed;
    state.reboot_required = state.reboot_required || state.executed_at_unix_ms.is_some();
    state.post_driver = Some(driver.clone());
    let at_unix_ms = rocm_core::unix_time_millis();
    state.reconciled_at_unix_ms = Some(at_unix_ms);
    let check_summary = summarize_driver_passive_checks(&checks);
    state.reconciliation = Some(DriverReconciliationState {
        at_unix_ms,
        driver,
        reboot_observed,
        check_summary,
        checks,
    });
    write_driver_install_state(paths, state)?;
    Ok(render_driver_reconciliation(paths, state))
}

fn render_driver_reconciliation(paths: &AppPaths, state: &DriverInstallState) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "driver install reconciliation");
    let _ = writeln!(
        output,
        "  state: {}",
        driver_install_state_path(paths).display()
    );
    let _ = writeln!(output, "  approval: not required");
    let _ = writeln!(output, "  privileged_commands: <none>");
    let _ = writeln!(
        output,
        "  approved_at_unix_ms: {}",
        state.approved_at_unix_ms
    );
    let _ = writeln!(
        output,
        "  executed_at_unix_ms: {}",
        state
            .executed_at_unix_ms
            .map_or_else(|| "<not executed>".to_owned(), |value| value.to_string())
    );
    let _ = writeln!(output, "  reboot_required: {}", state.reboot_required);
    let _ = writeln!(output, "  reboot_observed: {}", state.reboot_observed);
    if let Some(reconciliation) = &state.reconciliation {
        let _ = writeln!(
            output,
            "  reconciled_at_unix_ms: {}",
            reconciliation.at_unix_ms
        );
        let _ = writeln!(output, "  driver_status: {}", reconciliation.driver.status);
        let _ = writeln!(
            output,
            "  driver_detail: {}",
            reconciliation
                .driver
                .detail
                .as_deref()
                .unwrap_or("<unknown>")
        );
        let _ = writeln!(
            output,
            "  passive_check_summary: total={} present={} missing={}",
            reconciliation.check_summary.total,
            reconciliation.check_summary.present,
            reconciliation.check_summary.missing
        );
        if reconciliation.checks.is_empty() {
            let _ = writeln!(output, "  passive_checks: <none for this platform>");
        } else {
            let _ = writeln!(output, "  passive_checks:");
            for check in &reconciliation.checks {
                let _ = writeln!(
                    output,
                    "    {}: {} ({})",
                    check.name, check.status, check.detail
                );
            }
        }
        if state.reboot_required && !state.reboot_observed {
            let _ = writeln!(
                output,
                "  action: reboot is still required before post-install checks are meaningful"
            );
        } else if reconciliation
            .checks
            .iter()
            .any(|check| check.status != "present")
        {
            let _ = writeln!(
                output,
                "  action: reconciliation recorded missing passive checks; run `rocm examine` and inspect driver logs"
            );
        } else {
            let _ = writeln!(
                output,
                "  action: reconciliation complete; run `rocm examine` for the full host summary"
            );
        }
    }
    output
}

fn summarize_driver_passive_checks(checks: &[DriverPassiveCheck]) -> DriverPassiveCheckSummary {
    let total = checks.len();
    let present = checks
        .iter()
        .filter(|check| check.status == "present")
        .count();
    DriverPassiveCheckSummary {
        total,
        present,
        missing: total.saturating_sub(present),
    }
}

fn passive_driver_checks() -> Vec<DriverPassiveCheck> {
    if !rocm_core::runtime_is_linux() {
        return Vec::new();
    }
    vec![
        passive_path_check("/sys/module/amdgpu", "amdgpu kernel module path"),
        passive_path_check("/dev/kfd", "KFD device node"),
        passive_render_node_check(),
    ]
}

fn passive_path_check(path: &str, detail: &str) -> DriverPassiveCheck {
    DriverPassiveCheck {
        name: path.to_owned(),
        status: if Path::new(path).exists() {
            "present"
        } else {
            "missing"
        }
        .to_owned(),
        detail: detail.to_owned(),
    }
}

fn passive_render_node_check() -> DriverPassiveCheck {
    let present = fs::read_dir("/dev/dri")
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(std::result::Result::ok))
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("renderD"))
        });
    DriverPassiveCheck {
        name: "/dev/dri/renderD*".to_owned(),
        status: if present { "present" } else { "missing" }.to_owned(),
        detail: "DRM render node".to_owned(),
    }
}

fn driver_install_args_require_tui_approval(args: &[&str]) -> Result<bool> {
    let flags = parse_driver_install_flags(args)?;
    let examine = ExamineSummary::gather()?;
    let os_release = read_os_release().unwrap_or_default();
    let plan = build_driver_install_plan(&examine, &os_release, flags.dkms);
    Ok(driver_install_flags_require_tui_approval(&plan, &flags))
}

fn driver_install_flags_require_approval(
    plan: &DriverInstallPlan,
    flags: &DriverInstallFlags,
) -> bool {
    if flags.reconcile {
        return false;
    }
    driver_plan_approval_label(plan, flags.yes, flags.dry_run) == "required"
}

fn driver_install_flags_require_tui_approval(
    plan: &DriverInstallPlan,
    flags: &DriverInstallFlags,
) -> bool {
    let mut tui_flags = *flags;
    tui_flags.yes = false;
    driver_install_flags_require_approval(plan, &tui_flags)
}

fn parse_driver_install_flags(args: &[&str]) -> Result<DriverInstallFlags> {
    if args.first() != Some(&"driver") {
        bail!("Usage: /install driver [--dkms] [--dry-run] [--yes] [--reconcile]");
    }
    let mut flags = DriverInstallFlags::default();
    for arg in args.iter().skip(1) {
        match *arg {
            "--dkms" => flags.dkms = true,
            "--yes" => flags.yes = true,
            "--dry-run" => flags.dry_run = true,
            "--reconcile" => flags.reconcile = true,
            value if value.starts_with('-') => bail!("Unknown driver install option: {value}"),
            value => bail!("Unexpected driver install argument: {value}"),
        }
    }
    if flags.reconcile && (flags.dkms || flags.yes || flags.dry_run) {
        bail!(
            "Usage: /install driver --reconcile\n\n--reconcile cannot be combined with --dkms, --yes, or --dry-run"
        );
    }
    Ok(flags)
}

struct DriverInstallResult {
    output: String,
    executed: bool,
}

#[derive(Clone, Copy, Default)]
struct DriverInstallFlags {
    dkms: bool,
    yes: bool,
    dry_run: bool,
    reconcile: bool,
}

struct DriverInstallError {
    source: anyhow::Error,
    executed: bool,
}

impl DriverInstallError {
    const fn new(source: anyhow::Error, executed: bool) -> Self {
        Self { source, executed }
    }
}

#[derive(Debug, Clone)]
struct DriverInstallPlan {
    supported: bool,
    mutating: bool,
    policy: String,
    os_id: String,
    version_id: String,
    codename: String,
    repo_version_expr: String,
    reason: String,
    preflight_checks: Vec<String>,
    commands: Vec<DriverPlanCommand>,
    checks: Vec<String>,
}

impl DriverInstallPlan {
    fn execution_commands(&self) -> Vec<String> {
        self.commands
            .iter()
            .filter(|command| {
                matches!(
                    command.phase,
                    DriverCommandPhase::Prepare | DriverCommandPhase::Execute
                )
            })
            .map(|command| command.command.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DriverCommandPhase {
    Prepare,
    Execute,
    Verify,
}

#[derive(Debug, Clone)]
struct DriverPlanCommand {
    phase: DriverCommandPhase,
    command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverInstallState {
    approved_at_unix_ms: u128,
    executed_at_unix_ms: Option<u128>,
    pre_driver: rocm_core::DriverSummary,
    post_driver: Option<rocm_core::DriverSummary>,
    boot_id_at_execution: Option<String>,
    reboot_required: bool,
    reboot_observed: bool,
    commands: Vec<String>,
    #[serde(default)]
    reconciled_at_unix_ms: Option<u128>,
    #[serde(default)]
    reconciliation: Option<DriverReconciliationState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverReconciliationState {
    at_unix_ms: u128,
    driver: rocm_core::DriverSummary,
    reboot_observed: bool,
    #[serde(default)]
    check_summary: DriverPassiveCheckSummary,
    checks: Vec<DriverPassiveCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DriverPassiveCheckSummary {
    total: usize,
    present: usize,
    missing: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriverPassiveCheck {
    name: String,
    status: String,
    detail: String,
}

fn build_driver_install_plan(
    examine: &ExamineSummary,
    os_release_text: &str,
    dkms: bool,
) -> DriverInstallPlan {
    let repo_version_expr = "${ROCM_CLI_AMDGPU_VERSION:-7.2.4}".to_owned();
    if examine.os == "windows" {
        return DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "windows_validate_only".to_owned(),
            os_id: "windows".to_owned(),
            version_id: String::new(),
            codename: String::new(),
            repo_version_expr,
            reason: "Windows driver install is validate-only in rocm-cli; use `rocm examine` to inspect the AMD display driver.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned()],
        };
    }
    if examine.wsl.as_ref().is_some_and(|wsl| wsl.is_wsl) {
        return DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "wsl_rocdxg".to_owned(),
            os_id: "wsl".to_owned(),
            version_id: String::new(),
            codename: String::new(),
            repo_version_expr,
            reason: "WSL uses the Windows host driver plus ROCDXG; run `scripts/wsl_setup_rocdxg.sh` inside WSL instead of installing Linux DKMS.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned(), "scripts/wsl_preflight.py".to_owned()],
        };
    }

    let os_id = parse_os_release_field(os_release_text, "ID").unwrap_or_default();
    let version_id = parse_os_release_field(os_release_text, "VERSION_ID").unwrap_or_default();
    let codename = parse_os_release_field(os_release_text, "VERSION_CODENAME")
        .or_else(|| parse_os_release_field(os_release_text, "UBUNTU_CODENAME"))
        .or_else(|| codename_for_version(&os_id, &version_id).map(str::to_owned))
        .unwrap_or_default();

    match (os_id.as_str(), version_id.as_str()) {
        ("ubuntu", "22.04" | "24.04") => apt_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            true,
        ),
        ("debian", "12" | "13") => {
            let repo_codename = if version_id == "13" { "noble" } else { "jammy" };
            apt_driver_plan(
                os_id,
                version_id,
                repo_codename.to_owned(),
                repo_version_expr,
                dkms,
                false,
            )
        }
        ("rhel", "10.1" | "10.0" | "9.7" | "9.6" | "9.4" | "8.10") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            DnfDriverDistro::Rhel,
        ),
        ("ol", "10.1" | "9.7" | "8.10") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            DnfDriverDistro::Oracle,
        ),
        ("rocky", "9.7") => dnf_driver_plan(
            os_id,
            version_id,
            codename,
            repo_version_expr,
            dkms,
            DnfDriverDistro::Rocky,
        ),
        ("sles" | "sle", "15.7") => {
            sles_driver_plan(os_id, version_id, codename, repo_version_expr, dkms)
        }
        _ => DriverInstallPlan {
            supported: false,
            mutating: false,
            policy: "unsupported_linux_dkms_plan".to_owned(),
            os_id,
            version_id,
            codename,
            repo_version_expr,
            reason: "Linux DKMS driver install is currently planned only for AMD-documented Ubuntu, Debian, RHEL, Oracle Linux, SLES, and Rocky versions; no commands were guessed for this distro.".to_owned(),
            preflight_checks: Vec::new(),
            commands: Vec::new(),
            checks: vec!["rocm examine".to_owned()],
        },
    }
}

#[derive(Debug, Clone, Copy)]
enum DnfDriverDistro {
    Rhel,
    Oracle,
    Rocky,
}

fn apt_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version_expr: String,
    dkms: bool,
    include_linux_modules_extra: bool,
) -> DriverInstallPlan {
    let mut commands = Vec::new();
    if dkms {
        commands.extend([
            driver_command(DriverCommandPhase::Prepare, "sudo apt-get update"),
            driver_command(
                DriverCommandPhase::Prepare,
                "sudo apt-get install -y ca-certificates curl gnupg",
            ),
        ]);
        let header_command = if include_linux_modules_extra {
            "sudo apt-get install -y \"linux-headers-$(uname -r)\" \"linux-modules-extra-$(uname -r)\""
        } else {
            "sudo apt-get install -y \"linux-headers-$(uname -r)\""
        };
        commands.push(driver_command(DriverCommandPhase::Prepare, header_command));
        commands.extend([
            driver_command(
                DriverCommandPhase::Prepare,
                "sudo install -m 0755 -d /etc/apt/keyrings",
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                "curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg",
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "printf '%s\\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/{repo_version_expr}/ubuntu {codename} main' | sudo tee /etc/apt/sources.list.d/amdgpu.list >/dev/null"
                ),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                "printf '%s\\n' 'Package: *' 'Pin: release o=repo.radeon.com' 'Pin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null",
            ),
            driver_command(DriverCommandPhase::Prepare, "sudo apt-get update"),
        ]);
        commands.push(driver_command(
            DriverCommandPhase::Execute,
            "sudo apt-get install -y amdgpu-dkms",
        ));
        commands.extend([
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version_expr,
        reason: if dkms {
            "Plan uses AMD's package-manager DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            vec![
                "root access: run as root, or ensure `sudo -v` succeeds before approval".to_owned(),
                "`sudo` command is available when not running as root".to_owned(),
                "`apt-get` package manager is available".to_owned(),
            ]
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
    }
}

fn dnf_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version_expr: String,
    dkms: bool,
    distro: DnfDriverDistro,
) -> DriverInstallPlan {
    let mut commands = Vec::new();
    if dkms {
        match distro {
            DnfDriverDistro::Rhel => {
                commands.extend(
                    rhel_kernel_prepare_commands(&version_id)
                        .into_iter()
                        .map(|command| driver_command(DriverCommandPhase::Prepare, command)),
                );
            }
            DnfDriverDistro::Oracle => {
                commands.push(driver_command(
                    DriverCommandPhase::Prepare,
                    "sudo dnf install -y \"kernel-uek-devel-$(uname -r)\"",
                ));
            }
            DnfDriverDistro::Rocky => {
                commands.push(driver_command(
                    DriverCommandPhase::Prepare,
                    "sudo dnf install -y kernel-headers kernel-devel kernel-devel-matched",
                ));
            }
        }
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            &format!(
                "sudo dnf install -y {}",
                amdgpu_install_rpm_url(&repo_version_expr, &os_id, &version_id, distro)
            ),
        ));
        commands.push(driver_command(
            DriverCommandPhase::Prepare,
            "sudo dnf clean all",
        ));
        commands.push(driver_command(
            DriverCommandPhase::Execute,
            "sudo dnf install -y amdgpu-dkms",
        ));
        commands.extend([
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version_expr,
        reason: if dkms {
            "Plan uses AMD's documented DNF DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            vec![
                "root access: run as root, or ensure `sudo -v` succeeds before approval"
                    .to_owned(),
                "`sudo` command is available when not running as root".to_owned(),
                "`dnf` package manager is available".to_owned(),
                "enterprise Linux repositories are registered and current before approval"
                    .to_owned(),
            ]
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
    }
}

fn sles_driver_plan(
    os_id: String,
    version_id: String,
    codename: String,
    repo_version_expr: String,
    dkms: bool,
) -> DriverInstallPlan {
    let mut commands = Vec::new();
    if dkms {
        commands.extend([
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("sudo SUSEConnect -p sle-module-desktop-applications/{version_id}/x86_64"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("sudo SUSEConnect -p sle-module-development-tools/{version_id}/x86_64"),
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!("sudo SUSEConnect -p PackageHub/{version_id}/x86_64"),
            ),
            driver_command(DriverCommandPhase::Prepare, "sudo zypper refresh"),
            driver_command(
                DriverCommandPhase::Prepare,
                "sudo zypper install -y kernel-default-devel",
            ),
            driver_command(
                DriverCommandPhase::Prepare,
                &format!(
                    "sudo zypper --no-gpg-checks install -y {}",
                    amdgpu_install_sles_rpm_url(&repo_version_expr, &version_id)
                ),
            ),
            driver_command(DriverCommandPhase::Prepare, "sudo zypper refresh"),
            driver_command(
                DriverCommandPhase::Execute,
                "sudo zypper install -y amdgpu-dkms",
            ),
            driver_command(DriverCommandPhase::Verify, "dkms status amdgpu"),
            driver_command(DriverCommandPhase::Verify, "test -e /dev/kfd"),
            driver_command(
                DriverCommandPhase::Verify,
                "ls /dev/dri/renderD* >/dev/null",
            ),
        ]);
    }

    DriverInstallPlan {
        supported: true,
        mutating: dkms,
        policy: "linux_official_amd_dkms_wrapper".to_owned(),
        os_id,
        version_id,
        codename,
        repo_version_expr,
        reason: if dkms {
            "Plan uses AMD's documented SLES DKMS flow and requires explicit approval before execution."
        } else {
            "DKMS was not requested; this is a non-mutating preflight plan."
        }
        .to_owned(),
        preflight_checks: if dkms {
            vec![
                "root access: run as root, or ensure `sudo -v` succeeds before approval"
                    .to_owned(),
                "`sudo` command is available when not running as root".to_owned(),
                "`zypper` package manager is available".to_owned(),
                "`SUSEConnect` is available and the host is registered before approval".to_owned(),
            ]
        } else {
            Vec::new()
        },
        commands,
        checks: vec![
            "dkms status amdgpu".to_owned(),
            "/sys/module/amdgpu".to_owned(),
            "/dev/kfd".to_owned(),
            "/dev/dri/renderD*".to_owned(),
            "amd-smi version if present".to_owned(),
            "rocminfo if present".to_owned(),
        ],
    }
}

fn rhel_kernel_prepare_commands(version_id: &str) -> Vec<&'static str> {
    if version_id.starts_with("8.") {
        vec![
            "sudo dnf install -y \"kernel-headers-$(uname -r)\"",
            "sudo dnf install -y \"kernel-devel-$(uname -r)\"",
        ]
    } else {
        vec![
            "sudo dnf install -y \"kernel-headers-$(uname -r)\"",
            "sudo dnf install -y \"kernel-devel-$(uname -r)\"",
            "sudo dnf install -y \"kernel-devel-matched-$(uname -r)\"",
        ]
    }
}

fn amdgpu_install_rpm_url(
    repo_version_expr: &str,
    os_id: &str,
    version_id: &str,
    distro: DnfDriverDistro,
) -> String {
    let repo_family = match distro {
        DnfDriverDistro::Rhel => "rhel",
        DnfDriverDistro::Oracle | DnfDriverDistro::Rocky => "el",
    };
    let repo_version = dnf_repo_version_path(os_id, version_id);
    let el_major = linux_major_version(version_id);
    format!(
        "https://repo.radeon.com/amdgpu-install/{repo_version_expr}/{repo_family}/{repo_version}/amdgpu-install-{repo_version_expr}.${{ROCM_CLI_AMDGPU_PACKAGE_RELEASE:-70204}}-1.el{el_major}.noarch.rpm"
    )
}

fn amdgpu_install_sles_rpm_url(repo_version_expr: &str, version_id: &str) -> String {
    format!(
        "https://repo.radeon.com/amdgpu-install/{repo_version_expr}/sle/{version_id}/amdgpu-install-{repo_version_expr}.${{ROCM_CLI_AMDGPU_PACKAGE_RELEASE:-70204}}-1.noarch.rpm"
    )
}

fn dnf_repo_version_path(os_id: &str, version_id: &str) -> String {
    match (os_id, version_id) {
        ("rhel", "10.1" | "10.0") | ("ol", "10.1") => "10".to_owned(),
        ("rhel" | "ol", "8.10") => "8".to_owned(),
        _ => version_id.to_owned(),
    }
}

fn linux_major_version(version_id: &str) -> &str {
    version_id.split('.').next().unwrap_or(version_id)
}

fn driver_command(phase: DriverCommandPhase, command: &str) -> DriverPlanCommand {
    DriverPlanCommand {
        phase,
        command: command.to_owned(),
    }
}

fn render_driver_install_plan(plan: &DriverInstallPlan, yes: bool, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "driver install plan");
    let _ = writeln!(output, "  policy: {}", plan.policy);
    let _ = writeln!(output, "  supported: {}", plan.supported);
    let _ = writeln!(output, "  mutating: {}", plan.mutating);
    let _ = writeln!(
        output,
        "  approval: {}",
        driver_plan_approval_label(plan, yes, dry_run)
    );
    let _ = writeln!(output, "  dry_run: {dry_run}");
    let _ = writeln!(output, "  os_id: {}", empty_as_unknown(&plan.os_id));
    let _ = writeln!(
        output,
        "  version_id: {}",
        empty_as_unknown(&plan.version_id)
    );
    let _ = writeln!(output, "  codename: {}", empty_as_unknown(&plan.codename));
    let _ = writeln!(output, "  repo_version: {}", plan.repo_version_expr);
    let _ = writeln!(output, "  reason: {}", plan.reason);
    if !plan.preflight_checks.is_empty() {
        let _ = writeln!(output, "  preflight_checks:");
        for check in &plan.preflight_checks {
            let _ = writeln!(output, "    {check}");
        }
    }
    let execution_commands = plan
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command.phase,
                DriverCommandPhase::Prepare | DriverCommandPhase::Execute
            )
        })
        .collect::<Vec<_>>();
    if execution_commands.is_empty() {
        let _ = writeln!(output, "  execution_commands: <none>");
    } else {
        let _ = writeln!(output, "  execution_commands:");
        for command in execution_commands {
            let _ = writeln!(output, "    {:?}: {}", command.phase, command.command);
        }
    }
    let verification_commands = plan
        .commands
        .iter()
        .filter(|command| command.phase == DriverCommandPhase::Verify)
        .collect::<Vec<_>>();
    if !verification_commands.is_empty() {
        let _ = writeln!(output, "  post_reboot_check_commands:");
        for command in verification_commands {
            let _ = writeln!(output, "    {}", command.command);
        }
    }
    if !plan.checks.is_empty() {
        let _ = writeln!(output, "  post_reboot_checks:");
        for check in &plan.checks {
            let _ = writeln!(output, "    {check}");
        }
    }
    if plan.supported && plan.mutating && !yes && !dry_run {
        let _ = writeln!(
            output,
            "  action: rerun with --yes after reviewing this plan, or approve from the TUI"
        );
    } else if plan.supported && plan.mutating && dry_run {
        let _ = writeln!(
            output,
            "  action: dry run only; no driver commands executed"
        );
    } else if plan.supported && !plan.mutating {
        let _ = writeln!(
            output,
            "  action: no driver commands will be executed; add --dkms to plan a native DKMS driver install"
        );
    } else if !plan.supported {
        let _ = writeln!(output, "  action: no driver commands will be executed");
    }
    output
}

const fn driver_plan_approval_label(
    plan: &DriverInstallPlan,
    yes: bool,
    dry_run: bool,
) -> &'static str {
    if !plan.supported || !plan.mutating || dry_run {
        "not required"
    } else if yes {
        "approved"
    } else {
        "required"
    }
}

const fn empty_as_unknown(value: &str) -> &str {
    if value.is_empty() { "<unknown>" } else { value }
}

fn parse_os_release_field(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let Some((name, raw_value)) = line.split_once('=') else {
            continue;
        };
        if name != key {
            continue;
        }
        return Some(raw_value.trim().trim_matches('"').to_owned());
    }
    None
}

fn codename_for_version(os_id: &str, version_id: &str) -> Option<&'static str> {
    match (os_id, version_id) {
        ("ubuntu", "22.04") => Some("jammy"),
        ("ubuntu", "24.04") => Some("noble"),
        ("debian", "12") => Some("jammy"),
        ("debian", "13") => Some("noble"),
        _ => None,
    }
}

fn read_os_release() -> Result<String> {
    fs::read_to_string("/etc/os-release").context("failed to read /etc/os-release")
}

fn run_driver_shell_command(command: &str) -> Result<()> {
    let (program, args) = shell_command_for_host(command);
    let status = ProcessCommand::new(program)
        .args(args)
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("failed to launch `{command}`"))?;
    if !status.success() {
        bail!("`{command}` exited with {status}");
    }
    Ok(())
}

fn driver_install_state_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("driver").join("state.json")
}

fn write_driver_install_state(paths: &AppPaths, state: &DriverInstallState) -> Result<()> {
    let path = driver_install_state_path(paths);
    let parent = path.parent().context("driver state path has no parent")?;
    fs::create_dir_all(parent)?;
    fs::write(&path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

fn read_driver_install_state(paths: &AppPaths) -> Result<Option<DriverInstallState>> {
    let path = driver_install_state_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let state = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(state))
}

fn current_boot_id() -> Option<String> {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn driver_reboot_observed(executed_boot_id: Option<&str>) -> bool {
    let Some(executed_boot_id) = executed_boot_id else {
        return false;
    };
    current_boot_id()
        .as_deref()
        .is_some_and(|current| current != executed_boot_id)
}

fn engines(command: EnginesCommand) -> Result<()> {
    match command {
        EnginesCommand::List => {
            print!("{}", render_engine_inventory_text());
            Ok(())
        }
        EnginesCommand::Install {
            engine,
            runtime_id,
            python_version,
            reinstall,
        } => {
            let paths = AppPaths::discover()?;
            let mut config = RocmCliConfig::load(&paths)?;
            let runtime_id =
                resolve_engine_install_runtime_id(&paths, &config, &engine, runtime_id)?;
            if engine == "pytorch"
                && !reinstall
                && python_version.is_none()
                && let Some(manifest) = active_pytorch_runtime(&paths, &runtime_id)?
            {
                let engine_config = config.engine_config_mut(&engine);
                engine_config.last_installed_runtime_id = Some(runtime_id.clone());
                if engine_config.preferred_runtime_id.is_none()
                    && engine_config.preferred_env_id.is_none()
                {
                    engine_config.preferred_runtime_id = Some(runtime_id.clone());
                }
                config.save(&paths)?;
                println!("engine ready");
                println!("  engine: {engine}");
                println!("  runtime_id: {runtime_id}");
                println!("  env_path: {}", manifest.install_root.display());
                record_cli_audit_event(
                    &paths,
                    "engine",
                    "engine_install",
                    "info",
                    format!("ready engine={engine} runtime_id={runtime_id}"),
                    None,
                );
                return Ok(());
            }
            let env_root = env_root_for_engine_install(&paths, &config, &engine, &runtime_id)?;
            let response = engine_request_with_env_root::<_, InstallResponse>(
                Some(&paths),
                &engine,
                EngineMethod::Install,
                &InstallRequest {
                    runtime_id: runtime_id.clone(),
                    python_version,
                    reinstall,
                    env_root: env_root.clone(),
                },
                env_root.as_deref(),
            )?;
            println!("engine install");
            println!("  engine: {engine}");
            println!("  runtime_id: {runtime_id}");
            println!("  reinstall: {reinstall}");
            println!("  env_id: {}", response.env_id);
            println!("  env_path: {}", response.env_path);
            for warning in response.warnings {
                println!("  warning: {warning}");
            }
            if response.managed_env == Some(false) {
                println!("  note: external runtime");
            } else {
                let engine_config = config.engine_config_mut(&engine);
                engine_config.last_installed_runtime_id = Some(runtime_id.clone());
                engine_config.last_installed_env_id = Some(response.env_id.clone());
                let mut seeded_preference = false;
                if engine_config.preferred_runtime_id.is_none()
                    && engine_config.preferred_env_id.is_none()
                {
                    engine_config.preferred_env_id = Some(response.env_id.clone());
                    seeded_preference = true;
                }
                config.save(&paths)?;
                let _ = seeded_preference;
            }
            record_cli_audit_event(
                &paths,
                "engine",
                "engine_install",
                "info",
                format!(
                    "installed engine={} runtime_id={} env_id={} reinstall={}",
                    engine, runtime_id, response.env_id, reinstall
                ),
                None,
            );
            Ok(())
        }
        EnginesCommand::Shell {
            engine,
            runtime_id,
            env_id,
            shell,
        } => engine_shell(
            &engine,
            runtime_id.as_deref(),
            env_id.as_deref(),
            shell.as_deref(),
        ),
    }
}

fn resolve_engine_install_runtime_id(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<String>,
) -> Result<String> {
    if engine_manages_own_runtime(engine) {
        return Ok(runtime_id.unwrap_or_else(|| managed_engine_runtime_id(engine).to_owned()));
    }
    let Some(selector) = runtime_id
        .or_else(|| config.active_runtime_key.clone())
        .or_else(|| config.default_runtime_id.clone())
    else {
        bail!(
            "no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id"
        );
    };
    resolve_runtime_selector_to_exact_key(paths, &selector, "engine install runtime selection")
}

fn engine_manages_own_runtime(engine: &str) -> bool {
    engine == "lemonade"
}

fn env_root_for_runtime(
    paths: &AppPaths,
    engine: &str,
    runtime_id: &str,
) -> Result<Option<PathBuf>> {
    if engine_manages_own_runtime(engine) {
        return Ok(None);
    }
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, runtime_id)?;
    Ok(Some(manifest.install_root.join("engines")))
}

fn env_root_for_engine_install(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: &str,
) -> Result<Option<PathBuf>> {
    if engine_manages_own_runtime(engine) {
        return env_root_for_self_managed_engine(paths, config);
    }
    env_root_for_runtime(paths, engine, runtime_id)
}

fn env_root_for_self_managed_engine(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<PathBuf>> {
    recover_setup_runtime_registration(paths, config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    for selector in [
        config.active_runtime_key.as_deref(),
        config.default_runtime_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(manifest) = runtime_manifest_for_selector(&manifests, selector) {
            return Ok(Some(manifest.install_root.join("engines")));
        }
    }
    let ready = manifests
        .iter()
        .filter(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok())
        .collect::<Vec<_>>();
    Ok(match ready.as_slice() {
        [manifest] => Some(manifest.install_root.join("engines")),
        _ => None,
    })
}

fn runtime_manifest_for_selector<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Option<&'a therock::InstalledRuntimeManifest> {
    manifests
        .iter()
        .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
        .or_else(|| {
            let mut matches = manifests
                .iter()
                .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector));
            let first = matches.next()?;
            if matches.next().is_none() {
                Some(first)
            } else {
                None
            }
        })
}

fn env_root_for_service(
    paths: &AppPaths,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<Option<PathBuf>> {
    if env_id.is_some() {
        return Ok(None);
    }
    match runtime_id {
        Some(runtime_id) => env_root_for_runtime(paths, engine, runtime_id),
        None => Ok(None),
    }
}

fn active_pytorch_runtime(
    paths: &AppPaths,
    runtime_id: &str,
) -> Result<Option<therock::InstalledRuntimeManifest>> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, runtime_id)?;
    if !pytorch_runtime_ready(manifest) {
        return Ok(None);
    }
    Ok(Some(manifest.clone()))
}

fn pytorch_runtime_ready(manifest: &therock::InstalledRuntimeManifest) -> bool {
    manifest.format == "wheel"
        && validate_runtime_manifest_for_activation(manifest).is_ok()
        && manifest
            .rocm_sdk
            .as_ref()
            .is_some_and(|sdk| sdk.import_ok && sdk.root_path.is_some())
}

fn managed_engine_runtime_id(engine: &str) -> &'static str {
    match engine {
        "lemonade" => "lemonade-embeddable-10.6.0",
        _ => "managed-engine-runtime",
    }
}

fn ensure_self_managed_engine_ready(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    engine: &str,
) -> Result<()> {
    if !engine_manages_own_runtime(engine) {
        return Ok(());
    }
    let runtime_id = managed_engine_runtime_id(engine).to_owned();
    let env_root = env_root_for_self_managed_engine(paths, config)?;
    let detect = engine_request::<_, DetectResponse>(
        Some(paths),
        engine,
        EngineMethod::Detect,
        &DetectRequest {
            runtime_id: Some(runtime_id.clone()),
            device_filter: None,
        },
    )
    .ok();
    let installed = detect.as_ref().is_some_and(|detect| {
        detect.installed && detect_runtime_matches_env_root(detect, env_root.as_deref())
    });
    let response = if installed {
        None
    } else {
        eprintln!("Preparing {engine} for GPU serving...");
        Some(engine_request_with_env_root::<_, InstallResponse>(
            Some(paths),
            engine,
            EngineMethod::Install,
            &InstallRequest {
                runtime_id: runtime_id.clone(),
                python_version: None,
                reinstall: false,
                env_root: env_root.clone(),
            },
            env_root.as_deref(),
        )?)
    };

    let engine_config = config.engine_config_mut(engine);
    engine_config.last_installed_runtime_id = Some(runtime_id);
    if let Some(response) = response {
        engine_config.last_installed_env_id = Some(response.env_id.clone());
        if engine_config.preferred_runtime_id.is_none() && engine_config.preferred_env_id.is_none()
        {
            engine_config.preferred_env_id = Some(response.env_id);
        }
    }
    config.save(paths)?;
    Ok(())
}

fn detect_runtime_matches_env_root(detect: &DetectResponse, env_root: Option<&Path>) -> bool {
    let Some(env_root) = env_root else {
        return true;
    };
    detect
        .runtime_executable
        .as_deref()
        .map(PathBuf::from)
        .is_some_and(|runtime_executable| path_is_same_or_inside(&runtime_executable, env_root))
}

#[derive(Debug, Clone, Deserialize)]
struct ManagedEngineEnvManifest {
    env_id: String,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ResolvedEngineEnv {
    env_id: String,
    managed_env_id: Option<String>,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
    source: String,
}

fn engine_shell(
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    shell_override: Option<&str>,
) -> Result<()> {
    if !interactive_terminal() {
        bail!("`rocm engines shell` requires an interactive terminal");
    }

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    if engine == "llama.cpp" {
        bail!(
            "`rocm engines shell llama.cpp` is not available because llama.cpp uses an external llama-server binary, not a managed Python environment"
        );
    }
    let resolved = resolve_engine_env(&paths, &config, engine, runtime_id, env_id)?;
    let shell_program = shell_override
        .map(str::to_owned)
        .or_else(default_interactive_shell_program)
        .context("unable to determine an interactive shell; set --shell or SHELL")?;
    let venv_bin = runtime_python_env_bin_dir(&resolved.env_path);
    let shell_hint = runtime_python_activation_hint(&resolved.env_path);

    println!("engine shell");
    println!("  engine: {engine}");
    println!("  source: {}", resolved.source);
    println!("  env_id: {}", resolved.env_id);
    println!("  runtime_id: {}", resolved.runtime_id);
    println!("  env_path: {}", resolved.env_path.display());
    println!("  python: {}", resolved.python_executable);
    println!("  shell: {shell_program}");
    println!("  activate_hint: {shell_hint}");
    println!("  exit_hint: use `exit` or Ctrl-D to leave the managed env shell");

    let path_with_env = prepend_runtime_path(&venv_bin, std::env::var_os("PATH").as_deref())
        .context("failed to compose PATH for managed engine env shell")?;
    let mut command = ProcessCommand::new(&shell_program);
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("VIRTUAL_ENV", &resolved.env_path)
        .env("PATH", path_with_env)
        .env("ROCM_CLI_ENGINE", engine)
        .env("ROCM_CLI_ENV_ID", &resolved.env_id)
        .env("ROCM_CLI_RUNTIME_ID", &resolved.runtime_id)
        .env("ROCM_CLI_PYTHON", &resolved.python_executable);
    apply_app_path_env(&mut command, &paths);

    if !rocm_core::runtime_is_windows() {
        let prompt = format!("(rocm:{engine}) ");
        command.env("VIRTUAL_ENV_PROMPT", &prompt);
        command.env("PS1", format!("{prompt}${{PS1:-}}"));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to launch shell `{shell_program}`"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("managed engine shell exited with status {status}");
    }
}

fn resolve_engine_env(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<ResolvedEngineEnv> {
    let selection = validate_engine_selection_runtime(
        paths,
        resolve_engine_selection(config, engine, runtime_id, env_id),
    )?;
    if let Some(env_id) = selection.env_id.as_deref() {
        let manifest = load_engine_env_manifest(paths, engine, env_id)?;
        return Ok(ResolvedEngineEnv {
            managed_env_id: Some(manifest.env_id.clone()),
            env_id: manifest.env_id,
            runtime_id: manifest.runtime_id,
            python_executable: manifest.python_executable,
            env_path: manifest.env_path,
            source: selection
                .source
                .unwrap_or_else(|| "manifest_env_id".to_owned()),
        });
    }

    let runtime_id = selection.runtime_id.with_context(|| {
        "no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id"
    })?;
    if engine == "pytorch"
        && let Some(manifest) = active_pytorch_runtime(paths, &runtime_id)?
    {
        let python_executable = manifest.python_executable.clone().unwrap_or_else(|| {
            runtime_python_executable_in_env(&manifest.install_root)
                .display()
                .to_string()
        });
        return Ok(ResolvedEngineEnv {
            env_id: manifest.runtime_key.clone(),
            managed_env_id: None,
            runtime_id,
            python_executable,
            env_path: manifest.install_root,
            source: selection
                .source
                .unwrap_or_else(|| "active_therock_runtime".to_owned()),
        });
    }
    let env_root = env_root_for_engine_install(paths, config, engine, &runtime_id)?;
    let response = engine_request_with_env_root::<_, InstallResponse>(
        Some(paths),
        engine,
        EngineMethod::Install,
        &InstallRequest {
            runtime_id: runtime_id.clone(),
            python_version: None,
            reinstall: false,
            env_root: env_root.clone(),
        },
        env_root.as_deref(),
    )?;
    Ok(ResolvedEngineEnv {
        managed_env_id: Some(response.env_id.clone()),
        env_id: response.env_id,
        runtime_id,
        python_executable: response.python_executable,
        env_path: PathBuf::from(response.env_path),
        source: selection
            .source
            .unwrap_or_else(|| "auto_install".to_owned()),
    })
}

fn load_engine_env_manifest(
    paths: &AppPaths,
    engine: &str,
    env_id: &str,
) -> Result<ManagedEngineEnvManifest> {
    let path = paths
        .engine_manifests_dir(engine)
        .join(format!("{env_id}.json"));
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ServeEngineSelection {
    engine: String,
    source: &'static str,
}

fn select_serve_engine(
    explicit_engine: Option<&str>,
    configured_default: Option<&str>,
    recipe: Option<&ModelRecipeRecord>,
) -> ServeEngineSelection {
    if let Some(engine) = explicit_engine.filter(|value| !value.trim().is_empty()) {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "explicit --engine",
        };
    }

    if let Some(engine) = configured_default.filter(|value| !value.trim().is_empty()) {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "configured default_engine",
        };
    }

    if let Some(engine) = recipe
        .and_then(|recipe| recipe.preferred_engines.first())
        .filter(|value| !value.trim().is_empty())
    {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
        };
    }

    ServeEngineSelection {
        engine: default_engine_for_platform().to_owned(),
        source: "platform default",
    }
}

fn model_recipe_supports_engine(recipe: &ModelRecipeRecord, engine: &str) -> bool {
    recipe
        .preferred_engines
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(engine))
        || recipe
            .engine_recipes
            .iter()
            .any(|candidate| candidate.engine.eq_ignore_ascii_case(engine))
}

fn serve_model_ref_for_engine(
    model: &str,
    recipe: Option<&ModelRecipeRecord>,
    selected_engine: &str,
) -> String {
    recipe
        .filter(|recipe| model_recipe_supports_engine(recipe, selected_engine))
        .map_or_else(
            || model.to_owned(),
            |recipe| recipe.canonical_model_id.clone(),
        )
}

fn serve_engine_selection_line(selection: &ServeEngineSelection) -> String {
    format!("  engine_selection: {}", selection.source)
}

fn render_serve_engine_recipe_lines(engine_recipe: &EngineRecipeHint) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "  engine_recipe_contract: {}",
        engine_recipe.contract_version
    );
    let _ = writeln!(
        output,
        "  engine_recipe_policy: selected-engine required_flags are applied at launch; parser/endpoint metadata is forwarded to the adapter"
    );
    let _ = writeln!(output, "  engine_recipe_engine: {}", engine_recipe.engine);
    if !engine_recipe.required_flags.is_empty() {
        let _ = writeln!(
            output,
            "  engine_recipe_required_flags: {}",
            engine_recipe.required_flags.join(" ")
        );
    }
    output
}

fn protocol_engine_recipe_hint(
    recipe: &ModelRecipeRecord,
    engine: &str,
) -> Option<EngineRecipeHint> {
    recipe
        .engine_recipes
        .iter()
        .find(|engine_recipe| engine_recipe.engine == engine)
        .map(|engine_recipe| EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: engine_recipe.engine.clone(),
            required_flags: engine_recipe.required_flags.clone(),
            parser_settings: engine_recipe.parser_settings.clone(),
            preferred_endpoint: engine_recipe.preferred_endpoint.as_ref().map(|endpoint| {
                EngineRecipeEndpointHint {
                    endpoint_mode: endpoint.endpoint_mode.clone(),
                    settings: endpoint.settings.clone(),
                }
            }),
            unsupported_combinations: engine_recipe
                .unsupported_combinations
                .iter()
                .map(|combination| EngineRecipeUnsupportedCombinationHint {
                    combination: combination.combination.clone(),
                    reason: combination.reason.clone(),
                })
                .collect(),
            notes: engine_recipe.notes.clone(),
        })
}

#[allow(clippy::too_many_arguments)]
fn serve(
    model: String,
    engine: Option<String>,
    device: Option<String>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    foreground: bool,
    managed: bool,
    allow_public_bind: bool,
) -> Result<()> {
    validate_bind_host(&host, allow_public_bind)?;
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    let shared_recipe = resolve_model_recipe(&model)?;
    let serve_engine = select_serve_engine(
        engine.as_deref(),
        config.default_engine.as_deref(),
        shared_recipe.as_ref(),
    );
    let selected_engine = serve_engine.engine.clone();
    let engine_recipe = shared_recipe
        .as_ref()
        .filter(|recipe| model_recipe_supports_engine(recipe, &selected_engine))
        .and_then(|recipe| protocol_engine_recipe_hint(recipe, &selected_engine));
    let engine_model_ref =
        serve_model_ref_for_engine(&model, shared_recipe.as_ref(), &selected_engine);
    let device_policy = parse_device_policy(device.as_deref())?;
    let resolved_selection = resolve_engine_selection(
        &config,
        &selected_engine,
        runtime_id.as_deref(),
        env_id.as_deref(),
    );
    let resolved_selection = validate_engine_selection_runtime(&paths, resolved_selection)?;
    if !matches!(device_policy, DevicePolicy::CpuOnly)
        && resolved_selection.runtime_id.is_none()
        && resolved_selection.env_id.is_none()
        && !engine_manages_own_runtime(&selected_engine)
    {
        bail!(
            "device_policy: {}; no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id/--env-id",
            device_policy_name(&device_policy)
        );
    }
    if !matches!(device_policy, DevicePolicy::CpuOnly)
        && engine_manages_own_runtime(&selected_engine)
    {
        ensure_self_managed_engine_ready(&paths, &mut config, &selected_engine)?;
    }
    let resolve = engine_request::<_, ResolveModelResponse>(
        Some(&paths),
        &selected_engine,
        EngineMethod::ResolveModel,
        &ResolveModelRequest {
            model_ref: engine_model_ref,
            runtime_id: resolved_selection.runtime_id.clone(),
            device_policy: Some(device_policy),
            recipe_override: None,
            engine_recipe,
        },
    )?;
    let service_id = generate_service_id(&selected_engine, &resolve.canonical_model_id);

    println!("serve plan");
    println!("  requested model: {model}");
    println!("  resolved model: {}", resolve.canonical_model_id);
    println!("  engine: {selected_engine}");
    println!("{}", serve_engine_selection_line(&serve_engine));
    println!("  host: {host}");
    println!("  port: {port}");
    if let Some(runtime_id) = resolved_selection.runtime_id.as_deref() {
        println!("  runtime_id: {runtime_id}");
    }
    if let Some(env_id) = resolved_selection.env_id.as_deref() {
        println!("  env_id: {env_id}");
    }
    if let Some(source) = resolved_selection.source.as_deref() {
        println!("  selection_source: {source}");
    }
    println!(
        "  device_policy: {}",
        device_policy_name(&resolve.device_policy)
    );
    if let Some(engine_recipe) = &resolve.engine_recipe {
        print!("{}", render_serve_engine_recipe_lines(engine_recipe));
    }

    let mut managed_runtime_id = resolved_selection.runtime_id.clone();
    let mut managed_env_id = resolved_selection.env_id.clone();
    if managed && selected_engine == "pytorch" {
        let engine_env = resolve_engine_env(
            &paths,
            &config,
            &selected_engine,
            resolved_selection.runtime_id.as_deref(),
            resolved_selection.env_id.as_deref(),
        )?;
        println!("  engine_env_id: {}", engine_env.env_id);
        managed_runtime_id = Some(engine_env.runtime_id);
        managed_env_id = engine_env.managed_env_id;
    }

    if managed {
        start_managed_service(
            &selected_engine,
            &service_id,
            &model,
            &resolve,
            &host,
            port,
            &resolve.device_policy,
            managed_runtime_id.as_deref(),
            managed_env_id.as_deref(),
            resolve.engine_recipe.as_ref(),
        )?;
        ensure_background_helper_running()?;
        return Ok(());
    }

    if !foreground {
        println!("  mode: foreground (default)");
        println!("  note: use --managed to hand supervision to rocmd.");
    }
    run_foreground_service(
        &selected_engine,
        &service_id,
        &resolve.canonical_model_id,
        &host,
        port,
        &resolve.device_policy,
        resolved_selection.runtime_id.as_deref(),
        resolved_selection.env_id.as_deref(),
        resolve.engine_recipe.as_ref(),
    )
}

fn validate_bind_host(host: &str, allow_public_bind: bool) -> Result<()> {
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "`rocm serve --host {host}` is not loopback; pass `--allow-public-bind` before binding a non-local interface"
        );
    }
    Ok(())
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

#[cfg(not(windows))]
fn detach_background_command(command: &mut ProcessCommand) {
    rocm_core::detach_command_session(command);
}

#[cfg(not(windows))]
fn attach_background_stdio(command: &mut ProcessCommand, log_path: Option<&Path>) -> Result<()> {
    if let Some(log_path) = log_path {
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        command
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    Ok(())
}

#[cfg(not(windows))]
fn managed_service_process_command(program: &Path, args: &[String]) -> ProcessCommand {
    let mut command = ProcessCommand::new(program);
    command.args(args);
    command
}

#[allow(clippy::too_many_arguments)]
fn start_managed_service(
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.services_dir())?;

    // Idempotency guard: if a managed service for this engine+model is already
    // alive, surface it and spawn nothing. A second `serve --managed` (e.g. the
    // chat assistant re-issuing the same request) is treated as satisfied, not
    // an error. Keyed on engine+canonical model — the freshly generated
    // `service_id` is timestamp-unique and would never match an existing one.
    // Stale/dead services fall through and relaunch normally.
    if let Some(existing) =
        existing_live_managed_service(&paths, engine, &resolve.canonical_model_id)
    {
        println!("managed service already running");
        println!("  service_id: {}", existing.service_id);
        println!("  endpoint: {}", existing.endpoint_url);
        println!("  status: {}", existing.status);
        println!("  note: existing service detected; no second process spawned");
        record_cli_audit_event(
            &paths,
            "service",
            "managed_service_launch_skipped",
            "info",
            format!(
                "skipped duplicate managed launch engine={engine} model={} existing_service_id={} status={}",
                resolve.canonical_model_id, existing.service_id, existing.status
            ),
            Some(&existing.service_id),
        );
        return Ok(());
    }

    let mut record = ManagedServiceRecord::new(
        &paths,
        service_id,
        engine,
        requested_model,
        resolve.canonical_model_id.clone(),
        host,
        port,
        "managed",
        0,
        runtime_id.map(str::to_owned),
        env_id.map(str::to_owned),
        Some(device_policy_name(device_policy).to_owned()),
    );
    record.engine_recipe_json = engine_recipe
        .map(serde_json::to_string)
        .transpose()
        .context("failed to encode engine recipe hint")?;
    record.write()?;

    if let Some(parent) = record.engine_state_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::File::create(&record.log_path)
        .with_context(|| format!("failed to create {}", record.log_path.display()))?;
    let current_exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let serve_args = builtin_engine_serve_http_args(
        engine,
        service_id,
        &resolve.canonical_model_id,
        host,
        port,
        device_policy,
        runtime_id,
        env_id,
        engine_recipe,
        &record.engine_state_path,
        Some(&record.log_path),
    )?;
    let engine_envs_root = env_root_for_service(&paths, engine, runtime_id, env_id)?;
    #[cfg(windows)]
    let child_pid = {
        let env_values = app_path_env_var_values(&paths, engine_envs_root.as_deref());
        let env_refs = app_path_env_var_refs(&env_values);
        rocm_core::spawn_detached_no_inherit(&current_exe, &serve_args, &env_refs)
            .context("failed to launch managed engine process")?
    };
    #[cfg(not(windows))]
    let child_pid = {
        let mut command = managed_service_process_command(&current_exe, &serve_args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, Some(&record.log_path))?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, &paths);
        if let Some(engine_envs_root) = engine_envs_root.as_deref() {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        let mut child = command
            .spawn()
            .context("failed to launch managed engine process")?;
        let child_pid = child.id();
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check managed engine startup state")?
        {
            bail!(
                "managed engine exited immediately with status {}; inspect {}",
                status,
                record.log_path.display()
            );
        }
        child_pid
    };
    record.supervisor_pid = child_pid;
    record.engine_pid = Some(child_pid);
    record.status = "running".to_owned();
    record.write()?;

    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));

    let readiness = wait_for_service_http_ready(
        engine,
        host,
        port,
        &resolve.canonical_model_id,
        Duration::from_secs(45),
    );
    record.status = if readiness { "ready" } else { "starting" }.to_owned();
    record.write()?;
    let endpoint_url = format!("{}/v1", format_http_base_url(host, port));
    println!("managed service launched");
    println!("  service_id: {service_id}");
    println!("  process_pid: {child_pid}");
    println!("  endpoint: {endpoint_url}");
    println!("  log_path: {}", record.log_path.display());
    println!("  manifest_path: {}", record.manifest_path.display());
    println!(
        "  readiness: {}",
        if readiness { "ready" } else { "starting" }
    );
    record_cli_audit_event(
        &paths,
        "service",
        "managed_service_launch",
        "info",
        format!(
            "launched managed service engine={} model={} endpoint={} readiness={}",
            engine,
            resolve.canonical_model_id,
            endpoint_url,
            if readiness { "ready" } else { "starting" }
        ),
        Some(service_id),
    );
    Ok(())
}

/// Shared daemon-lifecycle entrypoint: ensures the background automation helper
/// (`rocm daemon`) is running, spawning it detached if not. Liveness is read from
/// the file-based automation runtime state. Intentionally `pub(crate)` — reused by
/// both the `serve --managed` path and `automations enable`. Only the spawn result
/// itself (`command.spawn()` / `spawn_detached_no_inherit`) is logged rather than
/// propagated; setup errors (path discovery, stdio attach) still return `Err`.
pub(crate) fn ensure_background_helper_running() -> Result<()> {
    let paths = AppPaths::discover()?;
    if let Some(state) = AutomationRuntimeState::load(&paths)?
        && state.running
        && rocm_core::process_is_running(state.daemon_pid)
    {
        return Ok(());
    }

    let exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let args = vec!["daemon".to_owned()];
    #[cfg(windows)]
    let spawn_result = {
        let env_values = app_path_env_var_values(&paths, None);
        let env_refs = app_path_env_var_refs(&env_values);
        rocm_core::spawn_detached_no_inherit(&exe, &args, &env_refs).map(|_| ())
    };
    #[cfg(not(windows))]
    let spawn_result = {
        let mut command = managed_service_process_command(&exe, &args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, None)?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, &paths);
        command.spawn().map(|_| ())
    };
    match spawn_result {
        Ok(()) => println!("  helper: started background automation daemon"),
        Err(error) => println!("  helper: could not start background automation daemon: {error}"),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_foreground_service(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.engine_state_dir(engine))?;

    let mut record = ManagedServiceRecord::new(
        &paths,
        service_id,
        engine,
        canonical_model_id,
        canonical_model_id,
        host,
        port,
        "foreground",
        std::process::id(),
        runtime_id.map(str::to_owned),
        env_id.map(str::to_owned),
        Some(device_policy_name(device_policy).to_owned()),
    );
    record.write()?;

    println!("foreground service starting");
    println!("  service_id: {service_id}");
    println!("  endpoint: {}/v1", format_http_base_url(host, port));
    println!("  stop: Ctrl-C");

    record.engine_pid = Some(std::process::id());
    record.status = "running".to_owned();
    record.write()?;

    let result = run_builtin_engine_serve_http(
        engine,
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        host.to_owned(),
        port,
        device_policy_name(device_policy),
        runtime_id.map(str::to_owned),
        env_id.map(str::to_owned),
        record.engine_state_path.clone(),
        None,
        engine_recipe.cloned(),
    );
    record.status = if result.is_ok() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;
    result
}

fn services(command: Option<ServicesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    match command.unwrap_or(ServicesCommand::List { all: false }) {
        ServicesCommand::List { all } => {
            print!("{}", render_services_text(&paths, all)?);
            Ok(())
        }
        ServicesCommand::Logs { service_id } => {
            print!("{}", render_service_logs_text(&paths, &service_id)?);
            Ok(())
        }
        ServicesCommand::Stop { service_id, yes } => {
            run_approved_service_action(&paths, "stop_server", &service_id, yes)
        }
        ServicesCommand::Restart { service_id, yes } => {
            run_approved_service_action(&paths, "restart_server", &service_id, yes)
        }
    }
}

fn comfyui(command: Option<ComfyuiCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    match command.unwrap_or(ComfyuiCommand::Status) {
        ComfyuiCommand::Status => {
            print!("{}", comfyui::render_status(&paths, &config)?);
            record_cli_audit_event(
                &paths,
                "app",
                "comfyui_status",
                "info",
                "rendered ComfyUI status",
                None,
            );
            Ok(())
        }
        ComfyuiCommand::ModelsPath => {
            print!("{}", comfyui::render_models_path(&paths)?);
            record_cli_audit_event(
                &paths,
                "app",
                "comfyui_models_path",
                "info",
                "rendered ComfyUI models path",
                None,
            );
            Ok(())
        }
        ComfyuiCommand::Logs { lines } => {
            print!("{}", comfyui::render_logs(&paths, lines)?);
            record_cli_audit_event(
                &paths,
                "app",
                "comfyui_logs",
                "info",
                "rendered ComfyUI logs",
                None,
            );
            Ok(())
        }
        ComfyuiCommand::Install {
            runtime_id,
            reinstall,
            dry_run,
        } => {
            match comfyui::install(
                &paths,
                &config,
                comfyui::ComfyUiInstallOptions {
                    runtime_id: runtime_id.clone(),
                    reinstall,
                    dry_run,
                },
            ) {
                Ok(text) => {
                    print!("{text}");
                    record_cli_audit_event(
                        &paths,
                        "app",
                        if dry_run {
                            "comfyui_install_dry_run"
                        } else {
                            "comfyui_install"
                        },
                        "info",
                        format!(
                            "ComfyUI install completed runtime={} reinstall={} dry_run={}",
                            runtime_id.as_deref().unwrap_or("<selected>"),
                            reinstall,
                            dry_run
                        ),
                        None,
                    );
                    Ok(())
                }
                Err(error) => {
                    record_cli_audit_event(
                        &paths,
                        "app",
                        if dry_run {
                            "comfyui_install_dry_run"
                        } else {
                            "comfyui_install"
                        },
                        "error",
                        format!(
                            "ComfyUI install failed runtime={} reinstall={} dry_run={}: {error}",
                            runtime_id.as_deref().unwrap_or("<selected>"),
                            reinstall,
                            dry_run
                        ),
                        None,
                    );
                    Err(error)
                }
            }
        }
        ComfyuiCommand::Start {
            host,
            port,
            no_open_browser,
        } => match comfyui::start(
            &paths,
            comfyui::ComfyUiStartOptions {
                host,
                port,
                no_open_browser,
            },
        ) {
            Ok(text) => {
                print!("{text}");
                record_cli_audit_event(
                    &paths,
                    "app",
                    "comfyui_start",
                    "info",
                    "ComfyUI start requested",
                    None,
                );
                Ok(())
            }
            Err(error) => {
                record_cli_audit_event(
                    &paths,
                    "app",
                    "comfyui_start",
                    "error",
                    format!("ComfyUI start failed: {error}"),
                    None,
                );
                Err(error)
            }
        },
        ComfyuiCommand::Stop => match comfyui::stop(&paths) {
            Ok(text) => {
                print!("{text}");
                record_cli_audit_event(
                    &paths,
                    "app",
                    "comfyui_stop",
                    "info",
                    "ComfyUI stop requested",
                    None,
                );
                Ok(())
            }
            Err(error) => {
                record_cli_audit_event(
                    &paths,
                    "app",
                    "comfyui_stop",
                    "error",
                    format!("ComfyUI stop failed: {error}"),
                    None,
                );
                Err(error)
            }
        },
    }
}

fn run_approved_service_action(
    paths: &AppPaths,
    tool: &str,
    service_id: &str,
    yes: bool,
) -> Result<()> {
    validate_service_id(service_id)?;
    if !yes {
        bail!(
            "{} local server `{service_id}` requires --yes.\n\nTry: rocm services {} {service_id} --yes",
            service_action_verb(tool),
            service_action_command(tool)
        );
    }
    let sandbox_tool = sandbox_tool_arg_from_service_tool(tool)?;
    let result = run_internal_sandbox_tool(paths, sandbox_tool, Some(service_id.to_owned()), true)?;
    print!("{}", render_service_action_result(tool, &result));
    record_cli_audit_event(
        paths,
        "service",
        tool,
        "info",
        format!(
            "{} managed service {service_id}",
            service_action_past_tense(tool)
        ),
        Some(service_id),
    );
    Ok(())
}

fn sandbox_tool_arg_from_service_tool(tool: &str) -> Result<SandboxToolArg> {
    match tool {
        "stop_server" => Ok(SandboxToolArg::StopServer),
        "restart_server" => Ok(SandboxToolArg::RestartServer),
        "list_servers" => Ok(SandboxToolArg::ListServers),
        other => bail!("unsupported service tool `{other}`"),
    }
}

fn service_action_command(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "restart",
        "stop_server" => "stop",
        _ => "run",
    }
}

fn service_action_verb(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "Restarting",
        "stop_server" => "Stopping",
        _ => "Changing",
    }
}

fn service_action_past_tense(tool: &str) -> &'static str {
    match tool {
        "restart_server" => "restarted",
        "stop_server" => "stopped",
        _ => "updated",
    }
}

fn runtimes(command: Option<RuntimesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;

    match command.unwrap_or(RuntimesCommand::List) {
        RuntimesCommand::List => {
            print!("{}", render_runtimes_text(&paths, &config)?);
        }
        RuntimesCommand::Activate { runtime } => {
            let result = activate_runtime(&paths, &mut config, &runtime)?;
            println!("runtime activated");
            println!("  runtime_id: {}", result.runtime_id);
            println!("  runtime_key: {}", result.runtime_key);
            println!(
                "  changed_from_runtime_key: {}",
                result.previous_runtime_key.as_deref().unwrap_or("<unset>")
            );
            println!(
                "  note: running services keep their recorded runtime until they are restarted"
            );
            println!("  marker: {}", active_runtime_marker_path(&paths).display());
            println!("  config: {}", paths.config_path().display());
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_activate",
                "info",
                format!(
                    "activated runtime_key={} runtime_id={}",
                    result.runtime_key, result.runtime_id
                ),
                None,
            );
        }
        RuntimesCommand::Rollback => {
            let result = rollback_runtime(&paths, &mut config)?;
            println!("runtime rolled back");
            println!("  runtime_id: {}", result.runtime_id);
            println!("  runtime_key: {}", result.runtime_key);
            println!(
                "  changed_from_runtime_key: {}",
                result.previous_runtime_key.as_deref().unwrap_or("<unset>")
            );
            println!(
                "  note: running services keep their recorded runtime until they are restarted"
            );
            println!("  marker: {}", active_runtime_marker_path(&paths).display());
            println!("  config: {}", paths.config_path().display());
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_rollback",
                "info",
                format!(
                    "rolled back to runtime_key={} runtime_id={}",
                    result.runtime_key, result.runtime_id
                ),
                None,
            );
        }
        RuntimesCommand::Uninstall { runtime } => {
            let result = uninstall_runtime(&paths, &mut config, &runtime)?;
            println!("runtime removed");
            println!("  runtime_id: {}", result.runtime_id);
            println!("  runtime_key: {}", result.runtime_key);
            println!("  registry_removed: {}", result.registry_path.display());
            match result.removed_install_root.as_ref() {
                Some(path) => println!("  folder_removed: {}", path.display()),
                None if result.read_only => {
                    println!("  folder_removed: no");
                    println!("  note: existing external runtime folder was left untouched");
                }
                None => println!("  folder_removed: no"),
            }
            if result.was_active {
                println!("  default_runtime: cleared");
                println!("  next step: rocm runtimes activate <runtime_key>");
            }
            println!("  config: {}", paths.config_path().display());
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_uninstall",
                "info",
                format!(
                    "removed runtime_key={} runtime_id={} removed_install_root={}",
                    result.runtime_key,
                    result.runtime_id,
                    result
                        .removed_install_root
                        .as_ref()
                        .map_or_else(|| "none".to_owned(), |path| path.display().to_string())
                ),
                None,
            );
        }
        RuntimesCommand::Import { manifest, replace } => {
            let imported = import_runtime_manifest(&paths, &manifest, replace)?;
            println!("runtime imported");
            println!("  runtime_id: {}", imported.runtime_id);
            println!("  runtime_key: {}", imported.runtime_key);
            println!("  mode: read-only");
            println!("  source: {}", manifest.display());
            println!(
                "  registry: {}",
                runtime_manifest_path(&paths, &imported.runtime_key).display()
            );
            println!(
                "  next step: rocm runtimes activate {}",
                imported.runtime_key
            );
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_import",
                "info",
                format!(
                    "imported read-only runtime_key={} runtime_id={} source={}",
                    imported.runtime_key,
                    imported.runtime_id,
                    manifest.display()
                ),
                None,
            );
        }
        RuntimesCommand::Adopt {
            python,
            root,
            runtime_id,
            runtime_key,
            channel,
            replace,
        } => {
            let adopted = adopt_runtime_from_python_options(
                &paths,
                AdoptRuntimeOptions {
                    python_input: python,
                    install_root: root,
                    runtime_id,
                    runtime_key,
                    channel,
                    replace,
                },
            )?;
            println!("runtime adopted");
            println!("  runtime_id: {}", adopted.runtime_id);
            println!("  runtime_key: {}", adopted.runtime_key);
            println!("  mode: read-only");
            println!(
                "  python_executable: {}",
                adopted.python_executable.as_deref().unwrap_or("<unset>")
            );
            println!("  root: {}", adopted.install_root.display());
            println!(
                "  registry: {}",
                runtime_manifest_path(&paths, &adopted.runtime_key).display()
            );
            println!(
                "  next step: rocm runtimes activate {}",
                adopted.runtime_key
            );
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_adopt",
                "info",
                format!(
                    "adopted read-only runtime_key={} runtime_id={} root={}",
                    adopted.runtime_key,
                    adopted.runtime_id,
                    adopted.install_root.display()
                ),
                None,
            );
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActiveRuntimeMarker {
    runtime_id: String,
    runtime_key: String,
    manifest_path: PathBuf,
    install_root: PathBuf,
    previous_runtime_id: Option<String>,
    previous_runtime_key: Option<String>,
    activated_at_unix_ms: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeActivationResult {
    runtime_id: String,
    runtime_key: String,
    previous_runtime_key: Option<String>,
}

#[derive(Debug, Clone)]
struct RuntimeUninstallResult {
    runtime_id: String,
    runtime_key: String,
    registry_path: PathBuf,
    removed_install_root: Option<PathBuf>,
    read_only: bool,
    was_active: bool,
}

pub(crate) fn render_runtimes_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    recover_setup_runtime_registration(paths, config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "registered ROCm runtimes");
    let _ = writeln!(
        output,
        "  active_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  registry: {}",
        runtime_registry_dir(paths).display()
    );
    let _ = writeln!(
        output,
        "  marker: {}",
        active_runtime_marker_path(paths).display()
    );
    if let Some(active_runtime_key) = config.active_runtime_key.as_deref()
        && !manifests
            .iter()
            .any(|manifest| manifest.runtime_key == active_runtime_key)
    {
        let _ = writeln!(
            output,
            "  active_status: missing manifest for active_runtime_key={active_runtime_key}"
        );
    }
    if config.active_runtime_key.is_none()
        && let Some(default_runtime_id) = config.default_runtime_id.as_deref()
    {
        let matches = manifests
            .iter()
            .filter(|manifest| manifest.runtime_id == default_runtime_id)
            .collect::<Vec<_>>();
        if matches.is_empty() {
            let _ = writeln!(
                output,
                "  active_status: missing manifest for active_runtime_id={default_runtime_id}"
            );
        } else if matches.len() > 1 {
            let _ = writeln!(
                output,
                "  active_status: ambiguous runtime_id={default_runtime_id}; activate one runtime_key: {}",
                runtime_keys_text(&matches)
            );
        }
    }
    if manifests.is_empty() {
        let _ = writeln!(output, "  installed: none");
        let _ = writeln!(
            output,
            "  next step: rocm install sdk --channel release --format wheel"
        );
        return Ok(output);
    }

    let default_runtime_matches = default_runtime_id_matches(config, &manifests);
    let single_default_runtime_key =
        if config.active_runtime_key.is_none() && default_runtime_matches.len() == 1 {
            Some(default_runtime_matches[0].runtime_key.clone())
        } else {
            None
        };
    drop(default_runtime_matches);

    let _ = writeln!(output, "  installed:");
    for manifest in manifests {
        let active = config
            .active_runtime_key
            .as_deref()
            .is_some_and(|runtime_key| runtime_key == manifest.runtime_key)
            || single_default_runtime_key.as_deref() == Some(manifest.runtime_key.as_str());
        let rollback = config
            .previous_runtime_key
            .as_deref()
            .is_some_and(|runtime_key| runtime_key == manifest.runtime_key);
        let marker = if active {
            "*"
        } else if rollback {
            "-"
        } else {
            " "
        };
        let status = runtime_usability_status(&manifest);
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        let _ = writeln!(
            output,
            "  {marker} {} runtime_id={} version={} format={} family={} mode={} status={}",
            manifest.runtime_key,
            manifest.runtime_id,
            therock::runtime_version_display(&manifest.version),
            manifest.format,
            manifest.family,
            mode,
            status
        );
        let _ = writeln!(
            output,
            "      install_root: {}",
            manifest.install_root.display()
        );
    }

    Ok(output)
}

pub(crate) fn activate_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    selector: &str,
) -> Result<RuntimeActivationResult> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, selector)?;
    validate_runtime_manifest_for_activation(manifest)?;
    let current = current_runtime_manifest(config, &manifests);
    let previous_runtime_key = current
        .as_ref()
        .map(|manifest| manifest.runtime_key.clone())
        .filter(|runtime_key| runtime_key != &manifest.runtime_key);
    let previous_runtime_id = current
        .as_ref()
        .map(|manifest| manifest.runtime_id.clone())
        .filter(|_| previous_runtime_key.is_some());

    config.default_runtime_id = Some(manifest.runtime_id.clone());
    config.active_runtime_key = Some(manifest.runtime_key.clone());
    config.previous_runtime_key = previous_runtime_key.clone();
    config.save(paths)?;
    write_active_runtime_marker(
        paths,
        ActiveRuntimeMarker {
            runtime_id: manifest.runtime_id.clone(),
            runtime_key: manifest.runtime_key.clone(),
            manifest_path: runtime_manifest_path(paths, &manifest.runtime_key),
            install_root: manifest.install_root.clone(),
            previous_runtime_id,
            previous_runtime_key: previous_runtime_key.clone(),
            activated_at_unix_ms: rocm_core::unix_time_millis(),
        },
    )?;

    Ok(RuntimeActivationResult {
        runtime_id: manifest.runtime_id.clone(),
        runtime_key: manifest.runtime_key.clone(),
        previous_runtime_key,
    })
}

fn rollback_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
) -> Result<RuntimeActivationResult> {
    let previous_key = config
        .previous_runtime_key
        .clone()
        .context("no previous runtime is recorded; activate another runtime before rollback")?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let previous = select_runtime_manifest(&manifests, &previous_key)?;
    validate_runtime_manifest_for_activation(previous)?;
    let current = current_runtime_manifest(config, &manifests);
    let new_previous_key = current
        .as_ref()
        .map(|manifest| manifest.runtime_key.clone())
        .filter(|runtime_key| runtime_key != &previous.runtime_key);
    let new_previous_id = current
        .as_ref()
        .map(|manifest| manifest.runtime_id.clone())
        .filter(|_| new_previous_key.is_some());

    config.default_runtime_id = Some(previous.runtime_id.clone());
    config.active_runtime_key = Some(previous.runtime_key.clone());
    config.previous_runtime_key = new_previous_key.clone();
    config.save(paths)?;
    write_active_runtime_marker(
        paths,
        ActiveRuntimeMarker {
            runtime_id: previous.runtime_id.clone(),
            runtime_key: previous.runtime_key.clone(),
            manifest_path: runtime_manifest_path(paths, &previous.runtime_key),
            install_root: previous.install_root.clone(),
            previous_runtime_id: new_previous_id,
            previous_runtime_key: new_previous_key.clone(),
            activated_at_unix_ms: rocm_core::unix_time_millis(),
        },
    )?;

    Ok(RuntimeActivationResult {
        runtime_id: previous.runtime_id.clone(),
        runtime_key: previous.runtime_key.clone(),
        previous_runtime_key: new_previous_key,
    })
}

fn uninstall_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    selector: &str,
) -> Result<RuntimeUninstallResult> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, selector)?.clone();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    let was_active = current_runtime_manifest(config, &manifests)
        .is_some_and(|current| current.runtime_key == manifest.runtime_key);
    let remove_install_root = should_remove_runtime_install_root(&manifest)?;

    let mut removed_install_root = None;
    if remove_install_root && manifest.install_root.exists() {
        fs::remove_dir_all(&manifest.install_root).with_context(|| {
            format!(
                "failed to remove runtime folder {}",
                manifest.install_root.display()
            )
        })?;
        removed_install_root = Some(manifest.install_root.clone());
    }

    if registry_path.exists() {
        fs::remove_file(&registry_path).with_context(|| {
            format!(
                "failed to remove runtime registry entry {}",
                registry_path.display()
            )
        })?;
    }

    let mut config_changed = false;
    if config
        .active_runtime_key
        .as_deref()
        .is_some_and(|runtime_key| runtime_key.eq_ignore_ascii_case(&manifest.runtime_key))
    {
        config.active_runtime_key = None;
        config_changed = true;
    }
    if config
        .previous_runtime_key
        .as_deref()
        .is_some_and(|runtime_key| runtime_key.eq_ignore_ascii_case(&manifest.runtime_key))
    {
        config.previous_runtime_key = None;
        config_changed = true;
    }
    if config
        .default_runtime_id
        .as_deref()
        .is_some_and(|runtime_id| runtime_id.eq_ignore_ascii_case(&manifest.runtime_id))
        && (was_active
            || !manifests.iter().any(|other| {
                other.runtime_key != manifest.runtime_key
                    && other.runtime_id.eq_ignore_ascii_case(&manifest.runtime_id)
            }))
    {
        config.default_runtime_id = None;
        config_changed = true;
    }
    if config
        .setup
        .therock_venv
        .as_ref()
        .is_some_and(|path| paths_equivalent(path, &manifest.install_root))
    {
        config.setup.therock_venv = None;
        config.setup.completed = false;
        config.onboarding_dismissed = false;
        config_changed = true;
    }
    if config_changed {
        config.save(paths)?;
    }

    if active_runtime_marker_matches(paths, &manifest.runtime_key)? {
        let marker_path = active_runtime_marker_path(paths);
        if marker_path.exists() {
            fs::remove_file(&marker_path).with_context(|| {
                format!(
                    "failed to remove active runtime marker {}",
                    marker_path.display()
                )
            })?;
        }
    }

    Ok(RuntimeUninstallResult {
        runtime_id: manifest.runtime_id,
        runtime_key: manifest.runtime_key,
        registry_path,
        removed_install_root,
        read_only: manifest.read_only,
        was_active,
    })
}

fn should_remove_runtime_install_root(
    manifest: &therock::InstalledRuntimeManifest,
) -> Result<bool> {
    if manifest.read_only || manifest.imported_from.is_some() {
        return Ok(false);
    }
    if !local_runtime_manifest_matches(manifest)? {
        return Ok(false);
    }
    ensure_runtime_install_root_is_safe_to_remove(&manifest.install_root)?;
    Ok(true)
}

fn local_runtime_manifest_matches(manifest: &therock::InstalledRuntimeManifest) -> Result<bool> {
    let local_path = manifest.install_root.join(".rocm-cli-runtime.json");
    if !local_path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&local_path)
        .with_context(|| format!("failed to read {}", local_path.display()))?;
    let local: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", local_path.display()))?;
    Ok(local.runtime_key == manifest.runtime_key
        && local.runtime_id == manifest.runtime_id
        && paths_equivalent(&local.install_root, &manifest.install_root))
}

fn ensure_runtime_install_root_is_safe_to_remove(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() || path.parent().is_none() || path.file_name().is_none() {
        bail!(
            "refusing to remove unsafe runtime folder {}",
            path.display()
        );
    }
    Ok(())
}

fn active_runtime_marker_matches(paths: &AppPaths, runtime_key: &str) -> Result<bool> {
    let marker_path = active_runtime_marker_path(paths);
    if !marker_path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&marker_path)
        .with_context(|| format!("failed to read {}", marker_path.display()))?;
    let marker: ActiveRuntimeMarker = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", marker_path.display()))?;
    Ok(marker.runtime_key.eq_ignore_ascii_case(runtime_key))
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = normalize_path_for_compare(left);
    let right = normalize_path_for_compare(right);
    rocm_core::runtime_paths_equivalent(&left, &right)
}

fn path_is_same_or_inside(path: &Path, base: &Path) -> bool {
    let path = normalize_path_for_compare(path);
    let base = normalize_path_for_compare(base);
    runtime_path_is_same_or_inside(&path, &base)
}

fn normalize_path_for_compare(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

fn import_runtime_manifest(
    paths: &AppPaths,
    manifest_path: &Path,
    replace: bool,
) -> Result<therock::InstalledRuntimeManifest> {
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let mut manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_key",
            manifest_path.display()
        );
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_id",
            manifest_path.display()
        );
    }

    manifest.read_only = true;
    manifest.imported_from = Some(
        manifest_path
            .canonicalize()
            .unwrap_or_else(|_| manifest_path.to_path_buf()),
    );
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("imported runtime `{}` is not usable", manifest.runtime_key))?;

    write_runtime_registry_manifest(paths, &manifest, replace)?;

    Ok(manifest)
}

#[derive(Debug, Clone)]
struct SdkInstallFinalization {
    runtime_key: String,
    install_root: PathBuf,
}

fn print_sdk_install_success(finalized: &SdkInstallFinalization) {
    print!("{}", render_sdk_install_success(finalized));
}

fn render_sdk_install_success(finalized: &SdkInstallFinalization) -> String {
    format!(
        "ROCm SDK installed successfully.\n  install folder: {}\n  active runtime: {}\n  next step: run `rocm help` to see how to use rocm-cli.\n",
        finalized.install_root.display(),
        finalized.runtime_key
    )
}

fn finalize_successful_sdk_install(paths: &AppPaths) -> Result<Option<SdkInstallFinalization>> {
    let Some(manifest) = newest_installed_runtime_manifest(paths)? else {
        return Ok(None);
    };
    let mut config = RocmCliConfig::load(paths)?;
    config.setup.completed = true;
    config.setup.therock_venv = Some(manifest.install_root.clone());
    config.save(paths)?;

    let activation_paths = paths
        .clone()
        .with_managed_root(manifest.install_root.clone(), false);
    if paths.config_dir != activation_paths.config_dir
        || paths.data_dir != activation_paths.data_dir
    {
        recover_setup_runtime_registration(paths, &config)?;
        let mut current_config = RocmCliConfig::load(paths)?;
        current_config.setup.completed = true;
        current_config.setup.therock_venv = Some(manifest.install_root.clone());
        let _ = activate_runtime(paths, &mut current_config, &manifest.runtime_key)?;
    }

    recover_setup_runtime_registration(&activation_paths, &config)?;

    let mut config = RocmCliConfig::load(&activation_paths)?;
    config.setup.completed = true;
    config.setup.therock_venv = Some(manifest.install_root.clone());
    let activation = activate_runtime(&activation_paths, &mut config, &manifest.runtime_key)?;

    Ok(Some(SdkInstallFinalization {
        runtime_key: activation.runtime_key,
        install_root: manifest.install_root,
    }))
}

fn newest_installed_runtime_manifest(
    paths: &AppPaths,
) -> Result<Option<therock::InstalledRuntimeManifest>> {
    let mut manifests = therock::load_runtime_manifests(paths)?;
    manifests.sort_by(|left, right| {
        right
            .installed_at_unix_ms
            .cmp(&left.installed_at_unix_ms)
            .then_with(|| left.runtime_key.cmp(&right.runtime_key))
    });
    Ok(manifests.into_iter().next())
}

#[derive(Debug, Clone)]
struct AdoptRuntimeRequest {
    python_executable: PathBuf,
    install_root: PathBuf,
    runtime_id: String,
    runtime_key: String,
    replace: bool,
}

#[derive(Debug, Clone)]
struct AdoptRuntimeOptions {
    python_input: PathBuf,
    install_root: Option<PathBuf>,
    runtime_id: Option<String>,
    runtime_key: Option<String>,
    channel: Option<String>,
    replace: bool,
}

fn adopt_runtime_from_python_options(
    paths: &AppPaths,
    options: AdoptRuntimeOptions,
) -> Result<therock::InstalledRuntimeManifest> {
    let (python_executable, inferred_root) = resolve_adopt_python_input(&options.python_input)?;
    let probe = therock::probe_rocm_sdk_runtime(&python_executable)
        .with_context(|| format!("failed to probe {}", python_executable.display()))?;
    let request = infer_adopt_runtime_request(
        python_executable,
        options.install_root.or(inferred_root),
        options.runtime_id,
        options.runtime_key,
        options.channel,
        options.replace,
        &probe,
    )?;
    adopt_runtime_from_probe(paths, request, probe)
}

fn infer_adopt_runtime_request(
    python_executable: PathBuf,
    install_root: Option<PathBuf>,
    runtime_id: Option<String>,
    runtime_key: Option<String>,
    channel: Option<String>,
    replace: bool,
    probe: &therock::RocmSdkPythonProbe,
) -> Result<AdoptRuntimeRequest> {
    let install_root = install_root.with_context(|| {
        format!(
            "could not infer the Python environment folder from {}; pass --root",
            python_executable.display()
        )
    })?;
    let runtime_id = match runtime_id {
        Some(value) if !value.trim().is_empty() => {
            if let Some(channel) = channel.as_deref() {
                let (parsed_channel, _) = parse_therock_runtime_id(&value)?;
                let requested_channel = normalize_adopt_channel(channel)?;
                if parsed_channel != requested_channel {
                    bail!(
                        "--channel {requested_channel} does not match runtime id channel {parsed_channel}"
                    );
                }
            }
            value
        }
        Some(_) => bail!("runtime_id must not be empty"),
        None => {
            let channel = normalize_adopt_channel(channel.as_deref().unwrap_or("release"))?;
            let family = probe
                .resolved_target_family
                .as_deref()
                .or(probe.default_target_family.as_deref())
                .filter(|value| !value.trim().is_empty())
                .context(
                    "rocm_sdk probe did not report a GPU package; pass --runtime-id explicitly",
                )?;
            format!("therock-{channel}:{family}")
        }
    };
    let runtime_key = match runtime_key {
        Some(value) if !value.trim().is_empty() => value,
        Some(_) => bail!("runtime_key must not be empty"),
        None => {
            let (channel, family) = parse_therock_runtime_id(&runtime_id)?;
            let version = probe
                .rocm_sdk_version
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("rocm_sdk probe did not report a version; cannot name adopted runtime")?;
            format!(
                "adopted-{channel}-pip-{}-{}",
                runtime_key_component(&family),
                runtime_key_component(version)
            )
        }
    };
    Ok(AdoptRuntimeRequest {
        python_executable,
        install_root,
        runtime_id,
        runtime_key,
        replace,
    })
}

fn normalize_adopt_channel(channel: &str) -> Result<String> {
    match channel.trim().to_ascii_lowercase().as_str() {
        "release" => Ok("release".to_owned()),
        "nightly" => Ok("nightly".to_owned()),
        other => bail!("adopt channel must be release or nightly, got `{other}`"),
    }
}

fn runtime_key_component(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    let trimmed = output.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "runtime".to_owned()
    } else {
        trimmed
    }
}

fn resolve_adopt_python_input(input: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let absolute = if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(input)
    };
    if absolute.is_dir() {
        let env_root = absolute.canonicalize().with_context(|| {
            format!(
                "failed to resolve Python environment folder {}",
                absolute.display()
            )
        })?;
        let python = runtime_python_executable_in_env(&env_root);
        if !python.is_file() {
            bail!(
                "Python executable is missing in {}",
                python.parent().unwrap_or(env_root.as_path()).display()
            );
        }
        return Ok((python, Some(env_root)));
    }
    if absolute.is_file() {
        let inferred_root = infer_python_env_root(&absolute);
        return Ok((absolute, inferred_root));
    }
    bail!(
        "Python executable or folder is missing: {}",
        absolute.display()
    );
}

fn infer_python_env_root(python_executable: &Path) -> Option<PathBuf> {
    let bin_dir = python_executable.parent()?;
    let bin_name = bin_dir.file_name()?.to_string_lossy();
    if bin_name.eq_ignore_ascii_case("Scripts") || bin_name == "bin" {
        return bin_dir.parent().map(Path::to_path_buf);
    }
    bin_dir.parent().map(Path::to_path_buf)
}

fn adopt_runtime_from_probe(
    paths: &AppPaths,
    request: AdoptRuntimeRequest,
    probe: therock::RocmSdkPythonProbe,
) -> Result<therock::InstalledRuntimeManifest> {
    if request.runtime_key.trim().is_empty() {
        bail!("runtime_key must not be empty");
    }
    if request.runtime_id.trim().is_empty() {
        bail!("runtime_id must not be empty");
    }
    let python_executable = absolute_existing_file_path_preserving_symlink(
        &request.python_executable,
        "runtime Python executable",
    )?;
    let install_root = request.install_root.canonicalize().with_context(|| {
        format!(
            "runtime install root is missing: {}",
            request.install_root.display()
        )
    })?;
    if !install_root.is_dir() {
        bail!(
            "runtime install root is missing: {}",
            install_root.display()
        );
    }
    let (channel, family) = parse_therock_runtime_id(&request.runtime_id)?;
    therock::validate_rocm_sdk_runtime_probe(&probe)?;
    let version = probe
        .rocm_sdk_version
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .context("rocm_sdk probe did not report a version; cannot adopt runtime explicitly")?
        .to_owned();

    let manifest = therock::InstalledRuntimeManifest {
        runtime_key: request.runtime_key,
        runtime_id: request.runtime_id,
        channel,
        format: "wheel".to_owned(),
        family,
        family_source: "runtime_id".to_owned(),
        version,
        install_root: install_root.clone(),
        selected_artifact_url: "adopted-read-only".to_owned(),
        index_url: None,
        tarball_file_name: None,
        python_launcher: None,
        python_executable: Some(python_executable.display().to_string()),
        pip_cache_dir: None,
        rocm_sdk: Some(probe),
        read_only: true,
        imported_from: Some(install_root),
        installed_at_unix_ms: rocm_core::unix_time_millis(),
    };
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("adopted runtime `{}` is not usable", manifest.runtime_key))?;
    write_runtime_registry_manifest(paths, &manifest, request.replace)?;
    Ok(manifest)
}

fn absolute_existing_file_path_preserving_symlink(path: &Path, label: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(path)
    };
    if !absolute.is_file() {
        bail!("{label} is missing: {}", absolute.display());
    }
    Ok(absolute)
}

fn parse_therock_runtime_id(runtime_id: &str) -> Result<(String, String)> {
    let (prefix, family) = runtime_id.split_once(':').with_context(|| {
        format!("runtime_id `{runtime_id}` must include a TheRock family suffix after ':'")
    })?;
    let family = family.trim();
    if family.is_empty() {
        bail!("runtime_id `{runtime_id}` has an empty TheRock family suffix");
    }
    let channel = match prefix.trim() {
        "therock-release" => "release",
        "therock-nightly" => "nightly",
        other => bail!(
            "runtime_id `{runtime_id}` must start with therock-release: or therock-nightly:, got `{other}`"
        ),
    };
    Ok((channel.to_owned(), family.to_owned()))
}

fn write_runtime_registry_manifest(
    paths: &AppPaths,
    manifest: &therock::InstalledRuntimeManifest,
    replace: bool,
) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    if registry_path.exists() && !replace {
        bail!(
            "runtime registry entry already exists: {}; pass --replace to overwrite it",
            registry_path.display()
        );
    }
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&manifest)
            .context("failed to serialize runtime registry manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;
    Ok(())
}

fn recover_setup_runtime_registration(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    let Some(setup_root) = config
        .setup
        .therock_venv
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
    else {
        return Ok(None);
    };
    if !setup_root.is_dir() {
        return Ok(None);
    }

    let local_manifest_path = setup_root.join(".rocm-cli-runtime.json");
    if !local_manifest_path.is_file() {
        return Ok(None);
    }

    let bytes = fs::read(&local_manifest_path)
        .with_context(|| format!("failed to read {}", local_manifest_path.display()))?;
    let manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", local_manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "setup runtime manifest {} has an empty runtime_key",
            local_manifest_path.display()
        );
    }
    if !paths_equivalent(&manifest.install_root, setup_root) {
        bail!(
            "setup runtime manifest {} points at {}, but setup is configured for {}",
            local_manifest_path.display(),
            manifest.install_root.display(),
            setup_root.display()
        );
    }
    validate_runtime_manifest_for_activation(&manifest).with_context(|| {
        format!(
            "setup runtime `{}` from {} is not usable",
            manifest.runtime_key,
            local_manifest_path.display()
        )
    })?;

    if !runtime_manifest_path(paths, &manifest.runtime_key).is_file() {
        write_runtime_registry_manifest(paths, &manifest, false).with_context(|| {
            format!(
                "failed to restore setup runtime `{}` into {}",
                manifest.runtime_key,
                runtime_registry_dir(paths).display()
            )
        })?;
    }
    Ok(Some(manifest.runtime_key))
}

fn current_runtime_manifest<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Option<&'a therock::InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref()
        && let Some(manifest) = manifests
            .iter()
            .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(active_key))
    {
        return Some(manifest);
    }

    let matches = default_runtime_id_matches(config, manifests);
    match matches.as_slice() {
        [manifest] => Some(*manifest),
        _ => None,
    }
}

fn default_runtime_id_matches<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Vec<&'a therock::InstalledRuntimeManifest> {
    let Some(default_runtime_id) = config.default_runtime_id.as_deref() else {
        return Vec::new();
    };
    manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id))
        .collect()
}

fn runtime_keys_text(manifests: &[&therock::InstalledRuntimeManifest]) -> String {
    manifests
        .iter()
        .map(|manifest| manifest.runtime_key.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn select_runtime_manifest<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    let selector = selector.trim();
    if selector.is_empty() {
        bail!("runtime selector must not be empty");
    }

    if let Some(manifest) = manifests
        .iter()
        .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
    {
        return Ok(manifest);
    }

    let matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [manifest] => Ok(manifest),
        [] => bail!("installed runtime not found: {selector}"),
        _ => {
            let keys = matches
                .iter()
                .map(|manifest| manifest.runtime_key.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "runtime selector `{selector}` matches multiple installed runtimes; activate one by runtime_key: {keys}"
            );
        }
    }
}

pub(crate) fn runtime_usability_status(manifest: &therock::InstalledRuntimeManifest) -> String {
    match validate_runtime_manifest_for_activation(manifest) {
        Ok(()) => "ready".to_owned(),
        Err(error) => format!("unusable ({error})"),
    }
}

fn validate_runtime_manifest_for_activation(
    manifest: &therock::InstalledRuntimeManifest,
) -> Result<()> {
    if manifest.runtime_key.trim().is_empty() {
        bail!("manifest runtime_key is empty");
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!("manifest runtime_id is empty");
    }
    if !manifest.install_root.is_dir() {
        bail!(
            "install root is missing: {}",
            manifest.install_root.display()
        );
    }
    let local_manifest = manifest.install_root.join(".rocm-cli-runtime.json");
    if !manifest.read_only && !local_manifest.is_file() {
        bail!(
            "local runtime manifest is missing: {}",
            local_manifest.display()
        );
    }

    match manifest.format.as_str() {
        "wheel" => validate_wheel_runtime_manifest(manifest),
        "tarball" => validate_tarball_runtime_manifest(manifest),
        other => bail!("unsupported runtime format in manifest: {other}"),
    }
}

fn validate_wheel_runtime_manifest(manifest: &therock::InstalledRuntimeManifest) -> Result<()> {
    let python_executable = manifest
        .python_executable
        .as_deref()
        .context("pip runtime manifest is missing python_executable")?;
    if !Path::new(python_executable).is_file() {
        bail!("runtime Python executable is missing: {python_executable}");
    }
    let probe = manifest
        .rocm_sdk
        .as_ref()
        .context("pip runtime manifest is missing rocm_sdk probe data")?;
    therock::validate_rocm_sdk_runtime_probe(probe)?;
    Ok(())
}

fn validate_tarball_runtime_manifest(manifest: &therock::InstalledRuntimeManifest) -> Result<()> {
    if runtime_install_root_has_payload(&manifest.install_root)? {
        Ok(())
    } else {
        bail!(
            "tarball runtime install root has no payload files: {}",
            manifest.install_root.display()
        )
    }
}

fn runtime_install_root_has_payload(path: &Path) -> Result<bool> {
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            return Ok(true);
        }
    }
    Ok(false)
}

fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

fn runtime_manifest_path(paths: &AppPaths, runtime_key: &str) -> PathBuf {
    runtime_registry_dir(paths).join(format!("{runtime_key}.json"))
}

fn active_runtime_marker_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("active.json")
}

fn write_active_runtime_marker(paths: &AppPaths, marker: ActiveRuntimeMarker) -> Result<()> {
    let path = active_runtime_marker_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("active runtime marker path has no parent directory")?,
    )?;
    let tmp_path = path.with_extension(format!("json.tmp-{}", rocm_core::unix_time_millis()));
    fs::write(
        &tmp_path,
        serde_json::to_vec_pretty(&marker).context("failed to serialize active runtime marker")?,
    )
    .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    if path.exists() {
        let _ = fs::remove_file(&path);
    }
    fs::rename(&tmp_path, &path).with_context(|| {
        format!(
            "failed to move active runtime marker {} into {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn config(command: ConfigCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;

    match command {
        ConfigCommand::Show => {
            print!("{}", render_config_text(&paths, &config));
        }
        ConfigCommand::SetEngine {
            engine,
            runtime_id,
            env_id,
            clear,
        } => {
            let entry = config.engine_config_mut(&engine);
            if clear {
                entry.preferred_runtime_id = None;
                entry.preferred_env_id = None;
            } else if let Some(runtime_id) = runtime_id {
                entry.preferred_runtime_id = Some(runtime_id);
                entry.preferred_env_id = None;
            } else if let Some(env_id) = env_id {
                entry.preferred_env_id = Some(env_id);
                entry.preferred_runtime_id = None;
            } else {
                bail!("set-engine requires --runtime-id, --env-id, or --clear");
            }
            config.save(&paths)?;
            println!("updated engine config for {engine}");
        }
        ConfigCommand::SetDefaultEngine { engine } => {
            config.default_engine = Some(engine.clone());
            config.save(&paths)?;
            println!("default engine set to {engine}");
        }
        ConfigCommand::ClearDefaultEngine => {
            config.default_engine = None;
            config.save(&paths)?;
            println!("default engine cleared");
        }
        ConfigCommand::SetDefaultRuntime { runtime_id } => {
            config.default_runtime_id = Some(runtime_id.clone());
            config.active_runtime_key = None;
            config.previous_runtime_key = None;
            config.save(&paths)?;
            let _ = fs::remove_file(active_runtime_marker_path(&paths));
            println!("default runtime set to {runtime_id}");
        }
        ConfigCommand::ClearDefaultRuntime => {
            config.default_runtime_id = None;
            config.active_runtime_key = None;
            config.previous_runtime_key = None;
            config.save(&paths)?;
            let _ = fs::remove_file(active_runtime_marker_path(&paths));
            println!("default runtime cleared");
        }
        ConfigCommand::SetTelemetry { mode } => {
            config.telemetry.mode = mode.as_str().to_owned();
            config.save(&paths)?;
            println!("telemetry mode set to {}", mode.as_str());
            println!("  policy: {}", telemetry_policy_summary(&config.telemetry));
        }
        ConfigCommand::SetPermissions { mode } => {
            config.permissions.mode = mode.as_str().to_owned();
            config.save(&paths)?;
            record_cli_audit_event(
                &paths,
                "permissions",
                "set_mode",
                "info",
                format!("permissions mode set to {}", mode.as_str()),
                None,
            );
            println!("permissions mode set to {}", mode.as_str());
        }
        ConfigCommand::SetPlannerProvider { provider } => {
            let provider = provider_name(provider);
            config.planner_provider = Some(provider.to_owned());
            config.save(&paths)?;
            println!("planner provider set to {provider}");
            if provider != "local" && !config.provider_enabled(provider) {
                println!(
                    "  next step: rocm config enable-provider {provider} before provider-assisted planning can send prompts"
                );
            }
        }
        ConfigCommand::ClearPlannerProvider => {
            config.planner_provider = None;
            config.save(&paths)?;
            println!("planner provider cleared");
        }
        ConfigCommand::EnableProvider { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider is always enabled and does not send prompts to a cloud API");
            }
            config.provider_config_mut(provider).enabled = true;
            config.save(&paths)?;
            println!("provider {provider} enabled for prompt sending");
            match providers::provider_key_status_text(provider) {
                Ok(status) if status.starts_with("no key saved") => {
                    println!("  key: {status}");
                    println!("  next step: rocm config set-provider-key {provider}");
                }
                Ok(status) => println!("  key: {status}"),
                Err(error) => println!("  key: unavailable ({error})"),
            }
        }
        ConfigCommand::DisableProvider { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider is always enabled and does not send prompts to a cloud API");
            }
            config.provider_config_mut(provider).enabled = false;
            config.save(&paths)?;
            println!("provider {provider} disabled for prompt sending");
        }
        ConfigCommand::SetProviderKey { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider does not use a cloud API key");
            }
            let key = read_provider_key_from_user(provider)?;
            let status = provider_keys::set_provider_api_key(provider, &key)?;
            println!("{provider} API key saved");
            println!(
                "  key: {}",
                provider_keys::provider_key_status_label(&status)
            );
            println!(
                "  prompt sending: {}",
                if config.provider_enabled(provider) {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            if !config.provider_enabled(provider) {
                println!("  next step: rocm config enable-provider {provider}");
            }
        }
        ConfigCommand::ClearProviderKey { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider does not use a cloud API key");
            }
            let status = provider_keys::clear_provider_api_key(provider)?;
            println!("{provider} API key cleared");
            println!(
                "  key: {}",
                provider_keys::provider_key_status_label(&status)
            );
            println!(
                "  prompt sending: {}",
                if config.provider_enabled(provider) {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }
    }

    Ok(())
}

fn read_provider_key_from_user(provider: &str) -> Result<String> {
    let key = if interactive_terminal() {
        rpassword::prompt_password(format!("Paste {provider} API key: "))
            .context("failed to read provider API key")?
    } else {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .context("failed to read provider API key from stdin")?;
        input
    };
    let key = key.trim().to_owned();
    if key.is_empty() {
        bail!("{provider} API key was empty; nothing was saved");
    }
    Ok(key)
}

pub(crate) fn render_launch_summary(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    let mut output = String::new();
    let _ = writeln!(output, "rocm interactive shell");
    let _ = writeln!(output, "  terminal: non-interactive");
    let _ = writeln!(output, "  default engine: {selected_default_engine}");
    let _ = writeln!(
        output,
        "  default runtime: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active runtime key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  config file: {}", paths.config_path().display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(output, "  cache dir: {}", paths.cache_dir.display());
    let _ = writeln!(
        output,
        "  note: launch from an interactive terminal to enter the TUI."
    );
    output
}

pub(crate) fn render_chat_text(paths: &AppPaths, provider: &str) -> Result<String> {
    let status = providers::provider_status(paths, provider)?;
    let mut output = String::new();
    let _ = writeln!(output, "Chat assistant");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Assistant source: {}",
        if status.provider == "local" {
            "local model on this computer"
        } else {
            status.provider.as_str()
        }
    );
    let _ = writeln!(
        output,
        "Status: {}",
        plain_provider_auth_status(&status.auth_status)
    );
    let _ = writeln!(
        output,
        "ROCm help: {}",
        if status.tool_call_schema.is_empty() {
            "not available"
        } else {
            "available"
        }
    );
    if status.models.is_empty() {
        let _ = writeln!(output, "Models: none found yet");
    } else {
        let _ = writeln!(output, "Models:");
        for model in status.models {
            let _ = writeln!(output, "  - {model}");
        }
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Choose Chat source to switch between local and remote assistants."
    );
    let _ = writeln!(output, "Choose Settings to manage saved keys and defaults.");
    Ok(output)
}

fn plain_provider_auth_status(status: &str) -> String {
    if status == "ready" {
        "Ready".to_owned()
    } else if status == "no_ready_local_service" {
        "No local model server is ready".to_owned()
    } else if status.starts_with("disabled:") {
        "Disabled until you enable this provider".to_owned()
    } else {
        status.replace('_', " ")
    }
}

pub(crate) fn render_chat_prompt_text(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
) -> Result<String> {
    Ok(render_chat_prompt_result(paths, provider, model, prompt, rocm_tools)?.rendered)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatToolApprovalRequest {
    pub pending_title: String,
    pub command_title: String,
    pub args: Vec<String>,
    pub display_command: Option<String>,
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatPromptResult {
    pub rendered: String,
    pub approval: Option<ChatToolApprovalRequest>,
}

struct ChatToolRunResult {
    approval: Option<ChatToolApprovalRequest>,
    follow_up_text: String,
    ran_read_only_tool: bool,
    read_only_tool_error: bool,
    needs_install_folder: bool,
}

pub(crate) fn render_chat_prompt_result(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
) -> Result<ChatPromptResult> {
    render_chat_prompt_result_with_progress(paths, provider, model, prompt, rocm_tools, None)
}

pub(crate) fn render_chat_prompt_result_with_progress(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
    progress: Option<&mut dyn FnMut(String)>,
) -> Result<ChatPromptResult> {
    let mut progress = progress;
    let user_prompt = latest_user_chat_message(prompt);
    let assistant_model = local_rocm_tools_assistant_model(provider, rocm_tools).or(model);
    let service_needed_model = if local_rocm_tools_assistant_model(provider, rocm_tools).is_some() {
        None
    } else {
        model
    };
    if rocm_tools && let Some(approval) = install_sdk_without_prefix_chat_approval(user_prompt) {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        report_chat_tool_progress(&mut progress, "Review needed: Install ROCm");
        report_chat_tool_progress(&mut progress, "Waiting for the ROCm install folder.");
        return Ok(ChatPromptResult {
            rendered: render_install_sdk_folder_needed_chat_text(user_prompt, &approval.args),
            approval: Some(approval),
        });
    }
    let mut messages = Vec::new();
    if rocm_tools {
        messages.push(providers::ChatMessage {
            role: "system".to_owned(),
            content: rocm_chat_tool_system_prompt(),
        });
    }
    messages.push(providers::ChatMessage {
        role: "user".to_owned(),
        content: prompt.to_owned(),
    });
    let mut response = if rocm_tools {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        if let Some(call) = deterministic_mutating_tool_call_for_prompt(user_prompt)? {
            report_chat_tool_progress(&mut progress, "Preparing a review card.");
            providers::ChatResponse {
                provider: provider.to_owned(),
                model: assistant_model.unwrap_or("local").to_owned(),
                content: deterministic_mutating_tool_intro(&call),
                tool_calls: vec![call],
            }
        } else if !local_assistant_service_ready_for_chat(paths, assistant_model)
            && let Some(response) =
                local_read_only_fallback_response_for_prompt(provider, assistant_model, user_prompt)
        {
            report_chat_tool_progress(&mut progress, "Running ROCm status check.");
            response
        } else {
            match providers::provider_chat(
                paths,
                provider,
                &providers::ChatRequest {
                    model: assistant_model.map(str::to_owned),
                    messages: messages.clone(),
                    max_tokens: None,
                    rocm_tools,
                },
            ) {
                Ok(response) => response,
                Err(error)
                    if provider == "local" && local_provider_missing_service_error(&error) =>
                {
                    if let Some(response) = local_read_only_fallback_response_for_prompt(
                        provider,
                        assistant_model,
                        user_prompt,
                    ) {
                        report_chat_tool_progress(&mut progress, "Running ROCm status check.");
                        response
                    } else {
                        return Ok(ChatPromptResult {
                            rendered: local_chat_service_needed_text(
                                service_needed_model,
                                user_prompt,
                                rocm_tools,
                            ),
                            approval: None,
                        });
                    }
                }
                Err(error) => return Err(error),
            }
        }
    } else {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        match providers::provider_chat(
            paths,
            provider,
            &providers::ChatRequest {
                model: assistant_model.map(str::to_owned),
                messages: messages.clone(),
                max_tokens: None,
                rocm_tools,
            },
        ) {
            Ok(response) => response,
            Err(error) if provider == "local" && local_provider_missing_service_error(&error) => {
                return Ok(ChatPromptResult {
                    rendered: local_chat_service_needed_text(
                        service_needed_model,
                        user_prompt,
                        rocm_tools,
                    ),
                    approval: None,
                });
            }
            Err(error) => return Err(error),
        }
    };
    let fallback_tool_call = if rocm_tools && response.tool_calls.is_empty() {
        fallback_rocm_tool_call_for_prompt(user_prompt)
    } else {
        None
    };
    let fallback_tool_call_used = fallback_tool_call.is_some();
    if let Some(call) = fallback_tool_call {
        response.tool_calls.push(call);
    }
    if rocm_tools
        && let Some(call) =
            supplemental_read_only_tool_call_for_prompt(user_prompt, &response.tool_calls)
    {
        response.tool_calls.push(call);
    }
    let mut output = String::new();
    let _ = writeln!(output, "chat response");
    let _ = writeln!(output, "  provider: {}", response.provider);
    let _ = writeln!(output, "  model: {}", response.model);
    let _ = writeln!(
        output,
        "  rocm tools: {}",
        if rocm_tools { "enabled" } else { "off" }
    );
    let _ = writeln!(output);
    let initial_content_is_intermediate = fallback_tool_call_used
        || local_tool_call_content_is_intermediate(provider, rocm_tools, &response);
    let initial_content = visible_chat_content(&response.content);
    if initial_content_is_intermediate {
        let _ = writeln!(output, "Assistant is preparing the next step.");
    } else if !initial_content.trim().is_empty() {
        let _ = writeln!(output, "{initial_content}");
    }
    let tool_result = if rocm_tools {
        let explanation = (!fallback_tool_call_used && !initial_content.trim().is_empty())
            .then_some(initial_content.as_str());
        append_chat_tool_results(paths, &response, &mut output, explanation, &mut progress)?
    } else {
        ChatToolRunResult {
            approval: None,
            follow_up_text: String::new(),
            ran_read_only_tool: false,
            read_only_tool_error: false,
            needs_install_folder: false,
        }
    };
    let deterministic_summary = deterministic_chat_tool_summary(&tool_result.follow_up_text);
    if let Some(summary) = deterministic_summary.as_deref() {
        let _ = writeln!(output);
        let _ = writeln!(output, "ROCm CLI summary");
        let _ = writeln!(output, "{summary}");
    }
    let follow_up_blocking_summary = deterministic_summary
        .as_deref()
        .filter(|_| deterministic_summary_can_stand_alone(&tool_result.follow_up_text));
    if should_request_local_tool_follow_up(provider, &tool_result, follow_up_blocking_summary) {
        let follow_up_messages = vec![
            providers::ChatMessage {
                role: "system".to_owned(),
                content: "You are ROCm CLI's local assistant. ROCm tools have already been checked for this turn. Use the supplied tool results for local facts about this machine. For service results, ready/running means running, starting/recovering means starting, failed/stopped means not running, and no matching service row means ROCm CLI is not managing that service as running. If the tool results do not contain enough information to answer the whole question, say what is known from the results and then answer the rest normally from your own model knowledge. Do not request another tool call. Do not invent local paths, GPU names, versions, or install state that are not shown in the tool results. Keep the answer concise.".to_owned(),
            },
            providers::ChatMessage {
                role: "user".to_owned(),
                content: format!(
                    "Original request:\n{}\n\nROCm tool results:\n{}\n\nAnswer the original request. Use the ROCm tool results for this machine's facts; if they are incomplete, answer the remaining part generally.",
                    user_prompt.trim(),
                    tool_result.follow_up_text.trim()
                ),
            },
        ];
        let follow_up = providers::provider_chat(
            paths,
            provider,
            &providers::ChatRequest {
                model: assistant_model.map(str::to_owned),
                messages: follow_up_messages,
                max_tokens: Some(256),
                rocm_tools: false,
            },
        )?;
        if follow_up.tool_calls.is_empty() {
            let follow_up_content = visible_chat_content(&follow_up.content);
            if local_follow_up_content_is_final(&follow_up) && !follow_up_content.trim().is_empty()
            {
                let _ = writeln!(output);
                let _ = writeln!(output, "Assistant after ROCm checks");
                let _ = writeln!(output, "{}", follow_up_content.trim());
            }
        } else {
            let _ = writeln!(output);
            let _ = writeln!(
                output,
                "Assistant asked for another ROCm check after the first tool results."
            );
            let _ = writeln!(
                output,
                "rocm-cli stopped there so the answer is not based on another guess. Nothing was changed."
            );
        }
    } else if tool_result.read_only_tool_error {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "Assistant did not answer from this result because a ROCm check reported an error."
        );
        let _ = writeln!(output, "Nothing was changed.");
    } else if tool_result.needs_install_folder {
        let _ = writeln!(output);
        let _ = writeln!(output, "Assistant after ROCm checks");
        let _ = writeln!(
            output,
            "I can install ROCm/TheRock, but I need the install folder first. Type the folder you want to use, for example D:\\ROCm\\therock_venvs. Nothing will download until ROCm CLI shows a review card and you approve it."
        );
    }
    if tool_result.approval.is_none()
        && tool_result.ran_read_only_tool
        && !output.contains("Nothing was changed.")
    {
        let _ = writeln!(output);
        let _ = writeln!(output, "Nothing was changed.");
    }
    Ok(ChatPromptResult {
        rendered: output,
        approval: tool_result.approval,
    })
}

fn report_chat_tool_progress(progress: &mut Option<&mut dyn FnMut(String)>, message: &str) {
    if let Some(progress) = progress.as_deref_mut() {
        progress(message.to_owned());
    }
}

fn latest_user_chat_message(prompt: &str) -> &str {
    const MARKER: &str = "\nNew message:\n";
    prompt
        .rfind(MARKER)
        .map_or(prompt, |index| &prompt[index + MARKER.len()..])
        .trim()
}

fn deterministic_mutating_tool_call_for_prompt(
    prompt: &str,
) -> Result<Option<providers::ChatToolCall>> {
    let Some(call) = fallback_rocm_tool_call_for_prompt(prompt) else {
        return Ok(None);
    };
    validate_chat_tool_call(&call)?;
    if chat_tool_call_is_read_only(&call) {
        Ok(None)
    } else {
        Ok(Some(call))
    }
}

fn supplemental_read_only_tool_call_for_prompt(
    prompt: &str,
    existing_calls: &[providers::ChatToolCall],
) -> Option<providers::ChatToolCall> {
    let normalized = latest_user_chat_message(prompt).to_ascii_lowercase();
    if prompt_mentions_serving_engine_or_service(&normalized)
        && prompt_asks_running_or_status(&normalized)
        && prompt_asks_engine_install_state(&normalized)
    {
        for call in [
            fallback_services_list_tool_call(),
            fallback_engine_list_tool_call(),
        ] {
            if !chat_tool_calls_include_equivalent(existing_calls, &call) {
                return Some(call);
            }
        }
        return None;
    }
    let call = fallback_rocm_tool_call_for_prompt(prompt)?;
    if !chat_tool_call_is_read_only(&call)
        || chat_tool_calls_include_equivalent(existing_calls, &call)
    {
        return None;
    }
    Some(call)
}

fn chat_tool_calls_include_equivalent(
    existing_calls: &[providers::ChatToolCall],
    required: &providers::ChatToolCall,
) -> bool {
    existing_calls.iter().any(|existing| {
        if existing.name != required.name {
            return false;
        }
        if required.name == "rocm_command" {
            return normalized_chat_rocm_command_args(existing).ok()
                == normalized_chat_rocm_command_args(required).ok();
        }
        if required.name == "port_status" {
            return chat_port_status_tool_calls_equivalent(existing, required);
        }
        existing.arguments == required.arguments
    })
}

fn chat_port_status_tool_calls_equivalent(
    left: &providers::ChatToolCall,
    right: &providers::ChatToolCall,
) -> bool {
    let Some(left_object) = left.arguments.as_object() else {
        return false;
    };
    let Some(right_object) = right.arguments.as_object() else {
        return false;
    };
    let left_port = left_object.get("port").and_then(serde_json::Value::as_u64);
    let right_port = right_object.get("port").and_then(serde_json::Value::as_u64);
    if left_port != right_port {
        return false;
    }
    let left_host =
        json_string(left_object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    let right_host =
        json_string(right_object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    loopback_host_key(&left_host) == loopback_host_key(&right_host)
}

fn loopback_host_key(host: &str) -> String {
    match host.trim().to_ascii_lowercase().as_str() {
        "localhost" | "127.0.0.1" => "127.0.0.1".to_owned(),
        "::1" | "[::1]" => "::1".to_owned(),
        other => other.to_owned(),
    }
}

fn deterministic_mutating_tool_intro(call: &providers::ChatToolCall) -> String {
    match chat_tool_approval_request(call, None) {
        Ok(approval) if approval.pending_title == "Install ROCm" => {
            "I can install ROCm/TheRock into the folder you chose. Review the card before anything downloads or changes."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Install ComfyUI" => {
            "I can install ComfyUI into ROCm CLI's managed app folder. Review the card before anything downloads or changes."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Start ComfyUI" => {
            "I can start ComfyUI for you. Review the card before ROCm CLI launches it."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Start local model server" => {
            "I can start the recommended local model server on the GPU. Review the card before ROCm CLI launches it."
                .to_owned()
        }
        Ok(approval) => format!(
            "I can prepare this ROCm change: {}. Review the card before anything runs.",
            approval.pending_title
        ),
        Err(_) => {
            "I can prepare this ROCm change. Review the card before anything runs.".to_owned()
        }
    }
}

fn local_tool_call_content_is_intermediate(
    provider: &str,
    rocm_tools: bool,
    response: &providers::ChatResponse,
) -> bool {
    provider == "local"
        && rocm_tools
        && !response.tool_calls.is_empty()
        && !response.content.trim().is_empty()
}

fn local_follow_up_content_is_final(response: &providers::ChatResponse) -> bool {
    response.tool_calls.is_empty() && !response.content.trim().is_empty()
}

fn local_read_only_fallback_response_for_prompt(
    provider: &str,
    assistant_model: Option<&str>,
    prompt: &str,
) -> Option<providers::ChatResponse> {
    if provider != "local" || !prompt_can_use_read_only_without_local_assistant(prompt) {
        return None;
    }
    let call = fallback_rocm_tool_call_for_prompt(prompt)?;
    if !chat_tool_call_is_read_only(&call) {
        return None;
    }
    Some(providers::ChatResponse {
        provider: provider.to_owned(),
        model: assistant_model.unwrap_or("local").to_owned(),
        content: String::new(),
        tool_calls: vec![call],
    })
}

fn local_assistant_service_ready_for_chat(paths: &AppPaths, assistant_model: Option<&str>) -> bool {
    let model = assistant_model.unwrap_or(providers::BUILTIN_ASSISTANT_MODEL_ID);
    load_managed_services(paths).is_ok_and(|records| {
        records.iter().any(|record| {
            matches!(record.status.as_str(), "ready" | "running")
                && (service_model_names_match(&record.canonical_model_id, model)
                    || service_model_names_match(&record.model_ref, model))
        })
    })
}

fn prompt_can_use_read_only_without_local_assistant(prompt: &str) -> bool {
    let normalized = latest_user_chat_message(prompt).to_ascii_lowercase();
    let mentions_status_subject = any_substring(
        &normalized,
        &[
            "comfyui",
            "comfy ui",
            "comfy",
            "vllm",
            "sglang",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "pytorch",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
            "port",
            "8188",
            "therock",
            "rocm",
        ],
    );
    mentions_status_subject
        && (prompt_asks_running_or_status(&normalized)
            || any_substring(
                &normalized,
                &["installed", "available", "detected", "engine status"],
            ))
}

fn prompt_mentions_serving_engine_or_service(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &[
            "vllm",
            "sglang",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "pytorch",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
        ],
    )
}

fn prompt_asks_engine_install_state(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &["installed", "available", "detected", "engine status"],
    )
}

fn fallback_engine_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-engine-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["engines", "list"] }),
    }
}

fn fallback_services_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-services-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["services", "list", "--all"] }),
    }
}

fn fallback_rocm_tool_call_for_prompt(prompt: &str) -> Option<providers::ChatToolCall> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    let asks_running_or_status = prompt_asks_running_or_status(&normalized);
    let mentions_comfyui = any_substring(&normalized, &["comfyui", "comfy ui", "comfy"]);
    if asks_running_or_status
        && !mentions_comfyui
        && any_substring(&normalized, &["8188", "port 8188"])
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-port-8188-status".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 }),
        });
    }
    let mentions_serving_engine_or_service = prompt_mentions_serving_engine_or_service(&normalized);
    if mentions_serving_engine_or_service && prompt_asks_engine_install_state(&normalized) {
        return Some(fallback_engine_list_tool_call());
    }
    if mentions_serving_engine_or_service && asks_running_or_status {
        return Some(fallback_services_list_tool_call());
    }
    let mentions_llm_or_model = any_substring(
        &normalized,
        &["llm", "llms", "model", "models", "assistant"],
    );
    let asks_support_or_fit = any_substring(
        &normalized,
        &[
            "support",
            "supported",
            "run",
            "runs",
            "fit",
            "fits",
            "can my machine",
            "can this machine",
        ],
    );
    if mentions_llm_or_model && asks_support_or_fit {
        return Some(providers::ChatToolCall {
            id: Some("fallback-rocm-model".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["model"] }),
        });
    }

    if mentions_comfyui {
        if any_substring(&normalized, &["log", "logs"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-logs".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "logs"] }),
            });
        }
        if asks_running_or_status {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-status".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
            });
        }
        if any_substring(&normalized, &["start", "run", "launch", "open"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-start".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "start"],
                    "reason": "Start ComfyUI locally after the user approves it."
                }),
            });
        }
        if any_substring(
            &normalized,
            &[
                "can you setup",
                "can you set up",
                "please setup",
                "please set up",
                "setup comfyui for me",
                "set up comfyui for me",
                "install comfyui",
                "download comfyui",
            ],
        ) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-install".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "install"],
                    "reason": "Install ComfyUI into ROCm CLI's managed app folder after the user approves it."
                }),
            });
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-comfyui-status".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
        });
    }

    if let Some(call) = fallback_config_tool_call_for_prompt(&normalized) {
        return Some(call);
    }

    if mentions_llm_or_model
        && any_substring(
            &normalized,
            &[
                "serve",
                "server",
                "start",
                "setup and serve",
                "set up and serve",
                "run locally",
                "local model",
                "local assistant",
            ],
        )
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-serve-qwen".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "lemonade", "--device", "gpu_required", "--managed"],
                "reason": "Start the recommended local assistant after the user approves it."
            }),
        });
    }

    let mentions_setup = any_substring(&normalized, &["setup", "set up", "install"]);
    let asks_how = any_substring(&normalized, &["how", "help", "what do i need"]);
    let mentions_rocm_or_therock = any_substring(&normalized, &["therock", "rocm"]);
    let requested_install_prefix = requested_install_prefix_from_prompt(prompt);
    if requested_install_prefix.is_none()
        && let Some(approval) = install_sdk_without_prefix_chat_approval(prompt)
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install-folder".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": approval.args,
                "reason": "Ask the user to choose the ROCm/TheRock install folder before installing."
            }),
        });
    }
    if mentions_rocm_or_therock
        && prompt_requests_install_action(&normalized)
        && let Some(prefix) = requested_install_prefix
    {
        let mut args = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            "release".to_owned(),
            "--format".to_owned(),
            "wheel".to_owned(),
            "--prefix".to_owned(),
            prefix,
        ];
        if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
            args.push("--build-date".to_owned());
            args.push(build_date);
        } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
            args.push("--version".to_owned());
            args.push(version);
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": args,
                "reason": "Install TheRock ROCm into the user-selected folder after the user approves it."
            }),
        });
    }
    if asks_how && mentions_setup && any_substring(&normalized, &["therock", "rocm"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    let asks_where_installed = any_substring(
        &normalized,
        &[
            "where is rocm",
            "where's rocm",
            "where is therock",
            "where's therock",
            "where did rocm",
            "where did therock",
            "where rocm is installed",
            "where therock is installed",
            "rocm install folder",
            "therock install folder",
            "rocm installed at",
            "therock installed at",
        ],
    );
    let asks_status = asks_where_installed
        || any_substring(
            &normalized,
            &[
                "is rocm installed",
                "is therock installed",
                "is therock setup",
                "is therock set up",
                "rocm installed",
                "therock installed",
                "check this rocm setup",
                "which gpu",
                "what gpu",
                "gpu is on",
                "gpu do i have",
                "my machine",
            ],
        );
    if asks_status && any_substring(&normalized, &["gpu", "rocm", "therock", "setup"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    None
}

fn prompt_asks_running_or_status(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "running",
            "is it up",
            "is this up",
            "is there",
            "are there",
            "status",
            "started",
            "listening",
            "on port",
            "port ",
        ],
    )
}

fn fallback_config_tool_call_for_prompt(normalized: &str) -> Option<providers::ChatToolCall> {
    if !any_substring(
        normalized,
        &[
            "config",
            "setting",
            "settings",
            "default engine",
            "telemetry",
        ],
    ) {
        return None;
    }
    if any_substring(normalized, &["show", "check", "what", "current", "list"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-config-show".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["config", "show"] }),
        });
    }
    if any_substring(normalized, &["default engine", "set engine", "use engine"]) {
        for engine in ["pytorch", "llama.cpp", "lemonade", "vllm", "sglang", "atom"] {
            if normalized.contains(engine) {
                return Some(providers::ChatToolCall {
                    id: Some("fallback-config-default-engine".to_owned()),
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({
                        "args": ["config", "set-default-engine", engine],
                        "reason": "Change ROCm CLI's default engine after the user approves it."
                    }),
                });
            }
        }
    }
    if normalized.contains("telemetry") {
        if any_substring(normalized, &["off", "disable", "disabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-off".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "off"],
                    "reason": "Turn ROCm CLI telemetry off after the user approves it."
                }),
            });
        }
        if any_substring(normalized, &["local", "on", "enable", "enabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-local".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "local"],
                    "reason": "Enable local-only ROCm inspection after the user approves it."
                }),
            });
        }
    }
    None
}

fn any_substring(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

pub(crate) fn install_sdk_without_prefix_chat_approval(
    prompt: &str,
) -> Option<ChatToolApprovalRequest> {
    install_sdk_chat_approval_for_prompt(prompt).and_then(|approval| {
        chat_cli_arg_value(&approval.args, "--prefix")
            .is_none()
            .then_some(approval)
    })
}

pub(crate) fn install_sdk_chat_approval_for_prompt(
    prompt: &str,
) -> Option<ChatToolApprovalRequest> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    if !prompt_requests_rocm_install_or_setup(&normalized) {
        return None;
    }
    let mut args = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        if normalized.contains("nightly") {
            "nightly".to_owned()
        } else {
            "release".to_owned()
        },
        "--format".to_owned(),
        "wheel".to_owned(),
    ];
    if let Some(prefix) = requested_install_prefix_from_prompt(prompt) {
        args.push("--prefix".to_owned());
        args.push(prefix);
    }
    if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
        args.push("--build-date".to_owned());
        args.push(build_date);
    } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
        args.push("--version".to_owned());
        args.push(version);
    }
    Some(ChatToolApprovalRequest {
        pending_title: "Install ROCm".to_owned(),
        command_title: "Install".to_owned(),
        display_command: Some(format_structured_tool_call("rocm", &args)),
        args,
        explanation: Some(
            "Install ROCm/TheRock after the user chooses the install folder.".to_owned(),
        ),
    })
}

fn render_install_sdk_folder_needed_chat_text(prompt: &str, args: &[String]) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "ROCm install");
    let _ = writeln!(output);
    let _ = writeln!(output, "I can install ROCm/TheRock for you.");
    let _ = writeln!(
        output,
        "First choose the folder where ROCm CLI should put the Python environment."
    );
    let _ = writeln!(
        output,
        "Nothing will download or change until you review and approve the install."
    );
    if let Some(date) = chat_cli_arg_value(args, "--build-date") {
        let _ = writeln!(output);
        let _ = writeln!(output, "Requested build date: {date}");
    }
    if let Some(version) = chat_cli_arg_value(args, "--version") {
        let _ = writeln!(output);
        let _ = writeln!(output, "Requested version: {version}");
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Your request");
    let _ = writeln!(output, "  {}", prompt.trim());
    output.trim_end().to_owned()
}

pub(crate) fn chat_install_folder_from_prompt(prompt: &str) -> Option<String> {
    requested_install_prefix_from_prompt(prompt).or_else(|| clean_requested_install_prefix(prompt))
}

fn prompt_requests_rocm_install_or_setup(normalized: &str) -> bool {
    let mentions_rocm_stack = any_substring(
        normalized,
        &[
            "rocm",
            "therock",
            "the rock",
            "amd gpu package",
            "amd gpu ready",
            "local ai",
        ],
    );
    if !mentions_rocm_stack {
        return false;
    }
    prompt_requests_install_action(normalized)
        || any_substring(
            normalized,
            &[
                "need rocm",
                "need therock",
                "need the rock",
                "need to get rocm",
                "need to get therock",
                "get rocm installed",
                "get therock installed",
                "get the rock installed",
                "make rocm work",
                "make therock work",
                "make the rock work",
                "prepare rocm",
                "prepare therock",
                "prepare the rock",
                "set up my amd gpu",
                "setup my amd gpu",
                "set up local ai",
                "setup local ai",
                "get local ai working",
                "make local ai work",
                "rocm please",
                "make my amd gpu ready",
                "make my gpu ready",
                "get my amd gpu ready",
                "get my gpu ready",
            ],
        )
}

fn prompt_requests_install_action(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "install this",
            "install specific",
            "install the",
            "install rocm",
            "install therock",
            "install the rock",
            "install amd gpu package",
            "install local ai",
            "can you install",
            "please install",
            "rocm please",
            "therock please",
            "the rock please",
            "need to install",
            "want to install",
            "i need rocm",
            "i need therock",
            "i need the rock",
            "get me rocm",
            "get me therock",
            "get my gpu ready",
            "make my gpu ready",
            "make my amd gpu ready",
            "get installed",
            "setup for me",
            "set up for me",
            "setup rocm",
            "set up rocm",
            "setup therock",
            "set up therock",
            "setup the rock",
            "set up the rock",
            "setup my gpu",
            "set up my gpu",
            "setup amd gpu",
            "set up amd gpu",
        ],
    )
}

fn requested_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    let lower = prompt.to_ascii_lowercase();
    for phrase in [
        "--prefix=",
        "--prefix ",
        " into folder ",
        " into ",
        " in folder ",
        " in ",
        " to folder ",
        " to ",
        " under ",
        " at ",
        " use folder ",
        " use ",
        "use ",
        " folder is ",
        " folder: ",
        "folder:",
    ] {
        let mut search_start = 0;
        while let Some(relative_index) = lower[search_start..].find(phrase) {
            let value_start = search_start + relative_index + phrase.len();
            if let Some(prefix) = clean_requested_install_prefix(&prompt[value_start..]) {
                return Some(prefix);
            }
            search_start = value_start;
        }
    }
    requested_bare_install_prefix_from_prompt(prompt)
}

fn requested_bare_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let mut remaining = prompt;
        while let Some(start) = remaining.find(quote) {
            let after_start = &remaining[start + quote.len_utf8()..];
            let Some(end) = after_start.find(quote) else {
                break;
            };
            if let Some(prefix) = clean_requested_install_prefix(&after_start[..end]) {
                return Some(prefix);
            }
            remaining = &after_start[end + quote.len_utf8()..];
        }
    }
    prompt.split_whitespace().find_map(|token| {
        clean_requested_install_prefix(
            token.trim_matches(|ch: char| {
                matches!(ch, '"' | '\'' | '`' | ',' | ';' | '.' | ')' | ']')
            }),
        )
    })
}

fn clean_requested_install_prefix(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars();
    let first = chars.next()?;
    let extracted = if first == '"' || first == '\'' {
        let close = trimmed[first.len_utf8()..].find(first)?;
        &trimmed[first.len_utf8()..first.len_utf8().saturating_add(close)]
    } else {
        trimmed
            .split(['\r', '\n', ',', ';'])
            .next()
            .unwrap_or_default()
            .split(" and ")
            .next()
            .unwrap_or_default()
            .split(" with ")
            .next()
            .unwrap_or_default()
            .split(" then ")
            .next()
            .unwrap_or_default()
            .trim_end_matches(['.', ')', ']'])
    };
    let prefix = extracted.trim();
    if prefix.is_empty() || !looks_like_user_path(prefix) {
        return None;
    }
    Some(prefix.to_owned())
}

fn looks_like_user_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with('~')
        || value.starts_with("\\\\")
        || value.contains(":\\")
        || value.contains(":/")
        || value.contains('\\')
}

fn requested_therock_build_date_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| token.len() == 8 && token.chars().all(|ch| ch.is_ascii_digit()))
        .find_map(
            |token| match therock::RuntimeVersionSelector::build_date(token).ok()? {
                therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                therock::RuntimeVersionSelector::Version(_) => None,
            },
        )
        .or_else(|| {
            normalized
                .split_whitespace()
                .map(|token| {
                    token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
                })
                .find_map(|token| {
                    match therock::RuntimeVersionSelector::build_date(token).ok()? {
                        therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                        therock::RuntimeVersionSelector::Version(_) => None,
                    }
                })
        })
}

fn requested_therock_version_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '+' | '-' | '_')))
        .filter(|token| looks_like_therock_runtime_version(token))
        .find_map(|token| {
            therock::RuntimeVersionSelector::version(token)
                .ok()
                .map(|_| token.to_owned())
        })
}

fn looks_like_therock_runtime_version(token: &str) -> bool {
    let Some((prefix, date)) = token.rsplit_once('a') else {
        return false;
    };
    !prefix.is_empty()
        && prefix.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        && prefix.contains('.')
        && date.len() == 8
        && date.chars().all(|ch| ch.is_ascii_digit())
        && therock::RuntimeVersionSelector::build_date(date).is_ok()
}

fn visible_chat_content(content: &str) -> String {
    let mut output = String::new();
    let mut remaining = content;
    while let Some(start) = find_ascii_case_insensitive(remaining, "<think>") {
        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];
        let Some(end) = find_ascii_case_insensitive(after_start, "</think>") else {
            remaining = "";
            break;
        };
        remaining = &after_start[end + "</think>".len()..];
    }
    output.push_str(remaining);
    output.trim().to_owned()
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

const ROCM_CHAT_TOOL_SYSTEM_PROMPT: &str = "You are ROCm CLI's local assistant. Speak in simple English for non-technical Windows users. Use the provided ROCm tools when you need to inspect this machine, preview setup, read service logs, check updates, inspect automations, install or start ROCm-managed apps, or request ROCm/TheRock, config, engine, app, and local model server changes. For simple greetings or thanks like hello, hi, hey, ok, or thank you, reply normally; do not inspect ROCm, do not call tools, and do not launch or propose a model server. Tool-use rules: inspect first with read-only tools; call rocm_command only with argv-style args and no shell text; use natural_language_plan for ROCm requests that do not fit another read-only tool; ask for a mutating tool call only after explaining why it is needed; summarize tool results after they are returned. Read-only tools may run immediately. Tools that install, launch, stop, delete, or change state require user approval; request rocm_command and explain why. For 'is X running?', 'what is running?', status, or port questions, inspect before answering and do not start, stop, install, or serve anything. For ComfyUI or port 8188 use [\"comfyui\",\"status\"] or port_status. For vLLM, SGLang, Lemonade, PyTorch, llama.cpp, qwen, or local model servers use [\"services\",\"list\",\"--all\"] for running state and [\"engines\",\"list\"] for installed/available engine state. Treat ready/running as running, starting/recovering as starting, failed/stopped as not running, and no matching record as unknown or not managed by ROCm CLI. Interpret Examine carefully: active_runtime_status=ready means ROCm CLI has an active managed TheRock/ROCm runtime; legacy_rocm_status=not_detected only means no global system ROCm install was found. If active_runtime_status=ready, tell the user ROCm/TheRock is installed and active for ROCm CLI. For 'is TheRock installed', 'is ROCm installed', or 'which GPU is on this machine', use examine or gpu_snapshot before answering. For 'how do I setup TheRock' or install/setup requests, guide the user to choose an install folder first; do not answer with only a status check. For 'which LLMs can this machine support', use rocm_command args [\"model\"] or natural_language_plan before answering. For TheRock installs, always let the user choose the install folder. If the user names a folder or prefix, preserve that exact folder with [\"--prefix\",\"PATH\"]; you may call path_exists first to check whether that user-provided folder or its parent exists. If the user asks you to install TheRock/ROCm but has not named a folder, ask for the folder or let the guided setup folder picker collect it; do not invent a hidden default folder and do not request an install command without --prefix. Use rocm_command args [\"install\",\"sdk\",\"--channel\",\"release\",\"--format\",\"wheel\",\"--prefix\",\"PATH\"] only when the user asks you to install it and a folder is known; for a requested build date add [\"--build-date\",\"YYYY-MM-DD\"] and for a requested exact version add [\"--version\",\"VERSION\"]. For config changes, inspect with [\"config\",\"show\"] first when useful, then request config subcommands such as [\"config\",\"set-default-engine\",\"lemonade\"], [\"config\",\"set-default-runtime\",\"RUNTIME_KEY\"], or [\"config\",\"set-telemetry\",\"local\"] only after explaining why. For ComfyUI, use rocm_command with args like [\"comfyui\",\"status\"], [\"comfyui\",\"logs\"], [\"comfyui\",\"install\"], [\"comfyui\",\"start\"], or [\"comfyui\",\"stop\"]. First-time setup is the same thing as bootstrap in ROCm CLI; it is a deterministic ROCm setup flow, not a separate model chat. The built-in local assistant is fixed to qwen, which maps to Qwen3-4B-Instruct-2507-GGUF served by Lemonade with gpu_required. vLLM, SGLang, PyTorch, and Lemonade are general serving engines; inspect or manage them when the user asks about general model serving, but do not switch the built-in assistant away from Lemonade. Use qwen-smoke only for a quick server smoke test. For llama.cpp, use the llama.cpp engine backed by upstream llama-server: request rocm_command args like [\"engines\",\"install\",\"llama.cpp\"] or [\"serve\",\"MODEL.gguf\",\"--engine\",\"llama.cpp\",\"--device\",\"gpu_required\",\"--managed\"]. On native Windows, vLLM and SGLang are skipped; use WSL/Linux for those ROCm GPU engines. For vLLM management, inspect engines first and use [\"engines\",\"install\",\"vllm\"] or [\"serve\",\"MODEL\",\"--engine\",\"vllm\",\"--device\",\"gpu_required\",\"--managed\"] only where the host supports it. Do not invent shell commands and do not request CPU fallback.";
const ROCM_CHAT_TOOL_SKILL: &str = include_str!("../../../skills/rocm-cli-assistant/SKILL.md");

fn rocm_chat_tool_system_prompt() -> String {
    format!("{ROCM_CHAT_TOOL_SYSTEM_PROMPT}\n\nROCm CLI assistant skill:\n{ROCM_CHAT_TOOL_SKILL}")
}

fn local_provider_missing_service_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("local provider has no ready managed service")
    })
}

fn local_rocm_tools_assistant_model(provider: &str, rocm_tools: bool) -> Option<&'static str> {
    (provider == "local" && rocm_tools).then_some(providers::BUILTIN_ASSISTANT_MODEL_ID)
}

pub(crate) fn local_chat_service_needed_text(
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "No local assistant is running yet.");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "To use LLM-assisted ROCm commands, start a ROCm GPU local model server first."
    );
    let _ = writeln!(
        output,
        "First-time ROCm setup does not need an LLM; run `rocm` and use Set Up ROCm for that."
    );
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Recommended path:\n  run `rocm`, choose Start a local model, use the recommended assistant model, then start it"
    );
    let _ = writeln!(output);
    if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
        let args = vec![
            "serve".to_owned(),
            model.to_owned(),
            "--device".to_owned(),
            "gpu_required".to_owned(),
            "--managed".to_owned(),
        ];
        let _ = writeln!(
            output,
            "Advanced manual command for the selected model:\n  {}",
            format_structured_tool_call("rocm", &args)
        );
    } else {
        let example_args = vec![
            "serve".to_owned(),
            providers::LEMONADE_ASSISTANT_MODEL_ID.to_owned(),
            "--engine".to_owned(),
            "lemonade".to_owned(),
            "--device".to_owned(),
            "gpu_required".to_owned(),
            "--managed".to_owned(),
        ];
        let _ = writeln!(
            output,
            "Advanced manual command for the recommended assistant model:\n  {}",
            format_structured_tool_call("rocm", &example_args)
        );
    }
    let mut chat_args = vec!["chat".to_owned()];
    if rocm_tools {
        chat_args.push("--tools".to_owned());
    }
    chat_args.push("--provider".to_owned());
    chat_args.push("local".to_owned());
    if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
        chat_args.push("--model".to_owned());
        chat_args.push(model.to_owned());
    }
    if !prompt.trim().is_empty() {
        chat_args.push("--prompt".to_owned());
        chat_args.push(prompt.to_owned());
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "After the server is ready, run:\n  {}",
        format_structured_tool_call("rocm", &chat_args)
    );
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "For the guided flow, run `rocm`, choose Start a local model, start a model, then choose Chat."
    );
    let _ = writeln!(output, "Nothing was changed.");
    output.trim_end().to_owned()
}

fn append_chat_tool_results(
    paths: &AppPaths,
    response: &providers::ChatResponse,
    output: &mut String,
    assistant_explanation: Option<&str>,
    progress: &mut Option<&mut dyn FnMut(String)>,
) -> Result<ChatToolRunResult> {
    if response.tool_calls.is_empty() {
        let _ = writeln!(output, "ROCm checks used");
        let _ = writeln!(output, "  none requested");
        return Ok(ChatToolRunResult {
            approval: None,
            follow_up_text: String::new(),
            ran_read_only_tool: false,
            read_only_tool_error: false,
            needs_install_folder: false,
        });
    }

    let _ = writeln!(output, "ROCm checks used");
    let mut approval = None;
    let mut follow_up_text = String::new();
    let mut ran_read_only_tool = false;
    let mut read_only_tool_error = false;
    let mut needs_install_folder = false;
    for call in &response.tool_calls {
        if chat_tool_call_requests_therock_install_without_prefix(call) {
            needs_install_folder = true;
            report_chat_tool_progress(progress, "Waiting for the ROCm install folder.");
            let _ = writeln!(output, "  Install ROCm: needs install folder");
            let _ = writeln!(
                output,
                "    not run: choose an install folder before the review card"
            );
            continue;
        }
        validate_chat_tool_call(call)?;
        let label = chat_tool_call_display_label(call);
        if chat_tool_call_is_read_only(call) {
            report_chat_tool_progress(progress, &format!("Running ROCm check: {label}."));
            let result = run_chat_read_only_tool(paths, call)?;
            ran_read_only_tool = true;
            let is_error = mcp_tool_result_is_error(&result);
            read_only_tool_error |= is_error;
            report_chat_tool_progress(
                progress,
                &format!(
                    "ROCm check finished: {label} ({})",
                    chat_read_only_tool_status_label(is_error)
                ),
            );
            let _ = writeln!(
                output,
                "  {}: {}",
                label,
                chat_read_only_tool_status_label(is_error)
            );
            let result_text = mcp_tool_result_text(&result);
            let _ = writeln!(follow_up_text, "{}:", call.name);
            let _ = writeln!(follow_up_text, "{result_text}");
            for line in result_text.lines() {
                let _ = writeln!(output, "    {line}");
            }
        } else {
            report_chat_tool_progress(progress, &format!("Review needed: {label}."));
            let _ = writeln!(output, "  {label}: needs your review");
            let _ = writeln!(
                output,
                "    not run: review the approval card before anything runs"
            );
            if let Some(command) = rocm_chat_tool_requested_command(call) {
                let _ = writeln!(output, "    advanced manual command: {command}");
            }
            if approval.is_none() {
                approval = Some(chat_tool_approval_request(call, assistant_explanation)?);
            }
        }
    }
    Ok(ChatToolRunResult {
        approval,
        follow_up_text,
        ran_read_only_tool,
        read_only_tool_error,
        needs_install_folder,
    })
}

fn chat_tool_call_requests_therock_install_without_prefix(call: &providers::ChatToolCall) -> bool {
    let Some(object) = call.arguments.as_object() else {
        return false;
    };
    match call.name.as_str() {
        "install_sdk" => json_string(object, "prefix").is_none(),
        "rocm_command" => rocm_command_args_install_sdk_without_prefix(object),
        _ => false,
    }
}

fn rocm_command_args_install_sdk_without_prefix(
    object: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let Some(args) = object.get("args").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let args = args
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    args.first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("install"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("sdk"))
        && chat_cli_arg_value(&args, "--prefix").is_none()
}

pub(crate) fn chat_tool_approval_request(
    call: &providers::ChatToolCall,
    assistant_explanation: Option<&str>,
) -> Result<ChatToolApprovalRequest> {
    validate_chat_tool_call(call)?;
    if call.name == "rocm_command" {
        let ChatRocmCommandAction::Approval {
            args,
            pending_title,
            command_title,
        } = chat_rocm_command_action(call)?
        else {
            bail!("ROCm command tool is read-only and does not need approval");
        };
        return Ok(ChatToolApprovalRequest {
            pending_title,
            command_title,
            display_command: Some(format_structured_tool_call("rocm", &args)),
            args,
            explanation: assistant_explanation
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        });
    }
    // `proposal_action` (approve/reject) executes in-process in
    // `run_internal_mcp_call` — it has NO CLI argv, so build its approval
    // request directly rather than through `rocm_chat_tool_requested_args`.
    if call.name == "proposal_action" {
        let object = call
            .arguments
            .as_object()
            .context("proposal_action arguments must be a JSON object")?;
        let proposal_id = json_string(object, "proposal_id")
            .context("proposal_action requires non-empty `proposal_id`")?;
        let action =
            json_string(object, "action").context("proposal_action requires non-empty `action`")?;
        let pending_title = match action.as_str() {
            "approve" => "Approve proposal",
            "reject" => "Reject proposal",
            other => bail!("proposal_action `{other}` does not require approval"),
        };
        return Ok(ChatToolApprovalRequest {
            pending_title: pending_title.to_owned(),
            command_title: "Reviews".to_owned(),
            display_command: Some(format!("proposal {proposal_id}")),
            args: Vec::new(),
            explanation: assistant_explanation
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        });
    }
    let args = rocm_chat_tool_requested_args(call)
        .with_context(|| format!("ROCm tool `{}` is missing required arguments", call.name))?;
    let (pending_title, command_title) = match call.name.as_str() {
        "install_sdk" => ("Install ROCm", "Install"),
        "install_engine" => ("Install engine", "Engine"),
        "launch_server" => ("Start local model server", "Serve"),
        "stop_server" => ("Stop local model server", "Services"),
        "watcher_enable" => ("Enable automation", "Automations"),
        "watcher_disable" => ("Disable automation", "Automations"),
        other => bail!("ROCm tool `{other}` is read-only or unsupported for approval"),
    };
    Ok(ChatToolApprovalRequest {
        pending_title: pending_title.to_owned(),
        command_title: command_title.to_owned(),
        display_command: rocm_chat_tool_requested_command(call),
        args,
        explanation: assistant_explanation
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
    })
}

pub(crate) fn validate_chat_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    if !call.arguments.is_object() {
        bail!("ROCm tool `{}` arguments must be a JSON object", call.name);
    }
    match call.name.as_str() {
        "examine"
        // `doctor` is the dash-side LLM tool + `/doctor` overlay name for the
        // same machine inspection the bin exposes as `examine`. Accept it as an
        // alias so a model-issued `doctor` call resolves end-to-end.
        | "doctor"
        | "bridge_snapshot"
        | "gpu_snapshot"
        | "engines"
        | "services"
        | "service_logs"
        | "automations"
        | "natural_language_plan"
        | "path_exists"
        | "port_status"
        | "rocm_command"
        | "update_check"
        | "install_sdk_dry_run"
        | "install_sdk"
        | "install_engine"
        | "launch_server"
        | "stop_server"
        | "watcher_enable"
        | "watcher_disable"
        | "proposal_action" => {}
        other => bail!("local assistant requested unsupported ROCm tool `{other}`"),
    }
    match call.name.as_str() {
        "install_sdk" => validate_chat_install_sdk_tool_call(call)?,
        "install_engine" => validate_required_chat_string(call, "engine")?,
        "launch_server" => validate_chat_launch_server_tool_call(call)?,
        "service_logs" | "stop_server" => validate_chat_service_tool_call(call)?,
        "automations" => validate_optional_chat_integer(call, "event_limit", 1, 64)?,
        "natural_language_plan" => validate_required_chat_string(call, "request")?,
        "path_exists" => validate_required_chat_string(call, "path")?,
        "port_status" => validate_chat_port_status_tool_call(call)?,
        "rocm_command" => validate_chat_rocm_command_tool_call(call)?,
        "watcher_enable" => validate_chat_watcher_tool_call(call, true)?,
        "watcher_disable" => validate_chat_watcher_tool_call(call, false)?,
        "proposal_action" => validate_chat_proposal_action_tool_call(call)?,
        _ => {}
    }
    Ok(())
}

/// Validate a `proposal_action` chat-tool call: `proposal_id` must be a
/// non-empty string and `action` must be one of show | approve | reject.
fn validate_chat_proposal_action_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("proposal_action arguments must be a JSON object")?;
    let proposal_id = json_string(object, "proposal_id")
        .context("proposal_action requires non-empty `proposal_id`")?;
    if proposal_id.len() > 128 {
        bail!("proposal_id too long");
    }
    let action =
        json_string(object, "action").context("proposal_action requires non-empty `action`")?;
    if !matches!(action.as_str(), "show" | "approve" | "reject") {
        bail!("proposal_action `action` must be one of show, approve, reject");
    }
    Ok(())
}

fn validate_chat_install_sdk_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("install_sdk arguments must be a JSON object")?;
    let channel = json_string(object, "channel").unwrap_or_else(|| "release".to_owned());
    if !matches!(channel.as_str(), "release" | "nightly") {
        bail!("local assistant requested unsupported TheRock channel `{channel}`");
    }
    let format = json_string(object, "format").unwrap_or_else(|| "wheel".to_owned());
    if !matches!(format.as_str(), "wheel" | "tarball") {
        bail!("local assistant requested unsupported TheRock install format `{format}`");
    }
    if rocm_core::runtime_is_windows() && format != "wheel" {
        bail!("local assistant cannot request `{format}` installs on Windows; use wheel");
    }
    let version = json_string(object, "version");
    let build_date = json_string(object, "build_date");
    if version.is_some() && build_date.is_some() {
        bail!("local assistant cannot request both `version` and `build_date`");
    }
    if format != "wheel" && (version.is_some() || build_date.is_some()) {
        bail!(
            "local assistant can only request specific TheRock wheel versions for wheel installs"
        );
    }
    if let Some(version) = version {
        therock::RuntimeVersionSelector::version(version)?;
    }
    if let Some(build_date) = build_date {
        therock::RuntimeVersionSelector::build_date(build_date)?;
    }
    let Some(prefix) = json_string(object, "prefix") else {
        bail!(
            "local assistant must ask the user for a ROCm/TheRock install folder before requesting install_sdk"
        );
    };
    let prefix_path = Path::new(&prefix);
    if chat_install_prefix_is_system(prefix_path) {
        bail!(
            "local assistant cannot request system install folder `{}`",
            prefix_path.display()
        );
    }
    Ok(())
}

fn validate_chat_launch_server_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("launch_server arguments must be a JSON object")?;
    if object
        .get("allow_public_bind")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        bail!("local assistant cannot request public network binding");
    }
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    if !is_loopback_host(&host) {
        bail!("local assistant cannot request non-local host `{host}`");
    }
    if object
        .get("device")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|device| device.to_ascii_lowercase().contains("cpu"))
    {
        bail!("local assistant cannot request CPU execution; ROCm GPU execution is required");
    }
    Ok(())
}

fn validate_chat_service_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("service tool arguments must be a JSON object")?;
    let service_id =
        json_string(object, "service_id").context("service tool requires `service_id`")?;
    validate_service_id(&service_id)?;
    if call.name == "service_logs" {
        validate_optional_chat_integer(call, "lines", 1, 500)?;
    }
    Ok(())
}

fn validate_chat_watcher_tool_call(call: &providers::ChatToolCall, allow_mode: bool) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("watcher tool arguments must be a JSON object")?;
    let watcher = json_string(object, "watcher").context("watcher tool requires `watcher`")?;
    if watcher.len() > 128 {
        bail!("watcher id too long");
    }
    if builtin_watcher(&watcher).is_none() {
        bail!("local assistant requested unknown watcher `{watcher}`");
    }
    if let Some(mode) = json_string(object, "mode") {
        if !allow_mode {
            bail!("local assistant cannot set `mode` when disabling a watcher");
        }
        if !matches!(mode.as_str(), "observe" | "propose" | "contained") {
            bail!("local assistant requested unsupported watcher mode `{mode}`");
        }
    }
    Ok(())
}

fn validate_chat_port_status_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("port_status arguments must be a JSON object")?;
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    if !is_loopback_host(&host) {
        bail!("local assistant cannot inspect non-local host `{host}`");
    }
    let Some(port) = object.get("port").and_then(serde_json::Value::as_u64) else {
        bail!("ROCm tool `port_status` requires integer `port`");
    };
    if !(1..=u64::from(u16::MAX)).contains(&port) {
        bail!("ROCm tool `port_status` argument `port` must be between 1 and 65535");
    }
    Ok(())
}

fn validate_required_chat_string(call: &providers::ChatToolCall, key: &str) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("tool arguments must be a JSON object")?;
    json_string(object, key)
        .with_context(|| format!("ROCm tool `{}` requires non-empty `{key}`", call.name))?;
    Ok(())
}

fn validate_optional_chat_integer(
    call: &providers::ChatToolCall,
    key: &str,
    min: u64,
    max: u64,
) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("tool arguments must be a JSON object")?;
    let Some(value) = object.get(key) else {
        return Ok(());
    };
    let Some(value) = value.as_u64() else {
        bail!(
            "ROCm tool `{}` argument `{key}` must be an integer",
            call.name
        );
    };
    if value < min || value > max {
        bail!(
            "ROCm tool `{}` argument `{key}` must be between {min} and {max}",
            call.name
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum ChatRocmCommandAction {
    ReadOnly(Vec<String>),
    Approval {
        args: Vec<String>,
        pending_title: String,
        command_title: String,
    },
}

fn validate_chat_rocm_command_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    chat_rocm_command_action(call).map(|_| ())
}

fn chat_rocm_command_action(call: &providers::ChatToolCall) -> Result<ChatRocmCommandAction> {
    let args = normalized_chat_rocm_command_args(call)?;
    chat_rocm_command_action_from_args(args)
}

fn normalized_chat_rocm_command_args(call: &providers::ChatToolCall) -> Result<Vec<String>> {
    let object = call
        .arguments
        .as_object()
        .context("rocm_command arguments must be a JSON object")?;
    let values = object
        .get("args")
        .and_then(serde_json::Value::as_array)
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
    if args.is_empty() {
        bail!("rocm_command args should omit the leading `rocm` program name");
    }
    if let Some(reason) = object.get("reason")
        && !reason.is_string()
    {
        bail!("rocm_command `reason` must be a string when present");
    }
    Ok(args)
}

fn chat_rocm_command_action_from_args(mut args: Vec<String>) -> Result<ChatRocmCommandAction> {
    canonicalize_chat_rocm_command(&mut args)?;
    validate_chat_rocm_command_safety(&args)?;
    let first = args.first().map(|value| value.to_ascii_lowercase());
    let second = args.get(1).map(|value| value.to_ascii_lowercase());
    match first.as_deref() {
        Some("examine" | "version" | "model" | "models" | "daemon" | "logs") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("update") if !args.iter().any(|arg| arg == "--apply") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("runtimes") if second.as_deref().is_none_or(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("engines") if second.as_deref().is_some_and(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("services")
            if second
                .as_deref()
                .is_none_or(|value| matches!(value, "list" | "logs")) =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("automations") if second.as_deref().is_none_or(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("config") if second.as_deref() == Some("show") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("comfyui")
            if second
                .as_deref()
                .is_none_or(|value| matches!(value, "status" | "logs" | "log")) =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("install") if second.as_deref() == Some("sdk") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install ROCm".to_owned(),
                command_title: "Install".to_owned(),
            })
        }
        Some("install") if second.as_deref() == Some("driver") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install driver".to_owned(),
                command_title: "Install".to_owned(),
            })
        }
        Some("update") if args.iter().any(|arg| arg == "--apply") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Apply ROCm update".to_owned(),
                command_title: "Update".to_owned(),
            })
        }
        Some("runtimes") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Change ROCm install".to_owned(),
            command_title: "Runtimes".to_owned(),
        }),
        Some("engines") if second.as_deref() == Some("install") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install engine".to_owned(),
                command_title: "Engine".to_owned(),
            })
        }
        Some("serve") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Start local model server".to_owned(),
            command_title: "Serve".to_owned(),
        }),
        Some("services")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "stop" | "restart")) =>
        {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Change local model server".to_owned(),
                command_title: "Services".to_owned(),
            })
        }
        Some("automations")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "enable" | "disable")) =>
        {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Change automation".to_owned(),
                command_title: "Automations".to_owned(),
            })
        }
        Some("config") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Change settings".to_owned(),
            command_title: "Config".to_owned(),
        }),
        Some("uninstall") if args.iter().any(|arg| arg == "--dry-run") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("uninstall") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Uninstall ROCm CLI".to_owned(),
                command_title: "Uninstall".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("install") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("start") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Start ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("stop") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Stop ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("setup") if second.as_deref().is_none_or(|value| value == "status") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("setup") if second.as_deref() == Some("reset") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Reset first-time setup".to_owned(),
                command_title: "Setup".to_owned(),
            })
        }
        Some(command) => bail!("local assistant cannot use unsupported rocm command `{command}`"),
        None => bail!("rocm_command requires at least one argument"),
    }
}

fn canonicalize_chat_rocm_command(args: &mut [String]) -> Result<()> {
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfy"))
    {
        args[0] = "comfyui".to_owned();
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("engine"))
    {
        bail!("use `engines` for rocm engine commands");
    }
    Ok(())
}

fn validate_chat_rocm_command_safety(args: &[String]) -> Result<()> {
    if let Some(first) = args.first()
        && first.eq_ignore_ascii_case("serve")
    {
        if chat_cli_has_flag(args, "--allow-public-bind") {
            bail!("local assistant cannot request public network binding");
        }
        if serve_args_request_cpu_device(args) {
            bail!("local assistant cannot request CPU execution; ROCm GPU execution is required");
        }
        if let Some(host) = chat_cli_arg_value(args, "--host")
            && !is_loopback_host(host)
        {
            bail!("local assistant cannot request non-local host `{host}`");
        }
        if chat_cli_has_flag(args, "--foreground") || !chat_cli_has_flag(args, "--managed") {
            bail!("local assistant must request managed serving with --managed");
        }
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfyui"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("start"))
        && let Some(host) = chat_cli_arg_value(args, "--host")
        && !is_loopback_host(host)
    {
        bail!("local assistant cannot start ComfyUI on non-local host `{host}`");
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("install"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("sdk"))
    {
        if rocm_core::runtime_is_windows()
            && chat_cli_arg_value(args, "--format")
                .is_some_and(|value| !value.eq_ignore_ascii_case("wheel"))
        {
            bail!("local assistant cannot request non-wheel ROCm installs on Windows");
        }
        let version = chat_cli_arg_value_checked(args, "--version")?;
        let build_date = chat_cli_arg_value_checked(args, "--build-date")?;
        if version.is_some() && build_date.is_some() {
            bail!("local assistant cannot request both --version and --build-date");
        }
        if version.is_some() || build_date.is_some() {
            let format = chat_cli_arg_value(args, "--format").unwrap_or("wheel");
            if !format.eq_ignore_ascii_case("wheel") {
                bail!(
                    "local assistant can only request specific TheRock wheel versions for wheel installs"
                );
            }
        }
        if let Some(version) = version {
            therock::RuntimeVersionSelector::version(version)?;
        }
        if let Some(build_date) = build_date {
            therock::RuntimeVersionSelector::build_date(build_date)?;
        }
        let Some(prefix) = chat_cli_arg_value_checked(args, "--prefix")? else {
            bail!(
                "local assistant must ask the user for a ROCm/TheRock install folder before requesting `rocm install sdk`"
            );
        };
        let prefix_path = Path::new(prefix);
        if chat_install_prefix_is_system(prefix_path) {
            bail!(
                "local assistant cannot request system install folder `{}`",
                prefix_path.display()
            );
        }
    }
    Ok(())
}

fn ensure_flag(args: &mut Vec<String>, flag: &str) {
    if !args.iter().any(|arg| arg == flag) {
        args.push(flag.to_owned());
    }
}

fn chat_cli_arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            return Some(value);
        }
        if arg == name {
            return args.get(index + 1).map(String::as_str);
        }
        index += 1;
    }
    None
}

fn chat_cli_arg_value_checked<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            if value.trim().is_empty() {
                bail!("local assistant provided empty value for {name}");
            }
            return Ok(Some(value));
        }
        if arg == name {
            let Some(value) = args.get(index + 1).map(String::as_str) else {
                bail!("local assistant omitted the value for {name}");
            };
            if value.starts_with("--") || value.trim().is_empty() {
                bail!("local assistant omitted the value for {name}");
            }
            return Ok(Some(value));
        }
        index += 1;
    }
    Ok(None)
}

fn chat_cli_has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| {
        arg == name
            || arg
                .strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

fn chat_install_prefix_is_system(prefix: &Path) -> bool {
    prefix.as_os_str().is_empty() || runtime_install_root_is_protected(prefix)
}

pub(crate) fn chat_tool_call_is_read_only(call: &providers::ChatToolCall) -> bool {
    if call.name == "rocm_command" {
        return matches!(
            chat_rocm_command_action(call),
            Ok(ChatRocmCommandAction::ReadOnly(_))
        );
    }
    // `proposal_action` is read-only only when showing a proposal; approve/reject
    // mutate the proposal status and must route through approval.
    if call.name == "proposal_action" {
        return call
            .arguments
            .as_object()
            .and_then(|object| json_string(object, "action"))
            .as_deref()
            == Some("show");
    }
    matches!(
        call.name.as_str(),
        "examine"
            | "doctor"
            | "bridge_snapshot"
            | "gpu_snapshot"
            | "engines"
            | "services"
            | "service_logs"
            | "automations"
            | "natural_language_plan"
            | "path_exists"
            | "port_status"
            | "update_check"
            | "install_sdk_dry_run"
    )
}

fn deterministic_rocm_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("examine:") {
        return None;
    }
    let mut lines = Vec::new();
    let gpu = chat_tool_value(tool_text, "driver_detail")
        .filter(|value| value != "<unknown>")
        .or_else(|| {
            let target = chat_tool_value(tool_text, "detected_gfx_target")?;
            Some(format!("AMD GPU target {target}"))
        });
    if let Some(gpu) = gpu {
        lines.push(format!("  GPU: {gpu}"));
    }

    let active_status = chat_tool_value(tool_text, "active_runtime_status");
    if active_status.as_deref() == Some("ready") {
        let family = chat_tool_value(tool_text, "active_runtime_family")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let version = chat_tool_value(tool_text, "active_runtime_version")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let mut detail = "installed and active for ROCm CLI".to_owned();
        if let Some(family) = family {
            let _ = write!(detail, " ({family})");
        }
        if let Some(version) = version {
            let _ = write!(detail, ", {version}");
        }
        lines.push(format!("  ROCm/TheRock: {detail}"));
        if let Some(root) = chat_tool_value(tool_text, "active_runtime_root")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_root"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Install folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "active_runtime_pip_cache_dir")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_pip_cache_dir"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    } else if let Some(status) = active_status.as_deref() {
        lines.push(format!("  ROCm/TheRock: active runtime status is {status}"));
        if let Some(root) = chat_tool_value(tool_text, "setup_runtime_root")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Selected setup folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "setup_runtime_pip_cache_dir")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    }

    if chat_tool_value(tool_text, "legacy_rocm_status").as_deref() == Some("not_detected")
        && active_status.as_deref() == Some("ready")
    {
        lines.push(
            "  Note: no global legacy ROCm install was found; ROCm CLI is using its managed TheRock runtime."
                .to_owned(),
        );
    }

    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_chat_tool_summary(tool_text: &str) -> Option<String> {
    deterministic_rocm_tool_summary(tool_text)
        .or_else(|| deterministic_model_tool_summary(tool_text))
        .or_else(|| deterministic_combined_status_tool_summary(tool_text))
}

fn deterministic_combined_status_tool_summary(tool_text: &str) -> Option<String> {
    let summaries = [
        deterministic_comfyui_tool_summary(tool_text),
        deterministic_port_status_tool_summary(tool_text),
        deterministic_engine_inventory_tool_summary(tool_text),
        deterministic_services_tool_summary(tool_text),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    (!summaries.is_empty()).then(|| summaries.join("\n"))
}

fn deterministic_summary_can_stand_alone(tool_text: &str) -> bool {
    !tool_text.contains("model recipes")
}

#[derive(Debug, Default)]
struct DeterministicModelRecipe {
    canonical_id: String,
    min_gpu_mem_gib: Option<u32>,
    engines: Vec<String>,
    engine_statuses: Vec<String>,
    warnings: Vec<String>,
}

fn deterministic_model_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("model recipes") {
        return None;
    }
    let recipes = parse_deterministic_model_recipes(tool_text);
    if recipes.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == providers::LEMONADE_ASSISTANT_MODEL_ID)
    {
        lines.push(format!(
            "  Recommended local assistant: qwen ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: needs about {} GPU memory.",
            recipe
                .min_gpu_mem_gib
                .map_or_else(|| "unknown".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engine: {}",
            deterministic_engine_fit_summary(recipe, "lemonade")
        ));
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "Qwen3-0.6B-GGUF")
    {
        lines.push(format!(
            "  Tiny smoke test: qwen-smoke ({})",
            recipe.canonical_id
        ));
        lines.push(
            "    Use this only to check that GPU serving starts; it is not the default assistant."
                .to_owned(),
        );
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "meta-llama/Llama-3.2-3B-Instruct")
    {
        lines.push(format!(
            "  8 GiB-class option: llama ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: asks for {}; this can be tight on APUs and depends on available shared GPU memory.",
            recipe
                .min_gpu_mem_gib.map_or_else(|| "unknown GPU memory".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engines: {}",
            deterministic_available_engine_names(recipe).join(", ")
        ));
    }

    let larger = recipes
        .iter()
        .filter(|recipe| recipe.min_gpu_mem_gib.is_some_and(|value| value > 8))
        .filter(|recipe| recipe.canonical_id != providers::BUILTIN_ASSISTANT_MODEL_ID)
        .map(|recipe| {
            format!(
                "{} asks for {}",
                recipe.canonical_id,
                recipe.min_gpu_mem_gib.map_or_else(
                    || "more GPU memory".to_owned(),
                    |value| format!("{value} GiB")
                )
            )
        })
        .collect::<Vec<_>>();
    if !larger.is_empty() {
        lines.push(format!(
            "  Larger recipes are not low-VRAM defaults: {}.",
            larger.join("; ")
        ));
    }

    let linux_wsl_only = recipes
        .iter()
        .flat_map(|recipe| {
            recipe
                .engine_statuses
                .iter()
                .filter(|status| {
                    status.contains("unsupported_native_windows")
                        || status.to_ascii_lowercase().contains("wsl/linux")
                        || status.to_ascii_lowercase().contains("linux/wsl")
                })
                .filter_map(|status| {
                    let (engine, _) = status.split_once(':')?;
                    Some(format!(
                        "{} uses {} through WSL/Linux on Windows",
                        recipe.canonical_id, engine
                    ))
                })
        })
        .collect::<Vec<_>>();
    if !linux_wsl_only.is_empty() {
        lines.push(format!(
            "  Native Windows note: {}.",
            linux_wsl_only.join("; ")
        ));
    }

    if !lines.is_empty() {
        lines.push(
            "  Run `rocm examine` to refresh GPU memory details before starting anything large."
                .to_owned(),
        );
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn parse_deterministic_model_recipes(tool_text: &str) -> Vec<DeterministicModelRecipe> {
    let mut recipes = Vec::new();
    let mut current: Option<DeterministicModelRecipe> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(recipe) = parse_model_recipe_header(trimmed) {
            if let Some(previous) = current.replace(recipe) {
                recipes.push(previous);
            }
            continue;
        }
        let Some(recipe) = current.as_mut() else {
            continue;
        };
        if is_engine_status_line(trimmed) {
            recipe.engine_statuses.push(trimmed.to_owned());
        } else if let Some(warning) = trimmed.strip_prefix("warning:") {
            recipe.warnings.push(warning.trim().to_owned());
        }
    }
    if let Some(recipe) = current {
        recipes.push(recipe);
    }
    recipes
}

fn parse_model_recipe_header(line: &str) -> Option<DeterministicModelRecipe> {
    if !line.contains(" aliases=[") || !line.contains("min_gpu_mem=") {
        return None;
    }
    let canonical_id = line.split_whitespace().next()?.to_owned();
    let engines = parse_bracketed_csv(line, "engines=[", "]");
    let min_gpu_mem_gib = line
        .split_once("min_gpu_mem=")
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u32>().ok());
    Some(DeterministicModelRecipe {
        canonical_id,
        min_gpu_mem_gib,
        engines,
        engine_statuses: Vec::new(),
        warnings: Vec::new(),
    })
}

fn parse_bracketed_csv(line: &str, start: &str, end: &str) -> Vec<String> {
    line.split_once(start)
        .and_then(|(_, rest)| rest.split_once(end))
        .map(|(values, _)| {
            values
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn is_engine_status_line(line: &str) -> bool {
    matches!(
        line.split_once(':').map(|(engine, _)| engine),
        Some("pytorch" | "llama.cpp" | "lemonade" | "vllm" | "sglang" | "atom")
    )
}

fn deterministic_engine_fit_summary(recipe: &DeterministicModelRecipe, engine: &str) -> String {
    let status = recipe
        .engine_statuses
        .iter()
        .find(|status| status.starts_with(&format!("{engine}:")))
        .map(std::string::String::as_str);
    match status {
        Some(status) if status.contains("unsupported_native_windows") => {
            format!("{engine}: WSL/Linux only on Windows")
        }
        Some(status)
            if status.split_once(':').is_some_and(|(_, rest)| {
                rest.trim_start().starts_with("available")
                    || rest.trim_start().starts_with("adapter_available")
            }) =>
        {
            format!("{engine}: available")
        }
        Some(status) => status.to_owned(),
        None if recipe.engines.iter().any(|candidate| candidate == engine) => {
            format!("{engine}: listed")
        }
        None => format!("{engine}: not listed"),
    }
}

fn deterministic_available_engine_names(recipe: &DeterministicModelRecipe) -> Vec<String> {
    let engines = recipe
        .engine_statuses
        .iter()
        .filter_map(|status| {
            let (engine, rest) = status.split_once(':')?;
            (rest.trim_start().starts_with("available")
                || rest.trim_start().starts_with("adapter_available"))
            .then_some(engine.to_owned())
        })
        .collect::<Vec<_>>();
    if engines.is_empty() {
        recipe.engines.clone()
    } else {
        engines
    }
}

fn deterministic_comfyui_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("ComfyUI") || !tool_text.contains("Running") {
        return None;
    }
    let mut lines = Vec::new();
    match chat_tool_value(tool_text, "installed").as_deref() {
        Some("yes") => lines.push("  ComfyUI: installed.".to_owned()),
        Some("no") => lines.push("  ComfyUI: not installed.".to_owned()),
        Some(value) => lines.push(format!("  ComfyUI: installed status is {value}.")),
        None => {}
    }
    if let Some(status) = chat_tool_value(tool_text, "status") {
        let running_text = if status.eq_ignore_ascii_case("running") {
            "running".to_owned()
        } else if status.eq_ignore_ascii_case("starting") {
            "starting".to_owned()
        } else {
            format!("not running ({status})")
        };
        lines.push(format!("  Running: {running_text}."));
    }
    if let Some(models_path) = chat_tool_value(tool_text, "models path") {
        lines.push(format!("  Models folder: {models_path}."));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[derive(Debug, Default)]
struct DeterministicServiceRow {
    service_id: String,
    engine: String,
    model: String,
    status: String,
    running_state: String,
    endpoint: String,
}

fn deterministic_services_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local Servers")
        && !tool_text.contains("managed_services:")
        && !tool_text.contains("services:")
    {
        return None;
    }
    let rows = parse_deterministic_service_rows(tool_text);
    let mut lines = Vec::new();
    let live = rows
        .iter()
        .filter(|row| matches!(row.running_state.as_str(), "running" | "starting"))
        .collect::<Vec<_>>();
    if live.is_empty() {
        lines.push("  Local model servers: none running under ROCm CLI.".to_owned());
    } else {
        lines.push(format!(
            "  Local model servers running/starting: {}.",
            live.len()
        ));
        for row in live.iter().take(4) {
            lines.push(format!(
                "    {} {} ({}) at {}.",
                row.engine,
                row.model,
                row.running_state,
                empty_as_unknown(&row.endpoint)
            ));
        }
    }
    let past = rows
        .iter()
        .filter(|row| !matches!(row.running_state.as_str(), "running" | "starting"))
        .take(3)
        .map(|row| {
            format!(
                "{} {} {}",
                empty_as_unknown(&row.engine),
                empty_as_unknown(&row.model),
                empty_as_unknown(&row.status)
            )
        })
        .collect::<Vec<_>>();
    if !past.is_empty() {
        lines.push(format!("  Past/non-running records: {}.", past.join("; ")));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn parse_deterministic_service_rows(tool_text: &str) -> Vec<DeterministicServiceRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicServiceRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("- service_id=") {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            let mut row = DeterministicServiceRow::default();
            for token in rest.split_whitespace() {
                if let Some((key, value)) = token.split_once('=') {
                    match key {
                        "service_id" => row.service_id = value.to_owned(),
                        "engine" => row.engine = value.to_owned(),
                        "model" => row.model = value.to_owned(),
                        "status" => row.status = value.to_owned(),
                        "running_state" => row.running_state = value.to_owned(),
                        "endpoint" => row.endpoint = value.to_owned(),
                        _ => {}
                    }
                }
            }
            if row.running_state.is_empty() {
                row.running_state = managed_service_running_state(&row.status).to_owned();
            }
            current = Some(row);
            continue;
        }
        if let Some(service_id) = trimmed.strip_prefix("- ") {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicServiceRow {
                service_id: service_id.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(value) = trimmed.strip_prefix("status:") {
            row.status = value.trim().to_owned();
            row.running_state = managed_service_running_state(&row.status).to_owned();
        } else if let Some(value) = trimmed.strip_prefix("engine:") {
            row.engine = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("model:") {
            row.model = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("endpoint:") {
            row.endpoint = value.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

fn deterministic_port_status_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("port_status:") && !tool_text.contains("listening:") {
        return None;
    }
    let port = chat_tool_value(tool_text, "port")?;
    let listening = chat_tool_value(tool_text, "listening").unwrap_or_else(|| "unknown".to_owned());
    let mut lines = Vec::new();
    let state = match listening.as_str() {
        "true" => "listening",
        "false" => "not listening",
        _ => "unknown",
    };
    lines.push(format!("  Port {port}: {state}."));
    if let Some(hint) = chat_tool_value(tool_text, "hint") {
        lines.push(format!("  Hint: {hint}."));
    }
    if chat_tool_value(tool_text, "managed_service").as_deref() == Some("none") {
        lines.push("  Managed service: none on that endpoint.".to_owned());
    }
    Some(lines.join("\n"))
}

#[derive(Debug, Default)]
struct DeterministicEngineRow {
    engine: String,
    runtime: String,
    note: String,
}

fn deterministic_engine_inventory_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local model engines") {
        return None;
    }
    let rows = parse_deterministic_engine_rows(tool_text);
    if rows.is_empty() {
        return None;
    }
    let mut lines = vec!["  Engine runtimes:".to_owned()];
    for row in rows {
        let mut line = format!(
            "    {}: {}",
            friendly_engine_label(&row.engine),
            empty_as_unknown(&row.runtime)
        );
        if !row.note.trim().is_empty() {
            let _ = write!(line, " ({})", row.note.trim());
        }
        line.push('.');
        lines.push(line);
    }
    Some(lines.join("\n"))
}

fn parse_deterministic_engine_rows(tool_text: &str) -> Vec<DeterministicEngineRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicEngineRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim().trim_start_matches("* ").trim_start();
        if let Some(engine) = engine_name_from_inventory_line(trimmed) {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicEngineRow {
                engine: engine.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(runtime) = trimmed.strip_prefix("runtime:") {
            row.runtime = runtime.trim().to_owned();
        } else if let Some(note) = trimmed.strip_prefix("note:") {
            row.note = note.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

fn engine_name_from_inventory_line(line: &str) -> Option<&'static str> {
    ["pytorch", "llama.cpp", "lemonade", "vllm", "sglang", "atom"]
        .into_iter()
        .find(|engine| {
            line == *engine
                || line
                    .strip_prefix(*engine)
                    .is_some_and(|rest| rest.starts_with(char::is_whitespace))
        })
}

fn should_request_local_tool_follow_up(
    provider: &str,
    tool_result: &ChatToolRunResult,
    deterministic_summary: Option<&str>,
) -> bool {
    provider == "local"
        && deterministic_summary.is_none()
        && tool_result.approval.is_none()
        && tool_result.ran_read_only_tool
        && !tool_result.read_only_tool_error
        && !tool_result.follow_up_text.trim().is_empty()
}

fn chat_tool_value(tool_text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    tool_text.lines().find_map(|line| {
        line.trim()
            .strip_prefix(&prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

// pub(crate): used by the dash execution seam (dash_seam.rs)
pub(crate) fn run_internal_mcp_call(
    paths: &AppPaths,
    name: &str,
    arguments: serde_json::Value,
    allow_mutation: bool,
) -> Result<serde_json::Value> {
    let arguments = internal_mcp_arguments(arguments);
    let call = providers::ChatToolCall {
        id: None,
        name: name.to_owned(),
        arguments: serde_json::Value::Object(arguments.clone()),
    };
    validate_chat_tool_call(&call)?;
    if !chat_tool_call_is_read_only(&call) && !allow_mutation {
        bail!(
            "MCP tool `{name}` changes local ROCm state; rerun `rocm mcp-call {name}` with --allow-mutation only after explicit user approval"
        );
    }

    match name {
        "examine" | "doctor" => {
            let examine = ExamineSummary::gather()?;
            let text = render_examine_text()?;
            Ok(internal_mcp_tool_success(text, serde_json::json!(examine)))
        }
        "bridge_snapshot" => {
            let snapshot = build_codex_bridge_snapshot(paths)?;
            Ok(internal_mcp_tool_success(
                format!(
                    "Captured bridge snapshot for {} / {} with default engine `{}`.",
                    snapshot.examine.os, snapshot.examine.arch, snapshot.examine.default_engine
                ),
                serde_json::json!(snapshot),
            ))
        }
        "gpu_snapshot" => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let gpu = build_codex_bridge_gpu_snapshot(&config);
            let status = if !config.telemetry.local_inspection_enabled() {
                "GPU telemetry is disabled by rocm-cli config."
            } else if gpu.amd_smi_available {
                "Captured amd-smi GPU snapshot."
            } else {
                "Use `rocm examine` for the current local AMD GPU summary."
            };
            Ok(internal_mcp_tool_success(
                status.to_owned(),
                serde_json::json!(gpu),
            ))
        }
        "engines" => {
            let engines = builtin_codex_bridge_engine_inventory();
            Ok(internal_mcp_tool_success(
                format!("Found {} engine entries.", engines.len()),
                serde_json::json!({ "engines": engines }),
            ))
        }
        "services" => {
            let services = load_managed_services(paths)?;
            Ok(internal_mcp_tool_success(
                render_services_tool_result_text(&services),
                serde_json::json!({ "services": services }),
            ))
        }
        "port_status" => run_chat_port_status_tool(paths, &call),
        "service_logs" => {
            let service_id = json_string(&arguments, "service_id")
                .context("service_logs requires `service_id`")?;
            let lines = arguments
                .get("lines")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(80)
                .clamp(1, 500) as usize;
            let record = load_managed_services(paths)?
                .into_iter()
                .find(|service| service.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            let tail = read_tail_lines(&record.log_path, lines, "service log")?.join("\n");
            Ok(internal_mcp_tool_success(
                format!(
                    "Read the last {} line(s) from service `{}`.",
                    lines, record.service_id
                ),
                serde_json::json!({
                    "service": record,
                    "lines": lines,
                    "tail": tail,
                }),
            ))
        }
        "automations" => {
            let event_limit = arguments
                .get("event_limit")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 64) as usize;
            let runtime = AutomationRuntimeState::load(paths)?;
            let events = load_recent_automation_events(paths, event_limit)?;
            Ok(internal_mcp_tool_success(
                format!(
                    "Loaded automation runtime and {} recent events.",
                    events.len()
                ),
                serde_json::json!({
                    "runtime": runtime,
                    "recent_events": events,
                }),
            ))
        }
        "natural_language_plan" => {
            let request = json_string(&arguments, "request")
                .context("natural_language_plan requires `request`")?;
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let text = render_freeform_plan(&request, paths, &config);
            let action = freeform_plan_next_action_with_context(&request, paths, &config).map(
                |action| {
                    serde_json::json!({
                        "title": action.title,
                        "args": action.args,
                        "approval_required": action.approval_required,
                        "has_placeholders": action.has_placeholders,
                        "reason": action.reason,
                    })
                },
            );
            Ok(internal_mcp_tool_success(
                "Planned the ROCm request.".to_owned(),
                serde_json::json!({
                    "request": request,
                    "text": text,
                    "action": action,
                }),
            ))
        }
        "rocm_command" => {
            let action = chat_rocm_command_action(&call)?;
            let output = match action {
                ChatRocmCommandAction::ReadOnly(args) => {
                    run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?
                }
                ChatRocmCommandAction::Approval { args, .. } if allow_mutation => {
                    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
                    run_rocm_capture_for_paths(paths, &refs, Duration::from_mins(2))?
                }
                ChatRocmCommandAction::Approval { .. } => {
                    bail!("rocm_command changes local ROCm state and needs approval")
                }
            };
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output =
                run_rocm_command_for_paths(paths, &["update".to_owned()], Duration::from_mins(1))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let args = internal_mcp_install_sdk_args(&arguments, true)?;
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "path_exists" => run_chat_path_exists_tool(&call),
        "proposal_action" => {
            // Executes IN-PROCESS (no CLI subprocess): show loads a proposal;
            // approve/reject mutate its status (allow_mutation already enforced
            // by the read-only split above). proposal_action executes in-process
            // (via update_automation_proposal_status); it never delegates to the
            // subprocess arm.
            let proposal_id = json_string(&arguments, "proposal_id")
                .context("proposal_action requires `proposal_id`")?;
            let action =
                json_string(&arguments, "action").context("proposal_action requires `action`")?;
            match action.as_str() {
                "show" => {
                    let proposal = rocm_core::find_automation_proposal(paths, &proposal_id)
                        .with_context(|| {
                            format!("automation proposal `{proposal_id}` not found")
                        })?;
                    Ok(internal_mcp_tool_success(
                        format!(
                            "Proposal {} ({}): {}",
                            proposal.proposal_id, proposal.status, proposal.title
                        ),
                        serde_json::json!({
                            "id": proposal.proposal_id,
                            "status": proposal.status,
                            "summary": proposal.title,
                            "reason": proposal.message,
                        }),
                    ))
                }
                "approve" | "reject" => {
                    let status = if action == "approve" {
                        "approved"
                    } else {
                        "rejected"
                    };
                    let updated =
                        rocm_core::update_automation_proposal_status(paths, &proposal_id, status)?;
                    record_cli_audit_event(
                        paths,
                        "automations",
                        if action == "approve" {
                            "proposal_approved"
                        } else {
                            "proposal_rejected"
                        },
                        "info",
                        format!("proposal {proposal_id} {status}"),
                        None,
                    );
                    Ok(internal_mcp_tool_success(
                        format!("Proposal {proposal_id} {status}."),
                        serde_json::json!({
                            "id": updated.proposal_id,
                            "status": updated.status,
                            "summary": updated.title,
                            "reason": updated.message,
                        }),
                    ))
                }
                other => bail!("proposal_action `{other}` is not supported"),
            }
        }
        "install_sdk" | "install_engine" | "launch_server" | "stop_server" | "watcher_enable"
        | "watcher_disable" => {
            let args = rocm_chat_tool_requested_args(&call)
                .with_context(|| format!("MCP tool `{name}` is missing required arguments"))?;
            let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture_for_paths(paths, &refs, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran approved `rocm` command.",
                output,
                false,
            ))
        }
        other => bail!("unsupported MCP tool `{other}`"),
    }
}

fn internal_mcp_arguments(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    if let Some(arguments) = value
        .get("arguments")
        .and_then(serde_json::Value::as_object)
    {
        return arguments.clone();
    }
    value.as_object().cloned().unwrap_or_default()
}

fn internal_mcp_tool_success(text: String, structured: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
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

fn internal_mcp_tool_result_from_command(
    prefix: &str,
    output: CommandCapture,
    is_error: bool,
) -> serde_json::Value {
    let text = format!("{prefix}\n\n{}", command_capture_text(&output));
    serde_json::json!({
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

fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    let rocm_binary = daemon_binary_path()?;
    let mut command = ProcessCommand::new(&rocm_binary);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_app_path_env(&mut command, paths);
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

fn run_rocm_command_for_paths(
    paths: &AppPaths,
    args: &[String],
    _timeout: Duration,
) -> Result<CommandCapture> {
    match run_rocm_read_only_in_process(paths, args) {
        Ok(stdout) => Ok(CommandCapture {
            argv: std::iter::once("rocm".to_owned())
                .chain(args.iter().cloned())
                .collect(),
            exit_status: 0,
            stdout,
            stderr: String::new(),
        }),
        Err(error) => Err(error).with_context(|| {
            format!(
                "read-only assistant command is not implemented in-process: {}",
                format_structured_tool_call("rocm", args)
            )
        }),
    }
}

fn run_rocm_read_only_in_process(paths: &AppPaths, args: &[String]) -> Result<String> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    match args {
        [] => bail!("rocm command requires at least one argument"),
        [command] if command.eq_ignore_ascii_case("examine") => {
            render_examine_text_with_paths(paths, &config)
        }
        [command]
            if command.eq_ignore_ascii_case("version")
                || command == "--version"
                || command == "-V" =>
        {
            Ok(format!("rocm {}\n", env!("CARGO_PKG_VERSION")))
        }
        [command]
            if command.eq_ignore_ascii_case("model") || command.eq_ignore_ascii_case("models") =>
        {
            Ok(render_model_registry_verbose_text_with_context_and_host(
                Some(paths),
                None,
                None,
            ))
        }
        [command] if command.eq_ignore_ascii_case("daemon") => {
            Ok(render_daemon_text(paths, &config))
        }
        [command] if command.eq_ignore_ascii_case("logs") => Ok(render_logs_text(paths)),
        [command, rest @ ..] if command.eq_ignore_ascii_case("logs") => {
            let query = parse_optional_query(rest)?;
            Ok(render_logs_browser_text(paths, query.as_deref()))
        }
        [command] if command.eq_ignore_ascii_case("runtimes") => {
            render_runtimes_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("runtimes")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_runtimes_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("engines")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            Ok(render_engine_inventory_text_with_paths(Some(paths)))
        }
        [command] if command.eq_ignore_ascii_case("services") => render_services_text(paths, false),
        [command, subcommand]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_services_text(paths, false)
        }
        [command, subcommand, flag]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list")
                && matches!(flag.as_str(), "-a" | "--all") =>
        {
            render_services_text(paths, true)
        }
        [command, subcommand, service_id]
            if command.eq_ignore_ascii_case("services")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            render_service_logs_text(paths, service_id)
        }
        [command] if command.eq_ignore_ascii_case("automations") => {
            render_automations_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("automations")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_automations_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("config")
                && subcommand.eq_ignore_ascii_case("show") =>
        {
            Ok(render_config_text(paths, &config))
        }
        [command] if command.eq_ignore_ascii_case("comfyui") => {
            comfyui::render_status(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("comfyui")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            comfyui::render_status(paths, &config)
        }
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("comfyui")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            comfyui::render_logs(
                paths,
                parse_optional_lines(rest).unwrap_or(DEFAULT_LOG_TAIL_LINES),
            )
        }
        [command, rest @ ..]
            if command.eq_ignore_ascii_case("update")
                && !rest.iter().any(|arg| arg.eq_ignore_ascii_case("--apply")) =>
        {
            render_update_text(paths)
        }
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("install")
                && subcommand.eq_ignore_ascii_case("sdk")
                && rest.iter().any(|arg| arg.eq_ignore_ascii_case("--dry-run")) =>
        {
            render_install_sdk_dry_run_for_args(paths, rest)
        }
        [command, rest @ ..]
            if command.eq_ignore_ascii_case("uninstall")
                && rest.iter().any(|arg| arg.eq_ignore_ascii_case("--dry-run")) =>
        {
            render_uninstall_dry_run(paths)
        }
        [command] if command.eq_ignore_ascii_case("setup") => {
            render_setup_status_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("setup")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            render_setup_status_text(paths, &config)
        }
        _ => bail!(
            "unsupported in-process read-only rocm command: {}",
            format_structured_tool_call("rocm", args)
        ),
    }
}

fn parse_optional_query(args: &[String]) -> Result<Option<String>> {
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--query" | "-q" => {
                let value = args
                    .get(index + 1)
                    .context("logs query flag requires a value")?;
                return Ok(Some(value.to_owned()));
            }
            value if value.starts_with("--") => {}
            value => return Ok(Some(value.to_owned())),
        }
        index += 1;
    }
    Ok(None)
}

fn parse_optional_lines(args: &[String]) -> Result<usize> {
    let mut index = 0;
    while index < args.len() {
        if matches!(args[index].as_str(), "--lines" | "-n") {
            let value = args.get(index + 1).context("lines flag requires a value")?;
            return value
                .parse::<usize>()
                .context("lines flag must be a positive number");
        }
        index += 1;
    }
    Ok(DEFAULT_LOG_TAIL_LINES)
}

fn render_install_sdk_dry_run_for_args(paths: &AppPaths, args: &[String]) -> Result<String> {
    let channel = chat_cli_arg_value(args, "--channel").unwrap_or("release");
    let format = chat_cli_arg_value(args, "--format").unwrap_or("wheel");
    let prefix = chat_cli_arg_value(args, "--prefix").map(PathBuf::from);
    let version = chat_cli_arg_value(args, "--version").map(str::to_owned);
    let build_date = chat_cli_arg_value(args, "--build-date").map(str::to_owned);
    let selector = therock_install_version_selector(version, build_date)?;
    therock::install_sdk(paths, channel, format, prefix, selector, None, true)
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

fn internal_mcp_install_sdk_args(
    arguments: &serde_json::Map<String, serde_json::Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = json_string(arguments, "channel").unwrap_or_else(|| "release".to_owned());
    let format = json_string(arguments, "format").unwrap_or_else(|| "wheel".to_owned());
    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel,
        "--format".to_owned(),
        format,
    ];
    if let Some(prefix) = json_string(arguments, "prefix") {
        let prefix_path = Path::new(&prefix);
        if chat_install_prefix_is_system(prefix_path) {
            bail!(
                "install_sdk prefix `{}` is a system folder; choose a user folder instead",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix);
    }
    if let Some(version) = json_string(arguments, "version") {
        argv.push("--version".to_owned());
        argv.push(version);
    }
    if let Some(build_date) = json_string(arguments, "build_date") {
        argv.push("--build-date".to_owned());
        argv.push(build_date);
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    }
    Ok(argv)
}

fn run_chat_read_only_tool(
    paths: &AppPaths,
    call: &providers::ChatToolCall,
) -> Result<serde_json::Value> {
    match call.name.as_str() {
        "path_exists" => run_chat_path_exists_tool(call),
        "port_status" => run_chat_port_status_tool(paths, call),
        "rocm_command" => {
            let action = chat_rocm_command_action(call)?;
            let ChatRocmCommandAction::ReadOnly(args) = action else {
                bail!("assistant read-only path cannot run mutating rocm_command");
            };
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output =
                run_rocm_command_for_paths(paths, &["update".to_owned()], Duration::from_mins(1))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let arguments = internal_mcp_arguments(call.arguments.clone());
            let args = internal_mcp_install_sdk_args(&arguments, true)?;
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        _ => run_internal_mcp_call(paths, &call.name, call.arguments.clone(), false),
    }
}

fn run_chat_path_exists_tool(call: &providers::ChatToolCall) -> Result<serde_json::Value> {
    let object = call
        .arguments
        .as_object()
        .context("path_exists arguments must be a JSON object")?;
    let path = json_string(object, "path").context("path_exists requires path")?;
    let path = Path::new(&path);
    let metadata = path.metadata().ok();
    let path_kind = metadata.as_ref().map_or("missing", |metadata| {
        if metadata.is_dir() {
            "directory"
        } else if metadata.is_file() {
            "file"
        } else {
            "other"
        }
    });
    let parent = path.parent();
    let parent_exists = parent.is_some_and(Path::exists);
    let parent_display = parent.map_or_else(
        || "<none>".to_owned(),
        |parent| parent.display().to_string(),
    );
    let text = format!(
        "path: {}\nexists: {}\nkind: {}\nparent: {}\nparent_exists: {}",
        path.display(),
        metadata.is_some(),
        path_kind,
        parent_display,
        parent_exists
    );
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": text,
        }]
    }))
}

fn run_chat_port_status_tool(
    paths: &AppPaths,
    call: &providers::ChatToolCall,
) -> Result<serde_json::Value> {
    let object = call
        .arguments
        .as_object()
        .context("port_status arguments must be a JSON object")?;
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    let port = object
        .get("port")
        .and_then(serde_json::Value::as_u64)
        .context("port_status requires port")? as u16;
    let reachable = loopback_tcp_port_is_reachable(&host, port);
    let host_key = loopback_host_key(&host);
    let matching_services = load_managed_services(paths)?
        .into_iter()
        .filter(|record| {
            record.port == port
                && loopback_host_key(&record.host) == host_key
                && managed_service_is_live(record)
        })
        .collect::<Vec<_>>();
    let app_hint = if port == comfyui::default_port() {
        Some("ComfyUI default port")
    } else if port == rocm_core::DEFAULT_LOCAL_PORT {
        Some("ROCm local model server default port")
    } else {
        None
    };
    let mut text = String::new();
    let _ = writeln!(text, "host: {host}");
    let _ = writeln!(text, "port: {port}");
    let _ = writeln!(text, "listening: {reachable}");
    if let Some(app_hint) = app_hint {
        let _ = writeln!(text, "hint: {app_hint}");
    }
    if matching_services.is_empty() {
        let _ = writeln!(text, "managed_service: none");
    } else {
        let _ = writeln!(text, "managed_services:");
        for service in &matching_services {
            let _ = writeln!(
                text,
                "  - service_id={} engine={} model={} status={} running_state={} endpoint={}",
                service.service_id,
                service.engine,
                service.model_ref,
                service.status,
                managed_service_running_state(&service.status),
                service.endpoint_url
            );
        }
    }
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": text,
        }],
        "structuredContent": {
            "host": host,
            "port": port,
            "listening": reachable,
            "hint": app_hint,
            "managed_services": matching_services,
        },
        "isError": false,
    }))
}

fn loopback_tcp_port_is_reachable(host: &str, port: u16) -> bool {
    let Ok(addresses) = (host, port).to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok())
}

fn mcp_tool_result_text(value: &serde_json::Value) -> String {
    value
        .get("content")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    (item.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                        .then(|| item.get("text").and_then(serde_json::Value::as_str))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| value.to_string())
}

fn mcp_tool_result_is_error(value: &serde_json::Value) -> bool {
    value
        .get("isError")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

const fn chat_read_only_tool_status_label(is_error: bool) -> &'static str {
    if is_error {
        "reported an error"
    } else {
        "done"
    }
}

fn chat_tool_display_label(name: &str) -> String {
    match name {
        "examine" => "Checked this computer".to_owned(),
        "gpu_snapshot" => "Checked GPU status".to_owned(),
        "engines" => "Checked local engines".to_owned(),
        "services" => "Checked model servers".to_owned(),
        "bridge_snapshot" => "Checked ROCm state".to_owned(),
        "service_logs" => "Read server logs".to_owned(),
        "automations" => "Checked automations".to_owned(),
        "natural_language_plan" => "Planned ROCm request".to_owned(),
        "path_exists" => "Checked folder path".to_owned(),
        "port_status" => "Checked local port".to_owned(),
        "update_check" => "Checked for ROCm updates".to_owned(),
        "install_sdk_dry_run" => "Previewed ROCm install".to_owned(),
        "install_sdk" => "Install ROCm".to_owned(),
        "install_engine" => "Install engine".to_owned(),
        "launch_server" => "Start local model server".to_owned(),
        "stop_server" => "Stop local model server".to_owned(),
        "watcher_enable" => "Enable automation".to_owned(),
        "watcher_disable" => "Disable automation".to_owned(),
        other => other.replace('_', " "),
    }
}

fn chat_tool_call_display_label(call: &providers::ChatToolCall) -> String {
    if call.name != "rocm_command" {
        return chat_tool_display_label(&call.name);
    }
    let Some(args) = normalized_chat_rocm_command_args(call).ok() else {
        return "rocm command".to_owned();
    };
    match args.as_slice() {
        [command] if command.eq_ignore_ascii_case("examine") => "Checked this computer".to_owned(),
        [command]
            if command.eq_ignore_ascii_case("model") || command.eq_ignore_ascii_case("models") =>
        {
            "Checked model recipes".to_owned()
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("engines")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            "Checked local engines".to_owned()
        }
        [command] if command.eq_ignore_ascii_case("services") => "Checked model servers".to_owned(),
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list")
                && rest
                    .iter()
                    .all(|arg| matches!(arg.as_str(), "-a" | "--all")) =>
        {
            "Checked model servers".to_owned()
        }
        [command] if command.eq_ignore_ascii_case("comfyui") => "Checked ComfyUI".to_owned(),
        [command, subcommand]
            if command.eq_ignore_ascii_case("comfyui")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            "Checked ComfyUI".to_owned()
        }
        [command, subcommand, ..]
            if command.eq_ignore_ascii_case("comfyui")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            "Read ComfyUI logs".to_owned()
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("config")
                && subcommand.eq_ignore_ascii_case("show") =>
        {
            "Checked ROCm config".to_owned()
        }
        _ => "rocm command".to_owned(),
    }
}

fn rocm_chat_tool_requested_command(call: &providers::ChatToolCall) -> Option<String> {
    let args = rocm_chat_tool_requested_args(call)?;
    Some(format_structured_tool_call("rocm", &args))
}

fn rocm_chat_tool_requested_args(call: &providers::ChatToolCall) -> Option<Vec<String>> {
    let object = call.arguments.as_object()?;
    match call.name.as_str() {
        "install_sdk" => {
            let mut args = vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                json_string(object, "channel").unwrap_or_else(|| "release".to_owned()),
                "--format".to_owned(),
                json_string(object, "format").unwrap_or_else(|| "wheel".to_owned()),
            ];
            if let Some(prefix) = json_string(object, "prefix") {
                args.push("--prefix".to_owned());
                args.push(prefix);
            }
            if let Some(version) = json_string(object, "version") {
                args.push("--version".to_owned());
                args.push(version);
            }
            if let Some(build_date) = json_string(object, "build_date") {
                args.push("--build-date".to_owned());
                args.push(build_date);
            }
            Some(args)
        }
        "install_engine" => {
            let engine = json_string(object, "engine")?;
            let mut args = vec!["engines".to_owned(), "install".to_owned(), engine];
            if let Some(runtime_id) = json_string(object, "runtime_id") {
                args.push("--runtime-id".to_owned());
                args.push(runtime_id);
            }
            if let Some(python_version) = json_string(object, "python_version") {
                args.push("--python-version".to_owned());
                args.push(python_version);
            }
            if object
                .get("reinstall")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                args.push("--reinstall".to_owned());
            }
            Some(args)
        }
        "launch_server" => {
            let model = json_string(object, "model")?;
            let mut args = vec!["serve".to_owned(), model, "--managed".to_owned()];
            push_optional_json_cli_arg(&mut args, object, "--engine", "engine");
            push_optional_json_cli_arg(&mut args, object, "--device", "device");
            push_optional_json_cli_arg(&mut args, object, "--runtime-id", "runtime_id");
            push_optional_json_cli_arg(&mut args, object, "--env-id", "env_id");
            push_optional_json_cli_arg(&mut args, object, "--host", "host");
            if let Some(port) = object.get("port").and_then(serde_json::Value::as_u64) {
                args.push("--port".to_owned());
                args.push(port.to_string());
            }
            Some(args)
        }
        "stop_server" => {
            let service_id = json_string(object, "service_id")?;
            Some(vec![
                "services".to_owned(),
                "stop".to_owned(),
                service_id,
                "--yes".to_owned(),
            ])
        }
        "watcher_enable" => {
            let watcher = json_string(object, "watcher")?;
            let mut args = vec!["automations".to_owned(), "enable".to_owned(), watcher];
            push_optional_json_cli_arg(&mut args, object, "--mode", "mode");
            Some(args)
        }
        "watcher_disable" => {
            let watcher = json_string(object, "watcher")?;
            Some(vec![
                "automations".to_owned(),
                "disable".to_owned(),
                watcher,
            ])
        }
        "rocm_command" => match chat_rocm_command_action(call).ok()? {
            ChatRocmCommandAction::Approval { args, .. }
            | ChatRocmCommandAction::ReadOnly(args) => Some(args),
        },
        _ => None,
    }
}

fn push_optional_json_cli_arg(
    args: &mut Vec<String>,
    object: &serde_json::Map<String, serde_json::Value>,
    flag: &str,
    key: &str,
) {
    if let Some(value) = json_string(object, key) {
        args.push(flag.to_owned());
        args.push(value);
    }
}

fn json_string(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
}

pub(crate) fn render_examine_text() -> Result<String> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    render_examine_text_with_paths(&paths, &config)
}

fn render_examine_text_with_paths(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    Ok(examine_human_report(paths, config)?.0)
}

/// Build the human examine report and return it alongside the `ExamineSummary`
/// it was built from, so callers can derive scope/exit-code without re-probing.
fn examine_human_report(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<(String, ExamineSummary)> {
    recover_setup_runtime_registration(paths, config)?;
    let summary = ExamineSummary::gather()?;
    let mut output = render_examine_plain_header(&summary);
    output.push_str(&summary.render_text());
    append_examine_runtime_state(&mut output, paths, config)?;
    append_examine_engine_inventory(&mut output, paths, config);
    Ok((output, summary))
}

fn render_examine_plain_header(summary: &ExamineSummary) -> String {
    let gpu = if summary.detected_gfx_target.is_some() {
        "AMD GPU detected"
    } else {
        "AMD GPU not detected yet"
    };
    let runtime = match summary.managed_runtime_count {
        0 => "No ROCm installs saved yet".to_owned(),
        1 => "1 ROCm install saved".to_owned(),
        count => format!("{count} ROCm installs saved"),
    };
    format!(
        "ROCm setup check\n  {gpu}\n  {runtime}\n  Driver: {}\n\nDetails\n",
        plain_status_label(&summary.driver.status)
    )
}

fn plain_status_label(status: &str) -> String {
    status.replace('_', " ")
}

pub(crate) fn render_engine_inventory_text() -> String {
    let paths = AppPaths::discover().ok();
    render_engine_inventory_text_with_paths(paths.as_ref())
}

fn render_engine_inventory_text_with_paths(paths: Option<&AppPaths>) -> String {
    let default_engine = default_engine_for_platform();
    let mut output = String::new();
    let _ = writeln!(output, "Local model engines");
    let _ = writeln!(
        output,
        "  Built-in engines are included with rocm-cli. External plugins are optional."
    );
    let _ = writeln!(output, "  ROCm GPU execution is required.");
    if let Some(paths) = paths {
        let _ = writeln!(output, "  Plugin folders:");
        for (index, path) in engine_plugin_dirs(paths).iter().enumerate() {
            let note = if index == 0 { "primary" } else { "legacy" };
            let _ = writeln!(output, "    {}. {} ({note})", index + 1, path.display());
        }
    } else {
        let _ = writeln!(output, "  Plugin folders: not checked");
    }
    for (name, note) in engine_inventory() {
        let marker = if *name == default_engine { "*" } else { " " };
        let _ = writeln!(output, "{marker} {name:10} {note}");
        append_engine_detect_summary(&mut output, name, paths);
    }
    let _ = writeln!(
        output,
        "  protocol: {}",
        rocm_engine_protocol::ENGINE_PROTOCOL_VERSION
    );
    output
}

fn append_examine_runtime_state(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<()> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let active = current_runtime_manifest(config, &manifests);
    let default_runtime_matches = default_runtime_id_matches(config, &manifests);
    let ambiguous_default_keys =
        if config.active_runtime_key.is_none() && default_runtime_matches.len() > 1 {
            Some(runtime_keys_text(&default_runtime_matches))
        } else {
            None
        };
    let _ = writeln!(output, "runtime_state:");
    let _ = writeln!(
        output,
        "  active_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let active_status = match active {
        Some(manifest) => runtime_usability_status(manifest),
        None if ambiguous_default_keys.is_some() => "ambiguous_runtime_id".to_owned(),
        None if config.active_runtime_key.is_some() || config.default_runtime_id.is_some() => {
            "missing_manifest".to_owned()
        }
        None => "unset".to_owned(),
    };
    let _ = writeln!(output, "  active_runtime_status: {active_status}");
    if let Some(keys) = ambiguous_default_keys {
        let _ = writeln!(output, "  active_runtime_matches: {keys}");
        let _ = writeln!(
            output,
            "  active_runtime_action: rocm runtimes activate <runtime_key>"
        );
    }
    if let Some(manifest) = active {
        let _ = writeln!(
            output,
            "  active_runtime_root: {}",
            manifest.install_root.display()
        );
        let pip_cache_dir = manifest
            .pip_cache_dir
            .clone()
            .unwrap_or_else(|| managed_pip_cache_dir(&manifest.install_root));
        let _ = writeln!(
            output,
            "  active_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
        );
        let _ = writeln!(
            output,
            "  active_runtime_version: {}",
            therock::runtime_version_display(&manifest.version)
        );
        let _ = writeln!(output, "  active_runtime_family: {}", manifest.family);
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        let _ = writeln!(output, "  active_runtime_mode: {mode}");
    }
    if let Some(setup_root) = config.setup.therock_venv.as_deref() {
        let _ = writeln!(output, "  setup_runtime_root: {}", setup_root.display());
        let _ = writeln!(
            output,
            "  setup_runtime_pip_cache_dir: {}",
            managed_pip_cache_dir(setup_root).display()
        );
    }
    let keys = if manifests.is_empty() {
        "<none>".to_owned()
    } else {
        manifests
            .iter()
            .map(|manifest| manifest.runtime_key.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(output, "  registered_runtime_keys: {keys}");
    Ok(())
}

fn append_examine_engine_inventory(output: &mut String, paths: &AppPaths, config: &RocmCliConfig) {
    let configured_default = config.default_engine.as_deref();
    let effective_default = match configured_default {
        Some(engine) => engine,
        None => default_engine_for_platform(),
    };
    let _ = writeln!(output, "engine_inventory:");
    let _ = writeln!(
        output,
        "  configured_default_engine: {}",
        configured_default.unwrap_or("<platform default>")
    );
    let _ = writeln!(output, "  effective_default_engine: {effective_default}");
    let _ = writeln!(
        output,
        "  plugin_policy: first-party engines are built in; external data-dir plugins are optional overrides"
    );
    let _ = writeln!(
        output,
        "  external_plugin_policy: optional overrides only; no fallback engine is selected automatically"
    );
    let _ = writeln!(
        output,
        "  plugin_dirs: {}",
        engine_plugin_dirs(paths)
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (engine, note) in engine_inventory() {
        let marker = if *engine == effective_default {
            "*"
        } else {
            " "
        };
        let adapter = if builtin_engine_available(engine) {
            "built_in".to_owned()
        } else {
            match resolve_engine_binary_path_with_paths(engine, paths) {
                Ok(path) => format!("external path={}", path.display()),
                Err(error) => format!("missing reason={error}"),
            }
        };
        let runtime_pref = config
            .engine_config(engine)
            .and_then(|entry| {
                entry
                    .preferred_runtime_id
                    .as_deref()
                    .or(entry.preferred_env_id.as_deref())
                    .or(entry.last_installed_runtime_id.as_deref())
                    .or(entry.last_installed_env_id.as_deref())
            })
            .unwrap_or("<unset>");
        let _ = writeln!(
            output,
            "  {marker} {engine} adapter={adapter} runtime_pref={runtime_pref} note={note}"
        );
    }
}

#[allow(dead_code)]
pub(crate) fn render_model_registry_text_with_context_and_host(
    _paths: Option<&AppPaths>,
    aggregate_gpu_vram_gib: Option<f64>,
    _host_ram_gib: Option<f64>,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Local models");
    let registry = match load_model_recipe_registry() {
        Ok(registry) => registry,
        Err(error) => {
            let _ = writeln!(output, "  Model list is unavailable: {error}");
            return output;
        }
    };
    for recipe in &registry.recipes {
        let _ = writeln!(
            output,
            "  {}  {}  {}",
            recipe_display_ref(recipe),
            model_recipe_memory_label(recipe),
            model_recipe_gpu_fit_label(recipe, aggregate_gpu_vram_gib)
        );
    }
    let _ = writeln!(
        output,
        "\nUse `rocm serve <model>` to start one. Use `rocm model --verbose` for details."
    );
    output
}

fn model_recipe_memory_label(recipe: &ModelRecipeRecord) -> String {
    recipe.min_gpu_mem_gb.map_or_else(
        || "no GPU minimum".to_owned(),
        |value| format!("{value} GiB GPU"),
    )
}

fn model_recipe_gpu_fit_label(
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) -> &'static str {
    match (recipe.min_gpu_mem_gb, aggregate_gpu_vram_gib) {
        (Some(required), Some(available)) if available >= f64::from(required) => "fits this GPU",
        (Some(_), Some(_)) => "needs a larger GPU",
        (Some(_), None) => "GPU fit unknown",
        (None, _) => "fits",
    }
}

pub(crate) fn render_model_registry_verbose_text_with_context_and_host(
    paths: Option<&AppPaths>,
    aggregate_gpu_vram_gib: Option<f64>,
    host_ram_gib: Option<f64>,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "model recipes");
    let registry = match load_model_recipe_registry() {
        Ok(registry) => registry,
        Err(error) => {
            let _ = writeln!(output, "  source_error: {error}");
            let _ = writeln!(
                output,
                "  source_action: configure or fix the signed recipe index before using /model"
            );
            return output;
        }
    };
    for recipe in &registry.recipes {
        let aliases = if recipe.aliases.is_empty() {
            "<none>".to_owned()
        } else {
            recipe.aliases.join(", ")
        };
        let engines = if recipe.preferred_engines.is_empty() {
            "<none>".to_owned()
        } else {
            recipe.preferred_engines.join(", ")
        };
        let memory = recipe.min_gpu_mem_gb.map_or_else(
            || "<not required>".to_owned(),
            |value| format!("{value} GiB"),
        );
        let _ = writeln!(
            output,
            "  {} aliases=[{}] task={} dtype={} device={} min_gpu_mem={} engines=[{}]",
            recipe.canonical_model_id,
            aliases,
            recipe.task,
            recipe.dtype,
            recipe.device_policy,
            memory,
            engines
        );
        append_model_recipe_metadata_lines(&mut output, recipe, paths);
        append_model_host_ram_fit_lines(&mut output, recipe, host_ram_gib);
        append_model_fit_lines(&mut output, recipe, aggregate_gpu_vram_gib);
        append_model_engine_support_lines(&mut output, recipe, paths);
        if recipe.trust_remote_code {
            let _ = writeln!(output, "      trust_remote_code: true");
        }
        for warning in &recipe.warnings {
            let _ = writeln!(output, "      warning: {warning}");
        }
    }
    append_model_recipe_registry_source(&mut output, &registry);
    output
}

#[allow(dead_code)]
fn append_model_recipe_registry_source(output: &mut String, registry: &ModelRecipeRegistry) {
    match &registry.source {
        ModelRecipeRegistrySource::BuiltIn => {
            let _ = writeln!(
                output,
                "  source: built-in recipe registry; external signed recipe index is not configured yet"
            );
        }
        ModelRecipeRegistrySource::SignedIndex {
            index_path,
            signature_path,
            public_key_path,
        } => {
            let _ = writeln!(
                output,
                "  source: signed model recipe index path={} signature={} public_key={}",
                index_path.display(),
                signature_path.display(),
                public_key_path.display()
            );
        }
    }
}

#[allow(dead_code)]
fn append_model_recipe_metadata_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
) {
    let _ = writeln!(
        output,
        "      recommended_system_ram: {}",
        recipe
            .recommended_system_ram_gb
            .map_or_else(|| "<unknown>".to_owned(), |value| format!("{value} GiB"))
    );
    let _ = writeln!(
        output,
        "      quantization: {}",
        recipe.quantization.as_deref().unwrap_or("<unspecified>")
    );
    append_model_engine_recipe_settings_lines(output, recipe);
    append_model_artifact_lines(output, recipe, paths);
}

#[allow(dead_code)]
fn append_model_engine_recipe_settings_lines(output: &mut String, recipe: &ModelRecipeRecord) {
    if recipe.engine_recipes.is_empty() {
        let _ = writeln!(output, "      engine_recipes: <none>");
        return;
    }
    let _ = writeln!(
        output,
        "      engine_recipes_policy: protocol_contract={ENGINE_RECIPE_CONTRACT_VERSION} selected-engine hint is passed to adapters during model resolution and required flags are forwarded at launch"
    );
    for engine_recipe in &recipe.engine_recipes {
        let required_flags = format_list_or_none(&engine_recipe.required_flags);
        let parser_settings = format_string_map_or_none(&engine_recipe.parser_settings);
        let endpoint = engine_recipe.preferred_endpoint.as_ref().map_or_else(
            || "<none>".to_owned(),
            |endpoint| {
                let settings = format_string_map_or_none(&endpoint.settings);
                format!("mode={} settings=[{}]", endpoint.endpoint_mode, settings)
            },
        );
        let unsupported = if engine_recipe.unsupported_combinations.is_empty() {
            "<none>".to_owned()
        } else {
            engine_recipe
                .unsupported_combinations
                .iter()
                .map(|item| format!("{} ({})", item.combination, item.reason))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let notes = format_list_or_none(&engine_recipe.notes);
        let _ = writeln!(
            output,
            "      engine_recipe {} required_flags=[{}] parser_settings=[{}] preferred_endpoint={} unsupported_combinations=[{}] notes=[{}]",
            engine_recipe.engine, required_flags, parser_settings, endpoint, unsupported, notes
        );
    }
}

#[allow(dead_code)]
fn format_list_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "<none>".to_owned()
    } else {
        values.join(",")
    }
}

#[allow(dead_code)]
fn format_string_map_or_none(values: &BTreeMap<String, String>) -> String {
    if values.is_empty() {
        "<none>".to_owned()
    } else {
        values
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[allow(dead_code)]
fn append_model_artifact_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
) {
    if recipe.artifacts.is_empty() {
        let _ = writeln!(output, "      artifact_check: not_checked");
        let _ = writeln!(
            output,
            "      artifact_reason: {}",
            recipe
                .artifact_hint
                .as_deref()
                .unwrap_or("recipe does not declare artifact requirements")
        );
        return;
    }

    let gated = recipe
        .artifacts
        .iter()
        .any(|artifact| artifact.gated.unwrap_or(false));
    let artifact_check = if gated {
        "blocked"
    } else {
        "metadata_available"
    };
    let artifact_reason = if gated {
        "signed recipe index declares one or more gated artifacts"
    } else {
        "signed recipe index declares artifact metadata"
    };
    let _ = writeln!(output, "      artifact_check: {artifact_check}");
    let _ = writeln!(
        output,
        "      artifact_reason: {artifact_reason}; this is not a live availability or cache check"
    );
    let _ = writeln!(output, "      artifact_count: {}", recipe.artifacts.len());
    for artifact in &recipe.artifacts {
        let engines = if artifact.engines.is_empty() {
            "<unspecified>".to_owned()
        } else {
            artifact.engines.join(",")
        };
        let size = artifact
            .size_bytes
            .map_or_else(|| "<unknown>".to_owned(), format_bytes);
        let gated = artifact.gated.unwrap_or(false);
        let _ = writeln!(
            output,
            "      artifact {} kind={} uri={} revision={} size={} sha256={} license={} gated={} quantization={} engines=[{}]",
            artifact.artifact_id,
            artifact.kind,
            artifact.uri,
            artifact.revision.as_deref().unwrap_or("<unspecified>"),
            size,
            artifact.sha256.as_deref().unwrap_or("<unspecified>"),
            artifact.license.as_deref().unwrap_or("<unspecified>"),
            gated,
            artifact.quantization.as_deref().unwrap_or("<unspecified>"),
            engines
        );
        if let Some(source_policy) = &artifact.source_policy {
            append_artifact_source_policy_lines(output, source_policy);
        }
        if let Some(paths) = paths {
            let cache = model_artifact_cache_status(paths, &recipe.canonical_model_id, artifact);
            let _ = writeln!(
                output,
                "      artifact_cache {} status={} marker={} reason={}",
                cache.artifact_id,
                cache.status,
                cache.marker_path.display(),
                cache.reason
            );
        } else {
            let _ = writeln!(
                output,
                "      artifact_cache {} status=unknown marker=<unavailable> reason=app paths unavailable; no live cache check performed",
                artifact.artifact_id
            );
        }
    }
}

fn append_artifact_source_policy_lines(
    output: &mut String,
    source_policy: &rocm_core::ModelRecipeArtifactSourcePolicyRecord,
) {
    let _ = writeln!(
        output,
        "      download rule: {}",
        artifact_source_policy_label(&source_policy.policy)
    );
    for host in &source_policy.required_hosts {
        let _ = writeln!(output, "        allowed site: {host}");
    }
    for note in &source_policy.notes {
        let _ = writeln!(output, "        note: {note}");
    }
}

fn artifact_source_policy_label(policy: &str) -> &str {
    match policy {
        "direct_https_sha256" => "Direct HTTPS download with checksum",
        "huggingface_public" => "Public Hugging Face download",
        "huggingface_authenticated" => "Hugging Face download, token required",
        "manual_only" => "Manual download only",
        _ => "Unknown download rule",
    }
}

#[allow(dead_code)]
fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else {
        format!("{bytes} bytes")
    }
}

#[allow(dead_code)]
fn append_model_host_ram_fit_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    host_ram_gib: Option<f64>,
) {
    let _ = writeln!(output, "      system_ram_policy: advisory");
    let Some(required) = recipe.recommended_system_ram_gb else {
        let _ = writeln!(output, "      system_ram_fit: unknown");
        let _ = writeln!(
            output,
            "      system_ram_reason: recipe does not declare a RAM recommendation"
        );
        return;
    };

    match host_ram_gib {
        Some(available) if available >= f64::from(required) => {
            let _ = writeln!(output, "      system_ram_fit: supported");
            let _ = writeln!(
                output,
                "      system_ram_reason: host RAM {} meets recipe recommendation {}",
                format_gib(available),
                format_gib(f64::from(required))
            );
        }
        Some(available) => {
            let _ = writeln!(output, "      system_ram_fit: below_recommendation");
            let _ = writeln!(
                output,
                "      system_ram_reason: host RAM {} is below recipe recommendation {}",
                format_gib(available),
                format_gib(f64::from(required))
            );
            let _ = writeln!(
                output,
                "      system_ram_action: consider a smaller recipe or a host with at least {} system RAM for smoother serving",
                format_gib(f64::from(required))
            );
        }
        None => {
            let _ = writeln!(output, "      system_ram_fit: unknown");
            let _ = writeln!(
                output,
                "      system_ram_reason: current host RAM telemetry is unavailable"
            );
            let _ = writeln!(
                output,
                "      system_ram_action: run /examine to refresh host telemetry"
            );
        }
    }
}

#[allow(dead_code)]
fn append_model_fit_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) {
    match recipe.min_gpu_mem_gb {
        None => {
            let _ = writeln!(output, "      gpu_fit: supported");
            let _ = writeln!(
                output,
                "      reason: recipe device policy `{}` does not require GPU VRAM",
                recipe.device_policy
            );
            let _ = writeln!(
                output,
                "      action: use /plan serve {} with {}",
                recipe_display_ref(recipe),
                preferred_engine_action_target(recipe)
            );
        }
        Some(required) => match aggregate_gpu_vram_gib {
            Some(available) if available >= f64::from(required) => {
                let _ = writeln!(output, "      gpu_fit: supported");
                let _ = writeln!(
                    output,
                    "      reason: aggregate GPU VRAM {} meets recipe minimum {}",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                let _ = writeln!(
                    output,
                    "      action: use /plan serve {} with {}",
                    recipe_display_ref(recipe),
                    preferred_engine_action_target(recipe)
                );
            }
            Some(available) => {
                let _ = writeln!(output, "      gpu_fit: unsupported");
                let _ = writeln!(
                    output,
                    "      reason: aggregate GPU VRAM {} is below recipe minimum {}",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                let _ = writeln!(
                    output,
                    "      action: choose a recipe with min_gpu_mem <= {} or use a GPU with at least {} before serving",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                append_manual_alternative_lines(output, recipe, aggregate_gpu_vram_gib);
            }
            None => {
                let _ = writeln!(output, "      gpu_fit: unknown");
                let _ = writeln!(
                    output,
                    "      reason: current telemetry has no aggregate GPU VRAM reading"
                );
                let _ = writeln!(
                    output,
                    "      action: run /examine or refresh GPU telemetry, then retry /model {}",
                    recipe_display_ref(recipe)
                );
            }
        },
    }
}

#[allow(dead_code)]
fn append_manual_alternative_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) {
    let alternatives = manual_alternative_recommendations(recipe, aggregate_gpu_vram_gib);
    if alternatives.is_empty() {
        let _ = writeln!(output, "      manual_alternatives: <none declared>");
    } else {
        let _ = writeln!(
            output,
            "      manual_alternatives: {}",
            alternatives.join(", ")
        );
        let _ = writeln!(
            output,
            "      manual_alternative_policy: user must choose one explicitly; none is selected automatically"
        );
    }
}

#[allow(dead_code)]
fn manual_alternative_recommendations(
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) -> Vec<String> {
    let declared = recipe
        .manual_alternatives
        .iter()
        .filter_map(|candidate_ref| {
            resolve_builtin_model_recipe(candidate_ref).map(|candidate| (candidate_ref, candidate))
        })
        .filter(|(_, candidate)| recipe_is_manual_fit(candidate, aggregate_gpu_vram_gib))
        .map(|(candidate_ref, candidate)| {
            format!(
                "{} ({})",
                candidate_ref,
                candidate.min_gpu_mem_gb.map_or_else(
                    || "CPU-only".to_owned(),
                    |value| format!("{} min GPU", format_gib(f64::from(value)))
                )
            )
        })
        .collect::<Vec<_>>();
    if !declared.is_empty() {
        return declared;
    }
    builtin_model_recipes()
        .into_iter()
        .filter(|candidate| candidate.canonical_model_id != recipe.canonical_model_id)
        .filter(|candidate| candidate.task == recipe.task)
        .filter(|candidate| recipe_is_manual_fit(candidate, aggregate_gpu_vram_gib))
        .take(3)
        .map(|candidate| recipe_display_ref(&candidate).to_owned())
        .collect()
}

#[allow(dead_code)]
fn recipe_is_manual_fit(recipe: &ModelRecipeRecord, aggregate_gpu_vram_gib: Option<f64>) -> bool {
    match recipe.min_gpu_mem_gb {
        None => true,
        Some(required) => {
            aggregate_gpu_vram_gib.is_some_and(|available| available >= f64::from(required))
        }
    }
}

#[allow(dead_code)]
fn append_model_engine_support_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
) {
    if recipe.preferred_engines.is_empty() {
        let _ = writeln!(output, "      engine_support: unknown");
        let _ = writeln!(
            output,
            "      engine_action: add a preferred engine to the signed recipe index before serving"
        );
        return;
    }

    let _ = writeln!(output, "      engine_support:");
    for engine in &recipe.preferred_engines {
        if builtin_engine_available(engine) {
            if let Some(note) = model_registry_adapter_availability_note(engine) {
                let _ = writeln!(
                    output,
                    "        {engine}: adapter_available path=<built-in> {note}"
                );
            } else {
                let _ = writeln!(output, "        {engine}: built_in");
            }
        } else if let Some(paths) = paths {
            match resolve_engine_binary_path_with_paths(engine, paths) {
                Ok(path) => {
                    if let Some(note) = model_registry_adapter_availability_note(engine) {
                        let _ = writeln!(
                            output,
                            "        {engine}: adapter_available path={} {note}",
                            path.display()
                        );
                    } else {
                        let _ = writeln!(
                            output,
                            "        {engine}: available path={}",
                            path.display()
                        );
                    }
                }
                Err(error) => {
                    let _ = writeln!(
                        output,
                        "        {engine}: unavailable reason={}",
                        model_registry_reason(error.to_string())
                    );
                }
            }
        } else if let Some(reason) = missing_packaged_engine_reason(engine) {
            let _ = writeln!(
                output,
                "        {engine}: unavailable reason={}",
                model_registry_reason(reason)
            );
        } else {
            let _ = writeln!(output, "        {engine}: not_checked");
        }
    }
    let _ = writeln!(
        output,
        "      engine_action: use /engine install <engine> for an unavailable preferred engine, or select an available listed engine explicitly; approval is still required before install or serve"
    );
}

#[allow(dead_code)]
const fn model_registry_adapter_availability_note(engine: &str) -> Option<&'static str> {
    if rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("vllm") {
        Some(
            "runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm; gpu_execution_required=true; run /engine for adapter details",
        )
    } else if rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("sglang") {
        Some(
            "runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux SGLang ROCm; gpu_execution_required=true; run /engine for adapter details",
        )
    } else if rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("atom") {
        Some(
            "runtime_status=unsupported_native_windows reason=use WSL/Linux ATOM ROCm; gpu_execution_required=true; run /engine for adapter details",
        )
    } else {
        None
    }
}

#[allow(dead_code)]
fn recipe_display_ref(recipe: &ModelRecipeRecord) -> &str {
    recipe
        .aliases
        .first()
        .map_or(recipe.canonical_model_id.as_str(), String::as_str)
}

#[allow(dead_code)]
fn preferred_engine_action_target(recipe: &ModelRecipeRecord) -> &str {
    recipe
        .preferred_engines
        .first()
        .map_or("<engine>", String::as_str)
}

#[allow(dead_code)]
fn model_registry_reason(reason: String) -> String {
    reason
        .replace(
            " No CPU fallback is used.",
            " Serving stays blocked until a matching GPU engine adapter is installed.",
        )
        .replace(
            " No fallback engine is used.",
            " Serving stays blocked until the requested engine adapter is installed.",
        )
}

#[allow(dead_code)]
fn format_gib(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

fn append_engine_detect_summary(output: &mut String, engine: &str, paths: Option<&AppPaths>) {
    let engine_binary = if builtin_engine_available(engine) {
        Ok(PathBuf::new())
    } else {
        match paths {
            Some(paths) => resolve_engine_binary_path_with_paths(engine, paths),
            None => resolve_engine_binary_path(engine),
        }
    };
    let binary_status = if builtin_engine_available(engine) {
        "adapter: built-in".to_owned()
    } else {
        match &engine_binary {
            Ok(_) => "adapter: available".to_owned(),
            Err(_) => "adapter: not found".to_owned(),
        }
    };
    let _ = writeln!(output, "    {binary_status}");
    if !builtin_engine_available(engine)
        && engine_binary.is_err()
        && let Some(reason) = missing_packaged_engine_reason(engine)
    {
        let _ = writeln!(output, "    note: {reason}");
        return;
    }

    let Ok(detect) = engine_request::<_, DetectResponse>(
        paths,
        engine,
        EngineMethod::Detect,
        &DetectRequest {
            runtime_id: None,
            device_filter: None,
        },
    ) else {
        return;
    };

    let _ = writeln!(
        output,
        "    runtime: {}",
        engine_runtime_status_label(engine, &detect)
    );
    if let Some(kind) = detect.runtime_kind.as_deref() {
        let _ = writeln!(
            output,
            "    runtime kind: {}",
            friendly_engine_runtime_kind(kind)
        );
    }
    if detect.runtime_executable.is_some() {
        let _ = writeln!(output, "    runtime executable: available");
    }
    if let Some(note) = friendly_engine_detect_notes(engine, &detect.notes) {
        let _ = writeln!(output, "    note: {note}");
    }
}

fn friendly_engine_runtime_kind(kind: &str) -> String {
    kind.replace('_', " ")
}

fn friendly_engine_detect_notes(engine: &str, notes: &[String]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }
    let combined = notes.join("; ");
    let lower = combined.to_ascii_lowercase();
    if engine.eq_ignore_ascii_case("pytorch") {
        if lower.contains("torch probe failed") || lower.contains("torch probe import failed") {
            return Some("PyTorch engine is not ready yet; reinstall the PyTorch engine from the TUI or run `rocm engines install pytorch`.".to_owned());
        }
        if lower.contains("no managed pytorch envs found") {
            return Some("PyTorch engine is not installed yet.".to_owned());
        }
        if lower.contains("torch probe:")
            || lower.contains("rocm_sdk:")
            || lower.contains("torch._rocm_init")
            || lower.contains("managed env detected")
        {
            return Some("PyTorch is ready on your AMD GPU.".to_owned());
        }
    }
    if engine.eq_ignore_ascii_case("lemonade") {
        if lower.contains("not installed") || lower.contains("not found") {
            return Some("Lemonade is not installed yet.".to_owned());
        }
        if lower.contains("lemonade embeddable")
            || lower.contains("llamacpp:rocm")
            || lower.contains("configured")
        {
            return Some("Lemonade is ready on your AMD GPU.".to_owned());
        }
    }
    if engine.eq_ignore_ascii_case("llama.cpp") {
        if lower.contains("llama-server not found") {
            return Some("llama.cpp server is not installed yet.".to_owned());
        }
        if lower.contains("hip runtime env available") || lower.contains("therock") {
            return Some("llama.cpp can use the active ROCm install.".to_owned());
        }
    }
    if engine.eq_ignore_ascii_case("vllm")
        && (lower.contains("not installed") || lower.contains("command was not found"))
    {
        return Some("vLLM is not installed in a Linux/WSL ROCm Python environment.".to_owned());
    }
    if engine.eq_ignore_ascii_case("sglang")
        && (lower.contains("not installed") || lower.contains("command was not found"))
    {
        return Some("SGLang is not installed in a Linux/WSL ROCm Python environment.".to_owned());
    }
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return Some(format!("{engine} is available from WSL/Linux on Windows."));
    }
    Some(friendly_engine_detect_note_fallback(&combined))
}

fn friendly_engine_detect_note_fallback(note: &str) -> String {
    let lower = note.to_ascii_lowercase();
    if lower.contains("torch probe failed") || lower.contains("torch probe import failed") {
        return "PyTorch engine is not ready yet; reinstall the PyTorch engine from the TUI or run `rocm engines install pytorch`.".to_owned();
    }
    if lower.contains("no managed pytorch envs found") {
        return "PyTorch engine is not installed yet.".to_owned();
    }
    if lower.contains("torch probe:")
        || lower.contains("rocm_sdk:")
        || lower.contains("torch._rocm_init")
    {
        return "PyTorch is ready on your AMD GPU.".to_owned();
    }
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return "This engine is available from WSL/Linux on Windows.".to_owned();
    }
    note.to_owned()
}

fn engine_runtime_status_label(engine: &str, detect: &DetectResponse) -> &'static str {
    if detect.installed {
        "ready"
    } else if engine_runtime_is_native_windows_unsupported(engine, detect) {
        "unsupported_native_windows"
    } else {
        "not found"
    }
}

fn engine_runtime_is_native_windows_unsupported(engine: &str, detect: &DetectResponse) -> bool {
    if !rocm_core::runtime_is_windows()
        || !(engine.eq_ignore_ascii_case("vllm")
            || engine.eq_ignore_ascii_case("sglang")
            || engine.eq_ignore_ascii_case("atom"))
    {
        return false;
    }

    detect
        .notes
        .iter()
        .chain(
            detect
                .available_devices
                .iter()
                .filter_map(|device| device.reason.as_ref()),
        )
        .any(|note| {
            let normalized = note.to_ascii_lowercase();
            normalized.contains("linux/wsl") && normalized.contains("native windows")
        })
}

pub(crate) fn render_config_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "rocm config");
    let _ = writeln!(output, "  file: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  default_engine: {}",
        config
            .default_engine
            .as_deref()
            .unwrap_or("<platform default>")
    );
    let _ = writeln!(
        output,
        "  default_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  onboarding_dismissed: {}",
        config.onboarding_dismissed
    );
    let _ = writeln!(
        output,
        "  telemetry_mode: {}",
        config.telemetry.mode_label()
    );
    let _ = writeln!(
        output,
        "  telemetry_policy: {}",
        telemetry_policy_summary(&config.telemetry)
    );
    let _ = writeln!(
        output,
        "  planner_provider: {}",
        config.planner_provider.as_deref().unwrap_or("<off>")
    );
    let _ = writeln!(output, "  providers:");
    for provider in ["local", "openai", "anthropic"] {
        let key_status = providers::provider_key_status_text(provider)
            .unwrap_or_else(|error| format!("key status unavailable: {error}"));
        let _ = writeln!(
            output,
            "    {provider}: {}",
            if config.provider_enabled(provider) {
                "enabled"
            } else {
                "disabled"
            }
        );
        let _ = writeln!(output, "      key: {key_status}");
    }
    if config.engines.is_empty() {
        let _ = writeln!(output, "  engines: none");
        return output;
    }
    for (engine, entry) in &config.engines {
        let _ = writeln!(output, "  engine: {engine}");
        let _ = writeln!(
            output,
            "    preferred_runtime_id: {}",
            entry.preferred_runtime_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    preferred_env_id: {}",
            entry.preferred_env_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_runtime_id: {}",
            entry
                .last_installed_runtime_id
                .as_deref()
                .unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_env_id: {}",
            entry.last_installed_env_id.as_deref().unwrap_or("<unset>")
        );
    }
    output
}

fn telemetry_policy_summary(telemetry: &rocm_core::TelemetryConfig) -> &'static str {
    if telemetry.local_inspection_enabled() {
        "local amd-smi inspection only; no external reporting is implemented"
    } else if telemetry.known_mode() {
        "disabled; no local polling or external reporting is implemented"
    } else {
        "unknown mode treated as disabled; set `rocm config set-telemetry local|off`"
    }
}

pub(crate) fn render_logs_text(paths: &AppPaths) -> String {
    render_logs_browser_text(paths, None)
}

pub(crate) fn render_logs_browser_text(paths: &AppPaths, query: Option<&str>) -> String {
    render_logs_browser_page_text(paths, query, 0, 24)
}

pub(crate) fn render_logs_browser_page_text(
    paths: &AppPaths,
    query: Option<&str>,
    page: usize,
    page_size: usize,
) -> String {
    render_logs_browser_page_text_with_options(paths, query, page, page_size, true)
}

pub(crate) fn render_logs_browser_page_text_for_tui(
    paths: &AppPaths,
    query: Option<&str>,
    page: usize,
    page_size: usize,
    show_file_locations: bool,
) -> String {
    render_logs_browser_page_text_with_options(paths, query, page, page_size, show_file_locations)
}

fn render_logs_browser_page_text_with_options(
    paths: &AppPaths,
    query: Option<&str>,
    page: usize,
    page_size: usize,
    show_file_locations: bool,
) -> String {
    let mut output = String::new();
    let lifecycle_path = cli_lifecycle_log_path(paths);
    let action_dir = paths.data_dir.join("logs").join("cli");
    let screen_dir = paths.data_dir.join("logs").join("tui");
    let query = query.map(str::trim).filter(|value| !value.is_empty());
    let page_size = page_size.max(1);
    let _ = writeln!(output, "Logs");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "File locations: {}",
        if show_file_locations {
            "shown"
        } else {
            "hidden"
        }
    );
    if show_file_locations {
        let _ = writeln!(
            output,
            "  Folder: {}",
            paths.data_dir.join("logs").display()
        );
        let _ = writeln!(output, "  Activity log: {}", lifecycle_path.display());
        let _ = writeln!(output, "  Command logs: {}", action_dir.display());
        let _ = writeln!(output, "  Screen command logs: {}", screen_dir.display());
        let _ = writeln!(
            output,
            "  Audit events: {}",
            paths.audit_events_path().display()
        );
    } else {
        let _ = writeln!(output, "  Choose Show file locations to see exact paths.");
    }
    let action_logs = list_log_files(&action_dir, 12);
    let screen_logs = list_log_files(&screen_dir, 12);
    if show_file_locations {
        if action_logs.is_empty() && screen_logs.is_empty() {
            let _ = writeln!(output, "  Recent command files: none yet");
        } else {
            let _ = writeln!(output, "  Recent command files:");
            for log in action_logs {
                let display = log.strip_prefix(&action_dir).unwrap_or(&log);
                let _ = writeln!(output, "    cli/{}", display.display());
            }
            for log in screen_logs {
                let display = log.strip_prefix(&screen_dir).unwrap_or(&log);
                let _ = writeln!(output, "    screen/{}", display.display());
            }
        }
    }
    let _ = writeln!(output);
    if query.is_some() {
        let _ = writeln!(output, "Recent activity: filtered by search");
    } else {
        let recent_lines = read_optional_tail_lines(&lifecycle_path, 8, "CLI lifecycle log");
        if recent_lines.is_empty() {
            let _ = writeln!(output, "Recent activity: no activity yet");
        } else {
            let _ = writeln!(
                output,
                "Recent activity: last {} line(s)",
                recent_lines.len()
            );
            for line in recent_lines {
                let _ = writeln!(output, "  {}", format_cli_lifecycle_tail_line(&line));
            }
        }
    }

    let entries = collect_log_browser_entries(paths);
    let matching_entries = filter_log_browser_entries(&entries, query);
    let total_pages = logs_browser_total_pages(matching_entries.len(), page_size);
    let page = page.min(total_pages.saturating_sub(1));
    let start = page.saturating_mul(page_size);
    let end = start.saturating_add(page_size).min(matching_entries.len());
    let _ = writeln!(output);
    let _ = writeln!(output, "Matching lines");
    let _ = writeln!(output, "  Search: {}", query.unwrap_or("none"));
    let _ = writeln!(
        output,
        "  Lines: {} of {} recent line(s)",
        matching_entries.len(),
        entries.len()
    );
    let _ = writeln!(output, "  Page: {} of {}", page + 1, total_pages);
    if matching_entries.is_empty() {
        let _ = writeln!(output, "  Showing: 0 of 0");
    } else {
        let _ = writeln!(
            output,
            "  Showing: {}-{} of {}",
            start + 1,
            end,
            matching_entries.len()
        );
    }
    if entries.is_empty() {
        let _ = writeln!(output, "  No logs found yet.");
    } else if matching_entries.is_empty() {
        let _ = writeln!(output, "  No matching lines.");
    } else {
        let _ = writeln!(output, "  Lines:");
        for entry in matching_entries.into_iter().skip(start).take(page_size) {
            let _ = writeln!(
                output,
                "    {}: {}",
                log_browser_source_label(&entry.source, show_file_locations),
                entry.line
            );
        }
    }
    output
}

pub(crate) fn logs_browser_page_count(
    paths: &AppPaths,
    query: Option<&str>,
    page_size: usize,
) -> usize {
    let entries = collect_log_browser_entries(paths);
    let matching_entries = filter_log_browser_entries(&entries, query);
    logs_browser_total_pages(matching_entries.len(), page_size.max(1))
}

fn logs_browser_total_pages(item_count: usize, page_size: usize) -> usize {
    item_count.div_ceil(page_size).max(1)
}

#[derive(Debug, Clone)]
struct LogBrowserEntry {
    source: String,
    line: String,
}

fn collect_log_browser_entries(paths: &AppPaths) -> Vec<LogBrowserEntry> {
    let lifecycle_path = cli_lifecycle_log_path(paths);
    let mut entries = Vec::new();
    for line in read_optional_tail_lines(&lifecycle_path, 8, "CLI lifecycle log") {
        entries.push(LogBrowserEntry {
            source: "lifecycle".to_owned(),
            line: format_cli_lifecycle_tail_line(&line),
        });
    }

    let action_dir = paths.data_dir.join("logs").join("cli");
    for path in list_log_files(&action_dir, 12) {
        let display = path
            .strip_prefix(&action_dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        for line in read_optional_tail_lines(&path, 12, "CLI action log") {
            entries.push(LogBrowserEntry {
                source: format!("action/{display}"),
                line,
            });
        }
    }

    let screen_dir = paths.data_dir.join("logs").join("tui");
    for path in list_log_files(&screen_dir, 12) {
        let display = path
            .strip_prefix(&screen_dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        for line in read_optional_tail_lines(&path, 12, "screen command log") {
            entries.push(LogBrowserEntry {
                source: format!("screen/{display}"),
                line,
            });
        }
    }
    entries
}

fn filter_log_browser_entries<'a>(
    entries: &'a [LogBrowserEntry],
    query: Option<&str>,
) -> Vec<&'a LogBrowserEntry> {
    let Some(query) = query else {
        return entries.iter().collect();
    };
    let query = query.to_ascii_lowercase();
    entries
        .iter()
        .filter(|entry| {
            entry.source.to_ascii_lowercase().contains(&query)
                || entry.line.to_ascii_lowercase().contains(&query)
        })
        .collect()
}

fn log_browser_source_label(source: &str, show_file_locations: bool) -> String {
    if source == "lifecycle" {
        "recent activity".to_owned()
    } else if let Some(path) = source.strip_prefix("action/") {
        if show_file_locations {
            format!("command log {path}")
        } else {
            "command output".to_owned()
        }
    } else if let Some(path) = source.strip_prefix("screen/") {
        if show_file_locations {
            format!("screen command log {path}")
        } else {
            "screen command output".to_owned()
        }
    } else {
        source.to_owned()
    }
}

pub(crate) fn render_services_text(paths: &AppPaths, all: bool) -> Result<String> {
    let records = load_managed_services(paths)?
        .into_iter()
        .filter(|record| all || managed_service_is_live(record))
        .collect::<Vec<_>>();
    let counts = managed_service_sidebar_counts(&records);
    let mut output = String::new();
    let _ = writeln!(output, "Local Servers");
    let _ = writeln!(output);
    let _ = writeln!(output, "Status: {}", local_server_sidebar_status(&counts));
    let _ = writeln!(output);
    if records.is_empty() {
        let _ = if all {
            writeln!(output, "No local server records yet.")
        } else {
            writeln!(output, "No local servers are running.")
        };
        let _ = writeln!(
            output,
            "Start one with `rocm serve <model> --managed`, or run `rocm` and choose Serve."
        );
        return Ok(output);
    }

    let _ = writeln!(output, "Servers");
    for record in records {
        let _ = writeln!(output, "- {}", record.service_id);
        let _ = writeln!(output, "  status: {}", record.status);
        let _ = writeln!(output, "  engine: {}", record.engine);
        let _ = writeln!(output, "  model: {}", record.model_ref);
        let _ = writeln!(output, "  endpoint: {}", record.endpoint_url);
        let _ = writeln!(output, "  logs: rocm services logs {}", record.service_id);
        if matches!(
            record.status.as_str(),
            "ready" | "running" | "starting" | "recovering"
        ) {
            let _ = writeln!(
                output,
                "  stop: rocm services stop {} --yes",
                record.service_id
            );
        } else {
            let _ = writeln!(
                output,
                "  restart: rocm services restart {} --yes",
                record.service_id
            );
        }
    }
    Ok(output)
}

fn render_services_tool_result_text(records: &[ManagedServiceRecord]) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "managed_services: {}", records.len());
    let _ = writeln!(
        output,
        "status_meaning: ready/running = running; starting/recovering = starting; failed/stopped = not running; no matching row = not managed by ROCm CLI"
    );
    if records.is_empty() {
        let _ = writeln!(output, "services: none");
        return output;
    }
    let _ = writeln!(output, "services:");
    for record in records {
        let _ = writeln!(
            output,
            "  - service_id={} engine={} model={} canonical_model={} status={} running_state={} endpoint={}",
            record.service_id,
            record.engine,
            record.model_ref,
            record.canonical_model_id,
            record.status,
            managed_service_running_state(&record.status),
            record.endpoint_url
        );
    }
    output
}

pub(crate) fn render_service_logs_text(paths: &AppPaths, service_id: &str) -> Result<String> {
    render_service_logs_text_with_options(paths, service_id, true)
}

pub(crate) fn render_service_logs_text_for_tui(
    paths: &AppPaths,
    service_id: &str,
    show_file_locations: bool,
) -> Result<String> {
    render_service_logs_text_with_options(paths, service_id, show_file_locations)
}

fn render_service_logs_text_with_options(
    paths: &AppPaths,
    service_id: &str,
    show_file_locations: bool,
) -> Result<String> {
    let record = load_managed_service(paths, service_id)?;
    let recent_lines = read_tail_lines(&record.log_path, DEFAULT_LOG_TAIL_LINES, "service log")?;

    let mut output = String::new();
    let _ = writeln!(output, "Service Log");
    let _ = writeln!(output);
    let _ = writeln!(output, "Service: {}", record.service_id);
    let _ = writeln!(output, "Engine: {}", record.engine);
    let _ = writeln!(output, "Status: {}", record.status);
    let _ = writeln!(output, "Endpoint: {}", record.endpoint_url);
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "File locations: {}",
        if show_file_locations {
            "shown"
        } else {
            "hidden"
        }
    );
    if show_file_locations {
        let _ = writeln!(output, "  Details file: {}", record.manifest_path.display());
        let _ = writeln!(output, "  Log file: {}", record.log_path.display());
    } else {
        let _ = writeln!(output, "  Choose Show file locations to see exact paths.");
    }
    let _ = writeln!(output);
    if recent_lines.is_empty() {
        let _ = writeln!(output, "Recent output: no output yet");
    } else {
        let _ = writeln!(output, "Recent output: last {} line(s)", recent_lines.len());
        for line in recent_lines {
            let _ = writeln!(output, "  {line}");
        }
    }
    Ok(output)
}

fn render_service_action_result(tool: &str, value: &serde_json::Value) -> String {
    let output = value.get("output").unwrap_or(value);
    let action = service_action_past_tense(tool);
    let service = output.get("service").or_else(|| {
        output
            .get("result")
            .and_then(|result| result.get("service"))
    });
    let mut text = String::new();
    let _ = writeln!(text, "Local server {action}");
    if let Some(service) = service {
        if let Some(service_id) = service
            .get("service_id")
            .and_then(serde_json::Value::as_str)
        {
            let _ = writeln!(text, "  service: {service_id}");
        }
        if let Some(status) = service.get("status").and_then(serde_json::Value::as_str) {
            let _ = writeln!(text, "  status: {status}");
        }
        if let Some(endpoint) = service
            .get("endpoint_url")
            .and_then(serde_json::Value::as_str)
        {
            let _ = writeln!(text, "  endpoint: {endpoint}");
        }
    }
    if let Some(result) = output.get("result")
        && let Some(count) = result
            .get("signaled_pids")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len)
    {
        let _ = writeln!(text, "  stopped processes: {count}");
    }
    text
}

fn load_managed_service(paths: &AppPaths, service_id: &str) -> Result<ManagedServiceRecord> {
    validate_service_id(service_id)?;
    let manifest_path = paths.service_manifest_path(service_id);
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "managed service `{service_id}` not found at {}",
            manifest_path.display()
        )
    })?;
    let mut record = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    record.normalize_paths_for_host();
    let refreshed_from_engine = record.refresh_from_engine_state().unwrap_or(false);
    let refreshed_liveness = refresh_managed_service_runtime_liveness(&mut record);
    if refreshed_from_engine || refreshed_liveness {
        let _ = record.write();
    }
    if record.service_id != service_id {
        bail!(
            "managed service manifest {} contains service_id `{}`, expected `{service_id}`",
            manifest_path.display(),
            record.service_id
        );
    }
    Ok(record)
}

fn render_internal_status_text(paths: &AppPaths) -> Result<String> {
    let services = load_managed_services(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "rocmd status");
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(
        output,
        "  policy: built into rocm; no separate rocmd binary is required"
    );
    let _ = writeln!(output, "  services: {}", services.len());
    Ok(output)
}

fn run_internal_sandbox_tool(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    allow_native_fallback: bool,
) -> Result<serde_json::Value> {
    if !allow_native_fallback {
        bail!(
            "isolated sandbox runner is unavailable in the single-binary build; pass --allow-native-fallback to run the restricted internal tool API"
        );
    }
    let output = match tool {
        SandboxToolArg::ListServers => {
            let services = load_managed_services(paths)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "listed",
                "mutating": false,
                "count": services.len(),
                "services": services,
            })
        }
        SandboxToolArg::StopServer => {
            let service_id = service_id.context("stop_server requires --service-id")?;
            let result = stop_internal_managed_service(paths, &service_id)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "stopped",
                "mutating": true,
                "result": result,
            })
        }
        SandboxToolArg::RestartServer => {
            let service_id = service_id.context("restart_server requires --service-id")?;
            let service = restart_internal_managed_service(paths, &service_id)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "restarted",
                "mutating": true,
                "service": service,
            })
        }
    };
    Ok(serde_json::json!({
        "protocol": "rocmd-sandbox-run-v0",
        "tool": tool.as_cli_value(),
        "ok": true,
        "ok_meaning": "sandbox wrapper completed; inspect output.status for the restricted tool result",
        "isolation": "native_restricted",
        "output": output,
    }))
}

fn stop_internal_managed_service(paths: &AppPaths, service_id: &str) -> Result<serde_json::Value> {
    let mut record = load_managed_service(paths, service_id)?;
    let engine_stop = if record.engine == "lemonade" {
        unload_lemonade_service_model(&record).map(|()| StopResponse {
            stopped: true,
            graceful: true,
        })
    } else {
        engine_request::<_, StopResponse>(
            Some(paths),
            &record.engine,
            EngineMethod::Stop,
            &StopRequest {
                service_id: record.service_id.clone(),
                force: true,
            },
        )
    };
    let mut signaled_pids = Vec::new();
    for pid in [record.engine_pid, Some(record.supervisor_pid)]
        .into_iter()
        .flatten()
        .filter(|pid| *pid != 0 && *pid != std::process::id())
    {
        if signal_process_tree(pid).is_ok() {
            signaled_pids.push(pid);
        }
    }
    record.status = "stopped".to_owned();
    record.write()?;
    let engine_stop = match engine_stop {
        Ok(response) => serde_json::json!({
            "attempted": true,
            "stopped": response.stopped,
            "graceful": response.graceful,
        }),
        Err(error) => serde_json::json!({
            "attempted": true,
            "error": error.to_string(),
        }),
    };
    Ok(serde_json::json!({
        "service_id": service_id,
        "status": record.status,
        "engine_stop": engine_stop,
        "signaled_pids": signaled_pids,
    }))
}

fn unload_lemonade_service_model(record: &ManagedServiceRecord) -> Result<()> {
    let body = serde_json::json!({
        "model_name": record.canonical_model_id,
    });
    let (status, response_body) = http_post_local_service_json(
        &record.host,
        record.port,
        "/v1/unload",
        &body,
        Duration::from_secs(5),
    )?;
    if status == 200 {
        thread::sleep(Duration::from_millis(500));
        Ok(())
    } else {
        bail!("lemonade unload returned HTTP {status}: {response_body}");
    }
}

fn restart_internal_managed_service(
    paths: &AppPaths,
    service_id: &str,
) -> Result<ManagedServiceRecord> {
    let mut record = load_managed_service(paths, service_id)?;
    let _ = stop_internal_managed_service(paths, service_id);
    let policy = parse_device_policy(record.device_policy.as_deref())?;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    if let Some(parent) = record.engine_state_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let current_exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let recipe = parse_engine_recipe_json_arg(record.engine_recipe_json.clone())?;
    let serve_args = builtin_engine_serve_http_args(
        &record.engine,
        &record.service_id,
        &record.canonical_model_id,
        &record.host,
        record.port,
        &policy,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
        recipe.as_ref(),
        &record.engine_state_path,
        Some(&record.log_path),
    )?;
    let engine_envs_root = env_root_for_service(
        paths,
        &record.engine,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
    )?;
    #[cfg(windows)]
    let child_pid = {
        let env_values = app_path_env_var_values(paths, engine_envs_root.as_deref());
        let env_refs = app_path_env_var_refs(&env_values);
        rocm_core::spawn_detached_no_inherit(&current_exe, &serve_args, &env_refs)
            .context("failed to restart managed engine process")?
    };
    #[cfg(not(windows))]
    let child_pid = {
        let mut command = managed_service_process_command(&current_exe, &serve_args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, Some(&record.log_path))?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, paths);
        if let Some(engine_envs_root) = engine_envs_root.as_deref() {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        let mut child = command
            .spawn()
            .context("failed to restart managed engine process")?;
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check restarted engine startup state")?
        {
            record.status = "failed".to_owned();
            record.write()?;
            bail!(
                "managed engine exited immediately with status {}; inspect {}",
                status,
                record.log_path.display()
            );
        }
        child.id()
    };
    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));
    record.status = "running".to_owned();
    record.supervisor_pid = child_pid;
    record.engine_pid = Some(child_pid);
    record.restart_count = record.restart_count.saturating_add(1);
    record.last_restart_unix_ms = Some(rocm_core::unix_time_millis());
    record.status = if wait_for_service_http_ready(
        &record.engine,
        &record.host,
        record.port,
        &record.canonical_model_id,
        Duration::from_secs(45),
    ) {
        "ready".to_owned()
    } else {
        "starting".to_owned()
    };
    record.write()?;
    Ok(record)
}

fn signal_process_tree(pid: u32) -> Result<()> {
    rocm_core::terminate_process(pid)?;
    thread::sleep(Duration::from_millis(300));
    Ok(())
}

fn validate_service_id(service_id: &str) -> Result<()> {
    if service_id.trim().is_empty() {
        bail!("service id must not be empty");
    }
    if service_id.contains('/') || service_id.contains('\\') {
        bail!("service id must not contain path separators");
    }
    Ok(())
}

fn list_log_files(dir: &Path, limit: usize) -> Vec<PathBuf> {
    if limit == 0 || !dir.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("log"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.into_iter().take(limit).collect()
}

fn read_optional_tail_lines(path: &Path, limit: usize, label: &str) -> Vec<String> {
    if !path.is_file() {
        return Vec::new();
    }
    match read_tail_lines(path, limit, label) {
        Ok(lines) => lines,
        Err(error) => vec![format!("<failed to read {label}: {error}>")],
    }
}

fn format_cli_lifecycle_tail_line(line: &str) -> String {
    let level = lifecycle_field(line, "level").unwrap_or("info");
    let category = lifecycle_field(line, "category").unwrap_or("cli");
    let action = lifecycle_field(line, "action").unwrap_or("event");
    let message = line
        .split_once(" message=")
        .map_or(line, |(_, message)| message)
        .trim();
    let label = lifecycle_event_label(category, action);
    if level == "info" {
        format!("{label}: {message}")
    } else {
        format!("{label} ({level}): {message}")
    }
}

fn lifecycle_event_label(category: &str, action: &str) -> String {
    match action {
        "runtime_activate" => "Runtime changed".to_owned(),
        "runtime_import" | "runtime_adopt" => "Runtime added".to_owned(),
        "runtime_uninstall" => "Runtime removed".to_owned(),
        "runtime_update" | "update_apply" | "update_check" => "Update check".to_owned(),
        "install_sdk" | "install_driver" => "Install".to_owned(),
        "engine_install" | "engine_switch" => "Engine changed".to_owned(),
        "service_start" | "service_stop" | "service_restart" | "serve" => {
            "Service event".to_owned()
        }
        "automation" | "watcher" => "Automation".to_owned(),
        "event" => humanize_log_label(category),
        other => humanize_log_label(other),
    }
}

fn humanize_log_label(value: &str) -> String {
    let mut words = value
        .split(['_', '-', '/'])
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return "Activity".to_owned();
    }
    let first = words.remove(0);
    let mut output = String::new();
    let mut chars = first.chars();
    if let Some(ch) = chars.next() {
        output.extend(ch.to_uppercase());
        output.push_str(chars.as_str());
    }
    for word in words {
        output.push(' ');
        output.push_str(word);
    }
    output
}

fn lifecycle_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    line.split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .filter(|value| !value.is_empty())
}

fn read_tail_lines(path: &Path, limit: usize, label: &str) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let file = fs::File::open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    let reader = io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {label} {}", path.display()))?;
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    Ok(lines.into_iter().collect())
}

pub(crate) fn render_update_text(paths: &AppPaths) -> Result<String> {
    let mut output = therock::render_update_report(paths)?;
    append_update_surfaces(&mut output);
    Ok(output)
}

fn append_update_surfaces(output: &mut String) {
    let _ = writeln!(output, "  update_surfaces:");
    let _ = writeln!(
        output,
        "    cli: installed={} status=not_configured reason=repository-owned CLI update feed is not published yet",
        env!("CARGO_PKG_VERSION")
    );
    let engine_ids = engine_inventory()
        .iter()
        .map(|(engine, _)| *engine)
        .collect::<Vec<_>>()
        .join(",");
    let _ = writeln!(
        output,
        "    engines: status=package_managed packaged=[{engine_ids}] reason=first-party engine binaries update with the rocm-cli package; data-dir plugins are user-managed"
    );
    match load_model_recipe_registry() {
        Ok(registry) => match registry.source {
            ModelRecipeRegistrySource::BuiltIn => {
                let _ = writeln!(
                    output,
                    "    model_recipes: status=built_in count={} reason=external signed recipe index is not configured",
                    registry.recipes.len()
                );
            }
            ModelRecipeRegistrySource::SignedIndex {
                index_path,
                signature_path,
                public_key_path,
            } => {
                let _ = writeln!(
                    output,
                    "    model_recipes: status=signed_index count={} index={} signature={} public_key={} reason=loaded signed recipe index; recipe update feed is not live in this build",
                    registry.recipes.len(),
                    index_path.display(),
                    signature_path.display(),
                    public_key_path.display()
                );
            }
        },
        Err(error) => {
            let _ = writeln!(
                output,
                "    model_recipes: status=error reason={}",
                sanitize_log_value(&error.to_string())
            );
        }
    }
    let runtime_status = if output.contains("  managed runtimes: none") {
        "none_configured"
    } else {
        "checked_above"
    };
    let _ = writeln!(
        output,
        "    runtimes: status={runtime_status} reason=TheRock runtime update checks above are the only live update checks in this build"
    );
    let _ = writeln!(
        output,
        "  note: `rocm update --apply` applies runtime updates only; CLI, engine, and recipe update feeds require published metadata before they can mutate state"
    );
}

fn apply_runtime_update(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    runtime_selector: Option<&str>,
    activate: bool,
    dry_run: bool,
) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let source = select_runtime_update_source(&manifests, config, runtime_selector)?;
    let plan = therock::runtime_update_plan(paths, source)?;
    let mut output = String::new();
    let _ = writeln!(output, "runtime update");
    let _ = writeln!(output, "  source_runtime_key: {}", source.runtime_key);
    let _ = writeln!(output, "  source_runtime_id: {}", source.runtime_id);
    let _ = writeln!(output, "  channel: {}", source.channel);
    let _ = writeln!(output, "  format: {}", source.format);
    let _ = writeln!(output, "  family: {}", source.family);
    let _ = writeln!(
        output,
        "  installed_version: {}",
        therock::runtime_version_display(&source.version)
    );
    let _ = writeln!(
        output,
        "  latest_version: {}",
        therock::runtime_version_display(&plan.latest_version)
    );
    let _ = writeln!(output, "  status: {}", plan.status);
    let _ = writeln!(output, "  activate_after_install: {activate}");
    if !plan.update_available {
        let _ = writeln!(output, "  result: no newer runtime found");
        return Ok(output);
    }

    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let install_plan = therock::install_sdk(
            paths,
            &source.channel,
            &source.format,
            None,
            None,
            None,
            true,
        )?;
        let _ = writeln!(output, "  install_plan:");
        for line in install_plan.lines() {
            let _ = writeln!(output, "    {line}");
        }
        return Ok(output);
    }

    let install_output = therock::install_sdk(
        paths,
        &source.channel,
        &source.format,
        None,
        None,
        None,
        false,
    )?;
    let manifests_after = therock::load_runtime_manifests(paths)?;
    let installed = select_installed_update_runtime(&manifests_after, source, &plan.latest_version)
        .context("updated runtime install completed but the new runtime manifest was not found")?;
    let _ = writeln!(output, "  installed_runtime_key: {}", installed.runtime_key);
    let _ = writeln!(
        output,
        "  installed_runtime_root: {}",
        installed.install_root.display()
    );
    if activate {
        let activation = activate_runtime(paths, config, &installed.runtime_key)?;
        config.save(paths)?;
        let _ = writeln!(
            output,
            "  activated_runtime_key: {}",
            activation.runtime_key
        );
        let _ = writeln!(
            output,
            "  previous_runtime_key: {}",
            activation
                .previous_runtime_key
                .as_deref()
                .unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "  note: running services keep their recorded runtime until they are restarted"
        );
    } else {
        let _ = writeln!(
            output,
            "  next step: rocm runtimes activate {}",
            installed.runtime_key
        );
    }
    let _ = writeln!(output, "  install_output:");
    for line in install_output.lines() {
        let _ = writeln!(output, "    {line}");
    }
    Ok(output)
}

fn select_runtime_update_source<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    config: &RocmCliConfig,
    runtime_selector: Option<&str>,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(selector) = runtime_selector {
        return select_runtime_manifest(manifests, selector);
    }
    if let Some(active) = current_runtime_manifest(config, manifests) {
        return Ok(active);
    }
    match manifests {
        [] => bail!(
            "no managed runtimes are registered; run `rocm install sdk --channel release --format wheel` first"
        ),
        [only] => Ok(only),
        _ => bail!(
            "multiple runtimes are registered and no active runtime is configured; pass `--runtime <runtime-key>`"
        ),
    }
}

fn select_installed_update_runtime<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    source: &therock::InstalledRuntimeManifest,
    latest_version: &str,
) -> Option<&'a therock::InstalledRuntimeManifest> {
    manifests.iter().find(|manifest| {
        manifest.channel == source.channel
            && manifest.format == source.format
            && manifest.family == source.family
            && manifest.version == latest_version
    })
}

pub(crate) fn render_automations_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let recent_events = load_recent_automation_events(paths, 5).unwrap_or_default();
    let recent_proposals = load_recent_automation_proposals(paths, 5).unwrap_or_default();
    let recent_audit_events = load_recent_audit_events(paths, 5).unwrap_or_default();
    let mut output = String::new();
    let _ = writeln!(output, "automation checks");
    let _ = writeln!(output, "  config: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  background checks: {}",
        if config.automation_daemon_enabled() {
            "on"
        } else {
            "off"
        }
    );
    if let Some(state) = runtime_state.as_ref() {
        let _ = writeln!(
            output,
            "  background service: {}",
            if state.running { "running" } else { "stopped" }
        );
        let _ = writeln!(
            output,
            "  local event intake: {}",
            state
                .local_webhook_endpoint
                .as_deref()
                .unwrap_or("disabled")
        );
    } else {
        let _ = writeln!(output, "  background service: not running");
        let _ = writeln!(output, "  local event intake: disabled");
    }
    for watcher in builtin_watchers() {
        let runtime_snapshot = runtime_state.as_ref().and_then(|state| {
            state
                .active_watchers
                .iter()
                .find(|item| item.id == watcher.id)
        });
        let _ = writeln!(
            output,
            "  {} ({})",
            watcher_plain_name(watcher.id),
            if config.watcher_enabled(watcher) {
                "on"
            } else {
                "off"
            }
        );
        let _ = writeln!(
            output,
            "    setting: {}",
            watcher_mode_plain_label(config.effective_watcher_mode(watcher))
        );
        let _ = writeln!(
            output,
            "    listens for: {}",
            watcher_plain_trigger(watcher.id)
        );
        let _ = writeln!(output, "    does: {}", watcher_plain_action(watcher.id));
        if let Some(note) = watcher_policy_note(watcher.id) {
            let _ = writeln!(output, "    policy: {note}");
        }
        if let Some(snapshot) = runtime_snapshot {
            let _ = writeln!(output, "    last check: {}", watcher_last_check(snapshot));
        }
    }
    if !recent_events.is_empty() {
        let _ = writeln!(output, "  recent automation activity:");
        for event in recent_events {
            let _ = writeln!(output, "    {}", automation_event_plain_summary(&event));
            if let Some(service_id) = event.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
        }
    }
    if !recent_proposals.is_empty() {
        let _ = writeln!(output, "  recent review requests:");
        for proposal in recent_proposals {
            let _ = writeln!(
                output,
                "    {} [{}] {}",
                proposal.proposal_id,
                proposal_status_label(&proposal.status),
                proposal_plain_summary(&proposal)
            );
            let _ = writeln!(output, "      why: {}", proposal_plain_reason(&proposal));
            if let Some(service_id) = proposal.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
            if let Some(artifact_ref) = proposal
                .arguments
                .get("artifact_ref")
                .and_then(serde_json::Value::as_str)
            {
                let _ = writeln!(output, "      model file: {artifact_ref}");
            }
            if proposal_bool_argument(&proposal, "allow_artifact_download") {
                let limit = proposal
                    .arguments
                    .get("artifact_max_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .map_or_else(|| "not set".to_owned(), format_bytes_for_user);
                let _ = writeln!(output, "      download: approved up to {limit}");
            } else if proposal_kind(&proposal) == ProposalKind::PrefetchArtifact {
                let _ = writeln!(output, "      download: not approved yet");
            }
            if proposal_kind(&proposal) == ProposalKind::DriverPlan {
                let _ = writeln!(
                    output,
                    "      effect: show a driver plan only; no driver install"
                );
            }
            if proposal.status == "pending" {
                let _ = writeln!(
                    output,
                    "      controls: /automations approve {} | /automations reject {}",
                    proposal.proposal_id, proposal.proposal_id
                );
            }
        }
    }
    if !recent_audit_events.is_empty() {
        let _ = writeln!(output, "  recent background activity:");
        for event in recent_audit_events {
            let _ = writeln!(output, "    {}", audit_event_plain_summary(&event));
            if let Some(service_id) = event.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalKind {
    RestartServer,
    StopServer,
    CheckUpdates,
    PrefetchArtifact,
    DriverPlan,
    Other,
}

fn proposal_kind(proposal: &AutomationProposalRecord) -> ProposalKind {
    match proposal
        .tool
        .as_deref()
        .or_else(|| fallback_tool_for_automation_action(&proposal.action))
    {
        Some("restart_server") => ProposalKind::RestartServer,
        Some("stop_server") => ProposalKind::StopServer,
        Some("check_updates") => ProposalKind::CheckUpdates,
        Some("prefetch_artifact") => ProposalKind::PrefetchArtifact,
        Some("driver_plan") => ProposalKind::DriverPlan,
        _ => ProposalKind::Other,
    }
}

fn fallback_tool_for_automation_action(action: &str) -> Option<&'static str> {
    match action {
        "queue_restart_proposal" => Some("restart_server"),
        "queue_stop_server_proposal" => Some("stop_server"),
        "queue_update_proposal" => Some("check_updates"),
        "queue_prefetch_proposal" => Some("prefetch_artifact"),
        "prepare_driver_plan" => Some("driver_plan"),
        _ => None,
    }
}

fn proposal_plain_summary(proposal: &AutomationProposalRecord) -> &'static str {
    match proposal_kind(proposal) {
        ProposalKind::RestartServer => "Restart a model server",
        ProposalKind::StopServer => "Stop a model server",
        ProposalKind::CheckUpdates => "Check for ROCm updates",
        ProposalKind::PrefetchArtifact => "Prepare a model file",
        ProposalKind::DriverPlan => "Show a driver plan",
        ProposalKind::Other => "Review an automation request",
    }
}

fn proposal_plain_reason(proposal: &AutomationProposalRecord) -> &'static str {
    match proposal_kind(proposal) {
        ProposalKind::RestartServer => "A managed server looks stopped or unhealthy.",
        ProposalKind::StopServer => "GPU pressure is high and serving should be reviewed.",
        ProposalKind::CheckUpdates => "A scheduled update check is due.",
        ProposalKind::PrefetchArtifact => "rocm-cli was asked to prepare this model file.",
        ProposalKind::DriverPlan => "A driver update signal was received.",
        ProposalKind::Other => "An enabled automation asked for review.",
    }
}

fn proposal_status_label(status: &str) -> &'static str {
    match status {
        "pending" => "waiting for review",
        "approved" => "approved",
        "completed" => "done",
        "rejected" => "rejected",
        "failed" => "failed",
        _ => "status unknown",
    }
}

fn proposal_bool_argument(proposal: &AutomationProposalRecord, key: &str) -> bool {
    proposal
        .arguments
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn automation_event_plain_summary(event: &AutomationEventRecord) -> &'static str {
    match event.watcher_id.as_str() {
        "therock-update" => "ROCm update check recorded.",
        "server-recover" => "Server recovery check recorded.",
        "gpu-metrics" => "GPU status check recorded.",
        "gpu-thermal-protect" => "GPU pressure check recorded.",
        "cache-warm" => "Model file preparation request recorded.",
        "driver-upgrade" => "Driver update review recorded.",
        _ => "Automation activity recorded.",
    }
}

fn audit_event_plain_summary(event: &AuditEventRecord) -> &'static str {
    match event.category.as_str() {
        "automation" => "Automation activity was recorded.",
        "proposal" => "A review request changed status.",
        "provider" => "Provider request completed.",
        "service" => "Managed server activity was recorded.",
        "install" => "Install activity was recorded.",
        "update" => "Update activity was recorded.",
        "runtime" => "ROCm runtime activity was recorded.",
        "engine" => "Engine activity was recorded.",
        _ => "Background activity was recorded.",
    }
}

fn format_bytes_for_user(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    let bytes = bytes as f64;
    if bytes >= GB {
        format!("{:.1} GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes / KB)
    } else {
        format!("{} bytes", bytes as u64)
    }
}

const fn watcher_mode_plain_label(mode: WatcherMode) -> &'static str {
    match mode {
        WatcherMode::Observe => "record only",
        WatcherMode::Propose => "ask before taking action",
        WatcherMode::Contained => "ask before changes; keep actions limited",
    }
}

fn watcher_last_check(snapshot: &rocm_core::WatcherRuntimeSnapshot) -> &'static str {
    let Some(event) = snapshot.last_event.as_deref() else {
        return "not yet";
    };
    match event {
        "remind_update_check" => "ROCm update check recorded",
        "queue_update_proposal" => "ROCm update review requested",
        "collect_failure_snapshot" => "server health check recorded",
        "restart_managed_service" | "queue_restart_proposal" => "server restart review requested",
        "record_gpu_metrics" => "GPU status check recorded",
        "queue_stop_server_proposal" => "GPU pressure review requested",
        "queue_prefetch_proposal" => "model file review requested",
        "prepare_driver_plan" => "driver plan review requested",
        _ => "activity recorded",
    }
}

fn watcher_plain_name(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "ROCm update checks",
        "server-recover" => "Server recovery",
        "gpu-metrics" => "GPU status checks",
        "gpu-thermal-protect" => "GPU pressure protection",
        "cache-warm" => "Model file preparation",
        "driver-upgrade" => "Driver update review",
        _ => "Automation check",
    }
}

fn watcher_plain_trigger(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "a scheduled check",
        "server-recover" => "a server that stops or becomes unhealthy",
        "gpu-metrics" => "local GPU status updates",
        "gpu-thermal-protect" => "high GPU temperature or memory pressure",
        "cache-warm" => "a request to prepare a model file",
        "driver-upgrade" => "a driver update signal",
        _ => "local automation activity",
    }
}

fn watcher_plain_action(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "checks for ROCm updates without installing them",
        "server-recover" => "asks before restarting a managed server",
        "gpu-metrics" => "records local GPU status only",
        "gpu-thermal-protect" => "asks before stopping a managed server",
        "cache-warm" => "asks before preparing or downloading a model file",
        "driver-upgrade" => "shows a driver plan only",
        _ => "records the event",
    }
}

fn watcher_policy_note(watcher_id: &str) -> Option<&'static str> {
    match watcher_id {
        "gpu-metrics" => Some(
            "read-only telemetry; propose/contained modes record events only and do not create review requests or mutate services",
        ),
        "gpu-thermal-protect" => Some(
            "GPU pressure protection is review-gated; propose/contained modes queue a stop review only and never stop servers automatically",
        ),
        "cache-warm" => Some(
            "artifact prefetch stays review-gated; contained mode creates a review request instead of downloading without reviewed source policy",
        ),
        "driver-upgrade" => Some(
            "local driver update signals stay review-gated; review requests run a read-only driver plan and never install drivers automatically",
        ),
        _ => None,
    }
}

pub(crate) fn render_daemon_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let mut output = String::new();
    let _ = writeln!(output, "Background helper");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Status: {}",
        match runtime_state {
            Some(ref state) if state.running => "running",
            _ => "not running",
        }
    );
    let _ = writeln!(output, "Startup: on demand");
    let _ = writeln!(
        output,
        "Why it runs: automation checks and local model servers use it when needed."
    );
    let _ = writeln!(
        output,
        "Automation checks: {}",
        if config.automation_daemon_enabled() {
            "on"
        } else {
            "off"
        }
    );
    let _ = writeln!(output, "Saved state: kept on this computer");
    let _ = writeln!(output);
    let _ = writeln!(output, "Choose Automations to review background checks.");
    let _ = writeln!(output, "Choose Local servers to see local model servers.");
    output
}

pub(crate) fn render_sidebar_text(
    paths: &AppPaths,
    config: &RocmCliConfig,
    provider: &str,
    setup_ready: bool,
) -> String {
    let records = load_managed_services(paths).unwrap_or_default();
    let server_counts = managed_service_sidebar_counts(&records);
    let default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    let mut output = String::new();
    let _ = writeln!(output, "ROCm CLI");
    let _ = writeln!(
        output,
        "Setup: {}",
        if setup_ready { "ready" } else { "not set up" }
    );
    let _ = writeln!(output, "Assistant: {}", friendly_provider_label(provider));
    let _ = writeln!(
        output,
        "Model runner: {}",
        friendly_engine_label(default_engine)
    );
    let _ = writeln!(
        output,
        "Changes: {}",
        if config.permissions.full_access_enabled() {
            "full access"
        } else {
            "ask first"
        }
    );
    let _ = writeln!(
        output,
        "Local servers: {}",
        local_server_sidebar_status(&server_counts)
    );
    let enabled_watchers = builtin_watchers()
        .iter()
        .filter(|watcher| config.watcher_enabled(watcher))
        .count();
    let _ = writeln!(output, "Background checks: {enabled_watchers}");
    let _ = writeln!(
        output,
        "Background helper: {}",
        if config.automation_daemon_enabled() {
            "needed"
        } else {
            "off"
        }
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "Choose Run setup check for details.");
    output
}

fn friendly_provider_label(provider: &str) -> &str {
    match provider {
        "local" => "local model",
        "openai" => "OpenAI",
        "anthropic" => "Anthropic",
        other => other,
    }
}

fn friendly_engine_label(engine: &str) -> &str {
    match engine {
        "lemonade" => "Lemonade",
        "pytorch" => "PyTorch",
        "llama.cpp" => "llama.cpp",
        "vllm" => "vLLM",
        "sglang" => "SGLang",
        "atom" => "ATOM",
        other => other,
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ManagedServiceSidebarCounts {
    pub(crate) ready: usize,
    pub(crate) starting: usize,
    pub(crate) past_attempts: usize,
}

pub(crate) fn managed_service_sidebar_counts(
    records: &[ManagedServiceRecord],
) -> ManagedServiceSidebarCounts {
    let mut counts = ManagedServiceSidebarCounts::default();
    for record in records {
        match record.status.as_str() {
            "ready" | "running" => counts.ready += 1,
            "starting" | "recovering" => counts.starting += 1,
            _ => counts.past_attempts += 1,
        }
    }
    counts
}

pub(crate) fn managed_service_is_live(record: &ManagedServiceRecord) -> bool {
    matches!(
        record.status.as_str(),
        "ready" | "running" | "starting" | "recovering"
    )
}

/// Idempotency guard for managed launches: returns an already-live managed
/// service for this engine+model, if any.
///
/// Keyed on `(engine, canonical_model_id)` — NOT `service_id`. `generate_service_id`
/// embeds `unix_time_millis()`, so every launch mints a unique id; matching on it
/// would never catch a duplicate. `load_managed_services` refreshes liveness, so
/// stale manifests (dead PIDs) demote to "stopped" and are skipped, letting a
/// genuine relaunch proceed. Records are sorted newest-first, so `find` returns
/// the newest live match. Prevents a second `serve --managed` for the same
/// engine+model from spawning a duplicate process once the TUI job-bridge guard
/// has cleared.
fn existing_live_managed_service(
    paths: &AppPaths,
    engine: &str,
    canonical_model_id: &str,
) -> Option<ManagedServiceRecord> {
    load_managed_services(paths)
        .ok()?
        .into_iter()
        .find(|record| {
            record.engine == engine
                && record.canonical_model_id == canonical_model_id
                && managed_service_is_live(record)
        })
}

fn managed_service_running_state(status: &str) -> &'static str {
    match status {
        "ready" | "running" => "running",
        "starting" | "recovering" => "starting",
        "failed" | "stopped" => "not_running",
        _ => "unknown",
    }
}

const SERVICE_LIVENESS_CHECK_TIMEOUT: Duration = Duration::from_millis(750);

fn refresh_managed_service_runtime_liveness(record: &mut ManagedServiceRecord) -> bool {
    if !managed_service_is_live(record) {
        return false;
    }

    let endpoint_ready = matches!(record.status.as_str(), "ready" | "running")
        && managed_service_endpoint_model_ready(record, SERVICE_LIVENESS_CHECK_TIMEOUT)
            .unwrap_or(false);
    if endpoint_ready {
        return false;
    }

    let tracked_pids = [record.engine_pid, Some(record.supervisor_pid)]
        .into_iter()
        .flatten()
        .filter(|pid| *pid != 0)
        .collect::<Vec<_>>();
    let has_tracked_pid = !tracked_pids.is_empty();
    let has_live_pid = tracked_pids.iter().any(|pid| process_is_running(*pid));
    if has_tracked_pid && !has_live_pid {
        if record.status != "stopped" {
            record.status = "stopped".to_owned();
            return true;
        }
        return false;
    }

    if matches!(record.status.as_str(), "ready" | "running") {
        let new_status = if has_live_pid { "starting" } else { "stopped" };
        if record.status != new_status {
            record.status = new_status.to_owned();
            return true;
        }
    }

    false
}

fn local_server_sidebar_status(counts: &ManagedServiceSidebarCounts) -> String {
    match (counts.ready, counts.starting) {
        (0, 0) => "none ready".to_owned(),
        (ready, 0) => format!("{ready} ready"),
        (0, starting) => format!("{starting} starting"),
        (ready, starting) => format!("{ready} ready, {starting} starting"),
    }
}

pub(crate) fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
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
        if let Ok(mut record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) {
            record.normalize_paths_for_host();
            let refreshed_from_engine = record.refresh_from_engine_state().unwrap_or(false);
            let refreshed_liveness = refresh_managed_service_runtime_liveness(&mut record);
            if refreshed_from_engine || refreshed_liveness {
                let _ = record.write();
            }
            records.push(record);
        }
    }

    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    Ok(records)
}

#[cfg(test)]
pub(crate) fn tui_help_text() -> String {
    let mut output = String::new();
    let _ = writeln!(output, "slash commands");
    let _ = writeln!(output, "  /home          return to the main dashboard");
    let _ = writeln!(output, "  /help          show this help");
    let _ = writeln!(
        output,
        "  /examine        check this computer and ROCm setup"
    );
    let _ = writeln!(
        output,
        "  /setup         reopen the simple ROCm setup screen"
    );
    let _ = writeln!(
        output,
        "                 status, reset, and skip are available when needed"
    );
    let _ = writeln!(output, "  /permissions   show or change confirmation mode");
    let _ = writeln!(
        output,
        "  /runtimes      install, choose, add, remove, or roll back ROCm installs"
    );
    let _ = writeln!(
        output,
        "  /install       install ROCm or check driver setup"
    );
    let _ = writeln!(
        output,
        "                 adopt adds an existing Python ROCm folder as read-only"
    );
    let _ = writeln!(
        output,
        "  /engine        choose or install local model engines"
    );
    let _ = writeln!(
        output,
        "  /model [NAME]  list recipes or choose a known model"
    );
    let _ = writeln!(output, "  /plan TEXT     render an explicit request plan");
    let _ = writeln!(output, "  /config        show saved settings");
    let _ = writeln!(
        output,
        "                 set-planner-provider enables optional provider help for ambiguous plans"
    );
    let _ = writeln!(
        output,
        "  /automations   show background checks and review requests"
    );
    let _ = writeln!(
        output,
        "  /reviews       open review requests that need a decision"
    );
    let _ = writeln!(
        output,
        "  /approve       review the selected request before approving"
    );
    let _ = writeln!(
        output,
        "  /reject        review the selected request before rejecting"
    );
    let _ = writeln!(
        output,
        "  /edit          open review request editing from Automations"
    );
    let _ = writeln!(output, "  /services      show local model servers");
    let _ = writeln!(output, "  /logs [QUERY]  browse or search recent activity");
    let _ = writeln!(
        output,
        "                 use /logs --search <query> when the query matches a service id"
    );
    let _ = writeln!(
        output,
        "                 use /logs --service <id> for managed service logs"
    );
    let _ = writeln!(
        output,
        "                 use /logs next, /logs prev, or /logs refresh after opening the log browser"
    );
    let _ = writeln!(
        output,
        "  /gpu           show the latest local GPU snapshot"
    );
    let _ = writeln!(output, "  /update        show update policy");
    let _ = writeln!(output, "  /daemon        show background helper status");
    let _ = writeln!(output, "  /chat TEXT     stream a provider chat response");
    let _ = writeln!(output, "  /provider X    switch provider for this session");
    let _ = writeln!(
        output,
        "  /uninstall     preview what uninstall would remove"
    );
    let _ = writeln!(
        output,
        "  /serve MODEL   open a guided local model server setup"
    );
    let _ = writeln!(output, "  /clear         clear the transcript");
    let _ = writeln!(output, "  /quit, /exit   exit the TUI");
    let _ = writeln!(output);
    let _ = writeln!(output, "keyboard");
    let _ = writeln!(output, "  F1             open interactive help");
    let _ = writeln!(output, "  F5             refresh the current screen");
    let _ = writeln!(
        output,
        "  /              open the command popup on normal menu screens"
    );
    let _ = writeln!(
        output,
        "  Tab            complete slash commands and arguments"
    );
    let _ = writeln!(
        output,
        "  Up/Down        choose items or recall input history"
    );
    let _ = writeln!(output, "  PgUp/PgDn      scroll the current view");
    let _ = writeln!(output, "  Mouse wheel    scroll long text and log cards");
    let _ = writeln!(output, "  Home/End       jump to top or bottom");
    let _ = writeln!(output);
    let _ = writeln!(output, "logs");
    let _ = writeln!(
        output,
        "  Targeted logs such as /comfyui logs and service logs open in a scrollable card"
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "natural language");
    let _ = writeln!(output, "  serve qwen with lemonade");
    let _ = writeln!(output, "  install ROCm");
    let _ = writeln!(output, "  uninstall rocm-cli");
    let _ = writeln!(output);
    let _ = writeln!(output, "examples");
    let _ = writeln!(output, "  config set-default-engine lemonade");
    let _ = writeln!(output, "  config set-telemetry local");
    let _ = writeln!(output, "  /engine install lemonade --reinstall");
    let _ = writeln!(output, "  automations enable server-recover");
    output
}

pub(crate) fn render_freeform_plan(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> String {
    let plan = build_freeform_plan_with_context(request, paths, config);
    render_structured_request_plan(&plan, paths)
}

pub(crate) fn freeform_plan_uses_provider(request: &str, config: &RocmCliConfig) -> bool {
    let plan = build_freeform_plan(request, config);
    freeform_plan_needs_ambiguity_resolution(&plan) && configured_planner_provider(config).is_some()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FreeformPlanAction {
    pub title: String,
    pub args: Vec<String>,
    pub approval_required: bool,
    pub reason: String,
    pub has_placeholders: bool,
    pub provider_assisted: bool,
}

#[cfg(test)]
pub(crate) fn freeform_plan_next_action(
    request: &str,
    config: &RocmCliConfig,
) -> Option<FreeformPlanAction> {
    let plan = build_freeform_plan(request, config);
    plan_next_action(plan)
}

pub(crate) fn freeform_plan_next_action_with_context(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Option<FreeformPlanAction> {
    let plan = build_freeform_plan_with_context(request, paths, config);
    plan_next_action(plan)
}

fn plan_next_action(plan: StructuredRequestPlan) -> Option<FreeformPlanAction> {
    plan.actions.last().map(|action| FreeformPlanAction {
        title: action.title.to_owned(),
        args: action.args.clone(),
        approval_required: action.approval == "required",
        reason: action.reason.to_owned(),
        has_placeholders: action
            .args
            .iter()
            .any(|arg| arg.starts_with('<') && arg.ends_with('>')),
        provider_assisted: plan.provider_assisted,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StructuredRequestPlan {
    request: String,
    planner: String,
    provider_assisted: bool,
    intent: PlannerIntent,
    confidence: &'static str,
    approval: &'static str,
    parsed: Vec<(String, String)>,
    actions: Vec<PlannedToolCall>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannerIntent {
    Ask,
    Serve,
    InstallSdk,
    InstallDriver,
    Update,
    Uninstall,
    Inspect,
}

impl PlannerIntent {
    const fn label(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Serve => "serve",
            Self::InstallSdk => "install sdk",
            Self::InstallDriver => "install driver",
            Self::Update => "update check",
            Self::Uninstall => "uninstall",
            Self::Inspect => "inspect",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedToolCall {
    title: &'static str,
    tool: &'static str,
    args: Vec<String>,
    approval: &'static str,
    reason: &'static str,
}

impl PlannedToolCall {
    const fn read_only(title: &'static str, args: Vec<String>, reason: &'static str) -> Self {
        Self {
            title,
            tool: "rocm",
            args,
            approval: "not required",
            reason,
        }
    }

    const fn approval_required(
        title: &'static str,
        args: Vec<String>,
        reason: &'static str,
    ) -> Self {
        Self {
            title,
            tool: "rocm",
            args,
            approval: "required",
            reason,
        }
    }
}

fn build_freeform_plan(request: &str, config: &RocmCliConfig) -> StructuredRequestPlan {
    build_freeform_plan_with_recipes(request, config, None)
}

fn build_freeform_plan_with_recipes(
    request: &str,
    config: &RocmCliConfig,
    recipes: Option<&[ModelRecipeRecord]>,
) -> StructuredRequestPlan {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());

    if planner_is_serve_request(&lower) {
        let requested_model = infer_model_from_request(trimmed)
            .filter(|model| !generic_model_phrase(model))
            .or_else(|| infer_small_local_model_from_request(&lower))
            .or_else(|| infer_recommended_assistant_model_from_request(&lower));
        let resolved_recipe =
            requested_model.and_then(|model| planner_resolve_model_recipe(model, recipes));
        let model = resolved_recipe
            .as_ref()
            .map(|recipe| recipe.canonical_model_id.as_str())
            .or(requested_model)
            .unwrap_or("<model>");
        let engine = infer_engine_from_request(&lower)
            .or_else(|| {
                resolved_recipe
                    .as_ref()
                    .and_then(|recipe| recipe.preferred_engines.first().map(String::as_str))
            })
            .unwrap_or(default_engine);
        let device_policy = infer_device_policy_from_request(&lower).or_else(|| {
            resolved_recipe
                .as_ref()
                .map(|recipe| recipe.device_policy.as_str())
        });
        if device_policy.is_some_and(device_policy_is_cpu_mode) {
            let mut parsed = vec![
                ("model".to_owned(), model.to_owned()),
                ("engine".to_owned(), engine.to_owned()),
                ("device_policy".to_owned(), "cpu_not_supported".to_owned()),
                ("mode".to_owned(), "managed".to_owned()),
            ];
            if model == "<model>" {
                parsed.push(("missing".to_owned(), "model".to_owned()));
            }
            if let Some(requested_model) = requested_model
                && requested_model != model
            {
                parsed.push(("model_alias".to_owned(), requested_model.to_owned()));
            }
            if let Some(recipe) = resolved_recipe.as_ref() {
                parsed.push(("recipe_source".to_owned(), recipe.source.clone()));
                parsed.push(("recipe_dtype".to_owned(), recipe.dtype.clone()));
            }
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Serve,
                confidence: if model == "<model>" { "medium" } else { "high" },
                approval: "not available; ROCm CLI requires ROCm GPU execution",
                parsed,
                actions: Vec::new(),
                notes: vec![
                    "CPU mode is not offered by rocm-cli.".to_owned(),
                    "Use a ROCm-capable AMD GPU and request GPU execution.".to_owned(),
                ],
            };
        }
        let device_policy = device_policy.map(planned_device_policy_without_fallback);
        let mut args = vec![
            "serve".to_owned(),
            model.to_owned(),
            "--engine".to_owned(),
            engine.to_owned(),
        ];
        if let Some(device_policy) = device_policy {
            args.push("--device".to_owned());
            args.push(device_policy.to_owned());
        }
        args.push("--managed".to_owned());
        let mut parsed = vec![
            ("model".to_owned(), model.to_owned()),
            ("engine".to_owned(), engine.to_owned()),
            (
                "device_policy".to_owned(),
                device_policy.unwrap_or("gpu_required").to_owned(),
            ),
            ("mode".to_owned(), "managed".to_owned()),
        ];
        if model == "<model>" {
            parsed.push(("missing".to_owned(), "model".to_owned()));
        }
        if let Some(requested_model) = requested_model
            && requested_model != model
        {
            parsed.push(("model_alias".to_owned(), requested_model.to_owned()));
        }
        if let Some(recipe) = resolved_recipe.as_ref() {
            parsed.push(("recipe_source".to_owned(), recipe.source.clone()));
            parsed.push(("recipe_dtype".to_owned(), recipe.dtype.clone()));
        }
        let mut notes = vec![
            "Final execution must come from the structured tool call shown above.".to_owned(),
            "GPU-preferred recipe policies are treated as GPU required; no CPU fallback is implied."
                .to_owned(),
        ];
        if let Some(recipe) = resolved_recipe.as_ref() {
            notes.extend(recipe.warnings.iter().cloned());
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Serve,
            confidence: if model == "<model>" { "medium" } else { "high" },
            approval: "required before launch; plan rendering is read-only",
            parsed,
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect host/runtime state",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::read_only(
                    "Show local model engines",
                    vec!["engines".to_owned(), "list".to_owned()],
                    "read-only engine list",
                ),
                PlannedToolCall::approval_required(
                    "Launch local endpoint",
                    args,
                    "starts or changes a local serving process",
                ),
            ],
            notes,
        };
    }

    if lower.contains("driver") {
        let mut args = vec!["install".to_owned(), "driver".to_owned()];
        if lower.contains("dkms") {
            args.push("--dkms".to_owned());
        }
        args.push("--yes".to_owned());
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::InstallDriver,
            confidence: "high",
            approval: "required before driver changes",
            parsed: vec![(
                "driver_flow".to_owned(),
                if lower.contains("dkms") {
                    "dkms"
                } else {
                    "platform default"
                }
                .to_owned(),
            )],
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect host/driver state",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::approval_required(
                    "Install driver",
                    args,
                    "driver changes are privileged or disruptive",
                ),
            ],
            notes: vec!["Driver changes are always explicit and never silent.".to_owned()],
        };
    }

    if planner_mentions_comfyui(&lower) {
        if any_substring(&lower, &["log", "logs"]) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Inspect,
                confidence: "high",
                approval: "not required for inspection",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::read_only(
                    "Read ComfyUI logs",
                    vec!["comfyui".to_owned(), "logs".to_owned()],
                    "read-only app log check",
                )],
                notes: Vec::new(),
            };
        }
        if any_substring(&lower, &["start", "run", "launch", "open"]) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Serve,
                confidence: "high",
                approval: "required before launch",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::approval_required(
                    "Start ComfyUI",
                    vec!["comfyui".to_owned(), "start".to_owned()],
                    "starts a local ComfyUI process",
                )],
                notes: Vec::new(),
            };
        }
        if planner_requests_comfyui_install(&lower) {
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::InstallSdk,
                confidence: "high",
                approval: "required before installing ComfyUI",
                parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
                actions: vec![PlannedToolCall::approval_required(
                    "Install ComfyUI",
                    vec!["comfyui".to_owned(), "install".to_owned()],
                    "installs ComfyUI into ROCm CLI's managed app folder",
                )],
                notes: vec!["ComfyUI uses the active ROCm CLI managed TheRock runtime.".to_owned()],
            };
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Inspect,
            confidence: "high",
            approval: "not required for inspection",
            parsed: vec![("app".to_owned(), "ComfyUI".to_owned())],
            actions: vec![PlannedToolCall::read_only(
                "Check ComfyUI",
                vec!["comfyui".to_owned(), "status".to_owned()],
                "read-only app status check",
            )],
            notes: Vec::new(),
        };
    }

    if planner_is_install_sdk_request(&lower) {
        let channel = if lower.contains("nightly") {
            "nightly"
        } else {
            "release"
        };
        let build_date = requested_therock_build_date_from_prompt(&lower);
        let Some(prefix) = requested_install_prefix_from_prompt(trimmed) else {
            let mut parsed = vec![
                ("channel".to_owned(), channel.to_owned()),
                ("format".to_owned(), "wheel".to_owned()),
            ];
            if let Some(build_date) = build_date.as_deref() {
                parsed.push(("build_date".to_owned(), build_date.to_owned()));
            }
            return StructuredRequestPlan {
                request: trimmed.to_owned(),
                planner: "hybrid-parser-v1".to_owned(),
                provider_assisted: false,
                intent: PlannerIntent::Ask,
                confidence: "medium",
                approval: "not applicable until an install folder is chosen",
                parsed,
                actions: Vec::new(),
                notes: vec![
                    "Choose a ROCm/TheRock install folder before approving an install.".to_owned(),
                    "Say something like: install TheRock into D:\\ROCm\\therock_venvs.".to_owned(),
                ],
            };
        };
        let mut args = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            channel.to_owned(),
            "--format".to_owned(),
            "wheel".to_owned(),
            "--prefix".to_owned(),
            prefix.clone(),
        ];
        if let Some(build_date) = build_date.as_deref() {
            args.push("--build-date".to_owned());
            args.push(build_date.to_owned());
        }
        let mut parsed = vec![
            ("channel".to_owned(), channel.to_owned()),
            ("format".to_owned(), "wheel".to_owned()),
            ("prefix".to_owned(), prefix),
        ];
        if let Some(build_date) = build_date.as_deref() {
            parsed.push(("build_date".to_owned(), build_date.to_owned()));
        }
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::InstallSdk,
            confidence: "high",
            approval: "required before installing or switching runtimes",
            parsed,
            actions: vec![
                PlannedToolCall::read_only(
                    "Inspect current runtime",
                    vec!["examine".to_owned()],
                    "read-only inspection",
                ),
                PlannedToolCall::approval_required(
                    "Install TheRock SDK",
                    args,
                    "changes managed runtime state",
                ),
            ],
            notes: vec!["TheRock pip venv install is the default managed runtime path.".to_owned()],
        };
    }

    if lower.contains("update") {
        let apply = lower.contains("apply") || lower.contains("upgrade");
        let args = if apply {
            vec!["update".to_owned(), "--apply".to_owned()]
        } else {
            vec!["update".to_owned()]
        };
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Update,
            confidence: "high",
            approval: if apply {
                "required before applying updates"
            } else {
                "not required for update inspection"
            },
            parsed: vec![(
                "mode".to_owned(),
                if apply { "apply" } else { "check" }.to_owned(),
            )],
            actions: vec![if apply {
                PlannedToolCall::approval_required(
                    "Apply update",
                    args,
                    "installs or switches managed runtime state",
                )
            } else {
                PlannedToolCall::read_only("Check updates", args, "read-only update inspection")
            }],
            notes: vec!["Update checks compare against the selected runtime channel.".to_owned()],
        };
    }

    if lower.contains("uninstall") || lower.contains("remove rocm") {
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Uninstall,
            confidence: "high",
            approval: "required before deleting installed files",
            parsed: Vec::new(),
            actions: vec![
                PlannedToolCall::read_only(
                    "Preview uninstall",
                    vec!["uninstall".to_owned(), "--dry-run".to_owned()],
                    "dry-run planning",
                ),
                PlannedToolCall::approval_required(
                    "Apply uninstall",
                    vec!["uninstall".to_owned(), "--yes".to_owned()],
                    "deletes installed files and state",
                ),
            ],
            notes: Vec::new(),
        };
    }

    if planner_is_inspect_request(&lower) {
        return StructuredRequestPlan {
            request: trimmed.to_owned(),
            planner: "hybrid-parser-v1".to_owned(),
            provider_assisted: false,
            intent: PlannerIntent::Inspect,
            confidence: "high",
            approval: "not required for inspection",
            parsed: vec![("engine".to_owned(), default_engine.to_owned())],
            actions: vec![PlannedToolCall::read_only(
                "Inspect local ROCm state",
                vec!["examine".to_owned()],
                "read-only inspection",
            )],
            notes: vec![
                "Use /chat <prompt> in the TUI for provider-backed answers.".to_owned(),
                "Use /plan with an explicit install, update, serve, or uninstall request for action planning.".to_owned(),
            ],
        };
    }

    StructuredRequestPlan {
        request: trimmed.to_owned(),
        planner: "hybrid-parser-v1".to_owned(),
        provider_assisted: false,
        intent: PlannerIntent::Ask,
        confidence: "low",
        approval: "not applicable",
        parsed: Vec::new(),
        actions: Vec::new(),
        notes: vec![
            "No ROCm action was selected from this request.".to_owned(),
            "Use /help in the TUI, or include install, update, serve, uninstall, check, or inspect in the request.".to_owned(),
        ],
    }
}

fn planned_device_policy_without_fallback(policy: &str) -> &str {
    match policy {
        "gpu_preferred" => "gpu_required",
        other => other,
    }
}

fn device_policy_is_cpu_mode(policy: &str) -> bool {
    matches!(
        policy.trim().to_ascii_lowercase().as_str(),
        "cpu" | "cpu_only"
    )
}

fn serve_args_request_cpu_device(args: &[String]) -> bool {
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--device=")
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
        if arg == "--device"
            && let Some(value) = iter.peek()
            && device_policy_is_cpu_mode(value)
        {
            return true;
        }
    }
    false
}

fn planner_resolve_model_recipe(
    model_ref: &str,
    recipes: Option<&[ModelRecipeRecord]>,
) -> Option<ModelRecipeRecord> {
    if let Some(recipes) = recipes {
        return recipes
            .iter()
            .find(|recipe| recipe.matches_ref(model_ref))
            .cloned();
    }
    resolve_builtin_model_recipe(model_ref)
}

fn build_freeform_plan_with_context(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> StructuredRequestPlan {
    let registry = match load_model_recipe_registry() {
        Ok(registry) => Some(registry),
        Err(error) => {
            let mut plan = build_freeform_plan_with_recipes(request, config, Some(&[]));
            plan.confidence = "medium";
            plan.notes.push(format!(
                "Model recipe registry could not be loaded: {error}. Fix the recipe index before using registry aliases."
            ));
            return plan;
        }
    };
    let mut deterministic = build_freeform_plan_with_recipes(
        request,
        config,
        registry
            .as_ref()
            .map(|registry| registry.recipes.as_slice()),
    );
    if !freeform_plan_needs_ambiguity_resolution(&deterministic) {
        return deterministic;
    }
    let Some(provider) = configured_planner_provider(config) else {
        return deterministic;
    };

    match resolve_freeform_plan_with_provider(request, paths, config, provider, &deterministic) {
        Ok(plan) => plan,
        Err(error) => {
            deterministic.notes.push(format!(
                "Provider-assisted planner `{provider}` could not resolve this request: {error}"
            ));
            deterministic.notes.push(
                "No provider-produced tool call was used. Fill the placeholder values or run /chat for help."
                    .to_owned(),
            );
            deterministic
        }
    }
}

fn configured_planner_provider(config: &RocmCliConfig) -> Option<&str> {
    config
        .planner_provider
        .as_deref()
        .map(str::trim)
        .filter(|provider| !provider.is_empty())
        .filter(|provider| matches!(*provider, "local" | "openai" | "anthropic"))
}

fn freeform_plan_needs_ambiguity_resolution(plan: &StructuredRequestPlan) -> bool {
    plan.confidence != "high"
        || plan.actions.iter().any(|action| {
            action
                .args
                .iter()
                .any(|arg| arg.starts_with('<') && arg.ends_with('>'))
        })
}

fn resolve_freeform_plan_with_provider(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
    provider: &str,
    deterministic: &StructuredRequestPlan,
) -> Result<StructuredRequestPlan> {
    if provider != "local" && !config.provider_enabled(provider) {
        bail!(
            "cloud provider is disabled; run `rocm config enable-provider {provider}` before sending planner prompts"
        );
    }
    let prompt = build_provider_planner_prompt(request, deterministic);
    let response = providers::provider_chat(
        paths,
        provider,
        &providers::ChatRequest {
            model: None,
            messages: vec![providers::ChatMessage {
                role: "user".to_owned(),
                content: prompt,
            }],
            max_tokens: Some(512),
            rocm_tools: false,
        },
    )?;
    provider_planner_response_to_plan(request, provider, &response.content)
}

fn build_provider_planner_prompt(request: &str, deterministic: &StructuredRequestPlan) -> String {
    let next_tool_call = deterministic.actions.last().map_or_else(
        || "rocm examine".to_owned(),
        |action| format_structured_tool_call(action.tool, &action.args),
    );
    format!(
        "You are resolving an ambiguous rocm-cli request. Return only JSON with this shape: \
{{\"intent\":\"serve|install_sdk|install_driver|update|uninstall|inspect\",\
\"confidence\":\"high|medium|low\",\
\"tool_call\":{{\"tool\":\"rocm\",\"args\":[\"...\"]}},\
\"notes\":[\"short note\"]}}.\n\
Allowed rocm actions: examine; engines list; install sdk; install driver; update; serve; uninstall. Install sdk must include --prefix PATH chosen by the user, and may include --build-date YYYY-MM-DD or --version VERSION.\n\
Do not invent CPU fallback. Do not include shell commands. Do not include markdown.\n\
User request: {request}\n\
Deterministic planner intent: {}\n\
Deterministic next tool call: {next_tool_call}",
        deterministic.intent.label()
    )
}

#[derive(Debug, Deserialize)]
struct ProviderPlannerResponse {
    intent: String,
    #[serde(default)]
    confidence: Option<String>,
    tool_call: ProviderPlannerToolCall,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProviderPlannerToolCall {
    tool: String,
    args: Vec<String>,
}

fn provider_planner_response_to_plan(
    request: &str,
    provider: &str,
    content: &str,
) -> Result<StructuredRequestPlan> {
    let response = parse_provider_planner_response(content)?;
    let args = validate_provider_planner_tool_call(&response.tool_call)?;
    let intent = planner_intent_from_provider_response(&response.intent, &args)?;
    let approval_required = provider_planner_args_require_approval(&args);
    let action = if approval_required {
        PlannedToolCall::approval_required(
            "Provider-resolved action",
            args,
            "provider-assisted ambiguity resolution; review before running",
        )
    } else {
        PlannedToolCall::read_only(
            "Provider-resolved action",
            args,
            "provider-assisted read-only plan",
        )
    };
    let mut notes = vec![
        "Provider-assisted planning is optional and only uses the configured planner provider."
            .to_owned(),
        "The provider output was reduced to a validated rocm tool call; no shell command is executed."
            .to_owned(),
    ];
    notes.extend(
        response
            .notes
            .into_iter()
            .filter(|note| !note.trim().is_empty()),
    );
    Ok(StructuredRequestPlan {
        request: request.trim().to_owned(),
        planner: format!("hybrid-parser-v1 + provider:{provider}"),
        provider_assisted: true,
        intent,
        confidence: sanitize_provider_confidence(response.confidence.as_deref()),
        approval: if approval_required {
            "required before execution"
        } else {
            "not required for inspection"
        },
        parsed: vec![("provider".to_owned(), provider.to_owned())],
        actions: vec![action],
        notes,
    })
}

fn parse_provider_planner_response(content: &str) -> Result<ProviderPlannerResponse> {
    let json = strip_json_fence(content.trim());
    serde_json::from_str::<ProviderPlannerResponse>(json)
        .context("provider planner response was not valid JSON")
}

fn strip_json_fence(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(rest) = trimmed.strip_prefix("```json")
        && let Some((json, _)) = rest.trim_start().split_once("```")
    {
        return json.trim();
    }
    if let Some(rest) = trimmed.strip_prefix("```")
        && let Some((json, _)) = rest.trim_start().split_once("```")
    {
        return json.trim();
    }
    trimmed
}

fn validate_provider_planner_tool_call(call: &ProviderPlannerToolCall) -> Result<Vec<String>> {
    if call.tool != "rocm" {
        bail!("provider planner returned unsupported tool `{}`", call.tool);
    }
    if call.args.is_empty() {
        bail!("provider planner returned an empty tool call");
    }
    if call
        .args
        .iter()
        .any(|arg| arg.trim().is_empty() || arg.contains('\0'))
    {
        bail!("provider planner returned an invalid empty or NUL argument");
    }
    let argv = std::iter::once("rocm".to_owned())
        .chain(call.args.iter().cloned())
        .collect::<Vec<_>>();
    Cli::try_parse_from(argv)
        .context("provider planner returned a rocm command that is not valid")?;

    match call.args.as_slice() {
        [command] if command == "examine" => {}
        [command, subcommand] if command == "engines" && subcommand == "list" => {}
        [command, subcommand, ..] if command == "install" && subcommand == "sdk" => {
            validate_chat_rocm_command_safety(&call.args)?;
        }
        [command, subcommand, ..] if command == "install" && subcommand == "driver" => {}
        [command, ..] if command == "update" => {}
        [command, ..] if command == "serve" => {
            if chat_cli_has_flag(&call.args, "--allow-public-bind") {
                bail!("provider planner cannot request public network binding");
            }
            if serve_args_request_cpu_device(&call.args) {
                bail!(
                    "provider planner cannot request CPU execution; rocm-cli requires ROCm GPU execution"
                );
            }
            if let Some(host) = chat_cli_arg_value(&call.args, "--host")
                && !is_loopback_host(host)
            {
                bail!("provider planner cannot request non-local host `{host}`");
            }
            if chat_cli_has_flag(&call.args, "--foreground")
                || !chat_cli_has_flag(&call.args, "--managed")
            {
                bail!("provider planner must request managed serving with --managed");
            }
        }
        [command, ..] if command == "uninstall" => {}
        _ => bail!(
            "provider planner returned an unsupported rocm action: {}",
            format_structured_tool_call("rocm", &call.args)
        ),
    }
    Ok(call.args.clone())
}

fn planner_intent_from_provider_response(intent: &str, args: &[String]) -> Result<PlannerIntent> {
    let args_intent = planner_intent_from_args(args)?;
    let declared = match intent.trim().to_ascii_lowercase().as_str() {
        "serve" => PlannerIntent::Serve,
        "install_sdk" | "install sdk" => PlannerIntent::InstallSdk,
        "install_driver" | "install driver" => PlannerIntent::InstallDriver,
        "update" => PlannerIntent::Update,
        "uninstall" => PlannerIntent::Uninstall,
        "inspect" | "examine" => PlannerIntent::Inspect,
        _ => bail!("provider planner returned unsupported intent `{intent}`"),
    };
    if declared != args_intent {
        bail!(
            "provider planner intent `{}` did not match tool call intent `{}`",
            declared.label(),
            args_intent.label()
        );
    }
    Ok(args_intent)
}

fn planner_intent_from_args(args: &[String]) -> Result<PlannerIntent> {
    match args.first().map(String::as_str) {
        Some("serve") => Ok(PlannerIntent::Serve),
        Some("install") if args.get(1).is_some_and(|arg| arg == "sdk") => {
            Ok(PlannerIntent::InstallSdk)
        }
        Some("install") if args.get(1).is_some_and(|arg| arg == "driver") => {
            Ok(PlannerIntent::InstallDriver)
        }
        Some("update") => Ok(PlannerIntent::Update),
        Some("uninstall") => Ok(PlannerIntent::Uninstall),
        Some("examine" | "engines") => Ok(PlannerIntent::Inspect),
        _ => bail!("unsupported provider planner args"),
    }
}

fn provider_planner_args_require_approval(args: &[String]) -> bool {
    matches!(
        args.first().map(String::as_str),
        Some("install" | "serve" | "uninstall")
    ) || args.iter().any(|arg| arg == "--apply")
}

fn sanitize_provider_confidence(confidence: Option<&str>) -> &'static str {
    match confidence
        .unwrap_or("medium")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "high" => "high",
        "low" => "low",
        _ => "medium",
    }
}

fn render_structured_request_plan(plan: &StructuredRequestPlan, paths: &AppPaths) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "request plan");
    let _ = writeln!(output, "  request: {}", plan.request);
    let _ = writeln!(output, "  planner: {}", plan.planner);
    let _ = writeln!(output, "  tool_schema: {}", providers::ROCM_TOOL_SCHEMA_ID);
    let _ = writeln!(output, "  intent: {}", plan.intent.label());
    let _ = writeln!(output, "  confidence: {}", plan.confidence);
    let _ = writeln!(output, "  approval: {}", plan.approval);
    if !plan.parsed.is_empty() {
        let _ = writeln!(output, "  parsed:");
        for (key, value) in &plan.parsed {
            let _ = writeln!(output, "    {key}: {value}");
        }
    }
    let _ = writeln!(output, "  plan:");
    if plan.actions.is_empty() {
        let _ = writeln!(output, "    No ROCm action selected.");
    } else {
        for (index, action) in plan.actions.iter().enumerate() {
            let _ = writeln!(output, "    {}. {}", index + 1, action.title);
            let _ = writeln!(
                output,
                "       tool_call: {}",
                format_structured_tool_call(action.tool, &action.args)
            );
            let _ = writeln!(output, "       approval: {}", action.approval);
            let _ = writeln!(output, "       reason: {}", action.reason);
        }
    }
    if let Some(next) = plan.actions.last() {
        let _ = writeln!(
            output,
            "  next_tool_call: {}",
            format_structured_tool_call(next.tool, &next.args)
        );
        let _ = writeln!(output, "  next_tool_approval: {}", next.approval);
    }
    if plan.intent == PlannerIntent::Uninstall {
        let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    }
    for note in &plan.notes {
        let _ = writeln!(output, "  note: {note}");
    }
    output
}

fn planner_is_serve_request(lower: &str) -> bool {
    lower.contains("serve")
        || lower.contains("run a small local model")
        || lower.contains("local model on cpu")
        || lower.contains("start a local model")
}

fn planner_is_install_sdk_request(lower: &str) -> bool {
    let installish = contains_planner_word(lower, "install")
        || contains_planner_word(lower, "setup")
        || lower.contains("set up");
    let target = lower.contains("therock") || contains_planner_word(lower, "sdk");
    installish && target
}

fn planner_is_inspect_request(lower: &str) -> bool {
    let inspectish = contains_planner_word(lower, "inspect")
        || contains_planner_word(lower, "check")
        || contains_planner_word(lower, "status")
        || contains_planner_word(lower, "examine")
        || contains_planner_word(lower, "which")
        || contains_planner_word(lower, "where")
        || lower.contains("what is installed")
        || lower.contains("what's installed")
        || lower.contains("is installed")
        || lower.contains("is rocm installed")
        || lower.contains("is therock installed")
        || lower.contains("is the rock installed");
    let target = lower.contains("rocm")
        || lower.contains("therock")
        || lower.contains("the rock")
        || lower.contains("gpu")
        || lower.contains("driver")
        || lower.contains("setup")
        || lower.contains("installed")
        || lower.contains("this computer")
        || lower.contains("this machine");
    inspectish && target
}

fn planner_mentions_comfyui(lower: &str) -> bool {
    any_substring(lower, &["comfyui", "comfy ui", "comfy"])
}

fn planner_requests_comfyui_install(lower: &str) -> bool {
    any_substring(
        lower,
        &[
            "can you setup",
            "can you set up",
            "please setup",
            "please set up",
            "setup comfyui for me",
            "set up comfyui for me",
            "install comfyui",
            "download comfyui",
        ],
    )
}

fn contains_planner_word(text: &str, expected: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.')
        .any(|word| word == expected)
}

fn infer_small_local_model_from_request(lower: &str) -> Option<&'static str> {
    (lower.contains("small local model") || lower.contains("tiny local model"))
        .then_some("sshleifer/tiny-gpt2")
}

fn infer_recommended_assistant_model_from_request(lower: &str) -> Option<&'static str> {
    (lower.contains("start a local model")
        || lower.contains("local assistant")
        || lower.contains("recommended model")
        || lower.contains("serve an llm")
        || lower.contains("serve a local llm"))
    .then_some(providers::LEMONADE_ASSISTANT_MODEL_ID)
}

fn generic_model_phrase(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "a model" | "an llm" | "a local llm" | "local assistant" | "the assistant"
    )
}

fn infer_device_policy_from_request(lower: &str) -> Option<&'static str> {
    if lower.contains("cpu") {
        Some("cpu")
    } else if lower.contains("gpu preferred") || lower.contains("prefer gpu") {
        Some("gpu_preferred")
    } else if lower.contains("gpu") {
        Some("gpu")
    } else {
        None
    }
}

fn format_structured_tool_call(tool: &str, args: &[String]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(tool.to_owned());
    parts.extend(args.iter().map(|arg| quote_tool_arg(arg)));
    parts.join(" ")
}

fn quote_tool_arg(value: &str) -> String {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}

#[derive(Debug, Clone, Default)]
struct UninstallOptions {
    yes: bool,
    dry_run: bool,
    keep_binaries: bool,
    keep_config: bool,
    keep_data: bool,
    keep_cache: bool,
    force_dev_binaries: bool,
}

#[derive(Debug, Clone)]
struct UninstallPlanEntry {
    kind: &'static str,
    path: PathBuf,
}

#[derive(Debug, Clone, Default)]
struct UninstallPlan {
    actions: Vec<UninstallPlanEntry>,
    skipped: Vec<String>,
    warnings: Vec<String>,
}

fn build_uninstall_plan(paths: &AppPaths, options: &UninstallOptions) -> Result<UninstallPlan> {
    let mut plan = UninstallPlan::default();

    if options.keep_binaries {
        plan.skipped
            .push("binary removal disabled by --keep-binaries".to_owned());
    } else {
        let current_exe =
            daemon_binary_path().context("failed to discover current rocm executable")?;
        if is_dev_binary_layout(&current_exe) && !options.force_dev_binaries {
            plan.skipped.push(format!(
                "binary removal skipped because {} looks like a cargo target build; pass --force-dev-binaries to remove sibling debug/release binaries",
                current_exe.display()
            ));
        } else {
            for path in collect_installed_binary_candidates(&current_exe)? {
                if rocm_core::runtime_is_windows() && path == current_exe {
                    plan.skipped.push(format!(
                        "skipping running executable on Windows: {}",
                        path.display()
                    ));
                    continue;
                }
                plan.actions.push(UninstallPlanEntry {
                    kind: "binary",
                    path,
                });
            }
        }
    }

    for (keep, kind, path) in [
        (options.keep_config, "config", paths.config_dir.clone()),
        (options.keep_data, "data", paths.data_dir.clone()),
        (options.keep_cache, "cache", paths.cache_dir.clone()),
    ] {
        if keep {
            plan.skipped
                .push(format!("{kind} removal disabled by command line flag"));
            continue;
        }
        if path.exists() {
            plan.actions.push(UninstallPlanEntry { kind, path });
        } else {
            plan.skipped
                .push(format!("{kind} path not present: {}", path.display()));
        }
    }

    let managed_services = load_managed_services(paths).unwrap_or_default();
    if !managed_services.is_empty() {
        plan.warnings.push(format!(
            "{} managed service record(s) exist under {}; background processes are not stopped automatically in this pass",
            managed_services.len(),
            paths.services_dir().display()
        ));
    }

    plan.actions
        .sort_by(|left, right| left.path.cmp(&right.path));
    plan.actions.dedup_by(|left, right| left.path == right.path);
    Ok(plan)
}

pub(crate) fn render_uninstall_dry_run(paths: &AppPaths) -> Result<String> {
    let options = UninstallOptions {
        dry_run: true,
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan(paths, &options)?;
    Ok(render_uninstall_plan(&plan, &options))
}

fn render_uninstall_plan(plan: &UninstallPlan, options: &UninstallOptions) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Uninstall review");
    let _ = writeln!(output);
    if plan.actions.is_empty() {
        let _ = writeln!(output, "Nothing managed by rocm-cli would be removed.");
    } else {
        let _ = writeln!(output, "{} item(s) would be removed:", plan.actions.len());
        for entry in &plan.actions {
            let _ = writeln!(output, "  - {}: {}", entry.kind, entry.path.display());
        }
    }
    if !plan.warnings.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Please review:");
    }
    for warning in &plan.warnings {
        let _ = writeln!(output, "  - {warning}");
    }
    if !plan.skipped.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Left alone:");
    }
    for skipped in &plan.skipped {
        let _ = writeln!(output, "  - {skipped}");
    }
    if options.dry_run {
        let _ = writeln!(output);
        let _ = writeln!(output, "Choose Review uninstall to approve removal.");
    }
    output
}

fn confirm_uninstall() -> Result<bool> {
    print!("Proceed with uninstall? [y/N]: ");
    io::stdout()
        .flush()
        .context("failed to flush uninstall prompt")?;
    let mut response = String::new();
    io::stdin()
        .read_line(&mut response)
        .context("failed to read uninstall confirmation")?;
    let normalized = response.trim().to_ascii_lowercase();
    Ok(matches!(normalized.as_str(), "y" | "yes"))
}

fn collect_installed_binary_candidates(current_exe: &Path) -> Result<Vec<PathBuf>> {
    let binary_dir = current_exe
        .parent()
        .context("current executable has no parent directory")?;
    let mut binaries = Vec::new();
    for entry in fs::read_dir(binary_dir)
        .with_context(|| format!("failed to read {}", binary_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !path.is_file() {
            continue;
        }
        if is_rocm_install_entry_name(name) {
            binaries.push(path);
        }
    }
    binaries.sort();
    Ok(binaries)
}

fn is_rocm_install_entry_name(name: &str) -> bool {
    if name == ".rocm-cli-manifest" {
        return true;
    }
    let normalized = name.strip_suffix(".exe").unwrap_or(name);
    normalized == "rocm"
        || normalized == "rocmd"
        || normalized == "rocm-codex"
        || normalized.starts_with("rocm-engine-")
}

fn is_dev_binary_layout(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(parent_name) = parent.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    if parent_name != "debug" && parent_name != "release" {
        return false;
    }
    parent
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        == Some("target")
}

fn remove_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else if metadata.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else {
        bail!("unsupported filesystem entry for {}", path.display());
    }
    Ok(())
}

pub(crate) const fn engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        ("pytorch", "TheRock PyTorch local serving engine"),
        ("llama.cpp", "external GGUF serving engine for llama-server"),
        (
            "lemonade",
            "default embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
        (
            "sglang",
            "Linux/WSL ROCm GPU serving engine through external SGLang",
        ),
        (
            "atom",
            "Linux/WSL ROCm GPU serving engine through external ATOM Python",
        ),
    ]
}

fn infer_engine_from_request(lower: &str) -> Option<&str> {
    for engine in ["pytorch", "llama.cpp", "lemonade", "vllm", "sglang", "atom"] {
        if lower.contains(engine) {
            return Some(engine);
        }
    }
    if lower.contains("llama cpp") {
        return Some("llama.cpp");
    }
    None
}

fn infer_model_from_request(request: &str) -> Option<&str> {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let serve_index = lower.find("serve")?;
    let after = trimmed.get(serve_index + "serve".len()..)?.trim();
    if after.is_empty() {
        return None;
    }
    let end = after
        .find(" with ")
        .or_else(|| after.find(" using "))
        .unwrap_or(after.len());
    let model = after[..end].trim();
    (!model.is_empty()).then_some(model)
}

const fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Local => "local",
        Provider::Anthropic => "anthropic",
        Provider::Openai => "openai",
    }
}

fn resolve_engine_binary_path(engine: &str) -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    resolve_engine_binary_path_with_paths(engine, &paths)
}

fn resolve_engine_binary_path_with_paths(engine: &str, paths: &AppPaths) -> Result<PathBuf> {
    if let Some(path) = find_engine_plugin_binary(engine, engine_plugin_dirs(paths))? {
        return Ok(path);
    }
    if let Some(reason) = missing_packaged_engine_reason(engine) {
        bail!("{reason}");
    }
    engine_binary_path(engine)
}

const fn missing_packaged_engine_reason(_engine: &str) -> Option<String> {
    None
}

fn find_engine_plugin_binary<I, P>(engine: &str, plugin_dirs: I) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
}

impl From<WatcherModeArg> for WatcherMode {
    fn from(value: WatcherModeArg) -> Self {
        match value {
            WatcherModeArg::Observe => Self::Observe,
            WatcherModeArg::Propose => Self::Propose,
            WatcherModeArg::Contained => Self::Contained,
        }
    }
}

fn engine_request<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    engine_request_with_env_root(paths, engine, method, request, None)
}

fn engine_request_with_env_root<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
    env_root: Option<&Path>,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let stream_progress = matches!(&method, EngineMethod::Install);
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request)
            .context("failed to encode engine request payload")?,
    };
    if let Some(envelope) = with_scoped_builtin_engine_env(paths, env_root, || {
        builtin_engine_request(engine, &envelope)
    }) {
        return decode_engine_response(envelope);
    }

    let engine_binary = resolve_engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;
    let mut command = ProcessCommand::new(engine_binary);
    command.arg("stdio");
    if let Some(paths) = paths {
        apply_app_path_env(&mut command, paths);
    }
    if let Some(env_root) = env_root {
        command.env("ROCM_CLI_ENGINE_ENVS_ROOT", env_root);
    }
    if stream_progress {
        command.env("ROCM_ENGINE_PROGRESS_STDERR", "1");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn engine stdio process")?;

    {
        let mut stdin = child
            .stdin
            .take()
            .context("engine stdio child did not expose stdin")?;
        serde_json::to_writer(&mut stdin, &envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
    }

    let stderr_handle = child.stderr.take().map(|stderr| {
        thread::spawn(move || {
            let mut reader = io::BufReader::new(stderr);
            let mut collected = Vec::new();
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(text) = std::str::from_utf8(&line) {
                            eprint!("{text}");
                        } else {
                            let _ = io::stderr().write_all(&line);
                        }
                        collected.extend_from_slice(&line);
                    }
                    Err(error) => {
                        let message = format!("failed to read engine progress: {error}\n");
                        eprint!("{message}");
                        collected.extend_from_slice(message.as_bytes());
                        break;
                    }
                }
            }
            collected
        })
    });
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .context("engine stdio child did not expose stdout")?
        .read_to_end(&mut stdout)
        .context("failed to read engine stdio response")?;
    let status = child
        .wait()
        .context("failed waiting for engine stdio response")?;
    let stderr = stderr_handle
        .map(|handle| handle.join().unwrap_or_else(|_| Vec::new()))
        .unwrap_or_default();

    if !status.success() && stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {status}");
        }
        bail!("engine stdio process exited with status {status}: {stderr}");
    }
    let envelope: EngineResponseEnvelope = serde_json::from_slice(&stdout).with_context(|| {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        if stderr.is_empty() {
            "failed to parse engine response envelope".to_owned()
        } else {
            format!("failed to parse engine response envelope; stderr: {stderr}")
        }
    })?;
    decode_engine_response(envelope)
}

fn decode_engine_response<R>(envelope: EngineResponseEnvelope) -> Result<R>
where
    R: DeserializeOwned,
{
    if !envelope.ok {
        let error = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |value| format!("{}: {}", value.code, value.message),
        );
        bail!("{error}");
    }
    let data = envelope
        .data
        .context("engine response envelope did not contain data")?;
    serde_json::from_value(data).context("failed to decode engine response payload")
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn set_path(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
    fn drop(&mut self) {
        unsafe {
            match self.previous.as_ref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn with_scoped_builtin_engine_env<R>(
    paths: Option<&AppPaths>,
    env_root: Option<&Path>,
    f: impl FnOnce() -> R,
) -> R {
    if paths.is_none() && env_root.is_none() {
        return f();
    }
    let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("builtin engine env lock poisoned");
    let mut vars = Vec::new();
    if let Some(paths) = paths {
        for (key, value) in app_path_env_vars(paths) {
            vars.push(ScopedEnvVar::set_path(key, value));
        }
    }
    if let Some(env_root) = env_root {
        vars.push(ScopedEnvVar::set_path(
            "ROCM_CLI_ENGINE_ENVS_ROOT",
            env_root,
        ));
    }
    let result = f();
    drop(vars);
    result
}

fn builtin_engine_request(
    engine: &str,
    envelope: &EngineRequestEnvelope,
) -> Option<EngineResponseEnvelope> {
    match engine {
        "atom" => Some(rocm_engine_atom::builtin_handle_envelope(envelope.clone())),
        "lemonade" => Some(rocm_engine_lemonade::builtin_handle_envelope(
            envelope.clone(),
        )),
        "llama.cpp" => Some(rocm_engine_llama_cpp::builtin_handle_envelope(
            envelope.clone(),
        )),
        "pytorch" => Some(rocm_engine_pytorch::builtin_handle_envelope(
            envelope.clone(),
        )),
        "sglang" => Some(rocm_engine_sglang::builtin_handle_envelope(
            envelope.clone(),
        )),
        "vllm" => Some(rocm_engine_vllm::builtin_handle_envelope(envelope.clone())),
        _ => None,
    }
}

fn run_builtin_engine_stdio(engine: &str) -> Result<()> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read engine stdio request")?;
    let envelope: EngineRequestEnvelope =
        serde_json::from_str(&input).context("failed to parse engine stdio request")?;
    let response = builtin_engine_request(engine, &envelope)
        .with_context(|| format!("engine `{engine}` is not built into this rocm binary"))?;
    print!("{}", serde_json::to_string(&response)?);
    Ok(())
}

fn builtin_engine_available(engine: &str) -> bool {
    matches!(
        engine,
        "atom" | "lemonade" | "llama.cpp" | "pytorch" | "sglang" | "vllm"
    )
}

#[allow(clippy::too_many_arguments)]
fn run_builtin_engine_serve_http(
    engine: &str,
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: &str,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    let parsed_policy = parse_device_policy(Some(device_policy))?;
    match engine {
        "atom" => rocm_engine_atom::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            runtime_id,
            env_id,
            state_path,
            engine_recipe,
        ),
        "lemonade" => rocm_engine_lemonade::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "llama.cpp" => rocm_engine_llama_cpp::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            Some(device_policy_name(&parsed_policy).to_owned()),
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "pytorch" => rocm_engine_pytorch::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            env_id,
            runtime_id,
            state_path,
            engine_recipe,
        ),
        "sglang" => rocm_engine_sglang::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            runtime_id,
            env_id,
            state_path,
            engine_recipe,
        ),
        "vllm" => rocm_engine_vllm::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            runtime_id,
            env_id,
            state_path,
            engine_recipe,
        ),
        other => bail!("engine `{other}` is not built into this rocm binary"),
    }
}

#[allow(clippy::too_many_arguments)]
fn builtin_engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    state_path: &Path,
    log_path: Option<&Path>,
) -> Result<Vec<String>> {
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
        device_policy_name(device_policy).to_owned(),
    ];
    if env_id.is_none() {
        args.extend(optional_arg("--runtime-id", runtime_id));
    }
    args.extend(optional_arg("--env-id", env_id));
    args.extend(engine_recipe_json_arg(engine_recipe)?);
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    if let Some(log_path) = log_path {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    Ok(args)
}

fn parse_device_policy(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "auto" | "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => bail!(
            "rocm serve requires ROCm GPU execution; CPU mode is not a fallback path in rocm-cli"
        ),
        other => bail!("unsupported device policy: {other}"),
    }
}

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

#[derive(Debug, Clone)]
struct EngineSelection {
    runtime_id: Option<String>,
    env_id: Option<String>,
    source: Option<String>,
}

fn resolve_engine_selection(
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> EngineSelection {
    if let Some(env_id) = env_id {
        return EngineSelection {
            runtime_id: None,
            env_id: Some(env_id.to_owned()),
            source: Some("cli_env_id".to_owned()),
        };
    }
    if let Some(runtime_id) = runtime_id {
        return EngineSelection {
            runtime_id: Some(runtime_id.to_owned()),
            env_id: None,
            source: Some("cli_runtime_id".to_owned()),
        };
    }

    if let Some(runtime_key) = config.active_runtime_key.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_key.clone()),
            env_id: None,
            source: Some("config_active_runtime_key".to_owned()),
        };
    }

    if let Some(entry) = config.engine_config(engine) {
        if let Some(env_id) = entry.preferred_env_id.as_ref() {
            return EngineSelection {
                runtime_id: None,
                env_id: Some(env_id.clone()),
                source: Some("config_preferred_env_id".to_owned()),
            };
        }
        if let Some(runtime_id) = entry.preferred_runtime_id.as_ref() {
            return EngineSelection {
                runtime_id: Some(runtime_id.clone()),
                env_id: None,
                source: Some("config_preferred_runtime_id".to_owned()),
            };
        }
        if let Some(env_id) = entry.last_installed_env_id.as_ref() {
            return EngineSelection {
                runtime_id: None,
                env_id: Some(env_id.clone()),
                source: Some("config_last_installed_env_id".to_owned()),
            };
        }
        if let Some(runtime_id) = entry.last_installed_runtime_id.as_ref() {
            return EngineSelection {
                runtime_id: Some(runtime_id.clone()),
                env_id: None,
                source: Some("config_last_installed_runtime_id".to_owned()),
            };
        }
    }

    if let Some(runtime_id) = config.default_runtime_id.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_id.clone()),
            env_id: None,
            source: Some("config_default_runtime_id".to_owned()),
        };
    }

    EngineSelection {
        runtime_id: None,
        env_id: None,
        source: None,
    }
}

fn validate_engine_selection_runtime(
    paths: &AppPaths,
    mut selection: EngineSelection,
) -> Result<EngineSelection> {
    if let Some(runtime_id) = selection.runtime_id.as_deref() {
        let source = selection.source.as_deref().unwrap_or("runtime selection");
        selection.runtime_id = Some(resolve_runtime_selector_to_exact_key(
            paths, runtime_id, source,
        )?);
    } else if selection.env_id.is_none()
        && let Some(runtime_key) = single_ready_runtime_key(paths)?
    {
        selection.runtime_id = Some(runtime_key);
        selection.source = Some("single_ready_runtime".to_owned());
    }
    Ok(selection)
}

fn single_ready_runtime_key(paths: &AppPaths) -> Result<Option<String>> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    recover_setup_runtime_registration(paths, &config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let ready = manifests
        .iter()
        .filter(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok())
        .collect::<Vec<_>>();
    Ok(match ready.as_slice() {
        [manifest] => Some(manifest.runtime_key.clone()),
        _ => None,
    })
}

fn resolve_runtime_selector_to_exact_key(
    paths: &AppPaths,
    selector: &str,
    source: &str,
) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    match select_runtime_manifest(&manifests, selector) {
        Ok(manifest) => Ok(manifest.runtime_key.clone()),
        Err(error) => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            if recover_setup_runtime_registration(paths, &config)?.is_some() {
                let manifests = therock::load_runtime_manifests(paths)?;
                if let Ok(manifest) = select_runtime_manifest(&manifests, selector) {
                    return Ok(manifest.runtime_key.clone());
                }
            }
            bail!(
                "runtime selector `{selector}` from {source} is not an exact usable runtime: {error}; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id <runtime_key>"
            )
        }
    }
}

fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

fn engine_recipe_json_arg(engine_recipe: Option<&EngineRecipeHint>) -> Result<Vec<String>> {
    match engine_recipe {
        Some(engine_recipe) => Ok(vec![
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe).context("failed to encode engine recipe hint")?,
        ]),
        None => Ok(Vec::new()),
    }
}

fn parse_engine_recipe_json_arg(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|value| {
            serde_json::from_str(&value).context("failed to parse --engine-recipe-json payload")
        })
        .transpose()
}

fn app_path_env_vars(paths: &AppPaths) -> [(&'static str, &Path); 3] {
    [
        ("ROCM_CLI_CONFIG_DIR", paths.config_dir.as_path()),
        ("ROCM_CLI_DATA_DIR", paths.data_dir.as_path()),
        ("ROCM_CLI_CACHE_DIR", paths.cache_dir.as_path()),
    ]
}

#[cfg_attr(not(windows), allow(dead_code))]
fn app_path_env_var_values(
    paths: &AppPaths,
    env_root: Option<&Path>,
) -> Vec<(&'static str, PathBuf)> {
    let mut vars = app_path_env_vars(paths)
        .into_iter()
        .map(|(key, value)| (key, value.to_path_buf()))
        .collect::<Vec<_>>();
    if let Some(env_root) = env_root {
        vars.push(("ROCM_CLI_ENGINE_ENVS_ROOT", env_root.to_path_buf()));
    }
    vars
}

#[cfg_attr(not(windows), allow(dead_code))]
fn app_path_env_var_refs<'a>(vars: &'a [(&'static str, PathBuf)]) -> Vec<(&'static str, &'a Path)> {
    vars.iter()
        .map(|(key, value)| (*key, value.as_path()))
        .collect()
}

/// Argv passed to the embedded `rocmd` library to run the real foreground
/// automation loop (the same path as `rocmd run --automations-enabled`).
fn daemon_run_argv() -> Vec<OsString> {
    vec![
        OsString::from("rocmd"),
        OsString::from("run"),
        OsString::from("--automations-enabled"),
    ]
}

fn managed_service_launcher_path() -> Result<PathBuf> {
    let current_exe = daemon_binary_path()?;
    if rocm_core::runtime_is_windows() {
        return Ok(rocm_core::normalize_runtime_path_for_storage(&current_exe));
    }
    Ok(current_exe)
}

fn apply_app_path_env(command: &mut ProcessCommand, paths: &AppPaths) {
    for (key, value) in app_path_env_vars(paths) {
        command.env(key, value);
    }
}

fn wait_for_service_http_ready(
    engine: &str,
    host: &str,
    port: u16,
    canonical_model_id: &str,
    timeout: Duration,
) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        for path in service_http_readiness_paths(engine) {
            if let Ok((status, body)) =
                http_get_local_service(host, port, path, Duration::from_millis(750))
                && service_http_readiness_response_ready(
                    engine,
                    path,
                    status,
                    &body,
                    canonical_model_id,
                )
            {
                return true;
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    false
}

fn service_http_readiness_paths(engine: &str) -> &'static [&'static str] {
    match engine {
        "lemonade" => &["/v1/health", "/v1/models"],
        "llama.cpp" => &["/v1/models", "/health"],
        "pytorch" => &["/v1/models", "/healthz"],
        _ => &["/v1/models", "/v1/health", "/health", "/healthz"],
    }
}

fn http_get_local_service(
    host: &str,
    port: u16,
    path: &str,
    timeout: Duration,
) -> Result<(u16, String)> {
    let mut stream = connect_tcp_stream(host, port, timeout)?;
    let host_header = format_host_port(host, port);
    let request =
        format!("GET {path} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n\r\n");
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write service readiness request")?;
    let response = read_tcp_stream_to_string(&mut stream)
        .context("failed to read service readiness response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or((response.as_str(), ""));
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status, body.to_owned()))
}

fn http_post_local_service_json(
    host: &str,
    port: u16,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<(u16, String)> {
    let mut stream = connect_tcp_stream(host, port, timeout)?;
    let host_header = format_host_port(host, port);
    let body = serde_json::to_string(body).context("failed to serialize service request")?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write service request")?;
    let response =
        read_tcp_stream_to_string(&mut stream).context("failed to read service response")?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or((response.as_str(), ""));
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status, body.to_owned()))
}

fn service_http_readiness_response_ready(
    engine: &str,
    path: &str,
    status: u16,
    body: &str,
    canonical_model_id: &str,
) -> bool {
    if status != 200 {
        return false;
    }
    match (engine, path) {
        ("lemonade", "/v1/health") => lemonade_health_ready_for_model(body, canonical_model_id),
        ("lemonade", "/v1/models") => model_list_ready_for_model(body, canonical_model_id, true),
        (_, "/v1/models") => model_list_ready_for_model(body, canonical_model_id, false),
        _ => false,
    }
}

fn lemonade_health_ready_for_model(body: &str, canonical_model_id: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else {
        return false;
    };
    value
        .get("all_models_loaded")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                let name_matches = ["model_name", "id", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(serde_json::Value::as_str))
                    .any(|loaded| service_model_names_match(loaded, canonical_model_id));
                name_matches && service_model_reports_rocm_backend(model)
            })
        })
}

fn model_list_ready_for_model(
    body: &str,
    canonical_model_id: &str,
    require_rocm_backend: bool,
) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else {
        return false;
    };
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                let name_matches = ["id", "model", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(serde_json::Value::as_str))
                    .any(|loaded| service_model_names_match(loaded, canonical_model_id));
                name_matches && (!require_rocm_backend || service_model_reports_rocm_backend(model))
            })
        })
}

fn service_model_reports_rocm_backend(model: &serde_json::Value) -> bool {
    model
        .get("recipe_options")
        .and_then(|options| options.get("llamacpp_backend"))
        .or_else(|| model.get("llamacpp_backend"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|backend| backend.trim().to_ascii_lowercase().starts_with("rocm"))
}

fn service_model_names_match(left: &str, right: &str) -> bool {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if left.eq_ignore_ascii_case(right) {
        return true;
    }
    let left = left
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .to_ascii_lowercase();
    let right = right
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .to_ascii_lowercase();
    left.contains(&right) || right.contains(&left)
}

fn treat_as_natural_language(args: &[String]) -> bool {
    const STRUCTURED: &[&str] = &[
        "examine",
        "diagnose",
        "fix",
        "status",
        "completions",
        "bridge-snapshot",
        "sandbox-run",
        "mcp-call",
        "__engine-serve-http",
        "__engine-stdio",
        "bootstrap",
        "setup",
        "chat",
        "install",
        "update",
        "runtimes",
        "engines",
        "model",
        "models",
        "serve",
        "comfyui",
        "comfy",
        "services",
        "automations",
        "config",
        "logs",
        "daemon",
        "dash",
        "uninstall",
        "help",
        "--help",
        "-h",
        "version",
        "--version",
        "-V",
    ];

    !args.is_empty() && !args[0].starts_with('-') && !STRUCTURED.contains(&args[0].as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // The previous `daemon_run_argv_targets_rocmd_run_with_automations` unit test
    // only re-asserted the literals `daemon_run_argv()` returns, so it tested
    // nothing real. The intended real behavior — that this argv actually drives
    // `rocmd` into its `run --automations-enabled` foreground loop — is proven
    // end-to-end by the `daemon_runs_real_foreground_loop` integration test in
    // tests/daemon_run.rs. A non-tautological unit test would require parsing the
    // argv through `rocmd::Cli`/`rocmd::Command`, but those clap structs are
    // crate-private in rocmd and exposing them (plus their private field types
    // like `SandboxToolArg`) is more than a trivial visibility change, so the
    // tautological unit test is removed in favor of the integration coverage.

    #[test]
    fn cli_command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn completions_generate_for_every_shell() {
        use clap_complete::Shell;
        // The hidden, internal-only verbs that `--help` omits and that must
        // therefore never appear in any generated completion script. These are
        // matched as substrings of the generated text, so the hidden `status`
        // verb is intentionally excluded here: it would collide with the
        // visible `comfyui status` / `setup status` subcommands. The hidden
        // `status` verb is covered by name equality in
        // `completion_command_excludes_hidden_subcommands` instead.
        let hidden = [
            "__engine-serve-http",
            "__engine-stdio",
            "mcp-call",
            "sandbox-run",
            "bridge-snapshot",
            "bootstrap",
        ];
        for &shell in Shell::value_variants() {
            let mut cmd = completion_command();
            let mut buf: Vec<u8> = Vec::new();
            clap_complete::generate(shell, &mut cmd, "rocm", &mut buf);
            assert!(!buf.is_empty(), "no completion output for {shell:?}");
            let output = String::from_utf8(buf).expect("completion output is valid UTF-8");
            for verb in hidden {
                assert!(
                    !output.contains(verb),
                    "hidden subcommand `{verb}` leaked into {shell:?} completions"
                );
            }
            // A known visible subcommand must still be present.
            assert!(
                output.contains("examine"),
                "visible subcommand `examine` missing from {shell:?} completions"
            );
        }
    }

    #[test]
    fn completion_command_excludes_hidden_subcommands() {
        let names: Vec<String> = completion_command()
            .get_subcommands()
            .map(|sc| sc.get_name().to_owned())
            .collect();
        // Visible subcommands are preserved.
        assert!(
            names.iter().any(|n| n == "examine"),
            "filtered command tree dropped a visible subcommand; got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "completions"),
            "filtered command tree dropped `completions`; got {names:?}"
        );
        // Hidden subcommands are excluded.
        for hidden in [
            "__engine-serve-http",
            "__engine-stdio",
            "mcp-call",
            "sandbox-run",
            "bridge-snapshot",
            "bootstrap",
            "status",
        ] {
            assert!(
                !names.iter().any(|n| n == hidden),
                "filtered command tree still exposes hidden subcommand `{hidden}`; got {names:?}"
            );
        }
        // The full derived command (used for runtime dispatch) keeps them.
        let full_names: Vec<String> = Cli::command()
            .get_subcommands()
            .map(|sc| sc.get_name().to_owned())
            .collect();
        assert!(
            full_names.iter().any(|n| n == "__engine-stdio"),
            "runtime command tree must retain hidden verbs for dispatch; got {full_names:?}"
        );
    }

    #[test]
    fn completions_command_is_structured_not_freeform() {
        use clap_complete::Shell;
        for &shell in Shell::value_variants() {
            let shell_arg = shell.to_string();
            let invocation =
                parse_freeform_invocation(&["completions".to_owned(), shell_arg.clone()]);
            assert!(
                !should_treat_as_freeform(&invocation),
                "`completions {shell_arg}` must dispatch as a structured command, not freeform"
            );
            // It must also parse cleanly through the structured clap parser.
            let cli = Cli::try_parse_from(["rocm", "completions", &shell_arg])
                .expect("completions <shell> should parse via Cli");
            assert!(matches!(cli.command, Some(Command::Completions { .. })));
        }
    }

    #[test]
    fn completions_rejects_unknown_shell() {
        // An unrecognized shell must be a hard parse error (non-zero exit in
        // `main`), not silently treated as natural language or accepted.
        let invocation =
            parse_freeform_invocation(&["completions".to_owned(), "notashell".to_owned()]);
        assert!(
            !should_treat_as_freeform(&invocation),
            "`completions notashell` must stay on the structured path so clap reports the error"
        );
        let parsed = Cli::try_parse_from(["rocm", "completions", "notashell"]);
        assert!(
            parsed.is_err(),
            "an unknown shell must fail to parse rather than being accepted"
        );
    }

    #[test]
    fn service_http_readiness_requires_loaded_lemonade_model() {
        let loading = json!({ "all_models_loaded": [] }).to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loading,
            "Qwen3-0.6B-GGUF"
        ));

        let loaded = json!({
            "all_models_loaded": [{
                "model_name": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "rocm" }
            }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loaded,
            "Qwen3-0.6B-GGUF"
        ));

        let loaded_cpu = json!({
            "all_models_loaded": [{
                "model_name": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "cpu" }
            }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/health",
            200,
            &loaded_cpu,
            "Qwen3-0.6B-GGUF"
        ));
    }

    #[test]
    fn service_http_readiness_requires_model_list_entry() {
        let empty = json!({ "data": [] }).to_string();
        assert!(!service_http_readiness_response_ready(
            "llama.cpp",
            "/v1/models",
            200,
            &empty,
            "tiny.gguf"
        ));

        let models = json!({ "data": [{ "id": "tiny.gguf" }] }).to_string();
        assert!(service_http_readiness_response_ready(
            "llama.cpp",
            "/v1/models",
            200,
            &models,
            "tiny.gguf"
        ));

        let lemonade_cpu_models = json!({
            "data": [{
                "id": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "cpu" }
            }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &lemonade_cpu_models,
            "Qwen3-0.6B-GGUF"
        ));

        let lemonade_rocm_models = json!({
            "data": [{
                "id": "Qwen3-0.6B-GGUF",
                "recipe_options": { "llamacpp_backend": "rocm" }
            }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &lemonade_rocm_models,
            "Qwen3-0.6B-GGUF"
        ));

        assert!(!service_http_readiness_response_ready(
            "llama.cpp",
            "/health",
            200,
            "OK",
            "tiny.gguf"
        ));
        assert!(!service_http_readiness_response_ready(
            "pytorch",
            "/healthz",
            200,
            "OK",
            "Qwen3-0.6B-GGUF"
        ));
    }

    #[test]
    fn lemonade_stop_unloads_selected_model_over_http() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut request = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&request);
                if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                    let expected = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if body.len() >= expected {
                        break;
                    }
                }
            }
            let text = String::from_utf8(request).context("request was not utf-8")?;
            sender.send(text).ok();
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"status\":\"success\",\"message\":\"ok\"}",
            )?;
            Ok(())
        });

        let (_root, paths) = test_paths("lemonade-stop-unload");
        let record = ManagedServiceRecord::new(
            &paths,
            "svc-qwen",
            "lemonade",
            "qwen",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            123,
            Some("therock-release".to_owned()),
            Some("lemonade-embeddable-10.6.0".to_owned()),
            Some("gpu_required".to_owned()),
        );
        unload_lemonade_service_model(&record)?;
        handle.join().expect("listener thread panicked")?;
        let request = receiver.recv_timeout(Duration::from_secs(1))?;
        assert!(request.starts_with("POST /v1/unload HTTP/1.1"));
        assert!(request.contains("\"model_name\":\"Qwen3-0.6B-GGUF\""));
        Ok(())
    }

    fn test_examine(os: &str, wsl: bool) -> ExamineSummary {
        ExamineSummary {
            os: os.to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("6.8.0-test".to_owned()),
            distro: Some("test distro".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "pytorch".to_owned(),
            detected_gfx_target: Some("gfx1201".to_owned()),
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: rocm_core::DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_missing".to_owned(),
                detail: Some("/dev/kfd missing".to_owned()),
            },
            legacy_rocm: rocm_core::LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
            },
            wsl: wsl.then_some(rocm_core::WslSummary {
                is_wsl: true,
                dxg_device: true,
                dxcore: true,
                librocdxg: false,
                rocdxg_dids: false,
                ldconfig_librocdxg: false,
                rocminfo: false,
                cargo: false,
                detail: Some("missing librocdxg".to_owned()),
            }),
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("/tmp/config"),
            data_dir: PathBuf::from("/tmp/data"),
            cache_dir: PathBuf::from("/tmp/cache"),
        }
    }

    fn test_app_paths() -> AppPaths {
        AppPaths {
            config_dir: PathBuf::from("C:/Users/test/.rocm"),
            data_dir: PathBuf::from("D:/rocm-data"),
            cache_dir: PathBuf::from("D:/rocm-data/cache"),
        }
    }

    #[test]
    fn app_path_env_vars_include_config_data_and_cache() {
        let paths = test_app_paths();
        let vars = app_path_env_vars(&paths);

        assert_eq!(vars[0], ("ROCM_CLI_CONFIG_DIR", paths.config_dir.as_path()));
        assert_eq!(vars[1], ("ROCM_CLI_DATA_DIR", paths.data_dir.as_path()));
        assert_eq!(vars[2], ("ROCM_CLI_CACHE_DIR", paths.cache_dir.as_path()));
    }

    #[test]
    fn app_path_env_var_values_include_engine_env_root_when_needed() {
        let paths = test_app_paths();
        let engine_root = PathBuf::from("D:/rocm-data/runtime/engines");
        let vars = app_path_env_var_values(&paths, Some(&engine_root));

        assert_eq!(
            vars.last().map(|(key, value)| (*key, value.as_path())),
            Some(("ROCM_CLI_ENGINE_ENVS_ROOT", engine_root.as_path()))
        );
    }

    #[test]
    fn uninstall_binary_matcher_includes_packaged_codex_binary() {
        assert!(is_rocm_install_entry_name("rocm-codex"));
        assert!(is_rocm_install_entry_name("rocm-codex.exe"));
    }

    #[test]
    fn hybrid_planner_normalizes_model_alias_and_structured_serve_call() {
        let plan = build_freeform_plan("serve qwen3.5 with vllm", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "Qwen/Qwen3.5-4B".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("model_alias".to_owned(), "qwen3.5".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("engine".to_owned(), "vllm".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("mode".to_owned(), "managed".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.approval == "required"
                && action.args
                    == vec![
                        "serve".to_owned(),
                        "Qwen/Qwen3.5-4B".to_owned(),
                        "--engine".to_owned(),
                        "vllm".to_owned(),
                        "--device".to_owned(),
                        "gpu_required".to_owned(),
                        "--managed".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_can_use_active_recipe_registry_aliases() {
        let mut recipe = resolve_builtin_model_recipe("tiny-gpt2").expect("tiny recipe");
        recipe.canonical_model_id = "Acme/SignedTiny".to_owned();
        recipe.aliases = vec!["signedtiny".to_owned()];
        recipe.source = "signed_recipe_index".to_owned();
        recipe.preferred_engines = vec!["llama.cpp".to_owned()];
        recipe.device_policy = "cpu_only".to_owned();
        recipe.dtype = "float16".to_owned();

        let plan = build_freeform_plan_with_recipes(
            "serve signedtiny",
            &RocmCliConfig::default(),
            Some(&[recipe]),
        );

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "Acme/SignedTiny".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("model_alias".to_owned(), "signedtiny".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("recipe_source".to_owned(), "signed_recipe_index".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("recipe_dtype".to_owned(), "float16".to_owned()))
        );
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("CPU mode is not offered"))
        );
    }

    #[test]
    fn hybrid_planner_builds_nightly_therock_install_call() {
        let plan = build_freeform_plan(
            "install the latest TheRock nightly for this GPU into D:\\ROCm\\therock_venvs",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::InstallSdk);
        assert!(
            plan.parsed
                .contains(&("channel".to_owned(), "nightly".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("prefix".to_owned(), "D:\\ROCm\\therock_venvs".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.title == "Install TheRock SDK"
                && action.approval == "required"
                && action.args
                    == vec![
                        "install".to_owned(),
                        "sdk".to_owned(),
                        "--channel".to_owned(),
                        "nightly".to_owned(),
                        "--format".to_owned(),
                        "wheel".to_owned(),
                        "--prefix".to_owned(),
                        "D:\\ROCm\\therock_venvs".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_builds_requested_therock_build_date_install_call() {
        let plan = build_freeform_plan(
            "install the TheRock wheel from date 06052026 into D:\\ROCm\\therock_venvs",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::InstallSdk);
        assert!(
            plan.parsed
                .contains(&("build_date".to_owned(), "2026-06-05".to_owned()))
        );
        assert!(plan.actions.iter().any(|action| {
            action.title == "Install TheRock SDK"
                && action.approval == "required"
                && action.args
                    == vec![
                        "install".to_owned(),
                        "sdk".to_owned(),
                        "--channel".to_owned(),
                        "release".to_owned(),
                        "--format".to_owned(),
                        "wheel".to_owned(),
                        "--prefix".to_owned(),
                        "D:\\ROCm\\therock_venvs".to_owned(),
                        "--build-date".to_owned(),
                        "2026-06-05".to_owned(),
                    ]
        }));
    }

    #[test]
    fn hybrid_planner_asks_for_folder_before_therock_install() {
        let plan = build_freeform_plan(
            "install the TheRock wheel from date 06052026",
            &RocmCliConfig::default(),
        );

        assert_eq!(plan.intent, PlannerIntent::Ask);
        assert!(plan.actions.is_empty());
        assert!(plan.approval.contains("install folder"));
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("install folder"))
        );
        assert!(
            plan.parsed
                .contains(&("build_date".to_owned(), "2026-06-05".to_owned()))
        );
    }

    #[test]
    fn hybrid_planner_handles_small_cpu_model_without_gpu_fallback() {
        let plan = build_freeform_plan("run a small local model on cpu", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert!(
            plan.parsed
                .contains(&("model".to_owned(), "sshleifer/tiny-gpt2".to_owned()))
        );
        assert!(
            plan.parsed
                .contains(&("device_policy".to_owned(), "cpu_not_supported".to_owned()))
        );
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("CPU mode is not offered"))
        );
    }

    #[test]
    fn hybrid_planner_defaults_generic_local_assistant_to_validated_qwen() {
        let plan = build_freeform_plan("start a local model", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert!(plan.parsed.contains(&(
            "model".to_owned(),
            providers::BUILTIN_ASSISTANT_MODEL_ID.to_owned()
        )));
        assert!(plan.actions.iter().any(|action| {
            action.approval == "required"
                && action.args
                    == vec![
                        "serve".to_owned(),
                        providers::BUILTIN_ASSISTANT_MODEL_ID.to_owned(),
                        "--engine".to_owned(),
                        "lemonade".to_owned(),
                        "--device".to_owned(),
                        "gpu_required".to_owned(),
                        "--managed".to_owned(),
                    ]
        }));
    }

    #[test]
    fn freeform_plan_next_action_rejects_cpu_mode_request() {
        assert!(
            freeform_plan_next_action("run a small local model on cpu", &RocmCliConfig::default())
                .is_none()
        );
    }

    #[test]
    fn freeform_plan_next_action_surfaces_approval_action() {
        let action =
            freeform_plan_next_action("serve qwen3.5 with llama.cpp", &RocmCliConfig::default())
                .expect("serve request should have next action");

        assert_eq!(action.title, "Launch local endpoint");
        assert!(action.approval_required);
        assert!(!action.has_placeholders);
        assert_eq!(
            action.args,
            vec![
                "serve".to_owned(),
                "Qwen/Qwen3.5-4B".to_owned(),
                "--engine".to_owned(),
                "llama.cpp".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
    }

    #[test]
    fn freeform_invocation_supports_leading_yes_for_natural_language_only() {
        let invocation = parse_freeform_invocation(&[
            "--yes".to_owned(),
            "please".to_owned(),
            "serve".to_owned(),
            "qwen3.5".to_owned(),
            "with".to_owned(),
            "llama.cpp".to_owned(),
        ]);

        assert!(invocation.approve);
        assert!(should_treat_as_freeform(&invocation));
        assert_eq!(
            invocation.request_args,
            vec![
                "please".to_owned(),
                "serve".to_owned(),
                "qwen3.5".to_owned(),
                "with".to_owned(),
                "llama.cpp".to_owned(),
            ]
        );

        let structured = parse_freeform_invocation(&[
            "--yes".to_owned(),
            "install".to_owned(),
            "sdk".to_owned(),
            "--dry-run".to_owned(),
        ]);
        assert!(structured.approve);
        assert!(!treat_as_natural_language(&structured.request_args));
        assert!(!should_treat_as_freeform(&structured));
    }

    #[test]
    fn freeform_invocation_rejects_unquoted_structured_command_names_after_yes() {
        let invalid_install = parse_freeform_invocation(&[
            "--yes".to_owned(),
            "install".to_owned(),
            "sdk".to_owned(),
            "--bad-flag".to_owned(),
        ]);
        let invalid_serve = parse_freeform_invocation(&[
            "--yes".to_owned(),
            "serve".to_owned(),
            "qwen3.5".to_owned(),
            "with".to_owned(),
            "llama.cpp".to_owned(),
        ]);

        assert!(!should_treat_as_freeform(&invalid_install));
        assert!(!should_treat_as_freeform(&invalid_serve));
    }

    #[test]
    fn freeform_invocation_rejects_flag_shaped_yes_request() {
        let help = parse_freeform_invocation(&["--yes".to_owned(), "--help".to_owned()]);
        let bad_flag = parse_freeform_invocation(&["--yes".to_owned(), "--bad-flag".to_owned()]);

        assert!(!should_treat_as_freeform(&help));
        assert!(!should_treat_as_freeform(&bad_flag));
    }

    #[test]
    fn freeform_execution_validation_rejects_placeholder_tool_calls() {
        let action = freeform_plan_next_action("serve", &RocmCliConfig::default())
            .expect("serve request should have next action");

        let error = validate_freeform_execution_action(&action)
            .unwrap_err()
            .to_string();

        assert!(action.has_placeholders);
        assert!(error.contains("placeholder values"));
        assert!(error.contains("rocm serve <model>"));
    }

    #[test]
    fn freeform_execution_validation_accepts_fully_structured_tool_call() -> Result<()> {
        let action =
            freeform_plan_next_action("serve qwen3.5 with llama.cpp", &RocmCliConfig::default())
                .expect("serve request should have next action");

        validate_freeform_execution_action(&action)?;
        assert_eq!(
            format_structured_tool_call("rocm", &action.args),
            "rocm serve Qwen/Qwen3.5-4B --engine llama.cpp --device gpu_required --managed"
        );
        Ok(())
    }

    #[test]
    fn freeform_execution_header_surfaces_explicit_approval_and_tool_call() {
        let action =
            freeform_plan_next_action("serve qwen3.5 with llama.cpp", &RocmCliConfig::default())
                .expect("serve request should have next action");
        let rendered = render_freeform_execution_header(&action);

        assert!(rendered.contains("execution"));
        assert!(rendered.contains("approval: granted by --yes"));
        assert!(rendered.contains(
            "tool_call: rocm serve Qwen/Qwen3.5-4B --engine llama.cpp --device gpu_required --managed"
        ));
    }

    #[test]
    fn hybrid_planner_driver_action_includes_yes_for_approved_execution() {
        let plan = build_freeform_plan(
            "install the linux driver with dkms",
            &RocmCliConfig::default(),
        );
        let action = plan
            .actions
            .iter()
            .find(|action| action.title == "Install driver")
            .expect("driver plan should include install action");

        assert_eq!(plan.intent, PlannerIntent::InstallDriver);
        assert_eq!(
            action.args,
            vec![
                "install".to_owned(),
                "driver".to_owned(),
                "--dkms".to_owned(),
                "--yes".to_owned(),
            ]
        );
    }

    #[test]
    fn hybrid_planner_unknown_request_is_read_only_inspection() {
        let plan = build_freeform_plan("what is installed here", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Inspect);
        assert!(
            plan.actions
                .iter()
                .all(|action| action.approval == "not required")
        );
        assert!(
            plan.actions
                .iter()
                .all(|action| action.args == vec!["examine".to_owned()])
        );
    }

    #[test]
    fn hybrid_planner_routes_common_status_questions_to_read_only_inspection() {
        for prompt in [
            "is rocm installed?",
            "which gpu is on my machine?",
            "where is therock installed?",
        ] {
            let plan = build_freeform_plan(prompt, &RocmCliConfig::default());

            assert_eq!(plan.intent, PlannerIntent::Inspect, "{prompt}");
            assert_eq!(plan.approval, "not required for inspection", "{prompt}");
            assert_eq!(plan.actions.len(), 1, "{prompt}");
            assert_eq!(plan.actions[0].approval, "not required", "{prompt}");
            assert_eq!(plan.actions[0].args, vec!["examine".to_owned()], "{prompt}");
        }
    }

    #[test]
    fn hybrid_planner_routes_comfyui_help_and_actions() {
        let status = build_freeform_plan("how do i setup comfyui", &RocmCliConfig::default());
        assert_eq!(status.intent, PlannerIntent::Inspect);
        assert_eq!(status.approval, "not required for inspection");
        assert_eq!(
            status.actions[0].args,
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert_eq!(status.actions[0].approval, "not required");

        let install =
            build_freeform_plan("can you setup comfyui for me", &RocmCliConfig::default());
        assert_eq!(install.approval, "required before installing ComfyUI");
        assert_eq!(
            install.actions[0].args,
            vec!["comfyui".to_owned(), "install".to_owned()]
        );
        assert_eq!(install.actions[0].approval, "required");

        let start = build_freeform_plan("can you start comfyui", &RocmCliConfig::default());
        assert_eq!(start.approval, "required before launch");
        assert_eq!(
            start.actions[0].args,
            vec!["comfyui".to_owned(), "start".to_owned()]
        );
        assert_eq!(start.actions[0].approval, "required");
    }

    #[test]
    fn hybrid_planner_casual_request_has_no_rocm_action() {
        let plan = build_freeform_plan("hi", &RocmCliConfig::default());

        assert_eq!(plan.intent, PlannerIntent::Ask);
        assert!(plan.actions.is_empty());
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("No ROCm action"))
        );
    }

    #[test]
    fn render_freeform_plan_exposes_structured_tool_calls() {
        let (_root, paths) = test_paths("hybrid-render");
        let rendered =
            render_freeform_plan("serve qwen3.5 with vllm", &paths, &RocmCliConfig::default());

        assert!(rendered.contains("planner: hybrid-parser-v1"));
        assert!(rendered.contains("tool_schema: rocm-tools-v0"));
        assert!(rendered.contains(
            "tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        assert!(rendered.contains(
            "next_tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        assert!(rendered.contains("next_tool_approval: required"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn provider_planner_response_reduces_to_validated_rocm_tool_call() -> Result<()> {
        let content = r#"{
            "intent": "serve",
            "confidence": "high",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "sshleifer/tiny-gpt2", "--engine", "pytorch", "--device", "gpu_required", "--managed"]
            },
            "notes": ["resolved the missing model to a tiny test model"]
        }"#;

        let plan = provider_planner_response_to_plan("start a local model", "local", content)?;

        assert!(plan.provider_assisted);
        assert!(plan.planner.contains("provider:local"));
        assert_eq!(plan.intent, PlannerIntent::Serve);
        assert_eq!(plan.confidence, "high");
        assert_eq!(plan.actions[0].approval, "required");
        assert_eq!(
            plan.actions[0].args,
            vec![
                "serve".to_owned(),
                "sshleifer/tiny-gpt2".to_owned(),
                "--engine".to_owned(),
                "pytorch".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
        assert!(
            plan.notes
                .iter()
                .any(|note| note.contains("validated rocm tool call"))
        );
        Ok(())
    }

    #[test]
    fn provider_planner_rejects_public_bind_requests() {
        for content in [
            r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "tiny.gguf", "--engine", "llama.cpp", "--allow-public-bind", "--managed"]
            }
        }"#,
            r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "tiny.gguf", "--engine", "llama.cpp", "--host", "0.0.0.0", "--managed"]
            }
        }"#,
        ] {
            let error = provider_planner_response_to_plan("serve publicly", "local", content)
                .unwrap_err()
                .to_string();

            assert!(
                error.contains("public network binding") || error.contains("non-local host"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn provider_planner_requires_managed_serve_requests() {
        for args in [
            vec!["serve", "qwen", "--engine", "pytorch"],
            vec!["serve", "qwen", "--engine", "pytorch", "--foreground"],
        ] {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: args.into_iter().map(str::to_owned).collect(),
            };
            let error = validate_provider_planner_tool_call(&call)
                .unwrap_err()
                .to_string();

            assert!(error.contains("--managed"), "unexpected error: {error}");
        }
    }

    #[test]
    fn provider_planner_requires_user_folder_for_therock_install() {
        let call = ProviderPlannerToolCall {
            tool: "rocm".to_owned(),
            args: vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
            ],
        };
        let error = validate_provider_planner_tool_call(&call)
            .unwrap_err()
            .to_string();

        assert!(error.contains("ask the user"), "unexpected error: {error}");
    }

    #[test]
    fn provider_planner_response_rejects_cpu_serve_device() {
        for args in [
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "pytorch",
                "--device",
                "cpu",
                "--managed",
            ],
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "pytorch",
                "--device=cpu",
                "--managed",
            ],
            vec![
                "serve",
                "sshleifer/tiny-gpt2",
                "--engine",
                "pytorch",
                "--device",
                "cpu_only",
                "--managed",
            ],
        ] {
            let call = ProviderPlannerToolCall {
                tool: "rocm".to_owned(),
                args: args.into_iter().map(str::to_owned).collect(),
            };
            let error = validate_provider_planner_tool_call(&call)
                .unwrap_err()
                .to_string();

            assert!(error.contains("CPU execution"));
            assert!(error.contains("ROCm GPU execution"));
        }
    }

    #[test]
    fn chat_tool_call_mutating_install_maps_to_reviewable_rocm_command() {
        let call = providers::ChatToolCall {
            id: Some("call-1".to_owned()),
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": "D:\\ROCm\\therock_venvs"
            }),
        };

        assert!(!chat_tool_call_is_read_only(&call));
        validate_chat_tool_call(&call).expect("install request should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some(
                "rocm install sdk --channel release --format wheel --prefix D:\\ROCm\\therock_venvs"
            )
        );
        let approval = chat_tool_approval_request(
            &call,
            Some("TheRock is not installed yet, so I need to install ROCm first."),
        )
        .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ROCm");
        assert_eq!(approval.command_title, "Install");
        assert_eq!(
            approval.explanation.as_deref(),
            Some("TheRock is not installed yet, so I need to install ROCm first.")
        );
        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
            ]
        );
    }

    #[test]
    fn chat_tool_call_mutating_install_accepts_requested_build_date() {
        let call = providers::ChatToolCall {
            id: Some("call-date".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["install", "sdk", "--channel", "release", "--format", "wheel", "--prefix", "D:\\ROCm\\therock_venvs", "--build-date", "06052026"],
                "reason": "The user asked for the TheRock build from 2026-06-05."
            }),
        };

        validate_chat_tool_call(&call).expect("date-specific install should validate for review");
        assert!(!chat_tool_call_is_read_only(&call));
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some(
                "rocm install sdk --channel release --format wheel --prefix D:\\ROCm\\therock_venvs --build-date 06052026"
            )
        );
        let approval =
            chat_tool_approval_request(&call, Some("Install the requested TheRock build."))
                .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ROCm");
        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
                "--build-date".to_owned(),
                "06052026".to_owned(),
            ]
        );
    }

    #[test]
    fn chat_tool_call_rejects_mutating_install_without_user_folder() {
        let structured = providers::ChatToolCall {
            id: Some("call-missing-prefix".to_owned()),
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({
                "channel": "release",
                "format": "wheel"
            }),
        };
        let error = validate_chat_tool_call(&structured)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ask the user"), "unexpected error: {error}");

        let command = providers::ChatToolCall {
            id: Some("call-command-missing-prefix".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["install", "sdk", "--channel", "release", "--format", "wheel"],
                "reason": "Install ROCm."
            }),
        };
        let error = validate_chat_tool_call(&command).unwrap_err().to_string();
        assert!(error.contains("ask the user"), "unexpected error: {error}");
    }

    #[test]
    fn chat_tool_call_service_and_watcher_changes_map_to_reviewable_rocm_commands() {
        let stop = providers::ChatToolCall {
            id: Some("call-stop".to_owned()),
            name: "stop_server".to_owned(),
            arguments: serde_json::json!({ "service_id": "svc-qwen" }),
        };
        validate_chat_tool_call(&stop).expect("stop request should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&stop).as_deref(),
            Some("rocm services stop svc-qwen --yes")
        );
        let approval =
            chat_tool_approval_request(&stop, Some("This server is using memory we need."))
                .expect("stop approval should be built");
        assert_eq!(approval.pending_title, "Stop local model server");
        assert_eq!(approval.command_title, "Services");
        assert_eq!(
            approval.args,
            vec![
                "services".to_owned(),
                "stop".to_owned(),
                "svc-qwen".to_owned(),
                "--yes".to_owned(),
            ]
        );
        assert_eq!(
            approval.explanation.as_deref(),
            Some("This server is using memory we need.")
        );

        let enable = providers::ChatToolCall {
            id: Some("call-watch".to_owned()),
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({
                "watcher": "server-recover",
                "mode": "propose"
            }),
        };
        validate_chat_tool_call(&enable).expect("watcher enable should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&enable).as_deref(),
            Some("rocm automations enable server-recover --mode propose")
        );
        let approval =
            chat_tool_approval_request(&enable, Some("Recovering failed servers would help."))
                .expect("watcher approval should be built");
        assert_eq!(approval.pending_title, "Enable automation");
        assert_eq!(approval.command_title, "Automations");
        assert_eq!(
            approval.args,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "propose".to_owned(),
            ]
        );

        let disable = providers::ChatToolCall {
            id: Some("call-disable".to_owned()),
            name: "watcher_disable".to_owned(),
            arguments: serde_json::json!({ "watcher": "server-recover" }),
        };
        validate_chat_tool_call(&disable).expect("watcher disable should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&disable).as_deref(),
            Some("rocm automations disable server-recover")
        );
    }

    #[test]
    fn proposal_action_rejects_over_long_proposal_id() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({
                "proposal_id": "p".repeat(129),
                "action": "show"
            }),
        };
        let err = validate_chat_proposal_action_tool_call(&call)
            .expect_err("over-long proposal_id must be rejected");
        assert!(err.to_string().contains("proposal_id too long"));
    }

    #[test]
    fn chat_tool_call_accepts_expanded_read_only_bridge_tools() {
        for call in [
            providers::ChatToolCall {
                id: None,
                name: "bridge_snapshot".to_owned(),
                arguments: serde_json::json!({}),
            },
            providers::ChatToolCall {
                id: None,
                name: "service_logs".to_owned(),
                arguments: serde_json::json!({
                    "service_id": "svc-qwen",
                    "lines": 120
                }),
            },
            providers::ChatToolCall {
                id: None,
                name: "automations".to_owned(),
                arguments: serde_json::json!({ "event_limit": 12 }),
            },
            providers::ChatToolCall {
                id: None,
                name: "natural_language_plan".to_owned(),
                arguments: serde_json::json!({ "request": "check whether ROCm needs an update" }),
            },
            providers::ChatToolCall {
                id: None,
                name: "port_status".to_owned(),
                arguments: serde_json::json!({ "host": "127.0.0.1", "port": 8188 }),
            },
            providers::ChatToolCall {
                id: None,
                name: "update_check".to_owned(),
                arguments: serde_json::json!({}),
            },
        ] {
            validate_chat_tool_call(&call).expect("read-only bridge tool should validate");
            assert!(
                chat_tool_call_is_read_only(&call),
                "{} should be read-only",
                call.name
            );
        }
    }

    #[test]
    fn local_assistant_prompt_instructions_cover_core_support_questions() {
        let prompt = rocm_chat_tool_system_prompt();
        for expected in [
            "is TheRock installed",
            "which GPU is on this machine",
            "active_runtime_status=ready",
            "legacy_rocm_status=not_detected",
            "[\"model\"]",
            "--build-date",
            "always let the user choose the install folder",
            "--prefix",
            "do not invent a hidden default folder",
            "config",
            "comfyui",
            "First-time setup is the same thing as bootstrap",
            "vllm",
            "Qwen3-4B-Instruct-2507-GGUF",
            "fixed to qwen",
            "served by Lemonade",
            "port_status",
            "[\"services\",\"list\",\"--all\"]",
            "qwen-smoke",
            "llama-server",
            "Do not invent shell commands",
            "ROCm CLI Assistant Skill",
            "Treat `localhost` and `127.0.0.1` as the same loopback endpoint",
        ] {
            assert!(
                prompt.contains(expected),
                "system prompt should mention {expected}"
            );
        }
    }

    #[test]
    fn deterministic_rocm_tool_summary_interprets_managed_runtime_as_installed() {
        let summary = deterministic_rocm_tool_summary(
            "\
examine:
  driver_detail: AMD Radeon RX 9070 XT driver 32.0.23033.1002
  legacy_rocm_status: not_detected
runtime_state:
  active_runtime_status: ready
  active_runtime_root: D:\\ROCm\\therock_venvs
  active_runtime_pip_cache_dir: D:\\ROCm\\therock_venvs\\pip-cache
  active_runtime_version: 7.13.0a20260511 (build 2026-05-11)
  active_runtime_family: gfx120X-all
",
        )
        .expect("examine output should summarize");

        assert!(summary.contains("GPU: AMD Radeon RX 9070 XT driver 32.0.23033.1002"));
        assert!(summary.contains("ROCm/TheRock: installed and active for ROCm CLI"));
        assert!(summary.contains("gfx120X-all"));
        assert!(summary.contains(r"Install folder: D:\ROCm\therock_venvs"));
        assert!(summary.contains(r"Downloads/cache: D:\ROCm\therock_venvs\pip-cache"));
        assert!(summary.contains("no global legacy ROCm install was found"));
    }

    #[test]
    fn fallback_tool_call_routes_where_installed_to_read_only_examine() {
        for prompt in [
            "where is rocm installed?",
            "where is TheRock installed?",
            "what is the ROCm install folder?",
            "where did rocm install to?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine", "{prompt}");
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn deterministic_rocm_tool_summary_suppresses_extra_local_model_follow_up() {
        let tool_result = ChatToolRunResult {
            approval: None,
            follow_up_text: "\
examine:
  legacy_rocm_status: not_detected
runtime_state:
  active_runtime_status: ready
"
            .to_owned(),
            ran_read_only_tool: true,
            read_only_tool_error: false,
            needs_install_folder: false,
        };
        let summary = deterministic_rocm_tool_summary(&tool_result.follow_up_text);

        assert!(summary.is_some());
        assert!(!should_request_local_tool_follow_up(
            "local",
            &tool_result,
            summary.as_deref()
        ));

        let mut model_list_result = tool_result;
        model_list_result.follow_up_text = "rocm_command:\nmodel recipes\n  qwen\n".to_owned();
        assert!(should_request_local_tool_follow_up(
            "local",
            &model_list_result,
            None
        ));
    }

    #[test]
    fn deterministic_model_tool_summary_identifies_low_vram_assistant() {
        let summary = deterministic_model_tool_summary(
            "\
rocm_command:
model recipes
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen, lemonade-qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: recommended Lemonade GGUF assistant for ROCm machines
  Qwen3-0.6B-GGUF aliases=[qwen-smoke, lemonade-tiny] task=chat dtype=gguf device=gpu_required min_gpu_mem=2 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: tiny Lemonade GGUF smoke-test model; not the default assistant
  Qwen/Qwen2.5-0.5B-Instruct aliases=[qwen-tiny] task=chat dtype=float16 device=gpu_required min_gpu_mem=4 GiB engines=[pytorch]
      engine_support:
        pytorch: available path=D:\\rocm\\rocm-engine-pytorch.exe
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
      engine_support:
        vllm: adapter_available path=D:\\rocm\\rocm-engine-vllm.exe runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm
  meta-llama/Llama-3.2-3B-Instruct aliases=[llama] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=8 GiB engines=[pytorch, llama.cpp]
      engine_support:
        pytorch: available path=D:\\rocm\\rocm-engine-pytorch.exe
        llama.cpp: available path=D:\\rocm\\rocm-engine-llama-cpp.exe
",
        )
        .expect("model output should summarize");

        assert!(summary.contains("Recommended local assistant: qwen"));
        assert!(summary.contains("Qwen3-4B-Instruct-2507-GGUF"));
        assert!(summary.contains("4 GiB"));
        assert!(summary.contains("Tiny smoke test: qwen-smoke"));
        assert!(summary.contains("Qwen3-0.6B-GGUF"));
        assert!(summary.contains("8 GiB-class option: llama"));
        assert!(summary.contains("pytorch, llama.cpp"));
        assert!(summary.contains("Qwen/Qwen3.5-4B asks for 12 GiB"));
        assert!(summary.contains("Native Windows note"));
        assert!(summary.contains("Run `rocm examine`"));
    }

    #[test]
    fn deterministic_model_tool_summary_suppresses_extra_local_model_follow_up() {
        let tool_result = ChatToolRunResult {
            approval: None,
            follow_up_text: "\
rocm_command:
model recipes
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
"
            .to_owned(),
            ran_read_only_tool: true,
            read_only_tool_error: false,
            needs_install_folder: false,
        };
        let summary = deterministic_chat_tool_summary(&tool_result.follow_up_text);

        assert!(summary.is_some());
        assert!(!should_request_local_tool_follow_up(
            "local",
            &tool_result,
            summary.as_deref()
        ));
    }

    #[test]
    fn chat_tool_call_accepts_assistant_support_command_shapes() {
        for (call, expected_command, read_only) in [
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["examine"] }),
                },
                Some("rocm examine"),
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "gpu_snapshot".to_owned(),
                    arguments: serde_json::json!({}),
                },
                None,
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["model"] }),
                },
                Some("rocm model"),
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "install_sdk".to_owned(),
                    arguments: serde_json::json!({
                        "channel": "release",
                        "format": "wheel",
                        "prefix": "D:\\ROCm\\therock_venvs"
                    }),
                },
                Some(
                    "rocm install sdk --channel release --format wheel --prefix D:\\ROCm\\therock_venvs",
                ),
                false,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["comfyui", "install"] }),
                },
                Some("rocm comfyui install"),
                false,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "launch_server".to_owned(),
                    arguments: serde_json::json!({
                        "model": "qwen",
                        "engine": "pytorch",
                        "device": "gpu_required"
                    }),
                },
                Some("rocm serve qwen --managed --engine pytorch --device gpu_required"),
                false,
            ),
        ] {
            validate_chat_tool_call(&call).expect("assistant support tool should validate");
            assert_eq!(
                chat_tool_call_is_read_only(&call),
                read_only,
                "{}",
                call.name
            );
            if let Some(expected_command) = expected_command {
                assert_eq!(
                    rocm_chat_tool_requested_command(&call).as_deref(),
                    Some(expected_command)
                );
            }
        }
    }

    #[test]
    fn chat_rocm_command_routes_comfyui_and_llama_cpp_actions() {
        let comfy_install = providers::ChatToolCall {
            id: Some("call-comfy".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "install"],
                "reason": "The user asked me to install ComfyUI."
            }),
        };
        validate_chat_tool_call(&comfy_install).expect("ComfyUI install should validate");
        assert!(!chat_tool_call_is_read_only(&comfy_install));
        assert_eq!(
            rocm_chat_tool_requested_command(&comfy_install).as_deref(),
            Some("rocm comfyui install")
        );
        let approval = chat_tool_approval_request(&comfy_install, Some("Install ComfyUI now."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ComfyUI");
        assert_eq!(approval.command_title, "ComfyUI");
        assert_eq!(
            approval.args,
            vec!["comfyui".to_owned(), "install".to_owned()]
        );

        let llama = providers::ChatToolCall {
            id: Some("call-llama".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["engines", "install", "llama.cpp"]
            }),
        };
        validate_chat_tool_call(&llama).expect("llama.cpp engine install should validate");
        assert!(!chat_tool_call_is_read_only(&llama));
        assert_eq!(
            rocm_chat_tool_requested_command(&llama).as_deref(),
            Some("rocm engines install llama.cpp")
        );
        let approval =
            chat_tool_approval_request(&llama, Some("Install llama.cpp for GGUF serving."))
                .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install engine");
        assert_eq!(approval.command_title, "Engine");

        let vllm = providers::ChatToolCall {
            id: Some("call-vllm".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["engines", "install", "vllm"]
            }),
        };
        validate_chat_tool_call(&vllm).expect("vLLM engine install should validate");
        assert!(!chat_tool_call_is_read_only(&vllm));
        assert_eq!(
            rocm_chat_tool_requested_command(&vllm).as_deref(),
            Some("rocm engines install vllm")
        );
        let approval = chat_tool_approval_request(&vllm, Some("Install vLLM for Linux/WSL."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install engine");
        assert_eq!(approval.command_title, "Engine");

        let comfy_start = providers::ChatToolCall {
            id: Some("call-comfy-start".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "start"]
            }),
        };
        validate_chat_tool_call(&comfy_start).expect("ComfyUI start should validate");
        assert!(!chat_tool_call_is_read_only(&comfy_start));
        let approval = chat_tool_approval_request(&comfy_start, Some("Start ComfyUI locally."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Start ComfyUI");
        assert_eq!(approval.command_title, "ComfyUI");

        let serve = providers::ChatToolCall {
            id: Some("call-serve".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "pytorch", "--device", "gpu_required", "--managed"]
            }),
        };
        validate_chat_tool_call(&serve).expect("managed serve should validate");
        assert!(!chat_tool_call_is_read_only(&serve));
        assert_eq!(
            rocm_chat_tool_requested_command(&serve).as_deref(),
            Some("rocm serve qwen --engine pytorch --device gpu_required --managed")
        );
        let approval = chat_tool_approval_request(&serve, Some("Start the recommended assistant."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Start local model server");
        assert_eq!(approval.command_title, "Serve");

        let vllm_serve = providers::ChatToolCall {
            id: Some("call-vllm-serve".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "Qwen/Qwen3.5-4B", "--engine", "vllm", "--device", "gpu_required", "--managed"]
            }),
        };
        validate_chat_tool_call(&vllm_serve).expect("managed vLLM serve should validate");
        assert!(!chat_tool_call_is_read_only(&vllm_serve));
        assert_eq!(
            rocm_chat_tool_requested_command(&vllm_serve).as_deref(),
            Some("rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed")
        );

        let config = providers::ChatToolCall {
            id: Some("call-config".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["config", "set-default-engine", "pytorch"]
            }),
        };
        validate_chat_tool_call(&config).expect("config change should validate");
        assert!(!chat_tool_call_is_read_only(&config));
        let approval =
            chat_tool_approval_request(&config, Some("Use PyTorch as the default engine."))
                .expect("approval should be built");
        assert_eq!(approval.pending_title, "Change settings");
        assert_eq!(approval.command_title, "Config");
    }

    #[test]
    fn setup_status_is_read_only() {
        for args in [
            vec!["setup".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
        ] {
            let action =
                chat_rocm_command_action_from_args(args.clone()).expect("setup status classifies");
            assert!(
                matches!(action, ChatRocmCommandAction::ReadOnly(_)),
                "setup {args:?} should be read-only, got {action:?}"
            );
        }
    }

    #[test]
    fn setup_reset_requires_approval() {
        let action =
            chat_rocm_command_action_from_args(vec!["setup".to_owned(), "reset".to_owned()])
                .expect("setup reset classifies");
        match action {
            ChatRocmCommandAction::Approval {
                pending_title,
                command_title,
                ..
            } => {
                assert_eq!(pending_title, "Reset first-time setup");
                assert_eq!(command_title, "Setup");
            }
            other @ ChatRocmCommandAction::ReadOnly(_) => {
                panic!("setup reset should require approval, got {other:?}")
            }
        }
    }

    #[test]
    fn proposal_action_show_is_read_only() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({ "proposal_id": "p1", "action": "show" }),
        };
        validate_chat_tool_call(&call).expect("show validates");
        assert!(
            chat_tool_call_is_read_only(&call),
            "proposal_action show must be read-only"
        );
    }

    #[test]
    fn proposal_action_approve_requires_approval() {
        for action in ["approve", "reject"] {
            let call = providers::ChatToolCall {
                id: None,
                name: "proposal_action".to_owned(),
                arguments: serde_json::json!({ "proposal_id": "p1", "action": action }),
            };
            validate_chat_tool_call(&call).expect("approve/reject validates");
            assert!(
                !chat_tool_call_is_read_only(&call),
                "proposal_action {action} must NOT be read-only"
            );
            let req = chat_tool_approval_request(&call, None).unwrap_or_else(|err| {
                panic!("proposal_action {action} should need approval: {err}")
            });
            assert_eq!(req.command_title, "Reviews");
            assert!(
                req.pending_title.contains("proposal") || req.pending_title.contains("Proposal")
            );
            assert!(
                req.display_command
                    .as_deref()
                    .unwrap_or_default()
                    .contains("p1"),
                "display command should show the proposal id"
            );
        }
    }

    #[test]
    fn proposal_action_rejects_unknown_action() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({ "proposal_id": "p1", "action": "delete" }),
        };
        assert!(
            validate_chat_tool_call(&call).is_err(),
            "unknown proposal_action `action` must be rejected"
        );
    }

    #[test]
    fn proposal_action_approve_updates_status() {
        let (root, paths) = test_paths("proposal-approve");
        // Seed a pending proposal.
        let proposal = rocm_core::AutomationProposalRecord {
            at_unix_ms: rocm_core::unix_time_millis(),
            proposal_id: "prop-approve-1".to_owned(),
            watcher_id: "therock-update".to_owned(),
            action: "prepare_driver_plan".to_owned(),
            title: "Apply driver plan".to_owned(),
            message: "A reviewed driver plan is ready.".to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: None,
            arguments: serde_json::Value::Null,
            reviewed_at_unix_ms: None,
        };
        rocm_core::append_automation_proposal(&paths, &proposal).expect("seed proposal");

        // show is read-only and returns the proposal.
        let shown = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-approve-1", "action": "show" }),
            false,
        )
        .expect("show ok");
        assert_eq!(shown["structuredContent"]["status"], "pending");

        // approve requires allow_mutation.
        assert!(
            run_internal_mcp_call(
                &paths,
                "proposal_action",
                serde_json::json!({ "proposal_id": "prop-approve-1", "action": "approve" }),
                false,
            )
            .is_err(),
            "approve without allow_mutation must bail"
        );

        // approve with allow_mutation sets status to approved.
        let approved = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-approve-1", "action": "approve" }),
            true,
        )
        .expect("approve ok");
        assert_eq!(approved["structuredContent"]["status"], "approved");
        let stored = rocm_core::find_automation_proposal(&paths, "prop-approve-1")
            .expect("proposal still present");
        assert_eq!(stored.status, "approved");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_action_reject_updates_status() {
        let (root, paths) = test_paths("proposal-reject");
        let proposal = rocm_core::AutomationProposalRecord {
            at_unix_ms: rocm_core::unix_time_millis(),
            proposal_id: "prop-reject-1".to_owned(),
            watcher_id: "server-recover".to_owned(),
            action: "queue_stop_server_proposal".to_owned(),
            title: "Stop overheating server".to_owned(),
            message: "GPU thermal pressure detected.".to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: None,
            arguments: serde_json::Value::Null,
            reviewed_at_unix_ms: None,
        };
        rocm_core::append_automation_proposal(&paths, &proposal).expect("seed proposal");

        let rejected = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-reject-1", "action": "reject" }),
            true,
        )
        .expect("reject ok");
        assert_eq!(rejected["structuredContent"]["status"], "rejected");
        let stored = rocm_core::find_automation_proposal(&paths, "prop-reject-1")
            .expect("proposal still present");
        assert_eq!(stored.status, "rejected");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_action_show_missing_proposal_errors() {
        let (root, paths) = test_paths("proposal-missing");
        assert!(
            run_internal_mcp_call(
                &paths,
                "proposal_action",
                serde_json::json!({ "proposal_id": "nope", "action": "show" }),
                false,
            )
            .is_err(),
            "showing a missing proposal must error"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn config_set_permissions_classifies_as_approval() {
        // Permission escalation MUST route through approval — the classifier
        // routes any `config <sub>` (catch-all) to Approval. Verify both modes.
        for mode in ["full_access", "ask"] {
            let action = chat_rocm_command_action_from_args(vec![
                "config".to_owned(),
                "set-permissions".to_owned(),
                mode.to_owned(),
            ])
            .expect("config set-permissions classifies");
            match action {
                ChatRocmCommandAction::Approval { command_title, .. } => {
                    assert_eq!(command_title, "Config");
                }
                other @ ChatRocmCommandAction::ReadOnly(_) => {
                    panic!("config set-permissions {mode} must need approval, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn config_set_permissions_sets_mode() {
        // The SetPermissions handler logic: set permissions.mode, save, reload.
        let (root, paths) = test_paths("config-permissions");
        let mut config = RocmCliConfig::load(&paths).expect("load default config");
        assert_eq!(config.permissions.mode_label(), PERMISSIONS_MODE_ASK);
        // Mirror the SetPermissions handler mutation.
        config.permissions.mode = PermissionsModeArg::FullAccess.as_str().to_owned();
        config.save(&paths).expect("save config");
        let reloaded = RocmCliConfig::load(&paths).expect("reload config");
        assert_eq!(
            reloaded.permissions.mode_label(),
            PERMISSIONS_MODE_FULL_ACCESS
        );
        assert!(reloaded.permissions.full_access_enabled());
        // And back to ask.
        let mut config = reloaded;
        config.permissions.mode = PermissionsModeArg::Ask.as_str().to_owned();
        config.save(&paths).expect("save config");
        let reloaded = RocmCliConfig::load(&paths).expect("reload config");
        assert_eq!(reloaded.permissions.mode_label(), PERMISSIONS_MODE_ASK);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn watcher_validator_rejects_unknown_and_invalid_mode() {
        // Unknown watcher id → rejected.
        let unknown = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "no-such-watcher" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&unknown, true).is_err(),
            "unknown watcher must be rejected"
        );
        // Invalid mode → rejected.
        let bad_mode = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "rampage" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&bad_mode, true).is_err(),
            "invalid watcher mode must be rejected"
        );
        // Valid watcher + valid mode → accepted.
        let ok = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "observe" }),
        };
        validate_chat_watcher_tool_call(&ok, true).expect("valid watcher+mode accepted");
        // Disable must reject a `mode`.
        let disable_with_mode = providers::ChatToolCall {
            id: None,
            name: "watcher_disable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "observe" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&disable_with_mode, false).is_err(),
            "disable must reject a mode argument"
        );
    }

    #[test]
    fn lifecycle_read_mutate_split_is_honest() {
        let read_only = [
            vec!["update".to_owned()],
            vec!["comfyui".to_owned(), "status".to_owned()],
            vec!["comfyui".to_owned(), "logs".to_owned()],
            vec!["uninstall".to_owned(), "--dry-run".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
        ];
        for args in read_only {
            let action = chat_rocm_command_action_from_args(args.clone())
                .unwrap_or_else(|err| panic!("{args:?} should classify: {err}"));
            assert!(
                matches!(action, ChatRocmCommandAction::ReadOnly(_)),
                "{args:?} should be read-only, got {action:?}"
            );
        }

        let mutating = [
            vec!["update".to_owned(), "--apply".to_owned()],
            vec!["comfyui".to_owned(), "install".to_owned()],
            vec!["comfyui".to_owned(), "start".to_owned()],
            vec!["comfyui".to_owned(), "stop".to_owned()],
            vec!["uninstall".to_owned()],
            vec!["setup".to_owned(), "reset".to_owned()],
        ];
        for args in mutating {
            let action = chat_rocm_command_action_from_args(args.clone())
                .unwrap_or_else(|err| panic!("{args:?} should classify: {err}"));
            assert!(
                matches!(action, ChatRocmCommandAction::Approval { .. }),
                "{args:?} should require approval, got {action:?}"
            );
        }
    }

    #[test]
    fn chat_rocm_command_runs_read_only_and_rejects_risky_shapes() {
        let status = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["rocm", "comfy", "status"]
            }),
        };
        validate_chat_tool_call(&status).expect("ComfyUI status should validate");
        assert!(chat_tool_call_is_read_only(&status));
        assert_eq!(
            rocm_chat_tool_requested_command(&status).as_deref(),
            Some("rocm comfyui status")
        );

        let logs = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "logs"]
            }),
        };
        validate_chat_tool_call(&logs).expect("ComfyUI logs should validate");
        assert!(chat_tool_call_is_read_only(&logs));
        assert_eq!(
            rocm_chat_tool_requested_command(&logs).as_deref(),
            Some("rocm comfyui logs")
        );

        let cpu = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "tiny.gguf", "--engine", "llama.cpp", "--device", "cpu"]
            }),
        };
        let error = validate_chat_tool_call(&cpu).unwrap_err().to_string();
        assert!(error.contains("CPU execution"));

        let public_flag = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "tiny.gguf", "--engine", "llama.cpp", "--allow-public-bind", "--managed"]
            }),
        };
        let error = validate_chat_tool_call(&public_flag)
            .unwrap_err()
            .to_string();
        assert!(error.contains("public network binding"));

        let foreground = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "pytorch", "--foreground"]
            }),
        };
        let error = validate_chat_tool_call(&foreground)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--managed"));

        let shell = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["powershell", "-Command", "whoami"]
            }),
        };
        let error = validate_chat_tool_call(&shell).unwrap_err().to_string();
        assert!(error.contains("unsupported rocm command"));
    }

    #[test]
    fn assistant_read_only_rocm_commands_do_not_fallback_to_child_process() {
        let (_root, paths) = test_paths("readonly-rocm-in-process-only");
        let args = vec!["services".to_owned(), "status".to_owned()];

        let error = run_rocm_command_for_paths(&paths, &args, Duration::from_secs(1))
            .unwrap_err()
            .to_string();

        assert!(error.contains("read-only assistant command is not implemented in-process"));
        assert!(error.contains("rocm services status"));
        assert!(!paths.data_dir.join("logs").exists());
    }

    #[test]
    fn internal_mcp_read_only_rocm_command_runs_in_process() {
        let (_root, paths) = test_paths("mcp-readonly-rocm-in-process");

        let result = run_internal_mcp_call(
            &paths,
            "rocm_command",
            serde_json::json!({ "args": ["version"] }),
            false,
        )
        .expect("read-only rocm mcp-call should run");

        assert_eq!(
            result.get("isError").and_then(serde_json::Value::as_bool),
            Some(false)
        );
        assert_eq!(
            result
                .pointer("/structuredContent/argv/0")
                .and_then(serde_json::Value::as_str),
            Some("rocm")
        );
        assert!(mcp_tool_result_text(&result).contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn natural_language_plan_returns_structured_mutating_action() {
        let (_root, paths) = test_paths("nl-plan-structured-mutating");

        let result = run_internal_mcp_call(
            &paths,
            "natural_language_plan",
            serde_json::json!({
                "request": "install TheRock into /opt/rocm-target"
            }),
            false,
        )
        .expect("natural_language_plan should plan the request");

        assert_eq!(
            result.get("isError").and_then(serde_json::Value::as_bool),
            Some(false)
        );
        // text (rendered plan) preserved.
        assert!(
            !result
                .pointer("/structuredContent/text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .trim()
                .is_empty()
        );
        // A known install folder yields a mutating, placeholder-free action.
        assert_eq!(
            result
                .pointer("/structuredContent/action/approval_required")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            result
                .pointer("/structuredContent/action/has_placeholders")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
        let args = result
            .pointer("/structuredContent/action/args")
            .and_then(serde_json::Value::as_array)
            .expect("action args should be present");
        assert!(args.iter().any(|arg| arg.as_str() == Some("install")));
        assert!(args.iter().any(|arg| arg.as_str() == Some("--prefix")));
    }

    #[test]
    fn natural_language_plan_returns_placeholder_action_when_incomplete() {
        let (_root, paths) = test_paths("nl-plan-structured-placeholder");

        let result = run_internal_mcp_call(
            &paths,
            "natural_language_plan",
            serde_json::json!({ "request": "serve" }),
            false,
        )
        .expect("natural_language_plan should plan the request");

        assert_eq!(
            result
                .pointer("/structuredContent/action/has_placeholders")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn setup_status_renders_in_process() {
        let (_root, paths) = test_paths("setup-status-in-process");

        for args in [
            vec!["setup".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
        ] {
            let text = run_rocm_read_only_in_process(&paths, &args)
                .expect("setup status read path should render in-process");
            assert!(
                !text.trim().is_empty(),
                "setup status text should be non-empty for {args:?}"
            );
            assert!(
                text.contains("ROCm setup"),
                "setup status text should be reached for {args:?}: {text}"
            );
        }
    }

    #[test]
    fn chat_tool_call_rejects_bad_service_and_watcher_suggestions() {
        for (call, expected) in [
            (
                providers::ChatToolCall {
                    id: None,
                    name: "service_logs".to_owned(),
                    arguments: serde_json::json!({ "service_id": "bad/name" }),
                },
                "must not contain path separators",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "automations".to_owned(),
                    arguments: serde_json::json!({ "event_limit": 1000 }),
                },
                "between 1 and 64",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "watcher_enable".to_owned(),
                    arguments: serde_json::json!({
                        "watcher": "unknown",
                        "mode": "propose"
                    }),
                },
                "unknown watcher",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "watcher_disable".to_owned(),
                    arguments: serde_json::json!({
                        "watcher": "server-recover",
                        "mode": "propose"
                    }),
                },
                "cannot set `mode`",
            ),
        ] {
            let error = validate_chat_tool_call(&call).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn chat_tool_call_rejects_cpu_and_public_bind_server_requests() {
        let cpu = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "llama.cpp",
                "device": "cpu"
            }),
        };
        let error = validate_chat_tool_call(&cpu).unwrap_err().to_string();
        assert!(error.contains("CPU execution"));

        let public = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "llama.cpp",
                "host": "0.0.0.0",
                "allow_public_bind": true
            }),
        };
        let error = validate_chat_tool_call(&public).unwrap_err().to_string();
        assert!(error.contains("public network binding"));

        let host = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "llama.cpp",
                "host": "0.0.0.0"
            }),
        };
        let error = validate_chat_tool_call(&host).unwrap_err().to_string();
        assert!(error.contains("non-local host"));
    }

    #[test]
    fn chat_tool_call_rejects_bad_install_suggestions() {
        for (arguments, expected) in [
            (
                serde_json::json!({ "channel": "stable" }),
                "unsupported TheRock channel",
            ),
            (
                serde_json::json!({ "format": "zip" }),
                "unsupported TheRock install format",
            ),
            (
                serde_json::json!({
                    "prefix": if cfg!(windows) { "C:\\Windows\\rocm" } else { "/opt/rocm" }
                }),
                "system install folder",
            ),
            (
                serde_json::json!({
                    "version": "7.13.0a20260605",
                    "build_date": "2026-06-05"
                }),
                "both `version` and `build_date`",
            ),
            (
                serde_json::json!({ "build_date": "not-a-date" }),
                "build date",
            ),
            (
                serde_json::json!({
                    "format": "tarball",
                    "build_date": "2026-06-05"
                }),
                if cfg!(windows) {
                    "tarball` installs on Windows"
                } else {
                    "specific TheRock wheel versions"
                },
            ),
        ] {
            let call = providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments,
            };
            let error = validate_chat_tool_call(&call).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    /// Guard: every read-only tool the dash side registers must have a home in
    /// the bin's accept-list. A name-only tool (the `doctor` regression caught
    /// in review) would otherwise fail end-to-end as "unsupported ROCm tool"
    /// while passing the dash-side, name-against-itself completeness checks.
    ///
    /// We assert the NAME is accepted — not that empty args validate — so the
    /// check stays hermetic (no real `BinToolExecutor::execute` / live I/O) and
    /// independent of each tool's argument schema.
    #[test]
    fn read_tool_names_are_subset_of_bin_accept_list() {
        for name in rocm_dash_tui::agent::ROCM_READ_TOOL_NAMES {
            let call = providers::ChatToolCall {
                id: None,
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            };
            if let Err(error) = validate_chat_tool_call(&call) {
                let message = error.to_string();
                assert!(
                    !message.contains("unsupported ROCm tool"),
                    "ROCM_READ_TOOL_NAMES advertises `{name}` but the bin rejects \
                     the name as unsupported: {message}"
                );
            }
        }
    }

    #[test]
    fn chat_tool_call_refusal_prioritizes_safe_review_wording() -> Result<()> {
        let mut output = String::new();
        let (_root, paths) = test_paths("chat-tool-refusal");
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "tiny.gguf".to_owned(),
            content: "I can install ROCm.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments: serde_json::json!({
                    "channel": "release",
                    "format": "wheel",
                    "prefix": "D:\\ROCm\\therock_venvs"
                }),
            }],
        };

        let mut progress = None;
        append_chat_tool_results(
            &paths,
            &response,
            &mut output,
            Some("I can install ROCm."),
            &mut progress,
        )?;

        assert!(output.contains("Install ROCm: needs your review"));
        assert!(!output.contains("install_sdk: approval required"));
        assert!(output.contains("not run: review the approval card before anything runs"));
        assert!(output.contains("advanced manual command: rocm install sdk"));
        assert!(!output.contains("run the shown ROCm command"));
        Ok(())
    }

    #[test]
    fn fallback_tool_call_runs_model_support_checks() {
        let call =
            fallback_rocm_tool_call_for_prompt("Which LLMs can this machine support?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["model".to_owned()]
        );

        assert!(fallback_rocm_tool_call_for_prompt("Which LLM are you using?").is_none());
    }

    #[test]
    fn local_rocm_tools_chat_uses_fixed_lemonade_qwen_assistant() {
        assert_eq!(
            local_rocm_tools_assistant_model("local", true),
            Some(providers::BUILTIN_ASSISTANT_MODEL_ID)
        );
        assert_eq!(local_rocm_tools_assistant_model("local", false), None);
        assert_eq!(local_rocm_tools_assistant_model("openai", true), None);
    }

    #[test]
    fn fallback_tool_call_routes_running_questions_to_status_tools() {
        let comfy = fallback_rocm_tool_call_for_prompt("Is ComfyUI running?").unwrap();
        assert_eq!(comfy.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy));

        let comfy_port =
            fallback_rocm_tool_call_for_prompt("Is ComfyUI running on port 8188?").unwrap();
        assert_eq!(comfy_port.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy_port).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy_port));

        for prompt in [
            "Is vLLM running?",
            "is sglang running?",
            "is the model server running?",
            "is qwen running?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["services".to_owned(), "list".to_owned(), "--all".to_owned(),],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }

        let port = fallback_rocm_tool_call_for_prompt("what is running on port 8188?").unwrap();
        assert_eq!(port.name, "port_status");
        assert_eq!(
            port.arguments,
            serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 })
        );
        assert!(chat_tool_call_is_read_only(&port));
    }

    #[test]
    fn fallback_tool_call_routes_engine_install_state_to_engines_list() {
        for prompt in ["is vLLM installed?", "is SGLang available?"] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["engines".to_owned(), "list".to_owned()],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn supplemental_tool_call_adds_missing_specific_status_check() {
        let generic_services = providers::ChatToolCall {
            id: Some("model-picked-services".to_owned()),
            name: "services".to_owned(),
            arguments: serde_json::json!({}),
        };

        let call =
            supplemental_read_only_tool_call_for_prompt("Is ComfyUI running?", &[generic_services])
                .unwrap();

        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn supplemental_tool_call_does_not_duplicate_equivalent_status_check() {
        let comfy_status = fallback_rocm_tool_call_for_prompt("Is ComfyUI running?").unwrap();

        assert!(
            supplemental_read_only_tool_call_for_prompt("Is ComfyUI running?", &[comfy_status])
                .is_none()
        );
    }

    #[test]
    fn supplemental_tool_call_adds_running_state_for_engine_install_question() {
        let engine_inventory =
            fallback_rocm_tool_call_for_prompt("Is vLLM installed and is it running?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&engine_inventory).unwrap(),
            vec!["engines".to_owned(), "list".to_owned()]
        );

        let services = supplemental_read_only_tool_call_for_prompt(
            "Is vLLM installed and is it running?",
            &[engine_inventory],
        )
        .unwrap();

        assert_eq!(services.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&services).unwrap(),
            vec!["services".to_owned(), "list".to_owned(), "--all".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&services));
    }

    #[test]
    fn supplemental_tool_call_treats_loopback_port_checks_as_equivalent() {
        let model_port_check = providers::ChatToolCall {
            id: Some("model-picked-port".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "localhost", "port": 8188 }),
        };

        assert!(
            supplemental_read_only_tool_call_for_prompt(
                "what is running on port 8188?",
                &[model_port_check]
            )
            .is_none()
        );
    }

    #[test]
    fn fallback_tool_call_runs_status_checks() {
        for prompt in [
            "Which GPU is on my machine, and is ROCm installed?",
            "Is TheRock setup on my machine?",
            "Check this ROCm setup.",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine");
            assert_eq!(call.arguments, serde_json::json!({}));
        }
    }

    #[test]
    fn fallback_tool_call_routes_therock_setup_to_install_folder_flow() {
        let call = fallback_rocm_tool_call_for_prompt("How do I setup TheRock?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn fallback_tool_call_preserves_requested_therock_install_prefix() {
        for (prompt, expected_prefix) in [
            (
                "install TheRock for me in D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            ("install ROCm to D:\\ROCm\\temp", "D:\\ROCm\\temp"),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                ]
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_build_date_install() {
        for (prompt, expected_prefix) in [
            (
                "Install this specific TheRock wheel from date 06052026 into D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            (
                "install ROCm at D:\\ROCm\\temp with build date 2026-06-05",
                "D:\\ROCm\\temp",
            ),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                    "--build-date".to_owned(),
                    "2026-06-05".to_owned(),
                ],
                "{prompt}"
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_does_not_install_therock_without_folder() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install this specific TheRock wheel from date 06052026",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn chat_install_intent_without_folder_uses_folder_picker_not_rocm_check() -> Result<()> {
        let (_root, paths) = test_paths("chat-install-intent-folder-picker");
        for prompt in [
            "i need to install rocm",
            "How do I setup TheRock?",
            "install therock",
            "How do I get it to install therock?",
            "rocm please",
            "make my AMD GPU ready",
            "get TheRock installed for local AI",
            "set up my AMD GPU for local AI",
            "install this specific TheRock wheel from date 06052026",
        ] {
            let result = render_chat_prompt_result(&paths, "local", None, prompt, true)?;
            let approval = result
                .approval
                .as_ref()
                .expect("install should need folder");
            assert_eq!(approval.pending_title, "Install ROCm", "{prompt}");
            assert_eq!(
                approval.args[..6],
                [
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                ],
                "{prompt}"
            );
            assert!(
                !approval.args.iter().any(|arg| arg == "--prefix"),
                "{prompt}"
            );
            assert!(
                result.rendered.contains("First choose the folder"),
                "{prompt}: {}",
                result.rendered
            );
            assert!(
                !result.rendered.contains("I checked ROCm"),
                "{prompt}: {}",
                result.rendered
            );
            assert!(
                !result.rendered.contains("ROCm CLI summary"),
                "{prompt}: {}",
                result.rendered
            );
            if prompt.contains("06052026") {
                assert!(approval.args.contains(&"--build-date".to_owned()));
                assert!(approval.args.contains(&"2026-06-05".to_owned()));
            }
        }
        Ok(())
    }

    #[test]
    fn chat_install_intent_ignores_old_conversation_words() -> Result<()> {
        let (_root, paths) = test_paths("chat-install-intent-latest-message");
        let prompt = "\
Conversation so far:
Assistant: Use /examine to refresh actual GPU memory fit before starting anything large.
Assistant: Native Windows note: models may use WSL/Linux through Windows.

New message:
install therock";

        let result = render_chat_prompt_result(&paths, "local", None, prompt, true)?;
        let approval = result
            .approval
            .as_ref()
            .expect("direct latest install request should need a folder");

        assert_eq!(approval.pending_title, "Install ROCm");
        assert!(!approval.args.iter().any(|arg| arg == "--prefix"));
        assert!(result.rendered.contains("I can install ROCm/TheRock"));
        assert!(result.rendered.contains("First choose the folder"));
        assert!(!result.rendered.contains("I checked ROCm"));
        assert!(!result.rendered.contains("ROCm CLI summary"));
        Ok(())
    }

    #[test]
    fn chat_how_to_setup_question_opens_install_folder_flow() {
        assert!(
            install_sdk_without_prefix_chat_approval("How do I setup TheRock?").is_some(),
            "a setup question should ask for the install folder"
        );
        assert!(
            install_sdk_without_prefix_chat_approval("install therock").is_some(),
            "a direct install command should open the folder picker"
        );
    }

    #[test]
    fn chat_install_intent_preserves_bare_folder_path() {
        let approval =
            install_sdk_chat_approval_for_prompt("install therock D:\\ROCm\\therock_venvs")
                .expect("direct install prompt should be recognized");

        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
            ]
        );
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_exact_version_install() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install the TheRock ROCm wheel version 7.13.0a20260605 into D:\\ROCm\\therock_venvs",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
                "--version".to_owned(),
                "7.13.0a20260605".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn path_exists_chat_tool_is_read_only() {
        let call = providers::ChatToolCall {
            id: Some("path-check".to_owned()),
            name: "path_exists".to_owned(),
            arguments: serde_json::json!({ "path": "D:\\ROCm\\temp" }),
        };

        validate_chat_tool_call(&call).unwrap();
        assert!(chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn port_status_chat_tool_is_read_only_and_loopback_only() {
        let call = providers::ChatToolCall {
            id: Some("port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "127.0.0.1", "port": 8188 }),
        };

        validate_chat_tool_call(&call).unwrap();
        assert!(chat_tool_call_is_read_only(&call));

        let public = providers::ChatToolCall {
            id: Some("public-port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "192.168.1.10", "port": 8188 }),
        };
        let error = validate_chat_tool_call(&public).unwrap_err().to_string();
        assert!(error.contains("non-local host"), "{error}");
    }

    #[test]
    fn port_status_matches_loopback_managed_services() -> Result<()> {
        let (root, paths) = test_paths("port-status-loopback");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-comfyui",
            "comfyui",
            "ComfyUI",
            "ComfyUI",
            "127.0.0.1",
            18188,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();
        record.write()?;

        let call = providers::ChatToolCall {
            id: Some("port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "localhost", "port": 18188 }),
        };
        let result = run_chat_port_status_tool(&paths, &call)?;
        let text = mcp_tool_result_text(&result);
        let managed_service_count = result
            .get("structuredContent")
            .and_then(|content| content.get("managed_services"))
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        let _ = fs::remove_dir_all(root);

        assert_eq!(managed_service_count, 1);
        assert!(text.contains("managed_services:"), "{text}");
        assert!(text.contains("service_id=svc-comfyui"), "{text}");
        assert!(text.contains("running_state=starting"), "{text}");
        Ok(())
    }

    #[test]
    fn fallback_tool_call_routes_simple_config_changes() {
        let show = fallback_rocm_tool_call_for_prompt("Show current ROCm CLI config").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&show).unwrap(),
            vec!["config".to_owned(), "show".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&show));

        let engine =
            fallback_rocm_tool_call_for_prompt("Set the default engine to pytorch").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&engine).unwrap(),
            vec![
                "config".to_owned(),
                "set-default-engine".to_owned(),
                "pytorch".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&engine));

        let telemetry =
            fallback_rocm_tool_call_for_prompt("Disable telemetry in settings").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&telemetry).unwrap(),
            vec![
                "config".to_owned(),
                "set-telemetry".to_owned(),
                "off".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&telemetry));
    }

    #[test]
    fn fallback_tool_call_routes_comfyui_support_and_actions() {
        let status = fallback_rocm_tool_call_for_prompt("How do I setup ComfyUI?").unwrap();
        assert_eq!(status.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&status).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&status));

        let install = fallback_rocm_tool_call_for_prompt("Can you setup ComfyUI for me?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&install).unwrap(),
            vec!["comfyui".to_owned(), "install".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&install));
        let approval =
            chat_tool_approval_request(&install, Some("Install ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Install ComfyUI");

        let start = fallback_rocm_tool_call_for_prompt("Can you start ComfyUI?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&start).unwrap(),
            vec!["comfyui".to_owned(), "start".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&start));
        let approval =
            chat_tool_approval_request(&start, Some("Start ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start ComfyUI");
    }

    #[test]
    fn fallback_tool_call_routes_local_llm_serve_requests() {
        let call =
            fallback_rocm_tool_call_for_prompt("Can you setup and serve an LLM for me?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "serve".to_owned(),
                "qwen".to_owned(),
                "--engine".to_owned(),
                "lemonade".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
        let approval =
            chat_tool_approval_request(&call, Some("Start qwen after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start local model server");
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some("rocm serve qwen --engine lemonade --device gpu_required --managed")
        );
    }

    #[test]
    fn local_chat_tool_call_content_is_treated_as_intermediate() {
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "Qwen/Qwen3-0.6B".to_owned(),
            content: "The active runtime root is /opt/rocm.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: Some("call-1".to_owned()),
                name: "examine".to_owned(),
                arguments: serde_json::json!({}),
            }],
        };

        assert!(local_tool_call_content_is_intermediate(
            "local", true, &response
        ));
        assert!(!local_tool_call_content_is_intermediate(
            "openai", true, &response
        ));
        assert!(!local_tool_call_content_is_intermediate(
            "local", false, &response
        ));

        let without_tools = providers::ChatResponse {
            tool_calls: Vec::new(),
            ..response
        };
        assert!(!local_tool_call_content_is_intermediate(
            "local",
            true,
            &without_tools
        ));
    }

    #[test]
    fn local_chat_follow_up_with_tool_call_is_not_final_answer() {
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "Qwen/Qwen3-0.6B".to_owned(),
            content: "The runtime root is /opt/rocml.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: Some("call-2".to_owned()),
                name: "examine".to_owned(),
                arguments: serde_json::json!({}),
            }],
        };

        assert!(!local_follow_up_content_is_final(&response));

        let final_answer = providers::ChatResponse {
            tool_calls: Vec::new(),
            content: "The runtime root is D:\\ROCm\\therock_venvs.".to_owned(),
            ..response
        };
        assert!(local_follow_up_content_is_final(&final_answer));
    }

    #[test]
    fn visible_chat_content_removes_reasoning_blocks() {
        assert_eq!(
            visible_chat_content(
                "<think>\nchecking the tool output\n</think>\nThe runtime root is D:\\ROCm\\temp."
            ),
            "The runtime root is D:\\ROCm\\temp."
        );
        assert_eq!(
            visible_chat_content("Before\n<THINK>hidden</THINK>\nAfter"),
            "Before\n\nAfter"
        );
        assert_eq!(visible_chat_content("<think>unfinished"), "");
    }

    #[test]
    fn chat_tool_result_errors_use_plain_failure_wording() {
        assert_eq!(chat_read_only_tool_status_label(false), "done");
        assert_eq!(chat_read_only_tool_status_label(true), "reported an error");
        assert_eq!(chat_tool_display_label("examine"), "Checked this computer");
        assert_eq!(
            chat_tool_display_label("gpu_snapshot"),
            "Checked GPU status"
        );
        assert_eq!(chat_tool_display_label("install_sdk"), "Install ROCm");
        assert!(mcp_tool_result_is_error(&serde_json::json!({
            "isError": true
        })));
        assert!(!mcp_tool_result_is_error(&serde_json::json!({
            "isError": false
        })));
        assert!(!mcp_tool_result_is_error(&serde_json::json!({})));
    }

    #[test]
    fn local_chat_without_service_explains_serve_before_chat_without_llm_setup() {
        let (_root, paths) = test_paths("local-chat-no-service-guidance");
        let result =
            render_chat_prompt_result(&paths, "local", None, "Check this ROCm setup", true)
                .expect("missing local assistant should render guidance");
        assert!(result.approval.is_none());
        let rendered = result.rendered;

        assert!(rendered.contains("No local assistant is running yet."));
        assert!(rendered.contains("First-time ROCm setup does not need an LLM"));
        assert!(rendered.contains("Recommended path:"));
        assert!(rendered.contains("Advanced manual command"));
        assert!(rendered.contains(
            "rocm serve Qwen3-4B-Instruct-2507-GGUF --engine lemonade --device gpu_required --managed"
        ));
        assert!(!rendered.contains("sshleifer/tiny-gpt2"));
        assert!(rendered.contains("rocm chat --tools --provider local --prompt"));
        assert!(rendered.contains("Nothing was changed."));
        assert!(!rendered.contains("install sdk"));
        assert!(!rendered.contains("setup TheRock with an LLM"));
    }

    #[test]
    fn local_chat_status_prompts_use_read_only_tools_without_assistant() -> Result<()> {
        let (_root, paths) = test_paths("local-chat-status-fallback");

        let running =
            render_chat_prompt_result(&paths, "local", None, "Is vLLM running?", true)?.rendered;
        assert!(!running.contains("No local assistant is running yet."));
        assert!(running.contains("Checked model servers: done"), "{running}");
        assert!(running.contains("ROCm CLI summary"), "{running}");
        assert!(
            running.contains("Local model servers: none running under ROCm CLI."),
            "{running}"
        );
        assert!(running.contains("Nothing was changed."));

        let installed =
            render_chat_prompt_result(&paths, "local", None, "Is vLLM installed?", true)?.rendered;
        assert!(!installed.contains("No local assistant is running yet."));
        assert!(installed.contains("Engine runtimes:"), "{installed}");
        assert!(installed.contains("vLLM:"), "{installed}");

        let installed_and_running = render_chat_prompt_result(
            &paths,
            "local",
            None,
            "Is vLLM installed and is it running?",
            true,
        )?
        .rendered;
        assert!(
            installed_and_running.contains("Checked local engines: done"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Checked model servers: done"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Engine runtimes:"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Local model servers: none running under ROCm CLI."),
            "{installed_and_running}"
        );

        let port = render_chat_prompt_result(
            &paths,
            "local",
            None,
            "What is running on port 8188?",
            true,
        )?
        .rendered;
        assert!(!port.contains("No local assistant is running yet."));
        assert!(port.contains("Checked local port: done"), "{port}");
        assert!(port.contains("Port 8188:"), "{port}");
        Ok(())
    }

    #[test]
    fn chat_tools_anthropic_reaches_provider_opt_in_boundary() {
        let (_root, paths) = test_paths("anthropic-chat-tools-opt-in");

        let error = render_chat_prompt_result(
            &paths,
            "anthropic",
            Some("claude-test"),
            "Check this ROCm setup",
            true,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("cloud provider `anthropic` is disabled"));
        assert!(error.contains("rocm config enable-provider anthropic"));
        assert!(!error.contains("OpenAI-compatible provider"));
    }

    #[test]
    fn freeform_execution_validation_rejects_provider_assisted_plans() -> Result<()> {
        let content = r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "sshleifer/tiny-gpt2", "--engine", "pytorch", "--managed"]
            }
        }"#;
        let plan = provider_planner_response_to_plan("start a local model", "local", content)?;
        let action = plan_next_action(plan).expect("provider plan should have an action");

        let error = validate_freeform_execution_action(&action)
            .unwrap_err()
            .to_string();

        assert!(action.provider_assisted);
        assert!(error.contains("reviewed interactively"));
        Ok(())
    }

    #[test]
    fn render_update_text_reports_all_update_surfaces() -> Result<()> {
        let (root, paths) = test_paths("update-surfaces");

        let rendered = render_update_text(&paths)?;
        fs::remove_dir_all(root).ok();

        assert!(rendered.contains("update_surfaces:"));
        assert!(rendered.contains("cli: installed="));
        assert!(rendered.contains("status=not_configured"));
        assert!(rendered.contains("engines: status=package_managed"));
        assert!(rendered.contains("model_recipes: status="));
        assert!(rendered.contains("runtimes: status=none_configured"));
        assert!(rendered.contains("`rocm update --apply` applies runtime updates only"));
        Ok(())
    }

    #[test]
    fn render_logs_text_preserves_directory_summary() {
        let (_root, paths) = test_paths("logs-summary");
        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("Logs"));
        assert!(rendered.contains("File locations: shown"));
        assert!(rendered.contains(&format!(
            "  Folder: {}",
            paths.data_dir.join("logs").display()
        )));
        assert!(rendered.contains(&format!(
            "  Activity log: {}",
            cli_lifecycle_log_path(&paths).display()
        )));
        assert!(rendered.contains("  Command logs:"));
        assert!(rendered.contains("  Screen command logs:"));
        assert!(rendered.contains(&format!(
            "  Audit events: {}",
            paths.audit_events_path().display()
        )));
        assert!(rendered.contains("  Recent command files: none yet"));
        assert!(rendered.contains("Recent activity: no activity yet"));
        assert!(rendered.contains("Matching lines"));
        assert!(rendered.contains("  Search: none"));
        assert!(rendered.contains("  No logs found yet."));
    }

    #[test]
    fn render_logs_text_lists_action_logs_and_recent_lifecycle_tail() -> Result<()> {
        let (root, paths) = test_paths("logs-navigation");
        fs::create_dir_all(paths.data_dir.join("logs").join("cli"))?;
        fs::write(
            cli_lifecycle_log_path(&paths),
            (0..10).fold(String::new(), |mut acc, index| {
                let _ = writeln!(
                    acc,
                    "{index} level=info category=runtime action=install_sdk message=event-{index}"
                );
                acc
            }),
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("runtime-install_sdk.log"),
            "install event\n",
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("update-update_check.log"),
            "update event\n",
        )?;

        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("  Recent command files:"));
        assert!(rendered.contains("runtime-install_sdk.log"));
        assert!(rendered.contains("update-update_check.log"));
        assert!(rendered.contains("Recent activity: last 8 line(s)"));
        assert!(!rendered.contains("event-0"));
        assert!(!rendered.contains("event-1"));
        assert!(rendered.contains("Install: event-2"));
        assert!(rendered.contains("event-2"));
        assert!(rendered.contains("event-9"));
        assert!(rendered.contains("  Lines: 10 of 10 recent line(s)"));
        assert!(rendered.contains("    command log runtime-install_sdk.log: install event"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_text_lists_screen_command_logs() -> Result<()> {
        let (root, paths) = test_paths("logs-screen-command");
        let screen_dir = paths.data_dir.join("logs").join("tui");
        fs::create_dir_all(&screen_dir)?;
        let screen_log = screen_dir.join("12345-install-the-rock-sdk.log");
        fs::write(
            &screen_log,
            "title: Install TheRock SDK\n\
             recent_live_output:\n\
             Output: resolving torch wheels\n\
             command_output:\n\
             stdout:\n\
             resolved torch\n",
        )?;

        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("  Screen command logs:"));
        assert!(rendered.contains("screen/12345-install-the-rock-sdk.log"));
        assert!(rendered.contains("screen command log 12345-install-the-rock-sdk.log"));
        assert!(rendered.contains("Output: resolving torch wheels"));
        assert!(rendered.contains("  Lines: 6 of 6 recent line(s)"));
        let filtered = render_logs_browser_text(&paths, Some("torch wheels"));
        assert!(filtered.contains("Search: torch wheels"));
        assert!(filtered.contains("Output: resolving torch wheels"));
        assert!(filtered.contains("screen command log 12345-install-the-rock-sdk.log"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_browser_text_filters_lifecycle_and_action_logs() -> Result<()> {
        let (root, paths) = test_paths("logs-browser-search");
        fs::create_dir_all(paths.data_dir.join("logs").join("cli"))?;
        fs::write(
            cli_lifecycle_log_path(&paths),
            "1 level=info category=runtime action=install_sdk message=installed sdk\n\
             2 level=info category=service action=serve message=server ready\n",
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("service-serve.log"),
            "server ready\nmodel warmed\n",
        )?;

        let rendered = render_logs_browser_text(&paths, Some("server"));

        assert!(rendered.contains("  Search: server"));
        assert!(rendered.contains("  Lines: 2 of 4 recent line(s)"));
        assert!(rendered.contains("    recent activity: Service event: server ready"));
        assert!(rendered.contains("    command log service-serve.log: server ready"));
        assert!(!rendered.contains("installed sdk"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_browser_page_text_paginates_matching_lines() -> Result<()> {
        let (root, paths) = test_paths("logs-browser-pages");
        let action_dir = paths.data_dir.join("logs").join("cli");
        fs::create_dir_all(&action_dir)?;
        fs::write(action_dir.join("a.log"), "alpha-1\nalpha-2\nalpha-3\n")?;
        fs::write(action_dir.join("b.log"), "alpha-4\nalpha-5\nalpha-6\n")?;

        let rendered = render_logs_browser_page_text(&paths, Some("alpha"), 1, 4);

        assert_eq!(logs_browser_page_count(&paths, Some("alpha"), 4), 2);
        assert!(rendered.contains("  Page: 2 of 2"));
        assert!(rendered.contains("  Showing: 5-6 of 6"));
        assert!(!rendered.contains("alpha-1"));
        assert!(rendered.contains("alpha-5"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn cli_lifecycle_tail_lines_render_compactly() {
        let rendered = format_cli_lifecycle_tail_line(
            "42 level=error category=runtime action=install_sdk service_id=<none> message=line one",
        );

        assert_eq!(rendered, "Install (error): line one");
    }

    #[test]
    fn render_service_logs_text_tails_manifest_log() -> Result<()> {
        let (root, paths) = test_paths("service-logs");
        paths.ensure()?;

        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_qwen35_primary",
            "pytorch",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_preferred".to_owned()),
        );
        record.status = "ready".to_owned();
        record.write()?;

        let mut log = String::new();
        for index in 1..=90 {
            let _ = writeln!(log, "entry-{index:03}");
        }
        fs::write(&record.log_path, log)?;

        let rendered = render_service_logs_text(&paths, "svc_qwen35_primary")?;
        assert!(rendered.contains("Service Log"));
        assert!(rendered.contains("Service: svc_qwen35_primary"));
        assert!(rendered.contains("Engine: pytorch"));
        assert!(rendered.contains("Status: starting"));
        assert!(rendered.contains("File locations: shown"));
        assert!(rendered.contains(&format!(
            "  Details file: {}",
            record.manifest_path.display()
        )));
        assert!(rendered.contains(&format!("  Log file: {}", record.log_path.display())));
        assert!(!rendered.contains("entry-010"));
        assert!(rendered.contains("entry-011"));
        assert!(rendered.contains("entry-090"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_services_text_lists_live_services_by_default_and_all_on_request() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let (root, paths) = test_paths("services-list");
        paths.ensure()?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let ready_port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut request = [0_u8; 512];
            let _ = stream.read(&mut request)?;
            let body = r#"{"data":[{"id":"Qwen/Qwen3.5"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(())
        });
        let current_pid = std::process::id();

        for (service_id, status, port) in [
            ("svc-ready", "ready", ready_port),
            ("svc-starting", "starting", 11436_u16),
            ("svc-failed", "failed", 11437_u16),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "pytorch",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                port,
                "managed",
                current_pid,
                Some("therock-release".to_owned()),
                None,
                Some("gpu_required".to_owned()),
            );
            record.status = status.to_owned();
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        server
            .join()
            .expect("fake models server should not panic")?;
        let all = render_services_text(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local Servers"));
        assert!(rendered.contains("Status: 1 ready, 1 starting"));
        assert!(!rendered.contains("Past attempts"));
        assert!(rendered.contains("- svc-ready"));
        assert!(rendered.contains("  stop: rocm services stop svc-ready --yes"));
        assert!(!rendered.contains("- svc-failed"));
        assert!(all.contains("- svc-failed"));
        assert!(all.contains("  restart: rocm services restart svc-failed --yes"));
        assert!(!rendered.contains("running servers"));
        Ok(())
    }

    #[test]
    fn render_services_text_demotes_stale_ready_record() -> Result<()> {
        let (root, paths) = test_paths("services-stale-ready");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-stale-ready",
            "lemonade",
            "qwen",
            providers::BUILTIN_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            9,
            "managed",
            999_999_999,
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();
        record.engine_pid = Some(999_999_999);
        record.write()?;

        let rendered = render_services_text(&paths, false)?;
        let all = render_services_text(&paths, true)?;
        let reloaded = load_managed_service(&paths, "svc-stale-ready")?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("No local servers are running."));
        assert!(all.contains("- svc-stale-ready"));
        assert!(all.contains("  status: stopped"));
        assert_eq!(reloaded.status, "stopped");
        Ok(())
    }

    #[test]
    fn duplicate_managed_launch_detected_across_distinct_service_ids() -> Result<()> {
        // `generate_service_id` embeds a timestamp, so a second launch for the
        // same engine+model has a DIFFERENT service_id. The guard must still
        // detect the live service by (engine, canonical_model_id), and return
        // the newest live match. `starting` skips the endpoint probe; the
        // current process id is a guaranteed-live PID.
        let (root, paths) = test_paths("dup-managed-distinct-ids");
        paths.ensure()?;

        // Older, dead manifest for the same engine+model (distinct service_id).
        let mut dead = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-1000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11500,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        dead.status = "ready".to_owned();
        dead.engine_pid = Some(999_999_999);
        dead.created_at_unix_ms = 1000;
        dead.write()?;

        // Newer, live manifest for the same engine+model (distinct service_id).
        let mut live = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-2000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11501,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        live.status = "starting".to_owned();
        live.engine_pid = Some(std::process::id());
        live.created_at_unix_ms = 2000;
        live.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        let found = found.expect("a live managed service should be detected by engine+model");
        assert_eq!(
            found.service_id, "lemonade-qwen-2000",
            "should return the newest live match"
        );
        assert!(managed_service_is_live(&found));
        Ok(())
    }

    #[test]
    fn dead_managed_service_allows_relaunch() -> Result<()> {
        // A stale manifest with dead PIDs must NOT block a relaunch: liveness
        // refresh demotes it to "stopped", so the guard returns None.
        let (root, paths) = test_paths("dup-managed-dead");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11502,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.engine_pid = Some(999_999_999);
        record.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a dead managed service must not block relaunch"
        );
        Ok(())
    }

    #[test]
    fn live_service_for_other_model_does_not_block() -> Result<()> {
        // A live service for a DIFFERENT model must not match — the guard keys
        // on the model, not just the engine.
        let (root, paths) = test_paths("dup-managed-other-model");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-other-1",
            "lemonade",
            "other",
            "other-canonical",
            "127.0.0.1",
            11503,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;

        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            found.is_none(),
            "a live service for a different model must not match"
        );
        Ok(())
    }

    #[test]
    fn missing_manifest_allows_launch() {
        // No services dir / manifests → nothing to detect, launch proceeds.
        let (root, paths) = test_paths("dup-managed-missing");
        let found = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        assert!(found.is_none());
    }

    #[test]
    fn services_tool_result_text_includes_running_interpretation() {
        let (_root, paths) = test_paths("services-tool-text");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-vllm",
            "vllm",
            "Qwen/Qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();

        let rendered = render_services_tool_result_text(&[record]);

        assert!(rendered.contains("status_meaning: ready/running = running"));
        assert!(rendered.contains("engine=vllm"));
        assert!(rendered.contains("running_state=running"));
    }

    #[test]
    fn service_actions_require_yes_and_render_sandbox_result() {
        let (_root, paths) = test_paths("services-action-approval");
        let error = run_approved_service_action(&paths, "stop_server", "svc-qwen", false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires --yes"));
        assert!(error.contains("rocm services stop svc-qwen --yes"));

        let rendered = render_service_action_result(
            "stop_server",
            &serde_json::json!({
                "output": {
                    "status": "stopped",
                    "result": {
                        "service": {
                            "service_id": "svc-qwen",
                            "status": "stopped",
                            "endpoint_url": "http://127.0.0.1:11435/v1"
                        },
                        "signaled_pids": [1234, 5678]
                    }
                }
            }),
        );

        assert!(rendered.contains("Local server stopped"));
        assert!(rendered.contains("service: svc-qwen"));
        assert!(rendered.contains("status: stopped"));
        assert!(rendered.contains("stopped processes: 2"));
    }

    #[test]
    fn services_is_structured_not_freeform() {
        let invocation = parse_freeform_invocation(&["services".to_owned()]);
        assert!(!should_treat_as_freeform(&invocation));
        Cli::try_parse_from(["rocm", "services"]).expect("services should be a real command");
        Cli::try_parse_from(["rocm", "services", "list"])
            .expect("services list should be a real command");
        Cli::try_parse_from(["rocm", "services", "logs", "svc-qwen"])
            .expect("services logs should be a real command");
        Cli::try_parse_from(["rocm", "services", "stop", "svc-qwen", "--yes"])
            .expect("services stop should accept --yes");
        Cli::try_parse_from(["rocm", "services", "restart", "svc-qwen", "--yes"])
            .expect("services restart should accept --yes");
    }

    #[test]
    fn install_sdk_accepts_family_override() {
        Cli::try_parse_from([
            "rocm",
            "install",
            "sdk",
            "--channel",
            "release",
            "--format",
            "wheel",
            "--prefix",
            "D:\\ROCm\\therock_venvs",
            "--family",
            "gfx110X-all",
        ])
        .expect("install sdk should accept a TheRock family override");
    }

    #[test]
    fn top_level_cli_commands_are_not_treated_as_freeform() {
        for command in [
            "examine",
            "bootstrap",
            "version",
            "setup",
            "chat",
            "install",
            "update",
            "runtimes",
            "engines",
            "model",
            "models",
            "serve",
            "comfyui",
            "comfy",
            "services",
            "automations",
            "config",
            "logs",
            "daemon",
            "dash",
            "uninstall",
            "completions",
            "help",
        ] {
            let invocation = parse_freeform_invocation(&[command.to_owned()]);
            assert!(
                !should_treat_as_freeform(&invocation),
                "{command} should parse as a structured CLI command, not natural language"
            );
        }
        Cli::try_parse_from(["rocm", "setup"]).expect("setup should parse");
        Cli::try_parse_from(["rocm", "bootstrap"]).expect("bootstrap setup should parse");
        Cli::try_parse_from(["rocm", "setup", "status"]).expect("setup status should parse");
        Cli::try_parse_from(["rocm", "setup", "reset"]).expect("setup reset should parse");
        Cli::try_parse_from(["rocm", "models"]).expect("models alias should parse");
        Cli::try_parse_from(["rocm", "comfyui", "status"]).expect("comfyui status should parse");
        Cli::try_parse_from(["rocm", "comfyui", "logs", "--lines", "3"])
            .expect("comfyui logs should parse");
        Cli::try_parse_from(["rocm", "comfyui", "stop"]).expect("comfyui stop should parse");
        Cli::try_parse_from(["rocm", "comfy", "logs"]).expect("comfy alias should parse");
    }

    #[test]
    fn setup_reset_cli_output_is_plain_and_persists_first_time_prompt() -> Result<()> {
        let (_root, paths) = test_paths("setup-reset-cli");
        let mut config = RocmCliConfig {
            onboarding_dismissed: true,
            setup: rocm_core::SetupConfig {
                completed: true,
                therock_venv: Some(paths.data_dir.join("envs").join("default")),
                cli_install_dir: None,
            },
            ..Default::default()
        };
        config.provider_config_mut("openai").enabled = true;
        config.save(&paths)?;

        let rendered = reset_setup_prompt_state(&paths, &mut config)?;

        assert!(rendered.contains("Setup will show again"));
        assert!(rendered.contains("ROCm installs were not deleted"));
        assert!(rendered.contains("API keys"));
        assert!(!rendered.contains("request plan"));
        assert!(!rendered.contains("planner:"));
        assert!(!rendered.contains("tool_schema"));

        let saved = RocmCliConfig::load(&paths)?;
        assert!(!saved.onboarding_dismissed);
        assert!(!saved.setup.completed);
        assert!(saved.setup.therock_venv.is_some());
        assert!(saved.provider_enabled("openai"));
        Ok(())
    }

    #[test]
    fn setup_status_reports_completed_active_runtime() -> Result<()> {
        let (root, paths) = test_paths("setup-status-completed-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-status",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key.clone()),
            setup: rocm_core::SetupConfig {
                completed: true,
                therock_venv: Some(manifest.install_root.clone()),
                cli_install_dir: None,
            },
            ..Default::default()
        };

        let rendered = render_setup_status_text(&paths, &config)?;

        assert!(rendered.contains("status: completed"), "{rendered}");
        assert!(
            rendered.contains(&format!(
                "install folder: {}",
                manifest.install_root.display()
            )),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_key: release-pip-gfx120x-all-status"),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_id: therock-release:gfx120X-all"),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_status: ready"),
            "{rendered}"
        );
        assert!(rendered.contains("rocm help"), "{rendered}");

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn setup_status_reports_first_time_when_not_completed() -> Result<()> {
        let (_root, paths) = test_paths("setup-status-first-time");
        let config = RocmCliConfig::default();

        let rendered = render_setup_status_text(&paths, &config)?;

        assert!(rendered.contains("status: first-time setup will show"));
        assert!(rendered.contains("active_runtime_status: <unset>"));
        Ok(())
    }

    #[test]
    fn serve_bind_validation_requires_public_ack() {
        validate_bind_host("127.0.0.1", false).unwrap();
        validate_bind_host("localhost", false).unwrap();
        validate_bind_host("::1", false).unwrap();
        let error = validate_bind_host("0.0.0.0", false).unwrap_err();
        assert!(
            error.to_string().contains("--allow-public-bind"),
            "{error:#}"
        );
        validate_bind_host("0.0.0.0", true).unwrap();
    }

    #[test]
    fn serve_engine_selection_uses_shared_recipe_when_no_override_exists() {
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");

        let selection = select_serve_engine(None, None, Some(&recipe));

        assert_eq!(
            selection,
            ServeEngineSelection {
                engine: "vllm".to_owned(),
                source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
            }
        );
        assert_eq!(
            serve_engine_selection_line(&selection),
            "  engine_selection: recipe preferred engine; pass --engine <engine> to override; no automatic fallback"
        );
        assert_eq!(
            serve_model_ref_for_engine("qwen32b", Some(&recipe), "vllm"),
            "Qwen/Qwen3-32B-FP8"
        );
    }

    #[test]
    fn explicit_engine_override_keeps_alias_when_shared_recipe_is_for_another_engine() {
        let recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");

        assert_eq!(
            serve_model_ref_for_engine("qwen", Some(&recipe), "lemonade"),
            "Qwen3-4B-Instruct-2507-GGUF"
        );
        assert_eq!(
            serve_model_ref_for_engine("qwen", Some(&recipe), "pytorch"),
            "qwen"
        );
    }

    #[test]
    fn serve_engine_selection_respects_explicit_and_configured_engines() {
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");

        let explicit = select_serve_engine(Some("llama.cpp"), Some("pytorch"), Some(&recipe));
        let configured = select_serve_engine(None, Some("pytorch"), Some(&recipe));

        assert_eq!(
            explicit,
            ServeEngineSelection {
                engine: "llama.cpp".to_owned(),
                source: "explicit --engine",
            }
        );
        assert_eq!(
            configured,
            ServeEngineSelection {
                engine: "pytorch".to_owned(),
                source: "configured default_engine",
            }
        );
    }

    #[test]
    fn protocol_engine_recipe_hint_maps_selected_engine_metadata() {
        let mut recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");
        recipe.engine_recipes = vec![
            rocm_core::ModelRecipeEngineRecord {
                engine: "vllm".to_owned(),
                required_flags: vec!["--enable-auto-tool-choice".to_owned()],
                parser_settings: BTreeMap::from([(
                    "reasoning_parser".to_owned(),
                    "qwen3".to_owned(),
                )]),
                preferred_endpoint: Some(rocm_core::ModelRecipeEndpointRecord {
                    endpoint_mode: "openai".to_owned(),
                    settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
                }),
                unsupported_combinations: vec![
                    rocm_core::ModelRecipeUnsupportedCombinationRecord {
                        combination: "native Windows GPU serving".to_owned(),
                        reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
                    },
                ],
                notes: vec!["adapter hint".to_owned()],
            },
            rocm_core::ModelRecipeEngineRecord {
                engine: "sglang".to_owned(),
                required_flags: vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()],
                parser_settings: BTreeMap::new(),
                preferred_endpoint: None,
                unsupported_combinations: Vec::new(),
                notes: Vec::new(),
            },
        ];

        let hint = protocol_engine_recipe_hint(&recipe, "vllm").expect("vllm hint");

        assert_eq!(hint.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
        assert_eq!(hint.engine, "vllm");
        assert_eq!(
            hint.required_flags,
            vec!["--enable-auto-tool-choice".to_owned()]
        );
        assert_eq!(
            hint.parser_settings
                .get("reasoning_parser")
                .map(String::as_str),
            Some("qwen3")
        );
        assert_eq!(
            hint.preferred_endpoint
                .as_ref()
                .map(|endpoint| endpoint.endpoint_mode.as_str()),
            Some("openai")
        );
        assert_eq!(
            hint.preferred_endpoint
                .as_ref()
                .and_then(|endpoint| endpoint.settings.get("streaming"))
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(hint.unsupported_combinations.len(), 1);
        assert_eq!(hint.notes, vec!["adapter hint".to_owned()]);
        let serve_lines = render_serve_engine_recipe_lines(&hint);
        assert!(serve_lines.contains(
            "engine_recipe_policy: selected-engine required_flags are applied at launch"
        ));
        assert!(serve_lines.contains("engine_recipe_required_flags: --enable-auto-tool-choice"));
        assert!(protocol_engine_recipe_hint(&recipe, "pytorch").is_none());
    }

    #[test]
    fn parse_device_policy_defaults_to_gpu_required_without_cpu_fallback() -> Result<()> {
        assert_eq!(parse_device_policy(None)?, DevicePolicy::GpuRequired);
        assert_eq!(parse_device_policy(Some("gpu"))?, DevicePolicy::GpuRequired);
        assert_eq!(
            parse_device_policy(Some("gpu_preferred"))?,
            DevicePolicy::GpuRequired
        );
        let cpu = parse_device_policy(Some("cpu")).unwrap_err().to_string();
        assert!(cpu.contains("CPU mode is not a fallback path"));
        Ok(())
    }

    #[test]
    fn driver_plan_ubuntu_2404_uses_official_dkms_commands() {
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let commands = plan
            .commands
            .iter()
            .map(|command| command.command.as_str())
            .collect::<Vec<_>>();

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        assert!(
            plan.preflight_checks
                .iter()
                .any(|check| check.contains("sudo -v"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("linux-headers-$(uname -r)"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("linux-modules-extra-$(uname -r)"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("repo.radeon.com/graphics"))
        );
        assert!(
            commands
                .iter()
                .any(|command| command.contains("amdgpu-dkms"))
        );
        let rendered = render_driver_install_plan(&plan, false, false);
        assert!(rendered.contains("approval: required"));
        assert!(rendered.contains("preflight_checks:"));
        assert!(rendered.contains("root access: run as root, or ensure `sudo -v` succeeds"));
        assert!(rendered.contains("execution_commands:"));
        assert!(rendered.contains("Prepare: sudo apt-get update"));
        assert!(rendered.contains("Execute: sudo apt-get install -y amdgpu-dkms"));
        assert!(rendered.contains("post_reboot_check_commands:"));
        assert!(rendered.contains("dkms status amdgpu"));
        assert!(rendered.contains("rerun with --yes"));
    }

    #[test]
    fn driver_install_approval_only_applies_to_supported_mutating_unapproved_plan() -> Result<()> {
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let dkms_plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let preflight_plan =
            build_driver_install_plan(&test_examine("linux", false), os_release, false);
        let windows_plan = build_driver_install_plan(&test_examine("windows", false), "", true);

        let dkms_flags = parse_driver_install_flags(&["driver", "--dkms"])?;
        let dry_run_flags = parse_driver_install_flags(&["driver", "--dkms", "--dry-run"])?;
        let approved_flags = parse_driver_install_flags(&["driver", "--dkms", "--yes"])?;
        let preflight_flags = parse_driver_install_flags(&["driver"])?;

        assert!(driver_install_flags_require_approval(
            &dkms_plan,
            &dkms_flags
        ));
        assert!(!driver_install_flags_require_approval(
            &dkms_plan,
            &dry_run_flags
        ));
        assert!(!driver_install_flags_require_approval(
            &dkms_plan,
            &approved_flags
        ));
        assert!(driver_install_flags_require_tui_approval(
            &dkms_plan,
            &approved_flags
        ));
        assert!(!driver_install_flags_require_approval(
            &preflight_plan,
            &preflight_flags
        ));
        assert!(!driver_install_flags_require_approval(
            &windows_plan,
            &dkms_flags
        ));
        let reconcile_flags = parse_driver_install_flags(&["driver", "--reconcile"])?;
        assert!(!driver_install_flags_require_approval(
            &dkms_plan,
            &reconcile_flags
        ));
        assert!(parse_driver_install_flags(&["driver", "--reconcile", "--dkms"]).is_err());
        Ok(())
    }

    #[test]
    fn driver_reconcile_without_state_gives_non_privileged_guidance() -> Result<()> {
        let (root, paths) = test_paths("driver-reconcile-empty");

        let rendered = reconcile_driver_install(&paths)?;

        assert!(rendered.contains("driver install reconciliation"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("privileged_commands: <none>"));
        assert!(rendered.contains("no prior driver execution state found"));
        assert!(rendered.contains("rocm install driver --dkms"));
        assert!(!driver_install_state_path(&paths).exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn driver_reconcile_updates_state_after_reboot() -> Result<()> {
        let (root, paths) = test_paths("driver-reconcile-state");
        let pre_driver = rocm_core::DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: "not_detected".to_owned(),
            detail: None,
        };
        let current_driver = rocm_core::DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: "amdgpu_available".to_owned(),
            detail: Some("/dev/kfd is present".to_owned()),
        };
        let mut state = DriverInstallState {
            approved_at_unix_ms: 1,
            executed_at_unix_ms: Some(2),
            pre_driver,
            post_driver: None,
            boot_id_at_execution: Some("old-boot".to_owned()),
            reboot_required: true,
            reboot_observed: false,
            commands: vec!["sudo apt-get install -y amdgpu-dkms".to_owned()],
            reconciled_at_unix_ms: None,
            reconciliation: None,
        };
        let checks = vec![
            DriverPassiveCheck {
                name: "/dev/kfd".to_owned(),
                status: "present".to_owned(),
                detail: "KFD device node".to_owned(),
            },
            DriverPassiveCheck {
                name: "/dev/dri/renderD*".to_owned(),
                status: "missing".to_owned(),
                detail: "DRM render node".to_owned(),
            },
        ];

        let rendered = reconcile_driver_install_state(
            &paths,
            &mut state,
            current_driver,
            Some("new-boot".to_owned()),
            checks,
        )?;
        let saved = read_driver_install_state(&paths)?.expect("state should be saved");

        assert!(rendered.contains("reboot_observed: true"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("privileged_commands: <none>"));
        assert!(rendered.contains("driver_status: amdgpu_available"));
        assert!(rendered.contains("passive_check_summary: total=2 present=1 missing=1"));
        assert!(rendered.contains("/dev/dri/renderD*: missing"));
        assert!(rendered.contains("missing passive checks"));
        assert!(saved.reboot_observed);
        assert!(saved.reconciled_at_unix_ms.is_some());
        assert_eq!(
            saved
                .reconciliation
                .as_ref()
                .map(|value| value.driver.status.as_str()),
            Some("amdgpu_available")
        );
        let reconciliation = saved.reconciliation.as_ref().expect("reconciliation saved");
        assert_eq!(reconciliation.check_summary.total, 2);
        assert_eq!(reconciliation.check_summary.present, 1);
        assert_eq!(reconciliation.check_summary.missing, 1);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn driver_passive_check_summary_counts_non_present_as_missing() {
        let summary = summarize_driver_passive_checks(&[
            DriverPassiveCheck {
                name: "/dev/kfd".to_owned(),
                status: "present".to_owned(),
                detail: "KFD".to_owned(),
            },
            DriverPassiveCheck {
                name: "/dev/dri/renderD*".to_owned(),
                status: "missing".to_owned(),
                detail: "render".to_owned(),
            },
            DriverPassiveCheck {
                name: "dkms".to_owned(),
                status: "error".to_owned(),
                detail: "dkms status failed".to_owned(),
            },
        ]);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.present, 1);
        assert_eq!(summary.missing, 2);
    }

    #[test]
    fn driver_plan_default_linux_preflight_has_no_execution_commands() {
        let os_release = r#"
ID=ubuntu
VERSION_ID="24.04"
VERSION_CODENAME=noble
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, false);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(!plan.mutating);
        assert!(plan.commands.is_empty());
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        assert!(!rendered.contains("sudo apt-get"));
        assert!(rendered.contains("add --dkms"));
    }

    #[test]
    fn driver_plan_debian_12_omits_linux_modules_extra() {
        let os_release = r#"
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("linux-headers-$(uname -r)"));
        assert!(!rendered.contains("linux-modules-extra-$(uname -r)"));
        assert!(rendered.contains("amdgpu-dkms"));
        assert!(rendered.contains("dry run only"));
    }

    #[test]
    fn driver_plan_rhel_97_uses_documented_dnf_commands() {
        let os_release = r#"
ID=rhel
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(plan.mutating);
        assert_eq!(plan.policy, "linux_official_amd_dkms_wrapper");
        assert!(rendered.contains("`dnf` package manager is available"));
        assert!(rendered.contains("kernel-headers-$(uname -r)"));
        assert!(rendered.contains("kernel-devel-$(uname -r)"));
        assert!(rendered.contains("kernel-devel-matched-$(uname -r)"));
        assert!(rendered.contains(
            "repo.radeon.com/amdgpu-install/${ROCM_CLI_AMDGPU_VERSION:-7.2.4}/rhel/9.7/"
        ));
        assert!(rendered.contains("amdgpu-install-${ROCM_CLI_AMDGPU_VERSION:-7.2.4}.${ROCM_CLI_AMDGPU_PACKAGE_RELEASE:-70204}-1.el9.noarch.rpm"));
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn driver_plan_oracle_linux_101_uses_el_10_uek_flow() {
        let os_release = r#"
ID=ol
VERSION_ID="10.1"
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(plan.supported);
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("kernel-uek-devel-$(uname -r)"));
        assert!(
            rendered.contains(
                "repo.radeon.com/amdgpu-install/${ROCM_CLI_AMDGPU_VERSION:-7.2.4}/el/10/"
            )
        );
        assert!(rendered.contains("amdgpu-install-${ROCM_CLI_AMDGPU_VERSION:-7.2.4}.${ROCM_CLI_AMDGPU_PACKAGE_RELEASE:-70204}-1.el10.noarch.rpm"));
        assert!(rendered.contains("dry run only"));
    }

    #[test]
    fn driver_plan_rocky_97_uses_el_dnf_flow() {
        let os_release = r#"
ID=rocky
VERSION_ID="9.7"
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(
            rendered
                .contains("sudo dnf install -y kernel-headers kernel-devel kernel-devel-matched")
        );
        assert!(
            rendered.contains(
                "repo.radeon.com/amdgpu-install/${ROCM_CLI_AMDGPU_VERSION:-7.2.4}/el/9.7/"
            )
        );
        assert!(rendered.contains("Execute: sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn driver_plan_sles_157_uses_documented_zypper_commands() {
        let os_release = r#"
ID=sles
VERSION_ID="15.7"
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(plan.supported);
        assert!(rendered.contains("`zypper` package manager is available"));
        assert!(rendered.contains("SUSEConnect"));
        assert!(rendered.contains("sle-module-desktop-applications/15.7/x86_64"));
        assert!(rendered.contains("sudo zypper install -y kernel-default-devel"));
        assert!(rendered.contains(
            "repo.radeon.com/amdgpu-install/${ROCM_CLI_AMDGPU_VERSION:-7.2.4}/sle/15.7/"
        ));
        assert!(rendered.contains("sudo zypper --no-gpg-checks install -y"));
        assert!(rendered.contains("Execute: sudo zypper install -y amdgpu-dkms"));
        assert!(rendered.contains("approval: required"));
    }

    #[test]
    fn driver_plan_unsupported_linux_is_non_mutating() {
        let os_release = r#"
ID=fedora
VERSION_ID="41"
"#;
        let plan = build_driver_install_plan(&test_examine("linux", false), os_release, true);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert!(rendered.contains("unsupported_linux_dkms_plan"));
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("no driver commands will be executed"));
        assert!(!rendered.contains("sudo dnf install -y amdgpu-dkms"));
    }

    #[test]
    fn windows_install_driver_is_validate_only() {
        let plan = build_driver_install_plan(&test_examine("windows", false), "", true);
        let rendered = render_driver_install_plan(&plan, false, true);

        assert!(!plan.supported);
        assert!(!plan.mutating);
        assert_eq!(plan.policy, "windows_validate_only");
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        assert!(rendered.contains("post_reboot_checks:"));
        assert!(rendered.contains("use `rocm examine`"));
        assert!(rendered.contains("rocm examine"));
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn wsl_install_driver_uses_rocdxg_guidance_without_dkms() {
        let plan = build_driver_install_plan(&test_examine("linux", true), "", true);
        let rendered = render_driver_install_plan(&plan, false, false);

        assert!(!plan.supported);
        assert_eq!(plan.policy, "wsl_rocdxg");
        assert!(rendered.contains("approval: not required"));
        assert!(rendered.contains("execution_commands: <none>"));
        assert!(rendered.contains("scripts/wsl_setup_rocdxg.sh"));
        assert!(!rendered.contains("amdgpu-dkms"));
    }

    #[test]
    fn resolve_engine_selection_uses_default_runtime_after_engine_prefs() {
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let selection = resolve_engine_selection(&config, "pytorch", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_default_runtime_id")
        );

        config.active_runtime_key = Some("release-pip-gfx120x-all-7-13-0".to_owned());
        let selection = resolve_engine_selection(&config, "pytorch", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.engine_config_mut("pytorch").preferred_runtime_id =
            Some("therock-nightly:gfx120X-all".to_owned());
        let selection = resolve_engine_selection(&config, "pytorch", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.active_runtime_key = None;
        let selection = resolve_engine_selection(&config, "pytorch", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("therock-nightly:gfx120X-all")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_preferred_runtime_id")
        );
    }

    #[test]
    fn engine_selection_uses_single_ready_runtime_without_active_marker() -> Result<()> {
        let (root, paths) = test_paths("single-ready-runtime-selection");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let selection = validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(&RocmCliConfig::default(), "pytorch", None, None),
        )?;

        assert_eq!(
            selection.runtime_id.as_deref(),
            Some(manifest.runtime_key.as_str())
        );
        assert_eq!(selection.source.as_deref(), Some("single_ready_runtime"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn engine_selection_keeps_multiple_ready_runtimes_explicit() -> Result<()> {
        let (root, paths) = test_paths("multiple-ready-runtime-selection");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let selection = validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(&RocmCliConfig::default(), "pytorch", None, None),
        )?;

        assert!(selection.runtime_id.is_none());
        assert!(selection.env_id.is_none());
        assert!(selection.source.is_none());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_config_text_includes_default_runtime() {
        let (_root, paths) = test_paths("config-default-runtime");
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_config_text(&paths, &config);

        assert!(rendered.contains("default_runtime_id: therock-release:gfx120X-all"));
        assert!(rendered.contains("active_runtime_key: <unset>"));
    }

    #[test]
    fn render_config_text_includes_telemetry_policy() {
        let (_root, paths) = test_paths("config-telemetry-policy");
        let mut config = RocmCliConfig::default();

        let local = render_config_text(&paths, &config);
        assert!(local.contains("telemetry_mode: local"));
        assert!(local.contains("telemetry_policy: local amd-smi inspection only"));
        assert!(local.contains("no external reporting is implemented"));
        assert!(local.contains("  providers:"));
        assert!(local.contains("    local: enabled"));
        assert!(local.contains("    openai: disabled"));
        assert!(local.contains("    anthropic: disabled"));

        config.telemetry.mode = TELEMETRY_MODE_OFF.to_owned();
        config.provider_config_mut("openai").enabled = true;
        let off = render_config_text(&paths, &config);
        assert!(off.contains("telemetry_mode: off"));
        assert!(off.contains("telemetry_policy: disabled"));
        assert!(off.contains("no local polling"));
        assert!(off.contains("    openai: enabled"));
    }

    #[test]
    fn engine_install_runtime_selection_requires_configured_runtime() -> Result<()> {
        let (root, paths) = test_paths("engine-install-runtime-selection");
        let error =
            resolve_engine_install_runtime_id(&paths, &RocmCliConfig::default(), "pytorch", None)
                .unwrap_err()
                .to_string();
        assert!(error.contains("no active ROCm runtime is configured"));
        assert_eq!(
            resolve_engine_install_runtime_id(&paths, &RocmCliConfig::default(), "lemonade", None)?,
            "lemonade-embeddable-10.6.0"
        );
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;

        let config = RocmCliConfig {
            active_runtime_key: Some("release-pip-gfx120x-all".to_owned()),
            ..RocmCliConfig::default()
        };
        assert_eq!(
            resolve_engine_install_runtime_id(&paths, &config, "pytorch", None)?,
            "release-pip-gfx120x-all"
        );
        assert_eq!(
            resolve_engine_install_runtime_id(
                &paths,
                &config,
                "pytorch",
                Some("therock-release:gfx120X-all".to_owned())
            )?,
            "release-pip-gfx120x-all"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_selector_recovers_setup_runtime_registry_from_local_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-selector-recover-setup");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-local-manifest",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let install_root = manifest.install_root.clone();
        let mut config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };
        config.setup.completed = true;
        config.setup.therock_venv = Some(install_root.clone());
        config.save(&paths)?;

        let rebased_paths = paths.with_managed_root(install_root, false);
        let rebased_registry = runtime_registry_dir(&rebased_paths);
        let _ = fs::remove_dir_all(&rebased_registry);

        assert!(!runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());
        assert_eq!(
            resolve_runtime_selector_to_exact_key(
                &rebased_paths,
                &manifest.runtime_key,
                "test active runtime"
            )?,
            manifest.runtime_key
        );
        assert!(runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());

        let rendered = render_runtimes_text(&rebased_paths, &config)?;
        assert!(rendered.contains("release-pip-gfx120x-all-local-manifest"));
        assert!(rendered.contains("status=ready"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn sdk_install_finalization_activates_runtime_and_setup_root() -> Result<()> {
        let (root, paths) = test_paths("sdk-install-finalization");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-finalized",
            "therock-release:gfx120X-all",
            "7.13.0",
            42,
        )?;

        let finalized = finalize_successful_sdk_install(&paths)?
            .context("sdk install finalization should select the installed runtime")?;
        let rebased_paths = paths.with_managed_root(manifest.install_root.clone(), false);
        let config = RocmCliConfig::load(&rebased_paths)?;

        assert_eq!(finalized.runtime_key, manifest.runtime_key);
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some(manifest.runtime_id.as_str())
        );
        assert!(config.setup.completed);
        assert_eq!(
            config.setup.therock_venv.as_deref(),
            Some(manifest.install_root.as_path())
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(manifest.runtime_key.as_str())
        );
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some(manifest.runtime_id.as_str())
        );
        assert!(runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());
        assert!(active_runtime_marker_path(&rebased_paths).is_file());

        let success = render_sdk_install_success(&finalized);
        assert!(success.contains("ROCm SDK installed successfully."));
        assert!(success.contains("next step: run `rocm help`"));
        assert!(success.contains(&manifest.install_root.display().to_string()));
        assert!(!success.contains("config:"));
        assert!(!success.contains("marker:"));

        let mut examine = String::new();
        append_examine_runtime_state(&mut examine, &rebased_paths, &config)?;
        assert!(examine.contains("active_runtime_status: ready"));
        assert!(examine.contains("setup_runtime_root:"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn env_root_for_runtime_uses_runtime_install_root() -> Result<()> {
        let (root, paths) = test_paths("engine-env-root-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;

        let engine_root = env_root_for_runtime(&paths, "pytorch", &manifest.runtime_key)?;

        assert_eq!(engine_root, Some(manifest.install_root.join("engines")));
        assert_eq!(
            env_root_for_runtime(&paths, "lemonade", &manifest.runtime_key)?,
            None
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn env_root_for_engine_install_uses_active_runtime_root_for_lemonade() -> Result<()> {
        let (root, paths) = test_paths("lemonade-engine-env-root-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        let engine_root =
            env_root_for_engine_install(&paths, &config, "lemonade", "lemonade-embeddable")?;

        assert_eq!(engine_root, Some(manifest.install_root.join("engines")));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn resolve_pytorch_env_uses_active_therock_runtime() -> Result<()> {
        let (root, paths) = test_paths("pytorch-active-runtime-env");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        let resolved = resolve_engine_env(&paths, &config, "pytorch", None, None)?;

        assert_eq!(resolved.runtime_id, manifest.runtime_key);
        assert_eq!(resolved.env_path, manifest.install_root);
        assert_eq!(resolved.managed_env_id, None);
        assert!(!resolved.env_path.join("engines").exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn engine_runtime_selection_rejects_ambiguous_default_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("engine-runtime-ambiguous-default");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        write_test_pip_runtime(
            &paths,
            "vllm-source-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0",
            2,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let error = resolve_engine_install_runtime_id(&paths, &config, "pytorch", None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple installed runtimes"));
        assert!(error.contains("rocm runtimes activate <runtime_key>"));

        let selection = resolve_engine_selection(&config, "pytorch", None, None);
        let error = validate_engine_selection_runtime(&paths, selection)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple installed runtimes"));

        let selection =
            resolve_engine_selection(&config, "pytorch", Some("release-pip-gfx120x-all"), None);
        let selection = validate_engine_selection_runtime(&paths, selection)?;
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all")
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_runtimes_text_reports_missing_configured_active_runtime() -> Result<()> {
        let (root, paths) = test_paths("runtime-active-missing");
        let config = RocmCliConfig {
            active_runtime_key: Some("missing-runtime-key".to_owned()),
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_runtimes_text(&paths, &config)?;

        assert!(rendered.contains("active_runtime_key: missing-runtime-key"));
        assert!(rendered.contains(
            "active_status: missing manifest for active_runtime_key=missing-runtime-key"
        ));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_lists_display_build_date_from_version_string() -> Result<()> {
        let (root, paths) = test_paths("runtime-build-date-display");
        let runtime_key = "release-pip-gfx120x-all-7-14-0a20260601";
        let manifest = write_test_pip_runtime(
            &paths,
            runtime_key,
            "therock-release:gfx120X-all",
            "7.14.0a20260601",
            20,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            default_runtime_id: Some(manifest.runtime_id),
            ..RocmCliConfig::default()
        };

        let runtimes = render_runtimes_text(&paths, &config)?;
        assert!(runtimes.contains("version=7.14.0a20260601 (build 2026-06-01)"));

        let mut examine = String::new();
        append_examine_runtime_state(&mut examine, &paths, &config)?;
        assert!(examine.contains("active_runtime_version: 7.14.0a20260601 (build 2026-06-01)"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_activation_records_exact_key_and_rollback() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-12-0",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;

        let mut config = RocmCliConfig::default();
        let first = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-12-0")?;
        assert_eq!(first.previous_runtime_key, None);
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let second = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-13-0")?;
        assert_eq!(
            second.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let marker: ActiveRuntimeMarker =
            serde_json::from_slice(&fs::read(active_runtime_marker_path(&paths))?)?;
        assert_eq!(marker.runtime_key, "release-pip-gfx120x-all-7-13-0");
        assert_eq!(
            marker.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("* release-pip-gfx120x-all-7-13-0"));
        assert!(rendered.contains("- release-pip-gfx120x-all-7-12-0"));
        assert!(rendered.contains("status=ready"));

        let rolled_back = rollback_runtime(&paths, &mut config)?;
        assert_eq!(rolled_back.runtime_key, "release-pip-gfx120x-all-7-12-0");
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_activation_rejects_ambiguous_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("runtime-ambiguous");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-12-0",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();

        let error = activate_runtime(&paths, &mut config, "therock-release:gfx120X-all")
            .unwrap_err()
            .to_string();

        assert!(error.contains("matches multiple installed runtimes"));
        assert!(error.contains("release-pip-gfx120x-all-7-12-0"));
        assert!(error.contains("release-pip-gfx120x-all-7-13-0"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_activation_rejects_unusable_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-unusable");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        fs::remove_file(manifest.install_root.join("Scripts").join("python.exe")).ok();
        fs::remove_file(manifest.install_root.join("bin").join("python")).ok();
        let mut config = RocmCliConfig::default();

        let error = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-13-0")
            .unwrap_err()
            .to_string();

        assert!(error.contains("runtime Python executable is missing"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_import_records_read_only_manifest_without_mutating_runtime_root() -> Result<()> {
        let (root, paths) = test_paths("runtime-import");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;
        fs::remove_file(runtime_manifest_path(&paths, &manifest.runtime_key))?;
        fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;

        let imported = import_runtime_manifest(&paths, &exported_manifest, false)?;

        assert!(imported.read_only);
        let canonical_export = exported_manifest.canonicalize()?;
        assert_eq!(
            imported.imported_from.as_deref(),
            Some(canonical_export.as_path())
        );
        assert!(
            !manifest
                .install_root
                .join(".rocm-cli-runtime.json")
                .exists()
        );

        let imported_registry: therock::InstalledRuntimeManifest = serde_json::from_slice(
            &fs::read(runtime_manifest_path(&paths, &manifest.runtime_key))?,
        )?;
        assert!(imported_registry.read_only);

        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &manifest.runtime_key)?;
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("external-pip-gfx120x-all-7-13-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("mode=read-only"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_import_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-import-replace");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;

        let error = import_runtime_manifest(&paths, &exported_manifest, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("already exists"));
        import_runtime_manifest(&paths, &exported_manifest, true)?;

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_records_read_only_manifest_from_probe() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root
            .join("Lib")
            .join("site-packages")
            .join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root.clone(),
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                runtime_roots: vec![sdk_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_bin],
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert!(adopted.read_only);
        assert_eq!(adopted.channel, "release");
        assert_eq!(adopted.family, "gfx120X-all");
        assert_eq!(adopted.version, "7.13.0");
        assert_eq!(
            adopted.imported_from.as_deref(),
            Some(external_root.canonicalize()?.as_path())
        );
        assert!(!external_root.join(".rocm-cli-runtime.json").exists());
        assert!(runtime_manifest_path(&paths, &adopted.runtime_key).is_file());

        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &adopted.runtime_key)?;
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("adopted-release-pip-gfx120x-all-7-13-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("* adopted-release-pip-gfx120x-all-7-13-0"));
        assert!(rendered.contains("mode=read-only"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_infers_ids_from_probe() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable.clone(),
            Some(external_root.clone()),
            None,
            None,
            None,
            false,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.13.0a20260423".to_owned()),
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.python_executable, python_executable);
        assert_eq!(request.install_root, external_root);
        assert_eq!(request.runtime_id, "therock-release:gfx120X-all");
        assert_eq!(
            request.runtime_key,
            "adopted-release-pip-gfx120x-all-7-13-0a20260423"
        );
        assert!(!request.replace);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_honors_nightly_channel() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer-nightly");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable,
            Some(external_root),
            None,
            None,
            Some("nightly".to_owned()),
            true,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.14.0a20260531".to_owned()),
                default_target_family: Some("gfx1151".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.runtime_id, "therock-nightly:gfx1151");
        assert_eq!(
            request.runtime_key,
            "adopted-nightly-pip-gfx1151-7-14-0a20260531"
        );
        assert!(request.replace);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_removes_managed_root_and_clears_active_state() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-managed");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &manifest.runtime_key)?;
        config.setup.completed = true;
        config.setup.therock_venv = Some(manifest.install_root.clone());
        config.save(&paths)?;

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(removed.runtime_key, manifest.runtime_key);
        assert!(removed.was_active);
        assert_eq!(
            removed.removed_install_root.as_deref(),
            Some(manifest.install_root.as_path())
        );
        assert!(!manifest.install_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());
        assert!(!active_runtime_marker_path(&paths).exists());
        assert_eq!(config.active_runtime_key, None);
        assert_eq!(config.default_runtime_id, None);
        assert_eq!(config.setup.therock_venv, None);
        assert!(!config.setup.completed);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_unregisters_read_only_runtime_without_deleting_external_root() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-uninstall-read-only");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;
        fs::remove_file(runtime_manifest_path(&paths, &manifest.runtime_key))?;
        fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;
        import_runtime_manifest(&paths, &exported_manifest, false)?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(removed.runtime_key, manifest.runtime_key);
        assert!(removed.read_only);
        assert_eq!(removed.removed_install_root, None);
        assert!(manifest.install_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_removes_rocm_cli_managed_custom_prefix_root() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-prefix");
        let mut manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let prefix_root = root.join("user-chosen-prefix");
        fs::rename(&manifest.install_root, &prefix_root)?;
        manifest.install_root = prefix_root.clone();
        let python = prefix_root
            .join(if cfg!(windows) { "Scripts" } else { "bin" })
            .join(if cfg!(windows) {
                "python.exe"
            } else {
                "python"
            });
        manifest.python_executable = Some(python.display().to_string());
        fs::write(
            runtime_manifest_path(&paths, &manifest.runtime_key),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::write(
            prefix_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(
            removed.removed_install_root.as_deref(),
            Some(prefix_root.as_path())
        );
        assert!(!prefix_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn runtime_adopt_preserves_venv_python_symlink_path() -> Result<()> {
        use std::os::unix::fs::symlink;

        let (root, paths) = test_paths("runtime-adopt-python-symlink");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join("bin");
        let system_python = root.join("python3.12");
        let python_executable = scripts_dir.join("python");
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join("libamdhip64.so");
        let hipblas = sdk_bin.join("libhipblas.so");
        fs::write(&system_python, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;
        symlink(&system_python, &python_executable)?;

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable: python_executable.clone(),
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-symlink-python".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                runtime_roots: vec![sdk_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_bin],
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(
            adopted.python_executable.as_deref(),
            Some(python_executable.as_path().to_string_lossy().as_ref())
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_rejects_runtime_id_without_family() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-no-family");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::write(&python_executable, "python")?;

        let error = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root),
                bin_path: Some(sdk_bin),
                ..therock::RocmSdkPythonProbe::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("must include a TheRock family suffix"));
        assert!(!runtime_registry_dir(&paths).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-replace");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let probe = therock::RocmSdkPythonProbe {
            import_ok: true,
            rocm_sdk_version: Some("7.13.0".to_owned()),
            root_path: Some(sdk_root.clone()),
            bin_path: Some(sdk_bin.clone()),
            runtime_roots: vec![sdk_root],
            bin_paths: vec![sdk_bin.clone()],
            library_paths: vec![sdk_bin],
            resolved_libraries: vec![
                therock::RocmSdkLibraryProbe {
                    shortname: "amdhip64".to_owned(),
                    paths: vec![amdhip],
                },
                therock::RocmSdkLibraryProbe {
                    shortname: "hipblas".to_owned(),
                    paths: vec![hipblas],
                },
            ],
            ..therock::RocmSdkPythonProbe::default()
        };
        let request = AdoptRuntimeRequest {
            python_executable: python_executable.clone(),
            install_root: external_root.clone(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
            replace: false,
        };

        adopt_runtime_from_probe(&paths, request.clone(), probe.clone())?;
        let error = adopt_runtime_from_probe(&paths, request, probe.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already exists"));

        adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: true,
            },
            probe,
        )?;

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("pytorch"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("pytorch", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_prefers_primary_plugin_dir() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin-precedence");
        let primary_dir = paths.primary_engine_plugin_dir();
        let compatibility_dir = paths.data_dir.join("engines");
        fs::create_dir_all(&primary_dir)?;
        fs::create_dir_all(&compatibility_dir)?;
        let name = rocm_engine_protocol::platform_engine_plugin_binary_name("pytorch");
        let primary_path = primary_dir.join(&name);
        fs::write(&primary_path, "primary")?;
        fs::write(compatibility_dir.join(&name), "compatibility")?;

        let discovered = find_engine_plugin_binary("pytorch", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(primary_path));
        Ok(())
    }

    #[test]
    fn render_engine_inventory_text_surfaces_external_plugin_policy() {
        let (root, paths) = test_paths("engine-plugin-policy");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local model engines"));
        assert!(rendered.contains("Built-in engines are included with rocm-cli"));
        assert!(rendered.contains("ROCm GPU execution is required"));
        assert!(rendered.contains("Plugin folders:"));
        assert!(rendered.contains(&paths.primary_engine_plugin_dir().display().to_string()));
    }

    #[test]
    fn friendly_engine_detect_notes_hide_probe_and_path_noise() {
        let pytorch = friendly_engine_detect_notes(
            "pytorch",
            &[r"torch probe: cuda_available=true device_count=1; managed env detected at C:\Users\user\.rocm\engines\pytorch\envs\release; rocm_sdk: version=7.13".to_owned()],
        )
        .expect("pytorch note");
        assert_eq!(pytorch, "PyTorch is ready on your AMD GPU.");

        let lemonade = friendly_engine_detect_notes(
            "lemonade",
            &["Lemonade embeddable 10.6.0 is installed at D:/ROCm/temp/runtime; Lemonade is configured for llamacpp:rocm; no CPU fallback is used".to_owned()],
        )
        .expect("lemonade note");
        assert_eq!(lemonade, "Lemonade is ready on your AMD GPU.");

        let llama = friendly_engine_detect_notes(
            "llama.cpp",
            &["llama-server not found; TheRock HIP runtime env available: root=D:\\ROCm\\temp\\therock".to_owned()],
        )
        .expect("llama.cpp note");
        assert_eq!(llama, "llama.cpp server is not installed yet.");
        let vllm = friendly_engine_detect_notes(
            "vllm",
            &["vLLM is not installed in a Linux/WSL ROCm Python environment. Native Windows is skipped; no CPU fallback is used.".to_owned()],
        )
        .expect("vllm note");
        assert_eq!(
            vllm,
            "vLLM is not installed in a Linux/WSL ROCm Python environment."
        );
        let atom = friendly_engine_detect_note_fallback(
            "ATOM Python environment was not found; set ROCM_CLI_ATOM_PYTHON.",
        );
        assert!(atom.contains("ROCM_CLI_ATOM_PYTHON"));
        assert!(!pytorch.contains("torch probe"));
        assert!(!lemonade.contains("D:/"));
        assert!(!llama.contains("D:\\"));
    }

    #[test]
    fn missing_packaged_engine_reason_has_no_deferred_first_party_engines() {
        assert!(missing_packaged_engine_reason("atom").is_none());
        assert!(missing_packaged_engine_reason("vllm").is_none());
        assert!(missing_packaged_engine_reason("sglang").is_none());
        assert!(missing_packaged_engine_reason("pytorch").is_none());
        assert!(missing_packaged_engine_reason("llama.cpp").is_none());
    }

    #[test]
    fn render_automations_text_marks_gpu_metrics_read_only() -> Result<()> {
        let (root, paths) = test_paths("automation-status-policy");
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        config.automations.daemon_enabled = true;

        let rendered = render_automations_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("GPU status checks (on)"));
        assert!(rendered.contains("setting: ask before changes; keep actions limited"));
        assert!(rendered.contains("listens for: local GPU status updates"));
        assert!(rendered.contains("does: records local GPU status only"));
        assert!(rendered.contains("policy: read-only telemetry"));
        assert!(rendered.contains("do not create review requests or mutate services"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("gpu-metrics"));
        Ok(())
    }

    #[test]
    fn render_automations_text_marks_gpu_thermal_protect_review_gated() -> Result<()> {
        let (root, paths) = test_paths("automation-gpu-thermal-policy");
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-thermal-protect");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        config.automations.daemon_enabled = true;

        let rendered = render_automations_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("GPU pressure protection (on)"));
        assert!(rendered.contains("setting: ask before changes; keep actions limited"));
        assert!(rendered.contains("listens for: high GPU temperature or memory pressure"));
        assert!(rendered.contains("does: asks before stopping a managed server"));
        assert!(rendered.contains("policy: GPU pressure protection is review-gated"));
        assert!(rendered.contains("never stop servers automatically"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("gpu-thermal-protect"));
        Ok(())
    }

    #[test]
    fn render_automations_text_surfaces_local_webhook_endpoint() -> Result<()> {
        let (root, paths) = test_paths("automation-local-webhook");
        AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 123,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 2,
            local_webhook_endpoint: Some("http://127.0.0.1:19191/automation-events".to_owned()),
            active_watchers: vec![rocm_core::WatcherRuntimeSnapshot {
                id: "gpu-metrics".to_owned(),
                enabled: true,
                mode: WatcherMode::Observe,
                summary: "record metrics".to_owned(),
                last_event: Some("record_gpu_metrics".to_owned()),
                last_event_unix_ms: Some(2),
            }],
        }
        .write(&paths)?;

        let rendered = render_automations_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("background service: running"));
        assert!(rendered.contains("local event intake: http://127.0.0.1:19191/automation-events"));
        assert!(rendered.contains("last check: GPU status check recorded"));
        assert!(!rendered.contains("pid=123"));
        assert!(!rendered.contains("last_tick_unix_ms"));
        assert!(!rendered.contains("local_webhook_endpoint"));
        assert!(!rendered.contains("record_gpu_metrics"));
        assert!(!rendered.contains("gpu-metrics"));
        Ok(())
    }

    #[test]
    fn sidebar_omits_saved_failed_service_history() -> Result<()> {
        let (root, paths) = test_paths("sidebar-service-counts");
        for index in 0..6 {
            let mut record = ManagedServiceRecord::new(
                &paths,
                format!("svc-failed-{index}"),
                "pytorch",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                11435 + index,
                "managed",
                1000 + u32::from(index),
                None,
                None,
                None,
            );
            record.status = "failed".to_owned();
            record.write()?;
        }

        let rendered = render_sidebar_text(&paths, &RocmCliConfig::default(), "local", true);
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local servers: none ready"));
        assert!(!rendered.contains("past attempts"));
        assert!(!rendered.contains("running servers: 6"));
        Ok(())
    }

    #[test]
    fn render_automations_text_uses_plain_proposal_history() -> Result<()> {
        let (root, paths) = test_paths("automation-plain-proposal-history");
        rocm_core::append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 42,
                proposal_id: "proposal-plain".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "queue restart proposal".to_owned(),
                message: "backend-authored recovery message".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-plain".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-plain" }),
                reviewed_at_unix_ms: None,
            },
        )?;
        rocm_core::append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 43,
                proposal_id: "proposal-file".to_owned(),
                watcher_id: "cache-warm".to_owned(),
                action: "queue_prefetch_proposal".to_owned(),
                title: "queue prefetch proposal".to_owned(),
                message: "cache warm requested internal artifact prefetch".to_owned(),
                status: "pending".to_owned(),
                service_id: None,
                tool: Some("prefetch_artifact".to_owned()),
                arguments: serde_json::json!({
                    "artifact_ref": "tiny/model#gguf",
                    "allow_artifact_download": true,
                    "artifact_max_bytes": 1_048_576
                }),
                reviewed_at_unix_ms: None,
            },
        )?;
        rocm_core::append_automation_event(
            &paths,
            &rocm_core::AutomationEventRecord {
                at_unix_ms: 44,
                watcher_id: "gpu-metrics".to_owned(),
                level: "info".to_owned(),
                action: "record_gpu_metrics".to_owned(),
                message: "raw amd-smi telemetry event".to_owned(),
                service_id: None,
            },
        )?;
        rocm_core::append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 45,
                source: "test".to_owned(),
                category: "proposal".to_owned(),
                actor: "watcher:server-recover".to_owned(),
                level: "info".to_owned(),
                action: "proposal_approved".to_owned(),
                message: "approved proposal-plain with backend detail".to_owned(),
                watcher_id: Some("server-recover".to_owned()),
                service_id: Some("svc-plain".to_owned()),
            },
        )?;

        let rendered = render_automations_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("recent automation activity:"));
        assert!(rendered.contains("GPU status check recorded."));
        assert!(rendered.contains("recent review requests:"));
        assert!(rendered.contains("proposal-plain [waiting for review] Restart a model server"));
        assert!(rendered.contains("why: A managed server looks stopped or unhealthy."));
        assert!(rendered.contains("server: svc-plain"));
        assert!(rendered.contains("proposal-file [waiting for review] Prepare a model file"));
        assert!(rendered.contains("model file: tiny/model#gguf"));
        assert!(rendered.contains("download: approved up to 1.0 MB"));
        assert!(rendered.contains(
            "controls: /automations approve proposal-plain | /automations reject proposal-plain"
        ));
        assert!(rendered.contains("recent background activity:"));
        assert!(rendered.contains("A review request changed status."));
        assert!(!rendered.contains("queue_restart_proposal"));
        assert!(!rendered.contains("restart_server"));
        assert!(!rendered.contains("queue restart proposal"));
        assert!(!rendered.contains("backend-authored recovery message"));
        assert!(!rendered.contains("record_gpu_metrics"));
        assert!(!rendered.contains("raw amd-smi telemetry event"));
        assert!(!rendered.contains("proposal_approved"));
        assert!(!rendered.contains("approved proposal-plain with backend detail"));
        assert!(!rendered.contains("created_unix_ms"));
        assert!(!rendered.contains("last_tick_unix_ms"));
        assert!(!rendered.contains("local_webhook_endpoint"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("1048576 bytes"));
        assert!(!rendered.contains("recent events:"));
        assert!(!rendered.contains("recent audit:"));
        assert!(!rendered.contains("tool:"));
        Ok(())
    }

    #[test]
    fn linux_only_engine_runtime_status_is_explicit_on_native_windows() {
        let detect = DetectResponse {
            installed: false,
            env_id: None,
            runtime_kind: Some("external_sglang".to_owned()),
            runtime_executable: None,
            managed_env: Some(false),
            python_version: None,
            torch_version: None,
            transformers_version: None,
            available_devices: vec![rocm_engine_protocol::EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: false,
                reason: Some(
                    "SGLang ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows SGLang is skipped. No CPU fallback is used."
                        .to_owned(),
                ),
            }],
            capabilities: rocm_engine_protocol::EngineCapabilities {
                cpu: false,
                rocm_gpu: false,
                multi_gpu: false,
                openai_compatible: true,
                tool_calling: false,
                quantized_models: "sglang-supported".to_owned(),
                distributed_serving: false,
                reasoning_parser: false,
            },
            notes: Vec::new(),
        };

        if cfg!(windows) {
            assert_eq!(
                engine_runtime_status_label("sglang", &detect),
                "unsupported_native_windows"
            );
            let mut atom_detect = detect;
            atom_detect.runtime_kind = Some("external_atom".to_owned());
            atom_detect.available_devices[0].reason = Some(
                "ATOM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows ATOM is not enabled. No CPU fallback is used."
                    .to_owned(),
            );
            assert_eq!(
                engine_runtime_status_label("atom", &atom_detect),
                "unsupported_native_windows"
            );
            assert!(
                model_registry_adapter_availability_note("atom")
                    .is_some_and(|note| note.contains("unsupported_native_windows"))
            );
        } else {
            assert_eq!(engine_runtime_status_label("sglang", &detect), "not found");
            assert!(model_registry_adapter_availability_note("atom").is_none());
        }
    }

    #[test]
    fn record_cli_audit_event_writes_cli_lifecycle_record() -> Result<()> {
        let (root, paths) = test_paths("cli-audit");

        record_cli_audit_event(
            &paths,
            "runtime",
            "runtime_activate",
            "info",
            "activated test runtime",
            None,
        );

        let events = load_recent_audit_events(&paths, 1)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, "rocm");
        assert_eq!(events[0].actor, "cli");
        assert_eq!(events[0].category, "runtime");
        assert_eq!(events[0].action, "runtime_activate");
        assert_eq!(events[0].message, "activated test runtime");
        let lifecycle_log = fs::read_to_string(cli_lifecycle_log_path(&paths))?;
        assert!(lifecycle_log.contains("category=runtime"));
        assert!(lifecycle_log.contains("action=runtime_activate"));
        assert!(lifecycle_log.contains("message=activated test runtime"));
        let action_log =
            fs::read_to_string(cli_action_log_path(&paths, "runtime", "runtime_activate"))?;
        assert_eq!(lifecycle_log, action_log);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn cli_lifecycle_log_sanitizes_components_and_newlines() -> Result<()> {
        let (root, paths) = test_paths("cli-lifecycle-sanitize");
        let event = AuditEventRecord {
            at_unix_ms: 42,
            source: "rocm".to_owned(),
            category: "Run Time".to_owned(),
            actor: "cli".to_owned(),
            level: "info".to_owned(),
            action: "install/sdk".to_owned(),
            message: "line one\nline two".to_owned(),
            watcher_id: None,
            service_id: Some("svc-1".to_owned()),
        };

        append_cli_lifecycle_logs(&paths, &event)?;

        let line = fs::read_to_string(cli_lifecycle_log_path(&paths))?;
        assert!(line.contains("service_id=svc-1"));
        assert!(line.contains("message=line one line two"));
        assert!(
            cli_action_log_path(&paths, "Run Time", "install/sdk")
                .ends_with("run-time-install-sdk.log")
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_model_registry_text_lists_builtin_recipes() {
        let rendered = render_model_registry_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("Local models"));
        assert!(rendered.contains("qwen3.5"));
        assert!(rendered.contains("GPU fit unknown"));
        assert!(rendered.contains("Use `rocm serve <model>`"));
        assert!(rendered.contains("rocm model --verbose"));
        assert!(!rendered.contains("recommended_system_ram:"));
        assert!(!rendered.contains("engine_support:"));
        assert!(!rendered.contains("artifact_check:"));
    }

    #[test]
    fn render_model_registry_verbose_text_keeps_diagnostics() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("model recipes"));
        assert!(rendered.contains("aliases=[qwen"));
        assert!(rendered.contains("recommended_system_ram: 16 GiB"));
        assert!(rendered.contains("system_ram_fit: unknown"));
        assert!(rendered.contains("gpu_fit: unknown"));
        assert!(rendered.contains("engine_support:"));
        assert!(rendered.contains("engine_action: use /engine install <engine>"));
        assert!(rendered.contains("source: built-in recipe registry"));
    }

    #[test]
    fn model_recipe_artifact_lines_surface_signed_index_metadata() -> Result<()> {
        let (root, paths) = test_paths("model-artifact-cache-lines");
        let mut recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");
        recipe.artifacts = vec![rocm_core::ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("b".repeat(64)),
            size_bytes: Some(2 * 1024 * 1024 * 1024),
            license: Some("apache-2.0".to_owned()),
            gated: Some(false),
            quantization: Some("bfloat16".to_owned()),
            engines: vec!["pytorch".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "huggingface_public".to_owned(),
                required_hosts: vec!["huggingface.co".to_owned()],
                notes: vec!["test metadata only".to_owned()],
            }),
        }];
        recipe.engine_recipes = vec![rocm_core::ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: BTreeMap::from([("reasoning_parser".to_owned(), "qwen3".to_owned())]),
            preferred_endpoint: Some(rocm_core::ModelRecipeEndpointRecord {
                endpoint_mode: "openai".to_owned(),
                settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
            }),
            unsupported_combinations: vec![rocm_core::ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
            }],
            notes: vec!["metadata only; not applied to launches yet".to_owned()],
        }];
        let mut output = String::new();

        append_model_recipe_metadata_lines(&mut output, &recipe, Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(output.contains("artifact_check: metadata_available"));
        assert!(output.contains("artifact_count: 1"));
        assert!(output.contains("artifact hf-main kind=huggingface"));
        assert!(output.contains("download rule: Public Hugging Face download"));
        assert!(output.contains("allowed site: huggingface.co"));
        assert!(output.contains("note: test metadata only"));
        assert!(!output.contains("source_policy=huggingface_public"));
        assert!(output.contains("size=2.0 GiB"));
        assert!(output.contains("engines=[pytorch]"));
        assert!(output.contains("artifact_cache hf-main status=missing"));
        assert!(output.contains("prefetch requires an approved source policy"));
        assert!(output.contains(
            "engine_recipes_policy: protocol_contract=0.1.0 selected-engine hint is passed to adapters during model resolution and required flags are forwarded at launch"
        ));
        assert!(output.contains("engine_recipe vllm required_flags=[--enable-auto-tool-choice]"));
        assert!(output.contains("parser_settings=[reasoning_parser=qwen3]"));
        assert!(output.contains("preferred_endpoint=mode=openai settings=[streaming=true]"));
        assert!(output.contains(
            "unsupported_combinations=[native Windows GPU serving (vLLM ROCm serving is Linux/WSL only)]"
        ));
        Ok(())
    }

    #[test]
    fn render_model_registry_text_reports_host_ram_fit() {
        let rendered =
            render_model_registry_verbose_text_with_context_and_host(None, None, Some(32.0));

        assert!(rendered.contains("system_ram_policy: advisory"));
        assert!(rendered.contains("system_ram_fit: supported"));
        assert!(rendered.contains("host RAM 32 GiB meets recipe recommendation 16 GiB"));
        assert!(rendered.contains("system_ram_fit: below_recommendation"));
        assert!(rendered.contains("host RAM 32 GiB is below recipe recommendation 64 GiB"));
        assert!(rendered.contains("host with at least 64 GiB system RAM for smoother serving"));
    }

    #[test]
    fn render_model_registry_text_reports_supported_and_unsupported_gpu_fit() {
        let rendered =
            render_model_registry_verbose_text_with_context_and_host(None, Some(16.0), None);

        assert!(rendered.contains("gpu_fit: supported"));
        assert!(rendered.contains("aggregate GPU VRAM 16 GiB meets recipe minimum 12 GiB"));
        assert!(rendered.contains("gpu_fit: unsupported"));
        assert!(rendered.contains("aggregate GPU VRAM 16 GiB is below recipe minimum 48 GiB"));
        assert!(rendered.contains("choose a recipe with min_gpu_mem <= 16 GiB"));
        assert!(rendered.contains("or use a GPU with at least 48 GiB before serving"));
        assert!(rendered.contains("manual_alternatives: qwen3.5-4b (12 GiB min GPU)"));
        assert!(rendered.contains(
            "manual_alternative_policy: user must choose one explicitly; none is selected automatically"
        ));
        assert!(rendered.contains("llama-3.2-3b-instruct (8 GiB min GPU)"));
        assert!(rendered.contains("tiny-gpt2 (2 GiB min GPU)"));
    }

    #[test]
    fn render_model_registry_text_marks_tiny_recipe_as_gpu_smoke_support() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("sshleifer/tiny-gpt2"));
        assert!(rendered.contains("device=gpu_required"));
        assert!(rendered.contains("min_gpu_mem=2 GiB"));
        assert!(!rendered.contains("recipe device policy `cpu_only`"));
    }

    #[test]
    fn model_registry_marks_windows_vllm_adapter_as_runtime_unsupported() -> Result<()> {
        let (root, paths) = test_paths("model-vllm-support");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "vllm")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&plugin_path, fs::Permissions::from_mode(0o755))?;
        }
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");
        let mut output = String::new();

        append_model_engine_support_lines(&mut output, &recipe, Some(&paths));
        let _ = fs::remove_dir_all(root);

        if cfg!(windows) {
            assert!(output.contains("vllm: adapter_available"), "{output}");
            assert!(
                output.contains("runtime_status=unsupported_native_windows"),
                "{output}"
            );
            assert!(output.contains("gpu_execution_required=true"), "{output}");
            assert!(!output.contains("CPU fallback"), "{output}");
        } else {
            assert!(output.contains("vllm: built_in"), "{output}");
        }
        Ok(())
    }

    #[test]
    fn model_registry_engine_reasons_do_not_mention_cpu_fallbacks() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(!rendered.contains("CPU fallback"));
        assert!(rendered.contains("engine_action: use /engine install <engine>"));
    }

    #[test]
    fn examine_runtime_state_reports_active_runtime_key_and_status() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-state");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key),
            previous_runtime_key: Some("older-release".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("runtime_state:"));
        assert!(output.contains("active_runtime_id: therock-release:gfx120X-all"));
        assert!(output.contains("active_runtime_key: release-pip-gfx120x-all"));
        assert!(output.contains("previous_runtime_key: older-release"));
        assert!(output.contains("active_runtime_status: ready"));
        assert!(output.contains("active_runtime_family: gfx120X-all"));
        assert!(output.contains("registered_runtime_keys: release-pip-gfx120x-all"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_runtime_state_reports_ambiguous_default_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-ambiguous");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        write_test_pip_runtime(
            &paths,
            "vllm-source-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            2,
        )?;
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ambiguous_runtime_id"));
        assert!(output.contains("active_runtime_matches:"));
        assert!(output.contains("release-pip-gfx120x-all"));
        assert!(output.contains("vllm-source-pip-gfx120x-all"));
        assert!(output.contains("active_runtime_action: rocm runtimes activate <runtime_key>"));
        assert!(!output.contains("active_runtime_status: missing_manifest"));

        config.active_runtime_key = Some("release-pip-gfx120x-all".to_owned());
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ready"));
        assert!(!output.contains("ambiguous_runtime_id"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_engine_inventory_reports_config_without_engine_detect() {
        let (root, paths) = test_paths("examine-engine-inventory");
        let mut config = RocmCliConfig {
            default_engine: Some("llama.cpp".to_owned()),
            ..RocmCliConfig::default()
        };
        config.engine_config_mut("llama.cpp").preferred_runtime_id =
            Some("therock-release:gfx120X-all".to_owned());
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config);

        assert!(output.contains("engine_inventory:"));
        assert!(output.contains("configured_default_engine: llama.cpp"));
        assert!(output.contains("effective_default_engine: llama.cpp"));
        assert!(output.contains("* llama.cpp"));
        assert!(output.contains("runtime_pref=therock-release:gfx120X-all"));
        assert!(output.contains("plugin_dirs:"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn runtime_update_source_uses_active_runtime_and_requires_selector_when_ambiguous() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-update-source");
        let first = write_test_pip_runtime(
            &paths,
            "release-pip-gfx110x-all",
            "therock-release:gfx110X-all",
            "7.13.0a20260416",
            1,
        )?;
        let second = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            2,
        )?;
        let manifests = therock::load_runtime_manifests(&paths)?;
        let active_config = RocmCliConfig {
            default_runtime_id: Some(second.runtime_id.clone()),
            active_runtime_key: Some(second.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        let selected = select_runtime_update_source(&manifests, &active_config, None)?;
        assert_eq!(selected.runtime_key, second.runtime_key);

        let explicit = select_runtime_update_source(
            &manifests,
            &RocmCliConfig::default(),
            Some(&first.runtime_key),
        )?;
        assert_eq!(explicit.runtime_key, first.runtime_key);

        let error = select_runtime_update_source(&manifests, &RocmCliConfig::default(), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple runtimes"));
        assert!(error.contains("--runtime <runtime-key>"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn installed_update_runtime_matches_latest_version_and_family() {
        let mut source = test_runtime_manifest_for_update(
            "old-gfx120",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.13.0a20260416",
        );
        source.channel = "release".to_owned();
        let wrong_family = test_runtime_manifest_for_update(
            "new-gfx110",
            "therock-release:gfx110X-all",
            "gfx110X-all",
            "7.14.0a20260531",
        );
        let target = test_runtime_manifest_for_update(
            "new-gfx120",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.14.0a20260531",
        );
        let manifests = vec![wrong_family, target.clone()];

        let selected = select_installed_update_runtime(&manifests, &source, "7.14.0a20260531")
            .expect("matching updated runtime should be selected");

        assert_eq!(selected.runtime_key, target.runtime_key);
    }

    fn write_test_pip_runtime(
        paths: &AppPaths,
        runtime_key: &str,
        runtime_id: &str,
        version: &str,
        installed_at_unix_ms: u128,
    ) -> Result<therock::InstalledRuntimeManifest> {
        let install_root = paths
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join(runtime_key);
        let scripts_dir = install_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = install_root.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let manifest = therock::InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: runtime_id.to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "test".to_owned(),
            version: version.to_owned(),
            install_root: install_root.clone(),
            selected_artifact_url: "https://example.invalid/therock".to_owned(),
            index_url: Some("https://example.invalid/therock".to_owned()),
            tarball_file_name: None,
            python_launcher: Some("python".to_owned()),
            python_executable: Some(python_executable.display().to_string()),
            pip_cache_dir: Some(paths.cache_dir.join("uv").join("therock")),
            rocm_sdk: Some(therock::RocmSdkPythonProbe {
                import_ok: true,
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                runtime_roots: vec![sdk_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_bin],
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                ..therock::RocmSdkPythonProbe::default()
            }),
            read_only: false,
            imported_from: None,
            installed_at_unix_ms,
        };
        fs::create_dir_all(runtime_registry_dir(paths))?;
        fs::write(
            runtime_manifest_path(paths, runtime_key),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::write(
            install_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(manifest)
    }

    fn test_runtime_manifest_for_update(
        runtime_key: &str,
        runtime_id: &str,
        family: &str,
        version: &str,
    ) -> therock::InstalledRuntimeManifest {
        therock::InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: runtime_id.to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: family.to_owned(),
            family_source: "test".to_owned(),
            version: version.to_owned(),
            install_root: PathBuf::from("runtime-root"),
            selected_artifact_url: "https://example.invalid/therock".to_owned(),
            index_url: Some("https://example.invalid/therock".to_owned()),
            tarball_file_name: None,
            python_launcher: Some("python".to_owned()),
            python_executable: Some("python".to_owned()),
            pip_cache_dir: None,
            rocm_sdk: None,
            read_only: false,
            imported_from: None,
            installed_at_unix_ms: 1,
        }
    }

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("main")
            .join(format!(
                "rocm-cli-main-test-{name}-{}-{}",
                std::process::id(),
                rocm_core::unix_time_millis()
            ));
        let _ = fs::remove_dir_all(&root);
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }
}
