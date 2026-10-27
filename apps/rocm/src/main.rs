// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod advised_commands;
mod api_docs_link;
/// Shared archive generator and filesystem oracle for the extraction property
/// tests in `comfyui` and `therock`.
#[cfg(all(test, unix))]
#[path = "../../../crates/rocm-core/src/archive_props.rs"]
mod archive_props;
mod automations;
mod bootstrap;
mod chat;
mod chat_host_facts;
mod cli_progress;
mod cli_report;
mod comfyui;
mod dash;
mod dash_seam;
// Unix-only: its generators plant symlinks with `std::os::unix::fs::symlink`.
#[cfg(all(test, unix))]
mod deletion_properties;
mod driver_install;
mod endpoint_keys;
mod engines_cmd;
mod freeform_planner;
mod logging;
mod provider_keys;
mod providers;
mod remote;
mod runtime_install_finalize;
mod runtime_services;
mod runtimes_cmd;
mod serve_cmd;
mod serve_summary;
mod service_hints;
mod service_launch;
mod storage;
#[cfg(test)]
mod test_support;
mod therock;
mod uninstall;

// Per-command handlers mechanically relocated into modules — fns, plus the
// types (e.g. `ServeArgs`, driver-plan types) that moved with their cluster.
// Dispatch call sites stay byte-identical via these re-imports (upstream-sync
// mergeability); several helpers are re-imported for reasons other than
// dispatch: `select_serve_engine` because `assess_model_for_host` calls it
// directly, `env_root_for_service`, `env_root_for_engine_install`,
// `runtime_manifest_for_selector`, `engine_manages_own_runtime`, and
// `runtime_key_for_python` because other root-level commands outside
// `dispatch()` call them directly, `attach_background_stdio`,
// `detach_background_command`, `managed_engine_startup_failure_detail`, and
// `managed_service_process_command` because `restart_internal_managed_service`
// (also outside `dispatch()`) calls them, and
// `ensure_public_service_has_endpoint_key` because the endpoint-key check
// below calls it directly. `install()`'s SDK
// path calls into `runtime_install_finalize` directly (not via `dispatch()`),
// and `apply_runtime_update`/`append_update_activate_summary` and several
// root-level chat/examine/update helpers call into `runtimes_cmd` directly for
// the same reason `engines_cmd`'s runtime-resolution helpers are re-imported.
use crate::automations::automations;
use crate::driver_install::{install_driver, reconcile_driver_install};
use crate::engines_cmd::{
    engine_manages_own_runtime, engines, env_root_for_service, runtime_manifest_for_selector,
};
use crate::runtime_install_finalize::{
    SdkInstallConsents, ensure_libatomic_for_torch, ensure_libnuma_for_torch,
    finalize_successful_sdk_install, finish_sdk_install, maybe_auto_install_sdk_preferred_engine,
    print_sdk_install_success,
};
use crate::runtimes_cmd::{
    RuntimeActivationResult, activate_runtime, active_runtime_marker_path,
    current_runtime_manifest, default_runtime_id_matches, recover_setup_runtime_registration,
    runtime_keys_text, runtime_usability_status, runtimes, select_runtime_manifest,
    setup_runtime_paths, toolchain_state_text, validate_runtime_manifest_for_activation,
};
use crate::serve_cmd::{ServeArgs, select_serve_engine, serve};
use crate::service_launch::ensure_public_service_has_endpoint_key;
#[cfg(not(windows))]
use crate::service_launch::{
    attach_background_stdio, detach_background_command, managed_engine_startup_failure_detail,
    managed_service_process_command,
};
use crate::uninstall::uninstall;

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use rocm_core::browser::Opener;
use rocm_core::model_readiness::{
    AcceleratorMemory, HostEngineChoice, HostFacts, ModelCatalogSource, ModelReadiness,
};
use rocm_core::{
    AppPaths, AuditEventRecord, AutomationEventRecord, AutomationProposalRecord,
    AutomationRuntimeState, CodexBridgeEngine, CodexBridgeGpuSnapshot, CodexBridgeSnapshot,
    DEFAULT_LOCAL_HOST, EndpointReadiness, EndpointReadinessOutcome, ExamineSummary,
    ManagedServiceRecord, ModelRecipeRecord, ModelRecipeRegistry, ModelRecipeRegistrySource,
    PERMISSIONS_MODE_ASK, PERMISSIONS_MODE_FULL_ACCESS, RocmCliConfig, TELEMETRY_MODE_LOCAL,
    TELEMETRY_MODE_OFF, WatcherMode, append_audit_event, builtin_model_recipes, builtin_watchers,
    connect_tcp_stream, daemon_binary_path, default_engine_for_platform, detect_host_gfx_target,
    detect_host_gpu_summary, engine_binary_path, engine_plugin_dirs, format_host_port,
    format_http_base_url, interactive_terminal, load_model_recipe_registry,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
    managed_pip_cache_dir, managed_service_endpoint_readiness, model_artifact_cache_status,
    model_catalog_platforms, model_recipe_featured, model_recipe_target_platform_label,
    normalize_therock_family, platform_matches_gfx_family, process_is_running,
    read_http_response_bounded, uv_cache_source, write_all_tcp_stream,
};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DetectRequest, DetectResponse, DevicePolicy,
    ENGINE_RECIPE_CONTRACT_VERSION, EngineMethod, EnginePluginDescriptor, EngineRecipeHint,
    EngineRequestEnvelope, EngineResponseEnvelope, GpuSelection, StopRequest, StopResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
#[cfg(all(test, not(windows)))]
use std::process::ExitStatus;
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime};

static BUILTIN_ENGINE_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// User-selectable serving engines, in the order shown in `--help` and shell
/// completions. Used as the clap `value_parser` for every user-facing `--engine`
/// argument so the possible values stay in sync across the CLI surface.
const SUPPORTED_ENGINES: [&str; 2] = ["lemonade", "vllm"];

#[derive(Parser, Debug)]
#[command(
    name = "rocm",
    about = "ROCm AI Command Center CLI",
    long_about = "ROCm AI Command Center CLI: install ROCm, manage local inference engines, \
and run OpenAI-compatible model servers on AMD GPUs.\n\n\
Run `rocm` with no subcommand to open the interactive launcher (TUI). Use `rocm examine` \
to check that your GPU and ROCm install are ready, then `rocm serve <model>` to start a server.",
    version,
    // The `chat` example was dropped from this list when `chat` became
    // `[preview]`: promoting a command as a headline example while marking it
    // preview elsewhere contradicts itself. `engines list` keeps the six-step
    // narrative and is in scope.
    after_help = "EXAMPLES:\n  \
rocm examine                      Check GPU, ROCm install, and engines\n  \
rocm install sdk                  Install ROCm wheels into a managed environment\n  \
rocm engines list                 Show the local inference engines available\n  \
rocm model                        List models this machine can run\n  \
rocm serve qwen                   Start a local OpenAI-compatible server\n  \
rocm services list                Show running model servers\n\n\
Commands marked [preview] work but are outside the Tech Preview's supported\n\
scope: treat them as unfinished and do not depend on their behaviour yet.\n\n\
Run `rocm <command> --help` for details and examples on any command."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Check this computer's GPU, ROCm install, engines, and setup folders.
    ///
    /// A GPU that is recognized but has no ROCm build at all - `gfx90c`, the
    /// Vega iGPU in Renoir, Lucienne, Cezanne and Barcelo APUs - is reported on
    /// a `rocm_support` line saying so, so that an empty
    /// `compatible_therock_family` does not read as a failed probe. Such a GPU
    /// still works through the Vulkan backend.
    Examine {
        /// Emit the machine-readable Examination JSON (for diagnosis tooling).
        #[arg(long)]
        json: bool,
        /// Which machine-learning framework to inspect (affects --json only).
        ///
        /// The probe imports the framework to read its ROCm build and the GPU
        /// architectures it was compiled for, which costs an interpreter start.
        /// Left alone it tries PyTorch and falls back to llama.cpp; name one to
        /// go straight there, or `skip` to leave frameworks out entirely.
        ///
        /// Only the machine-readable form reports framework state, so this has
        /// no effect on the human report.
        #[arg(long, value_enum, default_value_t = FrameworkArg::Auto)]
        framework: FrameworkArg,
    },
    /// Diagnose known ROCm/PyTorch/llama.cpp failure modes against a closed catalog.
    ///
    /// Matches this machine against a fixed catalog of known misconfigurations and
    /// ranks what it finds. It can only recognise failure modes that are in the
    /// catalog: no match means "not recognised", not "nothing is wrong" - in that
    /// case it points you at where to report the symptom.
    ///
    /// Each cause is printed with an `id:` and an `apply with:` command. The
    /// leading `#1`, `#2` are ranking positions for reading order only; `rocm fix`
    /// takes the id, not the position.
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
        /// Diagnose a WSL distribution from the Windows host instead of this
        /// machine. Needs nothing installed inside the distribution. Pass the
        /// name only when more than one is installed.
        #[arg(long, value_name = "NAME", num_args = 0..=1, default_missing_value = "")]
        distro: Option<String>,
        /// Also answer whether a model would run on this machine, before
        /// downloading it.
        ///
        /// Takes a curated model name or alias (see `rocm model`). Nothing is
        /// fetched: the answer comes from the recipe and this host's GPU. A
        /// model the catalog does not carry is reported as undetermined rather
        /// than blocked - `rocm serve` still accepts it, this just cannot say
        /// in advance whether it fits.
        ///
        /// A flag rather than a positional: `diagnose` takes no positional
        /// today, and one added now would be ambiguous against `--symptom`.
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,
        /// Show the report this machine would contribute, and send nothing.
        ///
        /// Nothing leaves the machine: this prints the exact content so it can
        /// be read before any of it is shared. Hardware that is not on AMD's
        /// published compatibility matrix produces no report at all.
        ///
        /// Not combinable with `--distro`: a report describes this machine, and
        /// a WSL distribution reached remotely is not fully examined (see
        /// `--distro`'s own help), so it cannot back the disclosure guard's
        /// architecture check.
        #[arg(long, conflicts_with = "distro")]
        report: bool,
        /// Also offer a prefilled mail, so the report can be sent.
        ///
        /// Still sends nothing. This opens a mail to ROCmCLI@amd.com with the
        /// same content `--report` printed, already filled in; it leaves your
        /// machine only when you send it yourself. On a machine with no
        /// desktop, or one reached over SSH, the link is printed instead of
        /// opened.
        ///
        /// Requires `--report`, so the content is always shown before the
        /// mail is offered. Not combinable with `--json`, which is for
        /// scripts, and a script is not a person who can read and send a mail.
        #[arg(long, requires = "report", conflicts_with = "json")]
        send: bool,
    },
    /// Apply a known fix by id (see `rocm diagnose`); run with no id to list fixes.
    ///
    /// Takes the `id:` that `rocm diagnose` reports against a cause - not the
    /// `#1`/`#2` ranking position, which belongs to one report and is not a name.
    /// With no id it lists the whole catalog.
    ///
    /// Every fix carries a marker saying what happens on the machine in front of
    /// you: AUTO means this command carries the change out; NEEDS-ARG means it
    /// will, once told what to act on; PRINT-ONLY means it prints the steps for
    /// you to run yourself (typically because they need sudo or a reboot); and
    /// DIAGNOSE-ONLY means no reliable fix exists, so running it names the
    /// problem and changes nothing (`info-20-gpu-no-rocm-support` is the entry
    /// marked that way today: a GPU ROCm has no runtime build for is a fact
    /// about the hardware, not something to put right).
    /// Use `--dry-run` to see any fix's plan without changing anything.
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
        /// Emit the catalog as JSON for tooling. Only valid without a fix id.
        #[arg(long)]
        json: bool,
    },
    /// Print the rocm-cli version, release tag or branch, and commit hash,
    /// plus the ROCm SDK and GPU driver this machine would use.
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
        #[arg(long)]
        gpu: Option<String>,
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
    /// [preview] Chat with a configured assistant provider.
    ///
    /// In a terminal with no --prompt, opens the interactive chat (switch
    /// providers there with /provider; --provider, --model, and --tools don't
    /// apply). With --prompt, or a prompt piped on standard input, sends that one
    /// prompt and prints the reply. Use `rocm config enable-provider` and
    /// `rocm config set-provider-key` first to configure a remote provider; the
    /// `local` provider talks to a running server.
    #[command(after_help = "EXAMPLES:\n  \
rocm chat --prompt \"Explain ROCm in one sentence\"\n  \
rocm chat --provider local --model qwen --prompt \"Hello\"\n  \
echo \"Summarize this\" | rocm chat --provider anthropic")]
    Chat {
        /// Assistant provider to use.
        #[arg(long)]
        provider: Option<Provider>,
        /// Model name to request from the provider, such as qwen.
        #[arg(long)]
        model: Option<String>,
        /// Prompt to send. If omitted, a terminal opens the interactive chat and
        /// piped input is read from standard input.
        #[arg(long)]
        prompt: Option<String>,
        /// Sampling temperature to send with the request (>= 0.0). Higher is more random.
        // `allow_negative_numbers` on the numeric value flags below (`--temperature`,
        // `--top-p`, `--max-tokens`) lets their space form (e.g. `--temperature -1`) reach
        // the value parser for a clear range error, instead of clap rejecting the token as
        // an unexpected argument. clap only treats a `-…` token as a number when the whole
        // remainder parses as one, so a missing value (`--temperature --top-p 0.5`) or a
        // malformed token (`--temperature -0.5abc`) still errors at parse time.
        #[arg(long, allow_negative_numbers = true, value_parser = chat::parse_non_negative_temperature)]
        temperature: Option<f32>,
        /// Nucleus sampling probability to send with the request (0.0-1.0).
        #[arg(long = "top-p", allow_negative_numbers = true, value_parser = chat::parse_unit_interval)]
        top_p: Option<f32>,
        /// Maximum number of tokens to generate in the response.
        #[arg(long = "max-tokens", allow_negative_numbers = true, value_parser = chat::parse_positive_u32)]
        max_tokens: Option<u32>,
        #[arg(
            long,
            help = "Allow an OpenAI-compatible provider to request ROCm tool calls."
        )]
        tools: bool,
        /// Use the offline chat mock when launching the interactive dash chat.
        #[arg(long)]
        chat_mock: bool,
    },
    /// Install ROCm, drivers, or related local AI components.
    Install {
        #[command(subcommand)]
        target: InstallTarget,
    },
    /// Check for a newer ROCm package and optionally install it.
    ///
    /// Without --apply or --dry-run, only reports whether an update is available.
    /// Pass --dry-run to preview what --apply would do, --apply to install it, and
    /// add --activate to make the new install the default afterward.
    #[command(after_help = "EXAMPLES:\n  \
rocm update\n  \
rocm update --dry-run\n  \
rocm update --apply --activate\n  \
rocm update --apply --dry-run\n  \
rocm update --json")]
    Update {
        /// Install the selected update instead of only checking.
        #[arg(long)]
        apply: bool,
        /// Runtime key to update.
        #[arg(long)]
        runtime: Option<String>,
        /// Use the updated ROCm install as the default after installing it.
        #[arg(long)]
        activate: bool,
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Accepted for consistency with other mutating commands; applying never prompts.
        #[arg(long)]
        yes: bool,
        /// Print the check result as a single line of JSON instead of text.
        #[arg(long, conflicts_with_all = ["apply", "dry_run"])]
        json: bool,
        /// Bound the version-check network calls to this many seconds each.
        #[arg(long, requires = "json", conflicts_with = "apply", value_parser = clap::value_parser!(u64).range(1..))]
        timeout_secs: Option<u64>,
    },
    /// List, choose, add, or remove ROCm runtimes.
    Runtimes {
        #[command(subcommand)]
        command: Option<RuntimesCommand>,
    },
    /// Show what ROCm CLI is storing on disk, and free space it no longer needs.
    Storage {
        #[command(subcommand)]
        command: Option<StorageCommand>,
    },
    /// List, install, or open shells for local model engines.
    Engines {
        #[command(subcommand)]
        command: EnginesCommand,
    },
    /// Show recommended local models and what this machine can run.
    #[command(
        alias = "models",
        after_help = "EXAMPLES:\n  \
rocm model\n  \
rocm model --verbose"
    )]
    Model {
        /// Show detailed recipe diagnostics.
        #[arg(long)]
        verbose: bool,
    },
    /// Start a local OpenAI-compatible model server.
    ///
    /// Picks an engine and ROCm runtime automatically unless overridden. By default the
    /// server runs as a managed background service and prints a deployment summary
    /// (status, endpoint, API docs link, model, and smoke-test metrics); use --verbose to
    /// stream engine logs in this terminal instead. When streaming, press Ctrl-D to detach and leave the
    /// server running, or Ctrl-C to stop it. Inspect or stop servers later with
    /// `rocm services`.
    #[command(after_help = "EXAMPLES:\n  \
rocm serve qwen\n  \
rocm serve qwen --engine vllm --port 8000\n  \
rocm serve qwen --verbose --device gpu_required")]
    Serve {
        /// Model name, alias, or local model file path.
        model: String,
        /// Engine to use.
        #[arg(long, value_parser = SUPPORTED_ENGINES)]
        engine: Option<String>,
        /// Device policy.
        #[arg(long)]
        device: Option<DevicePolicyArg>,
        /// GPU device to serve on: `auto` (default; first free GPU) or a single
        /// index like `1`.
        #[arg(long, value_name = "INDEX|auto")]
        gpu: Option<String>,
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
        /// Attach to the server in this terminal and stream its logs (Ctrl-D to
        /// detach and leave it running, Ctrl-C to stop). Same as --verbose.
        #[arg(long, conflicts_with = "managed")]
        foreground: bool,
        /// Keep the server managed by ROCm CLI.
        #[arg(long)]
        managed: bool,
        /// Stream every engine log line in this terminal instead of a deployment summary.
        #[arg(long, conflicts_with = "managed")]
        verbose: bool,
        /// Skip the post-startup inference smoke test (time-to-first-token, tokens/sec).
        #[arg(long)]
        no_smoke_test: bool,
        /// Allow binding to a non-local address.
        #[arg(long)]
        allow_public_bind: bool,
        /// Require an API key even on a loopback bind.
        ///
        /// Loopback serving is credential-free because only this machine can
        /// reach it. That stops being true when something else republishes the
        /// port - a tailnet publish, a reverse proxy, a container port map - at
        /// which point the bind address no longer describes who can call it.
        /// Pass this to keep the endpoint authenticated anyway. `rocm remote`
        /// sets it on every session it starts.
        #[arg(long)]
        require_api_key: bool,
        /// vLLM tool-call parser to enable OpenAI tool calling for this model
        /// (e.g. `hermes`, `llama3_json`, `mistral`). Overrides any catalog default
        /// and implies `--enable-auto-tool-choice`. Applies to vLLM only.
        #[arg(long, value_name = "NAME")]
        tool_call_parser: Option<String>,
        /// Fraction of each GPU's TOTAL VRAM that vLLM may claim for weights and
        /// KV cache (`0.0 < value <= 1.0`). Omitted by default, so vLLM applies
        /// its own default. Lower it when a small model should not reserve most
        /// of a large card. Applies to vLLM only.
        // Taken as a string with `allow_hyphen_values` rather than a clap `f64` so a
        // negative fraction reaches `parse_gpu_memory_utilization` and gets the domain
        // error naming the valid range, instead of clap rejecting `-0.2` as an unknown
        // flag. The cost is that a missing value swallows the next argument
        // (`--gpu-memory-utilization --verbose`), which then fails loudly on parse.
        #[arg(long, value_name = "FRACTION", allow_hyphen_values = true)]
        gpu_memory_utilization: Option<String>,
        /// Default sampling temperature for generation (>= 0.0).
        // `allow_negative_numbers` on the numeric value flags below (`--temperature`,
        // `--top-p`, `--max-tokens`) lets their space form (e.g. `--temperature -1`) reach
        // the value parser for a clear range error, instead of clap rejecting the token as
        // an unexpected argument. clap only treats a `-…` token as a number when the whole
        // remainder parses as one, so a missing value (`--temperature --top-p 0.5`) or a
        // malformed token (`--temperature -0.5abc`) still errors at parse time.
        #[arg(long, allow_negative_numbers = true, value_parser = chat::parse_non_negative_temperature)]
        temperature: Option<f32>,
        /// Default nucleus sampling probability for generation (0.0-1.0).
        #[arg(long = "top-p", allow_negative_numbers = true, value_parser = chat::parse_unit_interval)]
        top_p: Option<f32>,
        /// Default maximum number of tokens to generate per response.
        #[arg(long = "max-tokens", allow_negative_numbers = true, value_parser = chat::parse_positive_u32)]
        max_tokens: Option<u32>,
        /// API key that clients must present to a public (non-loopback) endpoint.
        /// When binding a public interface and this is omitted, a strong key is
        /// generated automatically. Prefer the `ROCM_SERVE_API_KEY` environment
        /// variable over this flag so the secret does not appear in shell history
        /// or the process table. Ignored for a loopback bind unless
        /// `--require-api-key` is also passed, which makes a loopback endpoint
        /// authenticated too.
        #[arg(long)]
        api_key: Option<String>,
    },
    /// [preview] Install, start, stop, or inspect ComfyUI.
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
    /// [preview] Work with GPU machines on your tailnet.
    Remote {
        #[command(subcommand)]
        command: remote::RemoteCommand,
    },
    /// [preview] Manage optional background checks and review requests.
    Automations {
        #[command(subcommand)]
        command: Option<AutomationsCommand>,
    },
    /// Show or change saved ROCm CLI settings.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Browse and search recent ROCm CLI logs.
    #[command(after_help = "EXAMPLES:\n  \
rocm logs\n  \
rocm logs --service <service-id>\n  \
rocm logs --search error timeout")]
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
    /// Run the background supervisor that powers automations, managed servers, and the dashboard.
    ///
    /// The daemon is a multi-role helper that, while running:
    ///   - executes enabled automation watchers (update checks, driver-plan
    ///     checks, artifact prefetch) on a 5s tick in a sandboxed subprocess
    ///   - health-checks and auto-recovers managed local model servers
    ///     (Lemonade, vLLM)
    ///   - collects GPU thermal/VRAM metrics every 60s for the TUI dashboard
    ///   - listens on a local webhook port for automation events from other
    ///     `rocm` commands
    ///   - serves API docs for running model servers on 127.0.0.1 (port
    ///     11430 by default; `api_docs` in config.json), forwarding their
    ///     "Try it out" requests to the server
    ///
    /// It is normally started on demand by `rocm automations enable` and
    /// `rocm serve --managed`, so you rarely need to launch it yourself. Run it
    /// directly only when you want to observe its behavior in the foreground,
    /// e.g. for debugging.
    #[command(verbatim_doc_comment)]
    Daemon {
        /// Print a one-shot snapshot of automation/watcher status, then exit without starting the supervisor loop.
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
    /// Saturate a local OpenAI-compatible endpoint and report rough client-side
    /// throughput (local smoke-test, NOT an official ROCm/AMD benchmark).
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    /// Remove ROCm CLI-managed files from this computer.
    ///
    /// Removes the config, data, and cache directories and the installed
    /// binaries, including the running rocm.exe on Windows. It also undoes the
    /// PATH change the installer made (its marked shell-profile block, or the
    /// user PATH entry it recorded adding on Windows), unless other programs
    /// share the install folder.
    Uninstall {
        /// Do not ask for interactive confirmation.
        #[arg(long)]
        yes: bool,
        /// Show what would be removed without deleting anything.
        #[arg(long)]
        dry_run: bool,
        /// Keep installed rocm-cli binaries, and the installer's PATH change.
        #[arg(long)]
        keep_binaries: bool,
        /// Keep saved settings.
        #[arg(long)]
        keep_config: bool,
        /// Keep app data such as logs, services, engines, and the uv package cache
        /// (often the largest directory rocm-cli manages).
        #[arg(long)]
        keep_data: bool,
        /// Keep caches under the cache directory. Does not cover the uv package cache,
        /// which lives under the data directory; use --keep-data for that.
        #[arg(long)]
        keep_cache: bool,
        /// Allow removing development binaries inside the current build tree.
        #[arg(long)]
        force_dev_binaries: bool,
    },
}

#[derive(Subcommand, Debug)]
enum BenchCommand {
    /// Run a concurrency sweep (RAW serving throughput, synthetic single-shot
    /// requests - not agent-shaped, not comparable to *-agent-bench harnesses).
    ///
    /// Measures RAW serving throughput (synthetic single-shot requests, vLLM
    /// benchmark_serving shape). It does NOT reproduce agent-shaped, multi-turn,
    /// long-context tool traffic and is not comparable to the *-agent-bench quality
    /// harnesses.
    Load {
        /// OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1 (as shown by
        /// `rocm services list`). A plain host address without /v1 also works.
        #[arg(long, value_name = "URL")]
        endpoint: String,
        /// Model name (defaults to the first model the endpoint reports)
        #[arg(long)]
        model: Option<String>,
        /// Concurrency levels to sweep, comma-separated
        #[arg(long, value_delimiter = ',', default_value = "1,8,32,64", value_parser = clap::value_parser!(u32).range(1..=128))]
        concurrency: Vec<u32>,
        /// Input sequence length in tokens
        #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(1..=32768))]
        isl: u32,
        /// Output sequence length in tokens
        #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(1..=32768))]
        osl: u32,
        /// Requests per concurrency cell
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=10000))]
        requests: u32,
        /// Output CSV file (default: ~/.rocm/bench/results.csv, the
        /// daemon-tailed path that populates the dashboard's Bench panel)
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
        /// Ramp concurrency automatically (1,2,4,8,16,32,64,128), stopping at saturation.
        ///
        /// Ignores --concurrency when set. Stops early when gen_tps plateaus
        /// or the request queue backs up.
        #[arg(long)]
        auto_ramp: bool,
    },
}

#[derive(Subcommand, Debug)]
enum InstallTarget {
    /// Install TheRock ROCm wheels into a Python folder managed by ROCm CLI.
    #[command(after_help = "EXAMPLES:\n  \
rocm install sdk\n  \
rocm install sdk --channel nightly --build-date 2025-01-15\n  \
rocm install sdk --family gfx110X-all --dry-run\n  \
rocm install sdk --devel")]
    Sdk {
        /// Package channel to install, such as release or nightly.
        #[arg(long, default_value = "release")]
        channel: String,
        /// Package format to install [possible values: wheel, tarball].
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
        ///
        /// A GPU target ROCm publishes no build for is refused rather than
        /// mapped to a near neighbour: `--family gfx90c` - the Vega iGPU in
        /// Renoir, Lucienne, Cezanne and Barcelo APUs - fails and installs
        /// nothing, instead of substituting another family. A plain `rocm
        /// install sdk` reports the same refusal when it detects such a GPU.
        #[arg(long)]
        family: Option<String>,
        /// Also install the ROCm compiler, headers, and static libraries for
        /// building GPU code. Roughly doubles the wheel download; already
        /// included by `--format tarball`.
        ///
        /// This does not add the toolchain to a runtime you already have: a
        /// runtime is identified by the packages it was installed from, so
        /// passing --devel over an existing plain install of the same version
        /// creates a SECOND side-by-side runtime and activates it. Remove the
        /// one you do not want with `rocm runtimes uninstall <runtime-key>`;
        /// `rocm runtimes list` shows which is which.
        #[arg(long)]
        devel: bool,
        /// Resolve the install plan without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Approve replacing the current active default ROCm runtime (and
        /// required system-package installs such as OpenMPI for vLLM) without
        /// prompting; required outside an interactive terminal whenever a
        /// managed runtime is already the active default - including when this
        /// install targets a different GPU family or channel, which takes over
        /// the active default just the same. An install with no active default
        /// runtime never prompts.
        #[arg(long)]
        yes: bool,
        /// Approve replacing the current active default ROCm runtime, and only
        /// that: unlike --yes it does not approve system-package installs, so it
        /// never runs sudo. ROCm CLI's own non-interactive surfaces (chat, MCP,
        /// the dashboard) pass this, because they spawn `rocm` with no terminal
        /// and so have no way to answer a sudo password prompt; a missing system
        /// package stays a warning there, as it was before. --yes implies this.
        #[arg(long)]
        approve_replacing_active_default: bool,
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
    #[command(after_help = "EXAMPLES:\n  \
rocm engines install lemonade\n  \
rocm engines install vllm --reinstall")]
    Install {
        /// Engine name.
        #[arg(value_parser = SUPPORTED_ENGINES)]
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
        /// Approve required system-package installs (such as OpenMPI for vLLM) without asking.
        #[arg(long)]
        yes: bool,
    },
    /// Open a shell with the selected engine environment activated.
    Shell {
        /// Engine name.
        #[arg(value_parser = SUPPORTED_ENGINES)]
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
    /// Show ROCm runtimes known to ROCm CLI.
    List,
    /// Use the selected ROCm runtime by default.
    Activate {
        /// Runtime key or friendly runtime selector.
        runtime: String,
        /// Restart running local servers so they use the newly active runtime.
        #[arg(long)]
        restart_services: bool,
        /// Approve restarting running local servers. Requires --restart-services.
        //
        // `requires` rather than a silent no-op: unlike `runtimes uninstall`,
        // where `--yes` skips a real prompt, activation has nothing to confirm
        // on its own, so a bare `--yes` here means the user believes they asked
        // for something they did not.
        #[arg(long, requires = "restart_services")]
        yes: bool,
    },
    /// Switch back to the previously selected ROCm runtime.
    #[command(
        after_help = "NOTE: rollback has no history - it remembers only the runtime you just \
left, so it cannot undo more than one activation."
    )]
    Rollback {
        /// Restart running local servers so they use the restored runtime.
        #[arg(long)]
        restart_services: bool,
        /// Approve restarting running local servers. Requires --restart-services.
        #[arg(long, requires = "restart_services")]
        yes: bool,
    },
    /// Remove a ROCm runtime from ROCm CLI.
    #[command(alias = "remove")]
    Uninstall {
        /// Runtime key or friendly runtime selector.
        runtime: String,
        /// Do not ask for interactive confirmation.
        #[arg(long)]
        yes: bool,
        /// Show what would be removed without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Add a ROCm runtime from a saved manifest file.
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
enum StorageCommand {
    /// Show what is using disk space.
    Report {
        /// Emit the machine-readable storage JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove older ROCm installs, keeping the most recent ones.
    #[command(name = "remove-old-installs", alias = "remove-old-runtimes")]
    RemoveOldInstalls {
        /// How many recent installs to keep for each channel, format, GPU
        /// family, and toolchain choice.
        ///
        /// "Recent" means most recently installed, not highest version, so
        /// after a deliberate downgrade the older version counts as the newer
        /// install. A `--devel` install and a plain one are separate runtimes
        /// and get separate counts, so a newer runtime-only install never
        /// evicts the toolchain. The one in use and the rollback target are
        /// always kept on top of this count, whatever it is set to.
        #[arg(long, default_value_t = storage::DEFAULT_KEEP)]
        keep: usize,
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
    },
    /// Remove downloaded files that ROCm CLI can fetch again.
    #[command(name = "remove-downloads", alias = "remove-downloaded-files")]
    RemoveDownloads {
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Do not ask for confirmation.
        #[arg(long)]
        yes: bool,
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
        /// ROCm runtime key or id to use (see `rocm runtimes list`).
        #[arg(long)]
        runtime_id: Option<String>,
        /// Replace ComfyUI's code even if it is already installed. Only what the installed
        /// ComfyUI release shipped is deleted. Models, user (saved workflows), output, input,
        /// custom_nodes, datasets and extra_model_paths.yaml are kept, and anything else of
        /// yours is left in place; an entry of yours named like code the new release adds is
        /// renamed to `<name>.rocm-cli-kept-<time>`. An install made before rocm-cli recorded
        /// what its release shipped instead replaces whatever has the names the new release
        /// ships, and lists it. Refused while the ComfyUI started by `rocm comfyui start` is
        /// running or still starting, checked before the download and again just before the code
        /// is replaced, unless rocm-cli's saved record of it cannot be read, which it then says.
        /// Over an existing install, a download that is not a ComfyUI release (no main.py) is
        /// discarded and the install is left as it was.
        #[arg(long)]
        reinstall: bool,
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Accepted for consistency with other mutating commands; installing never prompts.
        #[arg(long)]
        yes: bool,
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
        /// Accepted for consistency with other mutating commands; starting never prompts.
        #[arg(long)]
        yes: bool,
    },
    /// Stop a ROCm CLI-managed ComfyUI server.
    Stop {
        /// Accepted for consistency with other mutating commands; stopping never prompts.
        #[arg(long)]
        yes: bool,
    },
}

// The doc comments below are `--help` output, one of the four user-facing
// surfaces that describe these commands. When a command's flags, defaults,
// arguments, or observable behaviour change here, change the other three in the
// same commit too (AGENTS.md section 5): README.md, docs/testing.md, and
// docs/manual-testing.md. `--help` is the one most often left behind, because
// it is the only one of the four that is not a Markdown file.
#[derive(Subcommand, Debug)]
enum ServicesCommand {
    /// Show currently running local model servers, and count the records that
    /// are no longer running.
    List {
        /// Include failed, stopped, and old service records.
        #[arg(short, long)]
        all: bool,
        /// Emit the service records as JSON instead of a table.
        ///
        /// This is the machine-readable form `rocm remote` reads back over its
        /// control channel to discover which service a remote `rocm serve` just
        /// started, rather than scraping the human table.
        #[arg(long)]
        json: bool,
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
        /// Required; stop refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Restart a local model server.
    ///
    /// Stops the server first, and starts it again only once every process it
    /// recorded is confirmed gone. If the stop cannot confirm that, nothing is
    /// started and the stop request stays recorded on the server.
    Restart {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; restart refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Delete one local server record and its files.
    ///
    /// Only works on a record that is not running: stop the server first. The
    /// record's details file, log, engine state file, and endpoint key file are
    /// all deleted, so its log can no longer be read and it can no longer be
    /// restarted.
    Remove {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; remove refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Delete local server records that are no longer running.
    ///
    /// Running servers are always left alone. Leftover files whose record is
    /// already gone are cleaned up too.
    ///
    /// Waits for a managed launch already under way to publish its record
    /// before reading the directory, and waits as long as that launch takes.
    /// There is no timeout. On an interactive terminal it says so while it
    /// waits; piped or redirected output stays silent.
    Prune {
        /// Only remove records and files untouched for at least this many hours.
        ///
        /// Age is measured from when the record file was last written - a stop,
        /// a restart, or a status correction all count as touching it - so a
        /// server that has only just stopped keeps its log and stays
        /// restartable. Pass 0 to include everything that is not running.
        #[arg(long, default_value_t = DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS)]
        older_than_hours: u64,
        /// Remove every record that is not running, however recent.
        ///
        /// The same thing as `--older-than-hours 0`, named so it can be reached
        /// without knowing the age rule exists -- which is how most people will
        /// arrive here, after the summary tells them recent records were kept.
        #[arg(long, conflicts_with = "older_than_hours")]
        any_age: bool,
        /// Show what would be removed, without removing anything.
        #[arg(long)]
        dry_run: bool,
        /// Required unless --dry-run; prune refuses without it.
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
        #[arg(value_parser = SUPPORTED_ENGINES)]
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
        #[arg(value_parser = SUPPORTED_ENGINES)]
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
    /// Clear recorded setup completion/dismissal state.
    ///
    /// Does not by itself re-trigger onboarding in the TUI; open it manually
    /// from `rocm dash`'s Observe tab with `n`.
    Reset {
        /// Accepted for consistency with other mutating commands; resetting never prompts.
        #[arg(long)]
        yes: bool,
    },
}

/// Which framework `rocm examine` should probe.
///
/// The CLI-facing mirror of [`rocm_core::FrameworkProbe`], which cannot derive
/// `ValueEnum` itself without pulling clap into `rocm-core`. Every variant of
/// that enum existed and was reachable in the library, but no caller ever passed
/// anything but the default, so the choice could not be made from the command
/// line at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum FrameworkArg {
    /// Try PyTorch first, then llama.cpp (the default).
    Auto,
    /// Probe PyTorch only.
    Pytorch,
    /// Probe llama.cpp only.
    #[value(name = "llama-cpp", alias = "llamacpp")]
    LlamaCpp,
    /// Probe no framework at all - fastest, and enough for GPU and driver questions.
    Skip,
}

impl From<FrameworkArg> for rocm_core::FrameworkProbe {
    fn from(value: FrameworkArg) -> Self {
        match value {
            FrameworkArg::Auto => Self::Auto,
            FrameworkArg::Pytorch => Self::PyTorch,
            FrameworkArg::LlamaCpp => Self::LlamaCpp,
            FrameworkArg::Skip => Self::Skip,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InstallFormat {
    /// Python wheel packages (recommended; smaller, pip-installable).
    Wheel,
    /// Self-contained tarball archive of the ROCm SDK.
    Tarball,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Provider {
    /// A local OpenAI-compatible server started via `rocm serve` (no API key).
    Local,
    /// Anthropic Claude API (requires an API key).
    Anthropic,
    /// OpenAI API or OpenAI-compatible endpoint (requires an API key).
    Openai,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WatcherModeArg {
    /// Only report findings; never take action.
    Observe,
    /// Report findings and propose changes for you to approve.
    Propose,
    /// Apply changes automatically within a contained, reversible scope.
    Contained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TelemetryModeArg {
    /// Collect GPU telemetry on this machine only; nothing leaves the device.
    Local,
    /// Disable telemetry collection entirely.
    Off,
}

/// User-facing `rocm serve --device` policy.
///
/// Modeled as a clap `ValueEnum` so invalid input is rejected by clap's usage
/// validation (exit code 2, with the valid choices listed) rather than a generic
/// application error (exit code 1) — matching how every other enum-style CLI
/// argument behaves. The historical aliases (`auto`/`gpu` for `gpu_required`,
/// `cpu` for `cpu_only`) stay accepted for backward compatibility but are hidden
/// from the advertised list.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum DevicePolicyArg {
    /// Require a ROCm GPU; fail if none is usable (the default).
    #[value(alias = "auto", alias = "gpu")]
    GpuRequired,
    /// Accepted for compatibility; behaves identically to `gpu_required`
    /// (rocm serve has no CPU fallback path).
    GpuPreferred,
    /// Accepted but always rejected: rocm serve has no CPU fallback path, so
    /// this exits with an error. Hidden from `--help`, shell completions and the
    /// invalid-value suggestion list so it is never advertised as usable, while
    /// still parsing so the deliberate rejection message is preserved.
    #[value(alias = "cpu", hide = true)]
    CpuOnly,
}

impl DevicePolicyArg {
    /// Canonical policy string understood by [`parse_device_policy`]. Routing the
    /// parsed variant back through the string keeps `parse_device_policy` the
    /// single choke point that enforces the `cpu_only` rejection across every
    /// caller, rather than adding a second conversion path that could bypass it.
    const fn as_policy_str(self) -> &'static str {
        match self {
            Self::GpuRequired => "gpu_required",
            Self::GpuPreferred => "gpu_preferred",
            Self::CpuOnly => "cpu_only",
        }
    }
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

/// Restore the default SIGPIPE disposition on Unix.
///
/// Rust installs `SIG_IGN` for SIGPIPE at startup, so a downstream reader
/// closing the pipe early (e.g. `rocm fix … --dry-run | head`) turns the next
/// `println!` into a panic (`failed printing to stdout: Broken pipe`, exit 101)
/// instead of the conventional quiet exit. Resetting to `SIG_DFL` makes the
/// process terminate on SIGPIPE like every other Unix CLI. Socket writes are
/// unaffected: std uses `MSG_NOSIGNAL`, so the serve/daemon paths still see a
/// normal `BrokenPipe` error rather than a signal.
#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI
fn reset_sigpipe() {
    // SAFETY: installing the default handler for a signal is always safe.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn reset_sigpipe() {}

/// Run `f` with SIGPIPE temporarily ignored, restoring the previous
/// disposition afterwards.
///
/// [`reset_sigpipe`] sets `SIG_DFL` process-wide so stdout/stderr pipe writes
/// terminate the process conventionally. But a write to a spawned child's stdin
/// whose reader has already exited would then kill `rocm` with SIGPIPE before
/// the caller can turn the resulting `EPIPE` into a diagnostic. Wrap such writes
/// in this helper so the failure surfaces as an `io::Error` (`BrokenPipe`)
/// instead of a signal.
#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI
fn with_sigpipe_ignored<T>(f: impl FnOnce() -> T) -> T {
    // SAFETY: reading/installing a signal disposition is always safe.
    let previous = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    let result = f();
    // SAFETY: restoring the disposition captured above is always safe.
    unsafe {
        libc::signal(libc::SIGPIPE, previous);
    }
    result
}

#[cfg(not(unix))]
fn with_sigpipe_ignored<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// Marker error carrying `rocm fix`'s exit code back through `main()`'s
/// ordinary return path, instead of calling `std::process::exit` mid-stack
/// and skipping the `_log_guard` destructor held in `run()`.
///
/// `exit_code_for` recovers the code via `downcast_ref`, which searches the
/// whole error chain — wrapping the `fix()` call with `.context(...)` is
/// fine. What does break it is discarding this error instead of chaining it,
/// e.g. `.map_err(|e| anyhow!("fix failed: {e}"))`, which loses the
/// underlying type and silently falls through to the generic "Error: ..."
/// branch instead of the carried exit code.
#[derive(Debug)]
struct FixExitCode(i32);

impl std::fmt::Display for FixExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fix exited with code {}", self.0)
    }
}

impl std::error::Error for FixExitCode {}

/// Marker error carrying a clap usage/parse error's exit code back through
/// `main()`'s ordinary return path, for the same reason [`FixExitCode`]
/// exists: `clap::Error::exit()` calls `std::process::exit` mid-stack, which
/// would skip the `_log_guard` destructor held in `run()`. The error is
/// printed at the point it's constructed (clap knows which stream and
/// formatting a given error kind wants); this type only carries the exit
/// code onward.
#[derive(Debug)]
struct ClapExitCode(i32);

impl std::fmt::Display for ClapExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "clap exited with code {}", self.0)
    }
}

impl std::error::Error for ClapExitCode {}

/// Print a clap usage/parse error on the stream and in the format clap itself
/// chooses, then carry its exit code onward as a [`ClapExitCode`] instead of
/// calling `err.exit()`, which would `std::process::exit` mid-stack and skip
/// the `_log_guard` destructor.
fn clap_exit_code(err: clap::Error) -> anyhow::Error {
    let code = err.exit_code();
    let _ = err.print();
    ClapExitCode(code).into()
}

fn main() -> ExitCode {
    exit_code_for(run())
}

/// Maps `run()`'s result to a process exit code, unwrapping a `FixExitCode`
/// or `ClapExitCode` to its carried code and otherwise reproducing the
/// standard `Result<(), anyhow::Error>` `Termination` behavior (print the
/// error to stderr, exit 1).
fn exit_code_for(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if let Some(FixExitCode(code)) = e.downcast_ref::<FixExitCode>() {
                ExitCode::from(*code as u8)
            } else if let Some(ClapExitCode(code)) = e.downcast_ref::<ClapExitCode>() {
                ExitCode::from(*code as u8)
            } else {
                // Match the standard `Result<(), E>` `Termination` behavior exactly:
                // ignore a failed write here rather than `eprintln!`, which panics.
                // A caller with a closed stderr pipe must still see exit code 1, not
                // a panic that replaces it.
                let _ = writeln!(io::stderr(), "Error: {e:?}");
                ExitCode::FAILURE
            }
        }
    }
}

fn run() -> Result<()> {
    reset_sigpipe();

    // Held for the whole process lifetime: dropping it flushes and stops the
    // non-blocking file writer, so an early drop would silently truncate the
    // log. A failed/missing `AppPaths::discover()` degrades to no logging
    // rather than a startup failure.
    let _log_guard = AppPaths::discover()
        .ok()
        .and_then(|paths| logging::init(&paths));

    maybe_migrate_legacy_dashboard_config();
    maybe_notice_legacy_uv_cache();

    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() {
        return launch_default();
    }

    let freeform_invocation = parse_freeform_invocation(&raw_args);
    if should_treat_as_freeform(&freeform_invocation) {
        // Inputs that look like a botched command invocation (a mistyped
        // subcommand, optionally followed by flags such as `--help`) are routed
        // to clap so it can surface a friendly "did you mean"/usage error and
        // exit, instead of dumping a request plan from the natural-language
        // planner.
        if let Some(err) = command_invocation_error(&freeform_invocation.request_args) {
            return Err(clap_exit_code(err));
        }
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

    dispatch(parse_cli()?)
}

/// Build the root `rocm` command with its top-level subcommands ordered
/// alphabetically in `--help` output (EAI-7362), and `-V`/`--version` reporting
/// the same traceable string as `rocm version` (see [`cli_version_string`]).
///
/// clap assigns each subcommand an incrementing display order in declaration
/// order and renders the command list sorted by `(display_order, name)`.
/// Resetting every subcommand to the same display order collapses that first
/// sort key, so the visible list falls back to alphabetical name order.
///
/// The value used is clap's own default display order (999, i.e.
/// `Command::get_display_order`'s `unwrap_or`). clap appends its implicit `help`
/// subcommand *after* this runs and leaves it at that default, so matching the
/// default here lets `help` sort into its alphabetical position instead of being
/// pinned last. The regression test guards this if clap's default ever changes.
///
/// `display_name` only changes what `-V`/`--version` prints ahead of the
/// version string — verified against `--help`'s `Usage:` line and `completions`
/// output, both of which are generated from `Cli::command()` directly and so
/// bypass this override.
fn cli_command() -> clap::Command {
    Cli::command()
        .mut_subcommands(|sc| sc.display_order(999usize))
        .version(cli_version_string())
        .display_name("rocm-cli")
}

/// Parse process arguments through [`cli_command`] so `rocm --help` and
/// `rocm help` list subcommands alphabetically and `rocm -V`/`--version` print
/// the traceable version string. Mirrors the derived `Cli::parse()`, which
/// builds from `Cli::command()` directly and therefore cannot pick up either
/// override.
///
/// Returns a [`ClapExitCode`]-carrying error instead of calling
/// `clap::Error::exit()` directly, so `run()`'s caller can unwrap the code
/// after `_log_guard` has dropped rather than mid-stack. This covers both
/// places clap can fail here: `try_get_matches()` for an ordinary argv parse
/// error (a bad flag, `--help`, a missing required argument — the common
/// case), and `from_arg_matches()` for the derive step below it.
fn parse_cli() -> Result<Cli> {
    let matches = cli_command().try_get_matches().map_err(clap_exit_code)?;
    Cli::from_arg_matches(&matches).map_err(clap_exit_code)
}

/// What every version-reporting surface (`-V`/`--version`, `rocm version`, and
/// the MCP in-process handler) prints: the semantic version Cargo built, plus
/// the git ref and commit hash [`build.rs`] embedded, additively — never a
/// replacement, so a build with an "unknown" ref (no tag, no branch, no git)
/// still reports a real version number rather than none at all.
fn cli_version_string() -> String {
    format!(
        "{} ({}, {})",
        env!("CARGO_PKG_VERSION"),
        env!("ROCM_CLI_VERSION_REF"),
        env!("ROCM_CLI_GIT_HASH")
    )
}

/// `rocm version`: the traceable build string, plus the ROCm SDK and GPU
/// driver this machine would actually use -- unlike `-V`/`--version` and the
/// MCP fast path, which stay a single terse line for scripts and in-process
/// callers.
///
/// "The ROCm SDK" prefers the active managed TheRock runtime (what `rocm`
/// itself runs engines against), falling back to a detected but unmanaged
/// system ROCm install -- the same precedence `rocm`'s freeform "ROCm status"
/// answer already uses, just without its other, heavier probing.
fn version() -> Result<()> {
    println!("rocm-cli {}", cli_version_string());

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let manifests = therock::load_runtime_manifests(&paths).unwrap_or_default();
    match current_runtime_manifest(&config, &manifests) {
        Some(manifest) => println!(
            "ROCm SDK: {} ({})",
            therock::runtime_version_display(&manifest.version),
            manifest.install_root.display()
        ),
        None => match rocm_core::detect_legacy_rocm_sdk() {
            Some((version, path)) => {
                println!(
                    "ROCm SDK: {version} (unmanaged install at {})",
                    path.display()
                );
            }
            None => println!("ROCm SDK: not detected"),
        },
    }

    match rocm_core::detect_gpu_driver_version() {
        Some(version) => println!("GPU driver: {version}"),
        None => println!("GPU driver: not detected"),
    }

    Ok(())
}

/// Legacy `uv` cache location, used before the cache was colocated with the managed
/// data directory. Kept relative so the check works on every platform's home dir.
const LEGACY_UV_CACHE_RELATIVE: [&str; 2] = [".cache", "uv"];

/// One-shot notice that a pre-colocation `uv` cache is still occupying space at the
/// default `uv` location. Nothing is migrated or deleted: the cache is
/// content-addressed and may be shared with unrelated `uv` projects on the machine, so
/// removing it is the user's call. Silent when the managed cache does not exist yet
/// (nothing has moved) or when an override is in effect.
fn maybe_notice_legacy_uv_cache() {
    let Ok(paths) = AppPaths::discover() else {
        return;
    };
    let cache = uv_cache_source(&paths);
    if cache.is_override() {
        return;
    }
    // Only worth mentioning once the managed cache is actually in use; otherwise the
    // legacy directory is simply the cache still being used by other tools.
    if !cache.path().is_dir() {
        return;
    }
    let Some(home) = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
    else {
        return;
    };
    let legacy = LEGACY_UV_CACHE_RELATIVE
        .iter()
        .fold(home, |dir, part| dir.join(part));
    if !legacy.is_dir() {
        return;
    }
    // One-shot: a standing reminder on every invocation would be noise, and the user may
    // reasonably decide to keep the legacy cache for other uv projects.
    let marker = paths.data_dir.join(".legacy-uv-cache-notice");
    if marker.exists() {
        return;
    }
    eprintln!(
        "rocm: the uv cache now lives at {}; the previous cache at {} is no longer used by rocm-cli and can be removed if no other uv project needs it",
        cache.path().display(),
        legacy.display()
    );
    let _ = fs::write(&marker, b"");
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
        // Bare `rocm` opens the minimal launcher front door; "Open full
        // dashboard" / `d` escalates into the unified dash, "Chat" reaches the
        // Chat tab. `rocm dash` / `rocm chat` bypass the launcher. The legacy
        // TUI assistant has been retired.
        return dash::run_launcher(false);
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
    let plan = freeform_planner::build_freeform_plan_with_context(&request, &paths, &config);
    if !approve
        && let Some(answer) = render_freeform_read_only_answer(&request, &plan, &paths, &config)?
    {
        print!("{answer}");
        return Ok(());
    }
    print!(
        "{}",
        freeform_planner::render_structured_request_plan(&plan, &paths)
    );
    if approve {
        execute_freeform_next_action(&request, &paths, &config)?;
    }
    Ok(())
}

/// Returns clap's own error when a freeform request looks like a botched
/// command invocation rather than natural language, so the caller can surface
/// clap's "did you mean"/usage message and exit instead of feeding it to the
/// planner. clap's error is surfaced when it carries a subcommand suggestion (a
/// near-miss typo such as `instal` or `automatios list`) or when the request
/// includes flag-style arguments (such as `doctorgdfg --help`). Genuine prose,
/// which yields neither, is left to the planner.
fn command_invocation_error(request_args: &[String]) -> Option<clap::Error> {
    if request_args.is_empty() {
        return None;
    }
    // A real subcommand token never contains whitespace, so a single quoted
    // prose argument (such as `"is rocm installed?"`) is natural language, not a
    // mistyped subcommand. clap can still propose a near-miss suggestion for
    // such a string (`is rocm installed?` -> `install`), so guard against it
    // here and leave the request for the planner.
    if request_args[0].split_whitespace().count() > 1 {
        return None;
    }
    let argv = std::iter::once("rocm".to_owned()).chain(request_args.iter().cloned());
    let err = Cli::try_parse_from(argv).err()?;
    if err.kind() != clap::error::ErrorKind::InvalidSubcommand {
        return None;
    }
    let has_suggestion = err
        .get(clap::error::ContextKind::SuggestedSubcommand)
        .is_some();
    let has_flag = request_args.iter().any(|arg| arg.starts_with('-'));
    (has_suggestion || has_flag).then_some(err)
}

fn setup(command: Option<SetupCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(SetupCommand::Status) {
        SetupCommand::Status => {
            print!("{}", render_setup_status_text(&paths, &config)?);
        }
        SetupCommand::Reset { yes: _ } => {
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
        "first-time setup available - open manually via `rocm dash`'s Observe tab with `n`"
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
        "Onboarding will not reopen automatically - open it from `rocm dash`'s Observe tab with `n`.",
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
    let execution = prepare_freeform_execution(request, paths, config)?;
    print!("{}", render_freeform_execution_header(&execution));

    let mut argv = vec!["rocm".to_owned()];
    argv.extend(execution.action.args);
    let cli = Cli::try_parse_from(argv)?;
    dispatch(cli)
}

/// The argv `execute_freeform_next_action` is about to dispatch, plus whether
/// [`apply_freeform_execution_consent`] added a flag to it.
///
/// The flag is tracked rather than re-detected from `action.args` because the
/// execution header uses it to tell the operator *why* its `tool_call:` differs
/// from the one in the request plan above. Looking for the flag in the final
/// argv would report "added here" for a plan that already carried it.
pub(crate) struct FreeformExecution {
    pub action: freeform_planner::FreeformPlanAction,
    pub consent_added: bool,
}

/// Everything `execute_freeform_next_action` decides before it hands the argv to
/// clap: plan, refuse what must not run unattended, and grant the consent the
/// outer `--yes` already carries.
///
/// Split out so the consent injection is reachable from a test without
/// dispatching a real install — the header render and the `dispatch` call are
/// all that is left above it.
fn prepare_freeform_execution(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<FreeformExecution> {
    let mut action =
        freeform_planner::freeform_plan_next_action_with_context(request, paths, config)
            .context("natural-language plan did not produce a structured tool call")?;
    validate_freeform_execution_action(&action)?;
    let consent_added = apply_freeform_execution_consent(&mut action.args);
    Ok(FreeformExecution {
        action,
        consent_added,
    })
}

/// Grant the generated tool call the consent the operator already gave on the
/// outer command line.
///
/// Only ever reached from `run_freeform` with `approve` set, i.e. from
/// `rocm --yes <natural-language request>`. The planner builds a bare
/// `install sdk ...` argv and `execute_freeform_next_action` re-parses and
/// dispatches it **in process**, so `install()` would otherwise run with
/// `yes = false, approve_replacing_active_default = false` no matter what the
/// outer invocation said. That made this surface print `approval: granted by
/// --yes` and then, with an active default runtime, refuse with "re-run with
/// `--approve-replacing-active-default`" — a flag this surface offers no way to
/// pass — or prompt on a terminal it had just said it did not need to ask.
///
/// Not `--yes`, for the same reason every other internal caller picks the narrow
/// flag: `--yes` on `install sdk` carries a second, unrelated consent for
/// system-package installs that run `sudo`, and the outer `--yes` here means
/// "execute the plan you were just shown", not "install system packages".
///
/// Injected before the execution header renders, so the printed `tool_call:` is
/// the argv that actually runs, and skipped under `--dry-run`, which returns
/// before the gate and needs no consent — matching the dry-run-aware arms in
/// `chat_rocm_command_action_from_args`, `rocmd` and dash-tui. The plan
/// rendering path is deliberately untouched: `rocm <request>` without `--yes`
/// prints a command for a human to review, and it must not hand them a
/// pre-approved one.
///
/// That leaves `rocm --yes <request>` printing two `tool_call:` lines that
/// differ — `run_freeform` renders the plan section before calling
/// `execute_freeform_next_action`, and the flag is injected between them — and
/// that is also deliberate. The two sections report different things: "request
/// plan" is what the planner derived from the request, and "execution" is the
/// argv handed to clap, so the added consent showing up only under the
/// `execution` header is how this surface discloses that it granted it. Nothing
/// is being solicited in between; under `--yes` the operator already approved
/// on the outer command line, and under no `--yes` the execution section is
/// never reached, so neither line is an approval prompt whose subject could
/// drift from what runs. Injecting into the plan render instead would have to
/// reach `render_structured_request_plan`, which the no-`--yes` review path
/// shares, and would print a pre-approved command to a human being asked to
/// review it — the case the paragraph above rules out.
///
/// So the difference stays deliberate but stops being unexplained: this returns
/// whether it actually added the flag, and the execution section says so in
/// words. A doc comment reaches the next reader of this file; the operator
/// looking at two `tool_call:` lines that disagree is the one who needs it.
///
/// Returns `true` only when the flag was not already present, so the disclosure
/// is about a flag this function added and not one the argv arrived with.
fn apply_freeform_execution_consent(args: &mut Vec<String>) -> bool {
    let is_install_sdk = args.first().is_some_and(|arg| arg == "install")
        && args.get(1).is_some_and(|arg| arg == "sdk");
    if !is_install_sdk || args.iter().any(|arg| arg == "--dry-run") {
        return false;
    }
    let already_present = args
        .iter()
        .any(|arg| arg == "--approve-replacing-active-default");
    chat::ensure_flag(args, "--approve-replacing-active-default");
    !already_present
}

fn validate_freeform_execution_action(action: &freeform_planner::FreeformPlanAction) -> Result<()> {
    if action.has_placeholders {
        bail!(
            "cannot execute natural-language plan because the next tool call still contains placeholder values: {}",
            freeform_planner::format_structured_tool_call("rocm", &action.args)
        );
    }
    if action.provider_assisted {
        bail!(
            "provider-assisted plans must be reviewed interactively; run the displayed structured rocm command directly after reviewing it"
        );
    }
    Ok(())
}

fn render_freeform_execution_header(execution: &FreeformExecution) -> String {
    let action = &execution.action;
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
        freeform_planner::format_structured_tool_call("rocm", &action.args)
    );
    // Printed only when the two `tool_call:` lines actually disagree, and
    // immediately under the one that runs. Without it the operator sees a
    // consent flag on the executed command that the request plan above never
    // showed, with nothing on screen saying where it came from — the natural
    // reading being that something was approved behind their back rather than
    // that their own `--yes` was carried through.
    if execution.consent_added {
        let _ = writeln!(
            output,
            "  note: --approve-replacing-active-default was added here from your --yes, so this \
             tool_call differs from the one under `request plan` above; nothing was approved \
             between them."
        );
    }
    output
}

fn render_freeform_read_only_answer(
    request: &str,
    plan: &freeform_planner::StructuredRequestPlan,
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
            comfyui::render_chat_status_answer(paths).map(Some)
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
    let asks_where = chat::any_substring(&lower, &["where", "folder", "path"]);
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

/// Sampling controls a caller can pass through to the underlying chat request.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ChatInferenceParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

fn dispatch(cli: Cli) -> Result<()> {
    // `storage` is excluded alongside the other self-referential commands: it
    // reports and reclaims disk, and the startup update check can provision a
    // managed Python, which downloads instead of freeing.
    if !matches!(
        cli.command,
        Some(
            Command::Update { .. }
                | Command::Bootstrap { .. }
                | Command::Completions { .. }
                | Command::Storage { .. }
        )
    ) {
        refresh_startup_update_check_quietly();
    }

    match cli.command {
        Some(Command::Examine { json, framework }) => examine(json, framework.into()),
        Some(Command::Diagnose {
            symptom,
            top,
            json,
            distro,
            model,
            report,
            send,
        }) => diagnose(symptom, top, json, distro, model, report, send),
        // Keep this error chained rather than discarding it into a fresh
        // `anyhow!(...)` (e.g. via a `.map_err` that restringifies it) -- see
        // `FixExitCode`'s doc comment for why that would silently break its
        // exit-code-carrying downcast. `.context(...)` is fine.
        Some(Command::Fix {
            fix_id,
            yes,
            dry_run,
            device_index,
            json,
        }) => fix(fix_id, yes, dry_run, device_index, json),
        Some(Command::Version) => version(),
        Some(Command::Setup { command }) => setup(command),
        Some(Command::EngineServeHttp {
            engine,
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            gpu,
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
            parse_gpu_indices_arg(gpu.as_deref())?,
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
            let value = chat::run_internal_mcp_call(&paths, &name, arguments, allow_mutation)?;
            println!("{}", serde_json::to_string(&value)?);
            Ok(())
        }
        Some(Command::Chat {
            provider,
            model,
            prompt,
            temperature,
            top_p,
            max_tokens,
            tools,
            chat_mock,
        }) => {
            let paths = AppPaths::discover()?;
            if interactive_terminal() && prompt.is_none() {
                // Interactive `rocm chat` routes to the unified dash chat
                // (parity reached, Phases 3-8). The dash auto-detects the
                // provider and supports `/provider` switching; --provider is
                // honored only on the non-interactive render path below. The
                // legacy TUI assistant has been retired.
                if provider.is_some() {
                    // --provider isn't threaded into the interactive dash; say so
                    // instead of dropping it silently — the user can switch live.
                    eprintln!(
                        "note: launching the dash chat; switch providers with /provider <name>"
                    );
                }
                return dash::run_chat(
                    chat_mock,
                    ChatInferenceParams {
                        temperature,
                        top_p,
                        max_tokens,
                    },
                    false,
                );
            }
            // Resolve the prompt from `--prompt` or, when it is omitted and
            // stdin is not a terminal, from piped standard input — the
            // documented `echo "…" | rocm chat` path. Only when neither
            // supplies a prompt do we fall back to the status screen, which
            // two kinds of invocation reach: stdin is a TTY that the
            // interactive branch above declined (it also requires stdout to be
            // one), or stdin was redirected and carried nothing but whitespace
            // — an empty pipe or `< /dev/null`.
            let prompt = match prompt {
                Some(prompt) => Some(prompt),
                None => chat::read_piped_prompt()?,
            };
            match prompt {
                Some(prompt) => print!(
                    "{}",
                    chat::render_chat_prompt_text(
                        &paths,
                        provider.map_or("local", provider_name),
                        model.as_deref(),
                        &prompt,
                        tools,
                        ChatInferenceParams {
                            temperature,
                            top_p,
                            max_tokens,
                        },
                    )?
                ),
                None => print!(
                    "{}",
                    chat::render_chat_text(&paths, provider.map_or("local", provider_name))?
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
            yes: _,
            json,
            timeout_secs,
        }) => {
            let paths = AppPaths::discover()?;
            if !apply && !dry_run && (runtime.is_some() || activate) {
                bail!(
                    "--runtime and --activate require --apply or --dry-run; \
                     run `rocm update --dry-run` to preview or add --apply to update"
                );
            }
            if update_should_preview_or_apply(apply, dry_run) {
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
            if json {
                match therock::render_update_json(&paths, timeout_secs) {
                    Ok(document) => {
                        println!("{}", serde_json::to_string(&document)?);
                        record_cli_audit_event(
                            &paths,
                            "update",
                            "update_check",
                            "info",
                            "rendered update report (json)",
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
        Some(Command::Storage { command }) => storage::storage(command),
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
            gpu,
            runtime_id,
            env_id,
            host,
            port,
            foreground,
            managed,
            verbose,
            no_smoke_test,
            allow_public_bind,
            require_api_key,
            tool_call_parser,
            gpu_memory_utilization,
            temperature,
            top_p,
            max_tokens,
            api_key,
        }) => serve(ServeArgs {
            model,
            engine,
            device,
            gpu,
            runtime_id,
            env_id,
            host,
            port,
            foreground,
            managed,
            verbose,
            no_smoke_test,
            allow_public_bind,
            require_api_key,
            tool_call_parser,
            gpu_memory_utilization,
            temperature,
            top_p,
            max_tokens,
            api_key,
        }),
        Some(Command::Comfyui { command }) => comfyui(command),
        Some(Command::Services { command }) => services(command),
        Some(Command::Remote { command }) => remote::run(command),
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
        }) => dash::run(replay, demo, chat_mock, false),
        Some(Command::Bench { command }) => match command {
            BenchCommand::Load {
                endpoint,
                model,
                concurrency,
                isl,
                osl,
                requests,
                out,
                auto_ramp,
            } => dash::run_bench(dash::BenchLoadArgs {
                endpoint,
                model,
                concurrency,
                isl,
                osl,
                requests,
                out,
                auto_ramp,
            }),
        },
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
    if let Some(display_name) = cmd.get_display_name() {
        bare = bare.display_name(display_name.to_owned());
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
        (
            "lemonade",
            "default embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
    ]
}

/// The `--json` document.
///
/// `Examination` stays at the top level, verbatim: its field names deliberately
/// mirror `examine.py`'s dataclass so the JSON contract is identical, and moving
/// them would break every consumer written against it.
///
/// Everything the CLI knows but the probe cannot is added under `summary`. The
/// probe runs before any config is read, so it has no idea which engine this
/// host will serve on, whether an existing ROCm install was found, or where the
/// CLI keeps its files — all of which the human report prints. A tool reading
/// `--json` used to have to scrape that text to find out.
#[derive(serde::Serialize)]
struct ExamineJson<'a> {
    #[serde(flatten)]
    examination: &'a rocm_core::Examination,
    summary: ExamineJsonSummary<'a>,
}

/// The human report's own view, plus the few facts it derives from config while
/// rendering rather than storing on the summary.
#[derive(serde::Serialize)]
struct ExamineJsonSummary<'a> {
    #[serde(flatten)]
    host: &'a ExamineSummary,
    /// What the user configured — `None` means unset, exactly as the text form's
    /// `<platform default>` does. Distinct from `effective_default_engine`.
    configured_default_engine: Option<&'a str>,
    /// What `serve` will actually pick: the configured engine if there is one,
    /// else the host default. Mirrors `select_serve_engine`.
    effective_default_engine: &'a str,
    active_runtime_id: Option<&'a str>,
    active_runtime_key: Option<&'a str>,
    previous_runtime_key: Option<&'a str>,
    /// Where the active runtime lives, completing the `id`/`key`/`root` triple.
    /// Not derivable from the key: `install_root` is its own field on the
    /// manifest, and `install sdk --prefix`, `runtimes adopt` and `runtimes
    /// import` all set it freely.
    active_runtime_root: Option<String>,
    /// The folder setup was configured for, straight from config. Distinct from
    /// `active_runtime_root`, not a fallback for it: setup's folder can be a
    /// stale or removed install while a different runtime is active. Reachable
    /// without the registry, which is what makes it worth carrying — see
    /// `recover_setup_runtime_registration`, the write `--json` will not do.
    setup_runtime_root: Option<String>,
    /// The text form's sibling fact, derived from `setup_runtime_root` rather
    /// than stored, so it cannot drift from where the pip cache actually goes.
    setup_runtime_pip_cache_dir: Option<String>,
}

fn examine(json: bool, framework: rocm_core::FrameworkProbe) -> Result<()> {
    // `rocm examine` is the general system inspector: the exit code reports
    // whether it RAN, not what it found. Any finding (no GPU, WSL, degraded) is
    // surfaced in the output and the `--json` `status` field, and the command
    // exits 0; a genuine inability to examine propagates as an error via `?`.
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    if json {
        // Prefer the active runtime's interpreter: in the managed configuration
        // torch lives only in its site-packages, so probing PATH would report
        // `unknown` for a host that has one.
        let interpreter = rocm_core::active_managed_framework_interpreter(&paths, &config);
        let examination =
            rocm_core::Examination::probe_with_interpreter(framework, interpreter.as_ref());
        // `gather` rather than `examine_human_report`: the latter first runs
        // `recover_setup_runtime_registration`, which writes. Asking a machine a
        // question should not change it, and `--json` is the form tooling calls
        // in a loop.
        let host = ExamineSummary::gather()?;
        let configured_default_engine = config.default_engine.as_deref();
        // Same resolution the human report uses, minus the recovery write above:
        // an unreadable registry leaves the root `null` rather than failing the
        // inspection, which is the weaker answer but still an answer.
        let manifests = therock::load_runtime_manifests(&paths).unwrap_or_default();
        let active_runtime_root = current_runtime_manifest(&config, &manifests)
            .map(|manifest| manifest.install_root.display().to_string());
        let (setup_runtime_root, setup_runtime_pip_cache_dir) = setup_runtime_paths(&config)
            .map(|(root, cache)| (root.display().to_string(), cache.display().to_string()))
            .unzip();
        let document = ExamineJson {
            examination: &examination,
            summary: ExamineJsonSummary {
                configured_default_engine,
                effective_default_engine: configured_default_engine.unwrap_or(&host.default_engine),
                active_runtime_id: config.default_runtime_id.as_deref(),
                active_runtime_key: config.active_runtime_key.as_deref(),
                previous_runtime_key: config.previous_runtime_key.as_deref(),
                active_runtime_root,
                setup_runtime_root,
                setup_runtime_pip_cache_dir,
                host: &host,
            },
        };
        println!("{}", serde_json::to_string_pretty(&document)?);
        return Ok(());
    }
    let (text, summary) = examine_human_report(&paths, &config)?;
    print!("{text}");
    if summary.wsl.as_ref().is_some_and(|w| w.is_wsl) {
        // Informational platform guidance for humans (the verdict is also in
        // the `status` field for `--json` consumers).
        println!("\n{}", rocm_core::WSL_PLATFORM_NOTE);
    }
    Ok(())
}

fn diagnose(
    symptom: Option<String>,
    top: usize,
    json: bool,
    distro: Option<String>,
    model: Option<String>,
    report_requested: bool,
    send: bool,
) -> Result<()> {
    // The model verdict is about THIS machine, always. `--distro` retargets the
    // environment examination at another one, but the GPU memory and engine
    // selection behind a model verdict are read locally -- so answering for a
    // model here would describe the wrong host under a heading naming another.
    // Refuse rather than substitute, the same way `--distro` itself does when it
    // cannot reach a machine.
    //
    // Keyed on whether `--distro` was passed, not on anything the probe below
    // populates. An examination-derived signal (e.g. whether `examination.wsl`
    // ended up set, or `locally_probed`) would make this refusal depend on
    // probe internals instead of on the flag the user actually typed --
    // exactly the kind of coupling that lets a future change to the probe
    // silently let a `--distro` run fall through and answer for the local host
    // under a heading naming another machine.
    if model.is_some() && distro.is_some() {
        bail!(
            "--model answers for the machine running this command, and --distro points the \
             examination at a different one. Run `rocm diagnose --model <model>` inside the \
             distribution instead."
        );
    }
    // `rocm diagnose` is a query: it exits 0 whether it matched, found nothing,
    // or is out of scope. Callers read `has_match` / `out_of_scope` /
    // `route_when_no_match` from `--json` rather than branching on the exit code.
    //
    // `--distro` is the exception that still errors: the user named a machine to
    // inspect, and silently reporting on a different one would be worse than
    // failing. `--distro` with no value means "the only one installed".
    let examination = if let Some(name) = distro {
        let selected = (!name.is_empty()).then_some(name);
        rocm_core::probe_wsl_distro_from_host(selected.as_deref())
            .map_err(|reason| anyhow::anyhow!("{reason}"))?
    } else {
        // Same reasoning as `examine --json`: the catalog reasons over the torch
        // the engines will load, which is the active runtime's.
        //
        // Best-effort, unlike `examine`'s copy: this command already promises to
        // answer on a degraded host, so a path-discovery failure must cost only
        // the managed-runtime lookup, never the diagnosis.
        let interpreter = AppPaths::discover().ok().and_then(|paths| {
            let config = RocmCliConfig::load(&paths).unwrap_or_default();
            rocm_core::active_managed_framework_interpreter(&paths, &config)
        });
        rocm_core::Examination::probe_with_interpreter(
            rocm_core::FrameworkProbe::Auto,
            interpreter.as_ref(),
        )
    };
    let inspected_remotely = examination
        .wsl
        .as_ref()
        .is_some_and(|wsl| !wsl.locally_probed);
    let mut report = rocm_core::run_diagnose(&examination, &symptom.unwrap_or_default());
    if let Some(model_ref) = &model {
        report.model = Some(assess_model_on_this_host(model_ref, &examination));
    }
    if report_requested {
        return show_prepared_report(&examination, &report, json, send);
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", rocm_core::render_diagnose_text(&report, top));
        if let Some(readiness) = &report.model {
            println!();
            print!(
                "{}",
                rocm_core::model_readiness::render_model_readiness_text(readiness)
            );
        }
        // Inspecting a distribution from outside it collects no environment, so
        // the checks that read one never run. Without saying so, "no known
        // misconfiguration matched" reads as a clean bill of health for the
        // whole installation, when it only covers the WSL GPU stack.
        if inspected_remotely {
            println!(
                "\nNote: inspected from outside the distribution, which sees only its \
                 WSL GPU stack.\nChecks that read the environment (HSA_OVERRIDE_GFX_VERSION, \
                 PATH, the framework/ROCm pairing)\ndid not run. For those, run `rocm diagnose` \
                 inside the distribution."
            );
        }
    }
    Ok(())
}

/// Answer `--model` for the machine running the command.
///
/// Gathers only what the verdict needs, and keeps every "could not read it"
/// apart from every "it is not there": an unreadable recipe index becomes
/// `Unreachable` rather than an empty catalog, and GPU memory that `amd-smi`
/// could not report becomes `Unknown` rather than zero. Both collapses would
/// turn a gap in what the CLI can see into a confident refusal of the model.
fn assess_model_on_this_host(
    model_ref: &str,
    examination: &rocm_core::Examination,
) -> ModelReadiness {
    let paths = AppPaths::discover().ok();
    assess_model_with_host_paths(model_ref, examination, paths.as_ref())
}

/// Testable core of [`assess_model_on_this_host`].
///
/// Takes the discovered `AppPaths` as an explicit parameter so the GPU-summary
/// lookup and the config lookup read the same directories, instead of each
/// racing its own `AppPaths::discover()` call. That used to be two separate
/// calls: one (implicitly `None`) feeding [`detect_host_gpu_summary`], another
/// feeding the config lookup a few lines later. Both resolve identically today
/// because `AppPaths::discover()` is deterministic within a process, but
/// `detect_host_gpu_summary`'s paths-driven TheRock-manifest gfx-target
/// fallback only ever sees a path when one is passed in -- silently skipping
/// it here is exactly the kind of drift that would let this command name a
/// different engine than `rocm serve` for the same model.
fn assess_model_with_host_paths(
    model_ref: &str,
    examination: &rocm_core::Examination,
    paths: Option<&AppPaths>,
) -> ModelReadiness {
    let host_gpu_summary = detect_host_gpu_summary(paths);
    let host = HostFacts {
        accelerator_memory: host_accelerator_memory(&host_gpu_summary, examination),
        system_ram_gib: rocm_core::detect_system_ram_gib(),
    };
    let configured_engine = paths
        .and_then(|paths| RocmCliConfig::load(paths).ok())
        .and_then(|config| config.default_engine);
    match load_model_recipe_registry() {
        Ok(registry) => assess_model_for_host(
            model_ref,
            &ModelCatalogSource::Available(&registry),
            &host,
            configured_engine.as_deref(),
            Some(&host_gpu_summary),
        ),
        Err(error) => assess_model_for_host(
            model_ref,
            &ModelCatalogSource::Unreachable {
                detail: format!("{error:#}"),
            },
            &host,
            configured_engine.as_deref(),
            Some(&host_gpu_summary),
        ),
    }
}

/// The memory pool an engine on this host would allocate a model from.
///
/// The APU case is why this is not simply a sum of `total_vram`. An APU has no
/// private VRAM; `amd-smi` reports the fixed BIOS carve-out (often ~4 GiB) while
/// the allocator serves the model out of GTT-backed system RAM. Comparing a
/// recipe minimum against the carve-out would refuse a 22 GiB model on a 128 GiB
/// Strix Halo, which serves it fine. [`vram_capacity_is_meaningful`] is the same
/// test `serve`'s low-VRAM warning uses, so the two cannot disagree about which
/// hosts the figure describes.
///
/// The examination is what makes "there is no GPU" reachable at all. `amd-smi`
/// answers "how much memory", and its absence is ambiguous — no GPU, or no
/// `amd-smi`. The examination reads the devices themselves, so it can tell those
/// two apart, and they deserve different answers: one is a machine that needs a
/// GPU, the other a machine whose GPU could not be measured.
/// Whether the examination has positive or unresolved evidence of an AMD GPU,
/// used to tell "no GPU" apart from "GPU present but VRAM unmeasured" when
/// `amd-smi` produced nothing.
///
/// `examination.has_amd_gpu` cannot answer this on WSL: the hardware probes
/// that set it are skipped there (see [`rocm_core::Examination::probe_with_interpreter`]),
/// so it reads `false` on every WSL host, healthy or not, and this command
/// would report a working machine as having no GPU at all -- `Blocked`
/// instead of the `Undetermined` a genuinely unmeasurable host deserves.
/// `rocm_sees_gpu`, not `has_amd_gpu`, is the WSL-native signal, per the same
/// reasoning `fix-wsl-6-host-driver-too-old` already uses in `diagnose.rs`.
/// Only `Some(false)` -- rocminfo positively enumerating no device -- counts
/// as "no GPU"; `None` means rocminfo was absent so the question went unasked,
/// which is not evidence of anything and must not read as a confident no.
fn examination_reports_amd_gpu(examination: &rocm_core::Examination) -> bool {
    match examination.wsl.as_ref() {
        Some(wsl) => wsl.rocm_sees_gpu != Some(false),
        None => examination.has_amd_gpu,
    }
}

fn host_accelerator_memory(
    host_gpu_summary: &rocm_core::HostGpuSummary,
    examination: &rocm_core::Examination,
) -> AcceleratorMemory {
    classify_accelerator_memory(
        gpu_vram_usage().as_deref(),
        host_gpu_summary.gfx_target.as_deref(),
        examination_reports_amd_gpu(examination),
        rocm_core::usable_amd_gpu_indices().as_deref(),
    )
}

/// The pure classification behind [`host_accelerator_memory`], split out so it
/// can be exercised with fixed input instead of a live `amd-smi` subprocess.
/// Deciding whether a host is an APU and which figure represents its memory is
/// the actual judgement this command depends on; entangled with the
/// subprocess call it was untestable, and a checker that silently regressed
/// to always reporting `Dedicated` would leave every test green.
fn classify_accelerator_memory(
    vram: Option<&[GpuVramUsage]>,
    gfx_target: Option<&str>,
    gpu_reported: bool,
    usable_indices: Option<&[u32]>,
) -> AcceleratorMemory {
    let Some(vram) = vram else {
        if gpu_reported {
            return AcceleratorMemory::Unknown;
        }
        return AcceleratorMemory::None;
    };
    if vram_capacity_is_meaningful(gfx_target, vram.len()) {
        // `rocm serve` pins exactly one ordinal -- there is no tensor-parallel
        // path anywhere in this binary -- so summing every card's VRAM into one
        // figure can report a model READY when it would OOM on every individual
        // card. Narrow to the mask-visible rows (a `HIP_VISIBLE_DEVICES` mask
        // hides devices the same way it does for `rocm serve` itself) and take
        // the most capable one: that is the best case `--gpu auto` could
        // actually land on, so it is the right upper bound for "would this
        // fit", even though it cannot name which ordinal the verdict answers
        // for. `usable_indices` of `None` means the mask could not be probed,
        // so every reported row is treated as visible rather than refusing to
        // answer.
        let max_gib = vram
            .iter()
            .filter(|usage| usable_indices.is_none_or(|indices| indices.contains(&usage.index)))
            .map(|usage| usage.total_mb as f64 / 1024.0)
            .fold(0.0_f64, f64::max);
        return AcceleratorMemory::Dedicated(max_gib);
    }
    // This is an APU: the carve-out `amd-smi` reports above is the wrong pool,
    // per `vram_capacity_is_meaningful`'s own doc. The right pool is the GTT
    // aperture, and nothing in this binary can read it -- `amd-smi` does not
    // report it, and no code path here calls `rocm-smi`, the tool that does.
    // Total system RAM is not a substitute: the GTT aperture is capped well
    // below installed RAM by BIOS/kernel policy, and asserting installed RAM
    // as the engine's allocation pool trades the carve-out's under-estimate
    // for an over-estimate in exactly the direction that produces a false
    // `Ready` for a model that does not actually fit -- the wrong answer this
    // command exists to prevent. This is deliberately not `Unknown`: that
    // variant means no telemetry at all, for which "run `amd-smi metric
    // --json`" is a real next step. Here `amd-smi` already ran and answered --
    // it just named the wrong pool -- so the same remediation would send a
    // Strix Halo user in a circle forever. Until the real aperture can be
    // read, this reports `UnifiedMemoryUnreadable`, so the verdict reads
    // `Undetermined` with no dead-end command attached, rather than a
    // confident wrong one.
    AcceleratorMemory::UnifiedMemoryUnreadable
}

/// Assess a model against this host, with `serve`'s own engine decision.
///
/// The engine is not re-derived here. `select_serve_engine` is the one place
/// that answers "which engine serves this model on this host", and passing it in
/// is what keeps `rocm diagnose --model` from confidently naming an engine
/// `rocm serve` would never pick. An engine ruled out by the platform gate is
/// reported as such rather than silently swapped for another: `serve` does not
/// fall back either, and a verdict that pretended otherwise would be wrong in
/// the user's favour.
fn assess_model_for_host(
    model_ref: &str,
    catalog: &ModelCatalogSource<'_>,
    host: &HostFacts,
    configured_default_engine: Option<&str>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
) -> ModelReadiness {
    let engine_for = |recipe: &ModelRecipeRecord| {
        let selection = select_serve_engine(
            None,
            configured_default_engine,
            Some(recipe),
            host_gpu_summary,
        );
        HostEngineChoice {
            unsupported_here: unsupported_here_for(
                &selection.engine,
                engine_ruled_out_by_platform(&selection.engine),
            ),
            engine: selection.engine,
            source: selection.source.to_owned(),
        }
    };
    rocm_core::model_readiness::assess_model_readiness(model_ref, catalog, host, &engine_for)
}

/// Whether the platform gate rules this engine out on this host.
///
/// One place, so the `/model` adapter-availability note and the readiness
/// verdict cannot answer differently about the same host and engine.
const fn engine_ruled_out_by_platform(engine: &str) -> bool {
    rocm_core::runtime_is_windows() && engine.eq_ignore_ascii_case("vllm")
}

/// The `unsupported_here` message for an engine, given whether the platform
/// gate already ruled it out.
///
/// Split out from the `engine_for` closure in [`assess_model_for_host`] so it
/// can be unit-tested on both branches directly, instead of only indirectly
/// through whichever platform the test happens to run on. `ruled_out` is a
/// plain `bool` rather than re-deriving it from `engine` here, so a test can
/// exercise the "ruled out" branch on Linux CI and the "allowed" branch on a
/// Windows runner without either one being unreachable.
fn unsupported_here_for(engine: &str, ruled_out: bool) -> Option<String> {
    ruled_out
        .then(|| format!("{engine} has no adapter on native Windows; serve it from WSL or Linux"))
}

/// The catalog entry a report should name, and whether a fix was offered for it.
///
/// Reads `has_match` rather than taking the head of `matched`. Several checkers
/// open with a nonzero score for a situation that is merely *potentially*
/// relevant, so `matched` is rarely empty even on a healthy machine — taking its
/// head regardless would publish a sub-threshold signal as though it were an
/// established cause, and the counts built on those reports would be wrong in a
/// way nothing downstream could detect.
fn established_entry(report: &rocm_core::DiagnoseReport) -> (Option<&str>, bool) {
    if !report.has_match {
        return (None, false);
    }
    report.matched.first().map_or((None, false), |top| {
        (Some(top.id.as_str()), top.fix.is_some())
    })
}

/// Act on a delivery decision, and say what happened.
///
/// Takes the decision rather than making it, and takes the opener rather than
/// being one. Both for the same reason: the decision is tested in `rocm-core`
/// against every environment, and this half has to be tested against an opener
/// that does not exist, on a machine with no browser. A function that decided
/// and opened could be verified on neither.
fn perform_delivery(
    delivery: &rocm_core::report_delivery::Delivery,
    opener: &dyn Opener,
) -> String {
    use rocm_core::report_delivery::{DESTINATION, Delivery};
    match delivery {
        // No mail client is started here on purpose, and the reason is worth
        // the line: this is the branch for a machine held over SSH, or a
        // server with no mail client at all, where starting one would open on
        // somebody else's desktop or fail silently. The address is named as
        // well as the link, because a machine in this state often cannot act
        // on a `mailto:` at all and the user has to send the mail by hand.
        Delivery::Show(url) => format!(
            "Nothing has been sent. To send this yourself, mail the report above to \
             {DESTINATION}, or open:\n  {url}"
        ),
        Delivery::Open(url) => match opener.open(url) {
            Ok(()) => format!(
                "Nothing has been sent yet. A prefilled mail to {DESTINATION} was opened, and \
                 it is sent only when you send it:\n  {url}"
            ),
            // A failed open is not a failed command. The user still has the
            // address and the link, which is the whole of what this offers.
            Err(error) => format!(
                "Nothing has been sent. A mail client could not be started ({error}). To send \
                 this yourself, mail the report above to {DESTINATION}, or open:\n  {url}"
            ),
        },
    }
}

/// Print the report this machine would contribute, and send nothing.
fn show_prepared_report(
    examination: &rocm_core::Examination,
    report: &rocm_core::DiagnoseReport,
    json: bool,
    send: bool,
) -> Result<()> {
    let (entry, fix_offered) = established_entry(report);
    // Exit 0 either way. A refusal is this command working, not failing: it
    // decided correctly and said why, and a nonzero code would send a caller
    // looking for a fault. Anything scripting this reads the outcome from
    // `--json` rather than from the exit code, exactly as `rocm diagnose` itself
    // already asks callers to do.
    match rocm_core::prepare_report(examination, entry, fix_offered) {
        Ok(prepared) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&prepared)?);
            } else {
                println!("This is the whole of what a report would carry:");
                println!();
                println!("{}", serde_json::to_string_pretty(&prepared)?);
                println!();
                // The content is printed above before this decides anything,
                // so a report is always read before its form is offered. That
                // ordering is the promise `--send` makes, and `--send`
                // requires `--report` so it cannot be skipped.
                let delivery = rocm_core::report_delivery::deliver(&prepared, send, &|key| {
                    std::env::var(key).ok()
                });
                println!(
                    "{}",
                    perform_delivery(&delivery, &rocm_core::browser::SystemOpener)
                );
            }
            Ok(())
        }
        Err(refusal) => {
            let explanation = match refusal {
                rocm_core::ReportRefusal::UnreleasedHardware => {
                    // "Doctor" is what the epic calls this capability; the CLI
                    // has no such command, so a user reading this has nothing
                    // to run and nothing to look up.
                    "This machine holds hardware that is not on AMD's published ROCm \
                     compatibility matrix, so no report was prepared. A report describes only \
                     hardware the compatibility matrix lists as supported."
                }
                rocm_core::ReportRefusal::ArchitectureUnreadable => {
                    "No AMD GPU architecture could be read here, so nothing confirms this \
                     hardware is on the ROCm compatibility matrix. No report was prepared."
                }
                rocm_core::ReportRefusal::PlatformNotProbed => {
                    // Says what happened rather than dressing it as a finding
                    // about the machine. The earlier wording told a healthy WSL
                    // user their GPU could not be read, when nothing had looked.
                    "This CLI does not inspect the GPU on WSL yet, so it cannot confirm whether \
                     this hardware is on the ROCm compatibility matrix. No report was prepared. \
                     This is a gap in the tool, not a problem with the machine."
                }
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rocm_core::refusal_envelope(
                        refusal,
                        explanation
                    ))?
                );
            } else {
                println!("{explanation}");
            }
            Ok(())
        }
    }
}

fn fix(
    fix_id: Option<String>,
    yes: bool,
    dry_run: bool,
    device_index: Option<i64>,
    json: bool,
) -> Result<()> {
    let Some(fix_id) = fix_id else {
        if json {
            print!("{}", rocm_core::catalog_manifest_json()?);
        } else {
            print!("{}", rocm_core::list_fix_recipes());
        }
        return Ok(());
    };
    // Refused rather than ignored. "Apply this fix, as JSON" has no meaning, and
    // quietly dropping the flag would let a caller believe it had asked for
    // machine-readable output and got it.
    if json {
        anyhow::bail!(
            "`--json` describes the whole catalog, so it cannot be combined with a fix id. \
             Run `rocm fix --json` to read the catalog, or `rocm fix {fix_id}` to apply this fix."
        );
    }
    let opts = rocm_core::FixOptions {
        yes,
        dry_run,
        device_index,
    };
    let code = rocm_core::apply_fix(&fix_id, &opts);
    if code != 0 {
        return Err(FixExitCode(code).into());
    }
    Ok(())
}

pub(crate) fn record_cli_audit_event(
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
            devel,
            version,
            build_date,
            family,
            dry_run,
            yes,
            approve_replacing_active_default,
        } => {
            let consents = SdkInstallConsents::resolve(yes, approve_replacing_active_default);
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
                chat::sdk_install_request(
                    &channel,
                    format_name,
                    prefix,
                    version_selector,
                    family.as_deref(),
                    dry_run,
                    devel,
                    consents.replace_active_default,
                ),
            ) {
                Ok(result) => {
                    let therock::SdkInstallResult { output, mutated } = result;
                    let finalized = if mutated {
                        finalize_successful_sdk_install(&paths)?
                    } else {
                        None
                    };
                    print!("{output}");
                    if let Some(finalized) = &finalized {
                        print_sdk_install_success(finalized);
                        // The SDK runtime wheel bundles PyTorch, whose ROCm build
                        // links against libatomic.so.1 and the system numactl
                        // runtime (libnuma.so.1 / libnuma_1.2). Ensure both are
                        // present for every SDK install, independent of which
                        // engine (if any) is auto-installed below.
                        ensure_libatomic_for_torch(consents.system_packages);
                        ensure_libnuma_for_torch(consents.system_packages);
                    }
                    finish_sdk_install(
                        &paths,
                        finalized.as_ref(),
                        if dry_run {
                            "install_sdk_dry_run"
                        } else {
                            "install_sdk"
                        },
                        {
                            // A real install that did not mutate the system was
                            // declined at the approval prompt; recording it as
                            // "completed" would lie in the audit trail. Dry-run
                            // never mutates but legitimately completes a preview.
                            let status = if dry_run || mutated {
                                "completed"
                            } else {
                                "cancelled"
                            };
                            format!(
                                "sdk install {status} channel={channel} format={format_name} prefix={prefix_display} version_selector={version_selector_display} dry_run={dry_run}"
                            )
                        },
                        |paths, finalized| {
                            maybe_auto_install_sdk_preferred_engine(
                                paths,
                                finalized,
                                consents.system_packages,
                            )
                        },
                    )?;
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

/// GPU/device warnings folded into the interactive deployment summary. Mirrors the
/// inline warnings printed in the plain serve plan, in the same order.
#[allow(clippy::too_many_arguments)]
fn collect_serve_notes(
    cpu_only: bool,
    gpu_selection: &GpuSelection,
    rocr_visible_devices_set: bool,
    gpu_indices: &[u32],
    gpu_vram: Option<&[GpuVramUsage]>,
    engine_flag_note: Option<&str>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
    engine_is_vllm: bool,
) -> Vec<String> {
    let mut notes = Vec::new();
    // An engine-scoped flag the selected engine cannot honor must be reported here
    // too: the summary path is what an interactive `rocm serve` actually prints, so
    // a note only emitted on the plan path would never reach that user.
    if let Some(note) = engine_flag_note {
        notes.push(note.to_owned());
    }
    if cpu_only {
        if matches!(gpu_selection, GpuSelection::Index(_)) {
            notes.push(
                "--gpu was ignored because --device cpu_only runs the model on CPU".to_owned(),
            );
        }
    } else {
        if rocr_visible_devices_set {
            notes.push(
                "ROCR_VISIBLE_DEVICES is set; the selected amd-smi ordinal is exported via \
                 HIP_VISIBLE_DEVICES, which the runtime interprets relative to the ROCR-visible \
                 set, so the device the engine binds may differ. Verify the selected GPU or unset \
                 ROCR_VISIBLE_DEVICES."
                    .to_owned(),
            );
        }
        if let Some(warning) = serve_gpu_low_memory_warning(gpu_indices, gpu_vram, host_gpu_summary)
        {
            notes.push(warning);
            // vLLM's total-VRAM reservation is what turns a busy card into an OOM;
            // pair the generic warning with the concrete knob that avoids it.
            if engine_is_vllm {
                notes.push(rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT.to_owned());
            }
        }
    }
    notes
}

fn services(command: Option<ServicesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    match command.unwrap_or(ServicesCommand::List {
        all: false,
        json: false,
    }) {
        ServicesCommand::List { all, json } => {
            if json {
                print!("{}", render_services_json(&paths, all)?);
            } else {
                print!("{}", render_services_text(&paths, all)?);
            }
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
        ServicesCommand::Remove { service_id, yes } => {
            print!(
                "{}",
                remove_managed_service_record(&paths, &service_id, yes)?
            );
            record_cli_audit_event(
                &paths,
                "service",
                "remove_record",
                "info",
                format!("removed local server record {service_id}"),
                Some(&service_id),
            );
            Ok(())
        }
        ServicesCommand::Prune {
            older_than_hours,
            any_age,
            dry_run,
            yes,
        } => {
            let older_than_hours = service_prune_min_age_hours(older_than_hours, any_age);
            let outcome = prune_managed_service_records(&paths, older_than_hours, dry_run, yes)?;
            print!("{}", outcome.text);
            if !dry_run && (outcome.removed_records > 0 || outcome.removed_files > 0) {
                record_cli_audit_event(
                    &paths,
                    "service",
                    "prune_records",
                    "info",
                    format!(
                        "removed {} local server record(s) and {} leftover file(s) older than {older_than_hours}h",
                        outcome.removed_records, outcome.removed_files
                    ),
                    None,
                );
            }
            // Deliberately after the print and the audit event: a file this run
            // could not delete still has to fail the command, but not at the
            // cost of the record of what it *did* delete.
            if !outcome.failures.is_empty() {
                // Only reachable through this dispatch, so no unit test covers
                // it: `service-cleanup-06` in
                // `features/service_record_cleanup.feature` is its cover.
                bail!(
                    "{} file(s) could not be removed; see the list above",
                    outcome.failures.len()
                );
            }
            Ok(())
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
            yes: _,
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
            yes: _,
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
        ComfyuiCommand::Stop { yes: _ } => match comfyui::stop(&paths) {
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
            "{} local server `{service_id}` requires --yes.\n\nTry: {}",
            service_action_verb(tool),
            service_hints::service_action_hint(service_action_command(tool), service_id)
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

// ---------------------------------------------------------------------------
// Local server record removal
// ---------------------------------------------------------------------------

/// Default age gate for `rocm services prune`.
///
/// A record that has only just stopped is the one a user is most likely to still
/// want: `rocm services list --all` advertises `rocm services restart <id> --yes`
/// for exactly that record, and pruning it destroys both that affordance and the
/// log explaining why it died. Defaulting to a day means a bulk cleanup reclaims
/// the accumulated history without swallowing the failure the user is currently
/// looking at. `--older-than-hours 0` opts out.
const DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS: u64 = 24;

/// The on-disk files one managed-service record owns.
///
/// Note `engine_state` lives *outside* `services_dir`, under
/// `<data>/engines/<engine>/state/`, so deleting the two files in `services_dir`
/// leaves it behind — that is how orphaned engine state accumulates today.
///
/// Every path is rebuilt here from [`AppPaths`] plus the *validated* id and
/// engine, deliberately **not** read from the record's own `manifest_path` /
/// `log_path` / `engine_state_path` fields. Those are deserialized from a JSON
/// file under `~/.rocm` that anything can write, and this is the one code path
/// that deletes what they name.
#[derive(Debug, Clone)]
struct ServiceRecordArtifacts {
    manifest: PathBuf,
    log: PathBuf,
    engine_state: PathBuf,
    endpoint_key: PathBuf,
}

impl ServiceRecordArtifacts {
    /// Every artifact, in the order they are reported and deleted.
    fn paths(&self) -> [&Path; 4] {
        [
            &self.manifest,
            &self.log,
            &self.engine_state,
            &self.endpoint_key,
        ]
    }
}

/// Validate a manifest-supplied engine name as a single filesystem path
/// component.
///
/// [`AppPaths::service_engine_state_path`] joins the engine name into a path
/// this module deletes, so an engine of `../../..` in a hand-edited manifest
/// would otherwise aim the removal outside `<data>/engines`. Reuses the same
/// rules as [`validate_service_id`] — `ServiceId` is the repo's single source of
/// truth for "safe as one path component", and nothing about those rules is
/// specific to service ids.
fn validate_engine_component(engine: &str) -> Result<()> {
    rocm_core::ServiceId::new(engine).with_context(|| {
        format!("managed service engine `{engine}` is not a safe path component")
    })?;
    Ok(())
}

fn service_record_artifacts(
    paths: &AppPaths,
    service_id: &str,
    engine: &str,
) -> Result<ServiceRecordArtifacts> {
    validate_service_id(service_id)?;
    validate_engine_component(engine)?;
    Ok(ServiceRecordArtifacts {
        manifest: paths.service_manifest_path(service_id),
        log: paths.service_log_path(service_id),
        engine_state: paths.service_engine_state_path(engine, service_id),
        endpoint_key: endpoint_keys::endpoint_key_file_path(paths, service_id),
    })
}

/// Delete every artifact that exists, collecting failures instead of stopping at
/// the first one.
///
/// A missing file is not an error: a record whose log was already deleted by
/// hand (the workaround this command replaces) must still be removable. An
/// unremovable one is returned as a message rather than propagated, because
/// `prune` walks many records and aborting mid-loop would throw away the
/// rendered plan and the audit event covering everything already deleted in the
/// same run — losing the record of a destructive action exactly when something
/// went wrong.
fn try_remove_service_record_artifacts(
    artifacts: &ServiceRecordArtifacts,
) -> (Vec<PathBuf>, Vec<String>) {
    let mut removed = Vec::new();
    let mut failures = Vec::new();
    for path in artifacts.paths() {
        match fs::remove_file(path) {
            Ok(()) => removed.push(path.to_path_buf()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => failures.push(format!("{}: {error}", path.display())),
        }
    }
    (removed, failures)
}

/// Delete every artifact that exists, returning the paths actually removed.
///
/// The strict form, for `remove`, which owns exactly one record and has no
/// partial progress to report: any file it could not delete is the command
/// failing. Every artifact is still attempted first, so a single stubborn file
/// does not strand the other three.
fn remove_service_record_artifacts(artifacts: &ServiceRecordArtifacts) -> Result<Vec<PathBuf>> {
    let (removed, failures) = try_remove_service_record_artifacts(artifacts);
    if !failures.is_empty() {
        bail!("failed to remove {}", failures.join("; "));
    }
    Ok(removed)
}

/// Whether a record is too alive to delete: the CLI reads it as live, *or* a
/// process it recorded is still running.
///
/// The status string alone is not enough for a destructive command.
/// `refresh_from_engine_state` adopts `failed` straight from the engine's own
/// state file, and `refresh_managed_service_runtime_liveness` then returns early
/// for any non-live status without ever consulting the recorded pids — so a
/// server whose engine reported failure while its process is still up reads as
/// removable, and deleting it takes the log and the 0600 endpoint key out from
/// under a process that may still be serving.
fn managed_service_record_is_in_use(record: &ManagedServiceRecord) -> bool {
    managed_service_is_live(record)
        || recorded_service_pids(record)
            .iter()
            .any(|pid| process_is_running(*pid))
}

/// Remove one non-running local server record and everything it owns.
///
/// Refuses a live record rather than stopping it: a removal that silently killed
/// a serving process would be a very different action from the one the user
/// asked for. Liveness is read from the record `load_managed_service` returns,
/// which has already been refreshed against the engine state file and the real
/// processes — a manifest still saying `ready` for a dead PID has demoted to
/// `stopped` by then, and must stay removable.
fn remove_managed_service_record(paths: &AppPaths, service_id: &str, yes: bool) -> Result<String> {
    validate_service_id(service_id)?;
    let record = load_managed_service(paths, service_id)?;
    // Checked before `--yes` on purpose: for a running server "stop it first" is
    // the actionable error, and repeating the command with --yes must not be the
    // advice a user takes away from it.
    if managed_service_record_is_in_use(&record) {
        bail!(
            "local server `{service_id}` is {} and cannot be removed while it is running.\n\nTry: {}",
            record.status,
            service_hints::service_stop_hint(service_id)
        );
    }
    if !yes {
        bail!(
            "Removing local server record `{service_id}` requires --yes.\n\nTry: {}",
            service_hints::service_action_hint("remove", service_id)
        );
    }

    let artifacts = service_record_artifacts(paths, &record.service_id, &record.engine)?;
    let mut output = String::new();
    // Printed *before* the delete, because after it there is nothing left to
    // point at: this is the last moment the log path is useful.
    let _ = writeln!(
        output,
        "Removing local server record `{service_id}` (status: {}).",
        record.status
    );
    let _ = writeln!(output, "  log: {}", artifacts.log.display());
    let _ = writeln!(
        output,
        "After this, `rocm services logs {service_id}` and `{}` no longer work.",
        service_hints::service_action_hint("restart", service_id)
    );
    let _ = writeln!(output);

    let removed = remove_service_record_artifacts(&artifacts)?;
    let mut report = cli_report::ActionReport::new("Local server record removed")
        .detail("service", &record.service_id)
        .detail("engine", &record.engine)
        .detail("files removed", removed.len());
    for path in &removed {
        report = report.detail("removed", path.display());
    }
    let _ = write!(output, "{}", report.render());
    Ok(output)
}

/// One record `rocm services prune` would delete.
#[derive(Debug, Clone)]
struct ServicePruneEntry {
    service_id: String,
    engine: String,
    status: String,
    artifacts: ServiceRecordArtifacts,
}

#[derive(Debug, Default)]
struct ServicePrunePlan {
    remove: Vec<ServicePruneEntry>,
    /// Files whose record is already gone — chiefly engine state under
    /// `<data>/engines/<engine>/state/`, which no removal path has ever swept.
    orphans: Vec<PathBuf>,
    /// Human-readable reasons, one per record or file left in place.
    skipped: Vec<String>,
    /// How many records were skipped purely because they are still running.
    skipped_live: usize,
    /// How many were removable in every respect but too recent. Counted apart
    /// from `skipped` so the summary can name the flag that includes them: a
    /// silent "nothing to do" on a host full of fresh failures reads as the
    /// command being broken.
    skipped_recent: usize,
}

/// Modification time of `path`, or `None` when it cannot be read.
fn path_modified(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

/// Age of a modification time relative to `now`. `None` when the time is missing
/// or in the future, which is treated as "too new to touch".
fn age_from_modified(modified: Option<SystemTime>, now: SystemTime) -> Option<Duration> {
    now.duration_since(modified?).ok()
}

/// Whether a modification time is old enough to prune, given a threshold.
///
/// Fail-closed: a time that cannot be determined is kept, except when the
/// threshold is zero (the explicit "everything that is not running" opt-out).
fn prunable_by_modified(modified: Option<SystemTime>, min_age: Duration, now: SystemTime) -> bool {
    if min_age.is_zero() {
        return true;
    }
    age_from_modified(modified, now).is_some_and(|age| age >= min_age)
}

/// Whether `path` is old enough to prune, given a threshold in hours.
///
/// Only sound for files nothing in this run rewrites. Service manifests are
/// rewritten by [`load_managed_services`], so their times are snapshotted up
/// front by [`service_manifest_modified_times`] and gated with
/// [`prunable_by_modified`] instead.
fn prunable_by_age(path: &Path, min_age: Duration, now: SystemTime) -> bool {
    prunable_by_modified(path_modified(path), min_age, now)
}

/// Modification times of every `*.json` in the services directory, taken before
/// anything in this run can rewrite them.
///
/// [`load_managed_services`] refreshes each record against the engine state and
/// the real processes, and persists the result whenever that changes the status
/// — which is exactly what happens the first time anything observes that a
/// `ready`/`running`/`starting` server has died. That rewrite lands *after* the
/// `now` the age gate compares against, so a manifest read afterwards looks
/// newer than the run itself and the fail-closed branch keeps it forever. A host
/// whose servers died weeks ago but were never listed since would see
/// `rocm services prune --yes` remove nothing while reporting those records as
/// too recent to touch.
fn service_manifest_modified_times(paths: &AppPaths) -> HashMap<PathBuf, SystemTime> {
    let mut times = HashMap::new();
    let Ok(entries) = fs::read_dir(paths.services_dir()) else {
        return times;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        if let Some(modified) = path_modified(&path) {
            times.insert(path, modified);
        }
    }
    times
}

/// The age threshold one `prune` invocation runs with, in hours.
///
/// `--any-age` is the discoverable spelling of `--older-than-hours 0` — clap
/// rejects the two together — so this only has to collapse the flag. It is a
/// function rather than a line inside the dispatch `match` so a test can drive
/// the real parsed arguments through the same mapping the command uses.
const fn service_prune_min_age_hours(older_than_hours: u64, any_age: bool) -> u64 {
    if any_age { 0 } else { older_than_hours }
}

fn describe_hours(hours: u64) -> String {
    if hours == 1 {
        "1 hour".to_owned()
    } else {
        format!("{hours} hours")
    }
}

/// Files in `services_dir` and the engine state dirs that no longer belong to
/// any record.
///
/// A `<id>.log` or `<id>.endpoint-key` counts as orphaned only when `<id>.json`
/// is *absent from disk* — not merely absent from `records`. An unparseable
/// manifest is skipped by `load_managed_services`, and treating its siblings as
/// orphans would quietly delete the log of the one record a user most needs to
/// investigate. The corrupt manifest itself is never removed for the same
/// reason.
///
/// "No `<id>.json` beside it" is also exactly what a *launch in progress* looks
/// like: `serve` writes the 0600 `<id>.endpoint-key` before
/// `spawn_managed_engine_child` (in `service_launch.rs`) writes the first
/// `<id>.json`. Nothing about
/// the file distinguishes the two cases — under `--any-age` there is no age gate
/// left to ask — so this function does not try. It is only ever reached with the
/// managed-launch lock held, which excludes that window outright; see
/// [`prune_managed_service_records`].
///
/// `services_dir` also holds `launch.lock`, which is shared by every managed
/// launch rather than owned by one service (see
/// [`AppPaths::managed_launch_lock_path`]). Only the three per-service
/// extensions are considered, so the lock is never a candidate — including the
/// one this sweep is itself running under.
fn collect_service_orphans(paths: &AppPaths, min_age: Duration, now: SystemTime) -> Vec<PathBuf> {
    let services_dir = paths.services_dir();
    let mut orphans = Vec::new();

    let push_if_orphaned = |path: &Path, orphans: &mut Vec<PathBuf>| {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            return;
        };
        if services_dir.join(format!("{stem}.json")).exists() {
            return;
        }
        if prunable_by_age(path, min_age, now) {
            orphans.push(path.to_path_buf());
        }
    };

    if let Ok(entries) = fs::read_dir(&services_dir) {
        for path in entries.flatten().map(|entry| entry.path()) {
            if matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("log" | "endpoint-key")
            ) {
                push_if_orphaned(&path, &mut orphans);
            }
        }
    }

    // `<data>/engines/<engine>/state/<id>.json`. `engines/plugins` has no
    // `state` subdirectory, so it never yields candidates.
    if let Ok(engines) = fs::read_dir(paths.data_dir.join("engines")) {
        for engine_dir in engines.flatten().map(|entry| entry.path()) {
            let Some(engine) = engine_dir.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Ok(states) = fs::read_dir(paths.engine_state_dir(engine)) else {
                continue;
            };
            for path in states.flatten().map(|entry| entry.path()) {
                if path.extension().and_then(|value| value.to_str()) == Some("json") {
                    push_if_orphaned(&path, &mut orphans);
                }
            }
        }
    }

    orphans.sort();
    orphans
}

fn build_service_prune_plan(
    paths: &AppPaths,
    min_age: Duration,
    hours: u64,
    now: SystemTime,
) -> Result<ServicePrunePlan> {
    let mut plan = ServicePrunePlan::default();
    // Taken first, because `load_managed_services` below rewrites the manifest of
    // every record whose status its refresh corrects — see
    // `service_manifest_modified_times` for why reading the time afterwards makes
    // the age gate keep exactly the long-dead records prune exists to remove.
    let manifest_times = service_manifest_modified_times(paths);
    // `load_managed_services` refreshes each record against the engine state and
    // the real processes before returning it, so liveness below is read from the
    // refreshed view, never from the manifest as it was on disk.
    for record in load_managed_services(paths)? {
        if managed_service_record_is_in_use(&record) {
            plan.skipped_live += 1;
            plan.skipped.push(format!(
                "{} is {} - stop it first with `{}`",
                record.service_id,
                record.status,
                service_hints::service_stop_hint(&record.service_id)
            ));
            continue;
        }
        let artifacts = match service_record_artifacts(paths, &record.service_id, &record.engine) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                plan.skipped.push(format!("{}: {error}", record.service_id));
                continue;
            }
        };
        if !prunable_by_modified(
            manifest_times.get(&artifacts.manifest).copied(),
            min_age,
            now,
        ) {
            plan.skipped_recent += 1;
            plan.skipped.push(format!(
                "{} changed less than {} ago",
                record.service_id,
                describe_hours(hours)
            ));
            continue;
        }
        plan.remove.push(ServicePruneEntry {
            service_id: record.service_id.clone(),
            engine: record.engine.clone(),
            status: record.status.clone(),
            artifacts,
        });
    }
    plan.orphans = collect_service_orphans(paths, min_age, now);
    plan.remove
        .sort_by(|left, right| left.service_id.cmp(&right.service_id));
    plan.skipped.sort();
    Ok(plan)
}

fn render_service_prune_plan(plan: &ServicePrunePlan, hours: u64, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Local server record review");
    let _ = writeln!(output);
    if hours == 0 {
        let _ = writeln!(
            output,
            "Including every record that is not running, however recent."
        );
    } else {
        let _ = writeln!(
            output,
            "Only records and files untouched for at least {}.",
            describe_hours(hours)
        );
    }
    let _ = writeln!(output);

    if plan.remove.is_empty() && plan.orphans.is_empty() {
        let _ = writeln!(output, "Nothing would be removed.");
    }
    if plan.skipped_recent > 0 {
        let _ = writeln!(
            output,
            "{} record(s) changed less than {} ago and are kept; add --any-age to include them.",
            plan.skipped_recent,
            describe_hours(hours)
        );
    }
    if !plan.remove.is_empty() {
        let _ = writeln!(
            output,
            "{} local server record(s) would be removed:",
            plan.remove.len()
        );
        for entry in &plan.remove {
            let _ = writeln!(
                output,
                "  - {} (status: {}, engine: {})",
                entry.service_id, entry.status, entry.engine
            );
            for path in entry.artifacts.paths() {
                if path.exists() {
                    let _ = writeln!(output, "      {}", path.display());
                }
            }
        }
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "Their logs go with them, and `rocm services restart <service-id> --yes` stops working for them."
        );
    }
    if !plan.orphans.is_empty() {
        if !plan.remove.is_empty() {
            let _ = writeln!(output);
        }
        let _ = writeln!(
            output,
            "{} leftover file(s) with no local server record would be removed:",
            plan.orphans.len()
        );
        for path in &plan.orphans {
            let _ = writeln!(output, "  - {}", path.display());
        }
    }
    if !plan.skipped.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Left alone:");
        for skipped in &plan.skipped {
            let _ = writeln!(output, "  - {skipped}");
        }
    }
    if dry_run {
        let _ = writeln!(output);
        // "Nothing was removed", not "nothing was changed": building the plan
        // loads every record, and loading corrects a status that no longer
        // matches the real processes and persists that correction. No file is
        // deleted, which is the promise a dry run of a removal command makes.
        let _ = writeln!(
            output,
            "Nothing was removed. Re-run without --dry-run to remove."
        );
    }
    output
}

#[derive(Debug, Default)]
struct ServicePruneOutcome {
    text: String,
    removed_records: usize,
    removed_files: usize,
    /// Records left in place purely because they are still running.
    skipped_live: usize,
    skipped_recent: usize,
    /// Paths the run could not delete, one message each. Collected rather than
    /// propagated so the plan still prints and the audit event is still
    /// recorded; the caller turns a non-empty list into a failing exit.
    failures: Vec<String>,
}

/// Carry out everything `plan` names, recording what happened in `outcome`.
///
/// Split out of [`prune_managed_service_records`] rather than inlined there so a
/// test can hand it a plan entry whose record came back to life after the plan
/// was built. That window is the only reason the liveness re-check below exists,
/// and no test driving `prune` end to end can open it: one call builds the plan
/// and applies it, so a record is either live for both halves or dead for both.
fn apply_service_prune_plan(
    paths: &AppPaths,
    plan: &ServicePrunePlan,
    outcome: &mut ServicePruneOutcome,
) {
    for entry in &plan.remove {
        // Liveness was snapshotted while the plan was built. A
        // `rocm services restart <id> --yes` landing in that window would have
        // its log and 0600 endpoint key deleted out from under a live process,
        // so re-read the record immediately before touching its files —
        // `remove` narrows the same window by loading the record it deletes.
        // Neither closes it: the check and the delete are not atomic either way.
        if load_managed_service(paths, &entry.service_id)
            .is_ok_and(|record| managed_service_record_is_in_use(&record))
        {
            outcome.skipped_live += 1;
            let _ = writeln!(
                outcome.text,
                "  {} started again while this ran and was left alone.",
                entry.service_id
            );
            continue;
        }
        let (removed, failures) = try_remove_service_record_artifacts(&entry.artifacts);
        outcome.removed_files += removed.len();
        if failures.is_empty() {
            outcome.removed_records += 1;
        }
        outcome.failures.extend(failures);
    }
    for path in &plan.orphans {
        match fs::remove_file(path) {
            Ok(()) => outcome.removed_files += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => outcome
                .failures
                .push(format!("{}: {error}", path.display())),
        }
    }
}

/// Bulk-remove the local server records that are no longer running.
///
/// Unlike [`remove_managed_service_record`] a live record is *skipped*, not an
/// error: a bulk cleanup that aborted because one server happened to be serving
/// would be unusable on the hosts that need it most. The count is reported so
/// the skip is never silent.
///
/// # Why this takes the managed-launch lock
///
/// [`collect_service_orphans`] calls a file with no `<id>.json` beside it a
/// leftover, and that is also what a launch looks like mid-flight: `serve`
/// writes the 0600 `<id>.endpoint-key` at the top of the managed path, and the
/// first `<id>.json` only lands at `record.write()` inside
/// `spawn_managed_engine_child` (in `service_launch.rs`). Between those two writes a live server's
/// secret is indistinguishable from a leftover, and `--any-age` removes the age
/// gate that used to hide the window, so a concurrent
/// `rocm services prune --any-age --yes` deleted it.
///
/// `serve` already holds `managed_launch_lock_path` across the whole of that
/// window — [`select_gpu_indices_under_launch_lock`] acquires it, hands the
/// guard back to its caller, and `start_managed_service`/`run_attached_service`
/// only drop it once the record is persisted (see that function's own comment,
/// which this one must not drift from). The gap was simply that prune never
/// acquired it: the helper above was the single acquirer in the tree. Taking it
/// here closes the window deterministically, for any number of records, which no
/// age floor can — the window is not short and is not bounded, because
/// `spawn_managed_engine_child`'s idempotency guard runs `load_managed_services`
/// first, and that refreshes every `ready`/`running` record through a 750ms
/// listing plus an up-to-8s inference probe ([`rocm_core::INFERENCE_PROBE_TIMEOUT`]),
/// sequentially, over a record count the user controls.
///
/// **Scope: the whole operation, not just the scan.** Holding it across
/// [`build_service_prune_plan`] alone is very nearly enough — a launch already
/// under way blocks prune until its manifest exists, at which point its key is
/// no longer an orphan, and a launch starting after the scan has not written a
/// key yet so it cannot be in the plan. "Cannot be in the plan" leans on
/// `generate_service_id` minting a millisecond-unique id, though, which is a
/// property of an unrelated function and a backwards clock step would break.
/// Covering [`apply_service_prune_plan`] too makes the exclusion unconditional,
/// and costs nothing worth measuring next to the scan it already serializes:
/// the apply phase is `unlink` calls plus one manifest read per record.
///
/// That wider scope is *not* pinned by a test, and neither the unit test nor
/// `service-cleanup-07` fails if the guard is released after plan building. It
/// cannot be pinned at that level: the only behaviour the extra span changes is
/// the id-collision case above, which needs `generate_service_id` to mint an id
/// a previous run already used — i.e. a backwards clock step — and any test that
/// instead tried to catch a launch slipping in between the two phases would be
/// racing a microsecond-wide window, so it would pass on a timing-lucky run
/// rather than flake. Treat the scope as a deliberately conservative choice
/// argued from the code, not as a property under regression cover: narrowing it
/// will not turn anything red.
///
/// **What it costs.** Prune's own plan building calls `load_managed_services`,
/// so prune can now delay a launch for as long as its own scan runs, and a
/// launch already under way delays prune for as long as *it* runs. Neither
/// direction is bounded: the per-record worst case is the same 8.75s, over a
/// record count the user controls, and `FileLock::acquire` has no timeout. The
/// usual wait is milliseconds, because those are ceilings on probes that hang
/// rather than costs every record pays — but "usually imperceptible" is the
/// honest claim, not "a few seconds". Because the wait has no upper bound, it is
/// announced on screen ([`cli_progress::AnimatedSpinner`]) instead of leaving
/// the command silent. `--dry-run` takes the lock as well: a preview that
/// disagreed with what `--yes` would do is worth less than the launch it blocks.
///
/// **What `--dry-run` writes.** `FileLock::acquire` creates the lock file and
/// any missing parents, and `services` only calls `AppPaths::discover`, never
/// `ensure` — so a preview on a host that has never served now creates
/// `<data>/services/` and an empty `launch.lock`, and fails outright where the
/// data directory is not writable. Both are accepted. The lock file is a 0-byte
/// rendezvous point inside the CLI's own data directory, in the directory
/// `rocm serve` creates there anyway. And a data directory the CLI cannot write
/// is one no managed server could have written a record into, so the preview
/// that now errors had nothing to report; where records *do* exist, deleting
/// them needs write permission on the very directory the lock needs, so a
/// preview that still succeeded would be predicting a `--yes` run that cannot
/// run — the same kind of disagreement the paragraph above declines to ship.
///
/// **No nested acquire.** `FileLock::acquire` blocks with no `try_` variant, so
/// a second acquire on a path already held by this process would hang forever.
/// Nothing under either phase acquires this path: the only other
/// `FileLock::acquire` calls in the tree are `ensure_background_helper_running_quiet`
/// and the managed-uv bootstrap in `rocm_core::uv`, each on a different lock file,
/// and neither is reachable from here. In the other
/// direction `serve` holds this lock but never runs prune, in-process or as a
/// subprocess.
fn prune_managed_service_records(
    paths: &AppPaths,
    hours: u64,
    dry_run: bool,
    yes: bool,
) -> Result<ServicePruneOutcome> {
    if !dry_run && !yes {
        bail!(
            "Removing local server records requires --yes.\n\nTry: rocm services prune --dry-run\nThen: rocm services prune --yes"
        );
    }
    // Held until this function returns, so no managed launch can be between its
    // key write and its record write while the sweep looks at the directory.
    //
    // The acquire blocks with no timeout, and the thing it blocks behind is a
    // launch whose own duration is unbounded — so it is announced rather than
    // left as a silent hang, in the command a user reaches for precisely when a
    // launch has gone wrong. `AnimatedSpinner` rather than the caller-driven
    // `Spinner` because there is no loop here to tick one: the wait is a single
    // blocking syscall, so only a background ticker can keep it moving. It is
    // TTY-gated and erased on drop, so an uncontended prune — the normal case,
    // where this costs microseconds — leaves nothing behind, and piped or
    // redirected output never sees it at all.
    let _launch_lock = {
        let _waiting =
            cli_progress::AnimatedSpinner::start("Waiting for a launch already under way…");
        rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?
    };
    let min_age = Duration::from_secs(hours.saturating_mul(3600));
    let plan = build_service_prune_plan(paths, min_age, hours, SystemTime::now())?;
    let mut outcome = ServicePruneOutcome {
        text: render_service_prune_plan(&plan, hours, dry_run),
        skipped_live: plan.skipped_live,
        skipped_recent: plan.skipped_recent,
        ..ServicePruneOutcome::default()
    };
    if dry_run {
        return Ok(outcome);
    }

    apply_service_prune_plan(paths, &plan, &mut outcome);

    let _ = writeln!(outcome.text);
    let _ = write!(
        outcome.text,
        "{}",
        cli_report::ActionReport::new("Local server records removed")
            .detail("records removed", outcome.removed_records)
            .detail("files removed", outcome.removed_files)
            .detail("still running, left alone", outcome.skipped_live)
            .detail("too recent, kept", outcome.skipped_recent)
            .render()
    );
    // A bulk cleanup that silently keeps things is indistinguishable from one
    // that found nothing, and the records most worth reading are exactly the
    // ones this keeps.
    if outcome.skipped_recent > 0 {
        let _ = writeln!(
            outcome.text,
            "  Those are recent enough to still be worth reading: `rocm services logs <id>`.\n  \
             Run `rocm services prune --any-age --yes` to remove them too."
        );
    }
    if !outcome.failures.is_empty() {
        let _ = writeln!(outcome.text);
        let _ = writeln!(
            outcome.text,
            "{} file(s) could not be removed:",
            outcome.failures.len()
        );
        for failure in &outcome.failures {
            let _ = writeln!(outcome.text, "  - {failure}");
        }
        let _ = writeln!(
            outcome.text,
            "Re-running is safe: everything already removed stays removed."
        );
    }
    Ok(outcome)
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
            let status = provider_keys::store_provider_credential(provider, &key)?;
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
            let status = provider_keys::remove_provider_credential(provider)?;
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
    // User-facing text, so it must agree with what `serve` would pick — the same
    // contradiction `examine` used to show on Instinct.
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or_else(|| freeform_planner::host_default_engine(Some(paths)));
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
    append_examine_engine_inventory(&mut output, paths, config, &summary.default_engine);
    Ok((output, summary))
}

fn render_examine_plain_header(summary: &ExamineSummary) -> String {
    let gpu = if summary.detected_gfx_target.is_some() {
        "AMD GPU detected"
    } else {
        "AMD GPU not detected yet"
    };
    // This line counted only CLI-managed runtimes, so a machine with ROCm
    // already installed was greeted with "No ROCm installs saved yet" -- read,
    // reasonably, as "nothing is installed". Now that the resolver finds those
    // installs and their versions, say what is actually on the machine, and that
    // a pre-existing install is left alone on purpose rather than missed.
    let managed = match summary.managed_runtime_count {
        0 => None,
        1 => Some("1 ROCm install saved".to_owned()),
        count => Some(format!("{count} ROCm installs saved")),
    };
    let existing = (summary.legacy_rocm.status != "not_detected").then(|| {
        summary.legacy_rocm.version.as_deref().map_or_else(
            || "existing ROCm install found, left unmanaged by design".to_owned(),
            |version| format!("existing ROCm {version} found, left unmanaged by design"),
        )
    });
    let runtime = match (managed, existing) {
        (Some(managed), Some(existing)) => format!("{managed}; {existing}"),
        (Some(managed), None) => managed,
        (None, Some(existing)) => format!("No ROCm installs saved yet; {existing}"),
        (None, None) => "No ROCm installs saved yet".to_owned(),
    };
    format!(
        "ROCm setup check\n  {gpu}\n  {runtime}\n  Driver: {}\n\nDetails\n",
        plain_status_label(&summary.driver.status)
    )
}

fn plain_status_label(status: &str) -> String {
    status.replace('_', " ")
}

/// Marker shown beside the engine `serve`/CLI commands default to. Every
/// place that renders this glyph MUST use this constant so the rendered
/// character and the legend text stay in sync.
const DEFAULT_ENGINE_MARKER: &str = "*";

/// Writes the legend line explaining [`DEFAULT_ENGINE_MARKER`]. Shared by both
/// engine-inventory renderers so the two copies cannot drift apart.
fn write_default_engine_legend(output: &mut String) {
    let _ = writeln!(output, "  legend: {DEFAULT_ENGINE_MARKER} = default engine");
}

fn render_engine_inventory_text_with_paths(paths: Option<&AppPaths>) -> String {
    // Mark the engine this GPU actually serves on as primary. Using the platform
    // constant put the `*` on Lemonade even on Instinct, where `serve` picks vLLM.
    let host_gpu = rocm_core::detect_host_gpu_summary(paths);
    let host_default_engine = rocm_core::default_engine_for_host(&host_gpu);
    // A configured `default_engine` still wins over the host preference, mirroring
    // `select_serve_engine` (and `append_examine_engine_inventory`). Without this,
    // the legend could mark an engine `serve` will not actually default to.
    let configured_default_engine = paths.and_then(|paths| {
        RocmCliConfig::load(paths)
            .ok()
            .and_then(|config| config.default_engine)
    });
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or the marker would land on an engine name
    // that is empty rather than falling back.
    let default_engine = configured_default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == default_engine);
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
    if any_marked {
        write_default_engine_legend(&mut output);
    }
    for (name, note) in engine_inventory() {
        let marker = if *name == default_engine {
            DEFAULT_ENGINE_MARKER
        } else {
            " "
        };
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
        // The same fact `rocm runtimes list` reports as `toolchain=`, for the
        // one runtime that is actually in use. `examine` is where a user looks
        // when a build fails on a missing `hipcc`, and without this line the
        // command that exists to answer "what is my ROCm state" could not say
        // whether the active runtime has a compiler at all.
        let _ = writeln!(
            output,
            "  active_runtime_toolchain: {}",
            toolchain_state_text(manifest.includes_devel())
        );
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        let _ = writeln!(output, "  active_runtime_mode: {mode}");
    }
    if let Some((setup_root, pip_cache_dir)) = setup_runtime_paths(config) {
        let _ = writeln!(output, "  setup_runtime_root: {}", setup_root.display());
        let _ = writeln!(
            output,
            "  setup_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
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

/// `host_default_engine` is the engine this GPU serves on, already resolved by
/// the caller from the `ExamineSummary` it holds — passed in rather than
/// re-probed so the two halves of one `rocm examine` run cannot disagree.
fn append_examine_engine_inventory(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
    host_default_engine: &str,
) {
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or `effective_default` below would become
    // an engine name that is empty rather than falling back.
    let configured_default = config
        .default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    // A configured value still wins, mirroring `select_serve_engine`. Only the
    // fallback becomes GPU-aware: it used to be the platform constant, which
    // reported Lemonade on Instinct where serve picks vLLM.
    let effective_default = configured_default.unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == effective_default);
    let _ = writeln!(output, "engine_inventory:");
    // Unchanged on purpose: this line means "what the user configured", so an
    // unset value must keep reading as unset rather than borrowing the host default.
    let _ = writeln!(
        output,
        "  configured_default_engine: {}",
        configured_default.unwrap_or("<platform default>")
    );
    let _ = writeln!(output, "  effective_default_engine: {effective_default}");
    // Say so when a configured engine is holding this host off the one its GPU
    // would pick. Installs provisioned before the installer stopped seeding
    // `default_engine` still carry `lemonade` on disk, and nothing distinguishes
    // that seed from a deliberate choice -- so point at the fix rather than
    // rewriting a file the user owns on a guess. This also catches a genuinely
    // stale choice made by hand, which no migration would have covered.
    if let Some(configured) = configured_default
        && configured != host_default_engine
    {
        let _ = writeln!(
            output,
            "  configured_default_note: overrides this host's {host_default_engine} preference; run `rocm config clear-default-engine` to follow the host"
        );
    }
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
    if any_marked {
        write_default_engine_legend(output);
    }
    for (engine, note) in engine_inventory() {
        let marker = if *engine == effective_default {
            DEFAULT_ENGINE_MARKER
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

/// The per-hardware-group heading in `rocm model`, carrying the "this is the
/// box you are on" marker when the group matches the detected host GPU.
///
/// Split out of [`render_model_registry_text_with_context_and_host`] only so
/// the marked branch is reachable from a test. Taking it in situ needs a host
/// whose real gfx target matches a catalog platform, which the default no-GPU
/// CI lane never provides — leaving the marker (`←` before ROCMAI-543) as the
/// one ASCII conversion on this surface that nothing guards.
fn model_catalog_platform_heading(label: &str, your_gpu: bool) -> String {
    if your_gpu {
        format!("{label} <- your GPU")
    } else {
        label.to_owned()
    }
}

#[allow(dead_code)]
pub(crate) fn render_model_registry_text_with_context_and_host(
    _paths: Option<&AppPaths>,
    aggregate_gpu_vram_gib: Option<f64>,
    _host_ram_gib: Option<f64>,
) -> String {
    let mut output = String::new();
    let registry = match load_model_recipe_registry() {
        Ok(registry) => registry,
        Err(error) => {
            let _ = writeln!(output, "Recommended models unavailable: {error}");
            return output;
        }
    };
    let _ = writeln!(output, "{}", model_catalog_header(&registry));
    // Curated view: for the built-in catalog show only the featured short list;
    // a configured index is already curated, so show all of it. Group either way
    // by the hardware path each model targets.
    let show_all = !matches!(registry.source, ModelRecipeRegistrySource::BuiltIn);
    let visible = registry
        .recipes
        .iter()
        .filter(|recipe| show_all || model_recipe_featured(recipe))
        .collect::<Vec<_>>();
    let platforms = model_catalog_platforms(&registry);
    let host_gfx_family = detect_host_gfx_target()
        .as_deref()
        .and_then(normalize_therock_family);
    for platform in &platforms {
        let mut group = visible
            .iter()
            .filter(|recipe| {
                recipe.preferred_engines.first().is_some_and(|engine| {
                    platform
                        .engines
                        .iter()
                        .any(|e| e.eq_ignore_ascii_case(engine))
                })
            })
            .peekable();
        if group.peek().is_none() {
            continue;
        }
        let your_gpu = host_gfx_family
            .as_deref()
            .is_some_and(|family| platform_matches_gfx_family(platform, family));
        let heading = model_catalog_platform_heading(&platform.label, your_gpu);
        let _ = writeln!(output, "\n{heading}");
        for recipe in group {
            // Show the canonical Hugging Face id (the reliable serve target) and
            // the quant that fits this hardware. Append a fit verdict only when the
            // host GPU VRAM is known — otherwise every row would read "GPU fit
            // unknown", which is noise.
            let detail = recipe
                .quantization
                .clone()
                .unwrap_or_else(|| model_recipe_memory_label(recipe));
            let fit = if aggregate_gpu_vram_gib.is_some() {
                format!(
                    "  {}",
                    model_recipe_gpu_fit_label(recipe, aggregate_gpu_vram_gib)
                )
            } else {
                String::new()
            };
            let _ = writeln!(output, "  {}  {}{}", recipe.canonical_model_id, detail, fit);
        }
    }
    if matches!(registry.source, ModelRecipeRegistrySource::BuiltIn) {
        let _ = writeln!(
            output,
            "\nThese are recommendations - you can serve any compatible Hugging Face model:\n  \
             rocm serve <owner/repo>          # vLLM, e.g. Qwen/Qwen3.6-27B\n  \
             rocm serve <owner/repo>:<quant>  # Lemonade GGUF, e.g. unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M\n\
             \nUse `rocm model --verbose` for details."
        );
    } else {
        let _ = writeln!(output, "\nUse `rocm model --verbose` for details.");
    }
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
        let hidden_from_builtin = matches!(registry.source, ModelRecipeRegistrySource::BuiltIn)
            && !model_recipe_featured(recipe);
        if hidden_from_builtin {
            let _ = writeln!(
                output,
                "      catalog: hidden (resolvable via rocm serve, not in the curated list)"
            );
        } else {
            let platforms = model_catalog_platforms(&registry);
            let _ = writeln!(
                output,
                "      catalog: {}",
                model_recipe_target_platform_label(recipe, &platforms)
            );
        }
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

/// The `rocm model` header. For the default built-in list it just names the
/// action; a configured recipe index instead advertises its provenance (the
/// only case where the source differs from the default and is worth surfacing).
fn model_catalog_header(registry: &ModelRecipeRegistry) -> String {
    match &registry.source {
        ModelRecipeRegistrySource::BuiltIn => {
            "Recommended models - run one with `rocm serve <model>`".to_owned()
        }
        ModelRecipeRegistrySource::SignedIndex { index_path, .. } => {
            format!(
                "Recommended models - from recipe index {}",
                index_path.display()
            )
        }
    }
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

/// Human-readable size, for `rocm storage`'s cache report and the automations
/// proposal history. Both report values routinely in the tens of megabytes,
/// so the intermediate tiers matter: a bare byte count is unreadable there.
fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    // `{:.1}` rounds, so from 1023.95 of a unit upwards the text already reads
    // as a full 1024 of it. Step up there rather than printing "1024.0 KiB".
    const ROUNDS_UP_TO_NEXT_UNIT: f64 = 1023.95;

    let value = bytes as f64;
    if value < KIB {
        return format!("{bytes} bytes");
    }
    let kib = value / KIB;
    if kib < ROUNDS_UP_TO_NEXT_UNIT {
        return format!("{kib:.1} KiB");
    }
    let mib = kib / KIB;
    if mib < ROUNDS_UP_TO_NEXT_UNIT {
        return format!("{mib:.1} MiB");
    }
    format!("{:.1} GiB", mib / KIB)
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
                // Not `/examine`: an `Examination` is a static host snapshot and
                // carries no VRAM figure at all, so sending the user there for a
                // missing VRAM reading is a confident dead end. But naming
                // `amd-smi metric --json` directly is just as much of one here:
                // every production caller of this function hardcodes
                // `aggregate_gpu_vram_gib = None`, so the reading this function
                // sees never changes no matter what the user runs. `rocm
                // diagnose --model` is the command that actually threads a live
                // reading through (`host_accelerator_memory`), so it is pointed
                // at instead of a command this one cannot act on.
                let _ = writeln!(
                    output,
                    "      action: run `rocm diagnose --model {}` for a live reading",
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

/// Curated models to suggest in place of one that does not fit, for `/model`.
///
/// The *selection policy* — declared alternatives first, else same-task curated
/// recipes, capped — lives in `rocm_core::model_readiness::curated_alternatives`
/// and is shared with `rocm diagnose --model`, so the two commands cannot come
/// to answer "what should I run instead" differently. What stays local is the
/// predicate and the wording: `/model` asks only whether the VRAM minimum is
/// met, while `diagnose --model` asks the whole readiness question, and each is
/// right for its own report.
#[allow(dead_code)]
fn manual_alternative_recommendations(
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) -> Vec<String> {
    let catalog = builtin_model_recipes();
    rocm_core::model_readiness::curated_alternatives(Some(recipe), &catalog, &|candidate| {
        recipe_is_manual_fit(candidate, aggregate_gpu_vram_gib)
    })
    .into_iter()
    .map(|(candidate_ref, candidate)| {
        // A declared alternative is shown with its requirement, a fallback with
        // its name alone. The two branches are exclusive -- a fallback only runs
        // when no declared candidate survived the predicate, and a candidate
        // that failed it in one branch fails it in the other -- so testing the
        // declared list here recovers exactly which branch produced this pick.
        if recipe
            .manual_alternatives
            .iter()
            .any(|declared| declared == candidate_ref)
        {
            format!(
                "{} ({})",
                candidate_ref,
                candidate.min_gpu_mem_gb.map_or_else(
                    || "CPU-only".to_owned(),
                    |value| format!("{} min GPU", format_gib(f64::from(value)))
                )
            )
        } else {
            candidate_ref.to_owned()
        }
    })
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
    if engine_ruled_out_by_platform(engine) {
        Some(
            "runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm; gpu_execution_required=true; run /engine for adapter details",
        )
    } else {
        None
    }
}

/// One definition, shared with `rocm diagnose --model`: a model named in a
/// suggestion has to be a string the user can paste back into `rocm serve`, and
/// two answers to "what do I call this recipe" is one too many.
#[allow(dead_code)]
fn recipe_display_ref(recipe: &ModelRecipeRecord) -> &str {
    rocm_core::model_readiness::recipe_display_ref(recipe)
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
    if engine.eq_ignore_ascii_case("vllm")
        && (lower.contains("not installed") || lower.contains("command was not found"))
    {
        return Some("vLLM is not installed in a Linux/WSL ROCm Python environment.".to_owned());
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
    if !rocm_core::runtime_is_windows() || !engine.eq_ignore_ascii_case("vllm") {
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

/// Tell the default (`!all`) view about local server records it is hiding.
///
/// The default view lists only live servers, so a host whose serves all failed
/// saw an empty list with no hint that anything was recorded. Name the command
/// that lists them, the one that reads a specific log (with a real id, so that
/// line can be pasted as-is), and the one that deletes them.
///
/// `prune` is named last and without qualification on purpose. Every record
/// carries an unrotated engine log, so on a host that has served real models
/// this block is pointing at the only thing on disk the user can get back — and
/// the other two lines only ever let them *look*. Nothing is said here about
/// how `prune` decides what to remove: that is its own command's output, and
/// its behaviour is under change on a sibling branch.
fn write_past_attempts_hint(output: &mut String, past_attempts: usize, newest_id: Option<&str>) {
    if past_attempts == 0 {
        return;
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Past attempts: {past_attempts} local server record(s) that are no longer running."
    );
    let _ = writeln!(output, "  See them: rocm services list --all");
    if let Some(id) = newest_id {
        let _ = writeln!(output, "  Read the newest: rocm services logs {id}");
    }
    let _ = writeln!(output, "  Reclaim the space: rocm services prune");
}

pub(crate) fn render_services_text(paths: &AppPaths, all: bool) -> Result<String> {
    // Counts describe the WHOLE registry, not the rows this view keeps. The
    // default view hides records that are not live, and counting the filtered
    // list made the header say "none ready" on a host whose only local servers
    // had failed - the records were on disk and nothing mentioned them. Rows
    // stay filtered; only the header and the hint below see everything.
    let all_records = load_managed_services(paths)?;
    let counts = managed_service_sidebar_counts(&all_records);
    // `load_managed_services` sorts newest-first, so the first record that is
    // not live is the newest one - the id worth putting in a pasteable hint.
    let newest_past_attempt = all_records
        .iter()
        .find(|record| !managed_service_is_live(record))
        .map(|record| record.service_id.clone());
    let records = all_records
        .into_iter()
        .filter(|record| all || managed_service_is_live(record))
        .collect::<Vec<_>>();
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
        // Gate the separator on the same condition as the hint, not just on
        // `!all`: `write_past_attempts_hint` early-returns at zero, so a host
        // that has never served would otherwise gain a blank line between "No
        // local servers are running." and the next line - a visible change to
        // the one output this PR is not meant to touch.
        if !all && counts.past_attempts > 0 {
            write_past_attempts_hint(
                &mut output,
                counts.past_attempts,
                newest_past_attempt.as_deref(),
            );
            let _ = writeln!(output);
        }
        let _ = writeln!(
            output,
            "Start one with `rocm serve <model> --managed`, or run `rocm` and choose Serve."
        );
        return Ok(output);
    }

    let docs_links = api_docs_link::service_docs_links(
        paths,
        records
            .iter()
            .filter(|record| managed_service_is_live(record))
            .map(|record| record.service_id.as_str()),
    );
    let _ = writeln!(output, "Servers");
    for record in records {
        let _ = writeln!(output, "- {}", record.service_id);
        let _ = writeln!(output, "  status: {}", record.status);
        let _ = writeln!(output, "  engine: {}", record.engine);
        let _ = writeln!(output, "  model: {}", record.model_ref);
        let _ = writeln!(output, "  endpoint: {}", record.endpoint_url);
        if let Some((_, link)) = docs_links
            .iter()
            .find(|(service_id, _)| *service_id == record.service_id)
        {
            let _ = writeln!(output, "  docs: {link}");
        }
        let _ = writeln!(output, "  logs: rocm services logs {}", record.service_id);
        if matches!(
            record.status.as_str(),
            "ready" | "running" | "starting" | "recovering"
        ) {
            let _ = writeln!(
                output,
                "  stop: {}",
                service_hints::service_stop_hint(&record.service_id)
            );
        } else {
            let _ = writeln!(
                output,
                "  restart: {}",
                service_hints::service_action_hint("restart", &record.service_id)
            );
        }
    }
    // A host can have one server ready and three failed, so the populated
    // branch needs the hint just as much as the empty one.
    if !all {
        write_past_attempts_hint(
            &mut output,
            counts.past_attempts,
            newest_past_attempt.as_deref(),
        );
    }
    Ok(output)
}

/// The machine-readable counterpart to [`render_services_text`].
///
/// Applies the same liveness filter as the text form so `--json` and the table
/// agree on which services they consider current — the two must not disagree
/// about what is running. Emits the `ManagedServiceRecord`s verbatim rather than
/// a bespoke projection: `rocm remote` deserializes them back into the same type
/// on the other side of its control channel, so any field this dropped would be
/// a field the remote orchestration could never see.
pub(crate) fn render_services_json(paths: &AppPaths, all: bool) -> Result<String> {
    let records = load_managed_services(paths)?
        .into_iter()
        .filter(|record| all || managed_service_is_live(record))
        .collect::<Vec<_>>();
    let mut output = serde_json::to_string_pretty(&records)
        .context("failed to serialize the managed service records as JSON")?;
    output.push('\n');
    Ok(output)
}

pub(crate) fn render_service_logs_text(paths: &AppPaths, service_id: &str) -> Result<String> {
    render_service_logs_text_with_options(paths, service_id, true)
}

fn render_service_logs_text_with_options(
    paths: &AppPaths,
    service_id: &str,
    show_file_locations: bool,
) -> Result<String> {
    let record = load_managed_service(paths, service_id)?;
    let recent_lines =
        read_optional_tail_lines(&record.log_path, DEFAULT_LOG_TAIL_LINES, "service log");

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

/// A managed service's record exactly as it is on disk, without the engine-state
/// and liveness refresh [`load_managed_service`] applies (and may persist).
fn read_managed_service_manifest(
    paths: &AppPaths,
    service_id: &str,
) -> Result<(ManagedServiceRecord, PathBuf)> {
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
    Ok((record, manifest_path))
}

fn load_managed_service(paths: &AppPaths, service_id: &str) -> Result<ManagedServiceRecord> {
    let (mut record, manifest_path) = read_managed_service_manifest(paths, service_id)?;
    let refreshed_from_engine = record.refresh_from_engine_state().unwrap_or(false);
    let refreshed_liveness = refresh_managed_service_runtime_liveness(paths, &mut record);
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
            let service = restart_internal_managed_service(paths, &service_id, None)?;
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

/// How long a managed stop waits for each recorded process to actually exit
/// after `SIGTERM` before escalating to `SIGKILL` and then reporting a timeout.
const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// Terminate the processes recorded for a managed service, verifying each PID's
/// identity first so a recycled PID never causes an unrelated process tree to be
/// killed.
///
/// The supervisor (launcher) and the engine server are distinct processes, so
/// each PID is paired with **its own** start-time token (`supervisor_start_ticks`
/// vs `engine_start_ticks`). When they resolve to the same PID (e.g. before the
/// engine state has been refreshed) the entry is de-duplicated, preferring a
/// known token. Returns the PIDs actually signalled and whether every recorded
/// process is confirmed no longer running.
fn terminate_recorded_service_pids(record: &ManagedServiceRecord) -> (Vec<u32>, bool) {
    // Build the (pid, own-token) work list, de-duplicating on PID and preferring
    // an entry that carries a verifiable start-time.
    let mut entries: Vec<(u32, Option<u64>)> = Vec::new();
    for (pid, ticks) in [
        (Some(record.supervisor_pid), record.supervisor_start_ticks),
        (record.engine_pid, record.engine_start_ticks),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0 && *pid != std::process::id()) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _)| *seen == pid) {
            if existing.1.is_none() {
                existing.1 = ticks;
            }
        } else {
            entries.push((pid, ticks));
        }
    }

    let mut signaled_pids = Vec::new();
    let mut all_stopped = true;
    for (pid, ticks) in entries {
        let identity = rocm_core::ProcessIdentity::new(pid, ticks);
        // The managed stop contract is definitive: after the adapter gets its
        // graceful opportunity, this cleanup pass forces any verified survivor
        // so the command does not report success while a GPU worker remains.
        let outcome = rocm_core::terminate_verified(
            &identity,
            rocm_core::KillScope::Tree,
            MANAGED_STOP_GRACE,
            true,
        );
        if !outcome.stopped() {
            all_stopped = false;
        }
        // Report every PID that received a signal. TimedOut belongs here because
        // signalling was attempted even though the process was not confirmed
        // stopped; `all_stopped` separately carries the truthful completion state.
        if matches!(
            outcome,
            rocm_core::TerminationOutcome::Graceful
                | rocm_core::TerminationOutcome::Forced
                | rocm_core::TerminationOutcome::TimedOut
        ) {
            signaled_pids.push(pid);
        }
    }
    (signaled_pids, all_stopped)
}

fn stop_internal_managed_service(paths: &AppPaths, service_id: &str) -> Result<serde_json::Value> {
    stop_internal_managed_service_with(paths, service_id, terminate_recorded_service_pids)
}

/// [`stop_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller. Production passes
/// [`terminate_recorded_service_pids`]; a test passes a stand-in that observes
/// what is on disk at the moment the processes would be signalled. Everything
/// else — the engine stop, the manifest writes, the key cleanup — runs for real.
fn stop_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<serde_json::Value> {
    stop_internal_managed_service_confirmed_with(paths, service_id, terminate)
        .map(|(report, _)| report)
}

/// [`stop_internal_managed_service_with`], also returning whether the stop is
/// confirmed: every process it handled is gone and the record still names
/// them. The report's `status` cannot carry that on its own — a record that
/// already read `stopped` keeps that status through a stop that confirmed
/// nothing.
fn stop_internal_managed_service_confirmed_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<(serde_json::Value, bool)> {
    let mut stopping = load_managed_service(paths, service_id)?;
    // Record the request on disk before anything is stopped, not only after.
    // From the engine `Stop` below until the last recorded process is confirmed
    // gone, the record still reads `ready`/`running` while its endpoint stops
    // answering — exactly what the daemon's `server-recover` watcher restarts.
    // That watcher skips a record carrying this marker, so it has to be on disk
    // for the whole stop. Same order as `rocmd`'s `stop_managed_service_with`.
    let requested_at = rocm_core::unix_time_millis();
    stopping.stop_requested_unix_ms = Some(requested_at);
    stopping.write()?;
    let engine_stop = if stopping.engine == "lemonade" {
        unload_lemonade_service_model(&stopping).map(|()| StopResponse {
            stopped: true,
            graceful: true,
        })
    } else {
        engine_request::<_, StopResponse>(
            Some(paths),
            &stopping.engine,
            EngineMethod::Stop,
            &StopRequest {
                service_id: stopping.service_id.clone(),
                force: true,
            },
        )
    };
    let (signaled_pids, terminated_all) = terminate(&stopping);
    // Re-read the record rather than write back the copy loaded before the
    // stop, which can be some forty seconds old by now: `write` replaces the
    // whole file, and a supervisor, the daemon's restart or another stop may
    // have written in the meantime. Only this stop's own changes are applied to
    // what is there now — the same discipline as `rocmd`'s
    // `stop_managed_service_with`. Read as-is: `load_managed_service`'s
    // liveness refresh would act on this stop's own marker before the verdict
    // below has been reached.
    let (mut record, _) = read_managed_service_manifest(paths, service_id)?;
    // Only claim "stopped" when every recorded process is confirmed gone, and
    // only for the processes this stop handled. If the record names different
    // ones now — a restart recorded its new supervisor while the stop ran —
    // the verdict says nothing about them, and claiming the service stopped or
    // dropping the key it serves with would both be false.
    //
    // When a stop cannot confirm termination (rare: SIGKILL-resistant or
    // unverifiable PID), leave the status so the standard liveness refresh
    // reconciles it to "stopped" once the process actually dies — rather than
    // asserting a stop that did not happen.
    let all_stopped = terminated_all && stopping.names_same_processes(&record);
    if all_stopped {
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = None;
    } else {
        // The marker records that a stop was *asked for* even though it could
        // not be confirmed. `refresh_managed_service_runtime_liveness` needs it
        // to tell a service the operator stopped from one that merely crashed:
        // only the former should lose its endpoint key once its processes are
        // gone. Re-stamped, because a restart that ran meanwhile clears it.
        record.stop_requested_unix_ms = Some(requested_at);
    }
    record.write()?;
    // Drop the endpoint key with the service by deleting its 0600 key file.
    // Best-effort — a stopped service must not fail to stop just because key
    // cleanup did.
    //
    // Gated on `all_stopped` for the same reason the status is: an unconfirmed
    // stop may have left the engine alive and still enforcing the key, and
    // discarding our only copy would lock the CLI's own probes, chat, and
    // service discovery out of a service that is otherwise fine. The marker
    // written above hands that cleanup to the liveness refresh instead.
    if all_stopped {
        endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
    }
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
    let report = serde_json::json!({
        "service_id": service_id,
        "status": record.status,
        "engine_stop": engine_stop,
        "signaled_pids": signaled_pids,
    });
    Ok((report, all_stopped))
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

/// Re-apply a caller's runtime pin to a record that was just loaded for a
/// restart, in the same shape `pin_service_record_to_runtime` writes it: the
/// exact runtime key, and no env pin.
///
/// Split out from [`restart_internal_managed_service`] so the pin can be
/// asserted without launching an engine — see
/// `pinning_the_record_is_what_puts_the_new_runtime_in_the_restart_argv`.
fn apply_restart_runtime_pin(record: &mut ManagedServiceRecord, runtime_pin: Option<&str>) {
    if let Some(runtime_key) = runtime_pin {
        record.runtime_id = Some(runtime_key.to_owned());
        record.env_id = None;
    }
}

/// Load the record a restart relaunches from, with the caller's runtime pin
/// applied after the load's own refresh.
///
/// Everything a restart can say about WHICH runtime the child comes back on is
/// decided here and in [`restart_child_argv`], and neither spawns anything, so
/// `restart_internal_managed_service` is only the stop, the spawn and the
/// readiness wait around them. That split is what lets a test reach the part of
/// a restart that matters for a move without an engine: dropping the pin from
/// this function, or from the caller that supplies it, changes the argv a test
/// reads off [`restart_child_argv`].
fn load_service_for_restart(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
) -> Result<ManagedServiceRecord> {
    let mut record = load_managed_service(paths, service_id)?;
    apply_restart_runtime_pin(&mut record, runtime_pin);
    Ok(record)
}

/// The argv a restarted child is launched with, read entirely off the record.
///
/// Every field comes from the record rather than from a parameter on purpose: a
/// restart's runtime is whatever the record says after
/// [`load_service_for_restart`], and an input here that bypassed it would let a
/// test assert a runtime the restart never uses.
fn restart_child_argv(record: &ManagedServiceRecord) -> Result<Vec<String>> {
    let policy = parse_device_policy(record.device_policy.as_deref())?;
    let recipe = parse_engine_recipe_json_arg(record.engine_recipe_json.clone())?;
    builtin_engine_serve_http_args(
        &record.engine,
        &record.service_id,
        &record.canonical_model_id,
        &record.host,
        record.port,
        &policy,
        &record.gpu_indices,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
        recipe.as_ref(),
        &record.engine_state_path,
        Some(&record.log_path),
    )
}

/// Stop a managed service and bring it back up.
///
/// `runtime_pin` is for callers whose whole reason for restarting is to MOVE the
/// service — `runtime_services::restart_service_onto_runtime`. Passing the key
/// here is not a convenience over writing the record first: `load_managed_service`
/// below refreshes the record from the engine state file written by the process
/// being replaced, which re-adopts THAT process's `runtime_id` and `env_id`. For
/// a plain restart that is correct, since the record should describe what is
/// actually running. For a move it silently undoes the re-pin — the previous
/// `env_id` comes back, and `builtin_engine_serve_http_args` omits `--runtime-id`
/// whenever `env_id` is set, so the child would resolve a runtime for itself and
/// come back up on the one it was already using while the record claimed the new
/// key. `None` leaves the refreshed record exactly as it was.
fn restart_internal_managed_service(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
) -> Result<ManagedServiceRecord> {
    restart_internal_managed_service_with(
        paths,
        service_id,
        runtime_pin,
        terminate_recorded_service_pids,
    )
}

/// [`restart_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller — see [`stop_internal_managed_service_with`].
fn restart_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<ManagedServiceRecord> {
    let record = load_service_for_restart(paths, service_id, runtime_pin)?;
    // Preserve the endpoint key across the restart: `stop_internal_managed_service`
    // deletes the key file (correct on a real stop), but a restart must bring the
    // service back on the same public host with the same auth. Capture it first and
    // re-store it after the stop so the spawn below hands the child the same key.
    let preserved_endpoint_key = endpoint_keys::endpoint_api_key(paths, service_id);
    // Checked before the stop, so a refused restart leaves a running service
    // running instead of stopping it and then failing to bring it back.
    ensure_public_service_has_endpoint_key(
        &record.host,
        preserved_endpoint_key.is_some(),
        record.requires_api_key,
    )?;
    let (stop_report, stopped) =
        stop_internal_managed_service_confirmed_with(paths, service_id, terminate)
            .with_context(|| format!("failed to stop `{service_id}` before restarting it"))?;
    // Starting a new engine is only safe once the old one is confirmed gone: a
    // survivor still holds the port and the GPU, so the new child would fail to
    // bind or contend for the device, and recording it would hide the survivor
    // from the next stop. The unconfirmed stop has already re-stamped its
    // marker and kept the endpoint key, so the liveness refresh finishes the
    // stop once the process does exit.
    if !stopped {
        bail!(
            "did not restart `{service_id}`: its stop could not confirm that every recorded \
             process exited (signalled pids: {}), so no new engine was started; the stop \
             request stays recorded on the service",
            stop_report["signaled_pids"]
        );
    }
    if let Some(key) = preserved_endpoint_key.as_deref() {
        endpoint_keys::store_endpoint_api_key(paths, service_id, key)?;
    }
    // What the confirmed stop left on disk. Every later write is applied to the
    // record as it is then, guarded on it still being this — never a write-back
    // of the copy loaded before the stop, which would erase a stop marker or
    // anything else written while this restart ran. The confirmed stop cleared
    // its own marker, so a restart supersedes every stop asked for before it;
    // one asked for since is someone else's, and none of the writes below
    // touches it.
    let (stopped_record, _) = read_managed_service_manifest(paths, service_id)?;
    // Still after the stop, like the device-policy refusal inside it always was:
    // the callers' "stopped by the attempt" report and pin restore both cover a
    // restart that gets this far and fails.
    let serve_args = restart_child_argv(&record)?;
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
    let engine_envs_root = env_root_for_service(
        paths,
        &record.engine,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
    )?;
    // A restarted public service comes back authenticated: its 0600 key file
    // persists across restarts (removed only on stop), so hand the child the same
    // file and reuse the key for the readiness probe.
    let endpoint_key_file = endpoint_keys::endpoint_key_file_if_present(paths, &record.service_id);
    let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
    #[cfg(windows)]
    let child_pid = {
        let env_values = app_path_env_var_values(paths, engine_envs_root.as_deref());
        let mut env_refs = app_path_env_var_refs(&env_values);
        if let Some(key_file) = endpoint_key_file.as_deref() {
            env_refs.push((rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file));
        }
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
        if let Some(key_file) = endpoint_key_file.as_deref() {
            command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        }
        let mut child = command
            .spawn()
            .context("failed to restart managed engine process")?;
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check restarted engine startup state")?
        {
            update_restarting_record(paths, &stopped_record, |current| {
                current.status = "failed".to_owned();
            })?;
            bail!(
                "{}",
                managed_engine_startup_failure_detail(status, &record.log_path)
            );
        }
        child.id()
    };
    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));
    // Recorded before the readiness wait, so a stop issued during it finds the
    // new child rather than the processes already stopped.
    let started = record_restarted_child(paths, &stopped_record, runtime_pin, child_pid)?;
    let readiness = wait_for_service_http_ready(
        &started.engine,
        &started.host,
        started.port,
        &started.canonical_model_id,
        endpoint_api_key.as_deref(),
        Duration::from_secs(45),
    );
    record_restart_readiness(paths, &started, readiness)
}

/// Apply one of a restart's own changes to the record as it is on disk now,
/// rather than writing back a copy the restart holds. `write` replaces the
/// whole file, and a stop records `stop_requested_unix_ms` before it signals
/// anything — the same hazard `rocmd`'s `update_supervised_record` guards
/// against for the supervisor's writes.
///
/// Nothing is written, and `None` returned, when the record no longer reads
/// as `expected` left it — it is gone, names other processes, or has another
/// status: whoever wrote since holds a truer view than this restart. `update`
/// never touches the stop marker, so a stop asked for meanwhile survives.
fn update_restarting_record(
    paths: &AppPaths,
    expected: &ManagedServiceRecord,
    update: impl FnOnce(&mut ManagedServiceRecord),
) -> Result<Option<ManagedServiceRecord>> {
    if !paths.service_manifest_path(&expected.service_id).exists() {
        return Ok(None);
    }
    let (mut current, _) = read_managed_service_manifest(paths, &expected.service_id)?;
    if !current.names_same_processes(expected) || current.status != expected.status {
        return Ok(None);
    }
    update(&mut current);
    current.write()?;
    Ok(Some(current))
}

/// Record the child a restart just started, over the record its confirmed
/// stop left (`stopped`). Fails when that record has changed since: the child
/// is then running but unrecorded, and the caller has to hear that rather than
/// a claim that the service restarted.
///
/// `runtime_pin` is re-applied here because the record on disk does not carry
/// it: the restart's own load refreshed the record from the outgoing process's
/// engine state and persisted that, re-adopting the previous `runtime_id` and
/// `env_id` over the pin `runtime_services::pin_service_record_to_runtime`
/// wrote. The child was launched with the pin (see [`restart_child_argv`]), so
/// the record that names it has to say the same runtime.
fn record_restarted_child(
    paths: &AppPaths,
    stopped: &ManagedServiceRecord,
    runtime_pin: Option<&str>,
    child_pid: u32,
) -> Result<ManagedServiceRecord> {
    update_restarting_record(paths, stopped, |current| {
        apply_restart_runtime_pin(current, runtime_pin);
        current.status = "running".to_owned();
        // Both roles move to the restarted child, each with its own token —
        // see `spawn_managed_engine_child`.
        current.record_supervisor_identity(child_pid);
        current.record_engine_identity(child_pid);
        // Counts the restart and drops the previous run's inference
        // verification — the new child has an unloaded model, so the old
        // verdict says nothing about it.
        current.reset_for_restart();
    })?
    .with_context(|| {
        format!(
            "service `{}` changed on disk while it was restarting, so the restarted engine \
             (pid {child_pid}) was not recorded on it",
            stopped.service_id
        )
    })
}

/// The restart's last write: what the readiness wait found. Applied only while
/// the record still reads as [`record_restarted_child`] left it; otherwise a
/// stop or another restart has written since, and the record as it is now is
/// returned untouched.
fn record_restart_readiness(
    paths: &AppPaths,
    started: &ManagedServiceRecord,
    readiness: EndpointReadiness,
) -> Result<ManagedServiceRecord> {
    let updated = update_restarting_record(paths, started, |current| {
        current.status = status_for_readiness(readiness).to_owned();
        if readiness == EndpointReadiness::Serving {
            current.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
        }
    })?;
    match updated {
        Some(record) => Ok(record),
        None => Ok(read_managed_service_manifest(paths, &started.service_id)
            .map_or_else(|_| started.clone(), |(record, _)| record)),
    }
}

fn validate_service_id(service_id: &str) -> Result<()> {
    // Single source of truth for what makes an id safe as a filesystem path
    // component (rejects empty, path separators, `..`, control chars).
    rocm_core::ServiceId::new(service_id)?;
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

/// Whether `rocm update` should route into the runtime update path
/// (`apply_runtime_update`) instead of the read-only status report.
///
/// `--dry-run` alone must take this path too, since `apply_runtime_update`
/// only mutates anything when `dry_run` is false — a plain status report
/// would silently ignore `--dry-run` and never show what `--apply` would do.
const fn update_should_preview_or_apply(apply: bool, dry_run: bool) -> bool {
    apply || dry_run
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
    let plan = therock::runtime_update_plan(paths, source, &manifests, None)?;
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
    let _ = writeln!(output, "  target_runtime_key: {}", plan.target_runtime_key);
    let _ = writeln!(output, "  activate_after_install: {activate}");
    if !plan.update_available {
        let _ = writeln!(output, "  result: no newer runtime found");
        return Ok(output);
    }

    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let install_plan = therock::install_sdk_for_update(
            paths,
            &source.channel,
            &source.format,
            &source.family,
            plan.device_target.as_deref(),
            plan.source_layout_generation.as_deref(),
            source.includes_devel(),
            true,
            activate,
        )?;
        let _ = writeln!(output, "  install_plan:");
        for line in install_plan.output.lines() {
            let _ = writeln!(output, "    {line}");
        }
        return Ok(output);
    }

    // `activate` rather than a bare `true`: the update path is preapproved either
    // way (its approval comes from the runtime the user selected, not from a
    // flag, and `rocm update` has no terminal contract), but the approval line it
    // prints must not promise an activation that only `--activate` performs
    // below.
    let install_output = therock::install_sdk_for_update(
        paths,
        &source.channel,
        &source.format,
        &source.family,
        plan.device_target.as_deref(),
        plan.source_layout_generation.as_deref(),
        // Reinstall what the user originally chose rather than silently
        // adding or dropping the compiler toolchain on update.
        source.includes_devel(),
        false,
        activate,
    )?;
    let manifests_after = therock::load_runtime_manifests(paths)?;
    // By exact key, never by version: a same-version repair installs a sibling
    // that shares the source's channel, format, family AND version, so a
    // version match would just as happily return the stale runtime this update
    // was meant to replace, and then activate it.
    let installed = select_installed_update_runtime(&manifests_after, &plan.target_runtime_key)
        .with_context(|| {
            format!(
                "runtime install completed but no manifest was written for the planned runtime key `{}`",
                plan.target_runtime_key
            )
        })?;
    let _ = writeln!(output, "  installed_runtime_key: {}", installed.runtime_key);
    let _ = writeln!(
        output,
        "  installed_runtime_root: {}",
        installed.install_root.display()
    );
    if activate {
        // No `config.save` after this: `activate_runtime` already persisted the
        // same struct and nothing mutates it in between, so a second write can
        // only fail after an activation that fully succeeded — sending the user
        // to re-run a command that already worked.
        let activation = activate_runtime(paths, config, &installed.runtime_key)?;
        append_update_activate_summary(&mut output, &activation);
    } else {
        let _ = writeln!(
            output,
            "  next step: rocm runtimes activate {}",
            installed.runtime_key
        );
    }
    let _ = writeln!(output, "  install_output:");
    for line in install_output.output.lines() {
        let _ = writeln!(output, "    {line}");
    }
    Ok(output)
}

/// Shown after any activation (direct `runtimes activate`, `update --apply
/// --activate`, or an `install sdk` that switches the active runtime) that
/// recorded a previous runtime, so every path that reaches this state gives
/// the same advice — see `RuntimesCommand::Activate` in `runtimes()`,
/// `append_update_activate_summary`, and `render_sdk_install_success` for the
/// call sites.
const ROLLBACK_RECOVERY_HINT: &str = "  next step: if this causes problems, run `rocm runtimes rollback` \
(no history - undoes only this one activation)";

/// Appends the `--activate` branch of the `update --apply` summary. The
/// rollback hint is only shown when a previous runtime was actually recorded —
/// `rocm runtimes rollback` hard-errors otherwise (no prior runtime to return
/// to), so hinting it unconditionally would point users at a command that
/// immediately fails on their very first activation.
fn append_update_activate_summary(output: &mut String, activation: &RuntimeActivationResult) {
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
    let _ = write!(
        output,
        "{}",
        runtime_services::render_runtime_service_reconciliation(
            &activation.services,
            &activation.runtime_key
        )
    );
    if activation.previous_runtime_key.is_some() {
        let _ = writeln!(output, "{ROLLBACK_RECOVERY_HINT}");
    }
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
    target_runtime_key: &str,
) -> Option<&'a therock::InstalledRuntimeManifest> {
    manifests
        .iter()
        .find(|manifest| manifest.runtime_key == target_runtime_key)
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
                    .map_or_else(|| "not set".to_owned(), format_bytes);
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

fn friendly_engine_label(engine: &str) -> &str {
    match engine {
        "lemonade" => "Lemonade",
        "vllm" => "vLLM",
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
/// service for this engine+model, if any. The caller compares its effective
/// launch recipe before deciding whether reuse is safe.
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

const SERVICE_LIVENESS_CHECK_TIMEOUT: Duration = Duration::from_millis(750);

/// The real process ids a record tracks.
///
/// The launcher pid is always recorded; `engine_pid` is adopted once the engine
/// state file reports one. A `0` pid is a placeholder, never a real process.
fn recorded_service_pids(record: &ManagedServiceRecord) -> Vec<u32> {
    [record.engine_pid, Some(record.supervisor_pid)]
        .into_iter()
        .flatten()
        .filter(|pid| *pid != 0)
        .collect()
}

/// Complete a stop that was requested but could not confirm termination, once
/// the processes are actually observed gone: drop the endpoint key and clear the
/// marker. Returns whether the record changed.
///
/// This is deliberately keyed on `stop_requested_unix_ms` rather than on "the
/// tracked pids are dead". A *crashed* service also has dead pids, but its
/// operator never asked for it to go away and will want it back: dropping the
/// key there would make `rocm services restart` and daemon recovery refuse it
/// permanently — fail-closed guards have no way to re-mint a key. The cost of
/// this choice is that a crashed public service's 0600 key file outlives the
/// process until the service is stopped or successfully restarted.
///
/// Called before the liveness gate in
/// [`refresh_managed_service_runtime_liveness`], so a record that reached
/// "stopped" by another route still gets its deferred cleanup rather than
/// stranding the key.
fn settle_pending_stop_key_cleanup(paths: &AppPaths, record: &mut ManagedServiceRecord) -> bool {
    if record.stop_requested_unix_ms.is_none() {
        return false;
    }
    if recorded_service_pids(record)
        .iter()
        .any(|pid| process_is_running(*pid))
    {
        return false;
    }
    endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
    record.stop_requested_unix_ms = None;
    true
}

fn refresh_managed_service_runtime_liveness(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
) -> bool {
    let settled_pending_stop = settle_pending_stop_key_cleanup(paths, record);
    if !managed_service_is_live(record) {
        return settled_pending_stop;
    }

    // Probe with the service's key so a protected public service is not mistaken
    // for dead (an anonymous /v1/models would 401).
    let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
    let outcome = if matches!(record.status.as_str(), "ready" | "running") {
        managed_service_endpoint_readiness(
            record,
            endpoint_api_key.as_deref(),
            SERVICE_LIVENESS_CHECK_TIMEOUT,
            rocm_core::INFERENCE_PROBE_TIMEOUT,
        )
    } else {
        EndpointReadinessOutcome {
            readiness: EndpointReadiness::Unreachable,
            record_changed: false,
        }
    };
    let readiness = outcome.readiness;
    if readiness == EndpointReadiness::Serving {
        // Mirror `providers.rs::ready_local_services()`: a live, probe-passing
        // service should be persisted as "ready", not left stuck at "running".
        if record.status == "running" {
            record.status = "ready".to_owned();
            return true;
        }
        // A newly latched verification is itself worth persisting, so the next
        // check reads it instead of sending another inference request.
        return settled_pending_stop || outcome.record_changed;
    }
    if readiness == EndpointReadiness::Listing {
        // The server is up and advertising the model; the weights are simply not
        // usable yet. Leave the status alone rather than demoting to "starting" —
        // `rocmd` restarts a service that sits in "starting" past its stale
        // window, which for a slow-loading model would restart it mid-load and
        // begin the wait all over again. Still report a change when the probe
        // bookkeeping moved, so the retry throttle is persisted and the next
        // process does not spend another full probe timeout.
        return settled_pending_stop || outcome.record_changed;
    }

    let tracked_pids = recorded_service_pids(record);
    let has_tracked_pid = !tracked_pids.is_empty();
    let has_live_pid = tracked_pids.iter().any(|pid| process_is_running(*pid));
    if has_tracked_pid && !has_live_pid {
        // Reconcile the status only. The endpoint key is *not* dropped here:
        // dead pids alone do not distinguish a stopped service from a crashed
        // one, and a crashed public service needs its key to be restartable.
        // `settle_pending_stop_key_cleanup` above owns that cleanup, gated on a
        // stop having actually been requested.
        if record.status != "stopped" {
            record.status = "stopped".to_owned();
            return true;
        }
        return settled_pending_stop;
    }

    if matches!(record.status.as_str(), "ready" | "running") {
        let new_status = if has_live_pid { "starting" } else { "stopped" };
        if record.status != new_status {
            record.status = new_status.to_owned();
            return true;
        }
    }

    settled_pending_stop
}

/// One-line header for the local-servers views.
///
/// Records that are neither ready nor starting used to be invisible here, so a
/// host with three failed servers and nothing live read "none ready" - true of
/// the live set, misleading about what is on disk. They now get their own
/// clause. The ready/starting wording is unchanged, so a header with no past
/// attempts reads exactly as before; one with past attempts gains a third
/// clause ("1 ready, 1 starting, 1 not running").
fn local_server_sidebar_status(counts: &ManagedServiceSidebarCounts) -> String {
    let mut parts = Vec::new();
    if counts.ready > 0 {
        parts.push(format!("{} ready", counts.ready));
    }
    if counts.starting > 0 {
        parts.push(format!("{} starting", counts.starting));
    }
    if counts.past_attempts > 0 {
        parts.push(format!("{} not running", counts.past_attempts));
    }
    if parts.is_empty() {
        "none ready".to_owned()
    } else {
        parts.join(", ")
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
            let refreshed_liveness = refresh_managed_service_runtime_liveness(paths, &mut record);
            if refreshed_from_engine || refreshed_liveness {
                let _ = record.write();
            }
            records.push(record);
        }
    }

    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    Ok(records)
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
    /// Installer PATH changes `rocm uninstall` undoes.
    path_edits: Vec<uninstall::PathEdit>,
    /// The installer's record of its PATH change, removed once that change is
    /// undone.
    path_record: Option<PathBuf>,
    /// The running executable, on Windows, which is deleted as the command exits.
    self_delete: Option<PathBuf>,
}

fn build_uninstall_plan(paths: &AppPaths, options: &UninstallOptions) -> Result<UninstallPlan> {
    let mut plan = UninstallPlan::default();

    plan_installed_binaries(
        &mut plan,
        options,
        || daemon_binary_path().context("failed to discover current rocm executable"),
        rocm_core::runtime_home_dir().as_deref(),
    )?;

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
        // Planned under the spelling `remove_path` acts on, so the review names
        // the entry that is actually removed, and `dir/` and `dir` dedup to one
        // line. Existence is the entry's own, not its target's: a dangling link
        // is listed and then really removed, instead of reported as absent.
        // A failed check is not an absence either: it fails the plan, so the
        // uninstall stops before removing anything instead of leaving a folder
        // that may still be on disk out of the review.
        match existing_entry(&path)
            .with_context(|| format!("failed to check the {kind} folder {}", path.display()))?
        {
            Some((entry, _)) => plan.actions.push(UninstallPlanEntry { kind, path: entry }),
            None => plan
                .skipped
                .push(format!("{kind} path not present: {}", path.display())),
        }
    }

    for note in shared_cache_notes(paths) {
        plan.warnings.push(note);
    }

    let managed_services = load_managed_services(paths).unwrap_or_default();
    if !managed_services.is_empty() {
        plan.warnings.push(format!(
            "{} managed service record(s) exist under {}; background processes are not stopped automatically in this pass",
            managed_services.len(),
            paths.services_dir().display()
        ));
    }

    // Remote sessions are worse than local ones to drop silently. The model runs
    // on someone else's machine and its endpoint is published there, so removing
    // the record here does not stop either — it only destroys the last thing that
    // knew they existed. Name them and the command that tears them down properly.
    //
    // A read that fails is reported, not defaulted away. `load_all` already warns
    // past an individual unreadable record and keeps going, so an `Err` here is a
    // directory-level I/O failure, and a missing directory is `Ok`. Taking the
    // default would state "there are none" on the one path where we do not know —
    // and this plan goes on to delete the data directory those records live in,
    // so the warning that something is still published elsewhere would be lost at
    // exactly the moment it was the last copy.
    let remote_sessions = match remote::session::load_all(paths) {
        Ok(sessions) => sessions,
        Err(error) => {
            plan.warnings.push(format!(
                "could not read the remote session records under {}: {error:#}\n\
                 This pass cannot tell whether models are still running on other machines with \
                 their endpoints published. Check with `rocm remote status` before continuing.",
                paths.remote_sessions_dir().display()
            ));
            Vec::new()
        }
    };
    if !remote_sessions.is_empty() {
        plan.warnings.push(format!(
            "{} remote session record(s) exist under {}; their models keep running on the \
             remote machines and their endpoints stay published. Removing these records only \
             loses track of them - run `rocm remote stop <session>` for each first: {}",
            remote_sessions.len(),
            paths.remote_sessions_dir().display(),
            remote_sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    plan.actions
        .sort_by(|left, right| left.path.cmp(&right.path));
    plan.actions.dedup_by(|left, right| left.path == right.path);
    Ok(plan)
}

/// Plan removing the installed binaries beside the running one, which
/// `current_exe` names, and undoing the installer's PATH change for their
/// folder. `--keep-binaries` keeps both, and so does a cargo build's folder.
fn plan_installed_binaries(
    plan: &mut UninstallPlan,
    options: &UninstallOptions,
    current_exe: impl FnOnce() -> Result<PathBuf>,
    home: Option<&Path>,
) -> Result<()> {
    if options.keep_binaries {
        // The PATH change only serves the binaries, so it stays with them.
        plan.skipped.push(
            "binary removal disabled by --keep-binaries, so any PATH change the installer made \
             stays too"
                .to_owned(),
        );
        return Ok(());
    }
    let current_exe = current_exe()?;
    if is_dev_binary_layout(&current_exe) && !options.force_dev_binaries {
        plan.skipped.push(format!(
            "binary removal skipped because {} looks like a cargo target build; pass --force-dev-binaries to remove sibling debug/release binaries",
            current_exe.display()
        ));
        return Ok(());
    }
    let binaries = collect_installed_binary_candidates(&current_exe)?;
    for path in &binaries {
        // Windows will not unlink a running image; it is deleted as the
        // command exits instead.
        if is_running_executable(path, &current_exe, cfg!(windows)) {
            plan.self_delete = Some(path.clone());
            continue;
        }
        // The PATH record goes only once its change is undone; see
        // `uninstall::plan_path_edits`.
        if path.file_name().and_then(|name| name.to_str()) == Some(uninstall::PATH_EDIT_RECORD) {
            continue;
        }
        plan.actions.push(UninstallPlanEntry {
            kind: "binary",
            path: path.clone(),
        });
    }
    if let Some(bin_dir) = current_exe.parent() {
        uninstall::plan_path_edits(plan, bin_dir, &binaries, home);
    }
    Ok(())
}

/// Whether `path`, an entry of the running executable's folder, is the running
/// executable itself, which only Windows has to treat specially. Windows names
/// are case-insensitive, and the running image may report its own name in
/// another case, or in its 8.3 short form (`ROCM.EXE`), than the folder listing
/// does; missing it would plan an ordinary removal that Windows refuses.
fn is_running_executable(path: &Path, current_exe: &Path, windows: bool) -> bool {
    if !windows {
        return false;
    }
    path == current_exe
        || (path.parent() == current_exe.parent()
            && path
                .file_name()
                .zip(current_exe.file_name())
                .is_some_and(|(name, exe)| {
                    name.to_string_lossy()
                        .eq_ignore_ascii_case(&exe.to_string_lossy())
                }))
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
    let removals = uninstall::removal_count(plan);
    if uninstall::plan_is_empty(plan) {
        let _ = writeln!(output, "Nothing managed by rocm-cli would be removed.");
    } else if removals > 0 {
        let _ = writeln!(output, "{removals} item(s) would be removed:");
        for entry in &plan.actions {
            let _ = writeln!(output, "  - {}: {}", entry.kind, entry.path.display());
        }
    }
    uninstall::render_extra_plan(&mut output, plan);
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

/// Caches that ROCm CLI causes to be filled but does not own, so uninstall
/// leaves them in place.
///
/// These are usually the largest things on disk after an install, so a plan
/// that stays silent about them reads as a clean slate it does not deliver.
/// Only paths that exist and sit outside everything already being removed are
/// reported, so the notes stay truthful as caches move under the data
/// directory.
fn shared_cache_notes(paths: &AppPaths) -> Vec<String> {
    let removed_roots = [
        paths.config_dir.clone(),
        paths.data_dir.clone(),
        paths.cache_dir.clone(),
    ];
    shared_cache_notes_for(&removed_roots, &shared_cache_candidates())
}

/// The pure part of [`shared_cache_notes`], so the filtering is testable without
/// depending on the caller's real environment.
fn shared_cache_notes_for(
    removed_roots: &[PathBuf],
    candidates: &[(PathBuf, &'static str)],
) -> Vec<String> {
    candidates
        .iter()
        .filter(|(path, _)| path.exists())
        .filter(|(path, _)| !removed_roots.iter().any(|root| path.starts_with(root)))
        .map(|(path, description)| format!("{description} not removed: {}", path.display()))
        .collect()
}

/// The shared caches worth reporting, with why each is left alone.
fn shared_cache_candidates() -> Vec<(PathBuf, &'static str)> {
    let mut candidates = Vec::new();

    // `uv` writes to `UV_CACHE_DIR` when set, otherwise its own default. Read it
    // rather than assuming a location, so this stays correct wherever the cache
    // has been pointed.
    let uv_cache = env_path("UV_CACHE_DIR")
        .or_else(|| rocm_core::runtime_home_dir().map(|home| home.join(".cache").join("uv")));
    if let Some(path) = uv_cache {
        candidates.push((
            path,
            "the uv package cache is shared with other uv projects on this computer and is",
        ));
    }

    let hf_cache = env_path("HF_HOME")
        .map(|home| home.join("hub"))
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| {
            rocm_core::runtime_home_dir()
                .map(|home| home.join(".cache").join("huggingface").join("hub"))
        });
    if let Some(path) = hf_cache {
        candidates.push((
            path,
            "downloaded model files are slow to fetch again and may include your own, so they are",
        ));
    }

    candidates
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
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
    if name == ".rocm-cli-manifest" || name == uninstall::PATH_EDIT_RECORD {
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

/// `path` respelled so it names the final entry itself.
///
/// A trailing separator (`link/`) or a trailing `.` (`link/.`) makes the kernel
/// resolve a final symlink, so `symlink_metadata` reports the link's *target*
/// directory and a recursive delete walks into it. Rebuilding the path from its
/// components drops exactly those spellings (plus repeated separators and
/// interior `.`), none of which otherwise change which entry is named. `..` is
/// kept: dropping it lexically would change the meaning through a link.
fn entry_path(path: &Path) -> PathBuf {
    path.components().collect()
}

/// Does `path`'s text end in a separator or a final `.` (`x/`, `x/.`)? The
/// kernel only lets such a spelling name a directory, so `notes.txt/` names
/// nothing even though `notes.txt` exists.
fn spelled_as_directory(path: &Path) -> bool {
    let text = path.as_os_str().as_encoded_bytes();
    let is_separator = |byte: u8| std::path::is_separator(char::from(byte));
    match text {
        [.., last] if is_separator(*last) => true,
        [.., before, b'.'] => is_separator(*before),
        _ => false,
    }
}

/// The entry `path` names, as [`entry_path`] spells it, with its own
/// (link-not-followed) metadata — or `None` when nothing is there.
///
/// A spelling that can only name a directory (see [`spelled_as_directory`])
/// names nothing when the entry is a regular file, so that counts as absent
/// rather than letting the respelling turn `notes.txt/` into `notes.txt`. A
/// symlink is still the entry: it is what [`remove_path`] unlinks.
fn existing_entry(path: &Path) -> io::Result<Option<(PathBuf, fs::Metadata)>> {
    let entry = entry_path(path);
    // Not `path.exists()`: that follows links, so a dangling link would read as
    // already gone, be skipped, and still be reported as removed by the caller.
    // "Not a directory" means a folder on the way is really a file (as in
    // `notes.txt/../x`), so the path names nothing, the same as "not found".
    // Any other failure is a real error, not an absence.
    let metadata = match fs::symlink_metadata(&entry) {
        Ok(metadata) => metadata,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    if spelled_as_directory(path) && !metadata.is_dir() && !metadata.file_type().is_symlink() {
        return Ok(None);
    }
    Ok(Some((entry, metadata)))
}

/// Remove the entry `path` names: a file or symlink is unlinked (never
/// followed, however `path` is spelled), a real directory is removed
/// recursively, and an entry that is already gone is a no-op.
///
/// Normalizing here rather than only at plan time covers every caller —
/// `rocm uninstall` and `rocm storage remove-downloads` — whatever spelling
/// their plan carries.
fn remove_path(path: &Path) -> Result<()> {
    let Some((path, metadata)) =
        existing_entry(path).with_context(|| format!("failed to stat {}", path.display()))?
    else {
        return Ok(());
    };
    let path = path.as_path();
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
        (
            "lemonade",
            "default embedded Lemonade server with ROCm llama.cpp backend",
        ),
        (
            "vllm",
            "Linux/WSL ROCm GPU serving engine through external vLLM",
        ),
    ]
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
        // Ignore SIGPIPE for the duration of the write: with SIG_DFL in effect
        // process-wide, an engine that exits before reading its stdin would
        // otherwise kill `rocm` with SIGPIPE before the exit-status diagnostics
        // below can run. Under SIG_IGN the write returns EPIPE instead, which we
        // surface as the normal "failed to write engine request" error and then
        // let `wait()` report why the engine exited.
        with_sigpipe_ignored(|| {
            serde_json::to_writer(&mut stdin, &envelope)
                .context("failed to write engine request")?;
            stdin.write_all(b"\n")?;
            Ok::<(), anyhow::Error>(())
        })?;
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
        "lemonade" => Some(rocm_engine_lemonade::builtin_handle_envelope(
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
    matches!(engine, "lemonade" | "vllm")
}

#[allow(clippy::too_many_arguments)]
fn run_builtin_engine_serve_http(
    engine: &str,
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: &str,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    let parsed_policy = parse_device_policy(Some(device_policy))?;
    match engine {
        "lemonade" => rocm_engine_lemonade::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "vllm" => rocm_engine_vllm::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
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
    gpu_indices: &[u32],
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
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
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

/// Parse the user-facing `--gpu` value (`auto` or a single index like `1`)
/// into a `GpuSelection`. Comma lists are rejected (single-GPU serving only).
/// `None` defaults to `auto`.
fn parse_gpu_selection(value: Option<&str>) -> Result<GpuSelection> {
    let Some(raw) = value else {
        return Ok(GpuSelection::Auto);
    };
    GpuSelection::parse_cli_value(raw).map_err(|message| anyhow::anyhow!(message))
}

/// Parse the internal `--gpu` csv passed to the hidden `__engine-serve-http`
/// subcommand back into explicit device ordinals (empty when absent/`auto`).
fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    match parse_gpu_selection(value)? {
        GpuSelection::Auto => Ok(Vec::new()),
        GpuSelection::Index(index) => Ok(vec![index]),
    }
}

/// Take the managed-launch lock and decide, under it, which device ordinal this
/// server pins. This is the whole select-then-claim critical section's entry
/// half: the returned [`rocm_core::FileLock`] must be held by the caller until
/// the claiming service record is persisted, at which point the choice is
/// visible to any concurrent auto-selection and the lock is dropped (in
/// `start_managed_service`/`run_attached_service`, before the readiness wait).
/// Acquiring the lock and reading busy-GPU state are one operation here on
/// purpose: split apart, two concurrent `rocm serve --gpu auto` both read the
/// same GPU as free and land on it.
///
/// `pinned` is the already-validated explicit `--gpu <index>` (see
/// [`validate_pinned_gpu_index`], which `serve()` applies before runtime
/// resolution so a bad ordinal is refused without a runtime). Explicit-index and
/// CPU-only launches take the lock too — their selection is fixed, but their
/// claim-record write must stay serialized against a concurrent auto-select.
/// Only `--gpu auto` (`cpu_only` false, `pinned` `None`) reads live state under
/// it, picking the lowest-numbered GPU that is idle (high free VRAM) and not
/// already serving a rocm-cli managed/foreground model. The result holds at most
/// one ordinal; serving across multiple GPUs is not supported.
///
/// Ordinal semantics: the index produced here is fed to engines via
/// `HIP_VISIBLE_DEVICES`, but it is sourced from `amd-smi`'s `gpu` index
/// (probing/busy detection). On a normal single-host ROCm install without GPU
/// partitioning these orderings coincide, but they are not guaranteed to match
/// when `ROCR_VISIBLE_DEVICES`, CPX/partition modes, or non-default device
/// enumeration are in play. Validate on multi-GPU hardware before relying on a
/// specific `--gpu <index>` mapping in those configurations.
///
/// Selection is mask-aware: `visible` is the set of HIP ordinals still usable
/// after the active visibility mask (see [`rocm_core::usable_amd_gpu_indices`]) —
/// already in the space the choice is exported through, since a
/// `ROCR_VISIBLE_DEVICES` mask has its survivors re-indexed to `0..N`.
/// Auto-selection is restricted to it, so serve never targets a masked-out
/// device. `None` (an unprobeable host) keeps the previous mask-unaware
/// behavior.
///
/// The lock is held for as long as the caller keeps the guard, which includes
/// the already-running check inside `spawn_managed_engine_child` — that refreshes
/// managed-service liveness and can issue per-service endpoint probes, so an
/// unrelated concurrent serve can wait on the order of seconds. `FileLock`
/// blocks without a timeout; the slow first-use install is deliberately kept
/// outside this section so it is not also serialized.
///
/// That span is relied on by a second caller: because the guard outlives both
/// `store_endpoint_api_key` and the first `record.write()`, it also covers the
/// interval in which a launch's 0600 endpoint key sits on disk with no manifest
/// beside it. [`prune_managed_service_records`] acquires the same lock so its
/// leftover sweep cannot read the directory in that state. Shortening the
/// guard's life — releasing it before the record is persisted — would silently
/// reopen that window as well as the GPU double-booking one.
fn select_gpu_indices_under_launch_lock(
    paths: &AppPaths,
    cpu_only: bool,
    pinned: Option<Vec<u32>>,
    detect_count: impl FnOnce() -> Option<usize>,
    visible: Option<&[u32]>,
    vram: Option<&[GpuVramUsage]>,
) -> Result<(Vec<u32>, rocm_core::FileLock)> {
    let lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
    let indices = if cpu_only {
        Vec::new()
    } else if let Some(indices) = pinned {
        indices
    } else {
        // `detect_count` is invoked only here: it shells out to amd-smi, and the
        // CPU-only and explicit-index paths have no use for the count, so they do
        // not pay for that subprocess inside the critical section.
        auto_select_gpu_indices(paths, detect_count(), visible, vram)
    };
    Ok((indices, lock))
}

/// Validate an explicit `--gpu <index>` against the active visibility set,
/// returning the single pinned ordinal. `visible` is the set of HIP ordinals
/// still usable after the visibility mask (see
/// [`rocm_core::usable_amd_gpu_indices`]) — already in the same space the index
/// is exported through — so it is the authoritative answer whenever it could be
/// enumerated. `detected` is only a fallback range check for hosts where the
/// visible set is unknown (`visible` is `None`). Both `None` (an unprobeable
/// host) is allowed through so serving can still proceed.
fn validate_pinned_gpu_index(
    index: u32,
    detected: Option<usize>,
    visible: Option<&[u32]>,
    mask_active: bool,
) -> Result<Vec<u32>> {
    // Prefer the visibility-resolved set. It is authoritative and already in the
    // HIP-ordinal space `--gpu` is exported through, so it settles both range and
    // mask membership in one comparison — avoiding the earlier bug of checking
    // the index against `detected` (a differently-probed physical count) and
    // `visible` (HIP space) as if they shared one ordinal space.
    //
    // `mask_active` only shapes the message, not the accept/reject decision: with
    // no mask set the visible set is just `0..present`, so an index outside it is
    // simply absent from the host, and blaming HIP/ROCR variables the user never
    // set sends them chasing an environment problem that does not exist.
    if let Some(visible) = visible {
        if visible.is_empty() {
            // An empty set is authoritative "no GPU usable" (every device masked
            // out, or none present), not "could not enumerate" — that is `None`.
            // serve()'s no-usable-GPU fail-fast normally reports this first with a
            // fuller message; reject here too so a bad `--gpu` can never slip
            // through.
            if mask_active {
                bail!(
                    "no usable AMD GPU is available under the active visibility mask \
                     (HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES); `--gpu {index}` cannot be honoured"
                );
            }
            bail!("no usable AMD GPU is present on this host; `--gpu {index}` cannot be honoured");
        }
        if !visible.contains(&index) {
            let list = visible
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            if mask_active {
                bail!(
                    "--gpu index {index} is not available under the active visibility mask \
                     (HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES); usable GPU indices: [{list}]"
                );
            }
            bail!("--gpu index {index} is not present on this host; usable GPU indices: [{list}]");
        }
        return Ok(vec![index]);
    }
    // No visible set could be enumerated: fall back to the raw detected count so
    // an obviously out-of-range index is still refused.
    if let Some(count) = detected
        && (index as usize) >= count
    {
        bail!(
            "--gpu index {index} is out of range; {count} GPU(s) detected (valid indices 0..{})",
            count - 1
        );
    }
    Ok(vec![index])
}

/// The GPU count to rank `--gpu auto` candidates over. Prefers the `amd-smi`
/// device count, but only when that count is usable — `None` (amd-smi
/// unavailable) and `Some(0)` (amd-smi ran and saw no devices) both fall through
/// to the DRM sysfs VRAM fallback's row count, so auto-selection can still
/// consider devices amd-smi did not enumerate. Used only to decide whether
/// `--gpu auto` has anything at all to rank, never as a `--gpu <index>`
/// validation bound — see [`validate_pinned_gpu_index`]. `None` when neither
/// source yields a non-zero count.
fn effective_gpu_count(detected: Option<usize>, vram: Option<&[GpuVramUsage]>) -> Option<usize> {
    detected
        .filter(|&count| count > 0)
        .or_else(|| vram.map(<[GpuVramUsage]>::len).filter(|&count| count > 0))
}

/// A GPU's local VRAM occupancy as reported by `amd-smi metric --json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuVramUsage {
    index: u32,
    used_mb: u64,
    total_mb: u64,
}

impl GpuVramUsage {
    /// Fraction of total VRAM that is currently free (`0.0`..=`1.0`). Returns
    /// `None` when the total is unknown so callers do not divide by zero.
    fn free_fraction(self) -> Option<f64> {
        if self.total_mb == 0 {
            return None;
        }
        Some(self.total_mb.saturating_sub(self.used_mb) as f64 / self.total_mb as f64)
    }

    /// Absolute VRAM currently free, in MiB. Used to compare GPUs of differing
    /// total capacity: a higher free *fraction* on a smaller GPU can still mean
    /// less free memory than a larger GPU, so auto-selection ranks by this.
    const fn free_mb(self) -> u64 {
        self.total_mb.saturating_sub(self.used_mb)
    }
}

/// A GPU is treated as "free" for `--gpu auto` when at least this fraction of
/// its VRAM is unused (i.e. it is effectively idle).
const AUTO_FREE_VRAM_FRACTION: f64 = 0.90;

/// Pick a GPU ordinal for `--gpu auto`. Prefers the lowest-numbered GPU that is
/// idle (free VRAM at or above [`AUTO_FREE_VRAM_FRACTION`]) and not pinned by a
/// running rocm-cli service; otherwise falls back to the non-busy GPU with the
/// most free VRAM, then to the first non-busy GPU. When neither the amd-smi
/// device count nor the DRM sysfs VRAM fallback yields any device it returns no
/// selection rather than assuming device 0 — the engine's device probe then pins
/// the first present GPU or fails fast under the GPU-required policy.
///
/// Selection is mask-aware: candidates are restricted to `visible`, the HIP
/// ordinals still usable after the active visibility mask (see
/// [`rocm_core::usable_amd_gpu_indices`]), so auto-select never lands on a
/// masked-out GPU the engine would then reject. `None` (an unprobeable host)
/// leaves the detected range unrestricted, as before.
///
/// Under a `ROCR_VISIBLE_DEVICES` mask the *choice* is correct — `visible` is
/// already re-indexed into the HIP space the selection is exported through — but
/// the `vram` rows come from amd-smi, which does not honour that mask, so the
/// idleness ranking can read a different device's occupancy than the one the
/// ordinal binds. That is a best-effort heuristic, not a binding hazard: it can
/// pick a less idle visible GPU, never a hidden one. Translating amd-smi rows
/// through a ROCR mask is deferred (needs multi-GPU hardware to verify).
///
/// This reads service state (`busy_gpu_indices`) but does not lock on its own.
/// Concurrency safety is the caller's responsibility: `serve()` holds the
/// managed-launch lock (`AppPaths::managed_launch_lock_path`) across this
/// selection and the claiming record write, so two near-concurrent `--gpu auto`
/// launches are serialized and cannot land on the same idle GPU. Callers that
/// select outside that lock get only best-effort de-confliction from the
/// VRAM-occupancy fallback and the start-time low-memory warning.
fn auto_select_gpu_indices(
    paths: &AppPaths,
    detected: Option<usize>,
    visible: Option<&[u32]>,
    vram: Option<&[GpuVramUsage]>,
) -> Vec<u32> {
    let busy = busy_gpu_indices(paths);
    select_auto_gpu_index(detected, visible, &busy, vram)
}

/// Pure auto-selection used by [`auto_select_gpu_indices`], split out so the
/// preference order can be unit-tested without amd-smi or service state.
fn select_auto_gpu_index(
    detected: Option<usize>,
    visible: Option<&[u32]>,
    busy: &[u32],
    vram: Option<&[GpuVramUsage]>,
) -> Vec<u32> {
    // The amd-smi `list` device count is the primary source, but on a shared
    // node without amd-smi it is `None` even though the DRM sysfs fallback probe
    // may still have populated `vram`. Derive the count from those rows in that
    // case so `--gpu auto` can rank GPUs by free VRAM instead of returning no
    // selection and landing on a busy device 0.
    let count = effective_gpu_count(detected, vram).unwrap_or(0);
    if count == 0 {
        // No GPU count from amd-smi and no VRAM telemetry (both unavailable, or
        // genuinely zero devices). Do not assume device 0 exists: return no
        // selection and let the engine's device probe pin the first present GPU
        // or fail fast under the GPU-required policy (no GPU-0 fallback).
        return Vec::new();
    }
    // Ordinals this selection may hand out, lowest first. When VRAM telemetry is
    // present, drive the scan from the *actual* row indices it reports rather
    // than a synthetic `0..count` range: amd-smi metric (and the DRM sysfs
    // fallback) can report non-contiguous ordinals when devices are hidden by a
    // visibility mask, so `0..count` would look up rows that do not exist and
    // miss the real ones. Without telemetry, fall back to the dense detected
    // range so service-state-only selection still has candidates.
    //
    // Either way the set is then narrowed to the ordinals still visible under
    // the active mask, so auto-select never targets a masked-out GPU. `visible`
    // is in HIP space (a ROCR mask's survivors are re-indexed to `0..N`), which
    // is the space the selection is exported through, so the retained ordinals
    // are directly selectable. When the host is unprobeable (`visible` is
    // `None`) the set is left unrestricted (mask-unaware, as before).
    //
    // A short row set is not always a mask, though: `parse_gpu_vram_usage`
    // drops any device entry missing `/mem_usage/used_vram/value` or
    // `/mem_usage/total_vram/value`, so a device the lighter `list` enumeration
    // counts can simply have no row. Taking rows-only there would shrink the
    // candidate set for a reason that has nothing to do with visibility, and if
    // every *reported* device is busy the terminal fallback would hand back a
    // busy GPU while an idle, merely untelemetried one went unconsidered —
    // exactly the "serve pinned to an occupied GPU" fault this selection exists
    // to avoid. So the `0..count` range is unioned back in — but only where it
    // is evidence rather than invention, which is decided by the rows, not by
    // `visible`:
    //
    //   * `detected` must be `Some(n > 0)`, so `count` is the `list` device
    //     count — an independent source asserting those ordinals exist. When it
    //     is absent `effective_gpu_count` falls back to `rows.len()`, which is
    //     only the rows restating their own size and confirms nothing they omit.
    //   * every row index must be below `count`. A row at or above it is the one
    //     available proof that the ordinal space is re-indexed or sparse rather
    //     than dense from 0 (`[2, 3]` with `count == 2`), and that is exactly the
    //     masked shape the rows-only construction was introduced for, so there
    //     the rows stay authoritative. Below the count the rows are a subset of a
    //     dense range the count already asserts, and `0..count` is no stronger an
    //     assumption than the telemetry-less arm below already makes.
    //
    // Gating on `visible.is_some()` instead would have left the bug standing
    // wherever the mask is unknown, which is not a corner: `usable_amd_gpu_indices`
    // is `None` unconditionally off Linux, so `visible` is always `None` on
    // Windows. The retain below still runs on top whenever a mask *is* known, so
    // no synthesised ordinal survives that the mask contradicts.
    let mut reported: Vec<u32> = match vram {
        Some(rows) => {
            let mut indices: Vec<u32> = rows.iter().map(|row| row.index).collect();
            let count_is_independently_sourced = detected.is_some_and(|detected| detected > 0);
            let rows_are_a_dense_range_subset =
                indices.iter().all(|&index| (index as usize) < count);
            if count_is_independently_sourced && rows_are_a_dense_range_subset {
                indices.extend(0..count as u32);
            }
            indices
        }
        None => (0..count as u32).collect(),
    };
    reported.sort_unstable();
    // The union above can repeat an ordinal that both sources name.
    reported.dedup();
    if let Some(visible) = visible {
        reported.retain(|index| visible.contains(index));
    }
    if reported.is_empty() {
        // Every reported device is masked out (or telemetry reported no rows at
        // all). Do not assume device 0 exists: return no selection and let the
        // engine's device probe pin the first present GPU or fail fast under the
        // GPU-required policy (no GPU-0 fallback).
        return Vec::new();
    }
    // The subset still free to claim. Kept separate from `reported` so the
    // terminal all-busy fallback below can still name a device that was actually
    // reported instead of fabricating one.
    let candidate_indices: Vec<u32> = reported
        .iter()
        .copied()
        .filter(|index| !busy.contains(index))
        .collect();
    let usage_for = |index: u32| vram.and_then(|rows| rows.iter().find(|row| row.index == index));

    if vram.is_some() {
        // Pass 1: lowest-index idle GPU.
        for &index in &candidate_indices {
            if let Some(free) = usage_for(index).and_then(|usage| usage.free_fraction())
                && free >= AUTO_FREE_VRAM_FRACTION
            {
                return vec![index];
            }
        }
        // Pass 2: the non-busy GPU with the most free VRAM in absolute terms
        // (not free percentage, which can favor a smaller GPU on heterogeneous
        // VRAM systems). `usage_for` is only *partial* over `candidate_indices` —
        // the union above puts back ordinals the `list` count confirms but
        // telemetry never reported — yet no "does this ordinal have a row?" guard
        // is needed, because the comparator already orders them last: the keys are
        // `Option<u64>`, and `None` sorts below every `Some`, including `Some(0)`.
        // So a rowless ordinal can only be the maximum when *no* candidate has a
        // row; then every key is `None`, every comparison falls through to the
        // index tie-break, and the winner is the lowest candidate index — which is
        // precisely what Pass 3 would return from an ascending `reported`. Adding
        // the guard back would therefore change no outcome, only skip a pass that
        // already agrees.
        if let Some(&index) = candidate_indices.iter().max_by(|left, right| {
            let left_free = usage_for(**left).map(|usage| usage.free_mb());
            let right_free = usage_for(**right).map(|usage| usage.free_mb());
            left_free
                .cmp(&right_free)
                // Break ties toward the lowest index.
                .then(right.cmp(left))
        }) {
            return vec![index];
        }
    }

    // Pass 3: no VRAM telemetry; first GPU not pinned by a managed service.
    if let Some(&index) = candidate_indices.first() {
        return vec![index];
    }
    // Every reported, visible GPU is pinned by a running service. Return the
    // lowest of them (no CPU fallback is ever used); the caller surfaces a
    // low-memory warning so the user can free a device or pick another `--gpu`.
    // It must come from `reported`, not a hardcoded 0: once candidates are
    // sourced from the VRAM rows those ordinals can be sparse (a visibility mask
    // can leave only `[2, 3]`), so a literal 0 would pin a device nothing ever
    // reported — exported straight into `HIP_VISIBLE_DEVICES` as a silently
    // wrong choice. `reported` is non-empty here, guaranteed by the check above.
    vec![reported[0]]
}

/// Best-effort per-GPU VRAM occupancy. Prefers `amd-smi metric --json`; when
/// amd-smi is not installed (e.g. a shared node or container that ships no
/// amd-smi — this never tries `rocm-smi`) it falls back to the amdgpu DRM sysfs
/// VRAM counters so `--gpu auto` selection and the low-VRAM warning still have
/// telemetry to work with on a single-GPU host (see
/// [`read_drm_vram_usage`] for why the sysfs fallback withholds telemetry on a
/// multi-GPU one). Returns `None` only when neither source is readable
/// (callers then fall back to service-state-only auto-selection).
fn gpu_vram_usage() -> Option<Vec<GpuVramUsage>> {
    #[cfg(feature = "e2e-test-hooks")]
    if let Some(forced) = simulated_low_vram_usage() {
        return Some(forced);
    }
    gpu_vram_usage_amd_smi().or_else(gpu_vram_usage_sysfs)
}

/// E2E hook: when `ROCM_E2E_FORCE_LOW_VRAM` names a GPU ordinal, report that GPU
/// as almost entirely occupied, overriding the real telemetry.
///
/// The serve-plan low-VRAM OOM warning (and, for vLLM, its
/// `--gpu-memory-utilization` note) only fires when a *selected* GPU is below
/// [`AUTO_FREE_VRAM_FRACTION`] free. The GPU lane's real cards are comfortably
/// free, so without this seam the branch is unreachable in E2E and the note has
/// no live coverage. Synthesizing one near-full single-GPU reading makes that
/// path deterministic on the vLLM GPU lane (a non-APU host, so
/// [`vram_capacity_is_meaningful`] still holds) without touching the device the
/// engine actually launches on. Compiled only into the `e2e-test-hooks` binary;
/// production builds never see it.
#[cfg(feature = "e2e-test-hooks")]
fn simulated_low_vram_usage() -> Option<Vec<GpuVramUsage>> {
    let index: u32 = std::env::var("ROCM_E2E_FORCE_LOW_VRAM")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    // A single 24 GiB device with ~0.5 GiB free is well below the 0.90 free
    // threshold, so `gpu_low_memory_warning` fires; one GPU keeps
    // `vram_capacity_is_meaningful` honest on the non-APU vLLM lane.
    let total_mb = 24 * 1024;
    Some(vec![GpuVramUsage {
        index,
        used_mb: total_mb - 512,
        total_mb,
    }])
}

/// Per-GPU VRAM occupancy via `amd-smi metric --json`. Returns `None` when
/// amd-smi is unavailable or its output cannot be parsed.
fn gpu_vram_usage_amd_smi() -> Option<Vec<GpuVramUsage>> {
    let binary = rocm_core::resolve_amd_smi_binary();
    let output = ProcessCommand::new(&binary)
        .arg("metric")
        .arg("--json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let rows = parse_gpu_vram_usage(&value);
    if rows.is_empty() { None } else { Some(rows) }
}

/// Fallback per-GPU VRAM occupancy read straight from the amdgpu DRM sysfs
/// counters (`/sys/class/drm/card*/device/mem_info_vram_{total,used}`), used when
/// amd-smi is not installed. Linux-only; other platforms expose no such
/// interface and return `None`.
#[cfg(target_os = "linux")]
fn gpu_vram_usage_sysfs() -> Option<Vec<GpuVramUsage>> {
    let rows = read_drm_vram_usage(Path::new("/sys/class/drm"));
    if rows.is_empty() { None } else { Some(rows) }
}

#[cfg(not(target_os = "linux"))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn gpu_vram_usage_sysfs() -> Option<Vec<GpuVramUsage>> {
    None
}

/// Read per-GPU VRAM occupancy from the amdgpu DRM sysfs tree rooted at
/// `drm_dir`. Enumerates primary `card<N>` nodes (skipping connector sub-nodes
/// like `card0-DP-1`), keeps only AMD devices exposing readable
/// `mem_info_vram_{total,used}` counters, and assigns HIP ordinal 0 to the sole
/// AMD card to mirror HIP device ordering on a standard single-GPU host install.
///
/// HIP ordinals come from the KFD *compute* topology, not DRM card numbers, and
/// the two enumerations are only guaranteed to agree when there is exactly one
/// AMD DRM card: an APU passes the same AMD filter as a discrete GPU, so a host
/// pairing an APU with a dGPU can order `card<N>` differently than HIP's node
/// order, and that ordinal is fed to `HIP_VISIBLE_DEVICES` downstream. Rather
/// than guess, this returns no rows at all when more than one AMD card is
/// *found* — counted before the telemetry filters below, so a second AMD card
/// with an unreadable counter still trips the guard instead of leaving the
/// survivor mislabelled ordinal 0. Callers then fall back to service-state-only
/// auto-selection. Split from [`gpu_vram_usage_sysfs`] so it can be tested
/// against a planted sysfs layout.
#[cfg(any(target_os = "linux", test))]
fn read_drm_vram_usage(drm_dir: &Path) -> Vec<GpuVramUsage> {
    const BYTES_PER_MIB: u64 = 1024 * 1024;
    let Ok(entries) = fs::read_dir(drm_dir) else {
        return Vec::new();
    };
    // Every AMD card is counted here, *before* the telemetry filters below can
    // drop it, so the multi-card ordinal-ambiguity guard fires on how many AMD
    // cards exist rather than on how many produced readable counters.
    let mut amd_cards_found = 0usize;
    let mut cards: Vec<GpuVramUsage> = Vec::new();
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        if name
            .strip_prefix("card")
            .and_then(|digits| digits.parse::<u32>().ok())
            .is_none()
        {
            continue;
        }
        let device_dir = card_path.join("device");
        if !drm_device_is_amd(&device_dir) {
            continue;
        }
        amd_cards_found += 1;
        let Some(total_bytes) = read_sysfs_u64(&device_dir.join("mem_info_vram_total")) else {
            continue;
        };
        // An unreadable `used` counter is unknown occupancy, not zero: treating
        // it as `0` would make the card look 100% free, which is exactly what
        // `--gpu auto` prefers first. Skip the card instead of guessing.
        let Some(used_bytes) = read_sysfs_u64(&device_dir.join("mem_info_vram_used")) else {
            continue;
        };
        cards.push(GpuVramUsage {
            // The sole surviving AMD card is HIP ordinal 0 (guaranteed by the
            // multi-card guard below).
            index: 0,
            used_mb: used_bytes / BYTES_PER_MIB,
            total_mb: total_bytes / BYTES_PER_MIB,
        });
    }
    // More than one AMD card found: ascending `card<N>` order is not guaranteed
    // to match HIP's ordinal order (see doc comment above), so do not hand out
    // ordinals that might feed the wrong device into `HIP_VISIBLE_DEVICES`.
    if amd_cards_found > 1 {
        return Vec::new();
    }
    cards
}

/// Read a sysfs file whose entire contents are a single unsigned integer.
#[cfg(any(target_os = "linux", test))]
fn read_sysfs_u64(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

/// Whether a DRM `device` directory belongs to an AMD GPU.
///
/// Delegates rather than re-implementing. This probe and `rocm-core`'s KFD/DRM
/// count authority have to agree on what counts as an AMD card — if this one
/// recognises fewer, it under-counts and the multi-card ordinal guard it feeds
/// silently narrows — and a second copy of the test with a comment claiming the
/// two match is not agreement, it is a promise nothing enforces. Calling the
/// same function is.
#[cfg(any(target_os = "linux", test))]
fn drm_device_is_amd(device_dir: &Path) -> bool {
    rocm_core::is_amdgpu_device(device_dir)
}

/// Parse `amd-smi metric --json` output into per-GPU VRAM usage. Accepts both
/// the `{"gpu_data": [...]}` envelope and a bare top-level array, mirroring the
/// schema variance handled by the dashboard amd-smi collector.
///
/// A row reporting `total_vram: 0` is dropped rather than kept as a measurement
/// of an empty pool. `amd-smi` emits that shape when it enumerated the device
/// but could not read its memory controller (a driver hiccup, not a 0 GiB
/// GPU), and every other reader of this figure already treats zero as "not a
/// real reading": [`GpuVramUsage::free_fraction`] returns `None` on it rather
/// than reporting the GPU fully occupied. Keeping the row here would let
/// [`host_accelerator_memory`] construct `AcceleratorMemory::Dedicated(0.0)`,
/// which reads as "measured, and it is nothing" rather than "could not
/// measure" -- turning a diagnostic gap into a confident `Blocked` verdict.
fn parse_gpu_vram_usage(value: &serde_json::Value) -> Vec<GpuVramUsage> {
    let entries = value
        .get("gpu_data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.as_array());
    let Some(entries) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .enumerate()
        .filter_map(|(position, entry)| {
            let index = entry
                .get("gpu")
                .and_then(serde_json::Value::as_u64)
                .map_or(position as u32, |id| id as u32);
            let used_mb = entry
                .pointer("/mem_usage/used_vram/value")
                .and_then(serde_json::Value::as_u64)?;
            let total_mb = entry
                .pointer("/mem_usage/total_vram/value")
                .and_then(serde_json::Value::as_u64)?;
            if total_mb == 0 {
                return None;
            }
            Some(GpuVramUsage {
                index,
                used_mb,
                total_mb,
            })
        })
        .collect()
}

/// Build a serve-plan warning when a selected GPU is already heavily used, so
/// the user knows to free it or pick another `--gpu` before the engine fails on
/// insufficient VRAM. Returns `None` when telemetry is missing or the selected
/// GPUs are comfortably free.
fn gpu_low_memory_warning(gpu_indices: &[u32], vram: Option<&[GpuVramUsage]>) -> Option<String> {
    let vram = vram?;
    for &index in gpu_indices {
        // Skip indices without telemetry rather than abandoning the whole scan,
        // so a missing row for one GPU does not suppress warnings for others.
        let Some(usage) = vram.iter().find(|row| row.index == index) else {
            continue;
        };
        let Some(free) = usage.free_fraction() else {
            continue;
        };
        if free < AUTO_FREE_VRAM_FRACTION {
            let free_gib = (usage.total_mb.saturating_sub(usage.used_mb)) as f64 / 1024.0;
            let total_gib = usage.total_mb as f64 / 1024.0;
            return Some(format!(
                "warning: GPU {index} has only {free_gib:.1} GiB of {total_gib:.1} GiB free; \
                 serving may fail on VRAM. Free it or pick another with `--gpu <index|auto>`."
            ));
        }
    }
    None
}

/// Whether the reported VRAM capacity describes the memory the engine will
/// actually draw on.
///
/// An APU has no private VRAM. `amd-smi` reports the fixed BIOS carveout (often
/// ~4 GiB) as `total_vram`, while the allocator serves the model out of
/// GTT-backed system RAM — so on a 128 GiB Strix Halo, "only 1.6 GiB of 4.0 GiB
/// free" is measuring the wrong pool. It fires on healthy machines and would
/// stay quiet on a genuinely starved one, so a low-memory warning built on it
/// carries no information and is better withheld.
///
/// The single-GPU condition is what keeps this honest.
/// [`rocm_core::gfx_is_apu_family`] classifies a *part*, and host detection
/// yields one target for the whole machine, so on a laptop pairing an APU with
/// a discrete card an APU verdict would otherwise suppress a legitimate warning
/// about the discrete card. When more than one GPU is present we cannot
/// attribute the target to the selected ordinal, so the warning stands.
fn vram_capacity_is_meaningful(gfx_target: Option<&str>, gpu_count: usize) -> bool {
    if gpu_count != 1 {
        return true;
    }
    !gfx_target.is_some_and(rocm_core::gfx_is_apu_family)
}

/// The serve plan's low-VRAM warning, withheld on hosts where the VRAM figures
/// do not describe the memory the engine uses (see
/// [`vram_capacity_is_meaningful`]).
///
/// Resolves the gfx target only once a warning would actually be emitted. Host
/// GPU detection can probe sysfs/WSL — `serve` deliberately gates its own call
/// on whether engine selection needs it — so the quiet path must not start
/// paying for a lookup it will never use.
fn serve_gpu_low_memory_warning(
    gpu_indices: &[u32],
    vram: Option<&[GpuVramUsage]>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
) -> Option<String> {
    let warning = gpu_low_memory_warning(gpu_indices, vram)?;
    let gpu_count = vram.map_or(0, <[GpuVramUsage]>::len);
    let gfx_target = match host_gpu_summary {
        Some(summary) => summary.gfx_target.clone(),
        None => rocm_core::detect_host_gfx_target(),
    };
    vram_capacity_is_meaningful(gfx_target.as_deref(), gpu_count).then_some(warning)
}

/// GPU ordinals currently pinned by running rocm-cli managed/foreground
/// services, used to skip busy devices during `--gpu auto` selection.
fn busy_gpu_indices(paths: &AppPaths) -> Vec<u32> {
    let Ok(records) = load_managed_services(paths) else {
        return Vec::new();
    };
    let mut busy = Vec::new();
    for record in records {
        if !matches!(
            record.status.as_str(),
            "starting" | "running" | "recovering" | "ready"
        ) {
            continue;
        }
        for index in record.gpu_indices {
            if !busy.contains(&index) {
                busy.push(index);
            }
        }
    }
    busy
}

/// Best-effort count of local AMD GPUs via `amd-smi`. Returns `None` when
/// amd-smi is unavailable or its output cannot be parsed (callers then fall
/// back to conservative defaults).
fn detect_gpu_count() -> Option<usize> {
    let binary = rocm_core::resolve_amd_smi_binary();
    let output = ProcessCommand::new(&binary)
        .arg("list")
        .arg("--json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let count = value.as_array().map(Vec::len)?;
    if count == 0 { None } else { Some(count) }
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
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> EndpointReadiness {
    wait_for_service_http_ready_with_progress(
        engine,
        host,
        port,
        canonical_model_id,
        endpoint_api_key,
        timeout,
        &mut |_elapsed| {},
    )
}

/// Wait until the service can actually serve, reporting how far it got.
///
/// A health or model-listing endpoint answering is not enough to call a service
/// ready: engines advertise a model within seconds of accepting its name, while
/// the weights can take minutes to become usable, so a caller that sends its
/// first request on that signal gets a hang. Only [`EndpointReadiness::Serving`]
/// — an inference request that came back — means ready; a service that lists the
/// model without answering one is [`EndpointReadiness::Listing`], and the best
/// result seen before `timeout` is what gets returned.
///
/// Polls until `timeout` elapses, invoking `on_tick(elapsed)` once per iteration
/// so a caller can animate a spinner. Engine-neutral: `service_http_readiness_paths`
/// maps each engine to the right listing path. `endpoint_api_key` is sent as a
/// bearer token so both the listing check and the inference probe still succeed
/// against a public endpoint that requires authentication.
fn wait_for_service_http_ready_with_progress(
    engine: &str,
    host: &str,
    port: u16,
    canonical_model_id: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
    on_tick: &mut dyn FnMut(Duration),
) -> EndpointReadiness {
    let start = std::time::Instant::now();
    let endpoint = format_http_base_url(host, port);
    let mut best = EndpointReadiness::Unreachable;
    while start.elapsed() < timeout {
        let listed = service_http_readiness_paths(engine).iter().any(|path| {
            http_get_local_service(
                host,
                port,
                path,
                endpoint_api_key,
                Duration::from_millis(750),
            )
            .is_ok_and(|(status, body)| {
                service_http_readiness_response_ready(
                    engine,
                    path,
                    status,
                    &body,
                    canonical_model_id,
                )
            })
        });
        if listed {
            best = EndpointReadiness::Listing;
            if rocm_core::openai_chat_completion_probe(
                &endpoint,
                canonical_model_id,
                endpoint_api_key,
                rocm_core::INFERENCE_PROBE_TIMEOUT,
            )
            .unwrap_or(false)
            {
                return EndpointReadiness::Serving;
            }
        }
        on_tick(start.elapsed());
        thread::sleep(Duration::from_millis(250));
    }
    best
}

/// The record status matching how far the endpoint got.
///
/// `Listing` maps to `running`, not `starting`: the engine is up and loading
/// normally, and `rocmd` restarts a service left in `starting` past its stale
/// window, which would kill a slow-loading model mid-load.
const fn status_for_readiness(readiness: EndpointReadiness) -> &'static str {
    match readiness {
        EndpointReadiness::Serving => "ready",
        EndpointReadiness::Listing => "running",
        EndpointReadiness::Unreachable => "starting",
    }
}

fn service_http_readiness_paths(engine: &str) -> &'static [&'static str] {
    match engine {
        "lemonade" => &["/v1/health", "/v1/models"],
        _ => &["/v1/models", "/v1/health", "/health", "/healthz"],
    }
}

fn http_get_local_service(
    host: &str,
    port: u16,
    path: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<(u16, String)> {
    let deadline = std::time::Instant::now() + timeout;
    let mut stream = connect_tcp_stream(host, port, timeout)?;
    let host_header = format_host_port(host, port);
    // Authenticate the probe when the endpoint is protected; loopback endpoints
    // pass `None` and the header is omitted.
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\n{auth_header}Connection: close\r\n\r\n"
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .context("failed to write service readiness request")?;
    let response = read_http_response_bounded(&mut stream, deadline)
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
    let deadline = std::time::Instant::now() + timeout;
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
    let response = read_http_response_bounded(&mut stream, deadline)
        .context("failed to read service response")?;
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
                name_matches && (!require_rocm_backend || service_model_gpu_ready(model))
            })
        })
}

/// Whether a lemonade `/v1/models` entry is served on GPU. A stock `llama-server`
/// (direct-serve) entry has no `recipe_options` and is accepted (that path only runs
/// GPU backends); a Lemonade-router entry carries `recipe_options`, so it must report a
/// ROCm backend — which keeps a registered-but-unloaded model (empty `recipe_options`)
/// from reading as ready.
fn service_model_gpu_ready(model: &serde_json::Value) -> bool {
    match model.get("recipe_options") {
        None => true,
        Some(_) => service_model_reports_rocm_backend(model),
    }
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
        "storage",
        "engines",
        "model",
        "models",
        "serve",
        "comfyui",
        "comfy",
        "services",
        "remote",
        "automations",
        "config",
        "logs",
        "daemon",
        "dash",
        "bench",
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
pub(crate) mod tests {
    use std::cell::RefCell;

    use rocm_core::browser::Opener;
    use rocm_core::report_delivery::Delivery;
    use rocm_core::resolve_builtin_model_recipe;

    use super::perform_delivery;

    /// An opener that records rather than opens, and can be told to fail.
    ///
    /// The whole reason the opener is a trait: the real one spawns a browser
    /// against whatever desktop exists, so neither "it was opened" nor "it was
    /// deliberately not opened" can be observed in CI without this.
    struct RecordingOpener {
        opened: RefCell<Vec<String>>,
        fails: bool,
    }

    impl RecordingOpener {
        fn working() -> Self {
            Self {
                opened: RefCell::new(Vec::new()),
                fails: false,
            }
        }
        fn broken() -> Self {
            Self {
                opened: RefCell::new(Vec::new()),
                fails: true,
            }
        }
        fn opened(&self) -> Vec<String> {
            self.opened.borrow().clone()
        }
    }

    impl Opener for RecordingOpener {
        fn open(&self, url: &str) -> anyhow::Result<()> {
            self.opened.borrow_mut().push(url.to_owned());
            if self.fails {
                anyhow::bail!("no browser here");
            }
            Ok(())
        }
    }

    /// Nothing is opened unless the decision was to open.
    ///
    /// The assertion that matters is on the opener, not on the wording. A
    /// message saying no browser was started is satisfied by any string; an
    /// opener that recorded nothing is the actual claim.
    #[test]
    fn a_delivery_that_is_not_an_open_never_reaches_the_browser() {
        let delivery = Delivery::Show("mailto:nobody@example.invalid".to_owned());
        let opener = RecordingOpener::working();
        let said = perform_delivery(&delivery, &opener);

        assert!(
            opener.opened().is_empty(),
            "a mail client was started for {delivery:?}, which is the one thing this path must \
             not do on a machine the user is holding over SSH"
        );
        assert!(
            said.contains("Nothing has been sent"),
            "the user has to be told nothing left the machine: {said}"
        );
        // A machine in this state often cannot act on a `mailto:` at all, so
        // the address has to be readable on its own, not only inside the link.
        assert!(
            said.contains(rocm_core::report_delivery::DESTINATION),
            "a user who has to send the mail by hand needs the address: {said}"
        );
    }

    /// Opening is what an open decision does, and the user is told it is not
    /// filed yet.
    #[test]
    fn an_open_decision_reaches_the_browser_and_is_still_not_a_send() {
        let url = "https://example.invalid/new?body=x";
        let opener = RecordingOpener::working();
        let said = perform_delivery(&Delivery::Open(url.to_owned()), &opener);

        assert_eq!(
            opener.opened(),
            vec![url.to_owned()],
            "premise failed: an open decision must reach the opener, otherwise the cases above \
             are satisfied by never opening anything"
        );
        assert!(
            said.contains("only when you send it"),
            "opening a prefilled mail is not sending it, and the user has to know which one \
             happened: {said}"
        );
    }

    /// A browser that will not start still leaves the user the link.
    #[test]
    fn a_browser_that_fails_to_start_still_hands_the_user_the_link() {
        let url = "https://example.invalid/new?body=x";
        let said = perform_delivery(&Delivery::Open(url.to_owned()), &RecordingOpener::broken());

        assert!(
            said.contains(url),
            "the link is the whole of what this offers, so a failed browser must not lose it: \
             {said}"
        );
        assert!(said.contains("Nothing has been sent"));
    }

    /// A diagnosis report holding exactly one finding.
    ///
    /// `has_match` is passed independently of the score on purpose: the point
    /// under test is that the two are read together, so a fixture that derived
    /// one from the other could not express the case being guarded against.
    fn report_of(
        has_match: bool,
        id: &str,
        score: i32,
        fix: Option<rocm_core::Fix>,
    ) -> rocm_core::DiagnoseReport {
        rocm_core::DiagnoseReport {
            has_match,
            matched: vec![rocm_core::Diagnosis {
                id: id.to_owned(),
                title: "under test".to_owned(),
                score,
                evidence: Vec::new(),
                fix,
            }],
            min_score_for_match: 50,
            high_confidence_threshold: 80,
            route_when_no_match: rocm_core::diagnose::Route {
                target: String::new(),
                url: String::new(),
            },
            out_of_scope: None,
            model: None,
        }
    }

    /// The entry a report names is one the diagnosis established, not merely
    /// the strongest signal it saw.
    ///
    /// This is a wiring test, not a logic one. `established_entry` is correct in
    /// itself; what it could get wrong is being handed `matched.first()`
    /// unconditionally. Several checkers open with a nonzero score for a
    /// situation that is only potentially relevant, so a healthy machine
    /// produces a `matched` list full of sub-threshold entries — and a report
    /// naming one of those would look like an established cause to every
    /// counter downstream, with nothing able to tell the difference afterwards.
    #[test]
    fn a_report_names_an_established_cause_and_not_the_loudest_weak_signal() {
        let weak_only = report_of(false, "fix-10-container", 25, None);
        assert_eq!(
            established_entry(&weak_only),
            (None, false),
            "nothing cleared the bar, so the report has no entry to name"
        );

        // Non-vacuity: an established cause must come through, or the assertion
        // above is satisfied by never naming anything.
        let established = report_of(true, "fix-6-path", 90, Some(rocm_core::Fix::default()));
        assert_eq!(
            established_entry(&established),
            (Some("fix-6-path"), true),
            "an established cause with a fix is exactly what a report is for"
        );
    }
    use std::process::ExitCode;

    /// `Ok(())` must map to a clean exit so `rocm`'s successful commands don't
    /// regress to a nonzero code.
    #[test]
    fn exit_code_for_ok_is_success() {
        assert_eq!(super::exit_code_for(Ok(())), ExitCode::SUCCESS);
    }

    /// `fix()`'s marker error must carry its exact code through, since that
    /// code (2/3/4/5) is part of `rocm fix`'s documented contract.
    #[test]
    fn exit_code_for_fix_exit_code_carries_the_code() {
        let err = anyhow::Error::new(super::FixExitCode(3));
        assert_eq!(super::exit_code_for(Err(err)), ExitCode::from(3));
    }

    /// `ClapExitCode` exists so a usage/parse error can reach `main()` through
    /// the ordinary return path (letting `_log_guard` drop) instead of
    /// `clap::Error::exit()` calling `std::process::exit` mid-stack. Guard the
    /// downcast the same way `exit_code_for_fix_exit_code_carries_the_code`
    /// guards `FixExitCode`'s.
    #[test]
    fn exit_code_for_clap_exit_code_carries_the_code() {
        let err = anyhow::Error::new(super::ClapExitCode(2));
        assert_eq!(super::exit_code_for(Err(err)), ExitCode::from(2));
    }

    /// `run()`'s mistyped-subcommand branch and `parse_cli()` both route a
    /// real `clap::Error` through the production `clap_exit_code` helper
    /// (not a hand-built `ClapExitCode`), so this calls that same helper on a
    /// real parse failure to exercise the actual
    /// `clap parse failure -> clap_exit_code -> ClapExitCode -> exit_code_for`
    /// chain end to end. Reverting `clap_exit_code` to call `err.exit()`
    /// directly, or dropping its use from either call site, breaks this.
    #[test]
    fn clap_error_exit_code_survives_the_clap_exit_code_round_trip() {
        let err = command_invocation_error(&["instal".to_owned()])
            .expect("`instal` should read as a mistyped subcommand");
        let expected_code = err.exit_code();
        let result: Result<()> = Err(super::clap_exit_code(err));
        assert_eq!(
            super::exit_code_for(result),
            ExitCode::from(expected_code as u8)
        );
    }

    /// `parse_cli()` itself reads `std::env::args_os()` (via
    /// `Command::try_get_matches()`), which a unit test cannot redirect, so
    /// this exercises the same `cli_command()` builder with an explicit argv
    /// instead: an unrecognised flag is the common case `parse_cli()` was
    /// still routing through `err.exit()` before it switched from
    /// `get_matches()` to `try_get_matches()`.
    #[test]
    fn cli_command_rejects_unknown_flag_through_clap_exit_code() {
        let err = super::cli_command()
            .try_get_matches_from(["rocm", "--this-flag-does-not-exist"])
            .expect_err("an unknown flag must be a parse error");
        let expected_code = err.exit_code();
        let result: Result<()> = Err(super::clap_exit_code(err));
        assert_eq!(
            super::exit_code_for(result),
            ExitCode::from(expected_code as u8)
        );
    }

    /// Any other error must still fail with exit 1, matching what
    /// `Result<(), anyhow::Error>`'s `Termination` impl already does today for
    /// every subcommand other than `fix`.
    #[test]
    fn exit_code_for_generic_error_is_failure() {
        let err = anyhow::anyhow!("boom");
        assert_eq!(super::exit_code_for(Err(err)), ExitCode::FAILURE);
    }

    /// Exercises the real `dispatch -> fix -> FixExitCode -> exit_code_for`
    /// chain end to end, not just `exit_code_for` in isolation. Guards against
    /// a future change at the `Command::Fix` dispatch arm (e.g. discarding the
    /// error into a fresh `anyhow!(...)`) silently breaking the downcast and
    /// falling through to the generic exit 1.
    #[test]
    fn dispatch_carries_fixs_exit_code_through_to_exit_code_for() {
        // Skip the startup update check: it's a side effect unrelated to what
        // this test verifies, and could otherwise touch the network. Goes
        // through `ScopedTestEnv` so it's serialized against every other test
        // that touches process env and restored on drop even on panic.
        let mut env = ScopedTestEnv::new();
        env.set("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK", "1");
        let cli = super::Cli {
            command: Some(super::Command::Fix {
                fix_id: Some("fix-does-not-exist".to_owned()),
                yes: true,
                dry_run: false,
                device_index: None,
                json: false,
            }),
        };
        let result = super::dispatch(cli);
        drop(env);
        assert_eq!(super::exit_code_for(result), ExitCode::from(2));
    }

    /// A cache that has moved inside a directory uninstall already removes must
    /// not be reported as "not removed" — the note would be false.
    #[test]
    fn shared_cache_notes_skip_paths_already_being_removed() {
        let root = std::env::temp_dir().join(format!("rocm-shared-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let data = root.join("data");
        let inside = data.join("uv-cache");
        std::fs::create_dir_all(&inside).unwrap();

        let notes = super::shared_cache_notes_for(
            &[root.join("config"), data, root.join("cache")],
            &[(inside, "the uv package cache")],
        );

        assert!(
            notes.is_empty(),
            "a cache inside a removed directory must not be reported: {notes:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A cache that exists outside everything being removed is reported, with
    /// its path, so the plan does not imply a clean slate it will not deliver.
    #[test]
    fn shared_cache_notes_report_paths_left_behind() {
        let root = std::env::temp_dir().join(format!("rocm-shared-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let outside = root.join("elsewhere").join("uv");
        std::fs::create_dir_all(&outside).unwrap();

        let notes = super::shared_cache_notes_for(
            &[root.join("data")],
            &[(outside.clone(), "the uv package cache")],
        );

        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("not removed"), "{notes:?}");
        assert!(
            notes[0].contains(&outside.display().to_string()),
            "{notes:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A cache that does not exist is not worth mentioning.
    #[test]
    fn shared_cache_notes_ignore_missing_paths() {
        let missing = std::env::temp_dir().join("rocm-definitely-not-here-12345");
        let notes = super::shared_cache_notes_for(
            &[std::env::temp_dir()],
            &[(missing, "the uv package cache")],
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    use super::*;
    use serde_json::json;

    use crate::runtime_install_finalize::tests::finish_sdk_install_with;
    use crate::runtimes_cmd::runtime_manifest_path;
    use crate::test_support::write_test_pip_runtime;

    /// Serializes and isolates tests that read or mutate process-global env vars.
    ///
    /// `std::env::set_var`/`remove_var` are `unsafe` in edition 2024 because
    /// concurrent env mutation races with any concurrent `env::var` read anywhere
    /// in the process, and `cargo test` runs these functions multi-threaded in one
    /// binary. A unique variable name does not make a mutation safe — the hazard is
    /// the mutation racing a read, not a name collision. Every test that touches
    /// env therefore holds this one process-wide lock, and each mutation is saved
    /// and restored on drop so it cannot leak into another test.
    pub(crate) struct ScopedTestEnv {
        _lock: std::sync::MutexGuard<'static, ()>,
        saved: Vec<(String, Option<String>)>,
    }

    impl ScopedTestEnv {
        pub(crate) fn new() -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let lock = LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Self {
                _lock: lock,
                saved: Vec::new(),
            }
        }

        /// Clear the two AMD driver-install override vars for the duration of the
        /// test, so a value exported in the developer's or runner's shell cannot
        /// leak into a plan and make its resolved `repo_version`/package release
        /// disagree with the assertion.
        pub(crate) fn with_amd_overrides_cleared() -> Self {
            let mut env = Self::new();
            env.clear("ROCM_CLI_AMDGPU_VERSION");
            env.clear("ROCM_CLI_AMDGPU_PACKAGE_RELEASE");
            env
        }

        fn save(&mut self, key: &str) {
            if !self.saved.iter().any(|(saved_key, _)| saved_key == key) {
                self.saved.push((key.to_owned(), std::env::var(key).ok()));
            }
        }

        #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
        pub(crate) fn set(&mut self, key: &str, value: &str) {
            self.save(key);
            unsafe {
                std::env::set_var(key, value);
            }
        }

        #[allow(unsafe_code)] // std::env::remove_var is unsafe in edition 2024
        pub(crate) fn clear(&mut self, key: &str) {
            self.save(key);
            unsafe {
                std::env::remove_var(key);
            }
        }
    }

    impl Drop for ScopedTestEnv {
        #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
        fn drop(&mut self) {
            for (key, previous) in self.saved.iter().rev() {
                unsafe {
                    match previous {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

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

    /// EAI-7362: the top-level `rocm --help` command list must be alphabetical,
    /// including clap's implicit `help` entry (it must sort into place, not be
    /// pinned last).
    ///
    /// Renders the real help text (not just the command model) so the assertion
    /// covers what clap actually prints.
    #[test]
    fn top_level_help_lists_commands_alphabetically() {
        let help = cli_command().render_long_help().to_string();

        let commands_section = help
            .split_once("Commands:")
            .expect("help output has a Commands section")
            .1;

        // Each command entry starts at a fixed two-space indent; wrapped
        // description lines are indented further, so a non-space third column
        // uniquely identifies a command row. Stop at the blank line that ends
        // the section.
        let names: Vec<String> = commands_section
            .lines()
            .skip_while(|line| line.trim().is_empty())
            .take_while(|line| !line.trim().is_empty())
            .filter(|line| line.starts_with("  ") && !line.starts_with("   "))
            .filter_map(|line| line.split_whitespace().next())
            .map(str::to_owned)
            .collect();

        assert!(
            names.iter().any(|name| name == "help"),
            "expected the implicit `help` entry to appear in the list, got {names:?}"
        );
        assert!(
            names.len() > 1,
            "expected several visible subcommands, got {names:?}"
        );

        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            names, sorted,
            "top-level `rocm --help` commands are not alphabetized"
        );
    }

    #[test]
    fn every_framework_choice_reaches_the_library() {
        // The defect this flag fixes was not a missing enum -- the library had
        // all four variants -- but that no caller could select one. Pin the
        // mapping so a renamed variant cannot silently fall back to the default.
        use rocm_core::FrameworkProbe;
        assert_eq!(
            FrameworkProbe::from(FrameworkArg::Auto),
            FrameworkProbe::Auto
        );
        assert_eq!(
            FrameworkProbe::from(FrameworkArg::Pytorch),
            FrameworkProbe::PyTorch
        );
        assert_eq!(
            FrameworkProbe::from(FrameworkArg::LlamaCpp),
            FrameworkProbe::LlamaCpp
        );
        assert_eq!(
            FrameworkProbe::from(FrameworkArg::Skip),
            FrameworkProbe::Skip
        );
    }

    #[test]
    fn the_framework_choice_is_offered_on_the_command_line() {
        // Guards the actual regression: the variants existed in the library and
        // were unreachable from here. A user must be able to see and pass them.
        let help = Cli::command()
            .find_subcommand_mut("examine")
            .expect("examine subcommand")
            .render_long_help()
            .to_string();
        for choice in ["auto", "pytorch", "llama-cpp", "skip"] {
            assert!(
                help.contains(choice),
                "`{choice}` must be offered by `rocm examine --help`:\n{help}"
            );
        }
    }

    #[test]
    fn install_sdk_help_describes_the_gate_as_replacing_the_active_default() {
        // `rocm install sdk --help` is the most-read description of the `--yes`
        // gate, and it is the one surface a "reword every site" pass can miss —
        // this branch's own history has it being missed once and corrected in a
        // follow-up. So the assertions read the `yes` argument's own help text
        // rather than the whole rendered page: the sibling
        // `--approve-replacing-active-default` doc independently satisfies a
        // page-wide "active default" match, which would keep a reverted `--yes`
        // doc green.
        //
        // The effect is a displacement, not a deletion: in the default managed
        // install root `runtime_key` embeds the resolved version, so an upgrade
        // or downgrade lands in its own install root and the previous install
        // stays on disk — only the active default moves. Claiming an overwrite
        // here would promise a deletion that does not happen and contradict the
        // prompt and the README.
        let mut command = Cli::command();
        let sdk = command
            .find_subcommand_mut("install")
            .expect("install subcommand")
            .find_subcommand_mut("sdk")
            .expect("install sdk subcommand");
        let yes = sdk
            .get_arguments()
            .find(|arg| arg.get_id() == "yes")
            .expect("`install sdk` must offer --yes");
        let yes_help = yes
            .get_long_help()
            .or_else(|| yes.get_help())
            .expect("--yes must be documented")
            .to_string();

        assert!(
            yes_help.contains("active default"),
            "`--yes` must document itself as approving a replacement of the \
             active default:\n{yes_help}"
        );
        assert!(
            !yes_help.to_lowercase().contains("overwrit"),
            "`--yes` must not claim an overwrite; an upgrade or downgrade leaves \
             the previous install on disk:\n{yes_help}"
        );
    }

    #[test]
    fn install_sdk_help_separates_the_two_consents_yes_carries() {
        // `--yes` approves two unrelated things: replacing the active default
        // runtime, and running `sudo` for required system packages. The whole
        // point of the narrow flag is that a caller with no terminal can grant
        // the first without the second, so the help has to say so — a reader who
        // believes it is a synonym for `--yes` will reach for `--yes` from a
        // script and get a sudo prompt nothing can answer.
        let help = Cli::command()
            .find_subcommand_mut("install")
            .expect("install subcommand")
            .find_subcommand_mut("sdk")
            .expect("install sdk subcommand")
            .render_long_help()
            .to_string();
        assert!(
            help.contains("--approve-replacing-active-default"),
            "`rocm install sdk --help` must document the narrow consent flag:\n{help}"
        );
        assert!(
            help.contains("does not approve system-package installs"),
            "`rocm install sdk --help` must say the narrow flag excludes \
             system-package installs:\n{help}"
        );
    }

    #[test]
    fn update_apply_approval_never_credits_the_inert_yes_flag() {
        // Pins `SdkInstallApprovalSource::UpdateApply`: the update path is
        // preapproved, but it must never print a line crediting `--yes`.
        //
        // `rocm update` now *does* take a `--yes` flag, added for consistency
        // with the other mutating commands, and the deliberate decision the
        // flag's arrival called for has been made: the behaviour does not
        // change. That flag is inert by its own doc comment — applying never
        // prompts — and the dispatch above discards it (`yes: _`), so it grants
        // nothing. Crediting it would claim an approval the user never gave,
        // and on `rocm install sdk` `--yes` additionally approves running
        // `sudo`, so the claim would be doubly wrong.
        //
        // This replaces an assertion that `rocm update --help` contained no
        // `--yes` at all: a proxy for the invariant that only held while the
        // flag was absent. Pin the invariant itself so this still fails if the
        // update path is ever made to credit the flag.
        for activates in [true, false] {
            let line = therock::preapproved_install_line(
                therock::SdkInstallApprovalSource::UpdateApply { activates },
                "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
                "7.14.0",
            );
            assert!(
                !line.contains("--yes"),
                "`rocm update --apply` (activates={activates}) credited --yes, \
                 but that flag grants it nothing: {line}"
            );
            assert!(
                line.starts_with("Requested by `rocm update --apply"),
                "the update path must name itself as the approval source: {line}"
            );
        }
    }

    #[test]
    fn out_of_scope_commands_are_marked_preview_in_help() {
        let help = Cli::command().render_long_help().to_string();
        for command in ["chat", "comfyui", "automations"] {
            let line = help
                .lines()
                .find(|line| line.trim_start().starts_with(command))
                .unwrap_or_else(|| panic!("`{command}` missing from help:\n{help}"));
            assert!(
                line.contains("[preview]"),
                "`{command}` is outside the Tech Preview scope and must say so:\n{line}"
            );
        }
    }

    #[test]
    fn runtimes_help_uses_the_runtime_noun_throughout() {
        // `comfyui install`'s selection errors steer the user to `rocm runtimes`
        // and say "ROCm runtime". The help for the command they land on must use
        // the same noun — including its own about line, which `rocm runtimes
        // --help` prints above the subcommand list and which the rename missed
        // while every subcommand below it already said "runtime". The approval
        // modal raised by a mutating `rocm runtimes …` from chat says the same
        // noun too, but clap doesn't render that title — see
        // `chat::tests::runtimes_activate_approval_modal_uses_the_runtime_noun`.
        let help = Cli::command()
            .find_subcommand_mut("runtimes")
            .expect("runtimes subcommand")
            .render_long_help()
            .to_string();
        assert!(
            help.contains("ROCm runtimes"),
            "`rocm runtimes --help` should describe itself with the `runtime` noun:\n{help}"
        );
        assert!(
            !help.contains("ROCm install"),
            "`rocm runtimes --help` must not reintroduce the `ROCm install` noun:\n{help}"
        );
    }

    #[test]
    fn the_preview_marker_is_explained_and_not_contradicted() {
        let help = Cli::command().render_long_help().to_string();
        // A marker nobody can interpret is no better than no marker.
        assert!(
            help.contains("Commands marked [preview]"),
            "the marker must be explained in the help footer:\n{help}"
        );
        // A command cannot be promoted as a headline example while being marked
        // unfinished — that is the contradiction this pairing exists to prevent.
        let examples = help
            .split_once("EXAMPLES:")
            .map(|(_, rest)| rest)
            .unwrap_or_default();
        for command in ["chat", "comfyui", "automations"] {
            assert!(
                !examples.contains(&format!("rocm {command}")),
                "`{command}` is marked preview, so it must not headline the examples:\n{examples}"
            );
        }
    }

    /// An `ExamineSummary` with the install-reporting fields under test and
    /// everything else inert.
    fn summary_with_legacy(
        managed_runtime_count: usize,
        legacy_status: &str,
        legacy_version: Option<&str>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: rocm_core::DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
            },
            legacy_rocm: rocm_core::LegacyRocmSummary {
                status: legacy_status.to_owned(),
                paths: Vec::new(),
                detail: None,
                version: legacy_version.map(str::to_owned),
            },
            wsl: None,
            managed_runtime_count,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        }
    }

    #[test]
    fn header_does_not_claim_nothing_is_installed_when_rocm_is_present() {
        // The reported case: ROCm on the machine, none of it CLI-managed. The
        // header counted only managed runtimes, so it said "No ROCm installs
        // saved yet" — read as "nothing is installed".
        let header = render_examine_plain_header(&summary_with_legacy(
            0,
            "detected_unmanaged",
            Some("7.14.0"),
        ));
        assert!(
            header.contains("existing ROCm 7.14.0 found"),
            "the detected install and its version must appear:\n{header}"
        );
        assert!(
            header.contains("left unmanaged by design"),
            "it must read as a deliberate choice, not an oversight:\n{header}"
        );
    }

    #[test]
    fn header_reports_a_detected_install_whose_version_is_unknown() {
        // Degrades to naming the install without inventing a version.
        let header =
            render_examine_plain_header(&summary_with_legacy(0, "detected_unmanaged", None));
        assert!(
            header.contains("existing ROCm install found"),
            "the install must still be reported:\n{header}"
        );
        assert!(
            !header.contains("<unknown>"),
            "an unknown version must not leak a placeholder into the summary:\n{header}"
        );
    }

    #[test]
    fn header_accounts_for_managed_and_existing_installs_together() {
        let header = render_examine_plain_header(&summary_with_legacy(
            2,
            "detected_unmanaged",
            Some("6.4.1"),
        ));
        assert!(
            header.contains("2 ROCm installs saved"),
            "the managed count must survive:\n{header}"
        );
        assert!(
            header.contains("existing ROCm 6.4.1 found"),
            "the unmanaged install must be reported alongside it:\n{header}"
        );
    }

    #[test]
    fn header_still_says_nothing_is_installed_when_nothing_is() {
        // The wording only changes when there is something to report.
        let header = render_examine_plain_header(&summary_with_legacy(0, "not_detected", None));
        assert!(
            header.contains("No ROCm installs saved yet"),
            "an empty machine must keep the original wording:\n{header}"
        );
        assert!(
            !header.contains("existing ROCm"),
            "nothing should be claimed on an empty machine:\n{header}"
        );
    }

    /// Regression test for the engine child-stdin write under the process-wide
    /// `SIG_DFL` that `main` installs via [`reset_sigpipe`]. If an engine child
    /// exits before reading its stdin, the parent's write to that pipe must
    /// surface as a `BrokenPipe` error — handled by the caller's diagnostics —
    /// rather than killing `rocm` with SIGPIPE before those diagnostics run.
    ///
    /// The bug only reproduces under `SIG_DFL`; the test harness leaves SIGPIPE
    /// at Rust's default `SIG_IGN`, and flipping it process-wide would race
    /// sibling tests. So this re-execs itself in fresh processes that install
    /// `SIG_DFL` first, and checks BOTH directions:
    /// - unguarded write to a dead child's stdin is fatal (killed by SIGPIPE) —
    ///   this is the regression the plain `SIG_DFL` reset introduced;
    /// - the same write wrapped in [`with_sigpipe_ignored`] survives and returns
    ///   an error instead — this is the fix.
    ///
    /// Each child reaps its short-lived grandchild before writing, so the read
    /// end is closed deterministically and the outcome does not race.
    #[cfg(unix)]
    #[test]
    fn engine_stdin_write_under_sig_dfl_is_guarded_not_fatal() {
        use std::os::unix::process::ExitStatusExt as _;

        // Grandchild helper: reproduce main()'s SIG_DFL, spawn a process that
        // exits immediately, reap it, then write to its now-closed stdin. The
        // "guarded" arm wraps the write; the "unguarded" arm does not.
        if let Some(mode) = std::env::var_os("ROCM_TEST_SIGPIPE_ARM") {
            reset_sigpipe();
            let mut grandchild = ProcessCommand::new("true")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn a process that exits immediately");
            let mut stdin = grandchild.stdin.take().expect("grandchild stdin");
            grandchild.wait().expect("reap grandchild");
            // Heap-allocated so the buffer doesn't trip clippy::large-stack-arrays;
            // it must exceed the pipe buffer so the write actually reaches the
            // closed read end rather than being swallowed by kernel buffering.
            let payload = vec![b'x'; 64 * 1024];
            if mode == "guarded" {
                let result = with_sigpipe_ignored(|| stdin.write_all(&payload));
                assert!(
                    result.is_err(),
                    "guarded write to a dead pipe should return BrokenPipe, not succeed"
                );
            } else {
                // Unguarded: under SIG_DFL this write delivers SIGPIPE and never
                // returns. If the process somehow survives, exit 0 so the parent's
                // "should have been signalled" assertion fails loudly.
                let _ = stdin.write_all(&payload);
            }
            return;
        }

        let exe = std::env::current_exe().expect("current test executable");
        let run_arm = |mode: &str| -> ExitStatus {
            ProcessCommand::new(&exe)
                .args([
                    "tests::engine_stdin_write_under_sig_dfl_is_guarded_not_fatal",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_SIGPIPE_ARM", mode)
                .status()
                .expect("re-exec the test arm in a child process")
        };

        let unguarded = run_arm("unguarded");
        assert_eq!(
            unguarded.signal(),
            Some(libc::SIGPIPE),
            "sanity check: an unguarded stdin write under SIG_DFL must be killed by \
             SIGPIPE (got {unguarded:?}); if not, the test no longer proves the guard matters"
        );

        let guarded = run_arm("guarded");
        assert!(
            guarded.success(),
            "guarded child-stdin write was fatal under SIG_DFL ({guarded:?}) — \
             SIGPIPE reached the process instead of surfacing as an error"
        );
    }

    #[test]
    fn bench_load_rejects_zero_and_unbounded_numeric_arguments() {
        for args in [
            [
                "rocm",
                "bench",
                "load",
                "--endpoint",
                "http://localhost:8000",
                "--concurrency",
                "0",
            ]
            .as_slice(),
            [
                "rocm",
                "bench",
                "load",
                "--endpoint",
                "http://localhost:8000",
                "--isl",
                "32769",
            ]
            .as_slice(),
            [
                "rocm",
                "bench",
                "load",
                "--endpoint",
                "http://localhost:8000",
                "--osl",
                "32769",
            ]
            .as_slice(),
            [
                "rocm",
                "bench",
                "load",
                "--endpoint",
                "http://localhost:8000",
                "--requests",
                "10001",
            ]
            .as_slice(),
        ] {
            assert!(
                Cli::try_parse_from(args).is_err(),
                "accepted invalid args: {args:?}"
            );
        }
    }

    fn serve_possible_values(arg_id: &str) -> Vec<String> {
        let cli = Cli::command();
        let serve = cli
            .find_subcommand("serve")
            .expect("serve subcommand exists");
        serve
            .get_arguments()
            .find(|arg| arg.get_id().as_str() == arg_id)
            .unwrap_or_else(|| panic!("serve has no `{arg_id}` argument"))
            .get_possible_values()
            .iter()
            .map(|value| value.get_name().to_owned())
            .collect()
    }

    // Both `--engine` and `--device` restrict their input to a fixed set via a
    // clap `value_parser`/`ValueEnum`, so invalid input is rejected as a usage
    // error (exit code 2) with the accepted choices listed. Values are advertised
    // in `--help` and shell completion structurally (not a hand-written doc
    // string), with one deliberate exception: `--device cpu_only` is
    // `#[value(hide = true)]` — still accepted so its exit-1 rejection message
    // survives, but kept out of help and completion. The sync tests below keep the
    // advertised lists honest; `serve_device_help_lists_match_device_policy_names`
    // compares against the full `DevicePolicy` set (hidden entries included, via
    // `get_possible_values`), so a dropped or renamed variant still fails.
    #[test]
    fn serve_engine_help_lists_match_engine_inventory() {
        let mut listed = serve_possible_values("engine");
        let mut expected: Vec<String> = builtin_engine_inventory()
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect();
        listed.sort();
        expected.sort();
        assert_eq!(
            listed, expected,
            "serve --engine possible-values must stay in sync with builtin_engine_inventory()"
        );
    }

    #[test]
    fn serve_device_help_lists_match_device_policy_names() {
        let mut listed = serve_possible_values("device");
        let mut expected: Vec<String> = [
            DevicePolicy::GpuRequired,
            DevicePolicy::GpuPreferred,
            DevicePolicy::CpuOnly,
        ]
        .iter()
        .map(|policy| device_policy_name(policy).to_owned())
        .collect();
        listed.sort();
        expected.sort();
        assert_eq!(
            listed, expected,
            "serve --device possible-values must stay in sync with DevicePolicy names"
        );
    }

    #[test]
    fn serve_rejects_unknown_device_policy_as_usage_error() {
        // An invalid `--device` must fail clap's value validation (a usage error,
        // exit code 2) rather than parsing as a free-form string and failing later
        // in application logic (exit code 1). This keeps `--device` consistent with
        // every other enum-style argument and lists the valid choices in the error.
        let error = parse_serve(&["--device", "bogus"]).expect_err("invalid device rejected");
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn serve_accepts_device_policy_values_and_aliases() {
        for (value, expected) in [
            ("gpu_required", DevicePolicyArg::GpuRequired),
            ("gpu_preferred", DevicePolicyArg::GpuPreferred),
            ("cpu_only", DevicePolicyArg::CpuOnly),
            // Historical aliases stay accepted for backward compatibility.
            ("auto", DevicePolicyArg::GpuRequired),
            ("gpu", DevicePolicyArg::GpuRequired),
            ("cpu", DevicePolicyArg::CpuOnly),
        ] {
            let cli = parse_serve(&["--device", value]).expect("device value parses");
            match cli.command {
                Some(Command::Serve { device, .. }) => {
                    assert_eq!(device, Some(expected), "device value `{value}`");
                }
                other => panic!("expected Serve, got {other:?}"),
            }
        }
    }

    #[test]
    fn device_policy_arg_maps_through_parse_device_policy() {
        // `as_policy_str` feeds the parsed variant back through
        // `parse_device_policy`, the single choke point enforcing the strict
        // no-CPU policy. Guard the mapping so a mis-typed arm (e.g. `CpuOnly =>
        // "gpu_required"`) cannot silently turn the deliberate CPU rejection into
        // a GPU-required serve while every other test stays green.
        assert_eq!(
            parse_device_policy(Some(DevicePolicyArg::GpuRequired.as_policy_str()))
                .expect("gpu_required parses"),
            DevicePolicy::GpuRequired
        );
        assert_eq!(
            parse_device_policy(Some(DevicePolicyArg::GpuPreferred.as_policy_str()))
                .expect("gpu_preferred parses"),
            DevicePolicy::GpuRequired
        );
        // `cpu_only` must still be rejected outright rather than mapped to a
        // GPU policy.
        assert!(
            parse_device_policy(Some(DevicePolicyArg::CpuOnly.as_policy_str())).is_err(),
            "cpu_only must be rejected"
        );
    }

    fn parse_runtimes(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["rocm", "runtimes"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv)
    }

    /// The two restart flags as `activate` and `rollback` parsed them.
    fn runtimes_restart_flags(args: &[&str]) -> (bool, bool) {
        match parse_runtimes(args).expect("flags parse").command {
            Some(Command::Runtimes {
                command:
                    Some(
                        RuntimesCommand::Activate {
                            restart_services,
                            yes,
                            ..
                        }
                        | RuntimesCommand::Rollback {
                            restart_services,
                            yes,
                        },
                    ),
            }) => (restart_services, yes),
            other => panic!("expected runtimes activate or rollback, got {other:?}"),
        }
    }

    // `--yes` approves a restart, so on its own it asks for something that was
    // never requested. Mutation: removing `requires = "restart_services"` from
    // either subcommand's `yes` makes that subcommand's argv parse and fails
    // the loop on it.
    #[test]
    fn runtimes_activate_and_rollback_refuse_yes_without_restart_services() {
        for args in [&["activate", "x", "--yes"][..], &["rollback", "--yes"][..]] {
            let error = parse_runtimes(args)
                .expect_err("a bare --yes has nothing to approve and must not parse");
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::MissingRequiredArgument,
                "expected a missing-argument refusal for {args:?}"
            );
            assert!(
                error.to_string().contains("restart-services"),
                "the refusal must name the flag that is missing for {args:?}:\n{error}"
            );
        }
    }

    // Mutation: making `--restart-services` conflict with `--yes`, or dropping
    // the `yes` field from either variant, fails the pair assertions.
    #[test]
    fn runtimes_restart_flags_parse_together_and_alone() {
        for prefix in [&["activate", "x"][..], &["rollback"][..]] {
            let mut together = prefix.to_vec();
            together.extend(["--restart-services", "--yes"]);
            assert_eq!(
                runtimes_restart_flags(&together),
                (true, true),
                "{together:?}"
            );

            let mut alone = prefix.to_vec();
            alone.push("--restart-services");
            assert_eq!(runtimes_restart_flags(&alone), (true, false), "{alone:?}");

            assert_eq!(runtimes_restart_flags(prefix), (false, false), "{prefix:?}");
        }
    }

    #[test]
    fn runtimes_activate_parses_the_runtime_it_was_given() {
        let cli = parse_runtimes(&["activate", "release-pip-x", "--restart-services", "--yes"])
            .expect("activate parses");
        match cli.command {
            Some(Command::Runtimes {
                command: Some(RuntimesCommand::Activate { runtime, .. }),
            }) => assert_eq!(runtime, "release-pip-x"),
            other => panic!("expected runtimes activate, got {other:?}"),
        }
    }

    fn parse_serve(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["rocm", "serve", "qwen"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv)
    }

    #[test]
    fn serve_parses_verbose_and_no_smoke_test_flags() {
        let cli = parse_serve(&["--verbose", "--no-smoke-test"]).expect("flags parse");
        match cli.command {
            Some(Command::Serve {
                verbose,
                no_smoke_test,
                foreground,
                managed,
                ..
            }) => {
                assert!(verbose, "--verbose should set verbose");
                assert!(no_smoke_test, "--no-smoke-test should set no_smoke_test");
                assert!(!foreground);
                assert!(!managed);
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_verbose_conflicts_with_managed() {
        // `--verbose` streams logs in the foreground; a backgrounded managed
        // server has no foreground stream to attach to, so the two are mutually
        // exclusive (point users at `rocm logs` for a managed server instead).
        let error = parse_serve(&["--verbose", "--managed"]).expect_err("conflict rejected");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn serve_defaults_have_all_flags_off() {
        let cli = parse_serve(&[]).expect("bare serve parses");
        match cli.command {
            Some(Command::Serve {
                verbose,
                no_smoke_test,
                foreground,
                managed,
                ..
            }) => {
                assert!(!verbose && !no_smoke_test && !foreground && !managed);
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_parses_generation_flags() {
        let cli = parse_serve(&[
            "--temperature",
            "0.5",
            "--top-p",
            "0.25",
            "--max-tokens",
            "128",
        ])
        .expect("generation flags parse");
        match cli.command {
            Some(Command::Serve {
                temperature,
                top_p,
                max_tokens,
                ..
            }) => {
                assert_eq!(temperature, Some(0.5));
                assert_eq!(top_p, Some(0.25));
                assert_eq!(max_tokens, Some(128));
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_rejects_out_of_range_sampling() {
        // top-p is a probability; temperature must be non-negative. Both are
        // validated at the CLI boundary so a bad value never reaches the engine.
        assert_eq!(
            parse_serve(&["--top-p=1.5"])
                .expect_err("top-p above 1.0 is rejected")
                .kind(),
            clap::error::ErrorKind::ValueValidation
        );
        assert_eq!(
            parse_serve(&["--temperature=-0.1"])
                .expect_err("negative temperature is rejected")
                .kind(),
            clap::error::ErrorKind::ValueValidation
        );
        assert_eq!(
            parse_serve(&["--max-tokens=0"])
                .expect_err("zero max-tokens is rejected")
                .kind(),
            clap::error::ErrorKind::ValueValidation
        );
    }

    #[test]
    fn serve_negative_sampling_space_form_reaches_range_validator() {
        // The space form (`--temperature -1`) must reach the value parser and
        // report a value error, not be rejected by clap as an unexpected argument.
        // This is the classic negative-number-as-flag gotcha; `allow_negative_numbers`
        // makes both forms validate identically. `--max-tokens` shares the gotcha even
        // though it is a positive integer: the ambiguity is at the tokenizer, before the
        // value parser runs, so the space form must reach `parse_positive_u32` too.
        for args in [
            &["--temperature", "-1"][..],
            &["--temperature=-1"][..],
            &["--top-p", "-0.5"][..],
            &["--top-p=-0.5"][..],
            &["--max-tokens", "-1"][..],
            &["--max-tokens=-1"][..],
        ] {
            assert_eq!(
                parse_serve(args)
                    .expect_err("negative sampling value is rejected")
                    .kind(),
                clap::error::ErrorKind::ValueValidation,
                "expected range validation for {args:?}"
            );
        }
    }

    #[test]
    fn chat_negative_sampling_space_form_reaches_range_validator() {
        // Mirrors `serve_negative_sampling_space_form_reaches_range_validator`: the
        // same numeric value flags carry `allow_negative_numbers` on `chat`, so the
        // space form reaches the value parser instead of clap's unexpected-argument path.
        for args in [
            &["--temperature", "-1"][..],
            &["--temperature=-1"][..],
            &["--top-p", "-0.5"][..],
            &["--top-p=-0.5"][..],
            &["--max-tokens", "-1"][..],
            &["--max-tokens=-1"][..],
        ] {
            let mut argv = vec!["rocm", "chat", "--prompt", "hi"];
            argv.extend_from_slice(args);
            assert_eq!(
                Cli::try_parse_from(argv)
                    .expect_err("negative sampling value is rejected")
                    .kind(),
                clap::error::ErrorKind::ValueValidation,
                "expected range validation for {args:?}"
            );
        }
    }

    #[test]
    fn chat_parses_generation_flags() {
        let cli = Cli::try_parse_from([
            "rocm",
            "chat",
            "--prompt",
            "hello",
            "--temperature",
            "0.5",
            "--top-p",
            "0.25",
            "--max-tokens",
            "64",
        ])
        .expect("chat generation flags parse");
        match cli.command {
            Some(Command::Chat {
                temperature,
                top_p,
                max_tokens,
                ..
            }) => {
                assert_eq!(temperature, Some(0.5));
                assert_eq!(top_p, Some(0.25));
                assert_eq!(max_tokens, Some(64));
            }
            other => panic!("expected Chat, got {other:?}"),
        }
    }

    #[test]
    fn chat_rejects_zero_max_tokens() {
        let error = Cli::try_parse_from(["rocm", "chat", "--prompt", "hello", "--max-tokens", "0"])
            .expect_err("zero max-tokens is rejected");
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn update_rejects_zero_timeout_secs() {
        let error = Cli::try_parse_from(["rocm", "update", "--json", "--timeout-secs", "0"])
            .expect_err("zero timeout-secs is rejected");
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
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
            "vllm",
            "/v1/models",
            200,
            &empty,
            "tiny.gguf"
        ));

        let models = json!({ "data": [{ "id": "tiny.gguf" }] }).to_string();
        assert!(service_http_readiness_response_ready(
            "vllm",
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
            "vllm",
            "/health",
            200,
            "OK",
            "tiny.gguf"
        ));
        assert!(!service_http_readiness_response_ready(
            "vllm",
            "/healthz",
            200,
            "OK",
            "Qwen3-0.6B-GGUF"
        ));
    }

    #[test]
    fn lemonade_direct_serve_model_reads_ready_without_recipe_options() {
        // The HF direct-serve path runs a stock llama-server whose `/v1/models` entry
        // has no `recipe_options`. It must read as ready by name (that path is GPU-only),
        // while a registered-but-unloaded lemonade entry (empty `recipe_options`) must not.
        let direct = json!({
            "data": [{ "id": "LiquidAI/LFM2.5-230M-GGUF:Q4_0", "object": "model" }]
        })
        .to_string();
        assert!(service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &direct,
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        ));

        let registered = json!({
            "data": [{ "id": "LiquidAI/LFM2.5-230M-GGUF:Q4_0", "recipe_options": {} }]
        })
        .to_string();
        assert!(!service_http_readiness_response_ready(
            "lemonade",
            "/v1/models",
            200,
            &registered,
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
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

    #[test]
    fn lemonade_stop_unload_is_bounded_by_the_request_timeout() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::Instant;

        // Regression test for the stall this PR fixes: a peer that trickles the
        // response one byte at a time, never framing or closing, used to stall
        // `read_tcp_stream_to_string`'s read-to-EOF loop indefinitely. That hung
        // `unload_lemonade_service_model` past its 5s timeout during scenario
        // teardown, showing up as an unexplained multi-minute gap. The unload
        // call must now return an error at (not far past) its 5s budget.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 512];
            let _ = stream.read(&mut buffer);
            let body = b"{\"status\":\"success\",\"message\":\"ok\"}";
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            // One byte every 300ms never finishes framing the 35-byte body
            // inside the 5s unload timeout below, so the bound under test is
            // the deadline firing, not the response completing early.
            for byte in body {
                if stream.write_all(&[*byte]).is_err() {
                    return;
                }
                thread::sleep(Duration::from_millis(300));
            }
        });

        let (_root, paths) = test_paths("lemonade-stop-unload-dribble");
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
        let started = Instant::now();
        assert!(unload_lemonade_service_model(&record).is_err());
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(4),
            "bounded BY the 5s deadline, not failing early: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_secs(8), "{elapsed:?}");
        Ok(())
    }

    #[test]
    fn serve_readiness_wait_withholds_ready_while_the_model_only_lists() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // This is the wait behind `rocm serve`, and its verdict is what
        // `services list` later prints. A model listing on `/v1/models` while
        // inference still fails must come back as `Listing` — reporting it ready
        // is the false positive users and automation trip over.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<()> {
            // Serve until the wait below gives up and the test drops the socket.
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let (status_line, body) = if request.starts_with("POST /v1/chat/completions ") {
                    ("HTTP/1.1 503 Service Unavailable", r#"{"error":"loading"}"#)
                } else {
                    ("HTTP/1.1 200 OK", r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#)
                };
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            Ok(())
        });

        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_millis(600),
        );

        assert_eq!(
            readiness,
            EndpointReadiness::Listing,
            "a listed-but-unservable model is not ready"
        );
        assert_eq!(
            status_for_readiness(readiness),
            "running",
            "a loading service must not be recorded as `starting`, which rocmd \
             restarts once stale"
        );
        drop(server);
        Ok(())
    }

    #[test]
    fn serve_readiness_wait_reports_ready_once_inference_answers() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || -> Result<()> {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
            }
            Ok(())
        });

        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_secs(5),
        );

        assert_eq!(readiness, EndpointReadiness::Serving);
        assert_eq!(status_for_readiness(readiness), "ready");
        drop(server);
        Ok(())
    }

    #[test]
    fn serve_readiness_ready_verdict_does_not_wait_for_the_peer_to_close() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::Instant;

        // Regression test for the other half of this PR's fix: a response is
        // read to completion by its own framing, not by waiting for the peer
        // to close. Before this fix, `read_tcp_stream_to_string` blocked
        // until EOF, so a keep-alive engine that answers correctly but never
        // closes the socket looked identical to a hung one — the readiness
        // probe ran out its timeout and reported not-ready even though the
        // answer had already arrived.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                thread::spawn(move || {
                    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                    let mut buffer = [0_u8; 1024];
                    let Ok(read) = stream.read(&mut buffer) else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                    let body = if request.starts_with("POST /v1/chat/completions ") {
                        r#"{"choices":[{"message":{"content":"ok"}}]}"#
                    } else {
                        r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    // Hold the connection open well past the readiness wait's
                    // timeout below, and deliberately omit `Connection:
                    // close`. The client must not need EOF to recognize the
                    // response as complete.
                    thread::sleep(Duration::from_secs(10));
                });
            }
        });

        let started = Instant::now();
        let readiness = wait_for_service_http_ready(
            "vllm",
            "127.0.0.1",
            port,
            "Qwen3-0.6B-GGUF",
            None,
            Duration::from_secs(5),
        );

        assert_eq!(readiness, EndpointReadiness::Serving);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "a complete response must be recognized without waiting on the peer to close"
        );
        // `server`'s accept loop runs forever; dropping the JoinHandle detaches
        // it rather than joining, and the thread dies with the test process.
        drop(server);
        Ok(())
    }

    #[test]
    fn a_loading_service_keeps_its_status_instead_of_being_demoted() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // A model that is listed but cannot serve yet is coming up normally. It
        // must not be demoted to "starting": `rocmd` restarts a service that sits
        // in "starting" past its stale window, which would kill a slow-loading
        // model mid-load and start the wait over.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        // Keeps listing the model for as long as the test asks, so a second
        // readiness check reaches the probe throttle rather than an unanswered
        // socket. Counts the inference requests that actually got sent.
        let probes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let server_probes = std::sync::Arc::clone(&probes);
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                if request.starts_with("POST /v1/chat/completions ") {
                    server_probes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let body = r#"{"error":"loading model"}"#;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                } else {
                    let body = r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                }
            }
        });

        let (root, paths) = test_paths("liveness-loading-service");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-loading",
            "vllm",
            "Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert_eq!(
            record.status, "running",
            "a loading service keeps its status rather than being demoted"
        );
        assert!(
            record.inference_verified_at_unix_ms.is_none(),
            "nothing is latched until inference actually answers"
        );
        assert!(
            changed && record.inference_probe_attempted_at_unix_ms.is_some(),
            "the probe attempt is recorded and reported so the caller persists \
             the retry throttle"
        );

        // Second pass, inside the retry interval: the throttle holds, so the
        // model is not asked to generate again just because someone re-listed.
        let changed_again = refresh_managed_service_runtime_liveness(&paths, &mut record);
        assert!(!changed_again, "a throttled check changes nothing");
        assert_eq!(record.status, "running");
        assert_eq!(
            probes.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a warming service must not be re-probed by every poll"
        );

        drop(server);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn load_managed_services_promotes_running_to_ready_once_probe_passes() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        // Three requests hit this mock. `load_managed_services` lists the model
        // and then confirms inference, which promotes the record and latches the
        // verification; the `load_managed_service` re-read below (which proves the
        // promotion was persisted, not just returned in-memory) reads that latch
        // and so only re-lists.
        let server = thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for _ in 0..3 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    if String::from_utf8_lossy(&request_bytes).contains("\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                requests.push(request);
            }
            Ok(requests)
        });

        let (root, paths) = test_paths("load-managed-services-promote-ready");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-qwen-promote",
            "vllm",
            "Qwen3-0.6B-GGUF",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        // A supervisor that has already observed the engine come up reports
        // "running"; only the HTTP model-ready probe should promote it further.
        record.status = "running".to_owned();
        record.write()?;

        let records = load_managed_services(&paths)?;
        let promoted = records
            .iter()
            .find(|found| found.service_id == "svc-qwen-promote")
            .expect("service should be present");
        assert_eq!(promoted.status, "ready");

        // Re-read the manifest file directly (bypassing any code path that
        // could itself re-run the promotion) to prove the transition was
        // actually written to disk, not just returned in the in-memory
        // `Vec<ManagedServiceRecord>` above. `load_managed_service` below
        // also calls `refresh_managed_service_runtime_liveness` on every
        // read, so asserting only on its return value would pass even if
        // `load_managed_services` never persisted anything.
        let on_disk_bytes = fs::read(&record.manifest_path)?;
        let on_disk = serde_json::from_slice::<ManagedServiceRecord>(&on_disk_bytes)?;
        assert_eq!(on_disk.status, "ready");

        // The promotion must have been persisted to disk, not just returned
        // in-memory, since chat's `pick_managed_chat_endpoint` re-reads it.
        let reloaded = load_managed_service(&paths, "svc-qwen-promote")?;
        assert_eq!(reloaded.status, "ready");
        assert!(
            reloaded.inference_verified_at_unix_ms.is_some(),
            "the inference verification is persisted with the promotion"
        );

        let requests = server.join().expect("server thread should not panic")?;
        let lines: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            lines,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "promotion confirms inference once; the re-read reads the latch"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    pub(crate) fn test_app_paths() -> AppPaths {
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
    fn freeform_invocation_supports_leading_yes_for_natural_language_only() {
        let invocation = parse_freeform_invocation(&[
            "--yes".to_owned(),
            "please".to_owned(),
            "serve".to_owned(),
            "qwen3.5".to_owned(),
            "with".to_owned(),
            "vllm".to_owned(),
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
                "vllm".to_owned(),
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
            "vllm".to_owned(),
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
        let action =
            freeform_planner::freeform_plan_next_action("serve", &RocmCliConfig::default())
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
        let action = freeform_planner::freeform_plan_next_action(
            "serve qwen3.5 with vllm",
            &RocmCliConfig::default(),
        )
        .expect("serve request should have next action");

        validate_freeform_execution_action(&action)?;
        assert_eq!(
            freeform_planner::format_structured_tool_call("rocm", &action.args),
            "rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        );
        Ok(())
    }

    #[test]
    fn freeform_execution_grants_the_narrow_consent_the_outer_yes_already_gave() {
        // `rocm --yes <request>` dispatches the generated argv **in process**, so
        // nothing carries the outer `--yes` into `install()` unless this does.
        // Without it the surface prints `approval: granted by --yes` and then
        // either refuses non-interactively, asking for a flag it offers no way to
        // pass, or prompts on a terminal it just said it would not need to ask.
        let request =
            "install the latest TheRock nightly for this GPU into D:\\ROCm\\therock_venvs";
        let config = RocmCliConfig::default();

        let planned = freeform_planner::freeform_plan_next_action(request, &config)
            .expect("install request should have next action");
        assert!(
            !planned
                .args
                .iter()
                .any(|arg| arg.starts_with("--yes") || arg.starts_with("--approve-")),
            "the plan itself must stay unapproved so `rocm <request>` shows a \
             reviewable command: {:?}",
            planned.args
        );

        // Through the real pre-dispatch path, not the injector in isolation:
        // `execute_freeform_next_action` is this plus the header render and
        // `dispatch`, so dropping the injection from the pipeline fails here.
        let execution = prepare_freeform_execution(request, &test_app_paths(), &config)
            .expect("install request should prepare for execution");
        let action = &execution.action;

        assert_eq!(
            freeform_planner::format_structured_tool_call("rocm", &action.args),
            "rocm install sdk --channel nightly --format wheel --prefix \
             D:\\ROCm\\therock_venvs --approve-replacing-active-default"
        );
        // The narrow flag, never `--yes`: this surface has no terminal promise to
        // make about a sudo password prompt for system packages.
        assert!(!action.args.iter().any(|arg| arg == "--yes"));
        // The header renders after injection, so the printed tool call is the
        // argv that actually runs.
        let rendered = render_freeform_execution_header(&execution);
        assert!(rendered.contains("--approve-replacing-active-default"));
        // And the operator is told why this `tool_call:` carries a consent flag
        // the `request plan` section above it did not show. The plan assertion at
        // the top of this test is what makes the two lines differ here, so the
        // disclosure and the difference are pinned by the same test.
        assert!(
            execution.consent_added,
            "the plan arrived unapproved, so the injector must report adding the flag"
        );
        assert!(
            rendered.contains("was added here from your --yes"),
            "the execution section must explain the differing tool_call: {rendered}"
        );
        // Re-parsing must reach `install()` with the consent actually set.
        let mut argv = vec!["rocm".to_owned()];
        argv.extend(execution.action.args);
        let cli = Cli::try_parse_from(argv).expect("generated argv should parse");
        match cli.command {
            Some(Command::Install {
                target:
                    InstallTarget::Sdk {
                        yes,
                        approve_replacing_active_default,
                        ..
                    },
            }) => {
                assert!(approve_replacing_active_default);
                assert!(!yes);
            }
            other => panic!("expected `install sdk`, got {other:?}"),
        }
    }

    #[test]
    fn freeform_execution_consent_is_scoped_to_mutating_sdk_installs() {
        // Dry runs return before the consent gate, and the sibling dry-run-aware
        // arms in chat, rocmd and dash-tui all withhold the flag there.
        let mut dry_run = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--dry-run".to_owned(),
        ];
        assert!(!apply_freeform_execution_consent(&mut dry_run));
        assert_eq!(
            dry_run,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--dry-run".to_owned()
            ]
        );

        // Nothing else the planner can emit takes this flag; injecting it would
        // not even parse.
        for mut args in [
            vec!["install".to_owned(), "driver".to_owned()],
            vec!["serve".to_owned(), "qwen".to_owned()],
            vec!["comfyui".to_owned(), "install".to_owned()],
        ] {
            let before = args.clone();
            assert!(!apply_freeform_execution_consent(&mut args));
            assert_eq!(args, before);
        }

        // Idempotent: a plan that already carries the flag is not given it twice,
        // and reports that it added nothing — the execution section must not
        // claim to have added a flag the argv arrived with.
        let mut already = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--approve-replacing-active-default".to_owned(),
        ];
        assert!(!apply_freeform_execution_consent(&mut already));
        assert_eq!(
            already
                .iter()
                .filter(|arg| *arg == "--approve-replacing-active-default")
                .count(),
            1
        );
    }

    #[test]
    fn freeform_execution_header_surfaces_explicit_approval_and_tool_call() {
        let action = freeform_planner::freeform_plan_next_action(
            "serve qwen3.5 with vllm",
            &RocmCliConfig::default(),
        )
        .expect("serve request should have next action");
        let rendered = render_freeform_execution_header(&FreeformExecution {
            action,
            consent_added: false,
        });

        assert!(rendered.contains("execution"));
        assert!(rendered.contains("approval: granted by --yes"));
        assert!(rendered.contains(
            "tool_call: rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed"
        ));
        // Nothing was injected on this path, so the two `tool_call:` lines agree
        // and the disclosure would be noise that contradicts the plan above.
        assert!(
            !rendered.contains("was added here"),
            "the note must be scoped to an argv this surface actually changed: {rendered}"
        );
    }

    #[test]
    fn close_subcommand_typo_yields_clap_suggestion() {
        let err = command_invocation_error(&["instal".to_owned()])
            .expect("a close typo should surface a clap subcommand suggestion");
        let message = err.to_string();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(message.contains("install"));
    }

    #[test]
    fn mistyped_command_with_flags_yields_clap_error() {
        // `doctorgdfg --help` is a botched command invocation, not prose, so it
        // should surface clap's usage error rather than a planner request plan.
        let err = command_invocation_error(&["doctorgdfg".to_owned(), "--help".to_owned()])
            .expect("a command-like token followed by a flag should yield a clap error");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(err.to_string().contains("doctorgdfg"));
    }

    #[test]
    fn mistyped_command_with_trailing_argument_yields_suggestion() {
        // A near-miss subcommand followed by a normal (non-flag) argument should
        // still surface clap's suggestion instead of falling to the planner.
        let err = command_invocation_error(&["automatios".to_owned(), "list".to_owned()])
            .expect("a near-miss subcommand with a trailing arg should yield a clap suggestion");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(err.to_string().contains("automations"));
    }

    #[test]
    fn natural_language_request_has_no_subcommand_suggestion() {
        // Multi-word prose requests stay with the planner.
        assert!(command_invocation_error(&["please".to_owned(), "install".to_owned()]).is_none());
        // A single token with no near match is left for the planner too.
        assert!(command_invocation_error(&["zzzzzzzz".to_owned()]).is_none());
        // Prose that happens to be a single quoted argument is not command-like.
        assert!(command_invocation_error(&["please install rocm".to_owned()]).is_none());
        // A single quoted prose request that clap can fuzzily match to a
        // subcommand (`is rocm installed?` -> `install`) must still reach the
        // planner rather than exiting with clap's suggestion.
        assert!(command_invocation_error(&["is rocm installed?".to_owned()]).is_none());
        assert!(command_invocation_error(&["how do i setup comfyui".to_owned()]).is_none());
    }

    /// Just below a unit boundary the value rounds up to a full 1024 of the
    /// SMALLER unit, which has to be reported as 1.0 of the larger one — the
    /// same defect `rocm_core::format_bytes` had. Each pair is the last input
    /// that still belongs to the smaller unit and the first that `{:.1}` rounds
    /// up to 1024.0 of it; the second used to print "1024.0 KiB" / "1024.0 MiB".
    #[test]
    fn format_bytes_steps_up_instead_of_printing_1024_of_the_smaller_unit() {
        assert_eq!(format_bytes(1023), "1023 bytes");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1_048_524), "1023.9 KiB");
        assert_eq!(format_bytes(1_048_525), "1.0 MiB");
        assert_eq!(format_bytes(1_048_575), "1.0 MiB");
        assert_eq!(format_bytes(1_048_576), "1.0 MiB");
        assert_eq!(format_bytes(1_073_689_395), "1023.9 MiB");
        assert_eq!(format_bytes(1_073_689_396), "1.0 GiB");
        assert_eq!(format_bytes(1_073_741_823), "1.0 GiB");
        // GiB is the top unit: nothing to promote to, so 1024 GiB stays in it.
        assert_eq!(format_bytes(1_099_511_627_776), "1024.0 GiB");
    }

    /// The units `format_bytes` prints, smallest first.
    const BYTE_UNITS: [&str; 4] = ["bytes", "KiB", "MiB", "GiB"];

    /// Byte counts that actually visit the unit boundaries. A uniform `u64`
    /// almost always lands far above the top unit, so on its own it never
    /// samples the band where `{:.1}` rounding reaches 1024.0. The other arms
    /// draw uniformly within one unit's range, and from a window just below
    /// each rounded boundary (KiB→MiB, MiB→GiB) that scales with the boundary,
    /// as the band does — see `rocm_core::disk_space`'s generator for the full
    /// reasoning.
    fn byte_count_strategy() -> impl proptest::strategy::Strategy<Value = u64> {
        use proptest::prelude::*;
        prop_oneof![
            any::<u64>(),
            (0u32..=3).prop_flat_map(|exponent| {
                let low = if exponent == 0 {
                    0
                } else {
                    1024u64.pow(exponent)
                };
                low..1024u64.pow(exponent + 1)
            }),
            (2u32..=3).prop_flat_map(|exponent| {
                let boundary = 1024u64.pow(exponent);
                (boundary - boundary / 16384)..=(boundary + 1)
            }),
        ]
    }

    proptest::proptest! {
        /// A size is rendered in the unit it belongs to, which has two edges.
        ///
        /// Upper: below the top unit, the printed mantissa is under 1024.0 —
        /// otherwise the size is shown in a unit it has outgrown.
        ///
        /// Lower: above `bytes`, the printed mantissa is at least 1.0, and the
        /// next smaller unit would have printed 1024.0 or more — otherwise the
        /// size was promoted before it reached a whole unit.
        ///
        /// Both edges compare the mantissa as printed, in tenths: that is the
        /// quantity a reader sees, so it is the one the scaling has to decide on.
        #[test]
        fn format_bytes_renders_a_size_in_its_own_unit(
            bytes in byte_count_strategy(),
        ) {
            let rendered = format_bytes(bytes);
            let (value, unit) = rendered
                .split_once(' ')
                .expect("rendered size is `<number> <unit>`");
            let value: f64 = value.parse().expect("numeric part parses");
            let tenths = (value * 10.0).round();
            let exponent = BYTE_UNITS
                .iter()
                .position(|name| *name == unit)
                .expect("rendered unit is one of the known units");
            if exponent + 1 < BYTE_UNITS.len() {
                proptest::prop_assert!(
                    tenths < 10_240.0,
                    "{bytes} rendered as {rendered}, which should have been \
                     promoted to the next unit",
                );
            }
            if exponent > 0 {
                proptest::prop_assert!(
                    tenths >= 10.0,
                    "{bytes} rendered as {rendered}, which was promoted before \
                     it reached a whole unit",
                );
                // Dividing by a power of two is exact, so this is the value the
                // smaller unit would have printed, not an approximation of it.
                let smaller = (1..exponent).fold(bytes as f64, |value, _| value / 1024.0);
                proptest::prop_assert!(
                    (smaller * 10.0).round() >= 10_240.0,
                    "{bytes} rendered as {rendered}, but still fits the smaller \
                     unit as {smaller:.1} {}",
                    BYTE_UNITS[exponent - 1],
                );
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
    fn freeform_execution_validation_rejects_provider_assisted_plans() -> Result<()> {
        let content = r#"{
            "intent": "serve",
            "tool_call": {
                "tool": "rocm",
                "args": ["serve", "sshleifer/tiny-gpt2", "--engine", "vllm", "--managed"]
            }
        }"#;
        let plan = freeform_planner::provider_planner_response_to_plan(
            "start a local model",
            "local",
            content,
        )?;
        let action =
            freeform_planner::plan_next_action(plan).expect("provider plan should have an action");

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
            "vllm",
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
        assert!(rendered.contains("Engine: vllm"));
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
        // The ready service is listed and then asked to complete a request:
        // readiness is not granted on the model listing alone.
        let server = thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                let mut request = [0_u8; 512];
                let read = stream.read(&mut request)?;
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen/Qwen3.5"}]}"#
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
            }
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
                "vllm",
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
        // The header counts the whole registry, so the hidden `failed` record
        // is admitted to rather than silently dropped from the default view.
        assert!(
            rendered.contains("Status: 1 ready, 1 starting, 1 not running"),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("Past attempts: 1 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-failed"),
            "{rendered}"
        );
        assert!(rendered.contains("- svc-ready"));
        assert!(rendered.contains("  stop: rocm services stop svc-ready --yes"));
        assert!(!rendered.contains("- svc-failed"));
        assert!(all.contains("- svc-failed"));
        assert!(all.contains("  restart: rocm services restart svc-failed --yes"));
        assert!(!rendered.contains("running servers"));
        Ok(())
    }

    /// The default view lists only live servers. The counts used to be taken
    /// from that already-filtered list, so a host whose local servers had all
    /// failed was told "none ready" and shown an empty list - the records were
    /// on disk, unmentioned, and nothing on screen said how to reach them.
    /// Counts now come from the unfiltered registry; the rows stay filtered.
    #[test]
    fn render_services_text_surfaces_records_the_default_view_hides() -> Result<()> {
        let (root, paths) = test_paths("services-past-attempts-only");
        paths.ensure()?;
        // Not live, so `refresh_managed_service_runtime_liveness` returns before
        // any endpoint probe or pid check: no network, no timing dependence.
        for (service_id, created_at) in [("svc-older", 1_000_u128), ("svc-newest", 2_000_u128)] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                11500,
                "managed",
                999_999_999,
                None,
                None,
                None,
            );
            record.status = "failed".to_owned();
            record.created_at_unix_ms = created_at;
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        let all = render_services_text(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Status: 2 not running"), "{rendered}");
        assert!(!rendered.contains("none ready"), "{rendered}");
        assert!(
            rendered.contains("No local servers are running."),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("Past attempts: 2 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  See them: rocm services list --all"),
            "{rendered}"
        );
        // A real id, newest first, so the line can be pasted as-is.
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-newest"),
            "{rendered}"
        );
        // The other two lines only let the user look. Each record keeps an
        // unrotated engine log, so `prune` is the only line here that gets any
        // of that space back - without it the block is a dead end on exactly
        // the host that needs it most.
        assert!(
            rendered.contains("  Reclaim the space: rocm services prune"),
            "{rendered}"
        );
        // The `--all` header is fixed by the same change; it lists the rows, so
        // it does not repeat the pointer.
        assert!(all.contains("Status: 2 not running"), "{all}");
        assert!(all.contains("- svc-newest"), "{all}");
        assert!(!all.contains("Past attempts:"), "{all}");
        Ok(())
    }

    /// The host this change is NOT meant to touch: nothing has ever been served,
    /// so there is no record to point at and the output must read exactly as it
    /// did before. Every other test here plants at least one record, so without
    /// this one the zero case has no coverage at all - and it is the case the
    /// hint's separator regressed, by printing a blank line the old output never
    /// had between "No local servers are running." and the next line.
    #[test]
    fn render_services_text_leaves_a_host_that_never_served_untouched() -> Result<()> {
        let (root, paths) = test_paths("services-never-served");
        paths.ensure()?;

        let rendered = render_services_text(&paths, false)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            rendered,
            "Local Servers\n\
             \n\
             Status: none ready\n\
             \n\
             No local servers are running.\n\
             Start one with `rocm serve <model> --managed`, or run `rocm` and choose Serve.\n",
            "{rendered}"
        );
        Ok(())
    }

    /// A host can have one server running and several that failed, so the
    /// populated branch of the default view needs the same pointer as the empty
    /// one - the failed records are hidden there too.
    #[test]
    fn render_services_text_points_at_past_attempts_beside_a_live_server() -> Result<()> {
        let (root, paths) = test_paths("services-past-attempts-mixed");
        paths.ensure()?;
        let current_pid = std::process::id();
        // `starting` skips the endpoint probe and this process is a
        // guaranteed-live pid, so the record stays live without any network.
        let mut live = ManagedServiceRecord::new(
            &paths,
            "svc-live",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11501,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        live.status = "starting".to_owned();
        live.engine_pid = Some(current_pid);
        live.created_at_unix_ms = 3_000;
        live.write()?;
        for (service_id, created_at) in [
            ("svc-failed-older", 1_000_u128),
            ("svc-failed-newest", 2_000_u128),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                11502,
                "managed",
                999_999_999,
                None,
                None,
                None,
            );
            record.status = "failed".to_owned();
            record.created_at_unix_ms = created_at;
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        let _ = fs::remove_dir_all(root);

        assert!(
            rendered.contains("Status: 1 starting, 2 not running"),
            "{rendered}"
        );
        // Rows stay filtered: only the live server is listed.
        assert!(rendered.contains("- svc-live"), "{rendered}");
        assert!(!rendered.contains("- svc-failed-newest"), "{rendered}");
        assert!(
            rendered
                .contains("Past attempts: 2 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  See them: rocm services list --all"),
            "{rendered}"
        );
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-failed-newest"),
            "{rendered}"
        );
        Ok(())
    }

    #[test]
    fn services_json_round_trips_and_applies_the_same_liveness_filter_as_the_table() -> Result<()> {
        // This JSON is a contract, not a convenience: `rocm remote` parses it
        // back over its control channel to learn which service a remote serve
        // just started. Two things have to hold — every record survives the
        // round trip, and `--json` agrees with the table about what is live. If
        // they disagreed, the remote orchestration would act on a different set
        // of services than the operator sees.
        let (root, paths) = test_paths("services-json");
        paths.ensure()?;
        let current_pid = std::process::id();
        for (service_id, status, port) in [
            ("svc-live", "starting", 11440_u16),
            ("svc-past", "failed", 11441_u16),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
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

        let live = render_services_json(&paths, false)?;
        let every = render_services_json(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        let live: Vec<ManagedServiceRecord> = serde_json::from_str(&live)?;
        let every: Vec<ManagedServiceRecord> = serde_json::from_str(&every)?;

        assert_eq!(
            live.iter()
                .map(|r| r.service_id.as_str())
                .collect::<Vec<_>>(),
            vec!["svc-live"],
            "the default listing must hide past attempts, exactly as the table does"
        );
        let mut every_ids = every
            .iter()
            .map(|r| r.service_id.as_str())
            .collect::<Vec<_>>();
        every_ids.sort_unstable();
        assert_eq!(every_ids, vec!["svc-live", "svc-past"]);

        // The fields remote orchestration actually reads must survive intact.
        let record = &live[0];
        assert_eq!(record.port, 11440);
        assert_eq!(record.status, "starting");
        // Note for remote orchestration: the recorded endpoint is already the
        // OpenAI-compatible base, `/v1` suffix included — not a bare origin.
        assert_eq!(record.endpoint_url, "http://127.0.0.1:11440/v1");
        assert_eq!(record.canonical_model_id, "Qwen/Qwen3.5");
        Ok(())
    }

    #[test]
    fn service_records_tolerate_unknown_fields_but_not_missing_required_ones() -> Result<()> {
        // A remote may run a different CLI version than the machine driving it.
        // Newer fields it emits must not break an older parser, or a version skew
        // turns every remote command into a parse error; a genuinely absent
        // required field must still fail, and name itself when it does.
        // Built from a real record rather than hand-written JSON, so the fixture
        // cannot drift out of step with the struct and quietly stop testing the
        // thing it claims to.
        let (root, paths) = test_paths("services-json-contract");
        paths.ensure()?;
        let record = ManagedServiceRecord::new(
            &paths,
            "svc-a",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11440,
            "managed",
            4242,
            None,
            None,
            None,
        );
        let mut value = serde_json::to_value(&record)?;
        let _ = fs::remove_dir_all(root);
        let fields = value
            .as_object_mut()
            .expect("a service record serializes as a JSON object");

        fields.insert(
            "a_field_from_a_newer_release".to_owned(),
            serde_json::Value::Bool(true),
        );
        let parsed: ManagedServiceRecord = serde_json::from_value(value.clone())
            .context("a newer remote's extra fields must not break an older parser")?;
        assert_eq!(parsed.service_id, "svc-a");
        assert_eq!(parsed.port, 11440);

        value
            .as_object_mut()
            .expect("still an object")
            .remove("port")
            .expect("port was present before removal");
        let error = serde_json::from_value::<ManagedServiceRecord>(value)
            .expect_err("a missing required field must be rejected, not defaulted")
            .to_string();
        assert!(
            error.contains("port"),
            "the error should name the missing field, got: {error}"
        );
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
        // The transition this test is named for reaches the header too: once the
        // record is demoted it is a past attempt, so the header says so instead
        // of the old "none ready" - which described the live set correctly and
        // the disk misleadingly.
        assert!(rendered.contains("Status: 1 not running"), "{rendered}");
        assert!(all.contains("- svc-stale-ready"));
        assert!(all.contains("  status: stopped"));
        assert_eq!(reloaded.status, "stopped");
        Ok(())
    }

    /// Plant a managed-service record plus every artifact it owns: the log and
    /// endpoint key beside the manifest in `services_dir`, and the engine state
    /// file under `<data>/engines/<engine>/state/`.
    fn plant_service_record(
        paths: &AppPaths,
        service_id: &str,
        status: &str,
        supervisor_pid: u32,
    ) -> Result<ManagedServiceRecord> {
        let record =
            plant_service_record_without_its_key(paths, service_id, status, supervisor_pid)?;
        endpoint_keys::store_endpoint_api_key(paths, service_id, "test-key")?;
        Ok(record)
    }

    /// Everything [`plant_service_record`] writes *except* the 0600 endpoint
    /// key: the manifest, the log, and the engine state file.
    ///
    /// Split out for the launch-lock test below, which writes the key itself —
    /// as `serve` does, before the record — and then has to publish the record
    /// without touching the key again. Planting the key a second time there
    /// would re-create whatever `prune` had already deleted, turning that test's
    /// key assertions into assertions about this helper: they would hold whether
    /// or not the sweep had swept.
    fn plant_service_record_without_its_key(
        paths: &AppPaths,
        service_id: &str,
        status: &str,
        supervisor_pid: u32,
    ) -> Result<ManagedServiceRecord> {
        plant_service_record_on_runtime(
            paths,
            service_id,
            "vllm",
            status,
            supervisor_pid,
            None,
            None,
        )
    }

    /// [`plant_service_record_without_its_key`] with the three fields the runtime-activation
    /// tests need to vary: the engine (self-managed engines are never stale),
    /// the recorded runtime key, and the environment pin.
    pub(crate) fn plant_service_record_on_runtime(
        paths: &AppPaths,
        service_id: &str,
        engine: &str,
        status: &str,
        supervisor_pid: u32,
        runtime_id: Option<&str>,
        env_id: Option<&str>,
    ) -> Result<ManagedServiceRecord> {
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            engine,
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            9,
            "managed",
            supervisor_pid,
            runtime_id.map(ToOwned::to_owned),
            env_id.map(ToOwned::to_owned),
            Some("gpu_required".to_owned()),
        );
        status.clone_into(&mut record.status);
        record.write()?;
        fs::write(&record.log_path, "engine output\n")?;
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .context("engine state path has a parent")?,
        )?;
        // Mirror the planted status: `refresh_from_engine_state` adopts whatever
        // this file says, so a mismatched value would silently demote the record
        // before the code under test ever sees it.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({ "status": status }))?,
        )?;
        Ok(record)
    }

    #[test]
    fn services_remove_deletes_every_artifact_and_leaves_the_shared_lock() -> Result<()> {
        let (root, paths) = test_paths("services-remove-artifacts");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-remove-me", "failed", 999_999_999)?;
        // `launch.lock` lives in `services_dir` but belongs to every managed
        // launch, not to one service. Removing a record must not touch it.
        let lock = paths.managed_launch_lock_path();
        fs::write(&lock, "")?;

        let rendered = remove_managed_service_record(&paths, "svc-remove-me", true);
        let rendered = match rendered {
            Ok(text) => text,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let engine_state_exists = record.engine_state_path.exists();
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, "svc-remove-me").exists();
        let lock_exists = lock.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(!manifest_exists, "the details file must be deleted");
        assert!(!log_exists, "the log must be deleted");
        assert!(
            !engine_state_exists,
            "the engine state file must be deleted — it lives outside the services folder, \
             so deleting only the two files beside the manifest is what orphans it"
        );
        assert!(!key_exists, "the endpoint key file must be deleted");
        assert!(lock_exists, "the shared launch lock must be left alone");
        assert!(rendered.contains("Local server record removed"));
        assert!(rendered.contains("  files removed: 4"));
        // The log path is the last thing worth knowing before it disappears.
        assert!(
            rendered.contains(&format!("  log: {}", record.log_path.display())),
            "the log path must be printed before the delete:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn services_remove_refuses_a_running_record_and_names_stop() -> Result<()> {
        let (root, paths) = test_paths("services-remove-live");
        paths.ensure()?;
        // The current process is a guaranteed-live PID, so the liveness refresh
        // keeps this record in a running state.
        plant_service_record(&paths, "svc-live", "ready", std::process::id())?;

        let error = remove_managed_service_record(&paths, "svc-live", true)
            .expect_err("removing a running local server must fail");
        let manifest_exists = paths.service_manifest_path("svc-live").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(
            message.contains("cannot be removed while it is running"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("rocm services stop svc-live --yes"),
            "the error must name the stop command: {message}"
        );
        assert!(manifest_exists, "a refused removal must delete nothing");
        Ok(())
    }

    /// A manifest still saying `ready` for a process that is gone is the common
    /// case this command exists for. `load_managed_services` demotes it to
    /// `stopped` while reading, so the live guard has to run on the refreshed
    /// record — checking the status as it sits on disk would refuse forever.
    #[test]
    fn services_remove_accepts_a_stale_ready_record_that_refreshed_to_stopped() -> Result<()> {
        let (root, paths) = test_paths("services-remove-stale-ready");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-stale", "ready", 999_999_999)?;
        assert_eq!(
            serde_json::from_slice::<ManagedServiceRecord>(&fs::read(&record.manifest_path)?)?
                .status,
            "ready",
            "premise: the manifest on disk still claims `ready`"
        );

        let result = remove_managed_service_record(&paths, "svc-stale", true);
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        result?;
        assert!(!manifest_exists, "the stale record must be removable");
        Ok(())
    }

    #[test]
    fn services_remove_requires_yes() -> Result<()> {
        let (root, paths) = test_paths("services-remove-yes");
        paths.ensure()?;
        plant_service_record(&paths, "svc-needs-yes", "failed", 999_999_999)?;

        let error = remove_managed_service_record(&paths, "svc-needs-yes", false)
            .expect_err("removal without --yes must fail");
        let manifest_exists = paths.service_manifest_path("svc-needs-yes").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(message.contains("requires --yes"), "unexpected: {message}");
        assert!(
            message.contains("rocm services remove svc-needs-yes --yes"),
            "unexpected: {message}"
        );
        assert!(manifest_exists, "a refused removal must delete nothing");
        Ok(())
    }

    #[test]
    fn service_record_artifacts_reject_an_engine_that_escapes_the_state_dir() -> Result<()> {
        let (root, paths) = test_paths("services-remove-traversal");
        paths.ensure()?;

        let by_engine = service_record_artifacts(&paths, "svc-ok", "../../../etc");
        let by_id = service_record_artifacts(&paths, "../../etc/passwd", "vllm");
        let _ = fs::remove_dir_all(&root);

        assert!(
            by_engine.is_err(),
            "an engine name with `..` must not reach a path this code deletes"
        );
        assert!(by_id.is_err(), "a traversing service id must be rejected");
        Ok(())
    }

    #[test]
    fn services_prune_skips_running_records_and_reports_the_count() -> Result<()> {
        let (root, paths) = test_paths("services-prune-live");
        paths.ensure()?;
        plant_service_record(&paths, "svc-dead", "failed", 999_999_999)?;
        plant_service_record(&paths, "svc-running", "ready", std::process::id())?;

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let dead_exists = paths.service_manifest_path("svc-dead").exists();
        let running_exists = paths.service_manifest_path("svc-running").exists();
        let _ = fs::remove_dir_all(&root);

        assert!(!dead_exists, "the stopped record must be pruned");
        assert!(running_exists, "a running record must be left alone");
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(outcome.skipped_live, 1);
        assert!(
            outcome.text.contains("  still running, left alone: 1"),
            "prune must report how many records it skipped:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("rocm services stop svc-running --yes"),
            "the skip reason must name the stop command:\n{}",
            outcome.text
        );
        // ROCMAI-543: the skip line is built by `build_service_prune_plan` as
        // "<id> is <status> - stop it first with `<cmd>`" and printed verbatim.
        // Asserting the whole rendered plan rather than that one sentence also
        // covers the counts, the headings, and any line added to the plan later.
        assert!(
            outcome.text.is_ascii(),
            "`rocm services prune` output must be ASCII-only; non-ASCII punctuation renders \
             as mojibake on a Windows console using the OEM code page (ROCMAI-543):\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The engine state file lives outside `services_dir`, so hand-deleting the
    /// manifest and log — the workaround this command replaces — leaves it
    /// behind forever. Prune has to sweep it.
    #[test]
    fn services_prune_sweeps_engine_state_left_behind_by_a_deleted_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-orphans");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-orphaned", "failed", 999_999_999)?;
        // Exactly what the manual workaround leaves: the two files in the
        // services folder are gone, the engine state and endpoint key are not.
        fs::remove_file(&record.manifest_path)?;
        let orphan_state = record.engine_state_path;
        let orphan_log = record.log_path;
        let orphan_key = endpoint_keys::endpoint_key_file_path(&paths, "svc-orphaned");

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let state_exists = orphan_state.exists();
        let log_exists = orphan_log.exists();
        let key_exists = orphan_key.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !state_exists,
            "the orphaned engine state file must be swept:\n{}",
            outcome.text
        );
        assert!(!log_exists, "the orphaned log must be swept");
        assert!(!key_exists, "the orphaned endpoint key must be swept");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 3);
        Ok(())
    }

    /// The window the launch lock exists to close here: `serve` writes
    /// the 0600 endpoint key *before* the first manifest, so between those two
    /// writes a live server's key has no `<id>.json` beside it — the exact shape
    /// the sweep calls a leftover, and `--any-age` leaves no age gate to hide it
    /// behind. `serve` holds the managed-launch lock across the whole of that
    /// interval, so the fix is for `prune` to acquire it too.
    ///
    /// Shape of the test, since `FileLock::acquire` blocks with no `try_`
    /// variant and calling `prune` on the thread that holds the lock would
    /// simply deadlock: the main thread stands in for the launch and holds the
    /// lock with only the key written, a second thread runs the real
    /// `--any-age --yes` argv, and the main thread then does what the launch
    /// does next — writes the record — *before* releasing. So the sweep sees the
    /// directory only in its published state.
    ///
    /// Two independent assertions each fail on their own if `prune` stops
    /// acquiring the lock: the negative wait (prune returns while the lock is
    /// held, so it never waited for it) and the endpoint key (prune scans before
    /// the record lands and deletes it). The negative wait is reported first
    /// only because it names the cause; it is not load-bearing on its own, so
    /// deleting it as "timing-sensitive" still leaves a pure filesystem fact
    /// behind it. For that second assertion to mean anything, the record has to
    /// be published *without* rewriting the key — hence
    /// [`plant_service_record_without_its_key`] below rather than the full
    /// planter. The negative wait in turn cannot pass vacuously: the worker
    /// physically cannot report completion without first holding the lock the
    /// main thread has.
    #[test]
    fn services_prune_waits_for_the_managed_launch_lock_before_sweeping() -> Result<()> {
        use std::sync::mpsc;

        /// Long enough that a `prune` over an all-but-empty data dir would have
        /// finished many times over, short enough to keep the suite quick. Only
        /// ever compared against "did not finish", so a slow machine makes this
        /// more conclusive, never flakier.
        const WAIT_PROOF: Duration = Duration::from_millis(500);
        const ID: &str = "svc-launching";

        let (root, paths) = test_paths("services-prune-launch-lock");
        paths.ensure()?;
        // Write 1 of 2, byte for byte as `serve` does it at the top of the
        // managed path. No manifest yet: `spawn_managed_engine_child` has not
        // reached `record.write()`.
        endpoint_keys::store_endpoint_api_key(&paths, ID, "live-secret")?;
        assert!(
            !paths.service_manifest_path(ID).exists(),
            "premise: the manifest must not be written yet"
        );
        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (waited, published, finished) = std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).expect("signal about to prune");
                let outcome =
                    parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
                        .and_then(|(hours, dry_run, yes)| {
                            prune_managed_service_records(&paths, hours, dry_run, yes)
                        });
                let _ = done_tx.send(
                    outcome
                        .map(|outcome| (outcome.removed_files, outcome.skipped_live, outcome.text))
                        .map_err(|error| error.to_string()),
                );
            });
            // The worker is running before the negative check below, so that
            // check is about the lock and not about scheduling latency.
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("prune thread started");
            let early = done_rx.recv_timeout(WAIT_PROOF);
            let waited = early.is_err();
            // Write 2 of 2, still under the lock: the launch publishes its
            // record, and the key stops looking like a leftover. `serve` does
            // this at `record.write()`, well before it drops the guard.
            // Only the record: the key was written above, by the launch, and
            // must not be re-planted here. `plant_service_record` would rewrite
            // it, and a rewritten key is present at the end of the test whether
            // or not the sweep deleted it — which is exactly how the two key
            // assertions below would stop being tripwires.
            let published =
                plant_service_record_without_its_key(&paths, ID, "starting", std::process::id())
                    .map(|_| ());
            // Nothing above this line may panic: the worker is blocked on this
            // guard, and `scope` would join it forever.
            drop(launch_lock);
            let finished = match early {
                Ok(finished) => finished,
                Err(_) => done_rx
                    .recv_timeout(Duration::from_mins(1))
                    .expect("prune completes once the launch lock is released"),
            };
            (waited, published, finished)
        });
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, ID).exists();
        let key_value = endpoint_keys::endpoint_api_key(&paths, ID);
        let _ = fs::remove_dir_all(&root);

        published.context("failed to publish the launch's record under the lock")?;
        assert!(
            waited,
            "prune finished while the managed-launch lock was held, so it never \
             acquired it — a launch between its key write and its record write \
             is still exposed:\n{finished:?}"
        );
        let (removed_files, skipped_live, text) =
            finished.map_err(|error| anyhow::anyhow!("prune failed: {error}"))?;
        assert!(
            key_exists,
            "the starting server's endpoint key must survive a concurrent \
             --any-age prune:\n{text}"
        );
        assert_eq!(
            key_value.as_deref(),
            Some("live-secret"),
            "the key the starting server needs must survive intact"
        );
        assert_eq!(
            removed_files, 0,
            "the published record makes its key a companion, not a leftover:\n{text}"
        );
        assert_eq!(
            skipped_live, 1,
            "the launch is live by the time prune reads the directory:\n{text}"
        );
        Ok(())
    }

    /// Persist a live-looking managed record claiming `gpu` — the same shape a
    /// real launch writes, with the current process id as the supervisor so the
    /// liveness refresh in `load_managed_services` keeps it "starting" (and thus
    /// counted by `busy_gpu_indices`).
    fn write_claiming_record(paths: &AppPaths, service_id: &str, port: u16, gpu: &[u32]) {
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "starting".to_owned();
        record.gpu_indices = gpu.to_vec();
        record.write().expect("write claiming record");
    }

    #[test]
    fn launch_lock_makes_gpu_select_and_claim_atomic() {
        // Regression for the serve read-select-launch race: the busy-GPU read and
        // the claiming record write must happen under one lock, or two concurrent
        // `--gpu auto` serves both read the same GPU as free and land on it.
        //
        // The test does NOT take the lock itself — that would only prove
        // `FileLock` excludes (already covered by
        // `file_lock_serializes_concurrent_holders` in rocm-core). It calls
        // `select_gpu_indices_under_launch_lock`, the production helper `serve()`
        // uses, whose contract is that it returns the guard *it* acquired together
        // with the selection; the test holds that guard across the claim exactly
        // as `serve()` holds it until `spawn_managed_engine_child` persists the
        // record. Delete the `FileLock::acquire` from that helper and this test
        // goes red: both threads then select GPU 0.
        //
        // Determinism: the barrier releases both threads together and each sleeps
        // between select and claim, so an unlocked helper double-books GPU 0
        // regardless of scheduling skew, while the locked helper forces the second
        // thread to observe the first thread's claim.
        let (root, paths) = test_paths("launch-lock-atomic-claim");
        paths.ensure().expect("prepare paths");
        let detected = Some(2_usize);

        let barrier = std::sync::Barrier::new(2);
        let selections = std::thread::scope(|scope| {
            let handles: Vec<_> = [("svc-race-a", 21001_u16), ("svc-race-b", 21002_u16)]
                .into_iter()
                .map(|(service_id, port)| {
                    let paths = &paths;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        // The exact call `serve()` makes: the helper acquires the
                        // launch lock and selects under it, handing the guard back.
                        // `None` visibility keeps selection mask-unaware for the
                        // test host; `pinned` `None` + `cpu_only` false is the
                        // `--gpu auto` path that reads live busy-GPU state.
                        let (gpu, lock) = select_gpu_indices_under_launch_lock(
                            paths,
                            false,
                            None,
                            || detected,
                            None,
                            None,
                        )
                        .expect("auto GPU selection under launch lock");
                        // Widen the select→claim window so an unlocked helper
                        // deterministically double-books GPU 0; under the lock the
                        // second thread cannot enter until we claim.
                        std::thread::sleep(Duration::from_millis(50));
                        write_claiming_record(paths, service_id, port, &gpu);
                        drop(lock);
                        gpu
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("selection thread joins"))
                .collect::<Vec<_>>()
        });

        let mut picked: Vec<u32> = selections.into_iter().flatten().collect();
        picked.sort_unstable();
        assert_eq!(
            picked,
            vec![0, 1],
            "serialized select-then-claim must hand out distinct GPUs, got {picked:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn services_prune_dry_run_reports_the_plan_and_removes_nothing() -> Result<()> {
        let (root, paths) = test_paths("services-prune-dry-run");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-dry", "failed", 999_999_999)?;

        let outcome = prune_managed_service_records(&paths, 0, true, false);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let engine_state_exists = record.engine_state_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(manifest_exists, "--dry-run must not delete the record");
        assert!(engine_state_exists, "--dry-run must not delete anything");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        assert!(outcome.text.contains("- svc-dry (status: failed"));
        assert!(
            outcome
                .text
                .contains("Nothing was removed. Re-run without --dry-run to remove.")
        );
        Ok(())
    }

    /// The `--all` view advertises `rocm services restart <id> --yes` for every
    /// record it lists. Pruning a server that died a minute ago would destroy
    /// that affordance and the log explaining the failure, so the default age
    /// gate keeps it.
    #[test]
    fn services_prune_default_age_keeps_a_just_stopped_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a record written seconds ago must survive the default prune:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert!(
            outcome
                .text
                .contains("svc-fresh changed less than 24 hours ago"),
            "prune must say why it was kept:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The age gate reads the manifest's modification time, and
    /// `load_managed_services` rewrites that manifest the first time it observes
    /// that a `ready` server has died — a crash, a kill, a reboot. Read after
    /// that rewrite the record looks newer than the prune run itself, so the
    /// fail-closed branch keeps it: on a host whose servers died weeks ago and
    /// have not been listed since, `rocm services prune --yes` removed nothing
    /// at all and called every one of them too recent to touch.
    #[test]
    fn services_prune_default_age_removes_a_record_the_refresh_rewrote() -> Result<()> {
        let (root, paths) = test_paths("services-prune-refresh-rewrite");
        paths.ensure()?;
        // `ready` with a dead pid is the state that triggers the rewrite: the
        // liveness refresh demotes the status and persists the demotion.
        let record = plant_service_record(&paths, "svc-long-dead", "ready", 999_999_999)?;
        let month_ago = SystemTime::now() - Duration::from_hours(24 * 30);
        fs::File::options()
            .write(true)
            .open(&record.manifest_path)?
            .set_modified(month_ago)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !manifest_exists,
            "a record last written a month ago must be pruned even though the \
             liveness refresh rewrote its manifest during this run:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(
            outcome.skipped_recent, 0,
            "the refresh's own rewrite must not make a month-old record recent:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// A cleanup that keeps things silently is indistinguishable from one that
    /// found nothing, and what it keeps is exactly what a user debugging a fresh
    /// failure still wants. The count and the way to override it both have to be
    /// on screen.
    #[test]
    fn services_prune_says_how_many_it_kept_for_being_recent() -> Result<()> {
        let (root, paths) = test_paths("services-prune-recent-report");
        paths.ensure()?;
        plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let _ = fs::remove_dir_all(&root);

        assert_eq!(outcome.skipped_recent, 1);
        assert!(
            outcome.text.contains("  too recent, kept: 1"),
            "the summary must count what it kept:\n{}",
            outcome.text
        );
        assert!(
            outcome.text.contains("rocm services prune --any-age --yes"),
            "the summary must name the flag that includes them:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// Parse a real `rocm services prune` command line and return exactly what
    /// the dispatch at `ServicesCommand::Prune` would hand
    /// [`prune_managed_service_records`]. Nothing here re-implements the flag
    /// mapping: it runs [`service_prune_min_age_hours`], the same function the
    /// command uses, so a test driving this covers the wiring and not a copy of
    /// it.
    fn parse_services_prune_args(argv: &[&str]) -> Result<(u64, bool, bool)> {
        let cli = Cli::try_parse_from(argv)?;
        let Some(Command::Services {
            command:
                Some(ServicesCommand::Prune {
                    older_than_hours,
                    any_age,
                    dry_run,
                    yes,
                }),
        }) = cli.command
        else {
            bail!("{argv:?} did not parse as `services prune`");
        };
        Ok((
            service_prune_min_age_hours(older_than_hours, any_age),
            dry_run,
            yes,
        ))
    }

    /// `--any-age` is the reachable form of `--older-than-hours 0`: the summary
    /// points at it, so it has to actually take the record the default kept.
    ///
    /// Driven from the argument vector rather than by passing 0 by hand —
    /// otherwise this is just another call with `hours = 0` and the flag's only
    /// wiring, the collapse in [`service_prune_min_age_hours`], is never
    /// executed by any test.
    #[test]
    fn services_prune_any_age_removes_a_just_stopped_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-any-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
                .and_then(|(hours, dry_run, yes)| {
                    assert_eq!(hours, 0, "--any-age must collapse to the zero-age rule");
                    assert!(!dry_run);
                    assert!(yes);
                    prune_managed_service_records(&paths, hours, dry_run, yes)
                });
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !manifest_exists,
            "--any-age must remove the record the default keeps:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(
            outcome.skipped_recent, 0,
            "nothing is 'too recent' once the age rule is off:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// Without the flag the very same parsed command line must keep the record,
    /// which is what makes the assertion above about `--any-age` and not about
    /// prune deleting things in general.
    #[test]
    fn services_prune_without_any_age_keeps_the_default_threshold() -> Result<()> {
        let (root, paths) = test_paths("services-prune-no-any-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let parsed = parse_services_prune_args(&["rocm", "services", "prune", "--yes"]);
        let outcome = parsed.and_then(|(hours, dry_run, yes)| {
            assert_eq!(
                hours, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS,
                "no --any-age means the default age rule still applies"
            );
            prune_managed_service_records(&paths, hours, dry_run, yes)
        });
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "the default must keep it:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.skipped_recent, 1);
        Ok(())
    }

    /// `--any-age` and an explicit `--older-than-hours` would be two answers to
    /// one question; clap has to reject the pair rather than silently pick one.
    #[test]
    fn services_prune_rejects_any_age_with_an_explicit_age() {
        let error = Cli::try_parse_from([
            "rocm",
            "services",
            "prune",
            "--any-age",
            "--older-than-hours",
            "5",
        ])
        .expect_err("the two age arguments must conflict");
        let rendered = error.to_string();
        assert!(
            rendered.contains("--any-age") && rendered.contains("--older-than-hours"),
            "the conflict must name both arguments:\n{rendered}"
        );
    }

    /// `collect_service_orphans`' doc comment leans on this: an *unparseable*
    /// manifest is skipped by `load_managed_services`, so widening the orphan
    /// rule from "no `<id>.json` on disk" to "no record in the list" would
    /// delete the log of the one record a user most needs to read, and the
    /// corrupt manifest with it. Nothing asserted that until now — the other
    /// tests only ever plant a fully absent manifest.
    #[test]
    fn services_prune_keeps_a_corrupt_manifest_and_its_siblings() -> Result<()> {
        let (root, paths) = test_paths("services-prune-corrupt");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-corrupt", "failed", 999_999_999)?;
        // Valid JSON, not a valid record: `serde_json::from_slice` fails, so
        // `load_managed_services` skips it without reporting an error.
        fs::write(&record.manifest_path, b"{\"service_id\": 12345}")?;
        let key_path = endpoint_keys::endpoint_key_file_path(&paths, "svc-corrupt");

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let state_exists = record.engine_state_path.exists();
        let key_exists = key_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a manifest that cannot be parsed must never be deleted:\n{}",
            outcome.text
        );
        assert!(
            log_exists,
            "the log of an unreadable record is exactly what a user needs:\n{}",
            outcome.text
        );
        assert!(state_exists, "the engine state must not look orphaned");
        assert!(key_exists, "the endpoint key must not look orphaned");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        Ok(())
    }

    /// The status string is not proof of death. `refresh_from_engine_state`
    /// adopts `failed` straight from the engine's own state file and the
    /// liveness refresh then returns early for a non-live status, so a server
    /// whose engine reported failure while its process is still up would be
    /// removable — taking the log and the 0600 endpoint key of a live process.
    #[test]
    fn services_removal_refuses_a_failed_record_whose_process_is_still_alive() -> Result<()> {
        let (root, paths) = test_paths("services-remove-live-pid");
        paths.ensure()?;
        // This test process: a pid that is unambiguously running.
        let record = plant_service_record(&paths, "svc-zombie", "failed", std::process::id())?;

        let remove_error = remove_managed_service_record(&paths, "svc-zombie", true);
        let prune = prune_managed_service_records(&paths, 0, false, true);
        let prune = match prune {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        let message = remove_error
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("cannot be removed while it is running"),
            "a record with a live pid must be refused whatever its status: {message}"
        );
        assert!(manifest_exists, "nothing may be deleted:\n{}", prune.text);
        assert_eq!(prune.removed_records, 0);
        assert_eq!(
            prune.skipped_live, 1,
            "prune must count it as still running:\n{}",
            prune.text
        );
        Ok(())
    }

    /// `rocm uninstall`'s review warns that a remote session keeps running on
    /// someone else's machine, and names the command that stops it properly.
    ///
    /// Nothing exercised that warning before: no test planted a remote session
    /// record, so the whole sentence -- including the ASCII punctuation this
    /// change put in it (ROCMAI-543) -- shipped uncovered. The remediation is
    /// asserted alongside the encoding because a warning that stops naming
    /// `rocm remote stop` leaves the user with no way to act on it.
    #[test]
    fn uninstall_warns_about_remote_sessions_in_ascii_and_names_the_stop_command() -> Result<()> {
        let (root, paths) = test_paths("uninstall-remote-session-warning");
        paths.ensure()?;
        let session = crate::remote::session::RemoteSessionRecord {
            session_id: crate::remote::session::RemoteSessionRecord::id_for("dev-box", 8000),
            target: "dev-box".to_owned(),
            peer_host: "dev-box".to_owned(),
            ssh_port: None,
            model: "qwen".to_owned(),
            remote_service_id: "svc-remote".to_owned(),
            remote_cli: "rocm".to_owned(),
            remote_port: 8000,
            tailnet_port: 8000,
            base_url: "http://dev-box:8000/v1".to_owned(),
            created_at_unix_ms: 1,
        };
        session.write(&paths)?;

        // `keep_binaries` so the plan never reaches outside the sandbox: the
        // binary paths are the host's real ones and have nothing to do with
        // this warning.
        let options = UninstallOptions {
            dry_run: true,
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let plan = build_uninstall_plan(&paths, &options);
        let _ = fs::remove_dir_all(&root);
        let plan = plan?;

        let warning = plan
            .warnings
            .iter()
            .find(|warning| warning.contains("remote session record(s)"))
            .unwrap_or_else(|| {
                panic!("a planted remote session must be warned about: {plan:#?}");
            });
        assert!(
            warning.contains(&session.session_id),
            "the warning must name the session it found: {warning}"
        );
        assert!(
            warning.contains("rocm remote stop <session>"),
            "the warning must name the command that stops the remote model: {warning}"
        );
        // ROCMAI-543: printed verbatim by `render_uninstall_plan`, so non-ASCII
        // punctuation here renders as mojibake like "ΓÇö" on a Windows console
        // using the OEM code page.
        assert!(
            warning.is_ascii(),
            "the remote-session warning must be ASCII-only (ROCMAI-543): {warning}"
        );
        Ok(())
    }

    /// A file `prune` cannot delete must be *reported*, not swallowed and not
    /// propagated: the plan, the per-record progress and the audit event all
    /// have to survive it, because a destructive command that loses its own
    /// account of what it deleted is worse than one that fails. The record whose
    /// file survived also must not be counted as removed.
    ///
    /// A non-empty directory standing where the engine state file belongs is the
    /// portable way to make `fs::remove_file` fail on both supported hosts; the
    /// premise is asserted rather than assumed.
    #[test]
    fn services_prune_reports_a_file_it_could_not_remove() -> Result<()> {
        let (root, paths) = test_paths("services-prune-undeletable");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-stuck", "failed", 999_999_999)?;
        let stuck = record.engine_state_path.clone();
        fs::remove_file(&stuck)?;
        fs::create_dir(&stuck)?;
        fs::write(stuck.join("held.json"), b"{}")?;
        assert!(
            fs::remove_file(&stuck).is_err(),
            "premise: {} must be undeletable by `remove_file`",
            stuck.display()
        );

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let stuck_exists = stuck.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(stuck_exists, "premise: the stuck path must survive");
        assert!(
            !manifest_exists,
            "one unremovable file must not strand the other three:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_files, 3);
        assert_eq!(
            outcome.removed_records, 0,
            "a record that still owns a file on disk is not removed:\n{}",
            outcome.text
        );
        assert_eq!(
            outcome.failures.len(),
            1,
            "the failure must be collected so the caller can fail the command:\n{}",
            outcome.text
        );
        assert!(
            outcome.failures[0].contains(&stuck.display().to_string()),
            "the failure must name the path: {:?}",
            outcome.failures
        );
        assert!(
            outcome
                .text
                .contains("1 local server record(s) would be removed"),
            "the plan must still be rendered:\n{}",
            outcome.text
        );
        assert!(
            outcome.text.contains("1 file(s) could not be removed:"),
            "the run must say what it could not delete:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("Re-running is safe: everything already removed stays removed."),
            "the user needs to know a re-run is not destructive twice over:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The window the pre-delete liveness re-check exists to narrow: the plan is
    /// built from a snapshot, and a `rocm services restart <id> --yes` landing
    /// between that snapshot and the delete would otherwise have its log and its
    /// 0600 endpoint key deleted out from under a serving process.
    ///
    /// Driven through [`apply_service_prune_plan`] with a hand-built plan
    /// because the window cannot be opened from outside: `prune` builds and
    /// applies the plan in one call, so a record is either live for both halves
    /// or dead for both. The two `skipped_live` assertions elsewhere in this
    /// module both plant an already-live record, which is satisfied by the
    /// *plan-building* skip and never reaches this branch.
    #[test]
    fn services_prune_leaves_a_record_that_restarted_after_the_plan_was_built() -> Result<()> {
        let (root, paths) = test_paths("services-prune-relaunched");
        paths.ensure()?;
        // Live at apply time. The current process is a guaranteed-live pid.
        let record = plant_service_record(&paths, "svc-relaunched", "ready", std::process::id())?;
        // The entry a plan built moments earlier, while the record was still
        // stopped, would carry into the delete loop.
        let artifacts = match service_record_artifacts(&paths, "svc-relaunched", "vllm") {
            Ok(artifacts) => artifacts,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let plan = ServicePrunePlan {
            remove: vec![ServicePruneEntry {
                service_id: "svc-relaunched".to_owned(),
                engine: "vllm".to_owned(),
                status: "stopped".to_owned(),
                artifacts,
            }],
            ..ServicePrunePlan::default()
        };

        let mut outcome = ServicePruneOutcome::default();
        apply_service_prune_plan(&paths, &plan, &mut outcome);
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, "svc-relaunched").exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a record that came back to life must not be deleted:\n{}",
            outcome.text
        );
        assert!(
            log_exists,
            "the log of a live process must survive:\n{}",
            outcome.text
        );
        assert!(
            key_exists,
            "the 0600 endpoint key of a live process must survive:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        assert_eq!(
            outcome.skipped_live, 1,
            "the re-check's skip must be counted like any other:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("  svc-relaunched started again while this ran and was left alone."),
            "the skip must be on screen, not silent:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// `prunable_by_modified` promises to fail *closed*: a modification time
    /// that yields no age — a file stamped in the future by clock skew or a
    /// stray `touch` — is kept, and only the explicit zero-age opt-out overrides
    /// that. Neither half was asserted anywhere: every other test plants a
    /// readable past time, for which `is_some_and` and `is_none_or` agree and
    /// the `min_age.is_zero()` early return is unreachable.
    #[test]
    fn services_prune_keeps_a_future_dated_record_until_any_age() -> Result<()> {
        let (root, paths) = test_paths("services-prune-future-mtime");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-future", "failed", 999_999_999)?;
        // `now.duration_since(future)` is an error, so `age_from_modified` has
        // no age to compare against the threshold.
        let backdate = |to: SystemTime| -> Result<()> {
            fs::File::options()
                .write(true)
                .open(&record.manifest_path)?
                .set_modified(to)?;
            Ok(())
        };
        let next_year = SystemTime::now() + Duration::from_hours(24 * 365);
        backdate(next_year)?;
        assert!(
            age_from_modified(path_modified(&record.manifest_path), SystemTime::now()).is_none(),
            "premise: a future-stamped manifest must have no age"
        );

        let kept =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let kept = match kept {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let kept_manifest = record.manifest_path.exists();

        // Re-stamped so the second run really does meet the no-age case rather
        // than a time the first run's refresh may have rewritten to `now`.
        // Ignored rather than `?`-ed: if the first run wrongly deleted the
        // manifest there is nothing left to stamp, and the `kept_manifest`
        // assertion below has to be what reports that, not an "os error 2".
        let _ = backdate(next_year);
        let taken = parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
            .and_then(|(hours, dry_run, yes)| {
                assert_eq!(hours, 0, "--any-age must collapse to the zero-age rule");
                prune_managed_service_records(&paths, hours, dry_run, yes)
            });
        let taken = match taken {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let taken_manifest = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            kept_manifest,
            "a time that cannot be aged must fail closed under the default rule:\n{}",
            kept.text
        );
        assert_eq!(kept.removed_records, 0);
        assert_eq!(
            kept.skipped_recent, 1,
            "the keep must be reported, not silent:\n{}",
            kept.text
        );
        assert!(
            !taken_manifest,
            "--any-age is the opt-out that takes it anyway:\n{}",
            taken.text
        );
        assert_eq!(taken.removed_records, 1);
        assert_eq!(taken.skipped_recent, 0);
        Ok(())
    }

    #[test]
    fn services_prune_requires_yes_unless_dry_run() -> Result<()> {
        let (root, paths) = test_paths("services-prune-yes");
        paths.ensure()?;
        plant_service_record(&paths, "svc-prune-yes", "failed", 999_999_999)?;

        let error = prune_managed_service_records(&paths, 0, false, false)
            .expect_err("prune without --yes must fail");
        let manifest_exists = paths.service_manifest_path("svc-prune-yes").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(message.contains("requires --yes"), "unexpected: {message}");
        assert!(
            message.contains("rocm services prune --dry-run"),
            "the error must name the preview command: {message}"
        );
        assert!(manifest_exists, "a refused prune must delete nothing");
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
    fn update_dry_run_does_not_require_apply() {
        Cli::try_parse_from(["rocm", "update", "--dry-run"])
            .expect("update --dry-run should parse without --apply");
        Cli::try_parse_from(["rocm", "update", "--apply", "--dry-run"])
            .expect("update --apply --dry-run should still parse");
        Cli::try_parse_from(["rocm", "update", "--dry-run", "--runtime", "rocm-6.2"])
            .expect("update --dry-run --runtime should parse without --apply");
        Cli::try_parse_from(["rocm", "update", "--dry-run", "--activate"])
            .expect("update --dry-run --activate should parse without --apply");
    }

    #[test]
    fn update_dry_run_conflicts_with_json() {
        Cli::try_parse_from(["rocm", "update", "--dry-run", "--json"]).expect_err(
            "update --dry-run --json should be rejected instead of silently dropping --json",
        );
    }

    // This only pins the free predicate's truth table. The actual dispatch
    // wiring — that `rocm update --dry-run` really does reach the preview
    // path without requiring --apply — is covered by e2e scenario
    // `update-dry-run-reaches-preview-path-without-apply`
    // (tests/e2e-cucumber/features/update.feature).
    #[test]
    fn update_should_preview_or_apply_includes_dry_run() {
        assert!(
            !update_should_preview_or_apply(false, false),
            "plain `rocm update` should stay on the read-only status report"
        );
        assert!(
            update_should_preview_or_apply(false, true),
            "the predicate must say dry-run alone should preview"
        );
        assert!(update_should_preview_or_apply(true, false));
        assert!(update_should_preview_or_apply(true, true));
    }

    #[test]
    fn remote_is_structured_not_freeform() {
        // `rocm remote …` reads like a plain-English request, so without an
        // entry in the structured allowlist the natural-language planner
        // swallows it and the real command becomes unreachable. This guards the
        // allowlist against losing `remote`.
        let invocation = parse_freeform_invocation(&[
            "remote".to_owned(),
            "targets".to_owned(),
            "--tag".to_owned(),
            "gpu".to_owned(),
        ]);
        assert!(!treat_as_natural_language(&invocation.request_args));
        assert!(!should_treat_as_freeform(&invocation));

        Cli::try_parse_from(["rocm", "remote", "targets"])
            .expect("remote targets should be a real command");
        Cli::try_parse_from(["rocm", "remote", "targets", "--tag", "gpu"])
            .expect("remote targets should accept a tag filter");
        // The group has no useful default action, so a bare `rocm remote` must
        // show help rather than silently doing something.
        Cli::try_parse_from(["rocm", "remote"])
            .expect_err("bare `rocm remote` should require a subcommand");
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
    fn install_sdk_devel_flag_defaults_off_and_wires_through_when_passed() {
        let cli = Cli::try_parse_from(["rocm", "install", "sdk"])
            .expect("install sdk should parse with no flags");
        match cli.command {
            Some(Command::Install {
                target: InstallTarget::Sdk { devel, .. },
            }) => assert!(!devel, "--devel should default to false"),
            other => panic!("expected an install sdk target, got {other:?}"),
        }

        let cli = Cli::try_parse_from(["rocm", "install", "sdk", "--devel"])
            .expect("install sdk should accept --devel");
        match cli.command {
            Some(Command::Install {
                target: InstallTarget::Sdk { devel, .. },
            }) => assert!(devel, "--devel should set the flag to true"),
            other => panic!("expected an install sdk target, got {other:?}"),
        }
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
            "storage",
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
            "bench",
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
    fn t5_bench_load_clap_parse_smoke() {
        // T5: verify BenchCommand::Load parses correctly including comma-separated concurrency.
        let cli = Cli::try_parse_from([
            "rocm",
            "bench",
            "load",
            "--endpoint",
            "http://x",
            "--concurrency",
            "1,8,32,64",
        ])
        .expect("rocm bench load should parse");
        match cli.command {
            Some(Command::Bench {
                command: BenchCommand::Load { concurrency, .. },
            }) => {
                assert_eq!(concurrency, vec![1u32, 8, 32, 64]);
            }
            other => panic!("expected Bench/Load, got {other:?}"),
        }
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

        // The claim itself (onboarding only opens via an explicit `n` on the
        // Observe tab, never automatically) is proven by
        // `crates/rocm-dash-tui/src/app/event_loop.rs`'s
        // `startup_focus_gate_only_opens_onboarding_for_explicit_setup_focus`
        // test and the `onboarding.rs` module doc — this assertion only
        // guards the string, not the behavior.
        assert!(rendered.contains("Onboarding will not reopen automatically"));
        assert!(rendered.contains("ROCm installs were not deleted"));
        assert!(rendered.contains("API keys"));
        // ROCMAI-543: the assertions above stop before the ` - open it from
        // `rocm dash`'s Observe tab` clause, so they stay green with the em
        // dash put back. This covers the whole block that `rocm setup --reset`
        // prints, including any line added to it later.
        assert!(
            rendered.is_ascii(),
            "the setup-reset prompt must be ASCII-only; non-ASCII punctuation renders as \
             mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );
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

        // See the pointer comment in
        // `setup_reset_cli_output_is_plain_and_persists_first_time_prompt`
        // above: this only guards the string, not the underlying behavior.
        assert!(rendered.contains("status: first-time setup available - open manually"));
        assert!(rendered.contains("active_runtime_status: <unset>"));
        Ok(())
    }

    #[test]
    fn restarting_a_keyless_public_service_names_the_public_bind_not_the_key_flag() -> Result<()> {
        // The same defect at a real call site rather than through the guard's own
        // arguments. `restart` reads `requires_api_key` off the record on disk, so
        // a record written by an ordinary public-bind launch must not claim the
        // service asked for `--require-api-key` — the branch order means a record
        // that claims it wins, and its remediation drops `--allow-public-bind`.
        let (root, paths) = test_paths("restart-public-no-key");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "vllm-public-2000",
            "vllm",
            "qwen",
            "qwen-canonical",
            "0.0.0.0",
            12000,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        // What a plain `--host 0.0.0.0 --allow-public-bind` launch records: the
        // bind is public, and the flag was never passed.
        record.requires_api_key = false;
        record.write()?;

        // No key file: the situation after a stop, which drops it.
        let error = restart_internal_managed_service(&paths, "vllm-public-2000", None)
            .expect_err("a keyless public service must not be restarted");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("--allow-public-bind"),
            "the refusal must name the command that brings it back public: {rendered}"
        );
        assert!(
            !rendered.contains("--require-api-key"),
            "a service that never passed the flag must not be told it did: {rendered}"
        );

        // The refusal happens before the stop, so the service is left alone.
        assert!(
            load_managed_service(&paths, "vllm-public-2000").is_ok(),
            "a refused restart must not have removed the record"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn restart_refuses_a_public_service_without_a_key_before_stopping_it() {
        // The guard runs before the stop, so a refused restart must leave the
        // record exactly as it was rather than taking down a running service.
        let (root, paths) = test_paths("restart-public-no-key");
        let service_id = "svc-public-nokey";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "model-ref",
            "canonical/model",
            "0.0.0.0",
            11435,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "running".to_owned();
        record.write().unwrap();

        let error = restart_internal_managed_service(&paths, service_id, None).unwrap_err();
        assert!(
            error.to_string().contains("without authentication"),
            "{error:#}"
        );

        // Proves the *ordering*, not just the refusal: had the guard run after
        // `stop_internal_managed_service`, the stop would have written
        // "stopped". It reaches that state here because the record's only pid is
        // the test's own (`engine_pid` is None and `terminate_recorded_service_pids`
        // skips the caller's pid), so the stop confirms termination trivially.
        let after = load_managed_service(&paths, service_id).unwrap();
        assert_ne!(
            after.status, "stopped",
            "a refused restart must not stop the service"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A public record with a dead pid and a stored endpoint key, for the
    /// liveness-refresh cases below. The port is one nothing listens on, so the
    /// refresh's endpoint probe fails and it falls through to the pid check.
    fn dead_public_service_with_key(
        paths: &AppPaths,
        service_id: &str,
        port: u16,
    ) -> ManagedServiceRecord {
        paths.ensure().unwrap();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "vllm",
            "model-ref",
            "canonical/model",
            "0.0.0.0",
            port,
            "managed",
            // A pid far above any plausible live process, as in
            // `dead_managed_service_allows_relaunch`.
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(paths, service_id, "secret-key").unwrap();
        record
    }

    #[test]
    fn crashed_public_service_keeps_its_endpoint_key() {
        // A crash (OOM kill, host reboot, panic) leaves the record at "running"
        // with dead pids and no stop marker. Dropping the key here would make
        // the fail-closed respawn guards refuse every later `rocm services
        // restart` and every daemon recovery attempt — permanently, because
        // nothing can re-mint the key. The refresh happens on every read
        // (`rocm services list`), so this must survive it.
        let (root, paths) = test_paths("liveness-crash-keeps-key");
        let service_id = "svc-crashed-public";
        let mut record = dead_public_service_with_key(&paths, service_id, 11982);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(changed, "a dead service must be demoted to stopped");
        assert_eq!(record.status, "stopped");
        assert_eq!(
            endpoint_keys::endpoint_api_key(&paths, service_id).as_deref(),
            Some("secret-key"),
            "a crashed public service must stay restartable"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unconfirmed_stop_clears_the_endpoint_key_once_the_processes_are_gone() {
        // The other half: a stop that could not confirm termination leaves the
        // key in place (the engine may still be alive and enforcing it) and
        // records the intent. Once the processes are observed gone, the deferred
        // cleanup runs, so no plaintext secret is stranded for a service the
        // operator did ask to stop.
        let (root, paths) = test_paths("liveness-pending-stop-clears-key");
        let service_id = "svc-pending-stop";
        let mut record = dead_public_service_with_key(&paths, service_id, 11983);
        record.stop_requested_unix_ms = Some(1);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(changed);
        assert_eq!(record.status, "stopped");
        assert_eq!(
            endpoint_keys::endpoint_api_key(&paths, service_id),
            None,
            "a requested stop must still drop the key"
        );
        assert_eq!(
            record.stop_requested_unix_ms, None,
            "the marker is consumed, so the cleanup does not run again"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pending_stop_cleanup_runs_even_once_the_record_reads_stopped() {
        // Proves the cleanup sits *before* the `managed_service_is_live` gate:
        // a record that reached "stopped" by another route (an engine state
        // refresh, a concurrent writer) would otherwise early-return and strand
        // the key of a service the operator stopped.
        let (root, paths) = test_paths("liveness-pending-stop-when-stopped");
        let service_id = "svc-pending-stop-stopped";
        let mut record = dead_public_service_with_key(&paths, service_id, 11984);
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = Some(1);

        let changed = refresh_managed_service_runtime_liveness(&paths, &mut record);

        assert!(
            changed,
            "consuming the marker is a record change worth writing"
        );
        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert_eq!(record.stop_requested_unix_ms, None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stop_records_its_marker_on_disk_before_any_process_is_signalled() {
        // The daemon's `server-recover` watcher restarts a managed service whose
        // endpoint stops answering unless the record carries a stop marker. A
        // stop that wrote the marker only after terminating the processes would
        // leave that whole window — the engine `Stop`, then up to the grace
        // period per PID — open for the daemon to bring the service back. So at
        // the moment the processes are signalled the marker must already be on
        // disk, not just in the stop's in-memory copy.
        let (root, paths) = test_paths("stop-marker-before-signal");
        paths.ensure().unwrap();
        let service_id = "svc-stop-marker-order";
        // A port nothing listens on, so the lemonade unload fails fast instead
        // of reaching a real server.
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "lemonade",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            port,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write().unwrap();

        // Read the file as-is: `load_managed_service` runs the liveness refresh,
        // which would consume the marker of a record whose PIDs are dead.
        let on_disk = || {
            serde_json::from_slice::<ManagedServiceRecord>(
                &fs::read(paths.service_manifest_path(service_id)).unwrap(),
            )
            .unwrap()
        };
        let seen_at_signal = std::cell::Cell::new(None);
        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let on_disk = on_disk();
            seen_at_signal.set(Some(on_disk.stop_requested_unix_ms));
            (Vec::new(), false)
        })
        .unwrap();

        assert!(
            matches!(seen_at_signal.get(), Some(Some(_))),
            "the stop marker must be persisted before the processes are signalled, \
             saw {:?}",
            seen_at_signal.get()
        );
        // The final-state gating is unchanged: an unconfirmed stop keeps the
        // marker it wrote up front.
        assert!(on_disk().stop_requested_unix_ms.is_some());
        assert_eq!(report["signaled_pids"], serde_json::json!([]));
        let _ = fs::remove_dir_all(root);
    }

    /// A stopped-lemonade record on a port nothing listens on, so the stop's
    /// unload fails fast instead of reaching a real server.
    fn stop_test_record(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "lemonade",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            port,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record
    }

    /// The manifest as it is on disk, without `load_managed_service`'s refresh.
    fn raw_manifest(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        serde_json::from_slice(&fs::read(paths.service_manifest_path(service_id)).unwrap()).unwrap()
    }

    #[test]
    fn stop_does_not_confirm_when_a_restart_recorded_new_processes_meanwhile() {
        // A stop's verdict covers the processes it read and signalled. A restart
        // landing while it ran records a new supervisor and clears the marker;
        // claiming that service stopped, or deleting the key it serves with,
        // would be false. Here every process the stop handled is confirmed gone,
        // so only the re-read can tell.
        let (root, paths) = test_paths("stop-restart-meanwhile");
        paths.ensure().unwrap();
        let service_id = "svc-stop-restart-meanwhile";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut restarted = raw_manifest(&paths, service_id);
            restarted.status = "recovering".to_owned();
            restarted.supervisor_pid = 4_242_424;
            restarted.supervisor_start_ticks = Some(7);
            restarted.stop_requested_unix_ms = None;
            restarted.write().unwrap();
            (Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["status"], "recovering", "{report}");
        assert_eq!(on_disk.status, "recovering");
        assert_eq!(on_disk.supervisor_pid, 4_242_424, "the restart's PID stays");
        assert!(
            on_disk.stop_requested_unix_ms.is_some(),
            "an unconfirmed stop re-stamps the marker the restart cleared"
        );
        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
    }

    #[test]
    fn stop_final_write_keeps_what_others_wrote_while_it_ran() {
        // The stop's last write applies its own fields to the record as it is
        // now, not to the copy it loaded before the engine stop and the wait.
        let (root, paths) = test_paths("stop-final-write-rereads");
        paths.ensure().unwrap();
        let service_id = "svc-stop-final-write-rereads";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut meanwhile = raw_manifest(&paths, service_id);
            meanwhile.restart_count = 7;
            meanwhile.write().unwrap();
            (Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["status"], "stopped", "{report}");
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.stop_requested_unix_ms, None);
        assert_eq!(on_disk.restart_count, 7, "a concurrent write survives");
        assert_eq!(key, None, "a confirmed stop drops the key");
    }

    #[test]
    fn restart_does_not_start_an_engine_over_an_unconfirmed_stop() {
        // A restart used to discard its stop's result and start the new engine
        // regardless, next to a survivor still holding the port and the GPU.
        // Here the stop cannot confirm termination: the restart must say so,
        // start nothing, and leave the stop's own state standing.
        let (root, paths) = test_paths("restart-unconfirmed-stop");
        paths.ensure().unwrap();
        let service_id = "svc-restart-unconfirmed-stop";
        let record = stop_test_record(&paths, service_id);
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            (vec![999_999_999], false)
        });

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let log_created = record.log_path.exists();
        let _ = fs::remove_dir_all(root);
        let error = result.expect_err("an unconfirmed stop must not be restarted over");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("could not confirm"), "{rendered}");
        assert!(rendered.contains("no new engine was started"), "{rendered}");
        assert!(rendered.contains("999999999"), "names the pids: {rendered}");
        // The claims in that message, checked against the state they describe.
        assert!(
            !log_created,
            "nothing was started: the restart never opened the new run's log"
        );
        assert_eq!(on_disk.restart_count, 0, "no restart was recorded");
        assert_eq!(
            on_disk.supervisor_pid, 999_999_999,
            "the record still names the survivor"
        );
        assert!(
            on_disk.stop_requested_unix_ms.is_some(),
            "the stop request stays recorded on the service"
        );
        assert_eq!(
            key.as_deref(),
            Some("key-123"),
            "the key stays with the survivor"
        );
    }

    /// A record as a restart's confirmed stop leaves it, on disk.
    fn stopped_restart_record(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        let mut record = stop_test_record(paths, service_id);
        record.status = "stopped".to_owned();
        record.write().unwrap();
        raw_manifest(paths, service_id)
    }

    #[test]
    fn restart_readiness_write_keeps_a_stop_asked_for_while_it_waited() {
        // The restart's last write lands after a readiness wait of up to 45
        // seconds. A stop asked for in that window persists its marker before
        // it signals anything; writing the restart's own copy back erased it.
        let (root, paths) = test_paths("restart-readiness-keeps-marker");
        paths.ensure().unwrap();
        let service_id = "svc-restart-readiness-keeps-marker";
        let stopped = stopped_restart_record(&paths, service_id);
        let started = record_restarted_child(&paths, &stopped, None, std::process::id()).unwrap();

        let mut stopping = raw_manifest(&paths, service_id);
        stopping.stop_requested_unix_ms = Some(1_234);
        stopping.write().unwrap();

        let returned =
            record_restart_readiness(&paths, &started, EndpointReadiness::Unreachable).unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(
            on_disk.stop_requested_unix_ms,
            Some(1_234),
            "the stop survives"
        );
        assert_eq!(on_disk.supervisor_pid, std::process::id());
        assert_eq!(on_disk.restart_count, 1);
        assert_eq!(returned.stop_requested_unix_ms, Some(1_234));
    }

    #[test]
    fn restart_readiness_write_leaves_a_stop_that_finished_meanwhile() {
        // A stop that confirmed the new child gone during the wait wrote
        // "stopped"; the restart's verdict on a process that no longer exists
        // must not overwrite it.
        let (root, paths) = test_paths("restart-readiness-after-stop");
        paths.ensure().unwrap();
        let service_id = "svc-restart-readiness-after-stop";
        let stopped = stopped_restart_record(&paths, service_id);
        let started = record_restarted_child(&paths, &stopped, None, std::process::id()).unwrap();

        let mut finished = raw_manifest(&paths, service_id);
        finished.status = "stopped".to_owned();
        finished.write().unwrap();

        let returned =
            record_restart_readiness(&paths, &started, EndpointReadiness::Serving).unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.inference_verified_at_unix_ms, None);
        assert_eq!(
            returned.status, "stopped",
            "the caller sees the record as it is"
        );
    }

    #[test]
    fn restarted_child_record_names_the_runtime_the_restart_was_pinned_to() {
        // A move (`rocm runtimes activate --restart-services`) launches the
        // child on the pinned runtime, but the restart's own load re-adopted the
        // outgoing process's `runtime_id`/`env_id` from its engine state and
        // persisted them over the pin. The record the restart writes for its
        // child is built from disk, so it has to carry the pin again or it
        // names the runtime the service just left.
        let (root, paths) = test_paths("restart-child-runtime-pin");
        paths.ensure().unwrap();
        let service_id = "svc-restart-child-runtime-pin";
        let mut stopped = stop_test_record(&paths, service_id);
        stopped.status = "stopped".to_owned();
        stopped.runtime_id = Some("runtime-old".to_owned());
        stopped.env_id = Some("env-old".to_owned());
        stopped.write().unwrap();
        let stopped = raw_manifest(&paths, service_id);

        let started =
            record_restarted_child(&paths, &stopped, Some("runtime-new"), std::process::id())
                .unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(on_disk.runtime_id.as_deref(), Some("runtime-new"));
        assert_eq!(
            on_disk.env_id, None,
            "the pin drops the env, as a move does"
        );
        assert_eq!(started.runtime_id.as_deref(), Some("runtime-new"));
        assert_eq!(on_disk.supervisor_pid, std::process::id());
    }

    #[test]
    fn restart_reports_a_child_it_could_not_record_and_writes_nothing() {
        // Between the restart's stop and its spawn, another restart recorded
        // its own processes. Writing this child over them would hide theirs;
        // the restart has to report its child unrecorded instead.
        let (root, paths) = test_paths("restart-child-unrecorded");
        paths.ensure().unwrap();
        let service_id = "svc-restart-child-unrecorded";
        let stopped = stopped_restart_record(&paths, service_id);

        let mut other = raw_manifest(&paths, service_id);
        other.status = "running".to_owned();
        other.supervisor_pid = 4_242_424;
        other.supervisor_start_ticks = Some(7);
        other.write().unwrap();

        let result = record_restarted_child(&paths, &stopped, None, std::process::id());
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        let rendered = format!(
            "{:#}",
            result.expect_err("a changed record must be reported")
        );
        assert!(rendered.contains("was not recorded"), "{rendered}");
        assert!(
            rendered.contains(&std::process::id().to_string()),
            "{rendered}"
        );
        assert_eq!(
            on_disk.supervisor_pid, 4_242_424,
            "the other restart's PID stays"
        );
        assert_eq!(on_disk.restart_count, 0, "nothing was written");
    }

    #[test]
    fn serve_notes_surface_the_ignored_engine_flag_in_summary_mode() {
        // The interactive summary is what a default `rocm serve` prints, so a flag
        // the selected engine cannot honor has to be reported through this path —
        // not only on the plan path that an interactive run never takes.
        let note = "--gpu-memory-utilization applies only to vLLM; ignored for engine 'lemonade'";
        let notes = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            None,
            Some(note),
            None,
            false,
        );
        assert!(
            notes.iter().any(|entry| entry == note),
            "the ignored-flag note must reach the summary: {notes:?}"
        );

        let quiet = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            None,
            None,
            None,
            false,
        );
        assert!(
            !quiet
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "nothing to report when the flag was honored: {quiet:?}"
        );
    }

    #[test]
    fn serve_notes_pair_low_vram_with_the_vllm_utilization_hint() {
        // A busy discrete card that trips the low-VRAM warning: on vLLM the note
        // must also carry the concrete `--gpu-memory-utilization` workaround, so
        // the interactive summary tells the user how to avoid the OOM.
        let busy = [vram(0, 182_000, 192_000)];
        let notes = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            Some(&busy),
            None,
            Some(&host_gpu("gfx1100")),
            true,
        );
        assert!(
            notes.iter().any(|entry| entry.contains("has only")),
            "the low-VRAM warning must be present: {notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "vLLM low-VRAM notes must hint the utilization workaround: {notes:?}"
        );

        // The literal fragments the GPU-lane scenario
        // `@id:serve-vllm-low-vram-oom-guidance` matches on. That scenario runs
        // only where a real card exists, so pin the wording here too: a rename
        // then fails on every lane rather than silently on the one lane that can
        // observe it. `GPU 0 has only` is what makes the warning provably about
        // the *pinned* device; `vLLM reserves ~90%` is what distinguishes the
        // pre-launch hint from any other line that merely names the flag.
        assert!(
            notes.iter().any(|entry| entry.contains("GPU 0 has only")),
            "the warning must name the selected GPU: {notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|entry| entry.contains("vLLM reserves ~90%")),
            "the hint must explain vLLM's total-VRAM reservation: {notes:?}"
        );

        // The same busy card on a non-vLLM engine keeps the warning but omits the
        // vLLM-only knob, which that engine cannot honor.
        let lemonade = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            Some(&busy),
            None,
            Some(&host_gpu("gfx1100")),
            false,
        );
        assert!(
            lemonade.iter().any(|entry| entry.contains("has only")),
            "the low-VRAM warning still fires for other engines: {lemonade:?}"
        );
        assert!(
            !lemonade
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "non-vLLM engines must not be told to pass a vLLM-only flag: {lemonade:?}"
        );
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

    fn vram(index: u32, used_mb: u64, total_mb: u64) -> GpuVramUsage {
        GpuVramUsage {
            index,
            used_mb,
            total_mb,
        }
    }

    #[test]
    fn auto_selection_prefers_lowest_index_idle_gpu() {
        // GPU 0 busy (only 5% free), GPU 1 idle, GPU 2 idle.
        let usage = [
            vram(0, 182_000, 192_000),
            vram(1, 1_000, 192_000),
            vram(2, 500, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(Some(3), None, &[], Some(&usage)),
            vec![1],
            "should skip the busy GPU 0 and pick the lowest idle GPU"
        );
    }

    #[test]
    fn auto_selection_skips_managed_and_busy_gpus_then_picks_most_free() {
        // GPU 0 pinned by a managed service; GPU 1 partly used; GPU 2 more free
        // but none is fully idle, so pass 2 (most free) applies.
        let usage = [
            vram(0, 10_000, 192_000),
            vram(1, 120_000, 192_000),
            vram(2, 60_000, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(Some(3), None, &[0], Some(&usage)),
            vec![2],
            "with no idle GPU, pick the non-busy GPU with the most free VRAM"
        );
    }

    #[test]
    fn auto_selection_pass_two_ranks_by_absolute_free_vram() {
        // Heterogeneous VRAM with no fully-idle GPU (so pass 2 applies):
        // GPU 0 is a small card with a high free *fraction* (75%) but little
        // absolute free memory; GPU 1 is large with a lower fraction (~48%)
        // but far more free memory. Auto-selection must prefer GPU 1.
        let usage = [vram(0, 6_000, 24_000), vram(1, 100_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[], Some(&usage)),
            vec![1],
            "pass 2 should rank by absolute free VRAM, not free percentage"
        );
    }

    #[test]
    fn auto_selection_falls_back_to_first_non_busy_without_vram() {
        assert_eq!(select_auto_gpu_index(Some(4), None, &[0, 1], None), vec![2]);
        // Unknown GPU count: no GPU-0 fallback — defer to the engine device probe.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], None),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_restricts_candidates_to_visible_set() {
        // GPUs 0 and 1 are masked out (only 2 and 3 visible). With no VRAM data
        // auto-select must pick the lowest *visible* ordinal, never a hidden one.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[], None),
            vec![2],
            "selection must stay inside the visibility mask"
        );
        // The lowest visible GPU (2) is busy, so fall through to the next visible.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[2], None),
            vec![3]
        );
    }

    #[test]
    fn auto_selection_all_visible_busy_falls_back_to_lowest_visible_not_zero() {
        // Every visible GPU is pinned; the fallback must be the lowest *visible*
        // ordinal (2), not a hardcoded 0 that the mask has hidden.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[2, 3]), &[2, 3], None),
            vec![2]
        );
    }

    #[test]
    fn auto_selection_returns_none_when_every_device_masked_out() {
        // An empty visible set means no candidate survives the mask: return no
        // selection rather than assuming device 0.
        assert_eq!(
            select_auto_gpu_index(Some(4), Some(&[]), &[], None),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_uses_vram_row_count_when_amd_smi_count_is_unknown() {
        // `detect_gpu_count()` is `None` while VRAM telemetry is present. With
        // *several* rows that is the amd-smi shape of the state -- `metric`
        // answered while `list` did not -- not the DRM sysfs fallback, which
        // withholds telemetry entirely once a second AMD card is found and so
        // can never produce more than one row (`read_drm_vram_usage`). Either
        // way auto-selection must derive the count from the rows and skip the
        // busy GPU 0 rather than returning no selection (which would land the
        // engine on GPU 0).
        let usage = [
            vram(0, 182_000, 192_000),
            vram(1, 1_000, 192_000),
            vram(2, 500, 192_000),
        ];
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&usage)),
            vec![1],
            "with no amd-smi count, rank the reported VRAM rows and pick the first idle GPU"
        );
        // The state the DRM sysfs fallback can actually reach: exactly one row.
        // It is still ranked (and still selected) rather than discarded for
        // being a single device.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&[vram(0, 1_000, 192_000)])),
            vec![0],
            "the single row the sysfs fallback can emit must still be selectable"
        );
        // Still nothing to go on when neither the count nor telemetry is present.
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&[])),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn auto_selection_ranks_the_actual_reported_row_indices() {
        // A visibility mask hides GPUs 0 and 1, so amd-smi metric reports only
        // the surviving devices with their absolute, non-contiguous ordinals
        // [2, 3] -- there is no row at index 0 or 1. Auto-selection must scan
        // those real indices, not a synthetic `0..count` range (which would
        // look up absent rows and fall through to a bogus GPU 0).
        let usage = [vram(2, 182_000, 192_000), vram(3, 1_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[], Some(&usage)),
            vec![3],
            "should skip the busy GPU 2 and pick the idle GPU 3 by its real ordinal"
        );
        // With GPU 3 also pinned by a managed service, pass 3 still returns a
        // real reported ordinal (GPU 2), never a fabricated index 0.
        assert_eq!(
            select_auto_gpu_index(None, None, &[3], Some(&usage)),
            vec![2],
            "the sole non-busy reported GPU must be selected by its real ordinal"
        );
    }

    #[test]
    fn auto_selection_all_reported_busy_falls_back_to_a_reported_ordinal() {
        // Same sparse-ordinal host as above ([2, 3] reported, nothing at 0 or 1)
        // but now BOTH reported GPUs are pinned by a running service, so every
        // pass falls through to the terminal "all busy, pick one anyway"
        // fallback. That fallback used to hand back a hardcoded `0` — safe only
        // while candidates were a dense `0..count`, and simply wrong once they
        // come from the reported rows: index 0 is a device nothing reported, yet
        // it would be exported verbatim as `HIP_VISIBLE_DEVICES`. It must return
        // the lowest ordinal that was actually reported instead.
        let usage = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[2, 3], Some(&usage)),
            vec![2],
            "the all-busy fallback must name a reported GPU, never a fabricated index 0"
        );
        // "Lowest reported" has to mean lowest by ordinal, not first in the
        // rows: `amd-smi` orders its output by enumeration, not by index, so
        // feed the same two rows in descending order. Varying the *busy* slice's
        // order instead would prove nothing — it is only ever read through
        // `.contains()`, so that assertion was a duplicate of the one above.
        let descending = [vram(3, 190_000, 192_000), vram(2, 182_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(None, None, &[2, 3], Some(&descending)),
            vec![2],
            "the fallback must take the lowest reported ordinal, not the first row"
        );
    }

    #[test]
    fn auto_selection_considers_a_detected_gpu_that_reported_no_vram_row() {
        // A short row set is not always a visibility mask. `parse_gpu_vram_usage`
        // drops any device entry missing its `used_vram`/`total_vram` pointers, so
        // here `list` counts two GPUs, both are visible, but only GPU 0 produced a
        // row — and GPU 0 is pinned by a managed service. Driving candidates from
        // the rows alone leaves `[0]`, the busy filter empties it, every pass
        // iterates nothing and the terminal fallback hands back GPU 0: `serve`
        // pinned to an already-occupied GPU, the exact failure this selection
        // exists to prevent. The untelemetried GPU 1 is confirmed by both the
        // count and the visible set, so it must be a candidate and must win.
        assert_eq!(
            select_auto_gpu_index(
                Some(2),
                Some(&[0, 1]),
                &[0],
                Some(&[vram(0, 182_000, 192_000)])
            ),
            vec![1],
            "a detected, visible GPU with no VRAM row must be preferred over a busy reported one"
        );
        // The same shape without telemetry for the busy device being conclusive:
        // GPU 1 is still the only non-busy ordinal either source confirms.
        assert_eq!(
            select_auto_gpu_index(
                Some(2),
                Some(&[0, 1]),
                &[0],
                Some(&[vram(0, 1_000, 192_000)])
            ),
            vec![1],
            "an idle-looking but service-pinned GPU 0 must still lose to the free GPU 1"
        );
        // The masking behaviour this rows-only construction was introduced for is
        // untouched: rows `[2, 3]` against `count == 2` put a row index at or above
        // the count, which is the proof the ordinal space is re-indexed, so the
        // union never happens and the rows stand alone. (The visible retain would
        // also have dropped `0`/`1` here — this pins the row-index rule itself, so
        // the masked case survives on a host where the mask is unknown too.)
        let masked = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), Some(&[2, 3]), &[2, 3], Some(&masked)),
            vec![2],
            "a masked host must not gain candidates 0/1 from the detected count"
        );
    }

    #[test]
    fn auto_selection_considers_an_untelemetried_gpu_when_the_visible_set_is_unknown() {
        // Same shape as the test above, but `visible` is `None`. That is not an
        // exotic host: `probe_usable_amd_gpu_indices` returns `None`
        // *unconditionally* off Linux, and `serve` threads that straight into
        // `visible`, so every Windows run takes this path — as does any Linux host
        // whose KFD topology and DRM cards are both unreadable. Gating the
        // `0..count` union on `visible.is_some()` therefore left the whole bug
        // standing on a supported platform.
        //
        // Nothing about the mask is needed to justify GPU 1 here: the amd-smi
        // `list` count asserts two devices and the rows name only ordinals below
        // that count, so the rows are a subset of a dense range, not a re-indexed
        // one. GPU 0 is pinned by a managed service, so the untelemetried GPU 1 is
        // the only free ordinal and must win over the busy reported fallback.
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[0], Some(&[vram(0, 182_000, 192_000)])),
            vec![1],
            "an unprobeable host must still consider a counted GPU that reported no VRAM row"
        );
        // A row index at or above the count is the only evidence available that the
        // ordinal space is *not* a dense `0..count`, so there the rows stay
        // authoritative even with no mask to retain against — otherwise an
        // unprobeable masked host would have `[0, 1]` invented for it.
        let masked = [vram(2, 182_000, 192_000), vram(3, 190_000, 192_000)];
        assert_eq!(
            select_auto_gpu_index(Some(2), None, &[2, 3], Some(&masked)),
            vec![2],
            "re-indexed rows must not gain unconfirmed candidates 0/1 from the detected count"
        );
        // The count must come from the `list` enumeration to confirm anything.
        // With `detected` absent it is `rows.len()`, i.e. the rows restating their
        // own size, and `parse_gpu_vram_usage` can repeat an ordinal: it falls back
        // to the array position only when an entry has no `gpu` field, so a payload
        // naming `"gpu": 0` twice yields two index-0 rows. That makes `rows.len()`
        // 2 with every index below it, and dropping the `detected` half of the
        // condition would invent a GPU 1 nothing ever enumerated.
        assert_eq!(
            select_auto_gpu_index(
                None,
                None,
                &[0],
                Some(&[vram(0, 182_000, 192_000), vram(0, 182_000, 192_000)])
            ),
            vec![0],
            "a row-derived count must not synthesise an ordinal no source reported"
        );
    }

    #[test]
    fn validate_pinned_gpu_index_rejects_out_of_range() {
        // Index equal to or beyond the detected count is rejected.
        let error = validate_pinned_gpu_index(4, Some(4), None, false)
            .expect_err("index 4 is out of range");
        assert!(error.to_string().contains("out of range"));
        assert!(validate_pinned_gpu_index(9, Some(2), None, false).is_err());
    }

    #[test]
    fn validate_pinned_gpu_index_accepts_in_range_or_unknown_count() {
        // In-range index pins exactly that ordinal.
        assert_eq!(
            validate_pinned_gpu_index(0, Some(1), None, false).unwrap(),
            vec![0]
        );
        assert_eq!(
            validate_pinned_gpu_index(3, Some(4), None, false).unwrap(),
            vec![3]
        );
        // Unknown count (amd-smi unavailable) is allowed through unvalidated.
        assert_eq!(
            validate_pinned_gpu_index(7, None, None, false).unwrap(),
            vec![7]
        );
    }

    #[test]
    fn effective_gpu_count_prefers_amd_smi_then_falls_back_to_vram_rows() {
        let usage = [vram(0, 1_000, 192_000), vram(1, 1_000, 192_000)];
        // amd-smi's count wins when it is known, even over a differing row count.
        assert_eq!(effective_gpu_count(Some(4), Some(&usage)), Some(4));
        // amd-smi unavailable (`None`): the sysfs VRAM fallback row count stands in.
        assert_eq!(effective_gpu_count(None, Some(&usage)), Some(2));
        // The second, previously untested fallback trigger: amd-smi ran and
        // reported zero devices. `Some(0)` must fall through to the VRAM rows
        // just as `None` does, or a host amd-smi cannot enumerate but sysfs can
        // would have nothing for `--gpu auto` to rank. Dropping the
        // `filter(|&count| count > 0)` on `detected` turns this line red.
        assert_eq!(effective_gpu_count(Some(0), Some(&usage)), Some(2));
        // Neither source available.
        assert_eq!(effective_gpu_count(None, None), None);
        assert_eq!(effective_gpu_count(None, Some(&[])), None);
        // Both present but both empty: still nothing to rank.
        assert_eq!(effective_gpu_count(Some(0), Some(&[])), None);
        assert_eq!(effective_gpu_count(Some(0), None), None);
    }

    #[test]
    fn validate_pinned_gpu_index_rejects_masked_out_device() {
        // The visible set is authoritative: an index outside it is rejected early
        // with the usable set, rather than deferring to a late engine error. A
        // mask is set here, so the message names the visibility variables.
        let error = validate_pinned_gpu_index(0, Some(4), Some(&[2, 3]), true)
            .expect_err("masked-out device must be rejected");
        let message = error.to_string();
        assert!(message.contains("not available under the active visibility mask"));
        assert!(message.contains("[2, 3]"));
        // A visible index still pins exactly that ordinal.
        assert_eq!(
            validate_pinned_gpu_index(2, Some(4), Some(&[2, 3]), true).unwrap(),
            vec![2]
        );
        // An empty visible set is authoritative "no GPU usable" (every device
        // masked out), not "could not enumerate" — the latter is `None`, per the
        // `usable_amd_gpu_indices` contract. So it rejects rather than falling
        // through to the count check; the detected count must not override it.
        let masked_out = validate_pinned_gpu_index(1, Some(4), Some(&[]), true)
            .expect_err("an all-masked host must reject an explicit --gpu index");
        assert!(
            masked_out
                .to_string()
                .contains("no usable AMD GPU is available")
        );
    }

    #[test]
    fn validate_pinned_gpu_index_absent_index_without_mask_reads_as_not_present() {
        // No visibility mask is set, yet amd-smi counts more devices than the
        // KFD/DRM probe finds, so the authoritative visible set (0..present) is
        // shorter than `detected`. An index in that gap is genuinely absent from
        // the host — the rejection must say so and must NOT blame HIP/ROCR
        // variables the user never set (that message sends them debugging a mask
        // that does not exist).
        let error = validate_pinned_gpu_index(2, Some(4), Some(&[0, 1]), false)
            .expect_err("an index past the visible set must be refused");
        let message = error.to_string();
        assert!(
            message.contains("not present on this host"),
            "unmasked rejection should read as not-present, got: {message}"
        );
        assert!(
            !message.contains("visibility mask"),
            "must not blame a mask when none is set, got: {message}"
        );
        assert!(message.contains("[0, 1]"));

        // The all-empty visible set with no mask is a not-present message too,
        // never a spurious mask advisory.
        let empty = validate_pinned_gpu_index(0, Some(2), Some(&[]), false)
            .expect_err("no present GPU must refuse an explicit --gpu index");
        assert!(empty.to_string().contains("present on this host"));
        assert!(!empty.to_string().contains("visibility mask"));
    }

    #[test]
    fn validate_pinned_gpu_index_prefers_visible_set_over_detected_count() {
        // detected and visible come from different probes. When both are known the
        // visible set wins, so an index inside the detected count but outside the
        // HIP-visible set is still rejected (and vice versa), instead of the two
        // being compared as one ordinal space.
        assert_eq!(
            validate_pinned_gpu_index(1, Some(4), Some(&[0, 1]), true).unwrap(),
            vec![1]
        );
        assert!(
            validate_pinned_gpu_index(2, Some(4), Some(&[0, 1]), true).is_err(),
            "index within the detected count but outside the visible set must reject"
        );
    }

    #[test]
    fn parse_gpu_vram_usage_reads_gpu_data_envelope() {
        let value = json!({
            "gpu_data": [
                {"gpu": 0, "mem_usage": {"used_vram": {"value": 1000}, "total_vram": {"value": 192_000}}},
                {"gpu": 1, "mem_usage": {"used_vram": {"value": 50000}, "total_vram": {"value": 192_000}}}
            ]
        });
        let rows = parse_gpu_vram_usage(&value);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].index, 0);
        assert_eq!(rows[0].used_mb, 1000);
        assert_eq!(rows[1].index, 1);
        assert!((rows[1].free_fraction().unwrap() - (142_000.0 / 192_000.0)).abs() < 1e-9);
    }

    #[test]
    fn read_drm_vram_usage_reads_a_single_amdgpu_card_and_assigns_ordinal_zero() {
        // Plant a minimal `/sys/class/drm`-shaped tree: one AMD card, a connector
        // sub-node that must be skipped, and a non-AMD card that must be ignored.
        let root = std::env::temp_dir().join(format!("rocm-drm-vram-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let plant_amd = |card: &str, total: u64, used: u64| {
            let device = root.join(card).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("vendor"), "0x1002\n").unwrap();
            fs::write(device.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
            fs::write(device.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
        };
        // 192 GiB total, mostly free (values in bytes).
        plant_amd("card0", 206_158_430_208, 1_073_741_824);
        // A connector sub-node under card0 — must be skipped, not parsed as a card.
        fs::create_dir_all(root.join("card0-DP-1")).unwrap();
        // A non-AMD primary card — different vendor, must be ignored.
        let intel = root.join("card1").join("device");
        fs::create_dir_all(&intel).unwrap();
        fs::write(intel.join("vendor"), "0x8086\n").unwrap();
        fs::write(intel.join("mem_info_vram_total"), "1000000\n").unwrap();

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert_eq!(rows.len(), 1, "only the one AMD card counts: {rows:?}");
        assert_eq!(rows[0].index, 0);
        // 206_158_430_208 bytes / 1 MiB == 196_608 MiB.
        assert_eq!(rows[0].total_mb, 196_608);
        assert!(rows[0].free_fraction().unwrap() > AUTO_FREE_VRAM_FRACTION);
    }

    #[test]
    fn read_drm_vram_usage_withholds_telemetry_when_multiple_amd_cards_are_present() {
        // Ascending `card<N>` order is not guaranteed to match HIP's compute
        // ordinal once more than one AMD card exists (e.g. an APU alongside a
        // dGPU), so the fallback must not hand out ordinals it cannot vouch for.
        let root = std::env::temp_dir().join(format!("rocm-drm-vram-multi-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let plant_amd = |card: &str, total: u64, used: u64| {
            let device = root.join(card).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("vendor"), "0x1002\n").unwrap();
            fs::write(device.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
            fs::write(device.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
        };
        plant_amd("card0", 206_158_430_208, 1_073_741_824);
        plant_amd("card1", 206_158_430_208, 189_284_651_008);

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "multi-card sysfs telemetry must be withheld: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_withholds_telemetry_when_a_second_amd_card_has_no_counter() {
        // The guard must count AMD cards *found*, not surviving rows: a genuine
        // two-AMD-card host where the second card's `used` counter is unreadable
        // would otherwise slip past a `rows.len() > 1` check and mislabel the
        // survivor ordinal 0 — the exact APU+dGPU misattribution the guard exists
        // to prevent.
        let root =
            std::env::temp_dir().join(format!("rocm-drm-vram-partial-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        // card0: fully readable.
        let card0 = root.join("card0").join("device");
        fs::create_dir_all(&card0).unwrap();
        fs::write(card0.join("vendor"), "0x1002\n").unwrap();
        fs::write(card0.join("mem_info_vram_total"), "206158430208\n").unwrap();
        fs::write(card0.join("mem_info_vram_used"), "1073741824\n").unwrap();
        // card1: AMD, but its `used` counter cannot be read.
        let card1 = root.join("card1").join("device");
        fs::create_dir_all(&card1).unwrap();
        fs::write(card1.join("vendor"), "0x1002\n").unwrap();
        fs::write(card1.join("mem_info_vram_total"), "206158430208\n").unwrap();

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "a second AMD card must trip the guard even without readable counters: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_skips_a_card_with_an_unreadable_used_counter() {
        // A card whose `mem_info_vram_used` cannot be read must not be treated as
        // 0 bytes used (100% free) — that would make a broken counter look like
        // the ideal `--gpu auto` pick. It should be skipped entirely instead.
        let root =
            std::env::temp_dir().join(format!("rocm-drm-vram-unreadable-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let device = root.join("card0").join("device");
        fs::create_dir_all(&device).unwrap();
        fs::write(device.join("vendor"), "0x1002\n").unwrap();
        fs::write(device.join("mem_info_vram_total"), "206158430208\n").unwrap();
        // No `mem_info_vram_used` file written at all.

        let rows = read_drm_vram_usage(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            rows.is_empty(),
            "a card with no readable used counter must be skipped: {rows:?}"
        );
    }

    #[test]
    fn read_drm_vram_usage_is_empty_without_a_drm_tree() {
        let missing = std::env::temp_dir().join("rocm-drm-vram-absent-98765");
        let _ = fs::remove_dir_all(&missing);
        assert!(read_drm_vram_usage(&missing).is_empty());
    }

    #[test]
    fn gpu_low_memory_warning_flags_busy_selected_gpu() {
        let usage = [vram(0, 182_000, 192_000), vram(1, 1_000, 192_000)];
        let warning = gpu_low_memory_warning(&[0], Some(&usage)).expect("warning for busy GPU 0");
        assert!(warning.contains("GPU 0"));
        assert!(warning.contains("free"));
        assert!(gpu_low_memory_warning(&[1], Some(&usage)).is_none());
        assert!(gpu_low_memory_warning(&[0], None).is_none());
    }

    fn host_gpu(gfx_target: &str) -> rocm_core::HostGpuSummary {
        rocm_core::HostGpuSummary {
            gfx_target: Some(gfx_target.to_owned()),
            ..rocm_core::HostGpuSummary::default()
        }
    }

    #[test]
    fn unified_memory_apu_suppresses_the_vram_capacity_warning() {
        // Strix Halo as reported by amd-smi: a 4 GiB BIOS carveout with 1.6 GiB
        // free (40%, under the 90% bar) on a machine whose engine actually
        // serves out of ~128 GiB of shared system RAM. The old reading —
        // "only 1.6 GiB of 4.0 GiB free" — describes a pool the allocator does
        // not use, so it must not reach the user.
        let carveout = [vram(0, 2_458, 4_096)];
        assert!(
            gpu_low_memory_warning(&[0], Some(&carveout)).is_some(),
            "the underlying threshold still trips; only the serve-plan wrapper withholds it"
        );
        assert_eq!(
            serve_gpu_low_memory_warning(&[0], Some(&carveout), Some(&host_gpu("gfx1151"))),
            None
        );
    }

    #[test]
    fn discrete_gpu_still_warns_when_genuinely_busy() {
        let busy = [vram(0, 182_000, 192_000)];
        let warning = serve_gpu_low_memory_warning(&[0], Some(&busy), Some(&host_gpu("gfx1100")))
            .expect("a discrete card that is 95% full still warrants a warning");
        assert!(warning.contains("GPU 0"));
        assert!(warning.contains("`--gpu <index|auto>`"));

        // Baseline: the same discrete card, mostly free, stays quiet.
        let idle = [vram(0, 1_000, 192_000)];
        assert_eq!(
            serve_gpu_low_memory_warning(&[0], Some(&idle), Some(&host_gpu("gfx1100"))),
            None
        );
    }

    #[test]
    fn apu_verdict_does_not_silence_a_second_gpu() {
        // `gfx_is_apu_family` classifies a part, while host detection reports one
        // target for the whole machine. On an APU+dGPU laptop that target cannot
        // be attributed to the selected ordinal, so the warning must survive
        // rather than be suppressed on the discrete card's behalf.
        let hybrid = [vram(0, 2_458, 4_096), vram(1, 182_000, 192_000)];
        assert!(
            serve_gpu_low_memory_warning(&[1], Some(&hybrid), Some(&host_gpu("gfx1151"))).is_some()
        );
        assert!(vram_capacity_is_meaningful(Some("gfx1151"), 2));
        // An unknown target is never treated as unified memory.
        assert!(vram_capacity_is_meaningful(None, 1));
    }

    /// The `ROCM_E2E_FORCE_LOW_VRAM` hook must synthesize a reading that actually
    /// trips the serve-plan low-VRAM warning on a non-APU (vLLM-lane) host, and
    /// stay inert when the var is unset. This is what the `@requires-gpu`
    /// `serve-vllm-low-vram-oom-guidance` scenario relies on to fire the note
    /// deterministically on cards that are really free.
    #[cfg(feature = "e2e-test-hooks")]
    #[test]
    fn forced_low_vram_hook_trips_the_serve_plan_warning() {
        let mut env = ScopedTestEnv::new();

        // Unset: the hook contributes nothing and real telemetry is consulted.
        env.clear("ROCM_E2E_FORCE_LOW_VRAM");
        assert!(
            simulated_low_vram_usage().is_none(),
            "no override without the env var"
        );

        // Set to ordinal 0: a single near-full device that the serve-plan wrapper
        // reports on a discrete (non-APU) host but that stays honest there.
        env.set("ROCM_E2E_FORCE_LOW_VRAM", "0");
        let forced = simulated_low_vram_usage().expect("override present when the var is set");
        assert_eq!(
            forced.len(),
            1,
            "single synthetic GPU keeps the APU guard honest"
        );
        assert_eq!(forced[0].index, 0);
        assert!(
            forced[0]
                .free_fraction()
                .is_some_and(|f| f < AUTO_FREE_VRAM_FRACTION),
            "the synthetic reading must be below the free-VRAM bar so the warning fires"
        );
        let warning = serve_gpu_low_memory_warning(&[0], Some(&forced), Some(&host_gpu("gfx1100")))
            .expect("a near-full discrete card warrants the serve-plan warning");
        assert!(warning.contains("GPU 0"));
    }

    #[test]
    fn resolve_engine_selection_uses_default_runtime_after_engine_prefs() {
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_default_runtime_id")
        );

        config.active_runtime_key = Some("release-pip-gfx120x-all-7-13-0".to_owned());
        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.engine_config_mut("vllm").preferred_runtime_id =
            Some("therock-nightly:gfx120X-all".to_owned());
        let selection = resolve_engine_selection(&config, "vllm", None, None);
        assert_eq!(
            selection.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            selection.source.as_deref(),
            Some("config_active_runtime_key")
        );

        config.active_runtime_key = None;
        let selection = resolve_engine_selection(&config, "vllm", None, None);
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
            resolve_engine_selection(&RocmCliConfig::default(), "vllm", None, None),
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
            resolve_engine_selection(&RocmCliConfig::default(), "vllm", None, None),
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

    /// The two runtimes every activation/reconciliation test switches between.
    pub(crate) const OLD_RUNTIME_KEY: &str = "release-pip-gfx120x-all-7-12-0";
    pub(crate) const NEW_RUNTIME_KEY: &str = "release-pip-gfx120x-all-7-13-0";

    #[test]
    fn runtime_activation_restart_services_requires_yes() {
        // The caller passes the runtime the user actually typed, so the hint is
        // copy-pasteable. A literal `<runtime_key>` here would assert a string
        // the CLI never prints.
        let error = runtime_services::ensure_service_restart_approved(
            true,
            false,
            &format!("activate {NEW_RUNTIME_KEY}"),
        )
        .expect_err("restarting local servers must not happen without --yes")
        .to_string();

        assert!(
            error.contains("requires --yes"),
            "the refusal must name the missing approval:\n{error}"
        );
        assert!(
            error.contains(&format!(
                "rocm runtimes activate {NEW_RUNTIME_KEY} --restart-services --yes"
            )),
            "this path never prompts, so the refusal must spell out the command \
             that does work, with the runtime the user asked for:\n{error}"
        );

        assert!(
            runtime_services::ensure_service_restart_approved(
                true,
                true,
                &format!("activate {NEW_RUNTIME_KEY}")
            )
            .is_ok(),
            "--yes is the approval this path accepts"
        );
        assert!(
            runtime_services::ensure_service_restart_approved(false, false, "rollback").is_ok(),
            "a rollback that restarts nothing needs no approval"
        );
        // The hint is built per command, so the one error it can make is naming
        // the other one — and `rocm runtimes activate ... --restart-services
        // --yes` told to a user who just ran `rollback` would switch the runtime
        // rather than restore it.
        let rollback_error =
            runtime_services::ensure_service_restart_approved(true, false, "rollback")
                .expect_err("rollback restarts local servers too, so it needs --yes as well")
                .to_string();
        assert!(
            rollback_error.contains("Try: rocm runtimes rollback --restart-services --yes"),
            "the rollback refusal must name rollback, not activate:\n{rollback_error}"
        );
    }

    #[test]
    fn pinning_a_service_record_to_a_runtime_drops_its_env_pin() -> Result<()> {
        let (root, paths) = test_paths("service-runtime-pin");
        paths.ensure()?;
        plant_service_record_on_runtime(
            &paths,
            "svc-pinned",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            Some("custom-env"),
        )?;

        let previous =
            runtime_services::pin_service_record_to_runtime(&paths, "svc-pinned", NEW_RUNTIME_KEY)?;
        let repinned = load_managed_service(&paths, "svc-pinned")?;
        assert_eq!(
            previous,
            runtime_services::ServiceRuntimePin {
                runtime_id: Some(OLD_RUNTIME_KEY.to_owned()),
                env_id: Some("custom-env".to_owned()),
            }
        );

        runtime_services::restore_service_record_pin(&paths, "svc-pinned", previous)?;
        let restored = load_managed_service(&paths, "svc-pinned")?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(repinned.runtime_id.as_deref(), Some(NEW_RUNTIME_KEY));
        assert_eq!(
            repinned.env_id, None,
            "`builtin_engine_serve_http_args` leaves `--runtime-id` out of the \
             child's argv whenever `env_id` is set, so a record that kept its \
             env pin would be restarted onto a runtime nobody chose while \
             claiming the new key"
        );
        assert_eq!(restored.runtime_id.as_deref(), Some(OLD_RUNTIME_KEY));
        assert_eq!(restored.env_id.as_deref(), Some("custom-env"));
        Ok(())
    }

    /// The value following `flag` in an argv, or `None` when the flag is absent.
    fn argv_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        let index = args.iter().position(|arg| arg == flag)?;
        args.get(index + 1).map(String::as_str)
    }

    /// The runtime pin a `--restart-services` restart moves a server with, at
    /// the only place it is observable without launching an engine: the argv the
    /// restarted child is given.
    ///
    /// The sibling test above asserts the record's fields; that is not the same
    /// claim. `builtin_engine_serve_http_args` omits `--runtime-id` whenever
    /// `env_id` is set, so what the record says and what the child is told can
    /// differ — and it is the child's argv that decides which runtime comes back
    /// up.
    ///
    /// The setup is what makes this non-vacuous, and it is the state a live
    /// restart really sees rather than an invented one. The service being moved is
    /// still running when the restart loads its record, so its engine state file
    /// still describes it, and `refresh_from_engine_state` re-adopts that
    /// process's `runtime_id` and `env_id` over the pin
    /// `pin_service_record_to_runtime` had just written. The `None` load at the
    /// end is that load: the argv carries no `--runtime-id` at all because the
    /// revived `env_id` suppresses it.
    ///
    /// The restart is the production `restart_service_onto_runtime_with`, handed
    /// a stand-in that does what `restart_internal_managed_service` does up to
    /// the spawn — `load_service_for_restart`, then `restart_child_argv` — and
    /// then fails. Two things are guarded:
    ///
    /// - that `load_service_for_restart` applies the pin it is given: removing
    ///   the `apply_restart_runtime_pin` call inside it leaves the revived
    ///   `env_id` in place, and the `--runtime-id` and `--env-id` assertions
    ///   fail;
    /// - that `restart_service_onto_runtime_with` hands the key to the restart:
    ///   passing `None` instead of `Some(runtime_key)` fails the assertion on the
    ///   pin the stand-in was given, and with it the `--runtime-id` assertion.
    ///
    /// The failing stand-in also drives the restore path, so the record on disk
    /// is read back to be on the runtime it ran on.
    ///
    /// What it does not reach is that a real child launched with this argv loads
    /// the runtime, which needs a GPU: `runtime-lifecycle-12` observes that.
    #[test]
    fn pinning_the_record_is_what_puts_the_new_runtime_in_the_restart_argv() -> Result<()> {
        let (root, paths) = test_paths("service-runtime-pin-argv");
        paths.ensure()?;
        let planted = plant_service_record_on_runtime(
            &paths,
            "svc-argv",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            Some("custom-env"),
        )?;
        // The outgoing process's own state, in the shape vLLM writes it: the
        // exact key it was launched with, the family form it resolved that to,
        // and an `env_id` it records on every launch whether or not the user
        // asked for one.
        fs::write(
            &planted.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "starting",
                "requested_runtime_id": OLD_RUNTIME_KEY,
                "runtime_id": "therock-release:gfx120X-all",
                "env_id": "custom-env",
            }))?,
        )?;
        let handed_pin = RefCell::new(None::<String>);
        let observed_argv = RefCell::new(None::<Vec<String>>);

        let error = runtime_services::restart_service_onto_runtime_with(
            &paths,
            "svc-argv",
            NEW_RUNTIME_KEY,
            |restart_paths, service_id, pin| -> Result<ManagedServiceRecord> {
                *handed_pin.borrow_mut() = pin.map(ToOwned::to_owned);
                let record = load_service_for_restart(restart_paths, service_id, pin)?;
                *observed_argv.borrow_mut() = Some(restart_child_argv(&record)?);
                bail!("stop here: the argv is what is under test");
            },
        )
        .expect_err("the stand-in restart always fails");
        let handed_pin = handed_pin.into_inner();
        let observed_argv = observed_argv.into_inner();
        // Read the file itself: a load would refresh the record from the engine
        // state again and hide what the restore wrote.
        let on_disk: ManagedServiceRecord =
            serde_json::from_slice(&fs::read(paths.service_manifest_path("svc-argv"))?)?;
        let unpinned = load_service_for_restart(&paths, "svc-argv", None)?;
        let unpinned_argv = restart_child_argv(&unpinned)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            format!("{error:#}").contains("stop here"),
            "the stand-in's own failure must reach the caller:\n{error:#}"
        );
        assert_eq!(
            handed_pin.as_deref(),
            Some(NEW_RUNTIME_KEY),
            "the key has to be passed to the restart as well as written to the \
             record: the restart's own load refreshes the record from the \
             outgoing process's state and undoes the write"
        );
        let argv = observed_argv.context("the stand-in restart never built an argv")?;
        assert_eq!(
            argv_flag_value(&argv, "--runtime-id"),
            Some(NEW_RUNTIME_KEY),
            "re-applying the pin after the refresh is what makes the restarted \
             child load the runtime that was just activated: {argv:?}"
        );
        assert_eq!(
            argv_flag_value(&argv, "--env-id"),
            None,
            "clearing `env_id` is half the pin, not tidying: leaving it set \
             would suppress the `--runtime-id` asserted above: {argv:?}"
        );
        assert_eq!(
            argv_flag_value(&unpinned_argv, "--runtime-id"),
            None,
            "without the pin the same record is restarted with no `--runtime-id`, \
             because the outgoing process's `env_id` is restored over it; this is \
             what makes the assertions above mean something: {unpinned_argv:?}"
        );
        assert_eq!(
            argv_flag_value(&unpinned_argv, "--env-id"),
            Some("custom-env"),
            "which is the engine's own env coming back, not the user's: {unpinned_argv:?}"
        );
        assert_eq!(
            on_disk.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the failed restart must put the record back on the runtime it ran on"
        );
        assert_eq!(on_disk.env_id.as_deref(), Some("custom-env"));
        Ok(())
    }

    #[test]
    fn a_refused_service_restart_leaves_the_record_on_the_runtime_it_ran_on() -> Result<()> {
        let (root, paths) = test_paths("service-restart-refused");
        paths.ensure()?;
        let mut record = plant_service_record_on_runtime(
            &paths,
            "svc-public",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            Some("custom-env"),
        )?;
        // A public bind with no endpoint key is refused by
        // `restart_internal_managed_service` before it stops anything, which
        // makes the failure branch reachable without spawning a process — and,
        // just as importantly, without signalling the pid the planted record
        // carries, which is this test process.
        record.host = "0.0.0.0".to_owned();
        record.write()?;
        endpoint_keys::clear_endpoint_api_key(&paths, "svc-public");

        let error =
            runtime_services::restart_service_onto_runtime(&paths, "svc-public", NEW_RUNTIME_KEY)
                .expect_err("a refused restart must fail the move");
        let after = load_managed_service(&paths, "svc-public")?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            format!("{error:#}").contains("endpoint API key"),
            "the restart's own refusal must survive to the caller:\n{error:#}"
        );
        assert_eq!(
            after.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "a service that did not move must not be described by a runtime it \
             never loaded"
        );
        assert_eq!(after.env_id.as_deref(), Some("custom-env"));
        Ok(())
    }

    #[test]
    fn a_restart_that_fails_after_the_stop_leaves_the_server_stopped_and_repinned() -> Result<()> {
        let (root, paths) = test_paths("service-restart-fails-after-stop");
        paths.ensure()?;
        let mut record = plant_service_record_on_runtime(
            &paths,
            "svc-cpu-policy",
            "vllm",
            "starting",
            // Skipped by `terminate_recorded_service_pids`, which never
            // signals this process — so the stop runs for real without the
            // test killing itself.
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;
        // A device policy the managed serve path refuses. It is parsed AFTER
        // `restart_internal_managed_service` has already stopped the service,
        // which is the failure the report's "stopped by the attempt" claim and
        // the pin restore are both written for — and which the pre-stop
        // refusal test deliberately does not reach.
        record.device_policy = Some("cpu_only".to_owned());
        record.write()?;

        let error = runtime_services::restart_service_onto_runtime(
            &paths,
            "svc-cpu-policy",
            NEW_RUNTIME_KEY,
        )
        .expect_err("a refused device policy must fail the move");
        let after = load_managed_service(&paths, "svc-cpu-policy")?;
        let _ = fs::remove_dir_all(&root);

        let error = format!("{error:#}");
        assert!(
            error.contains("CPU mode is not a fallback path"),
            "moving a server onto a new runtime goes through the same managed \
             serve path as a fresh launch, so it must refuse CPU execution \
             exactly as that path does:\n{error}"
        );
        assert_eq!(
            after.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the restart got past the stop but never came back up, so the \
             record must not describe a runtime the server never loaded"
        );
        assert_eq!(
            after.status, "stopped",
            "the report and the docs both tell the user a failed restart left \
             the server down; if the stop had not happened that sends them \
             away from a server that is still serving"
        );
        Ok(())
    }

    #[test]
    fn a_failed_restart_is_counted_by_whether_the_server_is_still_up() -> Result<()> {
        let (root, paths) = test_paths("service-restart-failure-counting");
        paths.ensure()?;
        // Refused BEFORE the stop, so this server is still serving from the
        // runtime the activation switched away from when the attempt returns.
        let mut still_up = plant_service_record_on_runtime(
            &paths,
            "svc-public",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;
        still_up.host = "0.0.0.0".to_owned();
        still_up.write()?;
        endpoint_keys::clear_endpoint_api_key(&paths, "svc-public");
        // Refused AFTER the stop, so this one really is down.
        let mut taken_down = plant_service_record_on_runtime(
            &paths,
            "svc-cpu-policy",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;
        taken_down.device_policy = Some("cpu_only".to_owned());
        taken_down.write()?;
        let stale_entry = |service_id: &str| runtime_services::RuntimeServiceEntry {
            service_id: service_id.to_owned(),
            engine: "vllm".to_owned(),
            recorded_runtime: Some(OLD_RUNTIME_KEY.to_owned()),
        };
        let mut services = runtime_services::RuntimeServiceReconciliation {
            stale: vec![stale_entry("svc-public"), stale_entry("svc-cpu-policy")],
            ..runtime_services::RuntimeServiceReconciliation::default()
        };

        runtime_services::restart_stale_runtime_services(&paths, NEW_RUNTIME_KEY, &mut services);
        let rendered =
            runtime_services::render_runtime_service_reconciliation(&services, NEW_RUNTIME_KEY);
        let exit_error = format!(
            "{:#}",
            runtime_services::bail_on_failed_service_restarts(&services)
                .expect_err("a failed restart must make the command exit non-zero")
        );
        let _ = fs::remove_dir_all(&root);

        assert_eq!(services.failed.len(), 2, "{services:?}");
        assert!(
            services.restarted.is_empty(),
            "a server that never came back up on the new runtime must not be \
             counted as restarted; `services_restarted` is what the report \
             offers as the good news: {services:?}"
        );
        assert_eq!(
            services.stale,
            vec![stale_entry("svc-public")],
            "a restart refused before the stop leaves the server serving from \
             the runtime it recorded, so dropping it under-reports a live \
             server on the previous runtime exactly the way counting a moved \
             one over-reported it; the server the attempt took down is not \
             serving from anything and must not be listed: {services:?}"
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 1")
                && rendered.contains(&format!(
                    "- svc-public engine=vllm recorded_runtime={OLD_RUNTIME_KEY}"
                )),
            "the count and the line beneath it both describe what is actually \
             still serving:\n{rendered}"
        );
        assert!(
            rendered.contains("services_restart_failed: 2"),
            "every failure is reported, whichever side of the stop it fell \
             on:\n{rendered}"
        );
        assert!(
            !rendered.contains("--restart-services --yes` to move them"),
            "advising a flag the user just passed is advice that cannot help:\n{rendered}"
        );
        assert!(
            exit_error.contains("stopped by the attempt and are no longer serving: svc-cpu-policy"),
            "the server the attempt took down has to be named as down:\n{exit_error}"
        );
        assert!(
            exit_error.contains("still serving from the runtime they recorded")
                && exit_error.contains("svc-public"),
            "telling this user the server is down sends them away from a live \
             server on the runtime they just switched off:\n{exit_error}"
        );
        Ok(())
    }

    #[test]
    fn a_failed_restart_with_nothing_left_serving_is_reported_as_stopped() {
        // No entry survives in `stale`, i.e. the failure left no server read
        // back as live. That input — not a failed restart in general — is what
        // makes "stopped by the attempt" the right thing to say here.
        let services = runtime_services::RuntimeServiceReconciliation {
            failed: vec![runtime_services::FailedServiceRestart {
                service_id: "svc-a".to_owned(),
                error: "boom".to_owned(),
            }],
            ..runtime_services::RuntimeServiceReconciliation::default()
        };

        let error = runtime_services::bail_on_failed_service_restarts(&services)
            .expect_err("a failed restart must make the command exit non-zero")
            .to_string();

        assert!(error.contains("svc-a"), "{error}");
        assert!(
            error.contains("stopped by the attempt"),
            "nothing was read back as still serving, so this failure is one the \
             attempt left down — a report that says it keeps serving sends the \
             user away from an outage:\n{error}"
        );
        assert!(
            !error.contains("still serving"),
            "the still-serving clause describes servers read back as live; \
             naming one when none was read back would count the same failure \
             twice and point the user at a server that is not there:\n{error}"
        );
        assert!(
            error.contains("rocm services restart svc-a --yes")
                || error.contains("rocm services restart <id> --yes"),
            "the error must name the command that brings it back:\n{error}"
        );
    }

    // A respawn is "serving" only when the restart saw an inference request
    // succeed, which is the one way it leaves the record at `ready`. The other
    // statuses it can return — the endpoint lists the model, or nothing answered
    // within the wait — are a server that was moved and not yet confirmed, and
    // neither is rolled back: the record keeps naming the runtime it was moved to.
    // Mutation: matching `running` or `starting` as well as `ready` in
    // `restart_service_onto_runtime_with` makes the outcomes list differ.
    #[test]
    fn a_restart_counts_as_serving_only_when_the_respawned_server_is_ready() -> Result<()> {
        let (root, paths) = test_paths("service-restart-outcome");
        paths.ensure()?;
        let mut outcomes = Vec::new();
        let mut recorded_runtimes = Vec::new();
        for (service_id, status) in [
            ("svc-ready", "ready"),
            ("svc-running", "running"),
            ("svc-starting", "starting"),
        ] {
            plant_service_record_on_runtime(
                &paths,
                service_id,
                "vllm",
                "starting",
                std::process::id(),
                Some(OLD_RUNTIME_KEY),
                None,
            )?;
            outcomes.push(runtime_services::restart_service_onto_runtime_with(
                &paths,
                service_id,
                NEW_RUNTIME_KEY,
                |restart_paths, id, _pin| {
                    let mut record = load_managed_service(restart_paths, id)?;
                    status.clone_into(&mut record.status);
                    Ok(record)
                },
            )?);
            recorded_runtimes.push(load_managed_service(&paths, service_id)?.runtime_id);
        }
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            outcomes,
            vec![
                runtime_services::ServiceRestartOutcome::Serving,
                runtime_services::ServiceRestartOutcome::NotConfirmed {
                    status: "running".to_owned()
                },
                runtime_services::ServiceRestartOutcome::NotConfirmed {
                    status: "starting".to_owned()
                },
            ]
        );
        assert!(
            recorded_runtimes
                .iter()
                .all(|runtime| runtime.as_deref() == Some(NEW_RUNTIME_KEY)),
            "a respawned server stays recorded on the runtime it was moved to, \
             confirmed or not: {recorded_runtimes:?}"
        );
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
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
        let name = rocm_engine_protocol::platform_engine_plugin_binary_name("vllm");
        let primary_path = primary_dir.join(&name);
        fs::write(&primary_path, "primary")?;
        fs::write(compatibility_dir.join(&name), "compatibility")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(primary_path));
        Ok(())
    }

    #[test]
    fn install_sdk_survives_an_engine_install_failure() -> Result<()> {
        let (_root, paths) = test_paths("install-sdk-engine-failure");

        // The complement, and the reason the catch exists at all: the SDK is installed
        // and usable, only a separately retryable step is missing. Failing here would
        // throw away a multi-gigabyte install over an unreachable engine index.
        let (result, _) = finish_sdk_install_with(
            &paths,
            Err(anyhow::anyhow!(
                "failed to reach the engine index: dns error"
            )),
        );
        assert!(
            result.is_ok(),
            "an engine install failure must not discard a good SDK install: {:?}",
            result.err()
        );

        let actions: Vec<String> = load_recent_audit_events(&paths, 10)?
            .into_iter()
            .map(|event| event.action)
            .collect();
        assert!(actions.iter().any(|action| action == "install_sdk"));
        assert!(
            actions.iter().any(|action| action == "engine_auto_install"),
            "the engine failure must still be recorded: {actions:?}"
        );
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
    fn render_engine_inventory_text_includes_marker_legend() {
        let (root, paths) = test_paths("engine-inventory-marker-legend");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = rendered.find(&legend).expect("legend line present");
        // Anchor on the marked row itself (not just `"{DEFAULT_ENGINE_MARKER} "`,
        // which also matches inside the legend line and would pass even if no
        // row were actually marked).
        let marker_pos = rendered
            .find(&format!("{DEFAULT_ENGINE_MARKER} {host_default}"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_omits_legend_when_configured_default_matches_nothing() {
        // A configured default naming an external plugin (or a stale/typo'd
        // name) matches zero rows in `engine_inventory()`'s built-ins list —
        // the legend would then explain a marker that appears nowhere.
        let (root, paths) = test_paths("engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            !rendered.contains("legend:"),
            "legend must be absent when the configured default matches no built-in engine; got:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_honors_configured_default_engine() {
        // Regression: this renderer used to mark only `default_engine_for_host`,
        // ignoring a configured `default_engine` — the same host-vs-configured
        // precedence `select_serve_engine` and `append_examine_engine_inventory`
        // already honor. Pick whichever engine the host does NOT prefer so the
        // configured value is guaranteed to actually change the marked engine.
        let (root, paths) = test_paths("engine-inventory-configured-default");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));
        let configured = if host_default == "vllm" {
            "lemonade"
        } else {
            "vllm"
        };
        let config = RocmCliConfig {
            default_engine: Some(configured.to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {configured}")),
            "configured default engine {configured} must be marked; got:\n{rendered}"
        );
        assert!(
            !rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {host_default}")),
            "host default {host_default} must not be marked once a different engine is configured; got:\n{rendered}"
        );
    }

    #[test]
    fn examine_treats_a_blank_configured_engine_as_unset() {
        // Mirrors `select_serve_engine`'s guard: a config file with
        // `default_engine = ""` must fall back to the host preference rather
        // than reporting an empty engine name as "effective" and marking none
        // of the real ones.
        let (root, paths) = test_paths("examine-engine-inventory-blank-configured");
        let config = RocmCliConfig {
            default_engine: Some(String::new()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("configured_default_engine: <platform default>"),
            "a blank configured value must read as unset:\n{output}"
        );
        assert!(
            output.contains("effective_default_engine: vllm"),
            "a blank configured value must fall back to the host default:\n{output}"
        );
        assert!(
            output.contains("  * vllm "),
            "the '*' marker must land on the host's default, not an empty name:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn friendly_engine_detect_notes_hide_probe_and_path_noise() {
        let lemonade = friendly_engine_detect_notes(
            "lemonade",
            &["Lemonade embeddable 10.6.0 is installed at D:/ROCm/temp/runtime; Lemonade is configured for llamacpp:rocm; no CPU fallback is used".to_owned()],
        )
        .expect("lemonade note");
        assert_eq!(lemonade, "Lemonade is ready on your AMD GPU.");

        let vllm = friendly_engine_detect_notes(
            "vllm",
            &["vLLM is not installed in a Linux/WSL ROCm Python environment. Native Windows is skipped; no CPU fallback is used.".to_owned()],
        )
        .expect("vllm note");
        assert_eq!(
            vllm,
            "vLLM is not installed in a Linux/WSL ROCm Python environment."
        );
        // Raw install paths from the probe body must not leak into the friendly note.
        assert!(!lemonade.contains("D:/"));
    }

    #[test]
    fn missing_packaged_engine_reason_has_no_deferred_first_party_engines() {
        assert!(missing_packaged_engine_reason("lemonade").is_none());
        assert!(missing_packaged_engine_reason("vllm").is_none());
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
                    // One byte under 1 MiB: `{:.1}` rounds it up to a full
                    // 1024 KiB, so the approved cap must read as "1.0 MiB",
                    // not "1024.0 KiB". It must also read as `MiB`, not `MB`
                    // — this is base-1024 math, not base-1000 (issue #190).
                    "artifact_max_bytes": 1_048_575
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
        assert!(rendered.contains("download: approved up to 1.0 MiB"));
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
            runtime_kind: Some("external_vllm".to_owned()),
            runtime_executable: None,
            managed_env: Some(false),
            python_version: None,
            torch_version: None,
            transformers_version: None,
            available_devices: vec![rocm_engine_protocol::EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: false,
                reason: Some(
                    "vLLM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows vLLM is skipped. No CPU fallback is used."
                        .to_owned(),
                ),
            }],
            capabilities: rocm_engine_protocol::EngineCapabilities {
                cpu: false,
                rocm_gpu: false,
                openai_compatible: true,
                tool_calling: false,
                quantized_models: "vllm-supported".to_owned(),
                reasoning_parser: false,
            },
            notes: Vec::new(),
        };

        if cfg!(windows) {
            assert_eq!(
                engine_runtime_status_label("vllm", &detect),
                "unsupported_native_windows"
            );
            assert!(
                model_registry_adapter_availability_note("vllm")
                    .is_some_and(|note| note.contains("unsupported_native_windows"))
            );
        } else {
            assert_eq!(engine_runtime_status_label("vllm", &detect), "not found");
            assert!(model_registry_adapter_availability_note("vllm").is_none());
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
    fn model_catalog_platform_heading_marks_the_host_gpu_in_ascii() {
        assert_eq!(
            model_catalog_platform_heading("AMD Instinct - MI300X", false),
            "AMD Instinct - MI300X",
            "an unmatched platform heading is the bare label"
        );

        // ROCMAI-543: this marker was `←` (U+2190), which a Windows console on
        // the OEM code page renders as mojibake. `render_model_registry_text_*`
        // only reaches this branch on a host whose real gfx target matches a
        // catalog platform, so the default no-GPU CI lane never covers it.
        // Asserting the exact string (not `contains`) is what makes a revert of
        // the punctuation alone fail, which a substring check would not.
        let marked = model_catalog_platform_heading("AMD Instinct - MI300X", true);
        assert_eq!(marked, "AMD Instinct - MI300X <- your GPU");
        assert!(
            marked.is_ascii(),
            "the host-GPU marker must be ASCII (ROCMAI-543): {marked}"
        );
    }

    /// ROCMAI-543: the clap doc comments are the largest single surface of
    /// user-facing text this fix rewrote, and nothing exercised them — no test
    /// renders `--help` at all. Walking the whole command tree, rather than
    /// only the top-level `--help`, is what makes this cover the edits that
    /// were actually made: every one of them sits on a nested subcommand's
    /// field or variant (`rocm runtime install --family`, `rocm services
    /// prune --older-than-hours`, `rocm runtime rollback`'s `after_help`,
    /// ...), none of which the top-level help renders. Asserting on the
    /// rendered help rather than on individual strings means a future option
    /// added anywhere in the tree is covered the day it lands.
    #[test]
    fn every_help_text_in_the_command_tree_is_ascii() {
        fn assert_ascii_help(cmd: &mut clap::Command, path: &str) {
            // Both renderings: `-h` and `--help` draw from different fields
            // (`about` vs `long_about`, and clap truncates the short form at
            // the first paragraph), so a non-ASCII character in a later
            // paragraph is only visible in the long form.
            for rendered in [cmd.render_help(), cmd.render_long_help()] {
                let rendered = rendered.to_string();
                assert!(
                    rendered.is_ascii(),
                    "`{path} --help` must be ASCII-only; non-ASCII punctuation renders as \
                     mojibake on a Windows console using the OEM code page (ROCMAI-543): \
                     {rendered}"
                );
            }
            // Collect first: `find_subcommand_mut` needs a mutable borrow that
            // cannot overlap the iterator borrowing `cmd`.
            let names: Vec<String> = cmd
                .get_subcommands()
                .map(|sub| sub.get_name().to_owned())
                .collect();
            for name in names {
                let child_path = format!("{path} {name}");
                let child = cmd
                    .find_subcommand_mut(&name)
                    .expect("subcommand name was just enumerated from this command");
                assert_ascii_help(child, &child_path);
            }
        }

        assert_ascii_help(&mut Cli::command(), "rocm");
    }

    #[test]
    fn render_model_registry_text_lists_builtin_recipes() {
        let rendered = render_model_registry_text_with_context_and_host(None, None, None);

        // ROCMAI-543: a Windows console on the OEM code page (437) renders a
        // UTF-8 em dash (and other non-ASCII punctuation) as mojibake like
        // "ΓÇö" instead of transliterating it, so user-facing CLI output must
        // stick to ASCII. `is_ascii()` is strictly stronger than a single
        // em-dash check: it also catches smart quotes, ellipses, and arrows
        // reintroduced anywhere in this output (a new catalog label, a new
        // hint line, a "your GPU" heading), not just U+2014.
        assert!(
            rendered.is_ascii(),
            "rocm model output must be ASCII-only; non-ASCII punctuation renders as \
             mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );

        // Clean header: what the list is + the one action, no implementation jargon.
        assert!(rendered.contains("Recommended models - run one with `rocm serve <model>`"));
        // Featured current models are grouped under their hardware target, each
        // with the quant that fits a single GPU.
        assert!(rendered.contains("AMD Ryzen AI - Strix Halo (Lemonade / llama.cpp)"));
        assert!(rendered.contains("AMD Instinct - MI300X, MI350X, MI355X (vLLM)"));
        // Strix Halo GGUF entries carry the servable owner/repo:variant ref.
        assert!(rendered.contains("unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"));
        assert!(rendered.contains("Q4_K_M GGUF"));
        assert!(rendered.contains("Qwen/Qwen3.6-27B"));
        assert!(rendered.contains("BF16"));
        // Multi-GPU-only models are not featured (single-GPU serving only).
        assert!(!rendered.contains("GLM-5.2"));
        assert!(!rendered.contains("DeepSeek-V4-Flash"));
        // Without a known host GPU, no misleading per-row fit verdict is shown.
        assert!(!rendered.contains("GPU fit unknown"));
        // Superseded / smoke / assistant recipes are hidden from the curated list.
        assert!(!rendered.contains("tiny-gpt2"));
        assert!(!rendered.contains("Qwen/Qwen3.5-4B"));
        assert!(!rendered.contains("Qwen3-4B-Instruct-2507-GGUF"));
        // Disclaimer: the list is a starting point; any compatible HF model works.
        assert!(rendered.contains("you can serve any compatible Hugging Face model"));
        assert!(rendered.contains("rocm serve <owner/repo>:<quant>"));
        assert!(rendered.contains("rocm model --verbose"));
        assert!(!rendered.contains("recommended_system_ram:"));
        assert!(!rendered.contains("engine_support:"));
        assert!(!rendered.contains("artifact_check:"));
    }

    #[test]
    fn render_model_registry_text_shows_fit_when_gpu_known() {
        // With a known aggregate VRAM, each row gains a concrete fit verdict.
        // 60 GiB fits Qwen3.6-27B (~54) but not Gemma-4-31B (~62).
        let rendered = render_model_registry_text_with_context_and_host(None, Some(60.0), None);

        assert!(rendered.contains("fits this GPU"));
        assert!(rendered.contains("needs a larger GPU"));
        assert!(!rendered.contains("GPU fit unknown"));
    }

    #[test]
    fn model_catalog_header_only_names_source_for_configured_index() {
        // Default built-in list: no implementation jargon, just the action.
        let builtin = ModelRecipeRegistry {
            recipes: Vec::new(),
            platforms: Vec::new(),
            source: ModelRecipeRegistrySource::BuiltIn,
        };
        let header = model_catalog_header(&builtin);
        assert_eq!(
            header,
            "Recommended models - run one with `rocm serve <model>`"
        );
        assert!(!header.contains("built-in"));
        assert!(!header.contains("fallback"));

        // A configured index is the only case worth advertising provenance for.
        let configured = ModelRecipeRegistry {
            recipes: Vec::new(),
            platforms: Vec::new(),
            source: ModelRecipeRegistrySource::SignedIndex {
                index_path: PathBuf::from("/etc/rocm/recipes.json"),
                signature_path: PathBuf::from("/etc/rocm/recipes.json.sig"),
                public_key_path: PathBuf::from("/etc/rocm/recipes.pub"),
            },
        };
        assert_eq!(
            model_catalog_header(&configured),
            "Recommended models - from recipe index /etc/rocm/recipes.json"
        );
        // ROCMAI-543: both header variants are printed verbatim, so neither may
        // carry non-ASCII punctuation that mojibakes on a Windows OEM code page.
        assert!(model_catalog_header(&builtin).is_ascii());
        assert!(model_catalog_header(&configured).is_ascii());
    }

    #[test]
    fn render_model_registry_verbose_text_keeps_diagnostics() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("model recipes"));
        assert!(rendered.contains("aliases=[qwen"));
        assert!(rendered.contains("recommended_system_ram: 16 GiB"));
        assert!(rendered.contains("system_ram_fit: unknown"));
        assert!(rendered.contains("gpu_fit: unknown"));
        // Every production caller of this function hardcodes
        // `aggregate_gpu_vram_gib = None` (see `Command::Model` and the `/model`
        // assistant tool), so this branch is the only one `rocm model` ever
        // actually reaches -- its remediation must name a command this same
        // path can act on, not one whose reading this function never sees.
        assert!(rendered.contains("action: run `rocm diagnose --model"));
        assert!(!rendered.contains("amd-smi metric --json"));
        assert!(rendered.contains("engine_support:"));
        assert!(rendered.contains("engine_action: use /engine install <engine>"));
        assert!(rendered.contains("source: built-in recipe registry"));
        // Verbose keeps every recipe (including hidden ones) and annotates each
        // with its curated catalog placement.
        assert!(rendered.contains("catalog: AMD Instinct - MI300X, MI350X, MI355X (vLLM)"));
        assert!(rendered.contains("catalog: hidden (resolvable via rocm serve"));
        assert!(rendered.contains("sshleifer/tiny-gpt2"));
        // ROCMAI-543: `rocm model --verbose` output must stick to ASCII too, not
        // only the default non-verbose listing.
        assert!(
            rendered.is_ascii(),
            "rocm model --verbose output must be ASCII-only; non-ASCII punctuation renders \
             as mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );
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
            engines: vec!["vllm".to_owned()],
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
            model_id_override: None,
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
        assert!(output.contains("engines=[vllm]"));
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

    /// `rocm examine` is the first command a user runs when something ROCm is
    /// wrong, and "my build cannot find `hipcc`" is now a reachable state by
    /// design. Reporting the toolchain only from `rocm runtimes list` leaves
    /// the primary diagnostic unable to answer the question the opt-in
    /// created. Both polarities are pinned, because a field hardcoded to either
    /// word reads as working from a single run.
    #[test]
    fn examine_runtime_state_reports_whether_the_active_runtime_has_the_toolchain() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-toolchain");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        // The fixture records `devel: true` with no composition.
        let mut output = String::new();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: included"),
            "a toolchain runtime must say so:\n{output}"
        );

        // A runtime-only install reports the other way. Written through the
        // recorded specs, which is what `includes_devel` actually reads, so
        // this exercises the same path a real `rocm install sdk` produces.
        let runtime_only = therock::InstalledRuntimeManifest {
            devel: false,
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "canonical".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some("gfx1201".to_owned()),
            }),
            ..manifest
        };
        fs::write(
            runtime_manifest_path(&paths, &runtime_only.runtime_key),
            serde_json::to_vec_pretty(&runtime_only)?,
        )?;
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: excluded"),
            "a runtime-only runtime must say so:\n{output}"
        );

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
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.engine_config_mut("vllm").preferred_runtime_id =
            Some("therock-release:gfx120X-all".to_owned());
        let mut output = String::new();

        // Host default deliberately disagrees with the configured value: an
        // engine the user chose must outrank whatever the GPU would prefer.
        append_examine_engine_inventory(&mut output, &paths, &config, "lemonade");

        assert!(output.contains("engine_inventory:"));
        assert!(output.contains("configured_default_engine: vllm"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(output.contains("* vllm"));
        assert!(output.contains("runtime_pref=therock-release:gfx120X-all"));
        assert!(output.contains("plugin_dirs:"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_engine_inventory_includes_marker_legend() {
        let (root, paths) = test_paths("examine-engine-inventory-marker-legend");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = output.find(&legend).expect("legend line present");
        let marker_pos = output
            .find(&format!("{DEFAULT_ENGINE_MARKER} vllm"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{output}"
        );
    }

    #[test]
    fn examine_engine_inventory_omits_legend_when_effective_default_matches_nothing() {
        // Same gap as `render_engine_inventory_text_with_paths`: an effective
        // default naming an external plugin (or a stale/typo'd name) matches
        // zero rows in `engine_inventory()`'s built-ins list.
        let (root, paths) = test_paths("examine-engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        assert!(
            !output.contains("legend:"),
            "legend must be absent when the effective default matches no built-in engine; got:\n{output}"
        );
    }

    // ---------- engine shell prompt shim ----------

    #[test]
    fn examine_engine_inventory_falls_back_to_the_host_engine_when_unconfigured() {
        // With nothing configured, the reported default must be the engine this
        // GPU actually serves on. On an Instinct host that is vLLM; reporting the
        // platform constant here is what made `examine` contradict `serve`.
        let (root, paths) = test_paths("examine-engine-inventory-host");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        // The configured line still reads as unset — the host default must not be
        // mistaken for something the user chose.
        assert!(output.contains("configured_default_engine: <platform default>"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(
            output.contains("* vllm"),
            "the primary marker should follow the host engine:\n{output}"
        );
        // Nothing is being overridden, so there is nothing to warn about.
        assert!(
            !output.contains("configured_default_note"),
            "an unconfigured host must not be warned about an override:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_flags_a_configured_engine_that_overrides_the_host() {
        // Installs provisioned before the installer stopped seeding
        // `default_engine` still carry `lemonade` on disk. That seed is
        // indistinguishable from a deliberate choice, so it is not rewritten --
        // but the user is told why their Instinct box is not on vLLM, and how to
        // change it.
        let (root, paths) = test_paths("examine-engine-inventory-override");
        let config = RocmCliConfig {
            default_engine: Some("lemonade".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("effective_default_engine: lemonade"),
            "the configured engine still wins:\n{output}"
        );
        assert!(
            output.contains("configured_default_note:"),
            "the override must be called out:\n{output}"
        );
        assert!(
            output.contains("rocm config clear-default-engine"),
            "the note must name the remedy, not just the problem:\n{output}"
        );
        assert!(
            output.contains("  * lemonade "),
            "the '*' marker must land on the configured engine, not the host's:\n{output}"
        );
        assert!(
            !output.contains("  * vllm "),
            "the host's preferred engine must not also be marked once overridden:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_is_silent_when_the_configured_engine_agrees_with_the_host() {
        let (root, paths) = test_paths("examine-engine-inventory-agrees");
        let config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            !output.contains("configured_default_note"),
            "there is no override to report when the two agree:\n{output}"
        );
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
    fn installed_update_runtime_is_selected_by_exact_target_key() {
        // Everything a version match would have keyed on is identical here:
        // same channel, format, family and version. Only the composition-keyed
        // runtime key tells the freshly installed repair apart from the stale
        // runtime it was installed to replace.
        let stale = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.14.0",
        );
        let wrong_family = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0-ffffffffffffffff",
            "therock-release:gfx110X-all",
            "gfx110X-all",
            "7.14.0",
        );
        let repaired = test_runtime_manifest_for_update(
            "release-wheel-multi-arch-7-14-0-0123456789abcdef",
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.14.0",
        );
        let manifests = vec![stale, wrong_family, repaired.clone()];

        let selected = select_installed_update_runtime(&manifests, &repaired.runtime_key)
            .expect("the side-by-side repair must be selected by its exact key");
        assert_eq!(selected.runtime_key, repaired.runtime_key);

        assert!(
            select_installed_update_runtime(&manifests, "release-wheel-multi-arch-7-15-0")
                .is_none(),
            "an install that wrote no manifest for the planned key must not resolve to a sibling"
        );
    }

    #[test]
    fn update_activate_summary_hints_rollback_only_when_a_previous_runtime_exists() {
        let with_previous = RuntimeActivationResult {
            runtime_id: "therock-release:gfx942".to_owned(),
            runtime_key: "release-wheel-gfx942".to_owned(),
            previous_runtime_key: Some("release-wheel-gfx942-old".to_owned()),
            reactivated: false,
            // No live services: this test is about the rollback hint, and an
            // empty reconciliation is what an update on a machine with nothing
            // serving produces.
            services: runtime_services::RuntimeServiceReconciliation::default(),
        };
        let mut rendered = String::new();
        append_update_activate_summary(&mut rendered, &with_previous);
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "the update summary reports service state the same way `runtimes \
             activate` does, including when there is none:\n{rendered}"
        );
        assert!(
            rendered.contains("next step: if this causes problems, run `rocm runtimes rollback`"),
            "a previous runtime is recorded, so rollback is a valid recovery path:\n{rendered}"
        );
        assert!(
            rendered.contains("undoes only this one activation"),
            "the hint should state the single-step limit up front, not just in --help:\n{rendered}"
        );

        let without_previous = RuntimeActivationResult {
            runtime_id: "therock-release:gfx942".to_owned(),
            runtime_key: "release-wheel-gfx942".to_owned(),
            previous_runtime_key: None,
            reactivated: false,
            services: runtime_services::RuntimeServiceReconciliation::default(),
        };
        let mut rendered = String::new();
        append_update_activate_summary(&mut rendered, &without_previous);
        assert!(
            !rendered.contains("rocm runtimes rollback"),
            "no previous runtime is recorded, so `rocm runtimes rollback` would hard-error; \
             must not hint at a command that immediately fails:\n{rendered}"
        );
    }

    pub(crate) fn test_runtime_manifest_for_update(
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
            source_layout_generation: None,
            index_url: Some("https://example.invalid/therock".to_owned()),
            tarball_file_name: None,
            python_launcher: Some("python".to_owned()),
            python_executable: Some("python".to_owned()),
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms: 1,
        }
    }

    /// How `classify_service_runtime_state` sorts a record against the runtime
    /// being activated when the manifest list is complete: a server already on
    /// that runtime matches, the same record is stale once a different runtime
    /// is activated, a family id shared by two installs is reported rather than
    /// guessed, one that resolves to a single install matches that install, and
    /// a record naming no runtime is unknown.
    ///
    /// It does not isolate the exact-key fast path: `runtime_manifest_for_selector`
    /// resolves an exact key too, so every exact-key assertion here still holds
    /// without that arm. `a_record_naming_the_active_runtime_matches_without_any_manifest_to_resolve_it`
    /// covers it.
    #[test]
    fn classifying_a_service_against_the_runtime_it_already_runs_on() {
        let (root, paths) = test_paths("classify-service-runtime-state");
        paths.ensure().unwrap();
        let manifests = vec![
            test_runtime_manifest_for_update(
                OLD_RUNTIME_KEY,
                "therock-release:gfx120X-all",
                "gfx120X-all",
                "7.12.0",
            ),
            test_runtime_manifest_for_update(
                NEW_RUNTIME_KEY,
                "therock-release:gfx120X-all",
                "gfx120X-all",
                "7.13.0",
            ),
        ];
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-classify",
            "vllm",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            9,
            "managed",
            1,
            None,
            None,
            None,
        );

        record.runtime_id = Some(OLD_RUNTIME_KEY.to_owned());
        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &manifests, OLD_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Matches,
            "a server already on the runtime being activated is not left behind"
        );
        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &manifests, NEW_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Stale {
                recorded: OLD_RUNTIME_KEY.to_owned()
            },
            "activating past it makes the same record stale"
        );

        // A family id both installed versions share resolves to neither, so it
        // stays Stale rather than silently matching one of them.
        record.runtime_id = Some("therock-release:gfx120X-all".to_owned());
        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &manifests, OLD_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Stale {
                recorded: "therock-release:gfx120X-all".to_owned()
            },
            "an ambiguous family id is reported rather than assumed correct"
        );

        // The family-resolution arm itself, which the ambiguous case above does
        // not reach: one install carrying the family id resolves to it, and a
        // server already on the runtime being activated must not be reported as
        // left behind — `--restart-services` would otherwise stop and respawn it
        // for nothing.
        let one_install_of_the_family = vec![test_runtime_manifest_for_update(
            NEW_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "gfx120X-all",
            "7.13.0",
        )];
        assert_eq!(
            runtime_services::classify_service_runtime_state(
                &record,
                &one_install_of_the_family,
                NEW_RUNTIME_KEY
            ),
            runtime_services::ServiceRuntimeState::Matches,
            "an unambiguous family id resolves to the install that carries it"
        );
        assert_eq!(
            runtime_services::classify_service_runtime_state(
                &record,
                &one_install_of_the_family,
                OLD_RUNTIME_KEY
            ),
            runtime_services::ServiceRuntimeState::Stale {
                recorded: "therock-release:gfx120X-all".to_owned()
            },
            "resolving is not excusing: the same family id is stale against a \
             runtime it does not resolve to"
        );

        record.runtime_id = None;
        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &manifests, OLD_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Unknown,
            "a record naming no runtime says nothing about which one it loaded"
        );

        let _ = fs::remove_dir_all(root);
    }

    // The exact-key arm of `classify_service_runtime_state` answers from the
    // record alone, so it still holds for a caller whose manifest list does not
    // contain the key. With the full list the arm cannot be told apart from the
    // resolution below it, because that resolves an exact key too; an empty list
    // is where it is the only thing that can say `Matches`.
    // Mutation: deleting the `Some(recorded) if recorded == runtime_key` arm
    // makes the `Matches` assertion fail, since nothing is left to resolve the
    // key.
    #[test]
    fn a_record_naming_the_active_runtime_matches_without_any_manifest_to_resolve_it() {
        let (_root, paths) = test_paths("classify-service-runtime-state-no-manifests");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-classify-empty",
            "vllm",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            9,
            "managed",
            1,
            Some(NEW_RUNTIME_KEY.to_owned()),
            None,
            None,
        );

        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &[], NEW_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Matches,
            "a record that names the runtime being activated needs no manifest to say so"
        );

        record.runtime_id = Some(OLD_RUNTIME_KEY.to_owned());
        assert_eq!(
            runtime_services::classify_service_runtime_state(&record, &[], NEW_RUNTIME_KEY),
            runtime_services::ServiceRuntimeState::Stale {
                recorded: OLD_RUNTIME_KEY.to_owned()
            },
            "a different recorded runtime is stale however short the manifest list is"
        );
    }

    #[cfg(unix)]
    fn managed_record_for_pid(
        paths: &AppPaths,
        pid: u32,
        start_ticks: Option<u64>,
    ) -> ManagedServiceRecord {
        let mut record = ManagedServiceRecord::new(
            paths,
            "svc-managed-stop",
            "vllm",
            "m",
            "m",
            "127.0.0.1",
            9,
            "managed",
            pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(pid);
        record.supervisor_pid = pid;
        record.supervisor_start_ticks = start_ticks;
        record
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_refuses_recycled_pid() {
        // A live process whose recorded identity does not match: this is exactly
        // what a recycled PID looks like. The stop must NOT signal it.
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let (root, paths) = test_paths("managed-stop-recycled");
        let real = rocm_core::process_start_ticks(pid).expect("start-ticks");
        let record = managed_record_for_pid(&paths, pid, Some(real.wrapping_add(1)));

        let (signaled, all_stopped) = terminate_recorded_service_pids(&record);

        assert!(signaled.is_empty(), "must not signal a mismatched identity");
        // A recycled PID means the recorded service process itself is gone.
        assert!(all_stopped);
        assert!(
            rocm_core::process_is_running(pid),
            "the unrelated live process must survive"
        );
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_verifies_distinct_supervisor_and_engine_pids() {
        // The launcher and the engine server are different processes; each PID
        // must be verified with its OWN start-time token and terminated.
        let sup = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn supervisor");
        let eng = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn engine");
        let (sup_pid, eng_pid) = (sup.id(), eng.id());
        let (root, paths) = test_paths("managed-stop-distinct");
        let mut record =
            managed_record_for_pid(&paths, sup_pid, rocm_core::process_start_ticks(sup_pid));
        record.engine_pid = Some(eng_pid);
        record.engine_start_ticks = rocm_core::process_start_ticks(eng_pid);

        let (mut signaled, all_stopped) = terminate_recorded_service_pids(&record);
        signaled.sort_unstable();
        let mut expected = vec![sup_pid, eng_pid];
        expected.sort_unstable();

        assert_eq!(signaled, expected);
        assert!(all_stopped);
        let (mut sup, mut eng) = (sup, eng);
        let _ = sup.wait();
        let _ = eng.wait();
        assert!(!rocm_core::process_is_running(sup_pid));
        assert!(!rocm_core::process_is_running(eng_pid));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_terminates_verified_pid() {
        // Correct identity: the recorded process is really ours, so it is stopped.
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let (root, paths) = test_paths("managed-stop-verified");
        let real = rocm_core::process_start_ticks(pid).expect("start-ticks");
        let record = managed_record_for_pid(&paths, pid, Some(real));

        let (signaled, all_stopped) = terminate_recorded_service_pids(&record);

        assert_eq!(signaled, vec![pid]);
        assert!(all_stopped);
        // Reap our own child (a detached managed process would be reaped by init)
        // so the liveness check does not observe a not-yet-reaped zombie.
        let mut child = child;
        let _ = child.wait();
        assert!(
            !rocm_core::process_is_running(pid),
            "the verified process must be terminated"
        );
        let _ = fs::remove_dir_all(root);
    }

    pub(crate) fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("main")
            .join(format!(
                "rocm-cli-main-test-{name}-{}",
                crate::test_support::unique_suffix()
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

    #[test]
    fn test_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| test_paths("same-label").0);
    }

    // ---- Phase 9: reroute dispatch (bare `rocm` + interactive `rocm chat`) ----
    //
    // The interactive branches require a real TTY (`interactive_terminal()`),
    // which is unavailable in CI, and the dash visuals are trust-prior. These
    // tests instead PROVE the dispatch TARGET changed: the two interactive
    // handlers now call `dash::run_chat` and no longer call `tui::run`. We read
    // this source file at test time and assert on the handler bodies.
    fn main_rs_source() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("main.rs");
        fs::read_to_string(path).expect("read apps/rocm/src/main.rs")
    }

    /// Extract the body of `fn launch_default()` (brace-balanced).
    fn launch_default_body(src: &str) -> String {
        slice_braced_block(src, "fn launch_default() -> Result<()> {")
    }

    /// Extract the whole `Some(Command::Chat { .. }) => { .. }` handler arm,
    /// brace-balanced from the destructure pattern through the block body, so
    /// both the destructured fields (e.g. `chat_mock`) and the dispatch code are
    /// visible. Restricts to the production handler (the `=> {` form), not the
    /// variant declaration which has no `=>`.
    fn command_chat_handler_body(src: &str) -> String {
        // Anchor on the handler arm specifically: the destructure that is
        // immediately followed (after the closing `}` + `)`) by `=> {`.
        let arm = "Some(Command::Chat {";
        let arm_at = src.find(arm).expect("Command::Chat handler present");
        // Balance from the `{` of the destructure to capture the full arm,
        // including the `=> { ... }` block that follows.
        slice_braced_to_arm_end(&src[arm_at..])
    }

    /// Given a slice beginning at `Some(Command::Chat {`, return the text from
    /// that point through the end of the arm's `=> { .. }` block.
    fn slice_braced_to_arm_end(src: &str) -> String {
        // 1. balance the destructure `{ .. }`.
        let pat_open = src.find('{').expect("destructure open brace");
        let bytes = src.as_bytes();
        let mut depth = 0usize;
        let mut pat_end = pat_open;
        for (i, &b) in bytes.iter().enumerate().skip(pat_open) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        pat_end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        // 2. find the `=> {` block opener after the destructure and balance it.
        let after = &src[pat_end..];
        let block_rel = after.find("=> {").expect("arm block opener `=> {`");
        let block_body = slice_braced_block(after, &after[block_rel..block_rel + 4]);
        // Return destructure + block body together.
        format!("{}{}", &src[pat_open..pat_end], block_body)
    }

    /// Return the brace-balanced block that follows `opener` (which must end in
    /// `{`), including the contents but excluding the trailing brace's tail.
    fn slice_braced_block(src: &str, opener: &str) -> String {
        let start = src.find(opener).unwrap_or_else(|| {
            panic!("opener not found: {opener}");
        });
        let brace_start = start + opener.len() - 1; // index of the `{`
        let bytes = src.as_bytes();
        let mut depth = 0usize;
        let mut end = brace_start;
        for (i, &b) in bytes.iter().enumerate().skip(brace_start) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        src[brace_start..end].to_string()
    }

    /// Strip `//` line comments so assertions key on real code, not the prose
    /// comments that still mention `tui::run` for documentation.
    fn strip_line_comments(block: &str) -> String {
        block
            .lines()
            .map(|line| match line.find("//") {
                Some(idx) => &line[..idx],
                None => line,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn dash_run_chat_is_the_bare_and_chat_dispatch_target() {
        // Symbol exists with the expected signature (bare `rocm` / `rocm chat`
        // interactive target). Coercing to the fn pointer fails to compile if
        // the entrypoint is removed or its signature drifts; casting the pointer
        // to an address gives the binding a real effect (no underscore-bind).
        let target: fn(bool, ChatInferenceParams, bool) -> Result<()> = dash::run_chat;
        assert_ne!(target as usize, 0);
    }

    #[test]
    fn launch_default_routes_to_dash_run_chat_not_tui() {
        let src = main_rs_source();
        let body = strip_line_comments(&launch_default_body(&src));
        assert!(
            body.contains("dash::run_launcher(false)"),
            "bare `rocm` interactive branch must route to the launcher; body:\n{body}"
        );
        assert!(
            !body.contains("tui::run"),
            "launch_default must NOT call tui::run after the reroute; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_interactive_routes_to_dash_run_chat_not_tui() {
        let src = main_rs_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("dash::run_chat("),
            "interactive `rocm chat` must route to dash::run_chat(...); body:\n{body}"
        );
        assert!(
            !body.contains("tui::run"),
            "Command::Chat handler must NOT call tui::run after the reroute; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_interactive_notes_dropped_provider_flag() {
        // Phase-9 polish: --provider on interactive `rocm chat` is no longer
        // silently ignored — the handler emits a one-line note when provider is
        // set before rerouting to the dash. Proven by reading the handler body
        // (the interactive branch requires a TTY, unavailable in CI).
        let src = main_rs_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("provider.is_some()"),
            "handler must gate the note on a set --provider; body:\n{body}"
        );
        assert!(
            body.contains("/provider"),
            "the note must point the user at /provider for live switching; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_honors_chat_mock_and_keeps_prompt_passthrough() {
        let src = main_rs_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        // --chat-mock is forwarded into the dash reroute.
        assert!(
            body.contains("chat_mock,") || body.contains("chat_mock\n"),
            "Command::Chat must destructure the --chat-mock field; body:\n{body}"
        );
        assert!(
            body.contains("dash::run_chat(") && body.contains("chat_mock,"),
            "--chat-mock must be forwarded to run_chat; body:\n{body}"
        );
        assert!(
            body.contains("ChatInferenceParams")
                && body.contains("temperature,")
                && body.contains("top_p,")
                && body.contains("max_tokens,"),
            "sampling controls must be forwarded to interactive dash chat; body:\n{body}"
        );
        // The scriptable prompt passthrough stays on the text render path,
        // NOT the dash/TUI.
        assert!(
            body.contains("chat::render_chat_prompt_text("),
            "prompt passthrough must still call render_chat_prompt_text; body:\n{body}"
        );
        assert!(
            body.contains("chat::render_chat_text("),
            "non-interactive no-prompt path must still call render_chat_text; body:\n{body}"
        );
    }

    #[test]
    fn command_chat_reads_prompt_from_piped_stdin() {
        // A structural guard on the wiring, not a behavioral test: the handler
        // reads the real fd 0, which an in-process test cannot pipe. The
        // behavior — piped text reaching the model unaltered — is covered end
        // to end by the `chat-09` scenario (`@id:chat-cli-stdin-prompt`); what
        // is checked here is that the `--prompt`-less arm still resolves the
        // prompt from `read_piped_prompt`, with the result feeding the dispatch
        // rather than being discarded or read after the send decision is
        // already made.
        let src = main_rs_source();
        let body = strip_line_comments(&command_chat_handler_body(&src));
        assert!(
            body.contains("chat::read_piped_prompt("),
            "no-prompt path must read piped stdin via read_piped_prompt; body:\n{body}"
        );
        assert!(
            body.contains("None => chat::read_piped_prompt()?"),
            "the piped read must supply the prompt that is dispatched (and \
             propagate its error), not be a discarded call; body:\n{body}"
        );
        let read_at = body
            .find("chat::read_piped_prompt(")
            .expect("asserted above");
        let send_at = body
            .find("chat::render_chat_prompt_text(")
            .unwrap_or_else(|| panic!("chat handler no longer sends a prompt; body:\n{body}"));
        assert!(
            read_at < send_at,
            "stdin must be read before the send/status-screen decision, or a \
             piped prompt cannot influence it; body:\n{body}"
        );
    }

    /// Hosts whose engine choice differs, so the assertion below is exercised on
    /// more than one branch of `select_serve_engine`.
    fn engine_selection_hosts() -> Vec<(&'static str, rocm_core::HostGpuSummary)> {
        vec![
            (
                "Instinct MI300X",
                rocm_core::HostGpuSummary {
                    name: Some("AMD Instinct MI300X".to_owned()),
                    gfx_target: Some("gfx942".to_owned()),
                    therock_family: Some("gfx94X-dcgpu".to_owned()),
                },
            ),
            (
                "Strix Halo",
                rocm_core::HostGpuSummary {
                    name: Some("AMD Radeon 8060S".to_owned()),
                    gfx_target: Some("gfx1151".to_owned()),
                    therock_family: Some("gfx1151".to_owned()),
                },
            ),
            ("no detected GPU", rocm_core::HostGpuSummary::default()),
        ]
    }

    /// I3 — `rocm diagnose --model` never names an engine `rocm serve` would not
    /// select.
    ///
    /// Two layers agreeing about one decision. The defect this catches is not a
    /// wrong `select_serve_engine`, it is the readiness path re-deriving the
    /// choice from `recipe.preferred_engines` and drifting: that reads correctly
    /// and is wrong on exactly the hosts where serve overrides the recipe. So
    /// both real call sites are driven with the same inputs and compared, rather
    /// than either being called with literal arguments.
    #[test]
    fn the_engine_doctor_names_is_the_engine_serve_would_select() {
        let registry = rocm_core::builtin_model_recipe_registry();
        let host = HostFacts {
            accelerator_memory: AcceleratorMemory::Dedicated(192.0),
            system_ram_gib: Some(1024.0),
        };

        // Non-vacuity: at least one pair must be a case where serve OVERRIDES the
        // recipe's own first preference, because that is the only case a
        // re-derivation would get wrong. Native Windows has no such case --
        // `preferred_serve_engine_for_host_gpu_summary` never prefers vLLM there
        // -- so the check is stated where it exists and the equality assertion
        // below still runs everywhere.
        if !rocm_core::runtime_is_windows() {
            let overridden = engine_selection_hosts().into_iter().any(|(_, summary)| {
                registry.recipes.iter().any(|recipe| {
                    let selected = select_serve_engine(None, None, Some(recipe), Some(&summary));
                    recipe
                        .preferred_engines
                        .first()
                        .is_some_and(|preferred| !preferred.eq_ignore_ascii_case(&selected.engine))
                })
            });
            assert!(
                overridden,
                "no host/recipe pair here exercises serve overriding the recipe's preferred \
                 engine, so the comparison below cannot catch the readiness path re-deriving \
                 the choice itself"
            );
        }

        for (label, summary) in engine_selection_hosts() {
            for recipe in &registry.recipes {
                let expected = select_serve_engine(None, None, Some(recipe), Some(&summary));
                let report = assess_model_for_host(
                    &recipe.canonical_model_id,
                    &ModelCatalogSource::Available(&registry),
                    &host,
                    None,
                    Some(&summary),
                );
                assert_eq!(
                    report.engine.as_deref(),
                    Some(expected.engine.as_str()),
                    "on {label}, `rocm diagnose --model {}` names {:?} but `rocm serve` would \
                     select `{}`",
                    recipe.canonical_model_id,
                    report.engine,
                    expected.engine
                );
            }
        }
    }

    /// When the platform gate rules an engine out, the message names it and
    /// points at WSL/Linux -- the actual escape hatch, not a guess.
    #[test]
    fn unsupported_here_for_names_the_engine_when_ruled_out() {
        let message =
            unsupported_here_for("vllm", true).expect("a ruled-out engine gets a message");
        assert!(
            message.contains("vllm"),
            "the message has to name the engine it is talking about: {message:?}"
        );
        assert!(
            message.to_lowercase().contains("wsl") || message.to_lowercase().contains("linux"),
            "the message has to point at the actual escape hatch: {message:?}"
        );
    }

    /// When the platform gate allows an engine, there is nothing to report --
    /// asserted directly against the pure helper, not by relying on whichever
    /// OS this test happens to run on.
    #[test]
    fn unsupported_here_for_is_none_when_allowed() {
        assert_eq!(unsupported_here_for("vllm", false), None);
        assert_eq!(unsupported_here_for("llamacpp", false), None);
    }

    /// Regression test for the engine-consistency bug: `assess_model_on_this_host`
    /// used to call `AppPaths::discover()` twice -- once (implicitly `None`) for
    /// the GPU summary, once for the config lookup a few lines later -- so the
    /// two could read different directories. Writing a `default_engine` to one
    /// known `AppPaths` and driving both the config lookup and the GPU-summary
    /// lookup off that same value pins it down: if a future change reintroduces
    /// a second, independent `AppPaths::discover()` for either half, the
    /// configured engine this test writes to disk stops reaching the verdict.
    #[test]
    fn the_configured_default_engine_reaches_the_readiness_verdict_from_the_same_paths_as_the_gpu_summary()
     {
        let dir = std::env::temp_dir().join(format!(
            "rocm-diagnose-model-engine-consistency-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = AppPaths {
            config_dir: dir.clone(),
            data_dir: dir.clone(),
            cache_dir: dir.clone(),
        };
        let config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..Default::default()
        };
        config
            .save(&paths)
            .expect("write a config.json under the temp AppPaths");

        let registry = rocm_core::builtin_model_recipe_registry();
        let recipe = registry
            .recipes
            .first()
            .expect("the builtin registry ships at least one recipe");
        let examination = rocm_core::Examination::default();

        // `configured_default` outranks the host GPU summary and the recipe's own
        // preference in `select_serve_engine`, so this stays deterministic
        // regardless of what this test happens to run on.
        let summary = detect_host_gpu_summary(Some(&paths));
        let expected = select_serve_engine(None, Some("vllm"), Some(recipe), Some(&summary));

        let readiness =
            assess_model_with_host_paths(&recipe.canonical_model_id, &examination, Some(&paths));

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            readiness.engine.as_deref(),
            Some(expected.engine.as_str()),
            "the config written under this AppPaths said default_engine = \"vllm\", but the \
             readiness verdict named {:?} -- the config lookup and the GPU-summary lookup must \
             read the same discovered AppPaths",
            readiness.engine
        );
    }

    /// A row reporting `total_vram: 0` is not a measurement of an empty GPU; it
    /// is `amd-smi` enumerating a device it could not read the memory
    /// controller for. Keeping the row would let `host_accelerator_memory`
    /// report `AcceleratorMemory::Dedicated(0.0)`, which the readiness verdict
    /// then treats as "measured, and it is nothing" (`Blocked`) rather than
    /// "could not measure" (`Undetermined`).
    #[test]
    fn parse_gpu_vram_usage_drops_a_zero_total_row() {
        let value = json!({
            "gpu_data": [
                {
                    "gpu": 0,
                    "mem_usage": {
                        "used_vram": {"value": 0},
                        "total_vram": {"value": 0},
                    },
                },
                {
                    "gpu": 1,
                    "mem_usage": {
                        "used_vram": {"value": 512},
                        "total_vram": {"value": 16384},
                    },
                },
            ],
        });

        let rows = parse_gpu_vram_usage(&value);

        assert_eq!(
            rows,
            vec![GpuVramUsage {
                index: 1,
                used_mb: 512,
                total_mb: 16384,
            }],
            "a zero-total row must be dropped, not kept as a 0 GiB measurement: {rows:?}"
        );
    }

    /// All rows reporting a zero total collapses to no telemetry at all, so
    /// `gpu_vram_usage()`'s own `is_empty` check turns it into `None` --
    /// "could not measure" -- rather than a `Vec` of one unusable row.
    #[test]
    fn parse_gpu_vram_usage_returns_nothing_when_every_row_is_zero_total() {
        let value = json!({
            "gpu_data": [{
                "gpu": 0,
                "mem_usage": {
                    "used_vram": {"value": 0},
                    "total_vram": {"value": 0},
                },
            }],
        });

        assert!(parse_gpu_vram_usage(&value).is_empty());
    }

    /// A single dedicated GPU reports its measured total, unmodified.
    #[test]
    fn classify_accelerator_memory_reports_dedicated_vram_for_a_discrete_gpu() {
        let vram = [GpuVramUsage {
            index: 0,
            used_mb: 1024,
            total_mb: 16384,
        }];
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        assert_eq!(result, AcceleratorMemory::Dedicated(16.0));
    }

    /// The defect this guards against: an APU's `amd-smi` carve-out figure is
    /// not the pool the engine actually allocates from (the real pool is the
    /// GTT aperture, which nothing in this binary can read), so asserting
    /// installed system RAM as a stand-in used to fabricate a number that
    /// could read `Ready` for a model that does not actually fit. The correct,
    /// honest answer is `UnifiedMemoryUnreadable`, not a number of any kind,
    /// and not the plain `Unknown` used for genuine no-telemetry hosts either
    /// -- confirming this never again quietly regresses to reporting a
    /// fabricated figure (or worse, the APU's own tiny carve-out as
    /// `Dedicated`), and never again collapses back into the generic
    /// `Unknown`, whose `amd-smi` remediation would be a dead end here.
    #[test]
    fn classify_accelerator_memory_reports_unifiedmemoryunreadable_not_total_ram_for_an_apu() {
        let apu_carveout = [GpuVramUsage {
            index: 0,
            used_mb: 2048,
            total_mb: 4096,
        }];
        let result = classify_accelerator_memory(Some(&apu_carveout), Some("gfx1151"), true, None);
        assert_eq!(
            result,
            AcceleratorMemory::UnifiedMemoryUnreadable,
            "an APU's memory pool must read UnifiedMemoryUnreadable, not a fabricated figure, \
             and not the generic Unknown either: {result:?}"
        );
    }

    /// No VRAM telemetry at all, but the examination positively saw an AMD
    /// GPU: unmeasurable, not absent.
    #[test]
    fn classify_accelerator_memory_reports_unknown_when_gpu_present_but_unmeasured() {
        let result = classify_accelerator_memory(None, None, true, None);
        assert_eq!(result, AcceleratorMemory::Unknown);
    }

    /// No VRAM telemetry and no evidence of an AMD GPU: there is nothing to
    /// allocate a model from.
    #[test]
    fn classify_accelerator_memory_reports_none_when_no_gpu_is_present() {
        let result = classify_accelerator_memory(None, None, false, None);
        assert_eq!(result, AcceleratorMemory::None);
    }

    /// An APU paired with a discrete card (more than one GPU reported) keeps
    /// the discrete card's warning meaningful, per `vram_capacity_is_meaningful`
    /// -- the multi-GPU case cannot attribute the gfx target to one ordinal, so
    /// it is never treated as the unified-memory case even when the target
    /// looks like an APU family.
    #[test]
    fn classify_accelerator_memory_treats_multi_gpu_as_dedicated_even_with_apu_gfx_target() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 1024,
                total_mb: 4096,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 2048,
                total_mb: 8192,
            },
        ];
        let result = classify_accelerator_memory(Some(&vram), Some("gfx1151"), true, None);
        assert_eq!(result, AcceleratorMemory::Dedicated(8.0));
    }

    /// The defect a reviewer caught: `rocm serve` pins exactly one GPU
    /// ordinal, never the sum of every card, so summing VRAM across GPUs can
    /// report a model READY when it would OOM on every individual card. On an
    /// 8x192 GiB host a sum of ~1536 GiB would clear even the largest curated
    /// recipe's minimum; the true per-card figure (192 GiB) is what the
    /// verdict must compare against. Were this still summing, the assertion
    /// below would see `Dedicated(1536.0)`, not `Dedicated(192.0)`.
    #[test]
    fn classify_accelerator_memory_takes_the_largest_card_not_the_sum() {
        let vram: Vec<GpuVramUsage> = (0..8)
            .map(|index| GpuVramUsage {
                index,
                used_mb: 0,
                total_mb: 192 * 1024,
            })
            .collect();
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        assert_eq!(
            result,
            AcceleratorMemory::Dedicated(192.0),
            "a homogeneous 8-GPU host must report one card's capacity, not the sum of all \
             eight, or a model that OOMs on every card reads READY: {result:?}"
        );
    }

    /// A `HIP_VISIBLE_DEVICES`-style mask hides a GPU from `rocm serve` the
    /// same way it hides one from auto-selection (see
    /// `select_auto_gpu_index`'s own mask handling) -- but `amd-smi` enumerates
    /// at the driver level and knows nothing about that mask, so its rows
    /// still include the hidden card. Narrowing by `usable_indices` is what
    /// keeps a masked-out GPU's capacity from inflating the verdict for a
    /// model that could never actually land on it.
    #[test]
    fn classify_accelerator_memory_ignores_a_gpu_masked_out_by_the_visibility_filter() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 0,
                total_mb: 192 * 1024,
            },
        ];
        // Only index 0 is visible: the 192 GiB card at index 1 is masked out.
        let result = classify_accelerator_memory(Some(&vram), Some("gfx942"), true, Some(&[0]));
        assert_eq!(
            result,
            AcceleratorMemory::Dedicated(8.0),
            "a masked-out GPU's VRAM must not count toward the verdict: {result:?}"
        );
    }

    /// The end-to-end proof, one layer above `classify_accelerator_memory`'s own
    /// unit tests: not just that the function picks the right number, but that
    /// picking the wrong one would have changed the verdict a user actually
    /// sees. Two 8 GiB cards sum to 16 GiB, comfortably above `qwen3.5-4b`'s 12
    /// GiB minimum -- a verdict built on the sum would read READY, and `rocm
    /// serve` pins exactly one ordinal, so that model would OOM on every card
    /// this host has. No single card clears 12 GiB, so the correct verdict is
    /// BLOCKED. Were `classify_accelerator_memory` still summing, this would
    /// observe `Ready`, not `Blocked`.
    #[test]
    fn a_host_whose_vram_sum_clears_the_minimum_but_no_single_card_does_is_blocked_not_ready() {
        let vram = [
            GpuVramUsage {
                index: 0,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
            GpuVramUsage {
                index: 1,
                used_mb: 0,
                total_mb: 8 * 1024,
            },
        ];
        let accelerator_memory =
            classify_accelerator_memory(Some(&vram), Some("gfx942"), true, None);
        // Premise check: if this is ever 16.0 (the sum) instead of 8.0 (the
        // largest single card), the scenario below no longer tests what it
        // claims to.
        assert_eq!(accelerator_memory, AcceleratorMemory::Dedicated(8.0));

        let host = HostFacts {
            accelerator_memory,
            system_ram_gib: Some(1024.0),
        };
        let registry = rocm_core::builtin_model_recipe_registry();
        let report = assess_model_for_host(
            "qwen3.5-4b",
            &ModelCatalogSource::Available(&registry),
            &host,
            None,
            None,
        );
        assert_eq!(
            report.verdict,
            rocm_core::model_readiness::ModelVerdict::Blocked,
            "qwen3.5-4b needs 12 GiB; two 8 GiB cards sum to 16 GiB (which would clear it) but \
             no single card does (8 GiB each) -- the verdict must be BLOCKED, not READY, or \
             `rocm serve` would pin one 8 GiB card to a model that does not fit on it: {report:#?}"
        );
    }

    /// `has_amd_gpu` is always `false` on WSL (the probes that set it are
    /// skipped there), so `host_accelerator_memory` must read `rocm_sees_gpu`
    /// instead on a WSL host -- and only a confirmed `Some(false)` counts as
    /// "no GPU"; `None` means rocminfo was absent, which is not evidence of
    /// anything, per the same reasoning `fix-wsl-6-host-driver-too-old` in
    /// `diagnose.rs` already uses.
    #[test]
    fn examination_reports_amd_gpu_reads_rocm_sees_gpu_on_wsl() {
        let wsl_examination = |rocm_sees_gpu: Option<bool>| rocm_core::Examination {
            is_wsl: true,
            wsl: Some(rocm_core::examine::WslFacts {
                rocm_sees_gpu,
                ..Default::default()
            }),
            ..rocm_core::Examination::default()
        };

        assert!(
            !examination_reports_amd_gpu(&wsl_examination(Some(false))),
            "rocminfo positively enumerating no device must read as no GPU"
        );
        assert!(
            examination_reports_amd_gpu(&wsl_examination(Some(true))),
            "rocminfo positively enumerating a device must read as a GPU present"
        );
        assert!(
            examination_reports_amd_gpu(&wsl_examination(None)),
            "rocminfo absent (the question went unasked) must not read as a confident no"
        );
    }

    /// Off WSL, the bare-metal probes are what set `has_amd_gpu`, so this must
    /// keep using it rather than a WSL-only signal that was never populated.
    #[test]
    fn examination_reports_amd_gpu_reads_has_amd_gpu_off_wsl() {
        let mut examination = rocm_core::Examination::default();
        assert!(!examination_reports_amd_gpu(&examination));
        examination.has_amd_gpu = true;
        assert!(examination_reports_amd_gpu(&examination));
    }
}
