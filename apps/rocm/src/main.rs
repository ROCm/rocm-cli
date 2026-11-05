// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod advised_commands;
/// Shared archive generator and filesystem oracle for the extraction property
/// tests in `comfyui` and `therock`.
#[cfg(all(test, unix))]
#[path = "../../../crates/rocm-core/src/archive_props.rs"]
mod archive_props;
mod automations;
mod bootstrap;
mod bridge_snapshot;
mod chat;
mod chat_cmd;
mod chat_host_facts;
mod cli_audit;
mod cli_progress;
mod cli_report;
mod comfyui;
mod config_cmd;
mod dash;
mod dash_seam;
// Unix-only: its generators plant symlinks with `std::os::unix::fs::symlink`.
#[cfg(all(test, unix))]
mod deletion_properties;
mod diagnose_cmd;
mod driver_install;
mod driver_plan;
mod endpoint_keys;
mod engine_install_env;
mod engine_ipc;
mod engine_runtime_select;
mod engines_cmd;
mod examine_cmd;
mod examine_report_text;
mod freeform_answer;
mod freeform_execution;
mod freeform_planner;
mod gpu_policy;
mod install_cmd;
mod logging;
mod managed_service_status;
mod model_fit;
mod prompt_text;
mod provider_keys;
mod providers;
mod remote;
mod rocm_argv;
mod runtime_adopt;
mod runtime_install_finalize;
mod runtime_registry;
mod runtime_services;
mod runtime_uninstall;
mod runtimes_cmd;
mod serve_cmd;
mod serve_summary;
mod service_hints;
mod service_launch;
mod service_records;
mod services_cmd;
mod services_text;
mod setup_cmd;
mod shell_quote;
mod storage;
mod system_packages;
#[cfg(test)]
mod test_support;
mod therock;
mod tool_policy;
mod uninstall;
mod uninstall_plan;
mod update_cmd;

// Per-command handlers mechanically relocated into modules — fns, plus the
// types (e.g. `ServeArgs`, driver-plan types) that moved with their cluster.
// Dispatch call sites stay byte-identical via these re-imports (upstream-sync
// mergeability); several helpers are re-imported for reasons other than
// dispatch: `prepare_freeform_execution`/`render_freeform_execution_header`
// because `execute_freeform_next_action` (itself only called from
// `run_freeform`, outside `dispatch()`) calls them. The engine runtime and
// env-root helpers (`engine_manages_own_runtime`,
// `runtime_manifest_for_selector` and the rest) are no longer re-imported at
// the crate root: they live in `engine_install_env.rs`, and each caller imports
// them from there. Several root-level chat/examine helpers call into
// `runtimes_cmd` directly; `current_runtime_manifest` stays re-imported here
// for other root-level commands outside `dispatch()` that call it directly.
// `select_runtime_manifest` is no longer needed here:
// `single_ready_runtime_key`/`resolve_runtime_selector_to_exact_key` moved out
// to `engine_runtime_select.rs` in ROCMAI-215 (Phase 6f), reaching it via that
// module's own grouped `use crate::runtimes_cmd::{...}` import instead.
// `update_cmd.rs`'s `apply_runtime_update`/`append_update_activate_summary`
// call into `runtimes_cmd` directly too, via their own imports rather than
// this re-export. `env_root_for_service`, `attach_background_stdio`,
// `detach_background_command`, `managed_engine_startup_failure_detail`,
// `managed_service_process_command`, and
// `ensure_public_service_has_endpoint_key` are no longer needed here either:
// their only caller outside `dispatch()`, `restart_internal_managed_service`
// (and the endpoint-key check inside it), moved out to `services_text.rs` in
// ROCMAI-209 (Phase 6f), reaching them via its own imports instead.
// `record_cli_audit_event`, `cli_lifecycle_log_path` and `sanitize_log_value`
// (from `cli_audit.rs`) are re-imported so other modules keep reaching them as
// `crate::<name>`; `main.rs` itself calls `record_cli_audit_event`.
// `therock_install_version_selector` (from `install_cmd.rs`) is re-imported for
// the same reason: `chat.rs` reaches it as `crate::therock_install_version_selector`.
// `build_codex_bridge_gpu_snapshot` and `builtin_codex_bridge_engine_inventory`
// (from `bridge_snapshot.rs`) are re-imported only so `chat.rs` keeps reaching
// them as `crate::<name>`; `build_codex_bridge_snapshot` is called from
// `dispatch()` as well.
// `DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS` (from `service_records.rs`) is imported
// for the `ServicesCommand::Prune` clap definition.
use crate::automations::{automations, render_daemon_text};
use crate::bridge_snapshot::{
    build_codex_bridge_gpu_snapshot, build_codex_bridge_snapshot,
    builtin_codex_bridge_engine_inventory,
};
use crate::chat_cmd::run_chat_command;
use crate::cli_audit::{cli_lifecycle_log_path, record_cli_audit_event, sanitize_log_value};
use crate::config_cmd::{config, render_config_text};
use crate::diagnose_cmd::{diagnose, fix};
use crate::engine_ipc::{run_builtin_engine_serve_http, run_builtin_engine_stdio};
use crate::engine_runtime_select::{daemon_run_argv, parse_engine_recipe_json_arg};
use crate::engines_cmd::engines;
use crate::examine_cmd::examine;
// `append_examine_runtime_state` has no caller in `main.rs` itself — it is
// re-imported only so `runtimes_cmd.rs`'s own test module can keep reaching it
// unqualified as `crate::append_examine_runtime_state`, its pre-extraction
// call-site form; gated to `#[cfg(test)]` so a non-test build doesn't carry an
// otherwise-unused import.
#[cfg(test)]
use crate::examine_report_text::append_examine_runtime_state;
use crate::examine_report_text::{
    format_bytes, render_engine_inventory_text_with_paths, render_examine_text,
    render_examine_text_with_paths, render_launch_summary,
    render_model_registry_text_with_context_and_host,
    render_model_registry_verbose_text_with_context_and_host,
};
use crate::freeform_answer::render_freeform_read_only_answer;
use crate::freeform_execution::{prepare_freeform_execution, render_freeform_execution_header};
use crate::gpu_policy::parse_gpu_indices_arg;
use crate::install_cmd::{install, therock_install_version_selector};
use crate::model_fit::{
    PlatformRuleOut, engine_ruled_out_by_platform, engine_ruled_out_by_platform_with,
};
use crate::runtimes_cmd::{current_runtime_manifest, runtimes};
use crate::serve_cmd::{ServeArgs, serve};
use crate::service_records::DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS;
// Reached only as `super::run_approved_service_action` from
// `advised_commands.rs`'s test, so a non-test build does not carry it.
#[cfg(test)]
use crate::services_cmd::run_approved_service_action;
use crate::services_cmd::services;
use crate::services_text::{
    render_internal_status_text, render_logs_browser_text, render_logs_text,
    render_service_logs_text, run_internal_sandbox_tool,
};
use crate::setup_cmd::{render_setup_status_text, setup};
use crate::uninstall::uninstall;
use crate::uninstall_plan::UninstallOptions;
use crate::update_cmd::update;

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_HOST, EndpointReadiness, PERMISSIONS_MODE_ASK,
    PERMISSIONS_MODE_FULL_ACCESS, RocmCliConfig, TELEMETRY_MODE_LOCAL, TELEMETRY_MODE_OFF,
    WatcherMode, interactive_terminal, uv_cache_source,
};
use rocm_engine_protocol::DevicePolicy;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(all(test, not(windows)))]
use std::process::Stdio;
#[cfg(all(test, not(windows)))]
use std::process::{Command as ProcessCommand, ExitStatus};
#[cfg(test)]
use std::sync::Mutex;

/// User-selectable serving engines, in the order shown in `--help` and shell
/// completions. Used as the clap `value_parser` for every user-facing `--engine`
/// argument so the possible values stay in sync across the CLI surface.
const SUPPORTED_ENGINES: [&str; 3] = ["lemonade", "vllm", "aim"];

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
        /// opened. The mail goes from your own mail account, so it carries
        /// your address; the report itself names no user, host, or file path.
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
        /// Without it, a fix that changes files or settings asks `[y/N]`
        /// (default: no); when stdin is not a terminal it prints the plan and
        /// refuses with exit 5, changing nothing. Answering no also exits 5.
        /// Fixes that only print steps never ask.
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
    ///
    /// The default list is the curated recommendations. The AMD Inference Microservices
    /// (AIM) catalog of pre-built Docker images is not part of it: use --verbose to see
    /// every AIM image with the GPU families it supports, how many GPUs it needs, and
    /// whether it requires a Hugging Face token. Serve one with
    /// `rocm serve <model> --engine aim` (needs a reachable Docker daemon).
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
    /// (status, endpoint, model, and smoke-test metrics); use --verbose to stream engine
    /// logs in this terminal instead. When streaming, press Ctrl-D to detach and leave the
    /// server running, or Ctrl-C to stop it. If that stop answers that it could not confirm
    /// the server exited, it says so on stderr and exits with an error; if the stop itself
    /// fails, it says why on stderr, signals the server process directly and exits 0,
    /// noting on stderr if that signal fails too. Inspect or stop servers later with
    /// `rocm services`.
    ///
    /// Under --device gpu_required (the default) this command refuses on a host with no
    /// usable AMD GPU. Reusing an already-running managed server is the one exception: it
    /// launches nothing, so it is allowed through. The exception applies only when the
    /// server would actually be reused: same engine and model, matching recipe settings,
    /// a key already present if --require-api-key is asked for, and a device policy at
    /// least as strong as this one. A live server for a different engine, or for a model
    /// whose name does not overlap, does not soften the refusal, and nothing is prepared
    /// before it. A same-model server this run turns out not to be able to reuse does not
    /// soften it either, but engine preparation has already run by the time that is known.
    /// Model names are matched leniently, so a live server whose name overlaps the one you
    /// asked for (`qwen` for a running `qwen3-8b-instruct`) does let engine preparation run
    /// before the refusal.
    /// With an explicit --gpu the index is still validated after the exception, so a
    /// pinned GPU that is not visible is refused even when a reusable server exists.
    ///
    /// When a vLLM server THIS command launched fails to become ready and its engine log
    /// shows an out-of-memory failure, the deployment summary adds a note naming
    /// --gpu-memory-utilization and --gpu, warning that lowering the reservation will not
    /// help if the model does not fit, and printing a `rocm diagnose --symptom` command
    /// for your own failing line -- or, when that line carries an apostrophe or terminal
    /// control bytes and so cannot be rendered as one intact single-quoted argument, for a
    /// canonical vLLM out-of-memory symptom instead, so the printed command is always
    /// runnable and always reports a cause. That note belongs to the interactive summary only: a
    /// piped or redirected run prints the plain machine-readable form, which carries no
    /// post-launch notes (it still prints the pre-launch planning ones), and --verbose
    /// streams the traceback itself. It is never shown for a healthy or
    /// still-loading server, for a non-memory failure, or for a run that reused an
    /// already-running server rather than launching one; read that server's own log with
    /// `rocm services logs <service-id>`.
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
        /// index like `1`. An index the machine does not have is refused, and
        /// the error lists the indices that exist. If the GPUs cannot be listed,
        /// the index is used with a warning that it was not checked. If a
        /// visibility mask cannot be applied (`HIP_VISIBLE_DEVICES` cannot be
        /// read as ordinals, or no GPU list was available to apply it to), an
        /// index at or past the GPU count is still refused, and a lower one is
        /// used with a warning that the mask was not checked.
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
        /// Extra environment variable for the engine's container (repeatable).
        /// Engine-specific; currently only `--engine aim` reads these (e.g.
        /// `AIM_PROFILE_ID=...`) and other engines ignore it. `KEY=VALUE` is
        /// recorded with the service, so for a secret such as `HF_TOKEN` pass a
        /// bare `KEY` instead: it is forwarded from your environment and never
        /// stored. A gated AIM image (one that needs a Hugging Face token) is
        /// refused up front unless `HF_TOKEN` is passed this way.
        #[arg(long = "engine-env", value_name = "KEY[=VALUE]")]
        engine_env: Vec<String>,
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
    /// Run the background supervisor that runs enabled automation watchers.
    ///
    /// The daemon runs enabled automation watchers, and only while at least one
    /// is enabled (all are off by default). While running it:
    ///   - executes the enabled watchers (update checks, driver-plan checks,
    ///     artifact prefetch) on a 5s tick in a sandboxed subprocess
    ///   - health-checks and auto-recovers managed local model servers
    ///     (Lemonade, vLLM) when the `server-recover` watcher is enabled
    ///   - records GPU metrics roughly every 60s when the `gpu-metrics` or
    ///     `gpu-thermal-protect` watcher is enabled
    ///   - opens no webhook listener: that exists only when `rocmd run` is
    ///     started directly with `--local-webhook-port`, which `rocm daemon`
    ///     does not pass
    ///
    /// The dashboard does not need it: `rocm dash` has its own embedded daemon.
    ///
    /// It is started on demand by `rocm automations enable`, and by
    /// `rocm serve --managed` when an automation watcher is enabled, so you
    /// rarely need to launch it yourself. It exits on its own once no watcher
    /// is enabled: `rocm automations disable <watcher>` for each enabled one
    /// stops it. Run it directly only when you want to observe its behavior in
    /// the foreground, e.g. for debugging.
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
        /// Do not ask for interactive confirmation. Without it, the command
        /// asks `[y/N]` (default: no) in an interactive terminal; outside one,
        /// when there is something to remove, it prints the plan and refuses
        /// with exit 1, removing nothing.
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
    /// Preview or install Linux AMD driver support. On Windows it installs
    /// nothing: it reports the AMD display driver version and date and whether
    /// the driver is old.
    Driver {
        /// Use the DKMS driver wrapper flow.
        #[arg(long)]
        dkms: bool,
        /// Apply the plan. This command never asks for confirmation: without
        /// --yes it prints the plan and changes nothing, in a terminal or not.
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
        /// Approve the system-package installs vLLM needs (OpenMPI, libatomic,
        /// libnuma), including a sudo password prompt. This command never asks
        /// for confirmation. Without --yes, a Linux host with root or
        /// passwordless sudo installs a missing package anyway, and one that
        /// would need a password prints the commands to run while the install
        /// continues. With --yes, a failed OpenMPI install is an error; a
        /// failed libatomic or libnuma install only warns. On Windows these
        /// installs are skipped.
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
        /// Approve restarting running local servers. Pass it with
        /// --restart-services, which is refused without it, in a terminal or
        /// not; this command never prompts.
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
        /// Approve restarting running local servers. Pass it with
        /// --restart-services, which is refused without it, in a terminal or
        /// not; this command never prompts.
        #[arg(long, requires = "restart_services")]
        yes: bool,
    },
    /// Remove a ROCm runtime from ROCm CLI.
    #[command(alias = "remove")]
    Uninstall {
        /// Runtime key or friendly runtime selector.
        runtime: String,
        /// Do not ask for interactive confirmation. Without it, the command
        /// asks `[y/N]` (default: no) in an interactive terminal; outside one
        /// it prints the plan and refuses with exit 1, removing nothing.
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
        /// Do not ask for confirmation. Without it, the command asks `[y/N]`
        /// (default: no) in an interactive terminal; outside one, when there
        /// is something to remove, it prints the plan and refuses with exit 1,
        /// removing nothing.
        #[arg(long)]
        yes: bool,
    },
    /// Remove downloaded files that ROCm CLI can fetch again.
    #[command(name = "remove-downloads", alias = "remove-downloaded-files")]
    RemoveDownloads {
        /// Show what would happen without changing files.
        #[arg(long)]
        dry_run: bool,
        /// Do not ask for confirmation. Without it, the command asks `[y/N]`
        /// (default: no) in an interactive terminal; outside one, when there
        /// is something to remove, it prints the plan and refuses with exit 1,
        /// removing nothing.
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
    ///
    /// Reports the server stopped only once every process it recorded is
    /// confirmed gone. Otherwise it reports on stderr that the stop was not
    /// confirmed and exits with an error, keeping the record, which still names
    /// the processes, and any endpoint key, which the server may still be
    /// enforcing. It names each process that did not exit, and for one it could
    /// not verify it signals nothing and names an operating-system command to
    /// check on it. It does not suggest running the stop again.
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
    /// Canonical policy string understood by [`gpu_policy::parse_device_policy`]. Routing the
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
        execute_freeform_next_action(plan)?;
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

/// Takes the `StructuredRequestPlan` the caller already built (and already
/// rendered to the operator) and dispatches its next action — never rebuilds
/// it. A second, independent build here could call out to a configured cloud
/// `planner_provider` a second time; since that call is not guaranteed
/// deterministic, the plan the operator saw and the plan actually dispatched
/// could silently diverge. Reusing the one `plan` the caller already has
/// closes that gap.
fn execute_freeform_next_action(plan: freeform_planner::StructuredRequestPlan) -> Result<()> {
    let execution = prepare_freeform_execution(plan)?;
    print!("{}", render_freeform_execution_header(&execution));

    let mut argv = vec!["rocm".to_owned()];
    argv.extend(execution.action.args);
    let cli = Cli::try_parse_from(argv)?;
    dispatch(cli)
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
        }) => run_chat_command(
            provider,
            model,
            prompt,
            temperature,
            top_p,
            max_tokens,
            tools,
            chat_mock,
        ),
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
        }) => update(apply, runtime, activate, dry_run, json, timeout_secs),
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
            engine_env,
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
            engine_env,
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

/// What the user asked to do with an engine the platform gate may rule out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GatedEngineAction {
    Serve,
    Install,
}

/// Refuse `serve`, `engines install` and `engines shell` (which installs the
/// engine itself when its environment is missing) up front when the platform gate rules
/// the engine out on this host.
///
/// Takes the same decision `rocm engines list` reports as
/// `runtime: unsupported_native_windows`, and runs before runtime resolution,
/// so a native-Windows vLLM request gets the platform reason rather than an
/// unrelated "no active ROCm runtime" error, a network fetch, or a launched
/// service that dies with no output.
fn ensure_engine_allowed_on_platform(engine: &str, action: GatedEngineAction) -> Result<()> {
    refuse_if_ruled_out(engine_ruled_out_by_platform(engine), action)
}

/// The part of [`ensure_engine_allowed_on_platform`] that needs no Docker probe:
/// treats the daemon as reachable, so only the OS gate can refuse. `serve` runs
/// it first and the full gate after its own AIM pre-flight, so a request that
/// pre-flight refuses never reaches `docker info`.
fn ensure_engine_allowed_on_os(engine: &str, action: GatedEngineAction) -> Result<()> {
    refuse_if_ruled_out(engine_ruled_out_by_platform_with(engine, || true), action)
}

/// [`ensure_engine_allowed_on_platform`] with the platform decision passed in,
/// so the refusing branch is testable on any host.
fn refuse_if_ruled_out(ruled_out: PlatformRuleOut, action: GatedEngineAction) -> Result<()> {
    match platform_refusal(ruled_out, action) {
        Some(refusal) => bail!("{refusal}"),
        None => Ok(()),
    }
}

/// The refusal message for an engine, given why the platform gate already
/// ruled it out.
///
/// `ruled_out` is passed in rather than re-derived, as in
/// [`unsupported_here_for`], so both branches are unit-testable on any host.
/// Only native-Windows vLLM is refused here. The AIM conditions are left to
/// the AIM adapter, which checks Docker and GPU passthrough itself when it
/// launches; refusing them up front would be a separate behavior change.
fn platform_refusal(ruled_out: PlatformRuleOut, action: GatedEngineAction) -> Option<String> {
    match ruled_out {
        PlatformRuleOut::NativeWindowsVllm => Some(match action {
            GatedEngineAction::Serve => "vLLM is not supported on native Windows; serve it from \
                 WSL or Linux, or use the Lemonade engine (`--engine lemonade`)"
                .to_owned(),
            GatedEngineAction::Install => "vLLM is not supported on native Windows; install it \
                 from WSL or Linux, or use the Lemonade engine (`rocm engines install lemonade`)"
                .to_owned(),
        }),
        PlatformRuleOut::Allowed
        | PlatformRuleOut::NativeWindowsAim
        | PlatformRuleOut::AimDockerUnavailable => None,
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

/// Shown after any activation (direct `runtimes activate`, `update --apply
/// --activate`, or an `install sdk` that switches the active runtime) that
/// recorded a previous runtime, so every path that reaches this state gives
/// the same advice — see `RuntimesCommand::Activate` in `runtimes()`,
/// `append_update_activate_summary`, and `render_sdk_install_success` for the
/// call sites.
const ROLLBACK_RECOVERY_HINT: &str = "  next step: if this causes problems, run `rocm runtimes rollback` \
(no history - undoes only this one activation)";

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
        (
            "aim",
            "AMD Inference Microservices (AIM) pre-built Docker serving containers",
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

const fn missing_packaged_engine_reason(_engine: &str) -> Option<String> {
    None
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

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
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
    /// The dashboard's review and the CLI give the same sender notice, word for
    /// word. The dashboard shows it above the report before Approve; the CLI
    /// prints it under the report, before the mail link. `rocm-dash-tui`
    /// cannot depend on `rocm-core`, so it restates the text; this crate links
    /// both and holds them together.
    #[test]
    fn the_dashboard_approval_and_the_cli_give_the_same_sender_notice() {
        assert_eq!(
            rocm_dash_tui::ui::examine_manager::REPORT_SENDER_NOTICE,
            rocm_core::report_delivery::SENDER_NOTICE
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

    use super::*;

    use rocm_core::ManagedServiceRecord;

    use crate::bridge_snapshot::builtin_engine_inventory;
    use crate::runtime_install_finalize::tests::finish_sdk_install_with;
    use crate::services_text::load_managed_service;

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

        /// Clear the AMD driver-version override for the duration of the test, so
        /// a value exported in the developer's or runner's shell cannot leak into
        /// a plan and make its resolved `repo_version_expr` disagree with the
        /// assertion.
        pub(crate) fn with_amd_overrides_cleared() -> Self {
            let mut env = Self::new();
            env.clear("ROCM_CLI_AMDGPU_DRIVER_VERSION");
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
    fn yes_help_says_what_happens_without_it_outside_a_terminal() {
        // Wording pin: the sentences are only worth having while they are true,
        // and the behavior itself is proven elsewhere —
        //   uninstall              uninstall-04
        //   runtimes uninstall     runtime-lifecycle-04
        //   storage remove-*       storage-01, `removal_requires_yes_when_there_is_no_terminal_to_prompt`
        //   fix                    diagnose-08, `consent_without_prompt` unit tests
        //   install driver         driver-install-02
        //   engines install        `system_package_install_action` unit tests
        //   runtimes activate and rollback   runtime-lifecycle-11 and -14
        //                          (--restart-services without --yes), -17 (--yes without it)
        // Each entry is read from the `--yes` argument's own help, not the whole
        // page, so a sibling flag cannot satisfy it.
        // The trailing comma keeps `exit 1` from matching `exit 10`. Everything
        // refuses with 1 except `fix`, whose exit 5 is its own declined status.
        const REFUSES: &[&str] = &["(default: no)", "refuses with exit 1,"];
        const NEVER_ASKS: &[&str] = &["never asks for confirmation"];
        const NEVER_PROMPTS: &[&str] = &["never prompts"];
        let cases: &[(&[&str], &[&str])] = &[
            (&["uninstall"], REFUSES),
            (&["runtimes", "uninstall"], REFUSES),
            (&["storage", "remove-old-installs"], REFUSES),
            (&["storage", "remove-downloads"], REFUSES),
            (
                &["fix"],
                &["(default: no)", "refuses with exit 5,", "also exits 5"],
            ),
            (&["install", "driver"], NEVER_ASKS),
            (
                &["engines", "install"],
                &[
                    "never asks for confirmation",
                    // The least obvious sentence: no `--yes` still installs.
                    "installs a missing package anyway",
                    // Only OpenMPI's failure is an error; libatomic/libnuma warn.
                    "only warns",
                ],
            ),
            (&["runtimes", "activate"], NEVER_PROMPTS),
            (&["runtimes", "rollback"], NEVER_PROMPTS),
        ];
        for (path, expected) in cases {
            let mut command = Cli::command();
            let mut subcommand = &mut command;
            for name in *path {
                subcommand = subcommand.find_subcommand_mut(name).unwrap_or_else(|| {
                    panic!("no `{name}` subcommand on `rocm {}`", path.join(" "))
                });
            }
            let yes = subcommand
                .get_arguments()
                .find(|arg| arg.get_id() == "yes")
                .unwrap_or_else(|| panic!("`rocm {}` must offer --yes", path.join(" ")));
            let yes_help = yes
                .get_long_help()
                .or_else(|| yes.get_help())
                .unwrap_or_else(|| panic!("`rocm {} --yes` must be documented", path.join(" ")))
                .to_string();
            for phrase in *expected {
                assert!(
                    yes_help.contains(phrase),
                    "`rocm {} --yes` help must contain `{phrase}`:\n{yes_help}",
                    path.join(" ")
                );
            }
        }
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
        // `tool_policy::tests::runtimes_activate_approval_modal_uses_the_runtime_noun`.
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

    pub(crate) fn test_app_paths() -> AppPaths {
        AppPaths {
            config_dir: PathBuf::from("C:/Users/test/.rocm"),
            data_dir: PathBuf::from("D:/rocm-data"),
            cache_dir: PathBuf::from("D:/rocm-data/cache"),
        }
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

    /// Plant a managed-service record plus every artifact it owns: the log and
    /// endpoint key beside the manifest in `services_dir`, and the engine state
    /// file under `<data>/engines/<engine>/state/`.
    pub(crate) fn plant_service_record(
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
    pub(crate) fn plant_service_record_without_its_key(
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

        let actions: Vec<String> = rocm_core::load_recent_audit_events(&paths, 10)?
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
    fn missing_packaged_engine_reason_has_no_deferred_first_party_engines() {
        assert!(missing_packaged_engine_reason("lemonade").is_none());
        assert!(missing_packaged_engine_reason("vllm").is_none());
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

    /// Return the brace-balanced block that follows `opener` (which must end in
    /// `{`), including the contents but excluding the trailing brace's tail.
    pub(crate) fn slice_braced_block(src: &str, opener: &str) -> String {
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
    pub(crate) fn strip_line_comments(block: &str) -> String {
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

    /// Native-Windows vLLM is refused for both serve and install, and each
    /// refusal names the way out for that action. Wording only: that the
    /// refusal fires before anything is launched or installed is proven by the
    /// binary-level tests in `tests/vllm_native_windows_gate.rs` and the
    /// `serve-vllm-refused-on-native-windows` /
    /// `serve-vllm-install-refused-on-native-windows` scenarios.
    #[test]
    fn platform_refusal_names_wsl_linux_and_lemonade_for_native_windows_vllm() {
        let serve = platform_refusal(PlatformRuleOut::NativeWindowsVllm, GatedEngineAction::Serve)
            .expect("native-Windows vLLM serve is refused");
        let install = platform_refusal(
            PlatformRuleOut::NativeWindowsVllm,
            GatedEngineAction::Install,
        )
        .expect("native-Windows vLLM install is refused");
        for message in [&serve, &install] {
            assert!(
                message.contains("vLLM is not supported on native Windows")
                    && message.contains("WSL or Linux")
                    && message.contains("Lemonade"),
                "{message:?}"
            );
        }
        assert!(serve.contains("`--engine lemonade`"), "{serve:?}");
        assert!(
            install.contains("`rocm engines install lemonade`"),
            "{install:?}"
        );
    }

    /// Only native-Windows vLLM is refused up front; an allowed engine, and
    /// the AIM conditions left to its own adapter, pass through.
    #[test]
    fn platform_refusal_is_none_unless_native_windows_vllm() {
        for ruled_out in [
            PlatformRuleOut::Allowed,
            PlatformRuleOut::NativeWindowsAim,
            PlatformRuleOut::AimDockerUnavailable,
        ] {
            for action in [GatedEngineAction::Serve, GatedEngineAction::Install] {
                assert_eq!(platform_refusal(ruled_out, action), None, "{ruled_out:?}");
            }
        }
    }

    /// The refusing branch of the gate, with an injected decision so it is
    /// exercised on Linux CI too (the host-decision test below is vacuous there
    /// for vLLM): native-Windows vLLM becomes an error carrying the action's
    /// message, and everything else passes.
    #[test]
    fn refuse_if_ruled_out_errors_with_the_refusal_on_any_host() {
        for action in [GatedEngineAction::Serve, GatedEngineAction::Install] {
            let error = refuse_if_ruled_out(PlatformRuleOut::NativeWindowsVllm, action)
                .expect_err("native-Windows vLLM must be refused");
            assert_eq!(
                error.to_string(),
                platform_refusal(PlatformRuleOut::NativeWindowsVllm, action)
                    .expect("native-Windows vLLM has a refusal")
            );
            assert!(refuse_if_ruled_out(PlatformRuleOut::Allowed, action).is_ok());
        }
    }

    /// The gate takes the host's own decision: vLLM is refused on native
    /// Windows and allowed everywhere else, and Lemonade is never refused.
    #[test]
    fn ensure_engine_allowed_on_platform_follows_the_host_decision() {
        for action in [GatedEngineAction::Serve, GatedEngineAction::Install] {
            assert_eq!(
                ensure_engine_allowed_on_platform("vllm", action).is_err(),
                rocm_core::runtime_is_windows()
            );
            assert_eq!(
                ensure_engine_allowed_on_platform("VLLM", action).is_err(),
                rocm_core::runtime_is_windows()
            );
            assert!(ensure_engine_allowed_on_platform("lemonade", action).is_ok());
        }
    }
}
