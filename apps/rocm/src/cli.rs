// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm` command line definition: the top-level `Cli` and `Command`, the
//! subcommand enums nested under it, and the clap value enums behind the
//! `--provider`, `--device`, `--framework` and similar arguments.
//!
//! Mechanically relocated from `main.rs` with no behavior change: attributes,
//! help text, defaults, aliases and declaration order (which is help order) are
//! unchanged. `main.rs` keeps the dispatch over `Command`, the process entry and
//! the completion builder, and re-imports these types so every other module
//! keeps reaching them as `crate::Command`, `crate::Provider` and so on.

use crate::service_records::DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS;
use crate::{bootstrap, chat, comfyui, remote, storage};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{
    DEFAULT_LOCAL_HOST, PERMISSIONS_MODE_ASK, PERMISSIONS_MODE_FULL_ACCESS, TELEMETRY_MODE_LOCAL,
    TELEMETRY_MODE_OFF, WatcherMode,
};
use std::path::PathBuf;

/// User-selectable serving engines, in the order shown in `--help` and shell
/// completions. Used as the clap `value_parser` for every user-facing `--engine`
/// argument so the possible values stay in sync across the CLI surface.
pub(crate) const SUPPORTED_ENGINES: [&str; 3] = ["lemonade", "vllm", "aim"];

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
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
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
pub(crate) enum BenchCommand {
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
pub(crate) enum InstallTarget {
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
pub(crate) enum EnginesCommand {
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
pub(crate) enum RuntimesCommand {
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
pub(crate) enum StorageCommand {
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
pub(crate) enum ComfyuiCommand {
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
pub(crate) enum ServicesCommand {
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
pub(crate) enum AutomationsCommand {
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
pub(crate) enum ConfigCommand {
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
pub(crate) enum SetupCommand {
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
pub(crate) enum FrameworkArg {
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
pub(crate) enum InstallFormat {
    /// Python wheel packages (recommended; smaller, pip-installable).
    Wheel,
    /// Self-contained tarball archive of the ROCm SDK.
    Tarball,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Provider {
    /// A local OpenAI-compatible server started via `rocm serve` (no API key).
    Local,
    /// Anthropic Claude API (requires an API key).
    Anthropic,
    /// OpenAI API or OpenAI-compatible endpoint (requires an API key).
    Openai,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum WatcherModeArg {
    /// Only report findings; never take action.
    Observe,
    /// Report findings and propose changes for you to approve.
    Propose,
    /// Apply changes automatically within a contained, reversible scope.
    Contained,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TelemetryModeArg {
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
pub(crate) enum DevicePolicyArg {
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
    /// Canonical policy string understood by [`crate::gpu_policy::parse_device_policy`]. Routing the
    /// parsed variant back through the string keeps `parse_device_policy` the
    /// single choke point that enforces the `cpu_only` rejection across every
    /// caller, rather than adding a second conversion path that could bypass it.
    pub(crate) const fn as_policy_str(self) -> &'static str {
        match self {
            Self::GpuRequired => "gpu_required",
            Self::GpuPreferred => "gpu_preferred",
            Self::CpuOnly => "cpu_only",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub(crate) enum SandboxToolArg {
    ListServers,
    RestartServer,
    StopServer,
}

impl SandboxToolArg {
    pub(crate) const fn as_cli_value(self) -> &'static str {
        match self {
            Self::ListServers => "list_servers",
            Self::RestartServer => "restart_server",
            Self::StopServer => "stop_server",
        }
    }
}

impl TelemetryModeArg {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Local => TELEMETRY_MODE_LOCAL,
            Self::Off => TELEMETRY_MODE_OFF,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PermissionsModeArg {
    FullAccess,
    Ask,
}

impl PermissionsModeArg {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::FullAccess => PERMISSIONS_MODE_FULL_ACCESS,
            Self::Ask => PERMISSIONS_MODE_ASK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge_snapshot::builtin_engine_inventory;
    use crate::{cli_command, device_policy_name};
    use clap::{CommandFactory, Parser};
    use rocm_engine_protocol::DevicePolicy;

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
}
