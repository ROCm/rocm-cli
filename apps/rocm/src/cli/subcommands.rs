// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The verb lists nested under `rocm`'s top-level commands: the
//! `#[derive(Subcommand)]` enums for `bench`, `install`, `engines`,
//! `runtimes`, `storage`, `comfyui`, `services`, `automations`, `config` and
//! `setup`.
//!
//! A pure relocation out of [`super`], which keeps `Cli` and the root
//! `Command` these verbs hang off, and re-exports everything here so every
//! caller still reaches them as `crate::ServicesCommand` and so on.
//! Declaration order is unchanged, because it sets the order of each
//! command's help list.

use super::SUPPORTED_ENGINES;
use crate::cli_args::{
    InstallFormat, PermissionsModeArg, Provider, TelemetryModeArg, WatcherModeArg,
};
use crate::service_records::DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS;
use crate::{comfyui, storage};
use clap::Subcommand;
use std::path::PathBuf;

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
rocm install sdk --channel nightly --build-date 2026-08-19\n  \
rocm install sdk --family gfx1100 --dry-run\n  \
rocm install sdk --version 7.14.1 --family gfx110X-all\n  \
rocm install sdk --devel\n  \
rocm install sdk --version 10.0.0 --python-version 3.12")]
    Sdk {
        /// TheRock ROCm SDK channel to install from: `release` or `nightly`.
        ///
        /// This picks the ROCm SDK stream only. It is separate from the channel
        /// the `rocm` binary itself was installed from (`install.sh`,
        /// `install.ps1`); `rocm update` updates SDK runtimes, not the binary.
        #[arg(long, default_value = "release")]
        channel: String,
        /// Package format to install [possible values: wheel, tarball].
        #[arg(long, default_value = "wheel")]
        format: InstallFormat,
        /// Full folder path where the ROCm Python environment should be created.
        #[arg(long)]
        prefix: Option<std::path::PathBuf>,
        /// Exact TheRock package version to install; with none, the channel's
        /// newest build. An earlier ROCm 7 release such as 7.14.1 is supported,
        /// for a wheel install only.
        #[arg(long, conflicts_with = "build_date")]
        version: Option<String>,
        /// Pick the TheRock package built on this date; nightlies from before
        /// 2026-08-23 are still installable, for a wheel install only.
        #[arg(long, value_name = "YYYY-MM-DD", conflicts_with = "version")]
        build_date: Option<String>,
        /// TheRock GPU package family to install, such as gfx1200.
        ///
        /// ROCm 10 and newer, which an install with no `--version` gets, need
        /// an exact arch such as `gfx1200`; a label such as gfx110X-all is for
        /// an earlier ROCm 7 (wheel installs only).
        ///
        /// A GPU target ROCm publishes no build for is refused rather than
        /// mapped to a near neighbour: `--family gfx90c` - the Vega iGPU in
        /// Renoir, Lucienne, Cezanne and Barcelo APUs - fails and installs
        /// nothing, instead of substituting another family. A plain `rocm
        /// install sdk` reports the same refusal when it detects such a GPU.
        #[arg(long)]
        family: Option<String>,
        /// Python version for a ROCm 10 or newer install, one of 3.10 to 3.14.
        ///
        /// ROCm 10 installs use Python 3.14 unless you name another, because
        /// vLLM, flash-attn and amd-aiter for ROCm 10 are published for 3.14
        /// only. With another version the SDK and PyTorch install and run, but
        /// the vLLM engine does not: the engine install that follows an SDK
        /// install on a vLLM GPU is skipped, and `rocm engines install vllm`
        /// refuses that runtime. Re-run without this flag to switch it to 3.14.
        /// Wheel installs only; a ROCm 7 install is built for Python 3.12.
        #[arg(long, value_name = "X.Y")]
        python_version: Option<String>,
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
        ///
        /// Downloads no uv and prepares no Python. For ROCm 10 and newer on a
        /// host with no uv installed, the plan marks the package versions it
        /// would need uv to choose, and everything made from them, as not
        /// resolved.
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
    ///
    /// Each runtime is listed with its version (and build date), GPU family,
    /// mode and toolchain, then its channel, package source, source layout
    /// generation and exact GPU target as its manifest recorded them. A fact
    /// recorded nowhere, or recorded blank, is shown as `unrecorded`; nothing
    /// is guessed from other facts.
    List {
        /// Print the runtime list as JSON instead of text.
        #[arg(long)]
        json: bool,
    },
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
        after_help = "NOTE: rollback remembers only the runtime you just left, so it cannot \
undo more than one activation."
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
        /// Python environment folder. Taken from --python when that sits in a bin or Scripts folder; required otherwise.
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
        /// is replaced, unless rocm-cli's saved record of it cannot be parsed, which it then says;
        /// a record that cannot be read at all (a directory at its path, permission denied) is an
        /// error naming the file.
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
    ///
    /// If rocm-cli's saved record of the ComfyUI it started cannot be parsed, it
    /// stops no process, since the record names none, and checks only the
    /// default address: if nothing answers there it moves the record aside to
    /// `running.json.unreadable`; if something does, it keeps the record and
    /// exits non-zero. A record that cannot be read at all (a directory at its
    /// path, permission denied) is an error naming the file.
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
    ///
    /// In the default table, each running server is asked which backend and GPU
    /// it is actually on, so this takes a few seconds rather than returning
    /// instantly. The asking is capped in total (about 18 seconds worst case,
    /// however many servers are listed), and a server that does not answer in
    /// time simply has no device line. `--json` asks nothing and carries no
    /// device field: it is the stored records, returned as fast as before.
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
        /// Keep printing new output until the server stops (Ctrl-C ends the follow and
        /// leaves the server running). Use it to watch a model download.
        #[arg(long, short = 'f')]
        follow: bool,
    },
    /// Stop a local model server.
    ///
    /// Reports the server stopped only once every process it recorded is
    /// confirmed gone. Otherwise it reports on stderr that the stop was not
    /// confirmed and exits with an error, keeping the record, which still names
    /// the processes, and any endpoint key, which the server may still be
    /// enforcing. It names each process that did not exit (on Windows, also one
    /// whose child processes may still be running), and for one it could not
    /// verify it signals nothing and names an operating-system command to check
    /// on it. It does not suggest running the stop again.
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
    /// started and the stop request stays recorded. A server that dies during
    /// the restart fails loudly: its exit status and a log pointer are printed,
    /// and the record is marked `failed`.
    Restart {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; restart refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Delete one local server record and its files.
    ///
    /// Only works on a record that is not running: stop the server first (on
    /// Windows a process `rocm` cannot query counts as running). The record's
    /// details file, log, engine state file, and endpoint key file (with any
    /// copy of the key an interrupted save left beside it) are all deleted, so
    /// its log can no longer be read and it can no longer be restarted.
    Remove {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; remove refuses without it.
        #[arg(long)]
        yes: bool,
        /// Remove a record whose recorded process `rocm` cannot query.
        ///
        /// Only when a stop was requested, no recorded process is confirmed
        /// running, and at least one could not be queried (on Windows, a live
        /// process the account may not open). Deletes the same files as a
        /// plain `remove`, and says which process; if it is still running it
        /// keeps running. Refused for a process confirmed running.
        #[arg(long)]
        force: bool,
    },
    /// Delete local server records that are no longer running.
    ///
    /// Running servers are always left alone; a record held only by a process
    /// `rocm` cannot query (Windows) is listed with the `remove --force` command.
    /// Leftover files whose record is already gone are cleaned up too.
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
