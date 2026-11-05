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
mod cli;
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
mod provider_keys;
mod providers;
mod remote;
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
// The clap command tree and its value enums (from `cli.rs`) are re-imported so
// every module keeps reaching them as `crate::Command`, `crate::Provider` and so
// on; `main.rs` itself matches on `Command` in `dispatch()`.
use crate::automations::{automations, render_daemon_text};
use crate::bridge_snapshot::{
    build_codex_bridge_gpu_snapshot, build_codex_bridge_snapshot,
    builtin_codex_bridge_engine_inventory,
};
use crate::chat_cmd::run_chat_command;
use crate::cli::{
    AutomationsCommand, BenchCommand, Cli, ComfyuiCommand, Command, ConfigCommand, DevicePolicyArg,
    EnginesCommand, InstallFormat, InstallTarget, Provider, RuntimesCommand, SandboxToolArg,
    ServicesCommand, SetupCommand, StorageCommand,
};
// Reached only as `super::SUPPORTED_ENGINES` from `advised_commands.rs`'s test,
// so a non-test build does not carry it.
#[cfg(test)]
use crate::cli::SUPPORTED_ENGINES;
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
use clap::{CommandFactory, FromArgMatches, Parser};
use rocm_core::{
    AppPaths, EndpointReadiness, RocmCliConfig, interactive_terminal, uv_cache_source,
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

    use crate::cli::PermissionsModeArg;
    use clap::ValueEnum;
    use rocm_core::{ManagedServiceRecord, PERMISSIONS_MODE_ASK, PERMISSIONS_MODE_FULL_ACCESS};

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
