// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Spawn, attach to, and stream logs from a managed background service
//! process, plus the endpoint-key/auth checks that gate it.
//!
//! Relocated from `main.rs`, mechanically and with no behavior change at the
//! time; it has since gained behavior of its own — the engine spawn seam the
//! launch and restart share, the startup-death check on both platforms, and
//! the retirement of a failed launch's record. This module is the
//! managed-service-spawning tail that was deliberately deferred out of
//! the Phase 6a (ROCMAI-82) split: `start_managed_service`,
//! `run_attached_service`, `spawn_managed_engine_child`, `stream_attached_logs`,
//! and the rest of this file. It was entangled with crate-root launch helpers
//! — `record_cli_audit_event`, `existing_live_managed_service`, and the stop
//! action of the service-record-management cluster (this module reaches only
//! stop, via `AttachOutcome::Stop`; restart/remove/prune are not called here)
//! — and this later pass (ROCMAI-91, Phase 6b) separates the spawning tail out
//! from them; they stay at the crate root (see below).
//!
//! `dispatch()`'s call sites are unaffected: `Command::Services { command } =>
//! services(command)` doesn't call anything in this cluster directly, and
//! `ServicesCommand`/`services()` stay in `main.rs` (a dedicated clap
//! subcommand's enum and dispatch function usually do, per
//! `docs/architecture.md`'s convention note). `record_cli_audit_event`,
//! `existing_live_managed_service`, and the broader "service record management"
//! cluster (stop/restart/remove/prune, reached from `services()`; stop/restart
//! are also reachable from the chat/LLM sandbox tool-calling path) stay at the
//! crate root too — reached via `crate::`, same as other mechanically relocated
//! modules. Several functions here are `pub(crate)` specifically because
//! `main.rs`, `serve_cmd.rs`, and `automations.rs` all call into this cluster
//! across the new module boundary.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
#[cfg(not(windows))]
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, AutomationRuntimeState, EndpointReadiness, ManagedServiceRecord};
use rocm_core::{format_http_base_url, process_is_running};
use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};

#[cfg(not(windows))]
use crate::apply_app_path_env;
use crate::endpoint_keys;
use crate::engines_cmd::env_root_for_service;
use crate::read_optional_tail_lines;
use crate::service_hints;
use crate::{
    SandboxToolArg, builtin_engine_serve_http_args, device_policy_name,
    existing_live_managed_service, managed_service_launcher_path, record_cli_audit_event,
    render_service_action_result, run_internal_sandbox_tool, status_for_readiness,
    wait_for_service_http_ready_with_progress,
};
#[cfg(windows)]
use crate::{app_path_env_var_refs, app_path_env_var_values};

pub(crate) fn validate_bind_host(host: &str, allow_public_bind: bool) -> Result<()> {
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "`rocm serve --host {host}` is not loopback; pass `--allow-public-bind` before binding a non-local interface"
        );
    }
    Ok(())
}

/// Inverse of [`rocm_engine_protocol::is_public_bind_host`], which owns the
/// policy so `rocmd` classifies a recorded `host` identically when it respawns
/// the service.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    !rocm_engine_protocol::is_public_bind_host(host)
}

/// Resolve the API key that will guard this endpoint, applying the
/// loopback-vs-public policy for `rocm serve`.
///
/// - **Loopback host** → `None`: local serving stays credential-free (the
///   unchanged default). Any key supplied for a loopback bind is ignored and
///   nothing is persisted — loopback needs no auth.
/// - **Public host** → `Some(key)`: use the user-supplied key when present,
///   otherwise generate a strong random one so a public endpoint can never come
///   up anonymous. An empty/whitespace supplied key is rejected rather than
///   silently treated as "no auth".
/// - **`required`** → treat a loopback bind as public for this purpose.
///
/// That last case exists because "loopback" is a statement about the bind
/// address, not about who can reach the port. Publishing the port onto a
/// tailnet, proxying it, or mapping it out of a container all leave the bind
/// loopback while widening the audience — and the policy above would then hand
/// out an unauthenticated endpoint. Whoever widens the reach is responsible for
/// asking for the credential, so this is an explicit flag rather than a guess.
pub(crate) fn resolve_endpoint_auth(
    host: &str,
    supplied: Option<&str>,
    required: bool,
) -> Result<Option<String>> {
    if is_loopback_host(host) && !required {
        return Ok(None);
    }
    match supplied {
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                bail!(
                    "`rocm serve --api-key` (or ROCM_SERVE_API_KEY) was empty; a public \
                     endpoint must be protected by a non-empty API key"
                );
            }
            // The key is later interpolated verbatim into raw `Authorization:
            // Bearer {key}\r\n` header lines; reject a control character (e.g. an
            // embedded CR/LF) here so a crafted key cannot inject extra headers.
            if rocm_core::endpoint_api_key_has_forbidden_chars(trimmed) {
                bail!(
                    "`rocm serve --api-key` (or ROCM_SERVE_API_KEY) contained a control \
                     character such as a carriage return or newline; an endpoint API key \
                     must be a single line of printable characters"
                );
            }
            Ok(Some(trimmed.to_owned()))
        }
        None => Ok(Some(rocm_core::generate_endpoint_api_key())),
    }
}

/// When an equivalent managed service is already running, the endpoint key
/// serve() freshly stored for this attempt is unused — drop it so it is not
/// orphaned in storage. The already-running service keeps its own key.
/// Best-effort and idempotent; a loopback attempt (`freshly_stored == None`)
/// is a no-op.
pub(crate) fn drop_orphaned_endpoint_key_on_already_running(
    paths: &AppPaths,
    service_id: &str,
    freshly_stored: Option<&str>,
) {
    if freshly_stored.is_some() {
        endpoint_keys::clear_endpoint_api_key(paths, service_id);
    }
}

/// Reject engine/platform combinations that cannot enforce a public endpoint's
/// API key, so a public bind fails closed instead of coming up unauthenticated.
///
/// The one such case today: Windows managed Lemonade. Its server reads the
/// value-typed `LEMONADE_API_KEY` env var, but the Windows detached-spawn
/// primitive only carries path-valued env overrides, so the key never reaches it.
/// vLLM enforces auth on every platform (`VLLM_API_KEY`), and loopback binds
/// (`public_bind == false`) need no key — both pass. `is_windows` is a parameter
/// so both branches are unit-testable off-Windows.
pub(crate) fn ensure_public_bind_engine_supported(
    engine: &str,
    public_bind: bool,
    is_windows: bool,
) -> Result<()> {
    if public_bind && is_windows && engine == "lemonade" {
        bail!(
            "public binding with the lemonade engine is not supported on Windows: the endpoint \
             API key cannot be enforced there. Use `--engine vllm`, or bind a loopback host \
             (the default 127.0.0.1)."
        );
    }
    Ok(())
}

/// Refuse to (re)spawn a managed service recorded on a public host when its
/// endpoint API key is gone, so a respawn cannot reopen the endpoint anonymously.
///
/// `serve()` applies the loopback-vs-public policy once, via
/// [`resolve_endpoint_auth`], and persists the resulting key. Every later spawn
/// — `rocm services restart`, and `rocmd`'s recovery supervisor — reads that key
/// file back and would otherwise treat "no key file" as "no auth wanted",
/// silently downgrading a protected public endpoint to an open one. The real
/// invariant is a property of the *host*, not of the file: a non-loopback bind
/// must always be authenticated.
///
/// The gap is reachable through ordinary commands, because a stop deletes the
/// key file and `rocm services restart` accepts a stopped service id (its help
/// points at `rocm services list --all`).
///
/// `key_present` is a plain `bool` rather than a path so both branches are
/// unit-testable without touching the filesystem, mirroring `is_windows` in
/// [`ensure_public_bind_engine_supported`].
pub(crate) fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
) -> Result<()> {
    // Two ways a service can need a key. A public bind is the obvious one. The
    // other is a service that asked for auth on a loopback bind, because
    // something outside this process republishes the port — a tailnet publish
    // survives a reboot, let alone a restart, so "loopback" stops meaning
    // "only this machine" and the bind address can no longer be trusted to
    // answer the question on its own.
    if requires_api_key && !key_present {
        bail!(
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so restarting it would reopen it without authentication. Something outside this \
             machine may still be publishing its port. The key is dropped when a service stops \
             and cannot be recovered. Launch it again with \
             `rocm serve --require-api-key` (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, \
             to choose the key instead of generating one)."
        );
    }
    if rocm_engine_protocol::is_public_bind_host(host) && !key_present {
        bail!(
            "managed service is bound to the public host `{host}` but has no endpoint API key, \
             so restarting it would reopen it without authentication. The key is dropped when a \
             service stops and cannot be recovered. Launch it again with \
             `rocm serve --host {host} --allow-public-bind` (add `--api-key <key>`, or set \
             ROCM_SERVE_API_KEY, to choose the key instead of generating one)."
        );
    }
    Ok(())
}

#[cfg(not(windows))]
pub(crate) fn detach_background_command(command: &mut ProcessCommand) {
    rocm_core::detach_command_session(command);
}

#[cfg(not(windows))]
pub(crate) fn attach_background_stdio(
    command: &mut ProcessCommand,
    log_path: Option<&Path>,
) -> Result<()> {
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

/// How long a freshly spawned managed engine is watched before it is treated as
/// started. Long enough to catch an engine that dies on the way up — arguments it
/// rejects, a runtime it cannot load — and short enough to be invisible next to
/// the readiness wait that follows. Both platforms use the same budget so a launch
/// that fails at startup fails the same way on each.
///
/// A window, not a guarantee: engines bind their port and load the model later
/// than this, so a death there (a port already taken, a model that will not
/// load) passes the check, and the launch is reported as still starting.
pub(crate) const MANAGED_ENGINE_STARTUP_SETTLE: Duration = Duration::from_millis(200);

/// E2E-only switch that makes the managed engine die during startup, so the
/// black-box suite can assert what a user sees when it does.
///
/// Read in the *parent*, which is what spawns the engine, so nothing test-only
/// leaks into the child's own code path. The child is the real `rocm` binary,
/// really spawned and really dead — only its arguments are swapped for ones the
/// CLI rejects outright, so the launch takes exactly the path a broken engine
/// takes.
#[cfg(feature = "e2e-test-hooks")]
const MANAGED_ENGINE_STARTUP_FAILURE_TEST_ENV: &str = "ROCM_E2E_MANAGED_ENGINE_STARTUP_FAILURE";

/// Whether the E2E startup-failure switch is armed.
#[cfg(feature = "e2e-test-hooks")]
pub(crate) fn scripted_managed_engine_startup_failure() -> bool {
    std::env::var_os(MANAGED_ENGINE_STARTUP_FAILURE_TEST_ENV).is_some()
}

/// Release builds do not enable `e2e-test-hooks`, so the switch does not exist
/// there and no environment variable can arm it.
#[cfg(not(feature = "e2e-test-hooks"))]
pub(crate) const fn scripted_managed_engine_startup_failure() -> bool {
    false
}

/// Arguments that make the spawned `rocm` exit immediately with a non-zero
/// status. Used only under [`scripted_managed_engine_startup_failure`].
///
/// An unrecognised *flag*, deliberately: clap rejects it with exit code 2 before
/// any work. An unrecognised subcommand would not do — the CLI falls back to
/// natural-language request parsing for those and exits 0, which is a different
/// (and much slower) thing than an engine dying.
#[cfg(feature = "e2e-test-hooks")]
fn managed_engine_startup_failure_args() -> Vec<String> {
    vec!["--e2e-managed-engine-startup-failure".to_owned()]
}

/// Retire the service record of a launch that did not get an engine running.
///
/// Callers must reach this on *every* path that abandons a record, because a
/// record left in a non-terminal state keeps claiming its engine + model:
/// `managed_service_is_live` counts `"starting"` as alive, so the idempotency
/// guard would report the corpse as `AlreadyRunning` and refuse every later
/// `rocm serve` for the same pair. `"failed"` is outside the live set, so the
/// next launch proceeds.
///
/// On the launch path the record is persisted *before* the spawn, to claim the
/// GPU for concurrent auto-selection, so until a child is put on record it still
/// carries the constructor's `"starting"` status and a `supervisor_pid` of 0 —
/// and pid 0 is skipped by `recorded_service_pids`, so the liveness refresh
/// cannot demote it either and the wedge is permanent. That is the record every
/// failure before the spawn retires, and the one a death the spawn itself
/// observed retires too (see `watch_managed_engine_startup`). A death found by a
/// check still pending retires a record already naming the dead child, which
/// the liveness refresh would eventually demote; retiring it here makes that
/// immediate. The restart path reaches this only for a child observed dead at
/// startup. A restart that fails before it has spawned anything retires nothing
/// — its record stays as the stop it began with wrote it.
///
/// `"failed"` is also a status `rocmd` acts on, and that is intended: a failed
/// launch *is* a failure. Its `server-recover` watcher is off until a user
/// enables it, and enabled in its default `contained` mode it restarts any
/// managed record whose status is `"failed"` — a launch retired here becomes
/// eligible on its next tick (subject to the watcher's backoff), where the
/// pid-0 `"starting"` record it replaces was eligible only after the 5-minute
/// stale-`"starting"` window. Two consequences follow. A restart whose engine
/// dies at startup is restarted again rather than left down; one that fails
/// before spawning usually is not, because it leaves the record its stop wrote,
/// which is `"stopped"` — unless that stop could not confirm, when the prior
/// status stands, so a record that was already `"failed"` stays `"failed"`,
/// and so stays eligible for recovery. And that recovery can race a user's own retry: `rocm
/// serve` for the same engine + model mints a new service id, so both may try
/// to bind the same port, and one of them loses.
///
/// Like every launch write after the claim, this goes to the record as it is on
/// disk and only while the launch still owns it (see [`LaunchRecord`]). A stop
/// that settled the record first keeps its `"stopped"`: retired over it, the
/// record would read `"failed"`, which the recover watcher restarts.
pub(crate) fn mark_managed_launch_failed(launch: &mut LaunchRecord<'_>) -> Result<()> {
    launch.update(|current| current.status = "failed".to_owned())?;
    Ok(())
}

/// Retire `record` on a failed launch without letting a failed write displace the
/// launch's own error, which is what diagnoses it. The write failure is still
/// reported, on stderr, because a record left unretired keeps claiming its
/// engine + model.
fn retire_managed_launch_or_warn<R: ManagedStartupRecord + ?Sized>(record: &mut R) {
    retire_managed_launch_or_warn_to(record, &mut io::stderr());
}

/// [`retire_managed_launch_or_warn`] with the warning written to `warnings`
/// instead of stderr, so a test can read what a user would be shown.
fn retire_managed_launch_or_warn_to<R: ManagedStartupRecord + ?Sized>(
    record: &mut R,
    warnings: &mut dyn Write,
) {
    if let Some(warning) = managed_launch_retirement_warning(record) {
        let _ = writeln!(warnings, "{warning}");
    }
}

/// Retire `record`, returning the warning to print if the retirement could not be
/// written.
fn managed_launch_retirement_warning<R: ManagedStartupRecord + ?Sized>(
    record: &mut R,
) -> Option<String> {
    record.retire().err().map(|error| {
        format!(
            "warning: could not mark managed service `{}` as failed: {error:#}",
            record.service_id()
        )
    })
}

/// Retire `record` when `outcome` failed, so a bail-out before the child exists
/// stops it claiming its engine + model.
///
/// Every fallible step of the launch path between its claiming `record.write()`
/// and the spawn has to go through here. Each one that does not is another way to strand the pid-0 `"starting"`
/// corpse — the failures differ, the wedge is identical.
///
/// **The span ends at the spawn, deliberately.** Once a child exists, retiring the
/// record is no longer the safe default: it frees this engine + model for the next
/// `rocm serve`, which would start a rival engine on the same port while the child
/// this launch spawned may still be alive. Past that point the caller has to decide
/// per failure whether the child is known dead — which is what
/// [`fail_managed_launch_if_engine_died`] does, and why a failed *liveness query*
/// is not routed here.
///
/// Takes an already-evaluated `Result` rather than a closure so it can wrap that
/// prefix without holding a second borrow of `record` across the call.
///
/// The returned error is always the one that aborted the launch; a failed
/// retirement is only warned about (see [`retire_managed_launch_or_warn`]).
pub(crate) fn retire_record_on_error<T>(
    record: &mut LaunchRecord<'_>,
    outcome: Result<T>,
) -> Result<T> {
    outcome.inspect_err(|_| retire_managed_launch_or_warn(record))
}

/// Fail the launch if the freshly spawned engine had already exited, retiring the
/// record on the way out so the dead launch stops claiming its engine + model.
///
/// Both platforms funnel their startup check through here — Unix from
/// `Child::try_wait`, Windows from the watched detached spawn — so a child that
/// is *observed* to have died produces the same error whichever platform the user
/// is on: it names the child's own service log, and carries a tail of it when
/// there is one. The platforms differ in when there is. On Unix the child's stdio
/// is redirected into the log, so whatever it printed before dying is there; the
/// Windows detached spawn redirects nothing, so the log has content only if the
/// engine got as far as opening it itself.
///
/// A liveness query that *fails* is a different case, and the platforms do not
/// agree on it. Windows degrades to "assume it is alive": `observe_early_exit`
/// reports `None` when the wait times out or the exit code cannot be read, so the
/// launch proceeds. Unix propagates the `try_wait` error and fails the launch.
/// Neither retires the record, because at that point the child may be running and
/// releasing the engine + model would invite a rival engine onto the same port.
/// The record already names the child by then ([`watch_managed_engine_startup`]),
/// so the failed launch stays stoppable.
pub(crate) fn fail_managed_launch_if_engine_died<R: ManagedStartupRecord + ?Sized>(
    record: &mut R,
    startup_exit: Option<ExitStatus>,
) -> Result<()> {
    let Some(status) = startup_exit else {
        return Ok(());
    };
    // Warned about, not propagated, for the same reason as
    // `retire_record_on_error`: the engine's own exit status and log tail are what
    // diagnose this launch, and a failed record write must not displace them.
    retire_managed_launch_or_warn(record);
    bail!(
        "{}",
        managed_engine_startup_failure_detail(status, record.log_path())
    );
}

/// Read a watched detached spawn's early exit as an `ExitStatus`.
///
/// `GetExitCodeProcess` yields a bare `u32`, which is exactly what
/// `ExitStatusExt::from_raw` takes on Windows; the conversion is isolated here so
/// the shared startup check above stays platform-neutral.
#[cfg(windows)]
pub(crate) fn managed_startup_exit(spawn: &rocm_core::DetachedSpawn) -> Option<ExitStatus> {
    use std::os::windows::process::ExitStatusExt as _;

    spawn.early_exit_code.map(ExitStatus::from_raw)
}

/// Point `record` at a freshly spawned engine child: its pid in both roles, each
/// with the start-time token that lets a later stop verify it is this process and
/// not a recycled pid.
pub(crate) fn adopt_spawned_engine_child(record: &mut ManagedServiceRecord, child_pid: u32) {
    // The child is both roles until the engine state file names a separate
    // server process. Each role gets its own token, captured while the child is
    // (most likely) alive: a stop pairs every recorded PID with its own token,
    // and the two PIDs diverge as soon as `rocmd` restarts the service — at
    // which point an engine PID recorded without one could only be signalled
    // blind. `None` — a child already gone, or any platform but Linux, which
    // has no start times to read — leaves a later stop unable to tell this
    // child from a process that reuses its pid, and it signals a live pid it
    // cannot verify. So a child already known to be dead is never adopted (see
    // `watch_managed_engine_startup`).
    record.record_supervisor_identity(child_pid);
    record.record_engine_identity(child_pid);
}

/// Persist the just-spawned child as a `"starting"` service naming it: its pid
/// and start-time token replace whatever pids the record carried before.
///
/// Fails, writing nothing, when the launch no longer owns the record — a stop
/// settled it or a restart took it over while the child was being spawned. The
/// child is then running but unrecorded, and the caller has to hear that rather
/// than a claim that the service launched; writing it over the stop would bring
/// back, as `"starting"`, a service the stop just reported stopped.
pub(crate) fn persist_spawned_engine_child(
    launch: &mut LaunchRecord<'_>,
    child_pid: u32,
) -> Result<()> {
    let recorded = launch.update(|current| {
        adopt_spawned_engine_child(current, child_pid);
        current.status = "starting".to_owned();
    })?;
    if !recorded {
        bail!(
            "managed service `{}` changed on disk while it was launching — stopped, restarted \
             or removed — so the engine it started (pid {child_pid}) was not recorded on it",
            launch.record.service_id
        );
    }
    Ok(())
}

/// Apply one change to a managed service's record as it is on disk now, rather
/// than writing back a copy the caller holds.
///
/// `ManagedServiceRecord::write` replaces the whole file, and a stop records
/// `stop_requested_unix_ms` before it signals anything — so a copy read before
/// the stop and written back after it erases the marker, and with it the only
/// thing that keeps the daemon's `server-recover` watcher off a service the
/// user asked to stop. The same hazard `rocmd`'s `update_supervised_record`
/// guards against for the supervisor's writes.
///
/// Nothing is written, and `None` returned, when the record is gone or
/// `still_owned` says it no longer belongs to the caller: whoever wrote since
/// holds a truer view. `update` should touch only the caller's own fields, and
/// never the stop marker, so a stop asked for meanwhile survives.
///
/// Shared by the launch ([`LaunchRecord`]) and the restart
/// (`update_restarting_record` in `main.rs`). They differ only in what "still
/// owned" means — see each.
pub(crate) fn update_managed_record_on_disk(
    paths: &AppPaths,
    service_id: &str,
    still_owned: impl FnOnce(&ManagedServiceRecord) -> bool,
    update: impl FnOnce(&mut ManagedServiceRecord),
) -> Result<Option<ManagedServiceRecord>> {
    if !paths.service_manifest_path(service_id).exists() {
        return Ok(None);
    }
    let (mut current, _) = crate::read_managed_service_manifest(paths, service_id)?;
    if !still_owned(&current) {
        return Ok(None);
    }
    update(&mut current);
    current.write()?;
    Ok(Some(current))
}

/// The service record of a managed launch, as the launch last wrote it, and the
/// way every launch write after the claim reaches disk.
///
/// The claim is written whole: its id is freshly minted, so nothing else can
/// have written that file. Every write after it — the retirements, the child put
/// on record, the promotion to `"running"`, the readiness verdict — is applied
/// through [`update_managed_record_on_disk`], because from the claim on the
/// record is visible to `rocm services stop`, to `rocmd`, and to the liveness
/// refresh of every `load_managed_services` caller. The launch lock keeps none
/// of them out: neither stop takes it, and the launch itself releases it before
/// its readiness wait.
///
/// The launch still owns the record while it names the supervisor the launch
/// last recorded (pid 0 before the child is on record, the child after) and
/// still reads as live. A confirmed stop writes `"stopped"`, a restart records
/// its own supervisor, and either means the record is no longer the launch's to
/// write. The test is looser than the restart's, which also requires the
/// status and the engine pid to be unchanged, on purpose: the liveness refresh
/// adopts the engine's server pid from its state file and demotes `"running"`
/// to `"starting"` while the port is still closed, and both happen routinely
/// during a launch's readiness wait — a launch that gave up its readiness
/// verdict to them would fail or under-report every serve watched from the
/// dashboard.
pub(crate) struct LaunchRecord<'a> {
    paths: &'a AppPaths,
    /// The record as this launch last wrote it.
    pub(crate) record: ManagedServiceRecord,
}

impl<'a> LaunchRecord<'a> {
    /// A launch whose record is on disk as `record`.
    pub(crate) const fn new(paths: &'a AppPaths, record: ManagedServiceRecord) -> Self {
        Self { paths, record }
    }

    /// Apply `update` to the record on disk if the launch still owns it.
    /// Returns whether it did; on `false` nothing was written.
    fn update(&mut self, update: impl FnOnce(&mut ManagedServiceRecord)) -> Result<bool> {
        let launched = &self.record;
        let updated = update_managed_record_on_disk(
            self.paths,
            &launched.service_id,
            |current| {
                current.supervisor_pid == launched.supervisor_pid
                    && current.supervisor_start_ticks == launched.supervisor_start_ticks
                    && crate::managed_service_is_live(current)
            },
            update,
        )?;
        Ok(updated.map(|current| self.record = current).is_some())
    }
}

/// The launch's last write: what its readiness wait found. Applied only while
/// the launch still owns the record (see [`LaunchRecord`]); otherwise a stop or
/// a restart has written since, and the record as it is now is returned
/// untouched — the launch reports that, not a verdict on a process somebody else
/// has since stopped or replaced.
pub(crate) fn record_launch_readiness(
    paths: &AppPaths,
    launched: ManagedServiceRecord,
    readiness: EndpointReadiness,
) -> Result<ManagedServiceRecord> {
    let mut launch = LaunchRecord::new(paths, launched);
    let recorded = launch.update(|current| {
        current.status = status_for_readiness(readiness).to_owned();
        if readiness == EndpointReadiness::Serving {
            // Latch the verification the wait just performed, so the readiness
            // checks behind `services list` and chat read it instead of
            // re-probing.
            current.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
        }
    })?;
    if recorded {
        return Ok(launch.record);
    }
    Ok(
        crate::read_managed_service_manifest(paths, &launch.record.service_id)
            .map_or(launch.record, |(current, _)| current),
    )
}

/// The service record a managed engine's startup check writes to: the child put
/// on record, then, if it is seen dead, the record retired.
///
/// Neither the launch nor the restart writes back a copy it holds: a stop may
/// write the same record while either runs, so each change is applied to the
/// record as it is on disk, through [`update_managed_record_on_disk`] — the
/// launch's via [`LaunchRecord`], the restart's via `RestartStartupRecord` in
/// `main.rs`. Both go through the one startup check,
/// [`watch_managed_engine_startup`].
pub(crate) trait ManagedStartupRecord {
    /// Put the spawned child on record as a `"starting"` service naming it.
    fn persist_spawned_child(&mut self, child_pid: u32) -> Result<()>;
    /// Retire the record as `"failed"`.
    fn retire(&mut self) -> Result<()>;
    fn service_id(&self) -> &str;
    /// The service log a startup failure names.
    fn log_path(&self) -> &Path;
}

impl ManagedStartupRecord for LaunchRecord<'_> {
    fn persist_spawned_child(&mut self, child_pid: u32) -> Result<()> {
        persist_spawned_engine_child(self, child_pid)
    }

    fn retire(&mut self) -> Result<()> {
        mark_managed_launch_failed(self)
    }

    fn service_id(&self) -> &str {
        &self.record.service_id
    }

    fn log_path(&self) -> &Path {
        &self.record.log_path
    }
}

/// Everything a platform needs to start a managed engine child. The launch and
/// restart paths both build one and hand it to the same spawner, so the two
/// cannot drift in how an engine is started.
pub(crate) struct ManagedEngineSpawnRequest<'a> {
    pub(crate) paths: &'a AppPaths,
    pub(crate) program: &'a Path,
    pub(crate) args: &'a [String],
    pub(crate) engine_envs_root: Option<&'a Path>,
    pub(crate) endpoint_key_file: Option<&'a Path>,
    /// Where the child's stdio goes. Unix only: the Windows detached spawn
    /// redirects nothing (see [`fail_managed_launch_if_engine_died`]).
    #[cfg_attr(windows, allow(dead_code))]
    pub(crate) log_path: &'a Path,
}

/// Where a spawned child's startup check stands when the spawn returns. Either
/// way the answer is `Some(status)` if the child had already exited and `None`
/// if it is still running.
pub(crate) enum ManagedEngineStartupCheck {
    /// Already answered by the spawn itself: Windows watches the child while its
    /// process handle is still open, inside the spawn call.
    // Only Windows builds one outside tests.
    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    Observed(Option<ExitStatus>),
    /// Still owed, and able to fail: Unix settles, then `try_wait`s, and an `Err`
    /// means the child's state could not be told.
    // Only non-Windows builds one outside tests.
    #[cfg_attr(all(windows, not(test)), allow(dead_code))]
    Pending(Box<dyn FnOnce() -> Result<Option<ExitStatus>>>),
}

/// A spawned managed engine child, and where its startup check stands.
pub(crate) struct SpawnedManagedEngine {
    pub(crate) pid: u32,
    pub(crate) startup: ManagedEngineStartupCheck,
}

/// How the launch and restart paths start an engine child. Production passes
/// [`spawn_managed_engine_process`]; a test passes a stand-in, which is what lets
/// a unit test drive either path through and past its spawn without starting an
/// engine — and so pin the record each step leaves behind at the real call site,
/// not just in the helpers.
pub(crate) type SpawnManagedEngine<'a> =
    dyn Fn(&ManagedEngineSpawnRequest<'_>) -> Result<SpawnedManagedEngine> + 'a;

/// Start a detached managed engine child, and say how to check its startup.
pub(crate) fn spawn_managed_engine_process(
    request: &ManagedEngineSpawnRequest<'_>,
) -> Result<SpawnedManagedEngine> {
    #[cfg(windows)]
    {
        let env_values = app_path_env_var_values(request.paths, request.engine_envs_root);
        let mut env_refs = app_path_env_var_refs(&env_values);
        if let Some(key_file) = request.endpoint_key_file {
            env_refs.push((rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file));
        }
        // The detached spawn hands back a bare PID, not a `Child`, so there is no
        // `try_wait()` to lean on. Watch the process while its handle is still
        // open instead — the Windows helper does that internally, which is what
        // makes the check free of the PID-reuse race a later `OpenProcess` would
        // have. The check has therefore already run by the time this returns.
        let spawn = rocm_core::spawn_detached_no_inherit_watching_startup(
            request.program,
            request.args,
            &env_refs,
            MANAGED_ENGINE_STARTUP_SETTLE,
        )?;
        Ok(SpawnedManagedEngine {
            pid: spawn.pid,
            startup: ManagedEngineStartupCheck::Observed(managed_startup_exit(&spawn)),
        })
    }
    #[cfg(not(windows))]
    {
        let mut command = managed_service_process_command(request.program, request.args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, Some(request.log_path))?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, request.paths);
        if let Some(engine_envs_root) = request.engine_envs_root {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        if let Some(key_file) = request.endpoint_key_file {
            command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        }
        let mut child = command.spawn()?;
        Ok(SpawnedManagedEngine {
            pid: child.id(),
            startup: ManagedEngineStartupCheck::Pending(Box::new(move || {
                thread::sleep(MANAGED_ENGINE_STARTUP_SETTLE);
                child
                    .try_wait()
                    .context("failed to check managed engine startup state")
            })),
        })
    }
}

/// Put a freshly spawned child on record and settle its startup check: fail (and
/// retire the record) if it is observed dead, otherwise return its pid with the
/// record on disk as `"starting"` and naming it.
///
/// A check still pending is preceded by the write, so every way it can end leaves
/// a record that names the child. That matters most for a check that cannot
/// answer — a Unix `try_wait` error — which fails the launch without retiring the
/// record, because the child's state is unknown and freeing the engine + model
/// could put a rival engine on its port. Without the write, what that error left
/// on disk described no process at all: a launch record still at pid 0, which
/// `rocm services stop` cannot reach and no liveness refresh can demote; or a
/// restart record still naming the previous run's pids, which a pending-stop
/// marker would then read as "stopped" and delete the endpoint key of a service
/// that may be running. With it, a stop can reach the child and the liveness
/// refresh reconciles the record either way.
///
/// A check the spawn already answered (Windows) is read first instead, and a
/// child it saw die is never put on record: a dead child has nothing for a stop
/// to reach, and off Linux its pid carries no start-time token, so a stop would
/// signal whatever process reuses that pid. Its launch is retired at pid 0, its
/// restart naming only the previous run's pids, as the stop left them — which
/// carry that same exposure where they have no token, but no more than the
/// stopped record already did. A pending check that then reports a death retires
/// a record naming the child, which on Linux carries the token that keeps
/// `rocm services stop` off a recycled pid.
///
/// This closes the window only for a child that dies inside the startup check.
/// Every Windows child is recorded without a token, so one that dies later and
/// has its pid reused is exposed in the same way; a creation-time token from the
/// watched spawn would be the durable fix.
///
/// Two exits leave a live child unnamed, and both fail the launch or restart
/// with the record as it was before the spawn: the write itself failing, and
/// the write being refused because a stop, a restart or a removal changed the
/// record while the child was spawned. The second is deliberate — the record
/// is no longer the caller's to write — but the child keeps running with
/// nothing on record that a stop could reach; the error names its pid.
pub(crate) fn watch_managed_engine_startup<R: ManagedStartupRecord + ?Sized>(
    record: &mut R,
    spawned: SpawnedManagedEngine,
) -> Result<u32> {
    let SpawnedManagedEngine { pid, startup } = spawned;
    let startup_exit = match startup {
        ManagedEngineStartupCheck::Observed(startup_exit) => {
            if startup_exit.is_none() {
                record.persist_spawned_child(pid)?;
            }
            startup_exit
        }
        ManagedEngineStartupCheck::Pending(check) => {
            record.persist_spawned_child(pid)?;
            // NOT routed through `retire_record_on_error`: the child is already
            // spawned, and a failed query means its state is unknown, not that it
            // died.
            check()?
        }
    };
    fail_managed_launch_if_engine_died(record, startup_exit)?;
    Ok(pid)
}

/// Render a managed engine's immediate exit for the user: the exit status plus a
/// tail of the child's own service log, which is the only record of why it died —
/// the child is detached, so nothing else reaches the terminal.
pub(crate) fn managed_engine_startup_failure_detail(status: ExitStatus, log_path: &Path) -> String {
    managed_engine_startup_failure_detail_polling(
        status,
        log_path,
        MANAGED_ENGINE_STARTUP_LOG_POLL_INTERVAL,
    )
}

/// How long to wait between re-reads of an empty service log before giving up on
/// a tail. A child that died at startup may still be flushing its last lines.
const MANAGED_ENGINE_STARTUP_LOG_POLL_INTERVAL: Duration = Duration::from_millis(120);

/// [`managed_engine_startup_failure_detail`] with the re-read interval injected,
/// so a test of the no-log branch need not sit through the real waits.
fn managed_engine_startup_failure_detail_polling(
    status: ExitStatus,
    log_path: &Path,
    poll_interval: Duration,
) -> String {
    let mut recent_lines = read_optional_tail_lines(log_path, 80, "service log");
    if recent_lines.is_empty() {
        for _ in 0..5 {
            thread::sleep(poll_interval);
            recent_lines = read_optional_tail_lines(log_path, 80, "service log");
            if !recent_lines.is_empty() {
                break;
            }
        }
    }
    if recent_lines.is_empty() {
        return format!(
            "managed engine exited immediately with status {status}; inspect {}",
            log_path.display()
        );
    }
    format!(
        "managed engine exited immediately with status {status}; inspect {}\n\nrecent startup log output:\n{}",
        log_path.display(),
        recent_lines.join("\n")
    )
}
#[cfg(not(windows))]
pub(crate) fn managed_service_process_command(program: &Path, args: &[String]) -> ProcessCommand {
    let mut command = ProcessCommand::new(program);
    command.args(args);
    command
}

/// Engine-neutral result of a managed launch. Returned rather than printed so the
/// caller can render it either as the rich deployment summary (interactive TTY) or
/// as the plain line-by-line form (piped output, chat assistant), from one code path.
pub(crate) struct ManagedLaunchReport {
    pub(crate) service_id: String,
    /// `http://host:port/v1`.
    pub(crate) endpoint_url: String,
    /// `"ready"` (inference confirmed), `"running"` (model listed but not serving
    /// yet), `"starting"` (endpoint not answering), or the existing service's
    /// status when nothing was spawned.
    pub(crate) status: String,
    /// True when an equivalent service was already live and nothing was spawned.
    pub(crate) already_running: bool,
    pub(crate) child_pid: Option<u32>,
    pub(crate) log_path: Option<PathBuf>,
    pub(crate) manifest_path: Option<PathBuf>,
}

/// Either an already-live service (nothing spawned) or a freshly spawned engine
/// child that is `running` but not yet HTTP-ready.
///
/// Split out of [`start_managed_service`] so the attached (`--verbose` /
/// `--foreground`) serve path can spawn the very same detached child and stream
/// its log live from the first line — including startup — instead of blocking on
/// the readiness wait before any output appears.
enum ManagedSpawn {
    AlreadyRunning(ManagedLaunchReport),
    // `ManagedServiceRecord` is large; box it so the two variants stay a similar
    // size (clippy::large_enum_variant).
    Spawned {
        record: Box<ManagedServiceRecord>,
        child_pid: u32,
    },
}

/// Spawn the detached engine child shared by the managed (background) and
/// attached (`--verbose`/`--foreground`) serve paths. Returns before the HTTP
/// readiness wait; callers decide whether to block on readiness
/// ([`start_managed_service`]) or start tailing the log immediately
/// ([`run_attached_service`]).
#[allow(clippy::too_many_arguments)]
fn spawn_managed_engine_child(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    require_api_key: bool,
    spawn_engine: &SpawnManagedEngine<'_>,
) -> Result<ManagedSpawn> {
    paths.ensure()?;
    fs::create_dir_all(paths.services_dir())?;

    // Idempotency guard: if a managed service for this engine+model is already
    // alive, surface it and spawn nothing. A second `serve --managed` (e.g. the
    // chat assistant re-issuing the same request) is treated as satisfied, not
    // an error. Keyed on engine+canonical model — the freshly generated
    // `service_id` is timestamp-unique and would never match an existing one.
    // Stale/dead services fall through and relaunch normally.
    let requested_recipe_json = engine_recipe
        .map(serde_json::to_string)
        .transpose()
        .context("failed to encode engine recipe hint")?;
    if let Some(existing) =
        existing_live_managed_service(paths, engine, &resolve.canonical_model_id)
    {
        if existing.engine_recipe_json != requested_recipe_json {
            bail!(
                "managed service `{}` is already running for engine `{engine}` and model `{}` with different serve options (recipe hint, tool-call parser, or generation defaults); stop it and run `rocm serve` again to apply the requested options",
                existing.service_id,
                resolve.canonical_model_id
            );
        }
        // Reuse cannot satisfy a demand for auth the running server never got.
        // The engine reads its key once, at launch, from the environment this
        // function builds below — so a server started without one keeps serving
        // anonymously no matter what is written afterwards. Upgrading the record
        // here would be worse than doing nothing: the record would claim auth
        // that the live process does not enforce, and
        // `ensure_public_service_has_endpoint_key` would pass on the strength of
        // a key file nothing reads.
        //
        // This is what `rocm remote serve` relies on. It publishes a loopback
        // port onto the tailnet and prints "the API key above is what stops
        // anyone else calling it". Reusing an unauthenticated service silently
        // would make that sentence false about an endpoint the whole tailnet can
        // reach. Refusing is the only answer that fails closed, and it is the
        // same shape as the recipe mismatch above.
        if require_api_key && !existing.requires_api_key {
            bail!(
                "managed service `{}` is already running for engine `{engine}` and model `{}` \
                 without authentication, and a running server cannot be given a key it did not \
                 start with; stop it with `{}` and run the command again to \
                 serve it with `--require-api-key`",
                existing.service_id,
                resolve.canonical_model_id,
                service_hints::service_stop_hint(&existing.service_id)
            );
        }
        record_cli_audit_event(
            paths,
            "service",
            "managed_service_launch_skipped",
            "info",
            format!(
                "skipped duplicate managed launch engine={engine} model={} existing_service_id={} status={}",
                resolve.canonical_model_id, existing.service_id, existing.status
            ),
            Some(&existing.service_id),
        );
        return Ok(ManagedSpawn::AlreadyRunning(ManagedLaunchReport {
            service_id: existing.service_id,
            endpoint_url: existing.endpoint_url,
            status: existing.status,
            already_running: true,
            child_pid: None,
            log_path: None,
            manifest_path: None,
        }));
    }

    let mut record = ManagedServiceRecord::new(
        paths,
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
    record.gpu_indices = gpu_indices.to_vec();
    record.engine_recipe_json = requested_recipe_json;
    // The flag the user actually passed, carried through rather than re-derived.
    //
    // Deriving it from key-file presence looked equivalent and was not:
    // `resolve_endpoint_auth` mints a key for *every* non-loopback bind whether or
    // not auth was demanded, so a plain `--host 0.0.0.0 --allow-public-bind`
    // recorded `true` here. The guard below tests this field before the bind
    // address, so that service was then refused with a message naming a flag it
    // never used and a relaunch command that drops `--allow-public-bind` — the
    // public-bind branch, which carries the right command, became unreachable.
    //
    // This field means "the user demanded auth on a bind that would not otherwise
    // require it". A public bind needs no such record; its address still says so.
    record.requires_api_key = require_api_key;
    // The claim: the launch's one blind write, safe because the id is fresh. Every
    // write after it goes through `launch`, to the record as it is on disk.
    record.write()?;
    let mut launch = LaunchRecord::new(paths, record);

    // From here up to and including the spawn, every fallible step goes through
    // `retire_record_on_error`: the record is on disk claiming this engine + model,
    // so any bail-out that skips the retirement wedges the service just as an
    // unretired startup death would. The span stops at the spawn — see that
    // helper's doc for why retiring is no longer the safe default once a child
    // exists.
    let engine_state_parent = launch
        .record
        .engine_state_path
        .parent()
        .map(Path::to_path_buf);
    if let Some(parent) = engine_state_parent {
        let created = fs::create_dir_all(&parent)
            .with_context(|| format!("failed to create {}", parent.display()));
        retire_record_on_error(&mut launch, created)?;
    }
    let log_created = fs::File::create(&launch.record.log_path)
        .with_context(|| format!("failed to create {}", launch.record.log_path.display()));
    retire_record_on_error(&mut launch, log_created)?;
    let launcher =
        managed_service_launcher_path().context("failed to resolve current rocm executable path");
    let current_exe = retire_record_on_error(&mut launch, launcher)?;
    let built_args = builtin_engine_serve_http_args(
        engine,
        service_id,
        &resolve.canonical_model_id,
        host,
        port,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
        &launch.record.engine_state_path,
        Some(&launch.record.log_path),
    );
    let serve_args = retire_record_on_error(&mut launch, built_args)?;
    // E2E-only: swap in arguments the child rejects immediately, so the launch
    // below observes a real engine that really died during startup.
    #[cfg(feature = "e2e-test-hooks")]
    let serve_args = if scripted_managed_engine_startup_failure() {
        managed_engine_startup_failure_args()
    } else {
        serve_args
    };
    let envs_root = env_root_for_service(paths, engine, runtime_id, env_id);
    let engine_envs_root = retire_record_on_error(&mut launch, envs_root)?;
    // Hand the child the *path* to the endpoint key file (public bind only) via the
    // environment. A path — not the secret value — is what the detached-spawn
    // primitives accept as an env override, and it keeps the key off both the argv
    // and the environment block. `serve()` wrote the file before spawning.
    // Validity, not mere existence: the engine adapters resolve the key with
    // `endpoint_api_key_from_file` and enforce nothing when it yields `None`, so
    // an empty or malformed key file would otherwise satisfy the guard below and
    // still produce an unauthenticated public listener.
    let endpoint_key_file = endpoint_keys::endpoint_key_file_if_present(paths, service_id)
        .filter(|path| rocm_engine_protocol::endpoint_api_key_from_file(path).is_some());
    // `serve()` already resolved and stored the key for a public bind, so the
    // public-bind branch cannot fire on the fresh-launch path today. It is the
    // shared choke point for managed spawns, so enforce the invariant here too
    // rather than relying on every future caller having done so.
    //
    // `record.requires_api_key` is passed, not a literal, and the two arguments
    // are deliberately different things: that field is the `--require-api-key`
    // flag the caller passed, `endpoint_key_file` is whether a *usable* key is on
    // disk. A present but empty or malformed key file is where they disagree, and
    // is exactly what the `requires_api_key` branch exists to refuse.
    //
    // The field is threaded, never derived from the key file. Deriving it marked
    // every public bind as having demanded auth — see the assignment above.
    //
    // Evaluated before the guard, not inside the call, so reading the record here
    // does not overlap the `&mut` borrow the retirement needs.
    let public_key_guard = ensure_public_service_has_endpoint_key(
        host,
        endpoint_key_file.is_some(),
        launch.record.requires_api_key,
    );
    retire_record_on_error(&mut launch, public_key_guard)?;
    let spawned = spawn_engine(&ManagedEngineSpawnRequest {
        paths,
        program: &current_exe,
        args: &serve_args,
        engine_envs_root: engine_envs_root.as_deref(),
        endpoint_key_file: endpoint_key_file.as_deref(),
        log_path: &launch.record.log_path,
    })
    .context("failed to launch managed engine process");
    // The last step the retirement span covers: past it a child exists.
    let spawned = retire_record_on_error(&mut launch, spawned)?;
    let child_pid = watch_managed_engine_startup(&mut launch, spawned)?;
    if !launch.update(|current| current.status = "running".to_owned())? {
        bail!(
            "managed service `{}` changed on disk while its engine (pid {child_pid}) was \
             starting — stopped, restarted or removed — so the launch did not mark it running",
            launch.record.service_id
        );
    }

    Ok(ManagedSpawn::Spawned {
        record: Box::new(launch.record),
        child_pid,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_managed_service(
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    endpoint_api_key: Option<&str>,
    launch_lock: rocm_core::FileLock,
    require_api_key: bool,
    on_wait_tick: &mut dyn FnMut(Duration),
) -> Result<ManagedLaunchReport> {
    let paths = AppPaths::discover()?;
    let (record, child_pid) = match spawn_managed_engine_child(
        &paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
        require_api_key,
        &spawn_managed_engine_process,
    )? {
        ManagedSpawn::AlreadyRunning(report) => return Ok(report),
        ManagedSpawn::Spawned { record, child_pid } => (*record, child_pid),
    };
    // The claiming service record is now persisted, so the selected GPU is
    // visible to any concurrent auto-selection. Release the launch lock before
    // the readiness wait below, which can block for many seconds — holding it
    // that long would needlessly serialize unrelated serves.
    //
    // What the lock *does* still cover is the startup check above: both platforms
    // spend `MANAGED_ENGINE_STARTUP_SETTLE` inside `spawn_managed_engine_child`,
    // so a concurrent serve waits that much longer for the lock. That is
    // deliberate — the check has to see the child before the record is promoted to
    // `"running"` — and bounded, unlike the readiness wait.
    drop(launch_lock);

    let readiness = wait_for_service_http_ready_with_progress(
        engine,
        host,
        port,
        &resolve.canonical_model_id,
        endpoint_api_key,
        Duration::from_secs(45),
        on_wait_tick,
    );
    let launch_status = status_for_readiness(readiness);
    // The lock is gone and the wait can run 45 s, so this is the launch write a
    // concurrent stop is most likely to meet: it reports the record as it is
    // after the write, which is the stop's if the stop got there first.
    let record = record_launch_readiness(&paths, record, readiness)?;
    let endpoint_url = format!("{}/v1", format_http_base_url(host, port));
    record_cli_audit_event(
        &paths,
        "service",
        "managed_service_launch",
        "info",
        format!(
            "launched managed service engine={} model={} endpoint={} readiness={}",
            engine, resolve.canonical_model_id, endpoint_url, launch_status
        ),
        Some(service_id),
    );
    Ok(ManagedLaunchReport {
        service_id: service_id.to_owned(),
        endpoint_url,
        status: record.status,
        already_running: false,
        child_pid: Some(child_pid),
        log_path: Some(record.log_path),
        manifest_path: Some(record.manifest_path),
    })
}

/// Reproduce the original plain, line-by-line managed-launch output. Used for
/// non-interactive output (piped, CI, the chat assistant's `serve --managed`),
/// where the animated summary is inappropriate. The interactive path renders the
/// summary table via [`crate::serve_summary`] instead.
pub(crate) fn print_managed_launch_plain(
    report: &ManagedLaunchReport,
    endpoint_api_key: Option<&str>,
) {
    if report.already_running {
        println!("managed service already running");
        println!("  service_id: {}", report.service_id);
        println!("  endpoint: {}", report.endpoint_url);
        println!("  status: {}", report.status);
        println!("  note: existing service detected; no second process spawned");
        return;
    }
    println!("managed service launched");
    println!("  service_id: {}", report.service_id);
    if let Some(child_pid) = report.child_pid {
        println!("  process_pid: {child_pid}");
    }
    println!("  endpoint: {}", report.endpoint_url);
    if let Some(key) = endpoint_api_key {
        // Intentional one-time display of a freshly generated API key to the
        // terminal so the user can copy it — the designed delivery channel
        // documented on `render_endpoint_client_config`, not a log. The tag below
        // is currently inert (Rust's CodeQL pack has no AlertSuppression.ql yet —
        // github/codeql#21637) but will start working once that lands, since the
        // tag must be the single line immediately before the flagged code.
        print!(
            // codeql[rust/cleartext-logging]
            "{}",
            render_endpoint_client_config(&report.endpoint_url, key)
        );
    }
    if let Some(log_path) = report.log_path.as_deref() {
        println!("  log_path: {}", log_path.display());
    }
    if let Some(manifest_path) = report.manifest_path.as_deref() {
        println!("  manifest_path: {}", manifest_path.display());
    }
    println!("  readiness: {}", report.status);
}

/// Render the one-time secure client configuration for a public, authenticated
/// endpoint. This is the *intended* channel for delivering the key to the user
/// (unlike logs/status, which must never contain it) — it prints the key once at
/// launch alongside a ready-to-use example. Callers only invoke this for a
/// non-loopback bind that generated/received a key.
fn render_endpoint_client_config(endpoint_url: &str, api_key: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "  api key: {api_key}");
    let _ = writeln!(
        out,
        "  note: this key is shown only now - clients must send `Authorization: Bearer <key>`"
    );
    let _ = writeln!(
        out,
        "  example: curl -H \"Authorization: Bearer {api_key}\" {endpoint_url}/models"
    );
    out
}

/// The "should we spawn?" decision for [`ensure_background_helper_running`],
/// factored out so it is testable hermetically (no spawn side effect). Returns
/// `true` when the file-based runtime state says the daemon is `running` AND its
/// recorded `daemon_pid` is a live process — i.e. a second spawn must be guarded.
/// A missing state file, `running=false`, or a dead/zero pid returns `false`.
fn background_helper_already_running(paths: &AppPaths) -> Result<bool> {
    Ok(AutomationRuntimeState::load(paths)?
        .is_some_and(|state| state.running && rocm_core::process_is_running(state.daemon_pid)))
}

/// Shared daemon-lifecycle entrypoint: ensures the background automation helper
/// (`rocm daemon`) is running, spawning it detached if not. Liveness is read from
/// the file-based automation runtime state. Intentionally `pub(crate)` — reused by
/// both the `serve --managed` path and `automations enable`. Only the spawn result
/// itself (`command.spawn()` / `spawn_detached_no_inherit`) is logged rather than
/// propagated; setup errors (path discovery, stdio attach) still return `Err`.
pub(crate) fn ensure_background_helper_running() -> Result<()> {
    ensure_background_helper_running_quiet(false)
}

/// As [`ensure_background_helper_running`], but suppresses the stdout status line
/// when `quiet` is set. The interactive `rocm serve` summary path uses `quiet` so
/// the daemon-spawn note does not appear above the deployment summary table.
pub(crate) fn ensure_background_helper_running_quiet(quiet: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    if background_helper_already_running(&paths)? {
        return Ok(());
    }

    // The check above and the spawn below are a TOCTOU window: two concurrent
    // callers (e.g. two `rocm serve`) can both read "not running" and each spawn
    // a daemon. Serialize the decision on a lock file and re-check under it — the
    // first holder spawns, later holders observe the now-running daemon and
    // return without spawning. The unlocked pre-check above keeps the common
    // already-running case lock-free.
    let _autostart_lock = rocm_core::FileLock::acquire(paths.automation_autostart_lock_path())?;
    if background_helper_already_running(&paths)? {
        return Ok(());
    }

    // The lock alone does not close the window: the spawned daemon does not
    // publish its `running` runtime state until well after `spawn()` (clap parse,
    // runtime build, config load, banner flush). A second caller that acquires
    // this lock during that gap still sees "not running" and would spawn a
    // duplicate. Bridge the gap with a short-lived claim recording the child PID
    // and spawn time: a holder that finds a live, recent claim defers instead.
    let claim_path = paths.automation_autostart_claim_path();
    if autostart_spawn_in_flight(
        read_autostart_claim(&claim_path),
        now_unix_millis(),
        AUTOSTART_CLAIM_TTL_MS,
        rocm_core::process_is_running,
    ) {
        return Ok(());
    }

    let exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let args = vec!["daemon".to_owned()];
    #[cfg(windows)]
    let spawn_result = {
        let env_values = app_path_env_var_values(&paths, None);
        let env_refs = app_path_env_var_refs(&env_values);
        rocm_core::spawn_detached_no_inherit(&exe, &args, &env_refs)
    };
    #[cfg(not(windows))]
    let spawn_result = {
        let mut command = managed_service_process_command(&exe, &args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, None)?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, &paths);
        command.spawn().map(|child| child.id())
    };
    match spawn_result {
        Ok(daemon_pid) => {
            // Record the claim before returning (and thus releasing the lock) so a
            // concurrent holder in the spawn→publish window defers. Best-effort: a
            // failed write only reopens the original, already-tolerated race.
            let _ = write_autostart_claim(
                &claim_path,
                AutostartClaim {
                    daemon_pid,
                    spawned_at_ms: now_unix_millis(),
                },
            );
            if !quiet {
                println!("  helper: started background automation daemon");
            }
        }
        Err(error) if !quiet => {
            println!("  helper: could not start background automation daemon: {error}");
        }
        Err(_) => {}
    }
    Ok(())
}

/// How long an autostart claim is honoured before it is treated as stale even if
/// its recorded PID is still alive. Comfortably longer than a cold daemon boot
/// (clap parse → runtime build → config load → state publish) yet short enough
/// that a crashed spawn cannot suppress autostart for long.
const AUTOSTART_CLAIM_TTL_MS: u128 = 30_000;

/// A just-spawned daemon's autostart claim: the child PID and the wall-clock time
/// (milliseconds since the Unix epoch) the spawn was recorded. It lets a
/// concurrent autostart holder distinguish a live, in-flight spawn from a stale
/// leftover. Serialized as a single `"<pid> <ms>"` line — no dependency and
/// trivially forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutostartClaim {
    daemon_pid: u32,
    spawned_at_ms: u128,
}

/// Milliseconds since the Unix epoch, or `0` if the clock is before the epoch
/// (which only makes a fresh claim look old — safe, it just permits a respawn).
fn now_unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis())
}

/// Read an autostart claim, returning `None` when the file is absent or
/// unparseable (either is treated as "no claim", so a respawn is permitted).
fn read_autostart_claim(path: &Path) -> Option<AutostartClaim> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut parts = text.split_whitespace();
    let daemon_pid = parts.next()?.parse().ok()?;
    let spawned_at_ms = parts.next()?.parse().ok()?;
    Some(AutostartClaim {
        daemon_pid,
        spawned_at_ms,
    })
}

/// Write an autostart claim as `"<pid> <ms>"`. Best-effort at the call site.
fn write_autostart_claim(path: &Path, claim: AutostartClaim) -> std::io::Result<()> {
    std::fs::write(
        path,
        format!("{} {}", claim.daemon_pid, claim.spawned_at_ms),
    )
}

/// Whether an existing autostart `claim` means a daemon spawn is still in flight,
/// so the current lock holder should defer rather than spawn a duplicate. A claim
/// counts as in-flight only while its child PID is alive *and* it is younger than
/// `ttl_ms` — the TTL bounds how long a crashed spawn (or a PID later reused by an
/// unrelated process) can suppress autostart. `pid_alive` is injected so the
/// decision is unit-testable without a live process.
fn autostart_spawn_in_flight(
    claim: Option<AutostartClaim>,
    now_ms: u128,
    ttl_ms: u128,
    pid_alive: impl Fn(u32) -> bool,
) -> bool {
    claim.is_some_and(|claim| {
        now_ms.saturating_sub(claim.spawned_at_ms) < ttl_ms && pid_alive(claim.daemon_pid)
    })
}

/// What ended an attached (`--verbose`/`--foreground`) streaming session. Kept
/// as a plain enum, separate from any terminal I/O, so the follow-up action
/// (detach note vs. stop the server) is unit-testable without a TTY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttachOutcome {
    /// Ctrl-D: leave the server running and hand the terminal back.
    Detach,
    /// Ctrl-C: stop the server, then hand the terminal back.
    Stop,
    /// The engine process exited on its own while we were streaming its log.
    ServerExited,
}

/// Attached serve path for `--verbose`/`--foreground`: spawn the engine as a
/// detached managed child (the same child the background path spawns) and stream
/// its log in this terminal. Unlike the old in-process foreground, the server
/// survives the session — Ctrl-D detaches and leaves it running, Ctrl-C stops it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attached_service(
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port: u16,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    endpoint_api_key: Option<&str>,
    launch_lock: rocm_core::FileLock,
    require_api_key: bool,
) -> Result<()> {
    let paths = AppPaths::discover()?;

    let spawn = spawn_managed_engine_child(
        &paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port,
        &resolve.device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        resolve.engine_recipe.as_ref(),
        require_api_key,
        &spawn_managed_engine_process,
    )?;
    // The claiming record is persisted (or an existing service was found), so the
    // selected GPU is now visible to concurrent auto-selection. Release the launch
    // lock before streaming logs, which blocks for the whole attached session.
    drop(launch_lock);

    let (service_id, log_path, child_pid) = match spawn {
        // A server for this engine+model is already live. Don't fight it for the
        // port — point the user at the existing one instead of tailing a log we
        // did not start.
        ManagedSpawn::AlreadyRunning(report) => {
            println!("model already being served");
            println!("  service_id: {}", report.service_id);
            println!("  endpoint: {}", report.endpoint_url);
            println!("  status: {}", report.status);
            println!("  logs: rocm logs {}", report.service_id);
            println!(
                "  stop: {}",
                service_hints::service_stop_hint(&report.service_id)
            );
            drop_orphaned_endpoint_key_on_already_running(&paths, service_id, endpoint_api_key);
            return Ok(());
        }
        ManagedSpawn::Spawned { record, child_pid } => {
            (service_id.to_owned(), record.log_path.clone(), child_pid)
        }
    };

    // The child is a managed service that outlives this session once detached, so
    // it needs the same supervision the background path gives it: the daemon
    // health-checks and auto-recovers managed servers, reconciles a self-exited
    // server's record, and feeds the dashboard. Match the background ordering
    // (spawn, then ensure the helper) and keep it quiet so no status line breaks
    // into the log stream.
    ensure_background_helper_running_quiet(true)?;

    // The resolution detail (model, engine, runtime, GPU, warnings) was already
    // printed as the "serve plan" block in `serve()`; extend it with the launch
    // coordinates and the streaming hint rather than repeating it.
    let endpoint = format!("{}/v1", format_http_base_url(host, port));
    println!("  service_id: {service_id}");
    println!("  endpoint: {endpoint}");
    if let Some(key) = endpoint_api_key {
        // Same intentional one-time key display as `print_managed_launch_plain`
        // above; see its rationale for why the tag below is currently inert.
        // codeql[rust/cleartext-logging]
        print!("{}", render_endpoint_client_config(&endpoint, key));
    }
    println!("  streaming engine logs - Ctrl-D detaches (leaves it running), Ctrl-C stops it");
    println!();

    let outcome = stream_attached_logs(&log_path, child_pid)?;
    println!();

    match outcome {
        AttachOutcome::Detach => {
            println!("detached - server still running");
            println!("  service_id: {service_id}");
            println!("  endpoint: {endpoint}");
            println!("  list: rocm services");
            println!("  logs: rocm logs {service_id}");
            println!("  stop: {}", service_hints::service_stop_hint(&service_id));
            record_cli_audit_event(
                &paths,
                "service",
                "serve_detach",
                "info",
                format!("detached from attached serve service_id={service_id} endpoint={endpoint}"),
                Some(&service_id),
            );
            Ok(())
        }
        AttachOutcome::Stop => {
            println!("stopping server...");
            match run_internal_sandbox_tool(
                &paths,
                SandboxToolArg::StopServer,
                Some(service_id.clone()),
                true,
            ) {
                Ok(result) => print!("{}", render_service_action_result("stop_server", &result)),
                Err(error) => {
                    // Best-effort direct signal so Ctrl-C never leaves the model
                    // orphaned when the sandbox stop path fails.
                    let _ = rocm_core::terminate_process_tree(child_pid);
                    println!("  note: {error}");
                }
            }
            record_cli_audit_event(
                &paths,
                "service",
                "serve_stop",
                "info",
                format!("stopped attached serve service_id={service_id}"),
                Some(&service_id),
            );
            Ok(())
        }
        AttachOutcome::ServerExited => {
            println!("server process exited");
            println!("  service_id: {service_id}");
            println!("  recent logs: rocm logs {service_id}");
            Ok(())
        }
    }
}

/// Restores cooked terminal mode when dropped, so [`stream_attached_logs`] leaves
/// the terminal usable on every exit path (normal return, `?` error, or panic).
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Map a key press (control modifier + lowercased character) to the attach
/// action it triggers, if any. Factored out of the raw-mode reader loop so the
/// Ctrl-D/Ctrl-C mapping is unit-testable without a terminal.
const fn detach_key_outcome(ctrl: bool, ch: char) -> Option<AttachOutcome> {
    if !ctrl {
        return None;
    }
    match ch {
        'c' => Some(AttachOutcome::Stop),
        'd' => Some(AttachOutcome::Detach),
        _ => None,
    }
}

/// Follow `log_path` in the terminal until the user presses Ctrl-D (detach) or
/// Ctrl-C (stop), or the engine process exits. Uses crossterm raw mode to
/// capture the keys directly (in raw mode Ctrl-C does not raise SIGINT, so we see
/// it as a key event). When stdin is not a TTY (piped/CI), keystroke capture is
/// impossible, so we follow the log until the process exits instead.
fn stream_attached_logs(log_path: &Path, child_pid: u32) -> Result<AttachOutcome> {
    use std::io::IsTerminal as _;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    if !std::io::stdin().is_terminal() {
        return stream_attached_logs_no_tty(log_path, child_pid);
    }

    // Enter raw mode *before* spawning the key reader. In raw mode Ctrl-C arrives
    // as a key event instead of SIGINT; if the reader started first, a Ctrl-C in
    // that window would kill the CLI outright (leaving the detached child alive
    // but printing no detach/stop message). The guard restores cooked mode on
    // every exit path (normal return, `?` error, panic).
    crossterm::terminal::enable_raw_mode().context("failed to enter raw terminal mode")?;
    let _raw_guard = RawModeGuard;

    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<AttachOutcome>();

    let reader_stop = Arc::clone(&stop);
    let reader = thread::spawn(move || {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
        while !reader_stop.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => {
                        let outcome = match key.code {
                            KeyCode::Char(ch) => detach_key_outcome(
                                key.modifiers.contains(KeyModifiers::CONTROL),
                                ch.to_ascii_lowercase(),
                            ),
                            _ => None,
                        };
                        if let Some(outcome) = outcome {
                            let _ = tx.send(outcome);
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });

    let mut stdout = io::stdout();
    let mut log_reader: Option<io::BufReader<fs::File>> = None;
    let mut line = String::new();
    let outcome = loop {
        if log_reader.is_none() {
            log_reader = fs::File::open(log_path).ok().map(io::BufReader::new);
        }
        if let Some(reader) = log_reader.as_mut() {
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        // Raw mode disables the terminal's own \n -> \r\n
                        // translation, so emit an explicit carriage return to
                        // keep the log left-aligned instead of stair-stepping.
                        let _ = write!(stdout, "{}\r\n", line.trim_end_matches('\n'));
                        let _ = stdout.flush();
                    }
                    Err(_) => break,
                }
            }
        }

        if let Ok(signal) = rx.try_recv() {
            break signal;
        }
        if !process_is_running(child_pid) {
            break AttachOutcome::ServerExited;
        }
        thread::sleep(Duration::from_millis(150));
    };

    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    Ok(outcome)
}

/// Non-interactive fallback for [`stream_attached_logs`]: no keystroke capture,
/// so just follow the log until the (detached) engine process exits. A Ctrl-C
/// here delivers SIGINT to this process and leaves the managed server running.
fn stream_attached_logs_no_tty(log_path: &Path, child_pid: u32) -> Result<AttachOutcome> {
    let mut stdout = io::stdout();
    let mut log_reader: Option<io::BufReader<fs::File>> = None;
    let mut line = String::new();
    loop {
        if log_reader.is_none() {
            log_reader = fs::File::open(log_path).ok().map(io::BufReader::new);
        }
        if let Some(reader) = log_reader.as_mut() {
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        let _ = write!(stdout, "{line}");
                        let _ = stdout.flush();
                    }
                    Err(_) => break,
                }
            }
        }
        if !process_is_running(child_pid) {
            return Ok(AttachOutcome::ServerExited);
        }
        thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(windows))]
    use crate::load_managed_services;
    use crate::tests::{
        exit_status_from_code, manifest_on_disk, spawn_must_not_be_reached, spawn_that_fails,
        stand_in_child_pid, stand_in_spawn, stand_in_spawn_observed, test_paths,
    };

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
    fn detach_key_ctrl_d_detaches_ctrl_c_stops() {
        assert_eq!(detach_key_outcome(true, 'd'), Some(AttachOutcome::Detach));
        assert_eq!(detach_key_outcome(true, 'c'), Some(AttachOutcome::Stop));
    }

    #[test]
    fn detach_key_ignores_plain_and_unrelated_keys() {
        // Without the control modifier, `d`/`c` are ordinary log-scroll input.
        assert_eq!(detach_key_outcome(false, 'd'), None);
        assert_eq!(detach_key_outcome(false, 'c'), None);
        // Other control combos are not detach/stop triggers.
        assert_eq!(detach_key_outcome(true, 'q'), None);
        assert_eq!(detach_key_outcome(true, 'z'), None);
    }

    /// An engine that dies during startup must reach the user with the child's
    /// own log tail, not just an exit status — the log is the only place the
    /// reason is recorded, since the child is detached and writes nowhere else.
    /// Both platforms feed this from the same startup check, so both name the log;
    /// the tail appears where the log has content, which on Windows (no stdio
    /// redirection) means only once the engine has opened the log itself.
    #[test]
    fn managed_engine_startup_failure_detail_carries_the_child_log_tail() {
        let (root, _paths) = test_paths("managed-startup-failure-detail");
        fs::create_dir_all(&root).expect("test dir");
        let log_path = root.join("service.log");
        fs::write(&log_path, "loading runtime\nfatal: no usable device\n").expect("write log");

        let detail = managed_engine_startup_failure_detail(exit_status_from_code(3), &log_path);

        let _ = fs::remove_dir_all(&root);
        assert!(
            detail.contains("managed engine exited immediately"),
            "unexpected detail: {detail}"
        );
        assert!(
            detail.contains("fatal: no usable device"),
            "log tail missing from detail: {detail}"
        );
        assert!(
            detail.contains(&log_path.display().to_string()),
            "log path missing from detail: {detail}"
        );
    }

    /// The same failure with no log written yet still has to name the log path so
    /// the user knows where to look once the child flushes.
    #[test]
    fn managed_engine_startup_failure_detail_without_a_log_still_points_at_it() {
        let (root, _paths) = test_paths("managed-startup-failure-no-log");
        fs::create_dir_all(&root).expect("test dir");
        let log_path = root.join("absent.log");

        // Zero interval only keeps this fast: the log never appears, so this pins
        // the no-tail message and nothing about the re-reads themselves. Those
        // are pinned by
        // `managed_engine_startup_failure_detail_picks_up_a_log_written_while_polling`.
        let detail = managed_engine_startup_failure_detail_polling(
            exit_status_from_code(1),
            &log_path,
            Duration::ZERO,
        );

        let _ = fs::remove_dir_all(&root);
        assert!(
            detail.contains(&log_path.display().to_string()),
            "log path missing from detail: {detail}"
        );
        assert!(
            !detail.contains("recent startup log output"),
            "empty log should not advertise a tail: {detail}"
        );
    }

    /// A child that died at startup may still be flushing its log when the
    /// failure is rendered. The re-reads exist for exactly that, so the tail
    /// must appear when the log lands only after the first read found nothing.
    #[test]
    fn managed_engine_startup_failure_detail_picks_up_a_log_written_while_polling() {
        let (root, _paths) = test_paths("managed-startup-failure-late-log");
        fs::create_dir_all(&root).expect("test dir");
        let log_path = root.join("late.log");

        // The writer starts the clock only once released, and then waits well
        // past the first (immediate) read but well inside the re-read window
        // (5 x 250 ms). Landing the file by rename means a read never sees it
        // half-written.
        let (release, released) = std::sync::mpsc::channel::<()>();
        let writer = {
            let log_path = log_path.clone();
            let staged = root.join("late.log.partial");
            thread::spawn(move || {
                released.recv().expect("release writer");
                thread::sleep(Duration::from_millis(60));
                fs::write(&staged, "fatal: flushed after exit\n").expect("stage log");
                fs::rename(&staged, &log_path).expect("land log");
            })
        };
        release.send(()).expect("release writer");
        let detail = managed_engine_startup_failure_detail_polling(
            exit_status_from_code(1),
            &log_path,
            Duration::from_millis(250),
        );
        writer.join().expect("writer thread");

        let _ = fs::remove_dir_all(&root);
        assert!(
            detail.contains("fatal: flushed after exit"),
            "a log written after the first read must still reach the user: {detail}"
        );
    }

    /// A launch that dies at startup must retire its own service record.
    ///
    /// The record is written before the spawn, so it still says `"starting"` with
    /// a `supervisor_pid` of 0. Pid 0 is filtered out of the liveness refresh, so
    /// nothing ever demotes such a record, while `"starting"` counts as live —
    /// leaving it would make the idempotency guard report the corpse as already
    /// running and refuse every later `rocm serve` for the same engine + model.
    #[test]
    fn a_failed_launch_record_stops_blocking_the_next_serve() -> Result<()> {
        let (root, paths) = test_paths("managed-failed-launch-unblocks");
        paths.ensure()?;
        let record = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11512,
            "managed",
            // The pre-spawn state: no child pid recorded yet.
            0,
            None,
            None,
            None,
        );
        record.write()?;

        // Precondition: abandoned as written, the record blocks the next launch.
        assert!(
            existing_live_managed_service(&paths, "lemonade", "qwen-canonical").is_some(),
            "a pid-0 `starting` record should look live — that is the trap being closed"
        );

        mark_managed_launch_failed(&mut LaunchRecord::new(&paths, record))?;

        let still_blocking = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        assert!(
            still_blocking.is_none(),
            "a failed launch must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// Build the pre-spawn record every managed launch persists to claim its
    /// engine + model: `"starting"`, with no child pid recorded yet.
    fn pre_spawn_record(paths: &AppPaths, port: u16) -> ManagedServiceRecord {
        ManagedServiceRecord::new(
            paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            port,
            "managed",
            0,
            None,
            None,
            None,
        )
    }

    /// The whole user-visible fix in one assertion: an engine that exited during
    /// startup must fail the launch, name its log, *and* let go of the engine +
    /// model so the next `rocm serve` can try again.
    ///
    /// Both platforms route their startup check through this function — Unix from
    /// `Child::try_wait`, Windows from the watched detached spawn — so this runs
    /// everywhere and covers the wiring between detecting the death and acting on
    /// it, which the two halves tested in isolation do not.
    #[test]
    fn a_dead_engine_fails_the_launch_and_frees_the_engine_for_the_next_serve() -> Result<()> {
        let (root, paths) = test_paths("managed-dead-engine-fails-launch");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11513);
        record.write()?;
        fs::write(&record.log_path, "fatal: no usable device\n")?;

        let mut launch = LaunchRecord::new(&paths, record);
        let outcome =
            fail_managed_launch_if_engine_died(&mut launch, Some(exit_status_from_code(3)));

        let still_blocking = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        let error = outcome
            .expect_err("an engine that already exited must fail the launch")
            .to_string();
        assert!(
            error.contains("fatal: no usable device"),
            "the child's log tail is the only account of why it died: {error}"
        );
        assert!(
            still_blocking.is_none(),
            "a launch that failed at startup must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// What the record a spawned child is persisted as is worth, once written: it
    /// keeps the claim while that child lives and is reconciled once it does not,
    /// without losing the endpoint key a crash must keep.
    ///
    /// This proves the record, not *when* it is written. That the launch and the
    /// restart write it before their startup checks is pinned at those call sites
    /// by `a_launch_names_its_child_on_record_before_the_startup_check` and
    /// `a_restart_whose_engine_dies_at_startup_fails_and_retires_the_record`.
    ///
    /// Restart-shaped on purpose — previous run's dead pid, endpoint key on disk,
    /// stop marker already cleared — because that is the case where an
    /// inaccurate record cost the most: a refresh read the dead pids as a
    /// finished stop and deleted the key of a service that may be running. So the
    /// child is put on record by the restart's own write, `record_restarted_child`:
    /// a launch's write refuses a `"stopped"` record, which a stop settled.
    #[cfg(not(windows))]
    #[test]
    fn a_record_naming_the_spawned_child_is_reconciled_once_the_child_is_gone() -> Result<()> {
        let (root, paths) = test_paths("managed-child-persisted-before-check");
        paths.ensure()?;
        let mut record = pre_spawn_record(&paths, 11517);
        record.supervisor_pid = 999_999_999;
        record.status = "stopped".to_owned();
        record.write()?;
        endpoint_keys::store_endpoint_api_key(&paths, &record.service_id, "k-restart")?;
        let mut child = ProcessCommand::new("sleep").arg("30").spawn()?;
        let child_pid = child.id();

        crate::record_restarted_child(&paths, &record, None, child_pid)?;
        let while_alive = load_managed_services(&paths)?;
        let _ = child.kill();
        let _ = child.wait();
        let after_exit = load_managed_services(&paths)?;
        let key_after_exit = endpoint_keys::endpoint_api_key(&paths, &record.service_id);
        let _ = fs::remove_dir_all(root);

        let alive = while_alive.first().expect("record on disk");
        assert_eq!(
            alive.supervisor_pid, child_pid,
            "the record must name the child"
        );
        assert_eq!(alive.engine_pid, Some(child_pid));
        assert!(
            alive.supervisor_start_ticks.is_some(),
            "a stop needs the start-time token to verify it is this child"
        );
        assert_eq!(alive.status, "starting", "a live child keeps the claim");
        let gone = after_exit.first().expect("record on disk");
        assert_eq!(
            gone.status, "stopped",
            "once the child is gone the refresh must reconcile the record"
        );
        assert_eq!(
            key_after_exit.as_deref(),
            Some("k-restart"),
            "no stop was requested, so the key must survive the child, as for a crash"
        );
        Ok(())
    }

    /// The converse, and the reason the check can be trusted: a child that is
    /// still running must leave the launch untouched. If this ever failed, every
    /// healthy serve would be rejected and its record retired.
    #[test]
    fn a_live_engine_leaves_the_launch_running() -> Result<()> {
        let (root, paths) = test_paths("managed-live-engine-proceeds");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11514);
        record.write()?;

        let mut launch = LaunchRecord::new(&paths, record);
        let outcome = fail_managed_launch_if_engine_died(&mut launch, None);

        let _ = fs::remove_dir_all(root);
        assert!(outcome.is_ok(), "a running engine must not fail the launch");
        assert_eq!(
            launch.record.status, "starting",
            "a running engine must not have its record retired"
        );
        Ok(())
    }

    /// The helper itself, fed a synthetic failure: it must retire the record and
    /// hand back the *original* error, since that is what reaches the user.
    ///
    /// This proves the helper, not that any launch or restart calls it — the
    /// call-site wiring is pinned by the real launch failures driven through
    /// `spawn_managed_engine_child` (the key-guard refusal and the missing
    /// runtime), which leave the record retired only if their site retires it.
    #[test]
    fn retire_record_on_error_retires_the_record_and_keeps_the_original_error() -> Result<()> {
        let (root, paths) = test_paths("managed-spawn-failure-unblocks");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11515);
        record.write()?;

        let outcome: Result<u32> = retire_record_on_error(
            &mut LaunchRecord::new(&paths, record),
            Err(anyhow::anyhow!("failed to launch managed engine process")),
        );

        let still_blocking = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        let error = outcome
            .expect_err("a failed spawn must fail the launch")
            .to_string();
        assert!(
            error.contains("failed to launch managed engine process"),
            "the spawn failure must survive the retirement: {error}"
        );
        assert!(
            still_blocking.is_none(),
            "a spawn that never started must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// A retirement that cannot be written must say so — the record it failed to
    /// write still claims the engine + model — while the launch's own error stays
    /// the one returned. The warning is read from the sink the retirement
    /// writes it to, so a retirement that built the warning and dropped it
    /// fails this.
    #[test]
    fn a_retirement_that_cannot_be_written_is_reported_not_swallowed() -> Result<()> {
        let (root, paths) = test_paths("managed-retirement-write-fails");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11518);
        // A directory where the manifest file should be makes the retirement
        // fail: the guarded write cannot read the record it would update.
        fs::create_dir_all(&record.manifest_path)?;

        let mut shown = Vec::new();
        let manifest_path = record.manifest_path.clone();
        let mut launch = LaunchRecord::new(&paths, record);
        retire_managed_launch_or_warn_to(&mut launch, &mut shown);
        let manifest_is_still_unwritten = manifest_path.is_dir();
        let outcome: Result<u32> =
            retire_record_on_error(&mut launch, Err(anyhow::anyhow!("spawn refused")));
        let _ = fs::remove_dir_all(root);

        let shown = String::from_utf8(shown)?;
        assert!(
            manifest_is_still_unwritten,
            "precondition: the retirement really did not reach disk"
        );
        assert!(
            shown.starts_with("warning: could not mark managed service `lemonade-qwen-3000`")
                && shown.contains(&manifest_path.display().to_string()),
            "the user must be shown which service and which record failed: {shown:?}"
        );
        let error = outcome.expect_err("the launch must still fail").to_string();
        assert_eq!(
            error, "spawn refused",
            "the launch's own error is what reaches the user"
        );
        Ok(())
    }

    /// The other half of [`retire_record_on_error`]: a step that succeeded must
    /// leave the record alone. A retirement on `Ok` would mark every healthy
    /// launch `"failed"` mid-launch, which `rocmd`'s recover watcher treats as
    /// restartable.
    #[test]
    fn retire_record_on_error_leaves_the_record_alone_on_success() -> Result<()> {
        let (root, paths) = test_paths("managed-retire-on-success");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11519);
        record.write()?;

        let mut launch = LaunchRecord::new(&paths, record.clone());
        let outcome = retire_record_on_error(&mut launch, Ok(7_u32));
        let on_disk: ManagedServiceRecord =
            serde_json::from_slice(&fs::read(&record.manifest_path)?)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(outcome?, 7, "the step's own value is handed back");
        assert_eq!(launch.record.status, "starting");
        assert_eq!(
            on_disk.status, "starting",
            "a step that succeeded must not retire the record on disk"
        );
        Ok(())
    }

    /// A child observed dead whose record cannot be retired still fails with
    /// the engine's own exit status and log tail. The retirement's write error
    /// is what that write produces, and it must not take the place of the only
    /// thing that says why the engine died.
    #[test]
    fn a_startup_death_keeps_its_diagnosis_when_the_retirement_cannot_be_written() -> Result<()> {
        let (root, paths) = test_paths("managed-death-unwritable-record");
        paths.ensure()?;
        let record = pre_spawn_record(&paths, 11520);
        if let Some(parent) = record.log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&record.log_path, "engine: GPU initialisation failed\n")?;
        // A directory where the manifest file should be makes the retirement fail.
        fs::create_dir_all(&record.manifest_path)?;
        let manifest_path = record.manifest_path.display().to_string();

        let outcome = fail_managed_launch_if_engine_died(
            &mut LaunchRecord::new(&paths, record),
            Some(crate::tests::exit_status_from_code(3)),
        );
        let _ = fs::remove_dir_all(root);

        let error = format!(
            "{:#}",
            outcome.expect_err("an engine that died at startup fails the launch")
        );
        assert!(
            error.contains("managed engine exited immediately with status"),
            "the engine's exit status must be what the launch reports: {error}"
        );
        assert!(
            error.contains("engine: GPU initialisation failed"),
            "the log tail must survive a retirement that could not be written: {error}"
        );
        assert!(
            !error.contains(&manifest_path),
            "the record's own error must not replace the diagnosis: {error}"
        );
        Ok(())
    }

    /// A Lemonade launch of `qwen` on loopback through `spawn`: the shape every
    /// launch-path test below shares. Lemonade, because it resolves no runtime,
    /// so nothing before the spawn can fail unless a test arranges it.
    fn launch_qwen_on_lemonade(
        paths: &AppPaths,
        port: u16,
        spawn: &SpawnManagedEngine<'_>,
    ) -> Result<ManagedSpawn> {
        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };
        spawn_managed_engine_child(
            paths,
            "lemonade",
            "lemonade-qwen-3000",
            "qwen",
            &resolve,
            "127.0.0.1",
            port,
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            false,
            spawn,
        )
    }

    /// The launch path's half of the persist-before-check ordering, driven through
    /// `spawn_managed_engine_child` itself. At the moment the startup check runs,
    /// the record on disk must already name the child — so if that check then
    /// cannot answer, what is left behind is a record a stop can reach rather than
    /// a pid-0 `"starting"` claim nothing can reach or demote.
    ///
    /// The check here cannot answer (as a Unix `try_wait` error), so this pins the
    /// record that outcome leaves too: still naming the child and still
    /// `"starting"`, not retired — the child's state is unknown, and freeing the
    /// engine + model could put a rival engine on its port.
    #[test]
    fn a_launch_names_its_child_on_record_before_the_startup_check() -> Result<()> {
        let (root, paths) = test_paths("managed-launch-persist-before-check");
        paths.ensure()?;
        let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
        let seen_at_check = std::rc::Rc::default();
        let spawn = stand_in_spawn(
            manifest_path.clone(),
            std::rc::Rc::clone(&seen_at_check),
            || bail!("cannot query the child"),
        );

        let outcome = launch_qwen_on_lemonade(&paths, 11521, &spawn);
        let after = manifest_on_disk(&manifest_path);
        let _ = fs::remove_dir_all(root);

        let Err(error) = outcome else {
            panic!("a launch whose startup check cannot answer must fail")
        };
        assert!(
            format!("{error:#}").contains("cannot query the child"),
            "the query failure is what reaches the user: {error:#}"
        );
        let at_check = seen_at_check
            .borrow_mut()
            .take()
            .expect("the record was on disk when the check ran");
        assert_eq!(
            (
                at_check.status.as_str(),
                at_check.supervisor_pid,
                at_check.engine_pid
            ),
            ("starting", stand_in_child_pid(), Some(stand_in_child_pid())),
            "the record must name the child before its startup check runs"
        );
        let after = after.expect("the record is still on disk");
        assert_eq!(
            (after.status.as_str(), after.supervisor_pid),
            ("starting", stand_in_child_pid()),
            "a check that cannot answer must leave the child on record, not retire it"
        );
        Ok(())
    }

    /// The launch path's startup death, through its real call site: the serve
    /// fails naming the child's log, and the record — already naming the child —
    /// is retired so the engine + model are free for the next `rocm serve`.
    #[test]
    fn a_launch_whose_engine_dies_at_startup_fails_and_frees_the_engine() -> Result<()> {
        let (root, paths) = test_paths("managed-launch-dies-at-startup");
        paths.ensure()?;
        let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
        let seen_at_check = std::rc::Rc::default();
        let spawn = stand_in_spawn(
            manifest_path.clone(),
            std::rc::Rc::clone(&seen_at_check),
            || Ok(Some(exit_status_from_code(3))),
        );

        let outcome = launch_qwen_on_lemonade(&paths, 11522, &spawn);
        let after = manifest_on_disk(&manifest_path);
        let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let log_path = paths.service_log_path("lemonade-qwen-3000");
        let _ = fs::remove_dir_all(root);

        let Err(error) = outcome else {
            panic!("an engine that died at startup must fail the launch")
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("managed engine exited immediately")
                && message.contains(&log_path.display().to_string()),
            "the failure must say the engine died and name its log: {message}"
        );
        assert_eq!(
            seen_at_check
                .borrow()
                .as_ref()
                .map(|record| record.supervisor_pid),
            Some(stand_in_child_pid()),
            "the record must name the child before its startup check runs"
        );
        let after = after.expect("the record is still on disk");
        assert_eq!(
            (after.status.as_str(), after.supervisor_pid),
            ("failed", stand_in_child_pid()),
            "a launch that died at startup is retired, still naming its child"
        );
        assert!(
            still_claimed.is_none(),
            "a launch that died at startup must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// The converse at the real call site: a child that survives its check is
    /// recorded `"running"` under its own pid and handed back for the readiness
    /// wait, so the check cannot reject a healthy launch — whether the check was
    /// still pending after the spawn (Unix) or answered inside it (Windows).
    #[test]
    fn a_launch_that_survives_its_startup_check_is_recorded_running() -> Result<()> {
        for (index, observed) in [false, true].into_iter().enumerate() {
            let (root, paths) = test_paths(&format!("managed-launch-survives-startup-{index}"));
            paths.ensure()?;
            let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
            let outcome = if observed {
                launch_qwen_on_lemonade(&paths, 11523, &stand_in_spawn_observed(None))
            } else {
                let spawn =
                    stand_in_spawn(manifest_path.clone(), std::rc::Rc::default(), || Ok(None));
                launch_qwen_on_lemonade(&paths, 11523, &spawn)
            };
            let after = manifest_on_disk(&manifest_path);
            let _ = fs::remove_dir_all(root);

            let ManagedSpawn::Spawned { child_pid, .. } = outcome? else {
                panic!("a fresh launch must spawn, not find an existing service")
            };
            assert_eq!(child_pid, stand_in_child_pid());
            let after = after.expect("the record is on disk");
            assert_eq!(
                (after.status.as_str(), after.supervisor_pid),
                ("running", stand_in_child_pid()),
                "observed by the spawn: {observed}"
            );
        }
        Ok(())
    }

    /// A death the spawn itself already saw (the Windows watched spawn) must not
    /// put the dead child on record. Off Linux its pid carries no start-time
    /// token, so a later `rocm services stop` on the retired record could not
    /// tell it from a process that reused the pid, and would signal that one.
    /// The launch is retired at pid 0 instead, exactly as if nothing had spawned.
    #[test]
    fn a_launch_the_spawn_saw_die_is_retired_without_naming_the_dead_child() -> Result<()> {
        let (root, paths) = test_paths("managed-launch-observed-death");
        paths.ensure()?;
        let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");

        let outcome = launch_qwen_on_lemonade(
            &paths,
            11530,
            &stand_in_spawn_observed(Some(exit_status_from_code(3))),
        );
        let after = manifest_on_disk(&manifest_path);
        let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        let Err(error) = outcome else {
            panic!("an engine that died at startup must fail the launch")
        };
        assert!(
            format!("{error:#}").contains("managed engine exited immediately"),
            "{error:#}"
        );
        let after = after.expect("the record is still on disk");
        assert_eq!(
            (
                after.status.as_str(),
                after.supervisor_pid,
                after.engine_pid
            ),
            ("failed", 0, None),
            "a child the spawn saw die must not be named by the retired record"
        );
        assert!(
            still_claimed.is_none(),
            "a launch that died at startup must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// A stand-in spawn that, before handing back its child, lets `stop` write
    /// to the service's record as a concurrent `rocm services stop` would — so
    /// the write lands between the launch's claim and its first write naming
    /// the child. The startup check is answered as `observed` says: by the
    /// spawn itself (Windows) or still pending, and then alive (Unix).
    fn spawn_racing_a_stop(
        manifest_path: PathBuf,
        observed: bool,
        stop: fn(&mut ManagedServiceRecord),
    ) -> impl Fn(&ManagedEngineSpawnRequest<'_>) -> Result<SpawnedManagedEngine> {
        move |_request| {
            let mut stopping =
                manifest_on_disk(&manifest_path).expect("the launch claimed its record first");
            stop(&mut stopping);
            stopping.write()?;
            Ok(SpawnedManagedEngine {
                pid: stand_in_child_pid(),
                startup: if observed {
                    ManagedEngineStartupCheck::Observed(None)
                } else {
                    ManagedEngineStartupCheck::Pending(Box::new(|| Ok(None)))
                },
            })
        }
    }

    /// A stop asked for while the launch spawns its engine persists its marker
    /// before it signals anything. The launch's writes after that go to the
    /// record as it is on disk, never the copy it claimed with, so the marker
    /// survives the child being put on record and the launch being promoted to
    /// `"running"` — and with it the daemon's `server-recover` watcher keeps
    /// off a service the user asked to stop.
    #[test]
    fn a_stop_asked_for_while_the_launch_spawns_survives_its_writes() -> Result<()> {
        for (index, observed) in [false, true].into_iter().enumerate() {
            let (root, paths) = test_paths(&format!("managed-launch-stop-marker-{index}"));
            paths.ensure()?;
            let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
            let spawn = spawn_racing_a_stop(manifest_path.clone(), observed, |stopping| {
                stopping.stop_requested_unix_ms = Some(4_242);
            });

            let outcome = launch_qwen_on_lemonade(&paths, 11533, &spawn);
            let after = manifest_on_disk(&manifest_path);
            let _ = fs::remove_dir_all(root);

            let ManagedSpawn::Spawned { child_pid, .. } = outcome? else {
                panic!("a fresh launch must spawn, not find an existing service")
            };
            assert_eq!(child_pid, stand_in_child_pid());
            let after = after.expect("the record is on disk");
            assert_eq!(
                (
                    after.status.as_str(),
                    after.supervisor_pid,
                    after.stop_requested_unix_ms
                ),
                ("running", stand_in_child_pid(), Some(4_242)),
                "the child is on record and the stop asked for meanwhile survives \
                 (observed by the spawn: {observed})"
            );
        }
        Ok(())
    }

    /// A stop that confirmed while the launch spawned — its claim names no
    /// process yet, so there is nothing for the stop to wait on — has written
    /// `"stopped"`. The launch must not write its child over that and bring the
    /// service back as `"running"`: it fails instead, saying the child it
    /// started was not recorded, and the stop's record stands.
    #[test]
    fn a_launch_does_not_resurrect_a_service_stopped_while_it_spawned() -> Result<()> {
        for (index, observed) in [false, true].into_iter().enumerate() {
            let (root, paths) = test_paths(&format!("managed-launch-stopped-{index}"));
            paths.ensure()?;
            let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
            let spawn = spawn_racing_a_stop(manifest_path.clone(), observed, |stopping| {
                stopping.status = "stopped".to_owned();
            });

            let outcome = launch_qwen_on_lemonade(&paths, 11534, &spawn);
            let after = manifest_on_disk(&manifest_path);
            let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
            let _ = fs::remove_dir_all(root);

            let Err(error) = outcome else {
                panic!(
                    "a launch whose service was stopped under it must fail (observed: {observed})"
                )
            };
            let message = format!("{error:#}");
            assert!(
                message.contains("was not recorded")
                    && message.contains(&stand_in_child_pid().to_string()),
                "the failure must say which child went unrecorded: {message}"
            );
            // The claim in that message, checked against the record it describes.
            let after = after.expect("the record is still on disk");
            assert_eq!(
                (
                    after.status.as_str(),
                    after.supervisor_pid,
                    after.engine_pid
                ),
                ("stopped", 0, None),
                "the stop's record stands and names no child (observed: {observed})"
            );
            assert!(
                still_claimed.is_none(),
                "a stopped service must not be claimed live again"
            );
        }
        Ok(())
    }

    /// A stop that confirmed during the startup check — the child was on record
    /// by then, so the stop could reach it — has written `"stopped"`. The check
    /// then reports the child dead, and the launch's retirement must leave the
    /// stop's verdict alone: rewritten as `"failed"`, the service is what the
    /// daemon's `server-recover` watcher restarts.
    #[test]
    fn a_launch_retiring_a_dead_child_leaves_a_stop_that_finished_during_its_check() -> Result<()> {
        let (root, paths) = test_paths("managed-launch-stop-during-check");
        paths.ensure()?;
        let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
        let spawn = {
            let manifest_path = manifest_path.clone();
            move |_request: &ManagedEngineSpawnRequest<'_>| -> Result<SpawnedManagedEngine> {
                let manifest_path = manifest_path.clone();
                Ok(SpawnedManagedEngine {
                    pid: stand_in_child_pid(),
                    startup: ManagedEngineStartupCheck::Pending(Box::new(move || {
                        let mut stopped = manifest_on_disk(&manifest_path)
                            .expect("the child is on record before its check");
                        stopped.status = "stopped".to_owned();
                        stopped.write()?;
                        Ok(Some(exit_status_from_code(143)))
                    })),
                })
            }
        };

        let outcome = launch_qwen_on_lemonade(&paths, 11535, &spawn);
        let after = manifest_on_disk(&manifest_path);
        let _ = fs::remove_dir_all(root);

        let Err(error) = outcome else {
            panic!("a launch whose engine is gone after its check must fail")
        };
        assert!(
            format!("{error:#}").contains("managed engine exited immediately"),
            "{error:#}"
        );
        let after = after.expect("the record is still on disk");
        assert_eq!(
            (after.status.as_str(), after.supervisor_pid),
            ("stopped", stand_in_child_pid()),
            "the stop's verdict stands; the launch does not retire over it"
        );
        Ok(())
    }

    /// A launch that got past its startup check, as `spawn_managed_engine_child`
    /// hands it to the readiness wait: on disk as `"running"`, naming the child.
    fn launched_record(paths: &AppPaths, port: u16) -> Result<ManagedServiceRecord> {
        let mut record = pre_spawn_record(paths, port);
        record.supervisor_pid = std::process::id();
        record.engine_pid = Some(std::process::id());
        record.status = "running".to_owned();
        record.write()?;
        Ok(manifest_on_disk(&record.manifest_path).expect("just written"))
    }

    /// The launch's last write lands after a readiness wait of up to 45 s, with
    /// the launch lock already released. A stop asked for in that window
    /// persists its marker first; the launch's verdict must not erase it.
    #[test]
    fn launch_readiness_write_keeps_a_stop_asked_for_while_it_waited() -> Result<()> {
        let (root, paths) = test_paths("launch-readiness-keeps-marker");
        paths.ensure()?;
        let launched = launched_record(&paths, 11536)?;
        let mut stopping = launched.clone();
        stopping.stop_requested_unix_ms = Some(1_234);
        stopping.write()?;

        let returned = record_launch_readiness(&paths, launched, EndpointReadiness::Serving)?;
        let on_disk = manifest_on_disk(&paths.service_manifest_path("lemonade-qwen-3000"));
        let _ = fs::remove_dir_all(root);

        let on_disk = on_disk.expect("the record is on disk");
        assert_eq!(
            (on_disk.status.as_str(), on_disk.stop_requested_unix_ms),
            ("ready", Some(1_234)),
            "the verdict is recorded and the stop survives it"
        );
        assert!(on_disk.inference_verified_at_unix_ms.is_some());
        assert_eq!(returned.stop_requested_unix_ms, Some(1_234));
        Ok(())
    }

    /// A stop that confirmed during the wait wrote `"stopped"`. The launch's
    /// verdict on a process that is gone must not overwrite it, and what the
    /// launch reports is the stop's record, not its own verdict.
    #[test]
    fn launch_readiness_write_leaves_a_stop_that_finished_meanwhile() -> Result<()> {
        let (root, paths) = test_paths("launch-readiness-after-stop");
        paths.ensure()?;
        let launched = launched_record(&paths, 11537)?;
        let mut finished = launched.clone();
        finished.status = "stopped".to_owned();
        finished.write()?;

        let returned = record_launch_readiness(&paths, launched, EndpointReadiness::Serving)?;
        let on_disk = manifest_on_disk(&paths.service_manifest_path("lemonade-qwen-3000"));
        let _ = fs::remove_dir_all(root);

        let on_disk = on_disk.expect("the record is on disk");
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.inference_verified_at_unix_ms, None);
        assert_eq!(
            returned.status, "stopped",
            "the caller reports the record as it is"
        );
        Ok(())
    }

    /// A restart that ran during the wait recorded its own child as the
    /// supervisor, still live. The record is that restart's now, and the
    /// launch's verdict on the child it replaced must not land on it.
    #[test]
    fn launch_readiness_write_leaves_a_record_a_restart_took_over() -> Result<()> {
        let (root, paths) = test_paths("launch-readiness-after-restart");
        paths.ensure()?;
        let launched = launched_record(&paths, 11540)?;
        let mut restarted = launched.clone();
        restarted.supervisor_pid = 4_242_424;
        restarted.supervisor_start_ticks = Some(7);
        restarted.status = "starting".to_owned();
        restarted.write()?;

        let returned = record_launch_readiness(&paths, launched, EndpointReadiness::Serving)?;
        let on_disk = manifest_on_disk(&paths.service_manifest_path("lemonade-qwen-3000"));
        let _ = fs::remove_dir_all(root);

        let on_disk = on_disk.expect("the record is on disk");
        assert_eq!(
            (on_disk.status.as_str(), on_disk.supervisor_pid),
            ("starting", 4_242_424),
            "the restart's record stands"
        );
        assert_eq!(on_disk.inference_verified_at_unix_ms, None);
        assert_eq!(returned.supervisor_pid, 4_242_424);
        Ok(())
    }

    /// What the guard must NOT refuse: the liveness refresh of any `services
    /// list` or dashboard during the wait demotes `"running"` to `"starting"`
    /// while the port is closed, and adopts the engine's server pid from its
    /// state file. Neither is a stop. The launch still records its verdict, on
    /// top of the adopted pid rather than over it.
    #[test]
    fn launch_readiness_write_survives_a_liveness_refresh_during_the_wait() -> Result<()> {
        let (root, paths) = test_paths("launch-readiness-after-refresh");
        paths.ensure()?;
        let launched = launched_record(&paths, 11538)?;
        let mut refreshed = launched.clone();
        refreshed.status = "starting".to_owned();
        refreshed.engine_pid = Some(4_242_424);
        refreshed.engine_start_ticks = Some(7);
        refreshed.write()?;

        let returned = record_launch_readiness(&paths, launched, EndpointReadiness::Serving)?;
        let on_disk = manifest_on_disk(&paths.service_manifest_path("lemonade-qwen-3000"));
        let _ = fs::remove_dir_all(root);

        let on_disk = on_disk.expect("the record is on disk");
        assert_eq!(
            (on_disk.status.as_str(), on_disk.engine_pid),
            ("ready", Some(4_242_424)),
            "the verdict lands, and the server pid the refresh adopted stays"
        );
        assert_eq!(returned.status, "ready");
        Ok(())
    }

    /// A spawn that fails outright is the last step of the launch's retirement
    /// span: no child exists, so the claiming record must be retired.
    #[test]
    fn a_launch_whose_spawn_fails_frees_the_engine_for_the_next_serve() -> Result<()> {
        let (root, paths) = test_paths("managed-launch-spawn-fails");
        paths.ensure()?;

        let outcome = launch_qwen_on_lemonade(&paths, 11524, &spawn_that_fails);
        let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        let Err(error) = outcome else {
            panic!("a launch whose spawn failed must fail")
        };
        assert!(
            format!("{error:#}").contains("failed to launch managed engine process"),
            "{error:#}"
        );
        assert!(
            still_claimed.is_none(),
            "a spawn that never started must not keep claiming the engine+model"
        );
        Ok(())
    }

    /// Two more steps inside the launch's retirement span, each made to fail for
    /// real: creating the service log (a directory sits where it goes) and
    /// creating the engine's state directory (a file sits where it goes).
    #[test]
    fn a_launch_that_cannot_prepare_its_files_frees_the_engine_for_the_next_serve() -> Result<()> {
        type Obstruct = fn(&AppPaths) -> std::io::Result<()>;
        let cases: [(&str, Obstruct); 2] = [
            ("service log", |paths| {
                fs::create_dir_all(paths.service_log_path("lemonade-qwen-3000"))
            }),
            ("engine state directory", |paths| {
                let state_dir = paths.engine_state_dir("lemonade");
                if state_dir.is_dir() {
                    fs::remove_dir_all(&state_dir)?;
                }
                fs::create_dir_all(state_dir.parent().expect("state dir has a parent"))?;
                fs::write(&state_dir, "not a directory")
            }),
        ];
        for (index, (what, obstruct)) in cases.into_iter().enumerate() {
            let (root, paths) = test_paths(&format!("managed-launch-files-{index}"));
            paths.ensure()?;
            obstruct(&paths)?;

            let outcome = launch_qwen_on_lemonade(&paths, 11525, &spawn_must_not_be_reached);
            let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
            let _ = fs::remove_dir_all(root);

            assert!(
                outcome.is_err(),
                "a launch that cannot create its {what} must fail"
            );
            assert!(
                still_claimed.is_none(),
                "a launch that could not create its {what} must not keep claiming the \
                 engine+model"
            );
        }
        Ok(())
    }

    /// The Windows adapter between the watched spawn and the shared startup
    /// check: an observed exit code has to arrive as that exit code, and a child
    /// still running has to arrive as "no exit".
    #[cfg(windows)]
    #[test]
    fn managed_startup_exit_reads_the_watched_exit_code() {
        let exited = rocm_core::DetachedSpawn {
            pid: 4242,
            early_exit_code: Some(7),
        };
        assert_eq!(
            managed_startup_exit(&exited).and_then(|status| status.code()),
            Some(7)
        );

        let running = rocm_core::DetachedSpawn {
            pid: 4242,
            early_exit_code: None,
        };
        assert!(managed_startup_exit(&running).is_none());
    }

    #[test]
    fn spawn_managed_engine_child_blocks_reuse_with_mismatched_recipe() -> Result<()> {
        // A live service recorded with one recipe (e.g. a tool-call parser flag)
        // must reject a relaunch requesting a different recipe rather than
        // silently reusing the old server, and the error must not claim the
        // mismatch is specifically about generation defaults when it could stem
        // from any recipe field.
        let (root, paths) = test_paths("dup-managed-recipe-mismatch");
        paths.ensure()?;
        let mut existing = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-1000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11510,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        existing.status = "ready".to_owned();
        existing.engine_pid = Some(std::process::id());
        existing.engine_recipe_json = Some(serde_json::to_string(&EngineRecipeHint {
            contract_version: rocm_engine_protocol::ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "lemonade".to_owned(),
            required_flags: vec!["--tool-call-parser".to_owned(), "hermes".to_owned()],
            ..EngineRecipeHint::default()
        })?);
        existing.write()?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };
        let requested_recipe = EngineRecipeHint {
            contract_version: rocm_engine_protocol::ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "lemonade".to_owned(),
            required_flags: vec!["--temperature".to_owned(), "0.5".to_owned()],
            ..EngineRecipeHint::default()
        };

        let result = spawn_managed_engine_child(
            &paths,
            "lemonade",
            "lemonade-qwen-2000",
            "qwen",
            &resolve,
            "127.0.0.1",
            11511,
            &resolve.device_policy,
            &[],
            None,
            None,
            Some(&requested_recipe),
            false,
            &spawn_managed_engine_process,
        );
        let _ = fs::remove_dir_all(root);

        let Err(error) = result else {
            panic!("mismatched recipe on a live service must be rejected")
        };
        let message = error.to_string();
        assert!(
            message.contains("different serve options"),
            "message should describe the mismatch generically: {message}"
        );
        assert!(
            message.contains("recipe hint, tool-call parser, or generation defaults"),
            "message should not single out generation defaults as the sole cause: {message}"
        );
        Ok(())
    }

    #[test]
    fn spawn_managed_engine_child_refuses_to_reuse_an_unauthenticated_service() -> Result<()> {
        // `--require-api-key` used to be accepted and dropped on this path. The
        // reuse branch returns before the flag is recorded and before the
        // endpoint key is checked, so a caller demanding auth got a server that
        // never had it, with no error. `rocm remote serve` then published that
        // endpoint onto the tailnet and printed a freshly minted key under "the
        // API key above is what stops anyone else calling it" — a false
        // assurance about an endpoint the whole tailnet can reach.
        //
        // Asserted through `spawn_managed_engine_child` rather than against the
        // guard's own arguments: the defect was the early return, so only the
        // real call site can fail for it.
        let (root, paths) = test_paths("dup-managed-unauthenticated-reuse");
        paths.ensure()?;
        let mut existing = ManagedServiceRecord::new(
            &paths,
            "lemonade-qwen-3000",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11520,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        existing.status = "ready".to_owned();
        existing.engine_pid = Some(std::process::id());
        // The state that matters: live, matching, and serving without auth.
        existing.requires_api_key = false;
        existing.write()?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };

        let result = spawn_managed_engine_child(
            &paths,
            "lemonade",
            "lemonade-qwen-3001",
            "qwen",
            &resolve,
            "127.0.0.1",
            11520,
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            // The demand that used to be silently discarded.
            true,
            &spawn_managed_engine_process,
        );
        let _ = fs::remove_dir_all(root);

        let Err(error) = result else {
            panic!(
                "reusing an unauthenticated service must not satisfy `--require-api-key`; a \
                 satisfied reuse leaves the endpoint open while the caller is told it is not"
            )
        };
        let message = error.to_string();
        assert!(
            message.contains("without authentication"),
            "the refusal must say why it refused: {message}"
        );
        // The printed hint must be the exact, directly-runnable command:
        // `main.rs`'s `service_actions_require_yes_and_render_sandbox_result`
        // proves this exact form (`rocm services stop <id> --yes`) is the one
        // the approval gate itself recommends. That running this form actually
        // clears *this* refusal — a service reused without the key it is now
        // being asked to require — is proven by
        // `@id:serve-require-api-key-refuses-unauthenticated-reuse`
        // (`tests/e2e-cucumber/features/model_serving.feature`, serve-24),
        // which reproduces this exact condition, runs the printed hint, and
        // asserts the record is left `stopped` — not by `stop_managed_services`
        // in `tests/e2e-cucumber/tests/e2e.rs`, which runs this same
        // invocation as best-effort teardown and never asserts success.
        assert!(
            message.contains("rocm services stop lemonade-qwen-3000 --yes"),
            "the refusal must name the way out, with the service to stop, directly runnable with --yes: {message}"
        );
        Ok(())
    }

    #[test]
    fn a_managed_spawn_refuses_an_invalid_key_file_on_a_service_that_requires_one() -> Result<()> {
        // Drives the real call site, not the guard's own arguments. The service
        // was launched with `--require-api-key`, and its key file is present but
        // empty — so `requires_api_key` is true while `key_present` is false,
        // which is the only way to reach the first branch. A test calling
        // `ensure_public_service_has_endpoint_key` directly cannot catch a
        // mis-wired call site, which is the defect that has occurred here twice.
        //
        // The flag is passed explicitly rather than inferred from the key file.
        // Inferring it marked every public bind as having demanded auth, because
        // a public bind always has a key file whether or not it asked for one.
        let (root, paths) = test_paths("managed-spawn-invalid-key");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // Empty, so the file exists (requires_api_key = true) but yields no
        // usable key (key_present = false).
        endpoint_keys::store_endpoint_api_key(&paths, "lemonade-qwen-3000", "")?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };

        let result = spawn_managed_engine_child(
            &paths,
            "lemonade",
            "lemonade-qwen-3000",
            "qwen",
            &resolve,
            // Loopback on purpose: the public-bind branch must not be what
            // refuses this, or the test would pass with the guard disabled.
            "127.0.0.1",
            11512,
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            true,
            &spawn_managed_engine_process,
        );
        // The refusal fires after the claiming record is written, so this is a
        // real launch path abandoning its pid-0 `"starting"` record — the one
        // `retire_record_on_error` exists for. Read before the cleanup below.
        let still_claimed = existing_live_managed_service(&paths, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        let Err(error) = result else {
            panic!("a service requiring a key must not spawn with an unusable key file")
        };
        let message = error.to_string();
        assert!(
            message.contains("--require-api-key"),
            "the refusal must name the flag the service was launched with: {message}"
        );
        assert!(
            message.contains("without authentication"),
            "the refusal must say what the risk is: {message}"
        );
        assert!(
            still_claimed.is_none(),
            "a refused launch must retire its record, or every later `rocm serve` for \
             this engine + model is refused as already running"
        );
        Ok(())
    }

    /// A second real launch-path failure after the record write, at a different
    /// call site from the key-guard refusal above: the runtime the launch names
    /// has no installed manifest, so resolving its engine environment fails
    /// before any child is spawned. No GPU, engine, or network is involved.
    #[test]
    fn a_managed_spawn_naming_a_missing_runtime_frees_the_engine_for_the_next_serve() -> Result<()>
    {
        let (root, paths) = test_paths("managed-spawn-missing-runtime");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "vllm".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        };

        let result = spawn_managed_engine_child(
            &paths,
            // vLLM, not Lemonade: Lemonade manages its own runtime and never
            // looks the runtime manifest up, so it would not reach this failure.
            "vllm",
            "vllm-qwen-3000",
            "qwen",
            &resolve,
            "127.0.0.1",
            11516,
            &resolve.device_policy,
            &[],
            Some("no-such-runtime"),
            None,
            None,
            false,
            &spawn_managed_engine_process,
        );
        let still_claimed = existing_live_managed_service(&paths, "vllm", "qwen-canonical");
        let _ = fs::remove_dir_all(root);

        assert!(
            result.is_err(),
            "a launch naming an uninstalled runtime must fail before spawning"
        );
        assert!(
            still_claimed.is_none(),
            "a launch that failed before its spawn must not keep claiming the engine+model"
        );
        Ok(())
    }

    #[test]
    fn resolve_endpoint_auth_loopback_stays_credential_free() {
        // Loopback binds never require auth, even if a key is supplied.
        for host in ["127.0.0.1", "localhost", "::1"] {
            assert_eq!(resolve_endpoint_auth(host, None, false).unwrap(), None);
            assert_eq!(
                resolve_endpoint_auth(host, Some("ignored"), false).unwrap(),
                None
            );
        }
    }

    #[test]
    fn resolve_endpoint_auth_loopback_can_be_required_when_something_republishes_it() {
        // "Loopback" describes the bind address, not who can reach the port. A
        // tailnet publish, a proxy, or a container port map all leave the bind
        // loopback while widening the audience, and the default policy would
        // hand out an unauthenticated endpoint. Whoever widens the reach asks
        // for the credential explicitly.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let generated = resolve_endpoint_auth(host, None, true)
                .unwrap()
                .expect("a required key must be generated, not skipped");
            assert!(!generated.trim().is_empty());

            assert_eq!(
                resolve_endpoint_auth(host, Some("supplied-key"), true).unwrap(),
                Some("supplied-key".to_owned()),
                "a supplied key must be honoured rather than ignored as it is by default"
            );
        }

        // The same validation a public bind gets: an empty key is a refusal, not
        // a silent downgrade to no auth.
        assert!(resolve_endpoint_auth("127.0.0.1", Some("  "), true).is_err());
    }

    #[test]
    fn resolve_endpoint_auth_public_uses_supplied_key_trimmed() {
        let key = resolve_endpoint_auth("0.0.0.0", Some("  my-key  "), false)
            .unwrap()
            .expect("public bind must have a key");
        assert_eq!(key, "my-key");
    }

    #[test]
    fn resolve_endpoint_auth_public_generates_key_when_absent() {
        let key = resolve_endpoint_auth("0.0.0.0", None, false)
            .unwrap()
            .expect("public bind must generate a key");
        assert_eq!(key.len(), 48);
        assert!(key.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn resolve_endpoint_auth_public_rejects_empty_supplied_key() {
        let error = resolve_endpoint_auth("0.0.0.0", Some("   "), false).unwrap_err();
        assert!(error.to_string().contains("non-empty"), "{error:#}");
    }

    #[test]
    fn resolve_endpoint_auth_public_rejects_embedded_crlf() {
        // A supplied key survives `trim()` with embedded CR/LF intact and would
        // otherwise be interpolated into a raw `Authorization: Bearer` header,
        // injecting an extra header line. It must be rejected at input validation.
        for supplied in [
            "good-key\r\nX-Injected: value",
            "good-key\nmore",
            "line\rreturn",
        ] {
            let error = resolve_endpoint_auth("0.0.0.0", Some(supplied), false).unwrap_err();
            assert!(error.to_string().contains("control character"), "{error:#}");
        }
    }

    #[test]
    fn drop_orphaned_endpoint_key_on_already_running_clears_stored_key() {
        let (root, paths) = test_paths("drop-orphaned-key-stored");
        let service_id = "svc-orphaned";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "secret-key").unwrap();

        drop_orphaned_endpoint_key_on_already_running(&paths, service_id, Some("secret-key"));

        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert!(!endpoint_keys::endpoint_key_file_path(&paths, service_id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn drop_orphaned_endpoint_key_on_already_running_is_noop_for_loopback() {
        // A loopback attempt never stores a key (`freshly_stored == None`), so the
        // helper must not panic or error, and no file must appear.
        let (root, paths) = test_paths("drop-orphaned-key-loopback");
        let service_id = "svc-loopback";

        drop_orphaned_endpoint_key_on_already_running(&paths, service_id, None);

        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert!(!endpoint_keys::endpoint_key_file_path(&paths, service_id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn public_bind_fails_closed_for_windows_lemonade_only() {
        // Windows + Lemonade + public bind: refuse (cannot enforce the key).
        let error = ensure_public_bind_engine_supported("lemonade", true, true).unwrap_err();
        assert!(error.to_string().contains("lemonade"), "{error:#}");
        // Every other combination is allowed:
        ensure_public_bind_engine_supported("vllm", true, true).unwrap(); // vLLM enforces auth on Windows
        ensure_public_bind_engine_supported("lemonade", true, false).unwrap(); // non-Windows
        ensure_public_bind_engine_supported("lemonade", false, true).unwrap(); // loopback needs no key
    }

    #[test]
    fn a_public_bind_is_refused_with_the_command_that_restores_it() {
        // `resolve_endpoint_auth` mints a key for every non-loopback bind whether
        // or not auth was demanded, so deriving `requires_api_key` from key-file
        // presence marked ordinary public binds as having asked for it. The guard
        // tests that field first, so those services were refused with a message
        // naming a flag they never passed and a relaunch command that drops
        // `--allow-public-bind` — coming back on loopback instead.
        //
        // A plain `--host 0.0.0.0 --allow-public-bind` launch: key file present,
        // `--require-api-key` never passed.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false)
            .expect_err("a public bind with no key must be refused");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("--allow-public-bind"),
            "the refusal must name the command that restores the public bind: {rendered}"
        );
        assert!(
            !rendered.contains("--require-api-key"),
            "a service that never passed the flag must not be told it did: {rendered}"
        );

        // And the loopback-with-forced-auth case still reports its own reason.
        let error = ensure_public_service_has_endpoint_key("127.0.0.1", false, true)
            .expect_err("a service that demanded auth must not come back without it");
        assert!(
            format!("{error:#}").contains("--require-api-key"),
            "{error:#}"
        );
    }

    #[test]
    fn respawn_fails_closed_for_a_public_service_whose_key_is_gone() {
        // A stop deletes the key file, so a later restart of a public service
        // would otherwise respawn it with no auth at all.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("0.0.0.0"), "{error:#}");
        assert!(message.contains("without authentication"), "{error:#}");
        // Actionable: name the command that mints a fresh key.
        assert!(message.contains("--allow-public-bind"), "{error:#}");

        // A public service that still has its key restarts normally.
        ensure_public_service_has_endpoint_key("0.0.0.0", true, false).unwrap();
    }

    #[test]
    fn respawn_allows_loopback_services_without_an_endpoint_key() {
        // Loopback stays credential-free, so every accepted spelling must pass
        // the guard with no key present — when nothing asked for auth.
        for host in ["127.0.0.1", "localhost", "::1"] {
            ensure_public_service_has_endpoint_key(host, false, false)
                .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
        }
    }

    #[test]
    fn respawn_refuses_a_loopback_service_that_was_launched_with_a_key() {
        // The hole this closes: a loopback bind that something else republishes
        // — a tailnet publish, a proxy, a container port map. The publish
        // outlives the process, so a restart after the key was dropped would
        // reopen a reachable endpoint with no authentication, and the bind
        // address gives the guard no way to notice.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let error = ensure_public_service_has_endpoint_key(host, false, true)
                .expect_err("a service launched with a key must not restart without one");
            let message = format!("{error:#}");
            assert!(message.contains("without authentication"), "{message}");
            assert!(message.contains("--require-api-key"), "{message}");
        }

        // With its key still present it restarts normally.
        ensure_public_service_has_endpoint_key("127.0.0.1", true, true).unwrap();
    }

    #[test]
    fn endpoint_client_config_shows_key_once_with_bearer_guidance() {
        let rendered = render_endpoint_client_config("http://0.0.0.0:11435/v1", "secret-123");
        // Dummy literal exercising the intentional one-time key display documented
        // on `render_endpoint_client_config`; not a real credential or a log write.
        // codeql[rust/cleartext-logging]
        assert!(rendered.contains("secret-123"), "{rendered}");
        // codeql[rust/cleartext-logging]: see rationale above.
        assert!(rendered.contains("Authorization: Bearer"), "{rendered}");
        // codeql[rust/cleartext-logging]: see rationale above.
        assert!(rendered.contains("shown only now"), "{rendered}");
    }

    /// Build an `AutomationRuntimeState` for the no-double-spawn guard tests.
    fn runtime_state(running: bool, daemon_pid: u32) -> AutomationRuntimeState {
        AutomationRuntimeState {
            running,
            automations_enabled: true,
            daemon_pid,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: Vec::new(),
        }
    }

    #[test]
    fn background_helper_already_running_true_for_live_pid() {
        // Phase-10 daemon no-double-spawn: a runtime-state.json with running=true
        // and a LIVE daemon_pid (this very test process) means the helper is
        // already up — the "should spawn?" decision must say NO (true ⇒ skip).
        // Hermetic + offline: no spawn, just the file-based liveness check.
        let (root, paths) = test_paths("helper-live-pid");
        runtime_state(true, std::process::id())
            .write(&paths)
            .expect("write runtime state");
        assert!(
            background_helper_already_running(&paths).expect("liveness check ok"),
            "live recorded pid + running=true ⇒ do not spawn a second daemon"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn background_helper_already_running_false_for_dead_or_missing() {
        // The inverse guard cases — each must report NOT running (false ⇒ spawn):
        // (1) no state file at all, (2) running=true but a dead/zero pid,
        // (3) a live pid but running=false. None must spawn from this decision.
        let (root, paths) = test_paths("helper-dead-or-missing");
        // (1) No state file yet.
        assert!(
            !background_helper_already_running(&paths).expect("missing state ⇒ ok"),
            "no runtime state ⇒ not running"
        );
        // (2) running=true but pid 0 is never a live process.
        runtime_state(true, 0).write(&paths).expect("write state");
        assert!(
            !background_helper_already_running(&paths).expect("dead pid ⇒ ok"),
            "running=true + dead pid ⇒ not running (spawn)"
        );
        // (3) live pid but running flag is false.
        runtime_state(false, std::process::id())
            .write(&paths)
            .expect("write state");
        assert!(
            !background_helper_already_running(&paths).expect("not-running flag ⇒ ok"),
            "running=false ⇒ not running even with a live pid"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn autostart_spawn_in_flight_defers_only_for_a_live_recent_claim() {
        let claim = AutostartClaim {
            daemon_pid: 4242,
            spawned_at_ms: 1_000,
        };
        // A recent claim whose PID is still alive ⇒ a spawn is in flight, defer.
        assert!(autostart_spawn_in_flight(
            Some(claim),
            1_000 + 5_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
        // Same claim but the child PID is gone (crashed spawn) ⇒ respawn.
        assert!(!autostart_spawn_in_flight(
            Some(claim),
            1_000 + 5_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| false,
        ));
        // Older than the TTL, even with a live PID (possible PID reuse) ⇒ respawn.
        assert!(!autostart_spawn_in_flight(
            Some(claim),
            1_000 + AUTOSTART_CLAIM_TTL_MS,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
        // No claim at all ⇒ nothing in flight, spawn.
        assert!(!autostart_spawn_in_flight(
            None,
            1_000,
            AUTOSTART_CLAIM_TTL_MS,
            |_| true,
        ));
    }

    #[test]
    fn autostart_claim_round_trips_and_absent_or_garbage_reads_as_none() {
        let (root, paths) = test_paths("autostart-claim-roundtrip");
        let path = paths.automation_autostart_claim_path();
        fs::create_dir_all(path.parent().expect("claim parent")).expect("mkdir claim dir");
        // Absent file ⇒ no claim.
        assert_eq!(read_autostart_claim(&path), None);
        // Round-trip a written claim.
        let claim = AutostartClaim {
            daemon_pid: 9987,
            spawned_at_ms: 1_726_000_000_123,
        };
        write_autostart_claim(&path, claim).expect("write claim");
        assert_eq!(read_autostart_claim(&path), Some(claim));
        // Garbage / partial content ⇒ no claim (permits a respawn, never panics).
        fs::write(&path, "not-a-pid").expect("write garbage");
        assert_eq!(read_autostart_claim(&path), None);
        let _ = fs::remove_dir_all(&root);
    }
}
