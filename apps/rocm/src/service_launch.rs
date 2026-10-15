// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Spawn, attach to, and stream logs from a managed background service
//! process, plus the endpoint-key/auth checks that gate it.
//!
//! Mechanically relocated from `main.rs` with no behavior change. This module
//! is the managed-service-spawning tail that was deliberately deferred out of
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
#[cfg(not(windows))]
use std::process::ExitStatus;
#[cfg(not(windows))]
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
#[cfg(test)]
use rocm_core::DEFAULT_LOCAL_HOST;
use rocm_core::{AppPaths, AutomationRuntimeState, EndpointReadiness, ManagedServiceRecord};
use rocm_core::{format_http_base_url, process_is_running};
use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};

use crate::endpoint_keys;
use crate::engines_cmd::env_root_for_service;
use crate::serve_port;
use crate::service_hints;
use crate::{
    SandboxToolArg, builtin_engine_serve_http_args, device_policy_name,
    existing_live_managed_service_in, load_managed_services, managed_service_is_live,
    managed_service_launcher_path, record_cli_audit_event, render_service_action_result,
    run_internal_sandbox_tool, status_for_readiness, wait_for_service_http_ready_with_progress,
};
#[cfg(windows)]
use crate::{app_path_env_var_refs, app_path_env_var_values};
#[cfg(not(windows))]
use crate::{apply_app_path_env, read_optional_tail_lines};

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

#[cfg(not(windows))]
pub(crate) fn managed_engine_startup_failure_detail(status: ExitStatus, log_path: &Path) -> String {
    let mut recent_lines = read_optional_tail_lines(log_path, 80, "service log");
    if recent_lines.is_empty() {
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(120));
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

/// Outcome of the locked allocation step.
///
/// `Debug` so tests can assert on `Result::expect_err`.
#[derive(Debug)]
enum ManagedPortDecision {
    /// An equivalent managed service is already live; nothing may be spawned.
    /// Boxed to keep the variants a similar size (clippy::large_enum_variant).
    AlreadyLive(Box<ManagedServiceRecord>),
    /// The concrete port this launch owns until it hands it to the child.
    Lease(rocm_core::LoopbackPortLease),
}

/// Live managed services, with liveness refreshed by [`load_managed_services`]
/// (dead PIDs demote to "stopped" and therefore neither satisfy the duplicate
/// check nor reserve a port).
///
/// A read failure degrades to "nothing is live": the OS preflight still guards
/// the port, and refusing to serve because the services directory could not be
/// listed would be a worse failure than launching.
fn live_managed_services(paths: &AppPaths) -> Vec<ManagedServiceRecord> {
    load_managed_services(paths)
        .unwrap_or_default()
        .into_iter()
        .filter(managed_service_is_live)
        .collect()
}

/// Ports that live managed services still own.
///
/// Conservative by design: a record reserves its port regardless of the host it
/// recorded, because a `0.0.0.0` (or `localhost`) listener also blocks the
/// loopback address a new local server would bind. Over-reserving costs one
/// candidate in a 101-port scan; under-reserving costs a collision.
fn reserved_service_ports(live: &[ManagedServiceRecord]) -> Vec<serve_port::ReservedPort> {
    live.iter()
        .map(|record| serve_port::ReservedPort {
            port: record.port,
            service_id: record.service_id.clone(),
            status: record.status.clone(),
        })
        .collect()
}

/// The decision half of the launch transaction: refresh liveness, apply the
/// `(engine, canonical_model_id)` idempotency guard, then lease a concrete port
/// against both live records and the real OS.
///
/// Callers MUST hold [`rocm_core::AppPaths::managed_launch_lock_path`] across
/// this call *and* the record publication that follows, or two launches can
/// pick the same candidate.
/// `probe` is injected so tests can drive deterministic bind outcomes without a
/// GPU or a spawn.
fn resolve_managed_port_in_transaction(
    paths: &AppPaths,
    engine: &str,
    canonical_model_id: &str,
    host: &str,
    request: serve_port::PortRequest,
    probe: &mut dyn FnMut(u16) -> std::io::Result<rocm_core::LoopbackPortLease>,
) -> Result<ManagedPortDecision> {
    let live = live_managed_services(paths);
    // Idempotency first: an already-live equivalent service means no port is
    // needed at all, so it must be answered before any candidate is leased.
    if let Some(existing) = existing_live_managed_service_in(&live, engine, canonical_model_id) {
        return Ok(ManagedPortDecision::AlreadyLive(Box::new(existing.clone())));
    }
    let reserved = reserved_service_ports(&live);
    serve_port::resolve_serve_port(host, request, &reserved, probe).map(ManagedPortDecision::Lease)
}

/// Grace given to a child that must be terminated because its record could not
/// be published. Short: the child is milliseconds old and has nothing to flush.
const MANAGED_LAUNCH_CLEANUP_GRACE: Duration = Duration::from_secs(2);

/// Undo a failed managed launch attempt before the launch lock is released, so
/// no failed transaction leaves a live reservation or a stray endpoint key.
///
/// Order matters: a spawned child is terminated and confirmed dead *first*, so
/// nothing is still holding the port or writing to the log by the time cleanup
/// runs. If no child was ever spawned (`record.engine_pid` is still `None`),
/// every artifact this attempt created is removed: there is nothing for
/// `rocm services`/`rocm logs` to show. If a child did run, its record and log
/// are kept instead (marked `status = "failed"`), since the error that led here
/// already told the user to go inspect that log; only the endpoint key, which a
/// failed attempt has no use for, is cleared.
/// Every step is best-effort and idempotent; cleanup must never mask the
/// original failure with a second one.
fn abandon_failed_managed_launch(paths: &AppPaths, record: &ManagedServiceRecord) {
    if let Some(pid) = record.engine_pid.filter(|pid| *pid != 0) {
        let outcome = rocm_core::terminate_verified(
            &rocm_core::ProcessIdentity::capture(pid),
            rocm_core::KillScope::Tree,
            MANAGED_LAUNCH_CLEANUP_GRACE,
            true,
        );
        record_cli_audit_event(
            paths,
            "service",
            "managed_service_launch_cleanup",
            "warn",
            format!(
                "terminated engine child pid={pid} after a failed launch publication outcome={}",
                outcome.as_str()
            ),
            Some(&record.service_id),
        );
        let mut failed = record.clone();
        failed.status = "failed".to_owned();
        let _ = failed.write();
        endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
        return;
    }
    for path in [
        record.manifest_path.as_path(),
        record.log_path.as_path(),
        record.engine_state_path.as_path(),
    ] {
        let _ = fs::remove_file(path);
    }
    endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
}

/// Spawn the detached engine child shared by the managed (background) and
/// attached (`--verbose`/`--foreground`) serve paths. Returns before the HTTP
/// readiness wait; callers decide whether to block on readiness
/// ([`start_managed_service`]) or start tailing the log immediately
/// ([`run_attached_service`]).
///
/// The whole body is one serialized transaction under the shared allocation
/// lock: duplicate detection, live-reservation refresh, candidate leasing,
/// manifest/log creation, argv preparation, the bounded immediate-exit check,
/// and record publication. Readiness deliberately runs after the lock is
/// released.
#[allow(clippy::too_many_arguments)]
fn spawn_managed_engine_child(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port_request: serve_port::PortRequest,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    require_api_key: bool,
) -> Result<ManagedSpawn> {
    paths.ensure()?;
    fs::create_dir_all(paths.services_dir())?;

    // No lock is acquired here: every production caller
    // ([`start_managed_service`], [`run_attached_service`]) already holds
    // [`rocm_core::AppPaths::managed_launch_lock_path`] for the whole of this
    // call, so no two transactions can choose the same port or publish
    // competing records for one engine+model. A second acquisition in this
    // frame would self-deadlock, since the lock is exclusive per open file
    // description, not per process.
    //
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
    let lease = match resolve_managed_port_in_transaction(
        paths,
        engine,
        &resolve.canonical_model_id,
        host,
        port_request,
        &mut serve_port::loopback_bind_probe,
    )? {
        ManagedPortDecision::AlreadyLive(existing) => {
            let existing = *existing;
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
        ManagedPortDecision::Lease(lease) => lease,
    };

    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine,
        requested_model,
        resolve.canonical_model_id.clone(),
        host,
        lease.port(),
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
    // Publishing the record inside the lock is what reserves the leased port for
    // every other transaction. From here on, any failure must remove it again.
    record.write()?;

    match launch_managed_engine_child_locked(
        paths,
        &mut record,
        lease,
        engine,
        service_id,
        resolve,
        host,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
    ) {
        Ok(child_pid) => Ok(ManagedSpawn::Spawned {
            record: Box::new(record),
            child_pid,
        }),
        Err(error) => {
            // The single cleanup choke point for every failure path below the
            // record's first write: `record.engine_pid` tells it whether a child
            // ever ran, so a pre-spawn failure deletes everything and a post-spawn
            // one keeps the record and log (marked `status = "failed"`) instead.
            abandon_failed_managed_launch(paths, &record);
            Err(error)
        }
    }
}

/// The side-effecting half of the launch transaction: create the log, prepare
/// concrete argv, hand the leased port to the child, spawn it, reject a child
/// that has already exited, and publish the running record.
///
/// Split out of [`spawn_managed_engine_child`] so that *every* failure below the
/// record's first write is funnelled through one cleanup path before the
/// launch lock is released. Returns the child pid.
#[allow(clippy::too_many_arguments)]
fn launch_managed_engine_child_locked(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    lease: rocm_core::LoopbackPortLease,
    engine: &str,
    service_id: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Result<u32> {
    let port = record.port;
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
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
        &record.engine_state_path,
        Some(&record.log_path),
    )?;
    let engine_envs_root = env_root_for_service(paths, engine, runtime_id, env_id)?;
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
    ensure_public_service_has_endpoint_key(
        host,
        endpoint_key_file.is_some(),
        record.requires_api_key,
    )?;
    // Hand the port over: close the probe socket immediately before the spawn,
    // still under the caller's launch lock, so no other ROCm transaction can
    // take it in the meantime. On a custom host the lease never held a socket
    // and this is a no-op — the engine owns that bind.
    debug_assert_eq!(lease.port(), port);
    lease.release();
    #[cfg(windows)]
    let child_pid = {
        let env_values = app_path_env_var_values(paths, engine_envs_root.as_deref());
        let mut env_refs = app_path_env_var_refs(&env_values);
        if let Some(key_file) = endpoint_key_file.as_deref() {
            env_refs.push((rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file));
        }
        rocm_core::spawn_detached_no_inherit(&current_exe, &serve_args, &env_refs)
            .context("failed to launch managed engine process")?
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
            .context("failed to launch managed engine process")?;
        let child_pid = child.id();
        // Set before the immediate-exit check below, so a bail! here still leaves
        // `record.engine_pid` telling cleanup that a child was spawned.
        record.engine_pid = Some(child_pid);
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check managed engine startup state")?
        {
            bail!(
                "{}",
                managed_engine_startup_failure_detail(status, &record.log_path)
            );
        }
        child_pid
    };
    // The child is both roles until the engine state file names a separate
    // server process. Each role gets its own token, captured while the child is
    // alive: a stop pairs every recorded PID with its own token, and the two
    // PIDs diverge as soon as `rocmd` restarts the service — at which point an
    // engine PID recorded without one could only be signalled blind.
    record.record_supervisor_identity(child_pid);
    record.record_engine_identity(child_pid);
    record.status = "running".to_owned();
    // A surviving child is `running`, never `ready`: it holds a process, not a
    // proven endpoint. If it cannot be recorded it would be unsupervised and
    // invisible while holding the port, so terminate it rather than leak it.
    if let Err(error) = record.write() {
        // Cleanup is the caller's job now: `record.engine_pid` (set above) is
        // already enough for `abandon_failed_managed_launch` to terminate this
        // child, so there is nothing left for this inner path to do beyond
        // surfacing the error.
        return Err(error.context("failed to publish the managed service record"));
    }

    Ok(child_pid)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_managed_service(
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port_request: serve_port::PortRequest,
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
    let (mut record, child_pid) = match spawn_managed_engine_child(
        &paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port_request,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        engine_recipe,
        require_api_key,
    )? {
        ManagedSpawn::AlreadyRunning(report) => return Ok(report),
        ManagedSpawn::Spawned { record, child_pid } => (*record, child_pid),
    };
    // The claiming service record is now persisted, so the selected GPU is
    // visible to any concurrent auto-selection. Release the launch lock before
    // the readiness wait below, which can block for many seconds — holding it
    // that long would needlessly serialize unrelated serves.
    drop(launch_lock);

    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));

    let readiness = wait_for_service_http_ready_with_progress(
        engine,
        host,
        record.port,
        &resolve.canonical_model_id,
        endpoint_api_key,
        Duration::from_secs(45),
        on_wait_tick,
    );
    let launch_status = status_for_readiness(readiness);
    record.status = launch_status.to_owned();
    if readiness == EndpointReadiness::Serving {
        // Latch the verification the wait just performed, so the readiness checks
        // behind `services list` and chat read it instead of re-probing.
        record.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
    }
    record.write()?;
    let endpoint_url = format!("{}/v1", format_http_base_url(host, record.port));
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
        status: launch_status.to_owned(),
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
        "  note: this key is shown only now — clients must send `Authorization: Bearer <key>`"
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
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port_request: serve_port::PortRequest,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    endpoint_api_key: Option<&str>,
    launch_lock: rocm_core::FileLock,
    require_api_key: bool,
) -> Result<()> {
    let spawn = spawn_managed_engine_child(
        paths,
        engine,
        service_id,
        requested_model,
        resolve,
        host,
        port_request,
        &resolve.device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        resolve.engine_recipe.as_ref(),
        require_api_key,
    )
    // Mirrors the background path's cleanup in `serve_cmd.rs`: a pre-spawn
    // failure (port reserved/in use, range exhausted, other bind error) returns
    // before any service record exists, so nothing else clears the key just
    // stored for this attempt.
    .inspect_err(|_| {
        drop_orphaned_endpoint_key_on_already_running(paths, service_id, endpoint_api_key);
    })?;
    // The claiming record is persisted (or an existing service was found), so the
    // selected GPU is now visible to concurrent auto-selection. Release the launch
    // lock before streaming logs, which blocks for the whole attached session.
    drop(launch_lock);

    let (service_id, log_path, port, child_pid) = match spawn {
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
            drop_orphaned_endpoint_key_on_already_running(paths, service_id, endpoint_api_key);
            return Ok(());
        }
        ManagedSpawn::Spawned { record, child_pid } => (
            service_id.to_owned(),
            record.log_path.clone(),
            record.port,
            child_pid,
        ),
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
    println!("  streaming engine logs — Ctrl-D detaches (leaves it running), Ctrl-C stops it");
    println!();

    let outcome = stream_attached_logs(&log_path, child_pid)?;
    println!();

    match outcome {
        AttachOutcome::Detach => {
            println!("detached — server still running");
            println!("  service_id: {service_id}");
            println!("  endpoint: {endpoint}");
            println!("  list: rocm services");
            println!("  logs: rocm logs {service_id}");
            println!("  stop: {}", service_hints::service_stop_hint(&service_id));
            record_cli_audit_event(
                paths,
                "service",
                "serve_detach",
                "info",
                format!("detached from attached serve service_id={service_id} endpoint={endpoint}"),
                Some(&service_id),
            );
            Ok(())
        }
        AttachOutcome::Stop => {
            println!("stopping server…");
            match run_internal_sandbox_tool(
                paths,
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
                paths,
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
    use crate::tests::test_paths;

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
            serve_port::PortRequest::Explicit(11511),
            &resolve.device_policy,
            &[],
            None,
            None,
            Some(&requested_recipe),
            false,
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
            serve_port::PortRequest::Explicit(11520),
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            // The demand that used to be silently discarded.
            true,
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
            serve_port::PortRequest::Explicit(11512),
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            true,
        );

        // This guard fires before any child is spawned, so `record.engine_pid`
        // is still `None` when `spawn_managed_engine_child`'s cleanup arm calls
        // `abandon_failed_managed_launch` — proving its pre-spawn branch
        // (delete everything) runs from the real call site, not just from the
        // `abandoned_launch_leaves_no_record_log_or_endpoint_key` test that
        // calls the helper directly.
        let manifest_path = paths.service_manifest_path("lemonade-qwen-3000");
        let log_path = paths.service_log_path("lemonade-qwen-3000");
        let engine_state_path = paths.service_engine_state_path("lemonade", "lemonade-qwen-3000");
        let key = endpoint_keys::endpoint_api_key(&paths, "lemonade-qwen-3000");
        let live = live_managed_services(&paths);
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
            !manifest_path.exists(),
            "a pre-spawn failure must not leave a record"
        );
        assert!(
            !log_path.exists(),
            "a pre-spawn failure must not leave a log"
        );
        assert!(
            !engine_state_path.exists(),
            "a pre-spawn failure must not leave engine state"
        );
        assert_eq!(
            key, None,
            "a pre-spawn failure must not leave the unusable key behind"
        );
        assert!(
            live.is_empty(),
            "a pre-spawn failure must leave no reservation"
        );
        Ok(())
    }

    #[test]
    fn a_failed_spawn_past_the_first_write_terminates_the_child_and_keeps_a_failed_record()
    -> Result<()> {
        // Drives the real call site's *post*-spawn cleanup branch, which
        // neither `abandoned_launch_*` test reaches (they call
        // `abandon_failed_managed_launch` directly). An engine name this
        // binary does not build in makes the real spawned child (a second
        // `rocm` process) exit almost immediately via
        // `run_builtin_engine_serve_http`'s `other => bail!(...)` arm, after
        // `launch_managed_engine_child_locked` has already set
        // `record.engine_pid` — so cleanup must terminate (already-dead) the
        // child, keep the record and log for inspection, mark the record
        // `failed`, and clear only the endpoint key.
        let (root, paths) = test_paths("managed-spawn-unknown-engine-cleanup");
        paths.ensure()?;
        let service_id = "no-such-engine-qwen-4000";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "test-key")?;

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
            "no-such-engine",
            service_id,
            "qwen",
            &resolve,
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11_514),
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            false,
        );

        let manifest_path = paths.service_manifest_path(service_id);
        let log_path = paths.service_log_path(service_id);
        let persisted = load_managed_services(&paths)
            .expect("re-read manifests")
            .into_iter()
            .find(|candidate| candidate.service_id == service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let live = live_managed_services(&paths);

        let Err(error) = result else {
            panic!("an engine this binary does not build in must not report success")
        };
        let message = error.to_string();
        assert!(
            message.contains("managed engine exited immediately"),
            "the failure must come from the spawned child dying, not an earlier guard: {message}"
        );
        assert!(
            manifest_path.exists(),
            "a post-spawn failure must keep the record for inspection"
        );
        assert!(
            log_path.exists(),
            "a post-spawn failure must keep the log for inspection"
        );
        let persisted =
            persisted.expect("a post-spawn failure must leave the record on disk, not delete it");
        assert_eq!(
            persisted.status, "failed",
            "a post-spawn failure must mark the record failed, not leave it stuck"
        );
        assert_eq!(
            key, None,
            "a post-spawn failure must still clear the endpoint key"
        );
        assert!(
            live.is_empty(),
            "a failed record must not still reserve its port"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn failed_port_reservation_leaves_no_orphaned_endpoint_key() -> Result<()> {
        // serve() stores the new attempt's endpoint key before calling
        // start_managed_service. If spawn_managed_engine_child fails inside
        // resolve_managed_port_in_transaction (explicit port already reserved
        // by another live service), it returns before any ManagedServiceRecord
        // exists for this attempt, so abandon_failed_managed_launch never runs.
        // The caller's .inspect_err() cleanup (drop_orphaned_endpoint_key_on_already_running)
        // must be what clears the key.
        let (root, paths) = test_paths("orphaned-key-on-reserved-port");
        paths.ensure()?;
        let mut other = ManagedServiceRecord::new(
            &paths,
            "lemonade-llama-1000",
            "lemonade",
            "llama",
            "llama-canonical",
            "127.0.0.1",
            11513,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        other.status = "ready".to_owned();
        other.engine_pid = Some(std::process::id());
        other.write()?;

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

        let service_id = "lemonade-qwen-2000";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "test-key")?;

        let result = spawn_managed_engine_child(
            &paths,
            "lemonade",
            service_id,
            "qwen",
            &resolve,
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11513),
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            false,
        );
        let Err(error) = result else {
            let _ = fs::remove_dir_all(root);
            panic!("explicit port already reserved by a live service must be rejected")
        };
        assert!(
            error
                .to_string()
                .contains("already reserved by managed service"),
            "unexpected error: {error}"
        );

        drop_orphaned_endpoint_key_on_already_running(&paths, service_id, Some("test-key"));
        assert!(
            endpoint_keys::endpoint_api_key(&paths, service_id).is_none(),
            "cleanup must drop the key this failed attempt staged before launching"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn attached_launch_failing_before_spawn_leaves_no_orphaned_endpoint_key() -> Result<()> {
        // The background path's cleanup lives in serve_cmd.rs, wrapping
        // start_managed_service. run_attached_service is the other caller of
        // spawn_managed_engine_child and needs the identical .inspect_err()
        // cleanup, not a simulated one: drive the real function end to end.
        let (root, paths) = test_paths("orphaned-key-attached-reserved-port");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let mut other = ManagedServiceRecord::new(
            &paths,
            "lemonade-llama-1000",
            "lemonade",
            "llama",
            "llama-canonical",
            "127.0.0.1",
            11514,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        other.status = "ready".to_owned();
        other.engine_pid = Some(std::process::id());
        other.write()?;

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

        let service_id = "lemonade-qwen-attached-2000";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "test-key")?;
        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let result = run_attached_service(
            &paths,
            "lemonade",
            service_id,
            "qwen",
            &resolve,
            "127.0.0.1",
            serve_port::PortRequest::Explicit(11514),
            &[],
            None,
            None,
            Some("test-key"),
            launch_lock,
            false,
        );
        let Err(error) = result else {
            let _ = fs::remove_dir_all(root);
            panic!("explicit port already reserved by a live service must be rejected")
        };
        assert!(
            error
                .to_string()
                .contains("already reserved by managed service"),
            "unexpected error: {error}"
        );
        assert!(
            endpoint_keys::endpoint_api_key(&paths, service_id).is_none(),
            "a pre-spawn failure must drop the key this attempt staged before launching"
        );

        let _ = fs::remove_dir_all(root);
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

        let live_services = live_managed_services(&paths);
        let found =
            existing_live_managed_service_in(&live_services, "lemonade", "qwen-canonical").cloned();
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

        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
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

        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
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
        let live = live_managed_services(&paths);
        let found = existing_live_managed_service_in(&live, "lemonade", "qwen-canonical");
        let _ = fs::remove_dir_all(root);
        assert!(found.is_none());
    }

    /// A live managed record: own PID + a live status, so liveness refresh keeps
    /// it (and therefore its port reservation) in place.
    fn write_live_service_record(
        paths: &AppPaths,
        service_id: &str,
        canonical_model_id: &str,
        port: u16,
        status: &str,
    ) -> Result<ManagedServiceRecord> {
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "lemonade",
            canonical_model_id,
            canonical_model_id.to_owned(),
            DEFAULT_LOCAL_HOST,
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = status.to_owned();
        record.engine_pid = Some(std::process::id());
        record.write()?;
        Ok(record)
    }

    fn never_probe(_port: u16) -> std::io::Result<rocm_core::LoopbackPortLease> {
        panic!("no port may be probed on this path")
    }

    fn always_free(port: u16) -> std::io::Result<rocm_core::LoopbackPortLease> {
        Ok(rocm_core::LoopbackPortLease::engine_owned(port))
    }

    #[test]
    fn allocation_transaction_answers_an_equivalent_live_service_before_leasing() -> Result<()> {
        let (root, paths) = test_paths("alloc-duplicate-first");
        paths.ensure()?;
        write_live_service_record(
            &paths,
            "lemonade-qwen-1",
            "qwen-canonical",
            11_501,
            "starting",
        )?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "qwen-canonical",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Auto,
            &mut never_probe,
        );
        let _ = fs::remove_dir_all(root);

        match decision? {
            ManagedPortDecision::AlreadyLive(existing) => {
                assert_eq!(existing.service_id, "lemonade-qwen-1");
                assert_eq!(existing.port, 11_501);
            }
            ManagedPortDecision::Lease(lease) => {
                panic!("expected the duplicate guard, leased {}", lease.port())
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_skips_ports_reserved_by_other_live_services() -> Result<()> {
        let (root, paths) = test_paths("alloc-skip-reserved");
        paths.ensure()?;
        write_live_service_record(&paths, "lemonade-a-1", "model-a", 11_435, "running")?;
        write_live_service_record(&paths, "lemonade-b-1", "model-b", 11_436, "recovering")?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-c",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Auto,
            &mut always_free,
        );
        let _ = fs::remove_dir_all(root);

        match decision? {
            // A `recovering` record owns its port just as firmly as a running
            // one: CLI automatic selection must not race daemon recovery for it.
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_437),
            ManagedPortDecision::AlreadyLive(existing) => {
                panic!("unexpected duplicate match {}", existing.service_id)
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_ignores_stopped_records() -> Result<()> {
        let (root, paths) = test_paths("alloc-ignore-stopped");
        paths.ensure()?;
        let mut stale = ManagedServiceRecord::new(
            &paths,
            "lemonade-stale-1",
            "lemonade",
            "model-a",
            "model-a".to_owned(),
            DEFAULT_LOCAL_HOST,
            11_435,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        stale.status = "ready".to_owned();
        stale.engine_pid = Some(999_999_999);
        stale.write()?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-a",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Auto,
            &mut always_free,
        );
        let _ = fs::remove_dir_all(root);

        match decision? {
            // Dead PIDs demote to "stopped", so neither the duplicate guard nor
            // the reservation applies: the first candidate is free again.
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_435),
            ManagedPortDecision::AlreadyLive(existing) => {
                panic!(
                    "a stopped record must not block relaunch: {}",
                    existing.service_id
                )
            }
        }
        Ok(())
    }

    #[test]
    fn stopping_the_reserving_record_frees_its_explicit_port_for_reuse() -> Result<()> {
        // Proves the mechanism `rocm services stop` relies on: it never touches
        // the port directly, it only flips the stopped record's `status` (see
        // the `all_stopped` branch in `main.rs`'s stop handler). This recreates
        // that exact disk state and checks the same explicit port a live
        // competitor refused now resolves, closing the loop the E2E
        // `assert_port_reserved_message` step only checks by wording.
        let (root, paths) = test_paths("alloc-explicit-port-freed-by-stop");
        paths.ensure()?;
        let mut holder = ManagedServiceRecord::new(
            &paths,
            "lemonade-a-1",
            "lemonade",
            "model-a",
            "model-a".to_owned(),
            DEFAULT_LOCAL_HOST,
            11_435,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        holder.status = "running".to_owned();
        holder.engine_pid = Some(std::process::id());
        holder.write()?;

        let still_reserved = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-b",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Explicit(11_435),
            &mut always_free,
        );
        assert!(
            matches!(still_reserved, Err(ref error) if error.to_string().contains("already reserved by managed service lemonade-a-1")),
            "unexpected result while still running: {still_reserved:?}"
        );

        holder.status = "stopped".to_owned();
        holder.write()?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-b",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Explicit(11_435),
            &mut always_free,
        );
        let _ = fs::remove_dir_all(root);

        match decision? {
            ManagedPortDecision::Lease(lease) => assert_eq!(lease.port(), 11_435),
            ManagedPortDecision::AlreadyLive(existing) => {
                panic!(
                    "a stopped record must not block reuse: {}",
                    existing.service_id
                )
            }
        }
        Ok(())
    }

    #[test]
    fn allocation_transaction_rejects_an_explicit_port_a_recovering_service_holds() -> Result<()> {
        let (root, paths) = test_paths("alloc-explicit-recovering");
        paths.ensure()?;
        write_live_service_record(&paths, "lemonade-rec-1", "model-a", 11_435, "recovering")?;

        let decision = resolve_managed_port_in_transaction(
            &paths,
            "lemonade",
            "model-b",
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Explicit(11_435),
            &mut never_probe,
        );
        let _ = fs::remove_dir_all(root);

        let error = decision.expect_err("a recovering service still owns its port");
        let message = format!("{error:#}");
        assert!(message.contains("lemonade-rec-1"), "{message}");
        assert!(message.contains("recovering"), "{message}");
        Ok(())
    }

    #[test]
    fn concurrent_auto_transactions_publish_distinct_ports() -> Result<()> {
        use std::sync::{Arc, Barrier};

        let (root, paths) = test_paths("alloc-concurrent-distinct");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let barrier = Arc::new(Barrier::new(2));

        let mut handles = Vec::new();
        for model in ["model-a", "model-b"] {
            let paths = paths.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || -> Result<u16> {
                barrier.wait();
                // Exactly the production transaction: one lock around decision
                // and publication, real sockets for the preflight. Unbounded,
                // like `select_gpu_indices_under_launch_lock`'s acquire that a
                // real `serve()` holds across this same window — not
                // `acquire_bounded`, which only `rocmd`'s recovery path uses.
                let _launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
                let lease = match resolve_managed_port_in_transaction(
                    &paths,
                    "lemonade",
                    model,
                    DEFAULT_LOCAL_HOST,
                    serve_port::PortRequest::Auto,
                    &mut serve_port::loopback_bind_probe,
                )? {
                    ManagedPortDecision::Lease(lease) => lease,
                    ManagedPortDecision::AlreadyLive(existing) => {
                        anyhow::bail!("unexpected duplicate {}", existing.service_id)
                    }
                };
                let port =
                    write_live_service_record(&paths, model, model, lease.port(), "starting")?.port;
                lease.release();
                Ok(port)
            }));
        }

        let ports: Vec<u16> = handles
            .into_iter()
            .map(|handle| handle.join().expect("transaction thread"))
            .collect::<Result<Vec<u16>>>()?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(ports.len(), 2);
        assert_ne!(
            ports[0], ports[1],
            "two concurrent automatic launches must not publish the same port"
        );
        for port in ports {
            assert!((serve_port::AUTO_PORT_FIRST..=serve_port::AUTO_PORT_LAST).contains(&port));
        }
        Ok(())
    }

    #[test]
    fn concurrent_equivalent_transactions_converge_on_one_record() -> Result<()> {
        use std::sync::{Arc, Barrier};

        let (root, paths) = test_paths("alloc-concurrent-equivalent");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;
        let barrier = Arc::new(Barrier::new(2));

        let mut handles = Vec::new();
        for id in ["lemonade-dup-1", "lemonade-dup-2"] {
            let paths = paths.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || -> Result<bool> {
                barrier.wait();
                let _launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
                match resolve_managed_port_in_transaction(
                    &paths,
                    "lemonade",
                    "shared-model",
                    DEFAULT_LOCAL_HOST,
                    serve_port::PortRequest::Auto,
                    &mut serve_port::loopback_bind_probe,
                )? {
                    ManagedPortDecision::Lease(lease) => {
                        write_live_service_record(
                            &paths,
                            id,
                            "shared-model",
                            lease.port(),
                            "starting",
                        )?;
                        lease.release();
                        Ok(true)
                    }
                    ManagedPortDecision::AlreadyLive(_) => Ok(false),
                }
            }));
        }

        let launched: Vec<bool> = handles
            .into_iter()
            .map(|handle| handle.join().expect("transaction thread"))
            .collect::<Result<Vec<bool>>>()?;
        let published = live_managed_services(&paths).len();
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            launched.iter().filter(|launched| **launched).count(),
            1,
            "exactly one of two equivalent requests may launch"
        );
        assert_eq!(published, 1, "the loser must not publish a second record");
        Ok(())
    }

    #[test]
    fn select_gpu_indices_under_launch_lock_blocks_for_the_full_hold_and_then_sees_the_holders_write()
    -> Result<()> {
        // The standing gap this closes: nothing exercised the real lock-acquisition
        // function production callers use (`select_gpu_indices_under_launch_lock`)
        // against a lock another thread actually holds, so a refactor that dropped
        // the lock early (or stopped threading it through) would not fail any test.
        // A real end-to-end spawn test is avoided here per the fix-forward plan:
        // `spawn_managed_engine_child` is still exercised for real, but only via its
        // `AlreadyLive` short-circuit (a pre-seeded duplicate record), so no child
        // process is ever launched.
        use std::sync::{Arc, Barrier};
        use std::time::Instant;

        let (root, paths) = test_paths("launch-lock-holds-across-spawn");
        paths.ensure()?;
        write_live_service_record(&paths, "lemonade-held-1", "held-canonical", 11_520, "ready")?;

        let barrier = Arc::new(Barrier::new(2));
        let hold_for = Duration::from_millis(250);

        let holder_paths = paths.clone();
        let holder_barrier = Arc::clone(&barrier);
        let holder = thread::spawn(move || -> Result<()> {
            let _lock = rocm_core::FileLock::acquire(holder_paths.managed_launch_lock_path())?;
            holder_barrier.wait();
            thread::sleep(hold_for);
            Ok(())
        });

        barrier.wait();
        let started = Instant::now();
        let (gpu_indices, launch_lock) =
            crate::select_gpu_indices_under_launch_lock(&paths, true, None, || None, None, None)?;
        let waited = started.elapsed();

        let resolve = ResolveModelResponse {
            canonical_model_id: "held-canonical".to_owned(),
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
        let spawn = spawn_managed_engine_child(
            &paths,
            "lemonade",
            "lemonade-held-2",
            "held-model",
            &resolve,
            DEFAULT_LOCAL_HOST,
            serve_port::PortRequest::Auto,
            &resolve.device_policy,
            &gpu_indices,
            None,
            None,
            None,
            false,
        );
        drop(launch_lock);
        holder.join().expect("holder thread")?;
        let _ = fs::remove_dir_all(root);

        // `thread::sleep` only guarantees *at least* `hold_for`, but under a
        // loaded host (e.g. a full parallel `cargo test` run on WSL2) timer
        // coalescing can make the measured wait a hair short of the nominal
        // hold. Allow a small margin so the test still fails loudly if the
        // call returns early, without being flaky on actual block duration.
        let margin = Duration::from_millis(20);
        assert!(
            waited + margin >= hold_for,
            "select_gpu_indices_under_launch_lock returned before the holder released the lock: waited {waited:?}, held {hold_for:?}"
        );
        match spawn? {
            ManagedSpawn::AlreadyRunning(report) => {
                assert_eq!(report.service_id, "lemonade-held-1");
            }
            ManagedSpawn::Spawned { .. } => {
                panic!(
                    "expected the pre-seeded AlreadyLive record to short-circuit, not a real spawn"
                )
            }
        }
        Ok(())
    }

    #[test]
    fn abandoned_launch_leaves_no_record_log_or_endpoint_key() -> Result<()> {
        let (root, paths) = test_paths("alloc-abandon-artifacts");
        paths.ensure()?;
        let mut record =
            write_live_service_record(&paths, "lemonade-fail-1", "model-a", 11_435, "starting")?;
        // Simulate a pre-spawn failure: nothing ever ran.
        record.engine_pid = None;
        fs::create_dir_all(record.log_path.parent().expect("log parent"))?;
        fs::write(&record.log_path, b"engine log")?;
        if let Some(parent) = record.engine_state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&record.engine_state_path, b"{}")?;
        endpoint_keys::store_endpoint_api_key(&paths, &record.service_id, "secret-key")?;

        abandon_failed_managed_launch(&paths, &record);

        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let state_exists = record.engine_state_path.exists();
        let key = endpoint_keys::endpoint_api_key(&paths, &record.service_id);
        // The reservation must be gone too: nothing may still claim the port.
        let live = live_managed_services(&paths);
        let _ = fs::remove_dir_all(root);

        assert!(!manifest_exists, "a failed attempt must not leave a record");
        assert!(!log_exists, "a failed attempt must not leave a log");
        assert!(
            !state_exists,
            "a failed attempt must not leave engine state"
        );
        assert_eq!(key, None, "a failed attempt must not leave an endpoint key");
        assert!(
            live.is_empty(),
            "a failed attempt must leave no reservation"
        );
        Ok(())
    }

    #[test]
    #[cfg(not(windows))]
    #[ignore = "spawned as a child process by abandoned_launch_terminates_the_spawned_child"]
    fn managed_launch_cleanup_sleeper_child() {
        if std::env::var_os("ROCM_TEST_SLEEPER_CHILD").is_none() {
            return;
        }
        thread::sleep(Duration::from_mins(2));
    }

    #[test]
    #[cfg(not(windows))]
    fn abandoned_launch_terminates_the_spawned_child() -> Result<()> {
        let (root, paths) = test_paths("alloc-abandon-child");
        paths.ensure()?;
        let mut record =
            write_live_service_record(&paths, "lemonade-fail-2", "model-a", 11_435, "starting")?;

        // A real, long-lived child stands in for an engine process that was
        // spawned but whose record could not be published.
        let mut child = ProcessCommand::new(std::env::current_exe().expect("test binary"))
            .arg("tests::managed_launch_cleanup_sleeper_child")
            .arg("--exact")
            .arg("--ignored")
            .env("ROCM_TEST_SLEEPER_CHILD", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleeper child");
        let child_pid = child.id();
        record.engine_pid = Some(child_pid);

        abandon_failed_managed_launch(&paths, &record);

        // On Unix a killed direct child stays a zombie — and therefore "running"
        // to a `kill(pid, 0)` probe — until its parent reaps it, so prove death
        // by reaping, then confirm the pid is gone.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut exited = None;
        while exited.is_none() && std::time::Instant::now() < deadline {
            exited = child.try_wait().expect("query sleeper child");
            if exited.is_none() {
                thread::sleep(Duration::from_millis(50));
            }
        }
        let terminated = exited.is_some();
        if !terminated {
            // Cleanup did not kill it: do so here rather than leaking a sleeper
            // for two minutes, and reap it so no zombie outlives the test.
            let _ = child.kill();
        }
        // Reaped on every path, including the deadline path above.
        let _ = child.wait();
        let still_running = !terminated || process_is_running(child_pid);
        let persisted = load_managed_services(&paths)
            .expect("re-read manifests")
            .into_iter()
            .find(|candidate| candidate.service_id == record.service_id);
        let _ = fs::remove_dir_all(root);

        assert!(
            terminated,
            "a child that could not be published must be terminated"
        );
        assert!(!still_running, "the child pid must be gone after cleanup");
        let persisted = persisted.expect(
            "a launch that spawned a child must keep its record so `rocm services`/`rocm logs` can show the attempt",
        );
        assert_eq!(
            persisted.status, "failed",
            "a spawned child's record must be marked failed, not deleted"
        );
        Ok(())
    }
}
