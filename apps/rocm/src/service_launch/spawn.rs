// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Spawning the detached managed engine child shared by the background and
//! attached serve paths: process and stdio plumbing, [`ManagedLaunchReport`] and
//! [`ManagedSpawn`], and the serialized spawn transaction itself.

use super::launch_transaction::{
    ManagedPortDecision, ManagedReuseRefusal, abandon_failed_managed_launch,
    managed_service_reuse_refusal, no_usable_gpu_refusal, resolve_managed_port_in_transaction,
};
use crate::endpoint_keys;
use crate::engine_ipc::builtin_engine_serve_http_args;
use crate::engine_runtime_select::managed_service_launcher_path;
#[cfg(windows)]
use crate::engine_runtime_select::{app_path_env_var_refs, app_path_env_var_values};
use crate::engines_cmd::env_root_for_service;
use crate::serve_port;
use crate::service_hints;
use crate::services_text::read_optional_tail_lines;
use crate::{device_policy_name, record_cli_audit_event};
use anyhow::{Context, Result, bail};
#[cfg(windows)]
use rocm_core::process_is_running;
use rocm_core::{AppPaths, ManagedServiceRecord};
use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};
use std::fs;
#[cfg(not(windows))]
use std::io;
use std::path::{Path, PathBuf};
#[cfg(not(windows))]
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

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

/// Windows spawn-liveness check: wait out the startup window, then fail with the
/// startup log when the child is already gone. The wait and the liveness probe
/// are injected so a unit test runs the exit-detected path on every platform,
/// including the Windows lane that actually calls it.
///
/// # Errors
/// Fails with the startup-failure detail when `is_running` reports the child dead.
#[cfg_attr(not(windows), allow(dead_code))]
fn ensure_spawned_child_survives_startup(
    child_pid: u32,
    log_path: &Path,
    settle: impl FnOnce(),
    is_running: impl FnOnce(u32) -> bool,
) -> Result<()> {
    settle();
    if is_running(child_pid) {
        Ok(())
    } else {
        bail!("{}", managed_engine_startup_failure_detail("", log_path))
    }
}

/// `status_clause` is a pre-formatted fragment inserted right after "exited
/// immediately" (e.g. `" with status exit status: 1"` on Unix, where a real
/// `ExitStatus` is available; empty on Windows, where the liveness check
/// below only has a yes/no answer, not an exit code).
pub(crate) fn managed_engine_startup_failure_detail(
    status_clause: &str,
    log_path: &Path,
) -> String {
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
            "managed engine exited immediately{status_clause}; inspect {}",
            log_path.display()
        );
    }
    format!(
        "managed engine exited immediately{status_clause}; inspect {}\n\nrecent startup log output:\n{}",
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
/// Split out of [`start_managed_service`](super::start_managed_service) so the
/// attached (`--verbose` / `--foreground`) serve path can spawn the very same
/// detached child and stream its log live from the first line — including
/// startup — instead of blocking on the readiness wait before any output appears.
pub(super) enum ManagedSpawn {
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
/// ([`start_managed_service`](super::start_managed_service)) or start tailing the
/// log immediately ([`run_attached_service`](super::run_attached_service)).
///
/// The whole body is one serialized transaction under the shared allocation
/// lock: duplicate detection, live-reservation refresh, candidate leasing,
/// manifest/log creation, argv preparation, the bounded immediate-exit check,
/// and record publication. Readiness deliberately runs after the lock is
/// released.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_managed_engine_child(
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
    gpu_refusal_waived_for_reuse: bool,
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
            // Asked through the shared predicate rather than re-tested here, so this
            // refusal and `serve()`'s reuse pre-gate cannot disagree about whether a
            // service is reusable. When they disagreed, the pre-gate let a
            // non-reusable service bypass the no-GPU refusal and the user was sent
            // here to be told to stop a working server for nothing.
            //
            // The same device policy the serve path passes. This side used to pass
            // `None` to keep the spawn's acceptance untouched, but that opted the
            // spawn out of a term it has the input for, left its own
            // `DevicePolicyTooWeak` bail unreachable, and made the serve path's
            // argument droppable. Nothing observable widens: `parse_device_policy`
            // is the only producer of a `DevicePolicy` on every path that reaches
            // here and it rejects `cpu_only`, so no record this CLI writes can
            // carry the policy this term refuses.
            match managed_service_reuse_refusal(
                &existing,
                requested_recipe_json.as_deref(),
                require_api_key,
                device_policy,
            ) {
                Some(ManagedReuseRefusal::RecipeMismatch) => bail!(
                    "managed service `{}` is already running for engine `{engine}` and model `{}` with different serve options (recipe hint, tool-call parser, or generation defaults); stop it and run `rocm serve` again to apply the requested options",
                    existing.service_id,
                    resolve.canonical_model_id
                ),
                Some(ManagedReuseRefusal::MissingApiKey) => bail!(
                    "managed service `{}` is already running for engine `{engine}` and model `{}` \
                     without authentication, and a running server cannot be given a key it did not \
                     start with; stop it with `{}` and run the command again to \
                     serve it with `--require-api-key`",
                    existing.service_id,
                    resolve.canonical_model_id,
                    service_hints::service_stop_hint(&existing.service_id)
                ),
                // Reachable only from a record whose `device_policy` says
                // `cpu_only`, which no `rocm serve` can write today
                // (`parse_device_policy` refuses that policy outright) — so this is
                // a guard against a record written by some future path, not a
                // refusal users see now. The advice still names a command that
                // clears the condition: stopping the service removes the only thing
                // this term can object to, and the rerun then launches normally
                // (AGENTS.md section 3).
                Some(ManagedReuseRefusal::DevicePolicyTooWeak) => bail!(
                    "managed service `{}` is already running under a device policy that cannot \
                     satisfy this request; stop it with `{}` and run the command again",
                    existing.service_id,
                    service_hints::service_stop_hint(&existing.service_id)
                ),
                None => {}
            }
            // An explicit port is a demand, not a hint: reporting success on a
            // different port than the one asked for would silently drop it.
            if let Some(port) = port_request.explicit()
                && port != existing.port
            {
                bail!(
                    "managed service `{}` is already serving engine `{engine}` and model `{}` on port {}, so port {port} would not be used; stop it with `{}` and run the command again to serve on that port",
                    existing.service_id,
                    resolve.canonical_model_id,
                    existing.port,
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

    // Nothing live to reuse — and if the only reason this invocation got past
    // `serve()`'s no-usable-GPU pre-flight was a reusable service, that premise
    // has just been falsified. `serve()` decided it from a snapshot taken before
    // GPU selection and the launch lock; the check above is the authoritative
    // one, so a service that died in between (an OOM, say) would otherwise fall
    // through to a real spawn on a host with no usable GPU, and the user would
    // get whatever the engine says about having no device instead of the
    // actionable refusal the pre-flight exists to give. Raise it here, where the
    // premise is known to be false.
    if gpu_refusal_waived_for_reuse {
        bail!("{}", no_usable_gpu_refusal(device_policy));
    }

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
    record.served_model_id = resolve.served_model_id.clone();
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
        #[cfg(not(windows))]
        None,
    ) {
        Ok(child_pid) => Ok(ManagedSpawn::Spawned {
            record: Box::new(record),
            child_pid,
        }),
        Err(error) => {
            // The single cleanup choke point for every failure path below the
            // record's first write: `record.engine_pid` tells it whether a child
            // ever ran, so a pre-spawn failure deletes everything and a post-spawn
            // one keeps the record and log (marked `status = "stopped"`) instead.
            abandon_failed_managed_launch(paths, &record);
            Err(error)
        }
    }
}

// Test-only, thread-scoped spawn interceptor for callers that cannot take a
// `spawn_override` argument (`start_managed_service`, `run_attached_service`):
// lets a test observe state at the instant production would spawn the child.
#[cfg(all(test, not(windows)))]
thread_local! {
    pub(super) static TEST_SPAWN_HOOK: std::cell::RefCell<Option<std::rc::Rc<SpawnOverride<'static>>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, not(windows)))]
fn test_spawn_hook() -> Option<std::rc::Rc<SpawnOverride<'static>>> {
    TEST_SPAWN_HOOK.with(|hook| hook.borrow().clone())
}

#[cfg(not(windows))]
type SpawnOverride<'a> = dyn Fn(&mut ProcessCommand) -> io::Result<std::process::Child> + 'a;

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
    // Test-only seam: lets a test assert the leased port is actually bindable
    // (i.e. released) at the exact point production would spawn the child,
    // without needing to fake or intercept the real spawn mechanics.
    #[cfg(not(windows))] spawn_override: Option<&SpawnOverride<'_>>,
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
    rocm_core::ensure_public_service_has_endpoint_key(
        host,
        endpoint_key_file.is_some(),
        record.requires_api_key,
        rocm_core::ServiceStart::Launch,
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
        let child_pid = rocm_core::spawn_detached_no_inherit(&current_exe, &serve_args, &env_refs)
            .context("failed to launch managed engine process")?;
        // Set before the immediate-exit check below, so a bail! here still leaves
        // `record.engine_pid` telling cleanup that a child was spawned. The
        // start-time token is captured here too, while the child is certainly
        // ours: cleanup binds its kill to it rather than to whatever owns the
        // pid by then.
        record.record_engine_identity(child_pid);
        ensure_spawned_child_survives_startup(
            child_pid,
            &record.log_path,
            || thread::sleep(Duration::from_millis(200)),
            process_is_running,
        )?;
        child_pid
    };
    #[cfg(not(windows))]
    let child_pid = {
        let mut command = managed_service_process_command(&current_exe, &serve_args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, Some(&record.log_path))?;
        detach_background_command(&mut command);
        paths.apply_env(&mut command);
        if let Some(engine_envs_root) = engine_envs_root.as_deref() {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        if let Some(key_file) = endpoint_key_file.as_deref() {
            command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        }
        #[cfg(all(test, not(windows)))]
        let hook = test_spawn_hook();
        #[cfg(all(test, not(windows)))]
        let spawn_override = spawn_override.or(hook.as_deref());
        let mut child = match spawn_override {
            Some(spawn) => spawn(&mut command),
            None => command.spawn(),
        }
        .context("failed to launch managed engine process")?;
        let child_pid = child.id();
        // Set before the immediate-exit check below, so a bail! here still leaves
        // `record.engine_pid` telling cleanup that a child was spawned. The
        // start-time token is captured here too, while the child is certainly
        // ours: cleanup binds its kill to it rather than to whatever owns the
        // pid by then.
        record.record_engine_identity(child_pid);
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check managed engine startup state")?
        {
            bail!(
                "{}",
                managed_engine_startup_failure_detail(
                    &format!(" with status {status}"),
                    &record.log_path
                )
            );
        }
        child_pid
    };
    // The child is both roles until the engine state file names a separate
    // server process. Each role gets its own token, captured while the child is
    // alive: a stop pairs every recorded PID with its own token, and the two
    // PIDs diverge as soon as `rocmd` restarts the service — at which point an
    // engine PID recorded without one could only be signalled blind.
    // The engine's token was already captured right after the spawn (so cleanup
    // can bind to it); re-capturing here could replace it with a later owner's.
    record.record_supervisor_identity(child_pid);
    record.status = "running".to_owned();
    if let Err(error) = record.write() {
        // Cleanup is the caller's job now: `record.engine_pid` (set above) is
        // already enough for `abandon_failed_managed_launch` to terminate this
        // child, so there is nothing left for this inner path to do beyond
        // surfacing the error.
        return Err(error.context("failed to publish the managed service record"));
    }

    Ok(child_pid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managed_service_status::load_managed_services;
    use crate::service_launch::launch_transaction::live_managed_services;
    #[cfg(not(windows))]
    use crate::service_launch::test_support::sleeper_test_name;
    use crate::service_launch::test_support::write_live_service_record;
    use crate::tests::test_paths;
    use rocm_core::DEFAULT_LOCAL_HOST;

    #[test]
    fn a_child_that_exited_during_the_startup_window_is_reported_with_its_log() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let log = dir.path().join("engine.log");
        fs::write(&log, "boom: engine could not start\n")?;
        let settled = std::cell::Cell::new(false);
        let error = ensure_spawned_child_survives_startup(
            4242,
            &log,
            || settled.set(true),
            |pid| {
                assert_eq!(pid, 4242);
                assert!(settled.get(), "liveness must be probed after the wait");
                false
            },
        )
        .expect_err("a dead child must fail the launch");
        assert!(settled.get(), "the startup window must elapse first");
        assert!(format!("{error:#}").contains("boom"), "{error:#}");
        ensure_spawned_child_survives_startup(4242, &log, || {}, |_| true)?;
        Ok(())
    }

    #[test]
    fn spawn_managed_engine_child_refuses_an_explicit_port_a_live_equivalent_does_not_serve()
    -> Result<()> {
        let (root, paths) = test_paths("dup-managed-explicit-port-differs");
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
        existing.write()?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
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
        let launch = |port_request| {
            spawn_managed_engine_child(
                &paths,
                "lemonade",
                "lemonade-qwen-2000",
                "qwen",
                &resolve,
                "127.0.0.1",
                port_request,
                &resolve.device_policy,
                &[],
                None,
                None,
                None,
                false,
                false,
            )
        };

        let different = launch(serve_port::PortRequest::Explicit(11511));
        // The same port, and no port at all, still reuse the live service.
        let same = launch(serve_port::PortRequest::Explicit(11510));
        let auto = launch(serve_port::PortRequest::Auto);
        let _ = fs::remove_dir_all(root);

        let Err(error) = different else {
            panic!("a port the live service does not serve must be refused")
        };
        let message = error.to_string();
        assert!(
            message.contains("on port 11510") && message.contains("port 11511 would not be used"),
            "{message}"
        );
        assert!(matches!(same?, ManagedSpawn::AlreadyRunning(_)));
        assert!(matches!(auto?, ManagedSpawn::AlreadyRunning(_)));
        Ok(())
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
            served_model_id: None,
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
            served_model_id: None,
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
            false,
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
        // invalid — so `requires_api_key` is true while `key_present` is false,
        // which is the only way to reach the first branch. A test calling
        // `rocm_core::ensure_public_service_has_endpoint_key` directly cannot catch a
        // mis-wired call site, which is the defect that has occurred here twice.
        //
        // The flag is passed explicitly rather than inferred from the key file.
        // Inferring it marked every public bind as having demanded auth, because
        // a public bind always has a key file whether or not it asked for one.
        let (root, paths) = test_paths("managed-spawn-invalid-key");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // Non-empty but invalid (embedded CR/LF), so the file exists and is not
        // merely empty (requires_api_key = true) but yields no usable key
        // (key_present = false) — `endpoint_api_key_from_file` rejects embedded
        // control characters. An empty seed here would let the post-condition
        // below pass even if the real cleanup call never ran, since
        // `endpoint_api_key` already reads back `None` for an empty file whether
        // or not it was ever deleted.
        endpoint_keys::store_endpoint_api_key(
            &paths,
            "lemonade-qwen-3000",
            "bad-key\r\nX-Injected: value",
        )?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
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
            false,
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
        let key_file_path = endpoint_keys::endpoint_key_file_path(&paths, "lemonade-qwen-3000");
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
        // The spawn's refusal is the shared guard's, worded for a launch — the
        // same text `rocm services restart` and the daemon print, bar the
        // action. A private copy here, or the wrong action passed, would differ.
        assert_eq!(
            message,
            rocm_core::ensure_public_service_has_endpoint_key(
                "127.0.0.1",
                false,
                true,
                rocm_core::ServiceStart::Launch,
            )
            .unwrap_err()
            .to_string()
        );
        assert!(
            message.contains("launching it would open it without authentication"),
            "{message}"
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
        assert!(
            !key_file_path.exists(),
            "a pre-spawn failure must delete the unusable key file from disk, not merely \
             leave it unreadable as a key"
        );
        assert!(
            live.is_empty(),
            "a pre-spawn failure must leave no reservation"
        );
        Ok(())
    }

    #[test]
    fn a_failed_spawn_past_the_first_write_terminates_the_child_and_keeps_an_abandoned_record()
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
        // `stopped` (a `failed` one would be respawned by rocmd recovery), and
        // clear only the endpoint key.
        //
        // The child always dies near-instantly, but `spawn_managed_engine_child`
        // only reports that within a fixed startup-check window. On a loaded or
        // antivirus-scanning Windows CI box, a fresh debug `rocm.exe` can
        // occasionally be slow enough to start that the window is missed even
        // though the child is already doomed — so poll by retrying a fresh
        // attempt (its own service id/port, so no stale manifest collides) up
        // to a deadline instead of asserting on one race-prone attempt.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut attempt: u16 = 0;
        let (root, paths, service_id, result) = loop {
            attempt += 1;
            let (root, paths) =
                test_paths(&format!("managed-spawn-unknown-engine-cleanup-{attempt}"));
            paths.ensure()?;
            let service_id = format!("no-such-engine-qwen-{}", 4000 + attempt);
            endpoint_keys::store_endpoint_api_key(&paths, &service_id, "test-key")?;

            let resolve = ResolveModelResponse {
                canonical_model_id: "qwen-canonical".to_owned(),
                served_model_id: None,
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
                &service_id,
                "qwen",
                &resolve,
                "127.0.0.1",
                serve_port::PortRequest::Explicit(11_514 + attempt),
                &resolve.device_policy,
                &[],
                None,
                None,
                None,
                false,
                false,
            );

            if result.is_ok() && std::time::Instant::now() < deadline {
                let _ = fs::remove_dir_all(&root);
                continue;
            }
            break (root, paths, service_id, result);
        };

        let manifest_path = paths.service_manifest_path(&service_id);
        let log_path = paths.service_log_path(&service_id);
        let persisted = load_managed_services(&paths)
            .expect("re-read manifests")
            .into_iter()
            .find(|candidate| candidate.service_id == service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, &service_id);
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
        // `stopped`, not `failed`: rocmd's recovery restarts a `failed` record
        // without a stop marker, which would respawn a launch the CLI just
        // reported as failed (pinned on the rocmd side by
        // `abandoned_launch_record_is_not_a_recovery_candidate`).
        assert_eq!(
            persisted.status, "stopped",
            "a post-spawn failure must mark the record abandoned, not leave it stuck"
        );
        assert_eq!(
            key, None,
            "a post-spawn failure must still clear the endpoint key"
        );
        assert!(
            live.is_empty(),
            "an abandoned record must not still reserve its port"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn select_gpu_indices_under_launch_lock_blocks_until_the_holder_releases() -> Result<()> {
        // Exercises the real lock-acquisition function production callers use
        // (`select_gpu_indices_under_launch_lock`) against a lock another thread
        // actually holds, so a refactor that dropped the lock early would fail
        // here. `spawn_managed_engine_child` is then run for real, but only via
        // its `AlreadyLive` short-circuit (a pre-seeded duplicate record), so no
        // child process is ever launched. The holder writes nothing: this does
        // not show a write made under the lock being observed.
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
        let (gpu_indices, launch_lock) = crate::gpu_policy::select_gpu_indices_under_launch_lock(
            &paths,
            true,
            None,
            || None,
            None,
            None,
        )?;
        let waited = started.elapsed();

        let resolve = ResolveModelResponse {
            canonical_model_id: "held-canonical".to_owned(),
            served_model_id: None,
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
    #[cfg(not(windows))]
    fn launch_releases_the_leased_port_before_spawning_the_child() -> Result<()> {
        // `lease.release()` must run before the child is spawned, not after,
        // or the engine is handed a port the CLI is still holding open.
        // Proven via the injected `spawn_override` seam: it asserts a fresh
        // bind on the leased port succeeds at the exact instant production
        // would spawn, which only holds if the lease was already released.
        let (root, paths) = test_paths("launch-releases-port-before-spawn");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // Ephemeral, not a fixed number in the automatic range, which a real
        // `rocm serve` on the developer's machine could be holding.
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))?
            .local_addr()?
            .port();
        let lease = rocm_core::lease_loopback_port(port)?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-release-before-spawn",
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
        );
        record.write()?;

        let resolve = ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
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

        let bind_succeeded = std::cell::Cell::new(false);
        let spawn_override: &dyn Fn(&mut ProcessCommand) -> io::Result<std::process::Child> =
            &|_command| {
                // Retry briefly: a sibling test's concurrent fork can hold a copy
                // of the just-released socket until its exec closes it.
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while !bind_succeeded.get() && std::time::Instant::now() < deadline {
                    bind_succeeded.set(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
                }
                // A long-lived no-op child stands in for the engine: `_command`
                // carries real serve args this test binary cannot execute as
                // itself (see `managed_launch_cleanup_sleeper_child`).
                ProcessCommand::new(std::env::current_exe().expect("test binary"))
                    .arg(sleeper_test_name())
                    .arg("--exact")
                    .arg("--ignored")
                    .env("ROCM_TEST_SLEEPER_CHILD", "1")
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
            };

        let result = launch_managed_engine_child_locked(
            &paths,
            &mut record,
            lease,
            "lemonade",
            "lemonade-release-before-spawn",
            &resolve,
            "127.0.0.1",
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            Some(spawn_override),
        );
        if let Ok(child_pid) = result.as_ref() {
            let _ = rocm_core::terminate_process(*child_pid);
        }
        let _ = fs::remove_dir_all(root);

        assert!(
            bind_succeeded.get(),
            "the port must already be bindable (released) by the time the child is spawned"
        );
        result?;
        Ok(())
    }
}
