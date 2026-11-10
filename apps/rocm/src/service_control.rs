// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Stopping and restarting a managed local server: the process control behind
//! `rocm services stop|restart` and the internal sandbox tool.
//!
//! Mechanically relocated from `services_text.rs` with no behavior change.
//!
//! This module owns: the sandbox-tool runner (`run_internal_sandbox_tool`), the
//! stop path (`stop_recorded_processes`, which hands the recorded PIDs to the
//! routine `rocm-core` shares with `rocmd`, the `scripted_stop_outcome`
//! end-to-end hook, `stop_internal_managed_service`,
//! `stop_internal_managed_service_with`,
//! `stop_internal_managed_service_confirmed_with`,
//! `unload_lemonade_service_model`), and the restart path
//! (`apply_restart_runtime_pin`, `load_service_for_restart`, `restart_child_argv`,
//! `restart_internal_managed_service`, `restart_internal_managed_service_with`,
//! `update_restarting_record`, `record_restarted_child`,
//! `record_restart_readiness`).
//!
//! Only `run_internal_sandbox_tool` and `restart_internal_managed_service` are
//! `pub(crate)`. `run_internal_sandbox_tool` is read from `main.rs`'s
//! `dispatch()`, `services_cmd.rs`'s `run_approved_service_action` and
//! `serve_cmd.rs`, which hands it to `service_launch.rs`'s
//! `run_attached_service` as the stop for a Ctrl-C;
//! `restart_internal_managed_service` from
//! `runtime_services.rs`. Every other item has no caller outside this module and
//! stays private.
//!
//! The stop path reads the record through `services_text.rs`
//! (`load_managed_service`, `read_managed_service_manifest`) and shapes the
//! `stop_server` answer with its `internal_stop_server_value`, which stays there
//! next to `service_stop_unconfirmed` and the renderer that reads the same
//! `stop_unconfirmed` status. This module takes the
//! spawn helpers from `service_launch.rs`, which does not import this module
//! back; its keyless-respawn check is `rocm_core`'s shared guard.
//!
//! The regression tests for the stop and restart paths live in this module's own
//! `mod tests`. They reach shared fixtures (`test_paths`, `ScopedTestEnv`,
//! `plant_service_record_on_runtime`, `OLD_RUNTIME_KEY`, `NEW_RUNTIME_KEY`)
//! through `crate::tests::`, and `runtime_services::` through its own
//! `use crate::runtime_services;` for the tests that drive a restart through
//! `runtime_services`'s public entry points.

use std::fs;
use std::io;
#[cfg(not(windows))]
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, EndpointReadiness, ManagedServiceRecord};
use rocm_engine_protocol::{EngineMethod, StopRequest, StopResponse};

use crate::endpoint_keys;
use crate::engine_ipc::{builtin_engine_serve_http_args, engine_request};
#[cfg(windows)]
use crate::engine_runtime_select::{app_path_env_var_refs, app_path_env_var_values};
use crate::engine_runtime_select::{
    http_post_local_service_json, managed_service_launcher_path, parse_engine_recipe_json_arg,
    wait_for_service_http_ready,
};
use crate::engines_cmd::env_root_for_service;
use crate::gpu_policy::parse_device_policy;
use crate::managed_service_status::load_managed_services;
use crate::service_hints;
#[cfg(not(windows))]
use crate::service_launch::{
    attach_background_stdio, detach_background_command, managed_engine_startup_failure_detail,
    managed_service_process_command,
};
use crate::services_text::{
    internal_stop_server_value, load_managed_service, read_managed_service_manifest,
};
use crate::{SandboxToolArg, status_for_readiness};

pub(crate) fn run_internal_sandbox_tool(
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
            internal_stop_server_value(stop_internal_managed_service(paths, &service_id)?)
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

/// Terminate the processes recorded for a managed service through the routine
/// `rocmd` shares, [`rocm_core::terminate_recorded_service_pids`], with
/// [`rocm_core::terminate_recorded_pid`] as the termination of each PID. The
/// one thing this adds is the e2e seam below.
fn stop_recorded_processes(record: &ManagedServiceRecord) -> Vec<rocm_core::RecordedPidStop> {
    rocm_core::terminate_recorded_service_pids(record, |identity| {
        scripted_stop_outcome().unwrap_or_else(|| rocm_core::terminate_recorded_pid(identity))
    })
}

/// E2E hook: when `ROCM_E2E_STOP_OUTCOME` is `timed_out` or `unverified`, a
/// managed stop reports that outcome for every recorded PID without signalling
/// anything.
///
/// Neither outcome can be staged from outside the binary: `TimedOut` needs a
/// process that survives `SIGKILL`, and `Unverified` one whose recorded
/// start-time cannot be read back, or (on Windows) that cannot be queried at
/// all. Without this seam the "stop not confirmed"
/// report, its exit status and what it keeps on disk have no live coverage.
/// The recorded process is left untouched. Compiled only into the
/// `e2e-test-hooks` binary; production builds never see it.
#[cfg(feature = "e2e-test-hooks")]
fn scripted_stop_outcome() -> Option<rocm_core::TerminationOutcome> {
    match std::env::var("ROCM_E2E_STOP_OUTCOME").ok()?.trim() {
        "timed_out" => Some(rocm_core::TerminationOutcome::TimedOut),
        "unverified" => Some(rocm_core::TerminationOutcome::Unverified),
        _ => None,
    }
}

#[cfg(not(feature = "e2e-test-hooks"))]
const fn scripted_stop_outcome() -> Option<rocm_core::TerminationOutcome> {
    None
}

fn stop_internal_managed_service(paths: &AppPaths, service_id: &str) -> Result<serde_json::Value> {
    stop_internal_managed_service_with(paths, service_id, stop_recorded_processes)
}

/// [`stop_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller. Production passes
/// [`stop_recorded_processes`]; a test passes a stand-in that observes
/// what is on disk at the moment the processes would be signalled. Everything
/// else — the engine stop, the manifest writes, the key cleanup — runs for real.
///
/// `pub(crate)` so `service_launch`'s tests can drive the real stop path too,
/// rather than writing a stopped status to disk themselves.
pub(crate) fn stop_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl FnOnce(&ManagedServiceRecord) -> Vec<rocm_core::RecordedPidStop>,
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
    terminate: impl FnOnce(&ManagedServiceRecord) -> Vec<rocm_core::RecordedPidStop>,
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
    let stops = terminate(&stopping);
    // Every PID that received a signal, including one that then outlived the
    // deadline: signalling was attempted even though the process was not
    // confirmed stopped, and `terminated_all` separately carries the truthful
    // completion state.
    let signaled_pids: Vec<u32> = stops
        .iter()
        .filter(|stop| stop.signaled())
        .map(|stop| stop.pid)
        .collect();
    let terminated_all = stops.iter().all(|stop| stop.stopped());
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
    // A container engine is stopped by its engine `Stop`, not by the PID of the
    // supervisor that follows it, so a failed `Stop` leaves the stop unconfirmed.
    let engine_stop_confirms =
        !rocm_core::engine_stop_decides_confirmation(&stopping.engine) || engine_stop.is_ok();
    let record_changed = !record.is_covered_by_stop_of(&stopping);
    let all_stopped = terminated_all && !record_changed && engine_stop_confirms;
    if all_stopped {
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = None;
        // The processes this stop confirmed gone are not the service's any more,
        // and their PIDs are stale the instant they exited. Left in the record,
        // a later stop would signal them again: a PID recorded without a
        // start-time token (every one a Windows record had before it recorded
        // a creation time) cannot be refuted, so if the
        // OS had given it to another process the Windows tree kill would take
        // that process and its subtree. The same call, on the same confirmed
        // path, as `rocmd`'s stop.
        record.clear_recorded_processes();
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
    // `endpoint_key_kept`: whether this unconfirmed stop left a *usable* key —
    // one the server may still be enforcing — on disk. Usable as
    // `endpoint_api_key` decides it, the same test the managed spawn applies
    // before handing a key to the engine: an empty or malformed key file is no
    // key, because nothing enforces it. A loopback service started without
    // `--require-api-key` never had one. The unconfirmed report names the key
    // only when this is set.
    let endpoint_key_kept =
        !all_stopped && endpoint_keys::endpoint_api_key(paths, &record.service_id).is_some();
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
        "pid_outcomes": stops
            .iter()
            .map(|stop| stop.report_entry())
            .collect::<Vec<_>>(),
        "record_changed": record_changed,
        // A container engine is confirmed stopped by its own `Stop`; when that
        // failed, the verdict is unconfirmed whatever the PIDs did.
        "engine_stop_unconfirmed": !engine_stop_confirms,
        "endpoint_key_kept": endpoint_key_kept,
        "stopped": all_stopped,
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
pub(crate) fn restart_internal_managed_service(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
) -> Result<ManagedServiceRecord> {
    restart_internal_managed_service_with(paths, service_id, runtime_pin, stop_recorded_processes)
}

/// [`restart_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller — see [`stop_internal_managed_service_with`].
///
/// The respawn is serialized on the same cross-process launch lock a fresh
/// `rocm serve` launch and `rocmd` recovery both use: the confirmed stop just
/// released this service's (host, port), and nothing reserves it again until
/// the new child binds and its record is published, so a concurrent launch or
/// recovery could otherwise claim the exact port this restart is about to
/// reuse.
fn restart_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
    terminate: impl FnOnce(&ManagedServiceRecord) -> Vec<rocm_core::RecordedPidStop>,
) -> Result<ManagedServiceRecord> {
    restart_internal_managed_service_inner(paths, service_id, runtime_pin, terminate, None)
}

/// How long a restart waits for the new child to serve.
#[allow(clippy::missing_const_for_fn)] // not const under `cfg(test)`, where a test shortens it
fn restart_readiness_timeout() -> Duration {
    #[cfg(test)]
    if let Some(timeout) = tests::READINESS_TIMEOUT_OVERRIDE.with(std::cell::Cell::get) {
        return timeout;
    }
    Duration::from_secs(45)
}

/// How long a restart waits for the launch lock once its stop has completed.
#[allow(clippy::missing_const_for_fn)] // not const under `cfg(test)`, where a test shortens it
fn restart_launch_lock_timeout() -> Duration {
    #[cfg(test)]
    if let Some(timeout) = tests::LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(std::cell::Cell::get) {
        return timeout;
    }
    rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT
}

/// Replaces the child `spawn()` of a restart (non-Windows only; Windows spawns
/// through `rocm_core::spawn_detached_no_inherit`). Test-only seam, so a test
/// can observe the leased port at the exact instant production would spawn.
type RestartSpawnOverride<'a> =
    dyn Fn(&mut std::process::Command) -> io::Result<std::process::Child> + 'a;

fn restart_internal_managed_service_inner(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
    terminate: impl FnOnce(&ManagedServiceRecord) -> Vec<rocm_core::RecordedPidStop>,
    spawn_override: Option<&RestartSpawnOverride<'_>>,
) -> Result<ManagedServiceRecord> {
    #[cfg(windows)]
    let _ = spawn_override;
    let record = load_service_for_restart(paths, service_id, runtime_pin)?;
    // Preserve the endpoint key across the restart: `stop_internal_managed_service`
    // deletes the key file (correct on a real stop), but a restart must bring the
    // service back on the same public host with the same auth. Capture it first and
    // re-store it after the stop so the spawn below hands the child the same key.
    let preserved_endpoint_key = endpoint_keys::endpoint_api_key(paths, service_id);
    // Checked before the stop, so a refused restart leaves a running service
    // running instead of stopping it and then failing to bring it back.
    rocm_core::ensure_public_service_has_endpoint_key(
        &record.host,
        preserved_endpoint_key.is_some(),
        record.requires_api_key,
        rocm_core::ServiceStart::Restart,
        service_id,
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
    // Publish that a relaunch has begun before anything is restored or spawned.
    // From here until `record_restarted_child` the record is an empty `stopped`
    // manifest, which an older stop still finalizing would take for its own
    // completion and delete the restored key under; the generation stamp is what
    // keeps `is_covered_by_stop_of` from covering it. Applied to the record as it
    // is on disk now, guarded on it still being what the confirmed stop left.
    let (confirmed_record, _) = read_managed_service_manifest(paths, service_id)?;
    if update_restarting_record(
        paths,
        &confirmed_record,
        ManagedServiceRecord::begin_relaunch,
    )?
    .is_none()
    {
        bail!(
            "`{service_id}` was stopped but NOT restarted: it changed on disk while the \
             restart was beginning; run `{}` to try again",
            service_hints::service_action_hint("restart", service_id)
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
    // restart that gets this far and fails. Pure computation — it parses the
    // record, it does not touch the network — so it stays ahead of the launch
    // lock below and a refused device policy still fails before anything binds.
    let serve_args = restart_child_argv(&record)?;
    // The confirmed stop just released this (host, port): nothing holds it again
    // until the new child binds and `record_restarted_child` publishes below.
    // Close that gap the same way a fresh launch and `rocmd` recovery do —
    // under the shared launch lock, with a re-check against the disk and a
    // loopback preflight — so a concurrent launch or recovery can't claim this
    // exact port out from under the restart. Bounded, unlike `rocm serve`'s
    // acquire of the same lock: the stop above has already happened, so a wedged
    // holder would otherwise leave the service down with nothing on screen. On
    // timeout the error says the service is stopped and names the command that
    // brings it back.
    let launch_lock = rocm_core::FileLock::acquire_bounded(
        paths.managed_launch_lock_path(),
        restart_launch_lock_timeout(),
    )
    .with_context(|| {
        format!(
            "`{service_id}` was stopped but NOT restarted: the launch lock stayed busy; once the \
             launch holding it finishes, run `{}`",
            service_hints::service_action_hint("restart", service_id)
        )
    })?;
    if let Some(collision) = load_managed_services(paths)
        .context("failed to re-read managed services under the launch lock")?
        .into_iter()
        .find(|other| {
            other.service_id != stopped_record.service_id && other.blocks_port(stopped_record.port)
        })
    {
        bail!(
            "cannot restart `{service_id}` on {}:{}: managed service {} is recorded on that \
             port (it may have claimed that port while the restart waited for the launch lock)",
            stopped_record.host,
            stopped_record.port,
            collision.service_id
        );
    }
    let port_lease = if rocm_core::is_canonical_loopback_host(&stopped_record.host) {
        Some(rocm_core::lease_loopback_port_or_bind_error(
            stopped_record.port,
            format_args!(
                "cannot restart `{service_id}` on {}:{}: its port is in use",
                stopped_record.host, stopped_record.port
            ),
        )?)
    } else {
        None
    };
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
    // Hand the port to the respawned child: close the probe socket immediately
    // before the spawn, still holding the launch lock.
    if let Some(lease) = port_lease {
        lease.release();
    }
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
        paths.apply_env(&mut command);
        if let Some(engine_envs_root) = engine_envs_root.as_deref() {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        if let Some(key_file) = endpoint_key_file.as_deref() {
            command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        }
        let mut child = match spawn_override {
            Some(spawn) => spawn(&mut command),
            None => command.spawn(),
        }
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
                managed_engine_startup_failure_detail(
                    &format!(" with status {status}"),
                    &record.log_path
                )
            );
        }
        child.id()
    };
    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));
    // Recorded before the readiness wait, so a stop issued during it finds the
    // new child rather than the processes already stopped.
    let started = record_restarted_child(paths, &stopped_record, runtime_pin, child_pid)?;
    // Published; a concurrent launch or recovery now sees this service as live
    // on its port again, so there is nothing left for the lock to protect, and
    // the readiness wait below can block for many seconds.
    drop(launch_lock);
    let readiness = wait_for_service_http_ready(
        &started.engine,
        &started.host,
        started.port,
        started.served_model_name(),
        endpoint_api_key.as_deref(),
        restart_readiness_timeout(),
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
/// as `expected` left it — it is gone, names other processes, has another
/// status, or is in another relaunch generation: whoever wrote since holds a truer view than this restart. `update`
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
    if !current.names_same_processes(expected)
        || current.status != expected.status
        || current.relaunch_started_unix_ms != expected.relaunch_started_unix_ms
    {
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::runtime_services;
    use crate::services_cmd::finish_service_action;
    use crate::services_text::tests::assert_no_retry_line;
    use crate::services_text::{process_check_command, render_service_action_result};
    use crate::tests::{
        NEW_RUNTIME_KEY, OLD_RUNTIME_KEY, ScopedTestEnv, plant_service_record_on_runtime,
        test_paths,
    };

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
        // The restart's refusal is the shared guard's, worded for a restart —
        // the same text a fresh launch and the daemon print, bar the action. A
        // private copy here, or the wrong action passed, would differ.
        assert_eq!(
            rendered,
            rocm_core::ensure_public_service_has_endpoint_key(
                "0.0.0.0",
                false,
                false,
                rocm_core::ServiceStart::Restart,
                "vllm-public-2000",
            )
            .unwrap_err()
            .to_string()
        );
        assert!(
            rendered.contains("restarting it would reopen it without authentication"),
            "{rendered}"
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
        // the test's own (`engine_pid` is None and the shared stop routine
        // skips the caller's pid), so the stop confirms termination trivially.
        let after = load_managed_service(&paths, service_id).unwrap();
        assert_ne!(
            after.status, "stopped",
            "a refused restart must not stop the service"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// What `rocm services restart` does about the endpoint key, at the real
    /// call site, one argument of the guard at a time. The guard is `rocm_core`'s
    /// and is tested there; what only this call site can get wrong is what it
    /// passes: the host, the recorded `requires_api_key`, whether a usable key
    /// file is on disk, and that it is a restart.
    fn restart_outcome_for(
        test: &str,
        host: &str,
        requires_api_key: bool,
        key: Option<&str>,
    ) -> String {
        let (root, paths) = test_paths(test);
        paths.ensure().unwrap();
        let service_id = "svc-restart-key-site";
        let mut record = stop_test_record(&paths, service_id);
        record.host = host.to_owned();
        record.requires_api_key = requires_api_key;
        record.write().unwrap();
        if let Some(key) = key {
            endpoint_keys::store_endpoint_api_key(&paths, service_id, key).unwrap();
        }
        // A restart the guard lets through stops the service and then waits for
        // the launch lock, which is held, so it ends there and starts nothing: the
        // error is what tells a refusal from a restart that went ahead.
        let holder = rocm_core::FileLock::acquire(paths.managed_launch_lock_path()).unwrap();
        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(Some(Duration::from_millis(50))));
        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            bare_termination(Vec::new(), true)
        });
        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(None));
        drop(holder);
        let _ = fs::remove_dir_all(root);
        format!(
            "{:#}",
            result.expect_err("the restart never starts a child here")
        )
    }

    #[test]
    fn restart_passes_the_recorded_key_requirement_to_the_guard() {
        // A loopback service that asked for `--require-api-key` and has lost its
        // key is refused. Only `requires_api_key` makes it so, and every other
        // test plants it false, so this is what pins the argument.
        let error = restart_outcome_for("restart-site-requires", "127.0.0.1", true, None);
        assert_eq!(
            error,
            rocm_core::ensure_public_service_has_endpoint_key(
                "127.0.0.1",
                false,
                true,
                rocm_core::ServiceStart::Restart,
                "svc-restart-key-site",
            )
            .unwrap_err()
            .to_string()
        );
        // The same service without the requirement is not refused: it is let
        // through to the stop and the lock wait.
        let error = restart_outcome_for("restart-site-not-required", "127.0.0.1", false, None);
        assert!(error.contains("stopped but NOT restarted"), "{error}");
    }

    #[test]
    fn restart_passes_whether_a_key_is_on_disk_to_the_guard() {
        // A public service with its key still stored is not refused; without it
        // the same service is. Together they pin the key-present argument.
        let error =
            restart_outcome_for("restart-site-key-present", "0.0.0.0", false, Some("a-key"));
        assert!(error.contains("stopped but NOT restarted"), "{error}");
        assert!(!error.contains("without authentication"), "{error}");
        let error = restart_outcome_for("restart-site-key-absent", "0.0.0.0", false, None);
        assert!(error.contains("without authentication"), "{error}");
    }

    #[test]
    fn restart_passes_the_recorded_host_to_the_guard() {
        // Whether a keyless service is refused for its bind depends on the
        // recorded host: a public one is, loopback is not.
        let error = restart_outcome_for("restart-site-host-public", "0.0.0.0", false, None);
        assert!(
            error.contains("bound to the public host `0.0.0.0`"),
            "{error}"
        );
        let error = restart_outcome_for("restart-site-host-loopback", "127.0.0.1", false, None);
        assert!(error.contains("stopped but NOT restarted"), "{error}");
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
            bare_termination(Vec::new(), false)
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
            bare_termination(Vec::new(), true)
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

    /// A liveness refresh that lands after the stop's processes have exited but
    /// before the stop reads the record again completes the same stop (key,
    /// marker, recorded processes cleared, `stopped`) and writes it. That record
    /// names no process, which must not read as a restart: the stop is confirmed,
    /// the marker is not put back, and a restart is not refused for it. A record
    /// that read `failed` is completed to `stopped` too, as a confirmed stop makes
    /// it, so it is covered as well and not left reading `failed` with no process.
    #[test]
    fn a_listing_that_completes_the_stop_before_finalization_does_not_unconfirm_it() {
        use rocm_core::TerminationOutcome::Graceful;
        for status in ["ready", "failed"] {
            let (root, paths) = test_paths(&format!("stop-listing-before-finalization-{status}"));
            paths.ensure().unwrap();
            let service_id = "svc-stop-listing-before-finalization";
            let mut planted = stop_test_record(&paths, service_id);
            planted.status = status.to_owned();
            planted.supervisor_pid = 4242;
            planted.supervisor_start_ticks = None;
            planted.engine_pid = Some(4243);
            planted.engine_start_ticks = Some(77);
            planted.write().unwrap();
            endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

            let report = stop_internal_managed_service_with(&paths, service_id, |stopping| {
                assert!(stopping.stop_requested_unix_ms.is_some());
                // The processes have exited; a concurrent `services` listing sees
                // it, completes the pending stop and persists that, before this
                // stop reads the record again.
                let (mut listed, _) = read_managed_service_manifest(&paths, service_id).unwrap();
                // The fake PIDs read as gone to the stop's own initial refresh,
                // which may already have demoted the status: put back the status
                // the listing under test starts from.
                status.clone_into(&mut listed.status);
                let changed =
                    crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
                        &paths,
                        &mut listed,
                        |_| rocm_core::ProcessProbe::Gone,
                    );
                assert!(
                    changed,
                    "premise: the listing completed the stop ({status})"
                );
                listed.write().unwrap();
                assert_eq!(raw_manifest(&paths, service_id).supervisor_pid, 0);
                vec![stop_of(4242, Graceful), stop_of(4243, Graceful)]
            })
            .unwrap();

            let on_disk = raw_manifest(&paths, service_id);
            let key = endpoint_keys::endpoint_api_key(&paths, service_id);
            let _ = fs::remove_dir_all(root);
            assert_eq!(report["stopped"], true, "{status}: {report}");
            assert_eq!(on_disk.status, "stopped", "{status}");
            assert_eq!(
                on_disk.stop_requested_unix_ms, None,
                "{status}: a confirmed stop does not put the marker back"
            );
            assert_eq!(on_disk.supervisor_pid, 0, "{status}");
            assert_eq!(key, None, "{status}: the key goes with the confirmed stop");
        }
    }

    /// The same race through `restart`: the restart's own stop is completed by a
    /// listing before it finalizes, and the restart must carry on past the stop
    /// (here to the launch lock, which a holder keeps, so it stops there)
    /// instead of refusing with "could not confirm".
    #[test]
    fn a_restart_is_not_refused_because_a_listing_completed_its_stop() -> Result<()> {
        use rocm_core::TerminationOutcome::Graceful;
        let (root, paths) = test_paths("restart-listing-before-finalization");
        paths.ensure()?;
        let service_id = "svc-restart-listing-before-finalization";
        stop_test_record(&paths, service_id).write()?;
        let holder = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(Some(Duration::from_millis(100))));
        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            let (mut listed, _) = read_managed_service_manifest(&paths, service_id).unwrap();
            let _ = crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
                &paths,
                &mut listed,
                |_| rocm_core::ProcessProbe::Gone,
            );
            listed.write().unwrap();
            vec![stop_of(999_999_999, Graceful)]
        });
        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(None));
        drop(holder);
        let _ = fs::remove_dir_all(root);

        let message = format!(
            "{:#}",
            result.expect_err("the held launch lock fails the restart")
        );
        assert!(
            !message.contains("could not confirm"),
            "the stop was completed, the restart must not be refused for it: {message}"
        );
        assert!(message.contains("stopped but NOT restarted"), "{message}");
        Ok(())
    }

    /// The other half of the same check: a restart that has begun launching but
    /// has recorded no process yet (status `starting`, nothing named) is not the
    /// stop completing, and the stop must not claim it or drop the key it will
    /// serve with.
    #[test]
    fn stop_does_not_confirm_over_a_restart_that_has_recorded_no_process_yet() {
        let (root, paths) = test_paths("stop-restart-in-flight");
        paths.ensure().unwrap();
        let service_id = "svc-stop-restart-in-flight";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut restarting = raw_manifest(&paths, service_id);
            restarting.status = "starting".to_owned();
            restarting.clear_recorded_processes();
            restarting.stop_requested_unix_ms = None;
            restarting.write().unwrap();
            bare_termination(Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["status"], "starting", "{report}");
        assert_eq!(on_disk.status, "starting", "the restart's status stays");
        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
    }

    /// The same as above when the record being stopped names no process either:
    /// a relaunch's empty `starting` record then equals the stopping snapshot's
    /// processes, which must not make it the stop's own. Its marker is cleared and
    /// it reads `starting`, so the stop does not claim it, overwrite its status or
    /// delete the key it will serve with. A stop of an empty record that nobody
    /// relaunched is still confirmed.
    #[test]
    fn stop_does_not_confirm_over_an_empty_relaunch_of_an_empty_record() {
        let (root, paths) = test_paths("stop-empty-relaunch");
        paths.ensure().unwrap();
        let service_id = "svc-stop-empty-relaunch";
        let mut empty = stop_test_record(&paths, service_id);
        empty.clear_recorded_processes();
        empty.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut relaunch = raw_manifest(&paths, service_id);
            relaunch.status = "starting".to_owned();
            relaunch.stop_requested_unix_ms = None;
            relaunch.write().unwrap();
            bare_termination(Vec::new(), true)
        })
        .unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);

        // And with nobody relaunching, the stop of the same empty record is a stop.
        let untouched = stop_internal_managed_service_with(&paths, service_id, |_| {
            bare_termination(Vec::new(), true)
        })
        .unwrap();
        let _ = fs::remove_dir_all(root);

        assert_eq!(report["stopped"], false, "{report}");
        assert_eq!(on_disk.status, "starting", "the relaunch's status stays");
        assert_eq!(key.as_deref(), Some("key-123"), "its key is kept");
        assert_eq!(untouched["stopped"], true, "{untouched}");
    }

    /// The interleaving the empty-relaunch guard above must not mistake for a
    /// relaunch: the record being stopped names no process and reads `starting` or
    /// `failed` (a liveness refresh leaves both as they are), and a concurrent
    /// listing finds the stop marker with nothing recorded to wait for and clears
    /// it before the stop reads the record again. Nobody relaunched, so the stop is
    /// confirmed as it is when no listing intervenes (the last block), and the
    /// record reads `stopped` afterwards with its key gone.
    #[test]
    fn a_listing_clearing_the_marker_of_an_empty_stop_does_not_make_it_unconfirmed() {
        for status in ["starting", "failed"] {
            let (root, paths) = test_paths(&format!("stop-empty-listing-{status}"));
            paths.ensure().unwrap();
            let service_id = "svc-stop-empty-listing";
            let mut empty = stop_test_record(&paths, service_id);
            status.clone_into(&mut empty.status);
            empty.clear_recorded_processes();
            empty.write().unwrap();
            endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

            let report = stop_internal_managed_service_with(&paths, service_id, |stopping| {
                assert!(stopping.stop_requested_unix_ms.is_some());
                // The listing under test: it reads the record with the stop's
                // marker on it, finds no process to wait for, and persists the
                // cleanup before this stop reads the record again.
                let (mut listed, _) = read_managed_service_manifest(&paths, service_id).unwrap();
                let changed =
                    crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
                        &paths,
                        &mut listed,
                        |_| rocm_core::ProcessProbe::Gone,
                    );
                assert!(
                    changed,
                    "premise: the listing cleaned the marker ({status})"
                );
                listed.write().unwrap();
                let after_listing = raw_manifest(&paths, service_id);
                assert_eq!(after_listing.stop_requested_unix_ms, None, "{status}");
                assert_eq!(after_listing.status, status, "premise: status unchanged");
                bare_termination(Vec::new(), true)
            })
            .unwrap();

            let on_disk = raw_manifest(&paths, service_id);
            let key = endpoint_keys::endpoint_api_key(&paths, service_id);
            let _ = fs::remove_dir_all(root);
            assert_eq!(report["stopped"], true, "{status}: {report}");
            assert_eq!(on_disk.status, "stopped", "{status}");
            assert_eq!(on_disk.stop_requested_unix_ms, None, "{status}");
            assert_eq!(key, None, "{status}: the key goes with the confirmed stop");
        }
    }

    /// The same interleaving followed by a real relaunch: the listing consumed the
    /// marker, then the daemon restarted the empty record (a restart counts itself,
    /// clears the marker and reads `recovering` or `starting`). The stamp the
    /// listing left is the stop's own, but the relaunch wipes it, so the stop still
    /// does not claim the relaunch's record.
    #[test]
    fn a_relaunch_after_a_listing_consumed_the_marker_is_still_not_the_stop() {
        let (root, paths) = test_paths("stop-empty-listing-then-relaunch");
        paths.ensure().unwrap();
        let service_id = "svc-stop-empty-listing-then-relaunch";
        let mut empty = stop_test_record(&paths, service_id);
        "starting".clone_into(&mut empty.status);
        empty.clear_recorded_processes();
        empty.write().unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let (mut listed, _) = read_managed_service_manifest(&paths, service_id).unwrap();
            let _ = crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
                &paths,
                &mut listed,
                |_| rocm_core::ProcessProbe::Gone,
            );
            listed.write().unwrap();
            let mut relaunch = raw_manifest(&paths, service_id);
            assert!(
                relaunch.settled_stop_unix_ms.is_some(),
                "premise: the listing left the stop's stamp"
            );
            "recovering".clone_into(&mut relaunch.status);
            relaunch.reset_for_restart();
            relaunch.stop_requested_unix_ms = None;
            relaunch.write().unwrap();
            bare_termination(Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["stopped"], false, "{report}");
        assert_eq!(on_disk.status, "recovering", "the relaunch's status stays");
        assert_eq!(on_disk.restart_count, 1);
    }

    #[test]
    fn stop_of_a_container_engine_is_unconfirmed_when_the_engine_stop_fails() {
        // For `aim` the recorded PID is only the supervisor; killing it says
        // nothing about the container. With no aim engine to answer `Stop`, the
        // PID verdict alone must not mark the service stopped or drop its key.
        let (root, paths) = test_paths("stop-aim-engine-stop-fails");
        paths.ensure().unwrap();
        // The built-in aim engine answers `Stop` through `docker`; a dead daemon
        // address makes that fail the same way on a host with or without docker.
        let mut env = ScopedTestEnv::new();
        env.set("DOCKER_HOST", "unix:///nonexistent/rocm-test-docker.sock");
        let service_id = "svc-stop-aim-engine-stop-fails";
        let mut record = stop_test_record(&paths, service_id);
        record.engine = "aim".to_owned();
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            bare_termination(Vec::new(), true)
        })
        .unwrap();
        drop(env);

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert!(report["engine_stop"]["error"].is_string(), "{report}");
        // Unconfirmed: the marker is re-stamped and the key kept for the liveness
        // refresh to settle, rather than cleared as for a confirmed stop.
        assert!(on_disk.stop_requested_unix_ms.is_some());
        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
        // The report gives the engine's failed stop as its reason, since no
        // PID outcome and no record change explains it.
        assert_eq!(report["engine_stop_unconfirmed"], true, "{report}");
        let rendered =
            render_service_action_result("stop_server", &internal_stop_server_value(report));
        assert!(
            rendered.contains("Local server stop not confirmed"),
            "{rendered}"
        );
        assert!(
            rendered.contains("not confirmed: the engine's own stop failed: "),
            "{rendered}"
        );
        assert!(rendered.contains("and its endpoint key"), "{rendered}");
        assert_no_retry_line(&rendered);
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
            bare_termination(Vec::new(), true)
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
    fn services_stop_reports_an_unconfirmed_stop_as_an_error_and_keeps_the_service() {
        // `rocm services stop --yes` printed "Local server stopped" and exited 0
        // from the tool name alone, even when the stop could not confirm the
        // recorded process gone and so kept the record, its stop request and
        // the endpoint key. The answer now carries the stop's own verdict: the
        // message, the exit and the state on disk are checked together here.
        let (root, paths) = test_paths("services-stop-unconfirmed");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-unconfirmed";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            timed_out_termination(999_999_999)
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let audit = last_audit_event(&paths);
        let _ = fs::remove_dir_all(root);
        assert_eq!(answer["status"], "stop_unconfirmed", "{answer}");
        assert_eq!(answer["result"]["stopped"], false, "{answer}");
        let rendered = format!(
            "{:#}",
            outcome.expect_err("an unconfirmed stop must not exit as a success")
        );
        assert!(
            !rendered.contains("Local server stopped"),
            "claims a stop that was not confirmed: {rendered}"
        );
        assert!(
            rendered.contains("Local server stop not confirmed"),
            "{rendered}"
        );
        assert!(rendered.contains("signaled processes: 1"), "{rendered}");
        assert!(
            rendered.contains("still running: process 999999999 (supervisor_and_engine)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("kept: the server record, which still names its processes, and its endpoint key, which the server may still be enforcing"),
            "{rendered}"
        );
        assert_no_retry_line(&rendered);
        assert!(
            rendered.contains(
                "process 999999999 (supervisor_and_engine) or a process under it did not exit"
            ),
            "{rendered}"
        );
        assert!(!rendered.contains("check:"), "{rendered}");
        assert!(
            !rendered.contains("status:"),
            "a record status would contradict the header: {rendered}"
        );
        assert_eq!(audit["level"], "error", "{audit}");
        assert_eq!(audit["action"], "stop_server", "{audit}");
        assert_eq!(
            audit["message"],
            format!("stop of managed service {service_id} not confirmed"),
            "{audit}"
        );
        // The claims in that message, checked against the state they describe.
        assert_eq!(
            on_disk.supervisor_pid, 999_999_999,
            "the record still names the process"
        );
        assert!(
            on_disk.stop_requested_unix_ms.is_some(),
            "the stop request stays recorded"
        );
        assert_eq!(key.as_deref(), Some("key-123"), "the endpoint key is kept");
    }

    /// One recorded process's stop, for a stand-in to return.
    fn stop_of(pid: u32, outcome: rocm_core::TerminationOutcome) -> rocm_core::RecordedPidStop {
        rocm_core::RecordedPidStop {
            pid,
            role: "supervisor_and_engine",
            outcome: rocm_core::RecordedPidOutcome::Terminated(outcome),
        }
    }

    /// A stop stand-in's bare verdict, without per-PID detail: `signaled_pids`
    /// were signalled and `all_stopped` says whether they exited. A stop that
    /// signalled nothing and is not confirmed has one unverifiable process.
    fn bare_termination(
        signaled_pids: Vec<u32>,
        all_stopped: bool,
    ) -> Vec<rocm_core::RecordedPidStop> {
        use rocm_core::TerminationOutcome::{Graceful, TimedOut, Unverified};
        if signaled_pids.is_empty() {
            return if all_stopped {
                Vec::new()
            } else {
                vec![stop_of(999_999_999, Unverified)]
            };
        }
        let outcome = if all_stopped { Graceful } else { TimedOut };
        signaled_pids
            .into_iter()
            .map(|pid| stop_of(pid, outcome))
            .collect()
    }

    /// A stop stand-in whose one recorded process (the record's shared
    /// supervisor/engine PID) was signalled and outlived the deadline.
    fn timed_out_termination(pid: u32) -> Vec<rocm_core::RecordedPidStop> {
        vec![stop_of(pid, rocm_core::TerminationOutcome::TimedOut)]
    }

    /// The most recent audit event written under `paths`.
    fn last_audit_event(paths: &AppPaths) -> serde_json::Value {
        let text = fs::read_to_string(paths.audit_events_path()).expect("audit log");
        serde_json::from_str(text.lines().last().expect("an audit event")).expect("audit json")
    }

    #[test]
    fn services_stop_reports_an_unverified_process_with_a_check() {
        // A process whose recorded start-time cannot be read back is not
        // signalled by this stop; the report says so and names a command to
        // check on it.
        let (root, paths) = test_paths("services-stop-unverified");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-unverified";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![stop_of(
                999_999_999,
                rocm_core::TerminationOutcome::Unverified,
            )]
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("an unverified stop fails"));
        assert!(
            rendered.contains("Local server stop not confirmed"),
            "{rendered}"
        );
        assert!(
            rendered.contains("not signaled: process 999999999 (supervisor_and_engine)"),
            "{rendered}"
        );
        assert_no_retry_line(&rendered);
        assert!(
            rendered.contains(&format!("check: `{}`", process_check_command(999_999_999))),
            "{rendered}"
        );
        assert!(on_disk.stop_requested_unix_ms.is_some());
        assert_eq!(on_disk.supervisor_pid, 999_999_999);
        assert_eq!(key.as_deref(), Some("key-123"));
    }

    #[test]
    fn services_stop_notes_a_changed_record_even_when_a_process_timed_out() {
        // Process A timed out, then a restart recorded B: the report must say
        // the record changed, beside A's own line.
        let (root, paths) = test_paths("services-stop-timeout-then-changed");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-timeout-then-changed";
        stop_test_record(&paths, service_id).write().unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut restarted = raw_manifest(&paths, service_id);
            restarted.supervisor_pid = 4_242_424;
            restarted.supervisor_start_ticks = Some(7);
            restarted.write().unwrap();
            timed_out_termination(999_999_999)
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("unconfirmed"));
        assert!(
            rendered.contains("the record changed while the stop ran"),
            "{rendered}"
        );
        assert_no_retry_line(&rendered);
    }

    #[test]
    fn services_stop_gives_a_mixed_result_a_line_each_and_a_check() {
        let (root, paths) = test_paths("services-stop-mixed");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-mixed";
        let mut record = stop_test_record(&paths, service_id);
        record.engine_pid = Some(888_888_888);
        record.write().unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![
                rocm_core::RecordedPidStop {
                    role: "supervisor",
                    ..stop_of(999_999_999, rocm_core::TerminationOutcome::TimedOut)
                },
                rocm_core::RecordedPidStop {
                    role: "engine",
                    ..stop_of(888_888_888, rocm_core::TerminationOutcome::Unverified)
                },
            ]
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("unconfirmed"));
        assert_no_retry_line(&rendered);
        assert!(
            rendered.contains(&format!("check: `{}`", process_check_command(888_888_888))),
            "{rendered}"
        );
        assert!(!rendered.contains("the record changed"), "{rendered}");
    }

    #[test]
    fn services_stop_mentions_no_endpoint_key_for_a_keyless_service() {
        // A loopback service started without `--require-api-key` has no key, so
        // an unconfirmed stop of it keeps only the record; the report must not
        // claim it kept a key.
        let (root, paths) = test_paths("services-stop-keyless");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-keyless";
        stop_test_record(&paths, service_id).write().unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            timed_out_termination(999_999_999)
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("unconfirmed"));
        assert_eq!(answer["result"]["endpoint_key_kept"], false, "{answer}");
        assert!(
            rendered
                .lines()
                .any(|line| line == "  kept: the server record, which still names its processes"),
            "{rendered}"
        );
        assert!(!rendered.contains("endpoint key"), "{rendered}");
        assert_eq!(key, None, "there was no key to keep");
        assert_eq!(on_disk.supervisor_pid, 999_999_999, "the record is kept");
        assert!(on_disk.stop_requested_unix_ms.is_some());
    }

    #[test]
    fn services_stop_reports_a_changed_record_with_a_note() {
        // Every process the stop handled is gone, but the record names others
        // now: the verdict says nothing about them, and the report says so.
        let (root, paths) = test_paths("services-stop-record-changed");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-record-changed";
        stop_test_record(&paths, service_id).write().unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut restarted = raw_manifest(&paths, service_id);
            restarted.supervisor_pid = 4_242_424;
            restarted.supervisor_start_ticks = Some(7);
            restarted.write().unwrap();
            bare_termination(Vec::new(), true)
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let _ = fs::remove_dir_all(root);
        let rendered = format!(
            "{:#}",
            outcome.expect_err("a changed record is unconfirmed")
        );
        assert!(
            rendered.contains("the record changed while the stop ran"),
            "{rendered}"
        );
        assert_no_retry_line(&rendered);
        assert!(
            rendered.contains("check: `rocm services` lists what the record names now"),
            "{rendered}"
        );
    }

    #[test]
    fn services_stop_reports_a_confirmed_stop_as_stopped() {
        let (root, paths) = test_paths("services-stop-confirmed");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-confirmed";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            bare_termination(vec![4242], true)
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(answer["status"], "stopped", "{answer}");
        let rendered = outcome.expect("a confirmed stop succeeds");
        assert!(rendered.contains("Local server stopped"), "{rendered}");
        assert!(rendered.contains("stopped processes: 1"), "{rendered}");
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.stop_requested_unix_ms, None);
        assert_eq!(key, None, "a confirmed stop drops the key");
    }

    /// A confirmed stop must not leave the processes it stopped named in the
    /// record. Their PIDs are stale the moment they exit, and a recorded PID
    /// with no start-time token is signalled on trust alone once the OS has
    /// handed it to another process; on Windows the tree kill then takes that
    /// process and everything under it. `rocmd`'s stop clears them for the same
    /// reason (`stop_managed_service_clears_recorded_pids_once_the_processes_are_gone`).
    #[test]
    fn a_confirmed_stop_clears_the_recorded_processes_so_a_second_stop_signals_nothing() {
        use rocm_core::TerminationOutcome::Graceful;
        let (root, paths) = test_paths("services-stop-clears-pids");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-clears-pids";
        let mut record = stop_test_record(&paths, service_id);
        // The dangerous shape: a supervisor recorded without a start-time token,
        // and an engine with one.
        record.supervisor_pid = 4242;
        record.supervisor_start_ticks = None;
        record.engine_pid = Some(4243);
        record.engine_start_ticks = Some(77);
        record.write().unwrap();

        // The termination of each recorded PID, observed: the real routine runs
        // (which PIDs it visits, in which order), only the signalling is replaced.
        let signalled = std::cell::RefCell::new(Vec::new());
        let stop = |record: &ManagedServiceRecord| {
            rocm_core::terminate_recorded_service_pids(record, |identity| {
                signalled.borrow_mut().push(*identity);
                Graceful
            })
        };

        let first = stop_internal_managed_service_with(&paths, service_id, stop).unwrap();
        let after_first = raw_manifest(&paths, service_id);
        let signalled_by_first = signalled.borrow().clone();

        let second = stop_internal_managed_service_with(&paths, service_id, stop).unwrap();
        let after_second = raw_manifest(&paths, service_id);
        let signalled_in_all = signalled.borrow().clone();
        let _ = fs::remove_dir_all(root);

        assert_eq!(first["stopped"], true, "{first}");
        assert_eq!(
            signalled_by_first,
            vec![
                rocm_core::ProcessIdentity::new(4242, None),
                rocm_core::ProcessIdentity::new(4243, Some(77)),
            ]
        );
        // The persisted state after the confirmed stop names no process.
        assert_eq!(after_first.status, "stopped");
        assert_eq!(after_first.stop_requested_unix_ms, None);
        assert_eq!(after_first.supervisor_pid, 0);
        assert_eq!(after_first.supervisor_start_ticks, None);
        assert_eq!(after_first.engine_pid, None);
        assert_eq!(after_first.engine_start_ticks, None);
        // So a stop of the same record signals nothing, and still reads stopped.
        assert_eq!(
            signalled_in_all, signalled_by_first,
            "the second stop reached a PID the first one had already stopped"
        );
        assert_eq!(second["stopped"], true, "{second}");
        assert_eq!(second["signaled_pids"], serde_json::json!([]), "{second}");
        assert_eq!(second["pid_outcomes"], serde_json::json!([]), "{second}");
        assert_eq!(after_second.status, "stopped");
        assert_eq!(after_second.supervisor_pid, 0);
        assert_eq!(after_second.engine_pid, None);
    }

    // Unix only: the processes must really be running. The second stop reads the
    // record through the liveness refresh, which forgets the PIDs of processes it
    // observes gone (`a_stop_completed_by_a_later_listing_leaves_no_pid_for_the_next_stop`),
    // so made-up PIDs would be forgotten and not kept.
    #[cfg(unix)]
    #[test]
    fn an_unconfirmed_stop_keeps_the_recorded_processes_for_the_next_stop() {
        use rocm_core::TerminationOutcome::TimedOut;
        let mut supervisor = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn supervisor");
        let mut engine = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn engine");
        let (supervisor_pid, engine_pid) = (supervisor.id(), engine.id());
        let (root, paths) = test_paths("services-stop-keeps-pids");
        paths.ensure().unwrap();
        let service_id = "svc-services-stop-keeps-pids";
        let mut record = stop_test_record(&paths, service_id);
        record.supervisor_pid = supervisor_pid;
        record.supervisor_start_ticks = Some(5);
        record.engine_pid = Some(engine_pid);
        record.engine_start_ticks = Some(77);
        record.write().unwrap();

        let signalled = std::cell::Cell::new(0_usize);
        let stop = |record: &ManagedServiceRecord| {
            rocm_core::terminate_recorded_service_pids(record, |_| {
                signalled.set(signalled.get() + 1);
                TimedOut
            })
        };
        let first = stop_internal_managed_service_with(&paths, service_id, stop).unwrap();
        let after_first = raw_manifest(&paths, service_id);
        let _ = stop_internal_managed_service_with(&paths, service_id, stop).unwrap();
        let signalled_in_all = signalled.get();
        let _ = supervisor.kill();
        let _ = supervisor.wait();
        let _ = engine.kill();
        let _ = engine.wait();
        let _ = fs::remove_dir_all(root);

        // Nothing was confirmed gone, so the record still names what survived:
        // that is the only thing a later stop can find them by.
        assert_eq!(first["stopped"], false, "{first}");
        assert_eq!(after_first.supervisor_pid, supervisor_pid);
        assert_eq!(after_first.supervisor_start_ticks, Some(5));
        assert_eq!(after_first.engine_pid, Some(engine_pid));
        assert_eq!(after_first.engine_start_ticks, Some(77));
        assert_eq!(signalled_in_all, 4, "each stop tried both processes");
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
            bare_termination(vec![999_999_999], false)
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

    /// The relaunch generation a restart stamps lives only until the restart
    /// publishes: recording the child clears it, and a record whose generation
    /// moved since the restart read it (another restart began, or this one's
    /// stamp is gone) is not written over, as one with other processes is not.
    #[test]
    fn a_restart_publishing_its_child_ends_the_relaunch_generation_and_respects_a_moved_one() {
        let (root, paths) = test_paths("restart-relaunch-generation");
        paths.ensure().unwrap();
        let service_id = "svc-restart-relaunch-generation";
        let mut stopped = stop_test_record(&paths, service_id);
        stopped.status = "stopped".to_owned();
        stopped.clear_recorded_processes();
        stopped.begin_relaunch();
        stopped.write().unwrap();
        let stopped = raw_manifest(&paths, service_id);
        assert!(stopped.relaunch_started_unix_ms.is_some(), "premise");

        // Another restart began meanwhile: the generation on disk is not the one
        // this restart read.
        let mut moved = stopped.clone();
        moved.begin_relaunch();
        moved.write().unwrap();
        assert!(
            moved.relaunch_started_unix_ms > stopped.relaunch_started_unix_ms,
            "a new generation is always distinct from the one before it"
        );
        let refused = record_restarted_child(&paths, &stopped, None, std::process::id());
        let untouched = raw_manifest(&paths, service_id);

        // With the generation as the restart read it, publication goes through.
        stopped.write().unwrap();
        let started = record_restarted_child(&paths, &stopped, None, std::process::id()).unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);

        assert!(refused.is_err(), "a moved generation is not written over");
        assert_eq!(untouched.status, "stopped");
        assert_eq!(untouched.supervisor_pid, 0, "no child was recorded");
        assert_eq!(started.relaunch_started_unix_ms, None);
        assert_eq!(on_disk.relaunch_started_unix_ms, None);
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

    /// The gap a manual restart used to leave open: the confirmed stop just
    /// released the service's port, and nothing reserves it again until the
    /// new child binds and `record_restarted_child` publishes — a concurrent
    /// `rocm serve` launch could claim that exact port in between. Recreate
    /// the race under the shared launch lock and require the restart to
    /// refuse rather than spawn a competitor onto it, following the same
    /// wait-then-publish-then-release shape as
    /// `services_prune_waits_for_the_managed_launch_lock_before_sweeping`.
    #[test]
    fn restart_refuses_a_port_a_launch_claimed_while_it_waited_for_the_lock() -> Result<()> {
        use std::sync::mpsc;

        const WAIT_PROOF: Duration = Duration::from_millis(500);

        let (root, paths) = test_paths("restart-launch-claimed-port");
        paths.ensure()?;
        let service_id = "svc-restart-launch-claimed-port";
        let record = stop_test_record(&paths, service_id);
        let host = record.host.clone();
        let port = record.port;
        record.write()?;

        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (waited, finished) = std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).expect("signal about to restart");
                let outcome =
                    restart_internal_managed_service_with(&paths, service_id, None, |_| {
                        bare_termination(Vec::new(), true)
                    });
                let _ = done_tx.send(outcome.map_err(|error| format!("{error:#}")));
            });
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("restart thread started");
            let early = done_rx.recv_timeout(WAIT_PROOF);
            let waited = early.is_err();
            // Simulate a fresh `rocm serve --gpu auto` publishing a brand-new
            // service on this exact port while the restart was still waiting
            // for the lock it itself now holds.
            let mut competitor = ManagedServiceRecord::new(
                &paths,
                "svc-fresh-launch",
                "lemonade",
                "other-model",
                "other-model-canonical",
                &host,
                port,
                "managed",
                0,
                None,
                None,
                None,
            );
            competitor.status = "starting".to_owned();
            competitor.write().expect("publish the competing record");
            drop(launch_lock);
            let finished = match early {
                Ok(finished) => finished,
                Err(_) => done_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("restart completes once the launch lock is released"),
            };
            (waited, finished)
        });
        let _ = fs::remove_dir_all(root);

        assert!(
            waited,
            "the restart must serialize on the shared launch lock"
        );
        let error =
            finished.expect_err("a live competitor on the same port must refuse the restart");
        assert!(error.contains("svc-fresh-launch"), "{error}");
        assert!(error.contains("claimed that port"), "{error}");
        Ok(())
    }

    /// The restart's stop has already happened by the time it waits for the
    /// launch lock, so a holder that never lets go must not leave the service
    /// down silently: the wait is bounded, and the error says what state the
    /// service is in and names a command that clears it. Both claims are
    /// asserted against the real record and the real CLI hint.
    #[test]
    fn restart_gives_up_on_a_stuck_launch_lock_and_says_the_service_is_stopped() -> Result<()> {
        let (root, paths) = test_paths("restart-launch-lock-timeout");
        paths.ensure()?;
        let service_id = "svc-restart-launch-lock-timeout";
        stop_test_record(&paths, service_id).write()?;
        let holder = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(Some(Duration::from_millis(100))));
        let started = std::time::Instant::now();
        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            bare_termination(Vec::new(), true)
        });
        LAUNCH_LOCK_TIMEOUT_OVERRIDE.with(|cell| cell.set(None));
        let waited = started.elapsed();
        drop(holder);
        let (record, _) = read_managed_service_manifest(&paths, service_id)?;
        let _ = fs::remove_dir_all(root);

        let message = format!(
            "{:#}",
            result.expect_err("a stuck lock must fail the restart")
        );
        assert!(waited < Duration::from_secs(5), "waited {waited:?}");
        assert!(message.contains("stopped but NOT restarted"), "{message}");
        // The named command must be runnable non-interactively: without
        // `--yes` it would bail outside a terminal and never clear the
        // condition. Asserted literally so it does not just echo the builder.
        assert!(
            message.contains(&format!("rocm services restart {service_id} --yes")),
            "{message}"
        );
        // The claim the message makes: the service really is stopped, and no
        // replacement child was started behind the failed wait.
        assert_eq!(record.status, "stopped");
        // The confirmed stop cleared the recorded processes (the record names
        // none), so a supervisor recorded behind the failed wait would show as a
        // nonzero PID.
        assert_eq!(
            record.supervisor_pid, 0,
            "no new supervisor may be recorded behind the failed wait"
        );
        Ok(())
    }

    /// The disk re-check above only catches a recorded competitor. Something
    /// bound to the exact port without ever publishing a manifest for it — a
    /// stray process, a one-off `nc -l` — would sail through that check; only
    /// the loopback lease actually binding the port catches it. A real
    /// listener held on the port is what proves that lease ran, not just a
    /// record on disk.
    #[test]
    fn restart_refuses_a_port_something_is_actually_bound_to() -> Result<()> {
        let (root, paths) = test_paths("restart-port-actually-bound");
        paths.ensure()?;
        let service_id = "svc-restart-port-actually-bound";
        let record = stop_test_record(&paths, service_id);
        let port = record.port;
        record.write()?;

        let _blocker = std::net::TcpListener::bind(("127.0.0.1", port))?;

        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            bare_termination(Vec::new(), true)
        });

        let _ = fs::remove_dir_all(root);
        let error =
            result.expect_err("a port actually bound by someone else must refuse the restart");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("its port is in use"), "{rendered}");
        Ok(())
    }

    /// Mirrors `launch_releases_the_leased_port_before_spawning_the_child` and
    /// its rocmd twin: the restart's probe lease must be closed by the time the
    /// child is spawned, or the child's own bind of the port fails. Without
    /// the `lease.release()` call every loopback restart would die at the
    /// engine bind, and nothing else here would notice.
    #[cfg(not(windows))]
    #[test]
    fn restart_releases_the_leased_port_before_spawning_the_child() -> Result<()> {
        let (root, paths) = test_paths("restart-releases-lease");
        paths.ensure()?;
        let service_id = "svc-restart-releases-lease";
        let record = stop_test_record(&paths, service_id);
        let port = record.port;
        record.write()?;

        let bind_succeeded = std::cell::Cell::new(false);
        let lock_still_held = std::cell::Cell::new(false);
        let lock_path = paths.managed_launch_lock_path();
        let spawn: &RestartSpawnOverride<'_> = &|_command| {
            // The launch lock must outlive the spawn: it is what keeps a
            // concurrent launch from claiming the port before this record is
            // republished.
            lock_still_held
                .set(rocm_core::FileLock::acquire_bounded(&lock_path, Duration::ZERO).is_err());
            // Retry briefly: a sibling test's concurrent fork can hold a copy of
            // the just-released socket until its exec closes it.
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !bind_succeeded.get() && std::time::Instant::now() < deadline {
                bind_succeeded.set(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
            }
            // Fail the spawn on purpose: the observation above is all this
            // test needs, and no real engine should start.
            Err(io::Error::other("spawn intercepted by test"))
        };
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| bare_termination(Vec::new(), true),
            Some(spawn),
        );
        let _ = fs::remove_dir_all(root);

        assert!(
            bind_succeeded.get(),
            "the leased port must already be bindable (released) by the time the child is spawned"
        );
        assert!(
            lock_still_held.get(),
            "the launch lock must still be held when the restart spawns"
        );
        let rendered = format!("{:#}", result.expect_err("the intercepted spawn must fail"));
        assert!(
            rendered.contains("spawn intercepted by test"),
            "the restart must have reached the spawn: {rendered}"
        );
        Ok(())
    }

    /// An older stop finalizing while a restart is between its own confirmed stop
    /// and the publication of its child. The record is then an empty `stopped`
    /// manifest, which is also what an older stop's own completion looks like, but
    /// the restart has restored the endpoint key and is about to serve with it.
    /// The older stop (here the outer one, whose termination step runs the whole
    /// restart, so its snapshot predates it) must neither claim that record nor
    /// delete the key. The restart's spawn is failed on purpose, which leaves the
    /// record exactly as it is at the spawn: published nothing, key restored.
    #[cfg(not(windows))]
    #[test]
    fn an_older_stop_finalizing_before_a_restart_publishes_does_not_delete_its_key() -> Result<()> {
        let (root, paths) = test_paths("older-stop-during-restart");
        paths.ensure()?;
        let service_id = "svc-older-stop-during-restart";
        // A live PID (this test's own), so the restart's own load does not take
        // the older stop's marker for a completed stop and clear the key before
        // the restart has captured it. Nothing signals it: every termination
        // step here is a stand-in.
        let mut planted = stop_test_record(&paths, service_id);
        planted.supervisor_pid = std::process::id();
        planted.write()?;
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123")?;
        let restart_reached_spawn = std::cell::Cell::new(false);

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let spawn: &RestartSpawnOverride<'_> = &|_command| {
                restart_reached_spawn.set(true);
                Err(io::Error::other("spawn intercepted by test"))
            };
            let restart = restart_internal_managed_service_inner(
                &paths,
                service_id,
                None,
                |_| bare_termination(Vec::new(), true),
                Some(spawn),
            );
            assert!(restart.is_err(), "the intercepted spawn fails the restart");
            bare_termination(Vec::new(), true)
        })?;

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert!(restart_reached_spawn.get(), "premise: the restart ran");
        assert_eq!(
            key.as_deref(),
            Some("key-123"),
            "the key the restart restored for its child is not the older stop's to delete"
        );
        assert_eq!(report["stopped"], false, "{report}");
        assert_eq!(on_disk.restart_count, 0, "nothing was published");
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn restart_is_not_blocked_by_a_stopped_or_failed_record_on_the_same_port() -> Result<()> {
        // Mirror of `restart_refuses_a_port_a_launch_claimed...`: `blocks_port` is
        // `is_live() && port == port`, so a `stopped`/`failed` sibling on the same
        // port must not trip the "claimed that port" refusal. Mutation-proof: if
        // `blocks_port` regressed to ignore status, the restart would bail there
        // instead of reaching the real (and, in this test harness, doomed) child
        // spawn below.
        let (root, paths) = test_paths("restart-non-live-same-port");
        paths.ensure()?;
        let service_id = "svc-restart-non-live-same-port";
        let record = stop_test_record(&paths, service_id);
        let host = record.host.clone();
        let port = record.port;
        record.write()?;

        let mut stopped_sibling = ManagedServiceRecord::new(
            &paths,
            "svc-stopped-sibling",
            "lemonade",
            "other-model",
            "other-model-canonical",
            &host,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        stopped_sibling.status = "stopped".to_owned();
        stopped_sibling.write()?;

        let mut failed_sibling = ManagedServiceRecord::new(
            &paths,
            "svc-failed-sibling",
            "lemonade",
            "other-model-2",
            "other-model-2-canonical",
            &host,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        failed_sibling.status = "failed".to_owned();
        failed_sibling.write()?;

        // Fail the spawn on purpose: reaching it is what proves neither sibling
        // blocked the restart, so any other error is a real failure here.
        let spawn: &RestartSpawnOverride<'_> =
            &|_command| Err(io::Error::other("spawn intercepted by test"));
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| bare_termination(Vec::new(), true),
            Some(spawn),
        );

        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", result.expect_err("the intercepted spawn must fail"));
        assert!(
            rendered.contains("spawn intercepted by test"),
            "a stopped or failed sibling on the same port must not block the restart: {rendered}"
        );
        Ok(())
    }

    thread_local! {
        /// Lets a test shorten the restart's launch-lock wait.
        pub(super) static LAUNCH_LOCK_TIMEOUT_OVERRIDE: std::cell::Cell<Option<Duration>> =
            const { std::cell::Cell::new(None) };

        /// Lets a test that lets a restart reach its readiness wait shorten it.
        pub(super) static READINESS_TIMEOUT_OVERRIDE: std::cell::Cell<Option<Duration>> =
            const { std::cell::Cell::new(None) };
    }

    /// The restart holds the launch lock until its record names the new child.
    /// A waiter blocked on the lock since the spawn must therefore read a record
    /// that already carries the child's pid the moment it gets in; releasing
    /// the lock right after the spawn lets it read the stale record instead.
    #[cfg(not(windows))]
    #[test]
    fn restart_holds_the_launch_lock_until_the_record_names_the_new_child() -> Result<()> {
        let (root, paths) = test_paths("restart-lock-until-published");
        paths.ensure()?;
        let service_id = "svc-restart-lock-until-published";
        let record = stop_test_record(&paths, service_id);
        record.write()?;

        READINESS_TIMEOUT_OVERRIDE.with(|cell| cell.set(Some(Duration::ZERO)));
        let (observed_tx, observed_rx) = std::sync::mpsc::channel();
        let lock_path = paths.managed_launch_lock_path();
        let manifest_path = paths.service_manifest_path(service_id);
        let spawn: &RestartSpawnOverride<'_> = &|command| {
            let child = std::process::Command::new("sleep")
                .arg("30")
                .spawn()
                .map_err(|error| io::Error::other(format!("{error} ({command:?})")))?;
            let (lock_path, manifest_path, tx) = (
                lock_path.clone(),
                manifest_path.clone(),
                observed_tx.clone(),
            );
            thread::spawn(move || {
                // Blocks until the restart drops the lock.
                let _lock = rocm_core::FileLock::acquire(&lock_path).expect("acquire");
                let record: ManagedServiceRecord =
                    serde_json::from_slice(&fs::read(manifest_path).expect("read")).expect("parse");
                let _ = tx.send(record.supervisor_pid);
            });
            Ok(child)
        };
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| bare_termination(Vec::new(), true),
            Some(spawn),
        );
        READINESS_TIMEOUT_OVERRIDE.with(|cell| cell.set(None));
        let started = result?;
        let observed = observed_rx.recv_timeout(Duration::from_secs(10));
        // Reap the stand-in engine.
        let _ = std::process::Command::new("kill")
            .arg(started.supervisor_pid.to_string())
            .status();
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            observed.expect("the waiter must get the lock after the restart"),
            started.supervisor_pid,
            "the lock was released before the record named the restarted child"
        );
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn restart_is_not_blocked_by_a_live_record_on_another_port() -> Result<()> {
        // `blocks_port` compares ports. A live service elsewhere must not trip
        // the "claimed that port" refusal: if it ignored the port, every
        // restart would be refused while any other service ran.
        let (root, paths) = test_paths("restart-live-other-port");
        paths.ensure()?;
        let service_id = "svc-restart-live-other-port";
        let record = stop_test_record(&paths, service_id);
        let host = record.host.clone();
        let port = record.port;
        record.write()?;

        let mut other = ManagedServiceRecord::new(
            &paths,
            "svc-live-other-port",
            "lemonade",
            "other-model",
            "other-model-canonical",
            &host,
            port.wrapping_add(1),
            "managed",
            0,
            None,
            None,
            None,
        );
        other.status = "starting".to_owned();
        other.write()?;

        let spawn: &RestartSpawnOverride<'_> =
            &|_command| Err(io::Error::other("spawn intercepted by test"));
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| bare_termination(Vec::new(), true),
            Some(spawn),
        );
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", result.expect_err("the intercepted spawn must fail"));
        assert!(
            rendered.contains("spawn intercepted by test"),
            "a live service on another port must not block the restart: {rendered}"
        );
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
            // Skipped by the shared stop routine, which never
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

        let stops = stop_recorded_processes(&record);

        assert!(
            stops.iter().all(|stop| !stop.signaled()),
            "must not signal a mismatched identity"
        );
        // A recycled PID means the recorded service process itself is gone.
        assert!(stops.iter().all(|stop| stop.stopped()));
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

        let stops = stop_recorded_processes(&record);
        let all_stopped = stops.iter().all(|stop| stop.stopped());
        let mut signaled: Vec<u32> = stops
            .iter()
            .filter(|stop| stop.signaled())
            .map(|stop| stop.pid)
            .collect();
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

        let stops = stop_recorded_processes(&record);
        let signaled: Vec<u32> = stops
            .iter()
            .filter(|stop| stop.signaled())
            .map(|stop| stop.pid)
            .collect();

        assert_eq!(signaled, vec![pid]);
        assert!(stops.iter().all(|stop| stop.stopped()));
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

    // ---- This module's stop, through the routine it shares with `rocmd`. ----
    //
    // The routine itself (visiting order, start-time pairing, the `self` row,
    // the per-outcome classification, the Windows tree-kill step) is tested in
    // `rocm-core`'s `proc_lifecycle.rs`. These check that this caller reaches it
    // and reports what it returns. Every PID below is one that cannot be
    // running, so nothing is signalled and the tests behave the same on every
    // host.

    /// A PID no host has in use, so the stop reports it as already gone.
    const ABSENT_SUPERVISOR_PID: u32 = 999_999_998;
    const ABSENT_ENGINE_PID: u32 = 999_999_997;

    /// Run the production stop of a record naming `supervisor` and `engine`,
    /// and return its report.
    fn stop_report_for(name: &str, supervisor: u32, engine: Option<u32>) -> serde_json::Value {
        let (root, paths) = test_paths(name);
        paths.ensure().unwrap();
        let service_id = "svc-stop-through-the-shared-routine";
        let mut record = stop_test_record(&paths, service_id);
        record.supervisor_pid = supervisor;
        record.engine_pid = engine;
        record.write().unwrap();

        let report =
            stop_internal_managed_service_with(&paths, service_id, stop_recorded_processes)
                .unwrap();
        let _ = fs::remove_dir_all(root);
        report
    }

    #[test]
    fn a_stop_reports_the_supervisor_then_the_engine_in_that_order() {
        let report = stop_report_for(
            "managed-stop-order",
            ABSENT_SUPERVISOR_PID,
            Some(ABSENT_ENGINE_PID),
        );

        assert_eq!(
            report["pid_outcomes"],
            serde_json::json!([
                {"pid": ABSENT_SUPERVISOR_PID, "role": "supervisor", "outcome": "already_gone"},
                {"pid": ABSENT_ENGINE_PID, "role": "engine", "outcome": "already_gone"},
            ])
        );
        assert_eq!(report["signaled_pids"], serde_json::json!([]));
        assert_eq!(report["stopped"], true);
    }

    #[test]
    fn a_stop_reports_a_pid_recorded_twice_once_under_both_roles() {
        let report = stop_report_for(
            "managed-stop-aliased",
            ABSENT_SUPERVISOR_PID,
            Some(ABSENT_SUPERVISOR_PID),
        );

        assert_eq!(
            report["pid_outcomes"],
            serde_json::json!([
                {"pid": ABSENT_SUPERVISOR_PID, "role": "supervisor_and_engine", "outcome": "already_gone"},
            ])
        );
    }

    /// Both binaries report the stopping process's own PID as `self`, never
    /// signal it, and do not let it stand in the way of a confirmed stop. `rocm`
    /// used to drop it from the report, so `pid_outcomes` showed one entry fewer
    /// than `rocmd`'s for the same record.
    #[test]
    fn a_stop_reports_its_own_pid_as_self() {
        let own = std::process::id();

        let report = stop_report_for("managed-stop-own-pid", own, Some(ABSENT_ENGINE_PID));

        assert_eq!(
            report["pid_outcomes"],
            serde_json::json!([
                {"pid": own, "role": "supervisor", "outcome": "self"},
                {"pid": ABSENT_ENGINE_PID, "role": "engine", "outcome": "already_gone"},
            ])
        );
        assert_eq!(
            report["signaled_pids"],
            serde_json::json!([]),
            "the stopping process is never signalled"
        );
        assert_eq!(report["stopped"], true);
    }

    #[test]
    fn a_stop_of_a_record_naming_only_its_own_pid_confirms() {
        let own = std::process::id();

        let report = stop_report_for("managed-stop-only-own-pid", own, None);

        assert_eq!(
            report["pid_outcomes"],
            serde_json::json!([{"pid": own, "role": "supervisor", "outcome": "self"}])
        );
        assert_eq!(report["stopped"], true);
    }

    /// The `self` row is not one of the outcomes that leave a stop unconfirmed,
    /// so it never reaches the report's `still running` or `not signaled` lines.
    #[test]
    fn a_self_row_adds_no_line_to_an_unconfirmed_report() {
        let (root, paths) = test_paths("managed-stop-self-row-unconfirmed");
        paths.ensure().unwrap();
        let service_id = "svc-self-row-unconfirmed";
        stop_test_record(&paths, service_id).write().unwrap();
        let own = std::process::id();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![
                rocm_core::RecordedPidStop {
                    pid: own,
                    role: "supervisor",
                    outcome: rocm_core::RecordedPidOutcome::OwnProcess,
                },
                rocm_core::RecordedPidStop {
                    role: "engine",
                    ..stop_of(999_999_999, rocm_core::TerminationOutcome::TimedOut)
                },
            ]
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("unconfirmed"));

        assert!(
            rendered.contains("still running: process 999999999 (engine)"),
            "{rendered}"
        );
        assert!(
            !rendered.contains(&format!("process {own}")),
            "the stopping process is not a survivor: {rendered}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_signals_the_supervisor_before_the_engine() {
        let sup = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn supervisor");
        let eng = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn engine");
        let (sup_pid, eng_pid) = (sup.id(), eng.id());
        let (root, paths) = test_paths("managed-stop-signal-order");
        let mut record =
            managed_record_for_pid(&paths, sup_pid, rocm_core::process_start_ticks(sup_pid));
        record.engine_pid = Some(eng_pid);
        record.engine_start_ticks = rocm_core::process_start_ticks(eng_pid);

        let stops = stop_recorded_processes(&record);
        let (mut sup, mut eng) = (sup, eng);
        let _ = sup.wait();
        let _ = eng.wait();
        let _ = fs::remove_dir_all(root);

        // Not sorted: the order is the routine's, and a copy that visited the
        // engine first would fail here.
        assert_eq!(
            stops
                .iter()
                .filter(|stop| stop.signaled())
                .map(|stop| stop.pid)
                .collect::<Vec<_>>(),
            vec![sup_pid, eng_pid]
        );
        assert!(stops.iter().all(|stop| stop.stopped()));
    }

    /// A second copy of the shared routine, or of its grace constant, is how the
    /// two binaries diverged in the first place; the shared one lives in
    /// `rocm-core`'s `proc_lifecycle.rs`. The needles are assembled so that this
    /// test does not match itself.
    #[test]
    fn this_module_carries_no_private_copy_of_the_shared_stop_routine() {
        let source = include_str!("service_control.rs");
        for needle in [
            ["const ", "MANAGED_STOP_GRACE"].concat(),
            ["fn ", "terminate_recorded_service_pids"].concat(),
            ["fn ", "terminate_recorded_pid"].concat(),
            ["Command::new(\"task", "kill\")"].concat(),
        ] {
            assert!(
                !source.contains(&needle),
                "`{needle}` is defined in `rocm-core`, not here"
            );
        }
    }

    /// The stop and the liveness read after it, end to end: a stop that cannot
    /// verify a recorded process leaves the key and the stop request, and a later
    /// refresh that still cannot query the process must not take them away or
    /// reconcile the record to `stopped`. The refresh reads the record the stop
    /// wrote, as a service listing does.
    #[test]
    fn an_unverified_stop_keeps_its_key_and_marker_through_a_later_liveness_refresh() {
        let (root, paths) = test_paths("stop-unverified-then-refresh");
        paths.ensure().unwrap();
        let service_id = "svc-unverified-then-refresh";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![stop_of(
                999_999_999,
                rocm_core::TerminationOutcome::Unverified,
            )]
        })
        .unwrap();
        assert_eq!(report["stopped"], false, "{report}");
        let (mut record, _) = read_managed_service_manifest(&paths, service_id).unwrap();
        assert!(
            record.stop_requested_unix_ms.is_some(),
            "the stop is pending"
        );

        let changed = crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
            &paths,
            &mut record,
            |_| rocm_core::ProcessProbe::Unqueryable,
        );
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);

        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
        assert!(
            record.stop_requested_unix_ms.is_some(),
            "the stop request is kept"
        );
        assert!(
            !changed,
            "nothing about the record may change while the process is unqueryable"
        );
    }

    /// An unconfirmed stop that a later listing completes (every recorded process
    /// is observed gone) must leave a record that names no process, as a confirmed
    /// stop does: otherwise the next `services stop` signals whatever the OS has
    /// since given those PIDs to, and a PID with no start-time token cannot be
    /// refuted. The same record, written back and stopped again, visits no PID.
    #[test]
    fn a_stop_completed_by_a_later_listing_leaves_no_pid_for_the_next_stop() {
        let (root, paths) = test_paths("stop-completed-by-listing-clears-pids");
        paths.ensure().unwrap();
        let service_id = "svc-completed-by-listing";
        let mut planted = stop_test_record(&paths, service_id);
        planted.supervisor_pid = 4242;
        planted.supervisor_start_ticks = None;
        planted.engine_pid = Some(4243);
        planted.engine_start_ticks = Some(77);
        planted.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![stop_of(4242, rocm_core::TerminationOutcome::Unverified)]
        })
        .unwrap();
        assert_eq!(report["stopped"], false, "{report}");
        let (mut record, _) = read_managed_service_manifest(&paths, service_id).unwrap();
        assert_eq!(
            record.supervisor_pid, 4242,
            "an unconfirmed stop keeps the PIDs"
        );

        // The listing that finds both processes gone completes the stop.
        let changed = crate::managed_service_status::refresh_managed_service_runtime_liveness_with(
            &paths,
            &mut record,
            |_| rocm_core::ProcessProbe::Gone,
        );
        assert!(changed);
        record.write().unwrap();
        let after_listing = raw_manifest(&paths, service_id);

        let visited = std::cell::RefCell::new(Vec::new());
        let second = stop_internal_managed_service_with(&paths, service_id, |record| {
            rocm_core::terminate_recorded_service_pids(record, |identity| {
                visited.borrow_mut().push(*identity);
                rocm_core::TerminationOutcome::Graceful
            })
        })
        .unwrap();
        let visited = visited.into_inner();
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);

        assert_eq!(after_listing.status, "stopped");
        assert_eq!(after_listing.stop_requested_unix_ms, None);
        assert_eq!(key, None, "the deferred cleanup drops the key");
        assert_eq!(
            (
                after_listing.supervisor_pid,
                after_listing.supervisor_start_ticks,
                after_listing.engine_pid,
                after_listing.engine_start_ticks
            ),
            (0, None, None, None)
        );
        assert!(
            visited.is_empty(),
            "a later stop reached a PID the listing had observed gone: {visited:?}"
        );
        assert_eq!(second["stopped"], true, "{second}");
        assert_eq!(second["pid_outcomes"], serde_json::json!([]), "{second}");
    }

    /// A stop whose recorded process was seen to exit but whose tree kill did not
    /// complete says so in those words, not as a timeout, and keeps what an
    /// unconfirmed stop keeps: the claim and the state are checked together.
    #[test]
    fn a_stop_with_an_unconfirmed_tree_kill_says_what_happened_and_keeps_the_service() {
        let (root, paths) = test_paths("services-stop-descendants-unconfirmed");
        paths.ensure().unwrap();
        let service_id = "svc-descendants-unconfirmed";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![stop_of(
                999_999_999,
                rocm_core::TerminationOutcome::DescendantsUnconfirmed,
            )]
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        // What the report says about the warning, checked against what happens
        // next: a later stop finds the root gone and, with no new evidence about
        // what was under it, reports the stop confirmed.
        let later = stop_internal_managed_service_with(&paths, service_id, |_| {
            vec![stop_of(
                999_999_999,
                rocm_core::TerminationOutcome::AlreadyGone,
            )]
        })
        .unwrap();
        let _ = fs::remove_dir_all(root);

        assert_eq!(answer["status"], "stop_unconfirmed", "{answer}");
        let rendered = format!("{:#}", outcome.expect_err("not a confirmed stop"));
        assert!(
            rendered.contains(
                "not confirmed: process 999999999 (supervisor_and_engine) exited, but a process under it may still be running"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains("rocm does not keep this warning, so a later stop or listing will find the process gone and report the stop confirmed"),
            "the report must not imply the warning outlives this run: {rendered}"
        );
        assert_eq!(
            later["stopped"], true,
            "the claim the report makes is true: {later}"
        );
        assert!(
            !rendered.contains("still running: process")
                && !rendered.contains("did not exit after it was signaled"),
            "the root was seen to exit, so nothing outlived a deadline: {rendered}"
        );
        assert!(!rendered.contains("Local server stopped"), "{rendered}");
        assert_no_retry_line(&rendered);
        // The claim of what was kept, against what was kept.
        assert!(rendered.contains("kept: the server record"), "{rendered}");
        assert_eq!(on_disk.supervisor_pid, 999_999_999);
        assert!(on_disk.stop_requested_unix_ms.is_some());
        assert_eq!(key.as_deref(), Some("key-123"));
    }

    /// A root that exited by itself before its tree could be taken down confirms
    /// the stop only through the record's other roles, through the real routine
    /// and this caller's own folding: the root's row alone is not enough, and the
    /// record, its PIDs, its stop request and its key are kept unless the engine
    /// role is confirmed gone too.
    #[test]
    fn a_root_that_exited_alone_confirms_the_stop_only_when_the_engine_role_does() {
        use rocm_core::TerminationOutcome::{ExitedBeforeTreeKill, Forced, TimedOut};

        for (engine, confirmed) in [(Forced, true), (TimedOut, false)] {
            let (root, paths) = test_paths("services-stop-root-exited-alone");
            paths.ensure().unwrap();
            let service_id = "svc-root-exited-alone";
            let mut record = stop_test_record(&paths, service_id);
            record.supervisor_pid = 999_999_998;
            record.engine_pid = Some(999_999_997);
            record.write().unwrap();
            endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

            let report = stop_internal_managed_service_with(&paths, service_id, |record| {
                rocm_core::terminate_recorded_service_pids(record, |identity| {
                    if identity.pid == 999_999_998 {
                        ExitedBeforeTreeKill
                    } else {
                        engine
                    }
                })
            })
            .unwrap();
            let on_disk = raw_manifest(&paths, service_id);
            let key = endpoint_keys::endpoint_api_key(&paths, service_id);
            let _ = fs::remove_dir_all(root);

            assert_eq!(report["stopped"], confirmed, "{engine:?}: {report}");
            if confirmed {
                assert_eq!(on_disk.status, "stopped");
                assert_eq!(
                    on_disk.supervisor_pid, 0,
                    "the confirmed stop clears the PIDs"
                );
                assert_eq!(key, None, "and the key");
            } else {
                assert_eq!(on_disk.supervisor_pid, 999_999_998, "the PIDs are kept");
                assert!(on_disk.stop_requested_unix_ms.is_some());
                assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
            }
        }
    }

    /// A lone root that exited by itself is not vouched for by anything else the
    /// record names, so this caller does not clear the record on its say-so.
    #[test]
    fn a_lone_root_that_exited_alone_leaves_the_stop_unconfirmed() {
        let (root, paths) = test_paths("services-stop-lone-root-exited-alone");
        paths.ensure().unwrap();
        let service_id = "svc-lone-root-exited-alone";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |record| {
            rocm_core::terminate_recorded_service_pids(record, |_| {
                rocm_core::TerminationOutcome::ExitedBeforeTreeKill
            })
        })
        .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);

        let rendered = format!("{:#}", outcome.expect_err("not a confirmed stop"));
        assert!(
            rendered.contains("was already gone when the processes under it were to be taken down"),
            "{rendered}"
        );
        assert!(!rendered.contains("Local server stopped"), "{rendered}");
        assert_no_retry_line(&rendered);
        assert_eq!(on_disk.supervisor_pid, 999_999_999);
        assert!(on_disk.stop_requested_unix_ms.is_some());
        assert_eq!(key.as_deref(), Some("key-123"));
    }
}
