// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Stopping and restarting a managed local server: the process control behind
//! `rocm services stop|restart` and the internal sandbox tool.
//!
//! Mechanically relocated from `services_text.rs` with no behavior change.
//!
//! This module owns: the sandbox-tool runner (`run_internal_sandbox_tool`), the
//! stop path (`terminate_recorded_service_pids`, the `MANAGED_STOP_GRACE`
//! constant, the private `RecordedPidOutcome`/`RecordedStopTermination` types
//! and the `scripted_stop_outcome` end-to-end hook, `stop_internal_managed_service`,
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
//! `service_launch.rs`; `restart_internal_managed_service` from
//! `runtime_services.rs`. Every other item has no caller outside this module and
//! stays private.
//!
//! The stop path reads the record through `services_text.rs`
//! (`load_managed_service`, `read_managed_service_manifest`) and shapes the
//! `stop_server` answer with its `internal_stop_server_value`, which stays there
//! next to `service_stop_unconfirmed` and the renderer that reads the same
//! `stop_unconfirmed` status. `service_launch.rs` and this module import each
//! other: this module takes `ensure_public_service_has_endpoint_key` and the
//! spawn helpers from it, and it takes `run_internal_sandbox_tool` from here.
//!
//! The regression tests for the stop and restart paths live in this module's own
//! `mod tests`. They reach shared fixtures (`test_paths`, `ScopedTestEnv`,
//! `plant_service_record_on_runtime`, `OLD_RUNTIME_KEY`, `NEW_RUNTIME_KEY`)
//! through `crate::tests::`, and `runtime_services::` through its own
//! `use crate::runtime_services;` for the tests that drive a restart through
//! `runtime_services`'s public entry points.

use std::fs;
#[cfg(not(windows))]
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, EndpointReadiness, ManagedServiceRecord};
use rocm_engine_protocol::{EngineMethod, StopRequest, StopResponse};

use crate::endpoint_keys;
use crate::engine_ipc::{builtin_engine_serve_http_args, engine_request};
#[cfg(not(windows))]
use crate::engine_runtime_select::apply_app_path_env;
#[cfg(windows)]
use crate::engine_runtime_select::{app_path_env_var_refs, app_path_env_var_values};
use crate::engine_runtime_select::{
    http_post_local_service_json, managed_service_launcher_path, parse_engine_recipe_json_arg,
    wait_for_service_http_ready,
};
use crate::engines_cmd::env_root_for_service;
use crate::gpu_policy::parse_device_policy;
use crate::managed_service_status::load_managed_services;
use crate::service_launch::ensure_public_service_has_endpoint_key;
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

/// How long a managed stop waits for each recorded process to actually exit
/// after `SIGTERM` before escalating to `SIGKILL` and then reporting a timeout.
const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// One recorded PID's fate in a managed stop: the role the record gave it, and
/// what [`rocm_core::terminate_verified`] did about it. Kept per PID because the
/// outcomes that leave a stop unconfirmed are reported differently: a
/// `TimedOut` PID was signalled and it or a process under it outlived the
/// deadline, while an `Unverified` one is not signalled by this stop, and the
/// report names a command to check on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedPidOutcome {
    pid: u32,
    /// `supervisor`, `engine`, or `supervisor_and_engine` when the record names
    /// one PID for both — the same labels `rocmd`'s `pid_outcomes` uses.
    role: &'static str,
    outcome: rocm_core::TerminationOutcome,
}

/// What a managed stop did to the processes its record names.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RecordedStopTermination {
    /// Every PID that received a signal, including one that then outlived the
    /// deadline.
    signaled_pids: Vec<u32>,
    /// Whether every recorded process is confirmed no longer running.
    all_stopped: bool,
    pid_outcomes: Vec<RecordedPidOutcome>,
}

/// Terminate the processes recorded for a managed service, verifying each PID's
/// identity first so a recycled PID never causes an unrelated process tree to be
/// killed.
///
/// The supervisor (launcher) and the engine server are distinct processes, so
/// each PID is paired with **its own** start-time token (`supervisor_start_ticks`
/// vs `engine_start_ticks`). When they resolve to the same PID (e.g. before the
/// engine state has been refreshed) the entry is de-duplicated, preferring a
/// known token. Returns the PIDs actually signalled, whether every recorded
/// process is confirmed no longer running, and each PID's own outcome.
fn terminate_recorded_service_pids(record: &ManagedServiceRecord) -> RecordedStopTermination {
    // Build the (pid, own-token, role) work list, de-duplicating on PID and
    // preferring an entry that carries a verifiable start-time.
    let mut entries: Vec<(u32, Option<u64>, &'static str)> = Vec::new();
    for (pid, ticks, role) in [
        (
            Some(record.supervisor_pid),
            record.supervisor_start_ticks,
            "supervisor",
        ),
        (record.engine_pid, record.engine_start_ticks, "engine"),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0 && *pid != std::process::id()) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _, _)| *seen == pid) {
            if existing.1.is_none() {
                existing.1 = ticks;
            }
            existing.2 = "supervisor_and_engine";
        } else {
            entries.push((pid, ticks, role));
        }
    }

    let mut termination = RecordedStopTermination {
        all_stopped: true,
        ..RecordedStopTermination::default()
    };
    for (pid, ticks, role) in entries {
        let identity = rocm_core::ProcessIdentity::new(pid, ticks);
        // The managed stop contract is definitive: after the adapter gets its
        // graceful opportunity, this cleanup pass forces any verified survivor
        // so the command does not report success while a GPU worker remains.
        let outcome = scripted_stop_outcome().unwrap_or_else(|| {
            rocm_core::terminate_verified(
                &identity,
                rocm_core::KillScope::Tree,
                MANAGED_STOP_GRACE,
                true,
            )
        });
        if !outcome.stopped() {
            termination.all_stopped = false;
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
            termination.signaled_pids.push(pid);
        }
        termination
            .pid_outcomes
            .push(RecordedPidOutcome { pid, role, outcome });
    }
    termination
}

/// E2E hook: when `ROCM_E2E_STOP_OUTCOME` is `timed_out` or `unverified`, a
/// managed stop reports that outcome for every recorded PID without signalling
/// anything.
///
/// Neither outcome can be staged from outside the binary: `TimedOut` needs a
/// process that survives `SIGKILL`, and `Unverified` one whose recorded
/// start-time cannot be read back. Without this seam the "stop not confirmed"
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
    terminate: impl FnOnce(&ManagedServiceRecord) -> RecordedStopTermination,
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
    terminate: impl FnOnce(&ManagedServiceRecord) -> RecordedStopTermination,
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
    let RecordedStopTermination {
        signaled_pids,
        all_stopped: terminated_all,
        pid_outcomes,
    } = terminate(&stopping);
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
    let record_changed = !stopping.names_same_processes(&record);
    let all_stopped = terminated_all && !record_changed && engine_stop_confirms;
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
        "pid_outcomes": pid_outcomes
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "pid": entry.pid,
                    "role": entry.role,
                    "outcome": entry.outcome.as_str(),
                })
            })
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
    terminate: impl FnOnce(&ManagedServiceRecord) -> RecordedStopTermination,
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
        started.served_model_name(),
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

    /// A stop stand-in's bare verdict, without per-PID detail.
    fn bare_termination(signaled_pids: Vec<u32>, all_stopped: bool) -> RecordedStopTermination {
        RecordedStopTermination {
            signaled_pids,
            all_stopped,
            pid_outcomes: Vec::new(),
        }
    }

    /// A stop stand-in whose one recorded process (the record's shared
    /// supervisor/engine PID) was signalled and outlived the deadline.
    fn timed_out_termination(pid: u32) -> RecordedStopTermination {
        RecordedStopTermination {
            signaled_pids: vec![pid],
            all_stopped: false,
            pid_outcomes: vec![RecordedPidOutcome {
                pid,
                role: "supervisor_and_engine",
                outcome: rocm_core::TerminationOutcome::TimedOut,
            }],
        }
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

        let report =
            stop_internal_managed_service_with(&paths, service_id, |_| RecordedStopTermination {
                signaled_pids: Vec::new(),
                all_stopped: false,
                pid_outcomes: vec![RecordedPidOutcome {
                    pid: 999_999_999,
                    role: "supervisor_and_engine",
                    outcome: rocm_core::TerminationOutcome::Unverified,
                }],
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

        let report =
            stop_internal_managed_service_with(&paths, service_id, |_| RecordedStopTermination {
                signaled_pids: vec![999_999_999],
                all_stopped: false,
                pid_outcomes: vec![
                    RecordedPidOutcome {
                        pid: 999_999_999,
                        role: "supervisor",
                        outcome: rocm_core::TerminationOutcome::TimedOut,
                    },
                    RecordedPidOutcome {
                        pid: 888_888_888,
                        role: "engine",
                        outcome: rocm_core::TerminationOutcome::Unverified,
                    },
                ],
            })
            .unwrap();
        let answer = internal_stop_server_value(report);
        let outcome = finish_service_action(&paths, "stop_server", service_id, &answer);
        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", outcome.expect_err("unconfirmed"));
        assert_no_retry_line(&rendered);
        assert!(
            rendered.contains("check: `ps -p 888888888 -o pid,args`"),
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

        let RecordedStopTermination {
            signaled_pids: signaled,
            all_stopped,
            ..
        } = terminate_recorded_service_pids(&record);

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

        let RecordedStopTermination {
            signaled_pids: mut signaled,
            all_stopped,
            ..
        } = terminate_recorded_service_pids(&record);
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

        let RecordedStopTermination {
            signaled_pids: signaled,
            all_stopped,
            ..
        } = terminate_recorded_service_pids(&record);

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
}
