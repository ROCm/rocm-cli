// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{AppPaths, ManagedServiceRecord};
use rocm_engine_protocol::{EngineMethod, StopRequest, StopResponse};
use serde_json::Value;
use serde_json::json;
#[cfg(windows)]
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

pub(crate) fn stop_managed_service(paths: &AppPaths, service_id: &str) -> Result<Value> {
    stop_managed_service_with(paths, service_id, terminate_recorded_pid)
}

fn find_managed_service(paths: &AppPaths, service_id: &str) -> Result<ManagedServiceRecord> {
    crate::persistence::load_managed_services(paths)?
        .into_iter()
        .find(|record| record.service_id == service_id)
        .with_context(|| format!("managed service `{service_id}` not found"))
}

/// [`stop_managed_service`], with the per-PID termination supplied by the
/// caller. Production passes [`terminate_recorded_pid`]; a test passes a
/// stand-in for the outcomes it cannot construct on demand. Only the signalling
/// and its verdict are replaced: building the PID list, the manifest writes,
/// the key cleanup and the report all run for real.
pub(crate) fn stop_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl Fn(&rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
) -> Result<Value> {
    stop_managed_service_via(paths, service_id, terminate, request_engine_stop)
}

fn request_engine_stop(paths: &AppPaths, record: &ManagedServiceRecord) -> Result<StopResponse> {
    crate::engine_ipc::engine_request::<_, StopResponse>(
        paths,
        &record.engine,
        &record.service_id,
        EngineMethod::Stop,
        &StopRequest {
            service_id: record.service_id.clone(),
            force: true,
        },
    )
}

/// [`stop_managed_service_with`], with the engine `Stop` request supplied by the
/// caller as well, so a test can see what was asked of the engine and make it
/// fail.
pub(crate) fn stop_managed_service_via(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl Fn(&rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
    engine_stop: impl Fn(&AppPaths, &ManagedServiceRecord) -> Result<StopResponse>,
) -> Result<Value> {
    let mut stopping = find_managed_service(paths, service_id)?;
    // Record the request before the stop starts, not only after it ends. A stop
    // can take up to `MANAGED_STOP_GRACE` twice per recorded PID, and for that
    // whole window the service still reads `ready`/`running` while its endpoint
    // stops answering — exactly what the daemon's `server-recover` watcher
    // restarts. The marker is what that watcher skips (`stop_requested`), so it
    // has to be on disk while the stop is in flight.
    let requested_at = rocm_core::unix_time_millis();
    stopping.stop_requested_unix_ms = Some(requested_at);
    stopping.write()?;
    // Ask the engine to stop before signalling recorded PIDs. Lemonade is
    // skipped: `rocm services stop` unloads its model over HTTP rather than
    // sending it an engine `Stop`, and this path has never touched it. For a
    // container-backed engine (`aim`) this is the only thing that stops the
    // container, since its recorded PIDs are only the supervisor that follows it.
    let engine_stop = (stopping.engine != "lemonade").then(|| engine_stop(paths, &stopping));
    let stops = terminate_recorded_service_pids(&stopping, terminate);
    let pids_where = |keep: fn(&RecordedPidStop) -> bool| -> Vec<u32> {
        stops
            .iter()
            .filter(|stop| keep(stop))
            .map(|s| s.pid)
            .collect()
    };
    let signaled_pids = pids_where(|stop| stop.signaled);
    let force_signaled_pids = pids_where(|stop| stop.forced);
    // Everything that was deliberately not signalled, which is no longer one
    // kind of thing: the stopping process itself; a PID that was already gone;
    // a PID that is live but provably belongs to somebody else now; and a live
    // PID whose identity could not be read at all, which may still be the engine
    // and may still hold the device. Only `pid_outcomes` separates them, and
    // only `stopped` reports that the last of those is not a completed stop.
    let skipped_pids = pids_where(|stop| !stop.signaled);
    let pid_outcomes = stops
        .iter()
        .map(|stop| {
            json!({
                "pid": stop.pid,
                "role": stop.role,
                "outcome": stop.outcome,
            })
        })
        .collect::<Vec<_>>();
    // Re-read the record rather than write back the copy loaded before the
    // stop: `write` replaces the whole file, and other writers — a supervisor
    // recording its startup phase, `rocm`'s liveness refresh, a restart — may
    // have persisted something while the stop ran. Only this stop's own changes
    // are applied on top of what is there now.
    let mut record = find_managed_service(paths, service_id)?;
    // The verdict covers only the processes this stop actually handled. If the
    // record names different ones now — a restart recorded fresh PIDs while the
    // stop was running — those are processes it never touched, and claiming the
    // service stopped, clearing their PIDs or dropping the key they serve with
    // would each be false.
    let engine_stop_confirms = !rocm_core::engine_stop_decides_confirmation(&stopping.engine)
        || engine_stop.as_ref().is_some_and(Result::is_ok);
    let all_stopped = stops.iter().all(|stop| stop.stopped)
        && stopping.names_same_processes(&record)
        && engine_stop_confirms;
    // A PID whose identity cannot be read at all — `/proc/<pid>/stat`
    // unreadable while `kill(pid, 0)` still answers: another uid, `hidepid`, an
    // LSM — comes back `Unverified`, which never counts as stopped. Nor can it
    // be forced: `terminate_verified` returns before it consults `force`, so a
    // service holding such a PID stays unconfirmed and cannot be stopped from
    // the CLI at all. Deliberate: signalling a process nobody could identify is
    // the blind kill the identity check exists to prevent.
    // Claim a stop only when every recorded process is confirmed gone, and
    // otherwise record that one was *asked for*. `rocm services stop`
    // (`stop_internal_managed_service`) keeps the same contract on these same
    // records — marker persisted before anything is signalled, then status,
    // marker and key changed only on a confirmed stop — so a daemon watching
    // either one sees the same thing. Clearing the recorded PIDs below is this
    // stop's alone.
    if all_stopped {
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = None;
        // Recorded PIDs go stale the instant their processes exit, and a stale
        // PID is exactly what the identity check exists to catch — so do not
        // leave one behind for the next stop to find.
        record.supervisor_pid = 0;
        record.supervisor_start_ticks = None;
        record.engine_pid = None;
        record.engine_start_ticks = None;
    } else {
        // The recorded PIDs stay: an unconfirmed survivor is still the
        // service's, and the record is what names it.
        //
        // The marker that tells a service the operator stopped from one that
        // merely crashed. While it is set the daemon will not recover the
        // service (`stop_requested`), so an unconfirmed stop is not undone.
        //
        // It is also what the deferred key cleanup keys on — but that cleanup
        // lives in the `rocm` binary (`settle_pending_stop_key_cleanup`, run by
        // its liveness refresh), not here. A caller that only ever drives
        // `rocmd` keeps both the marker and the key on disk until a later stop
        // confirms; that confirmation is only as good as the check above it,
        // which cannot see a survivor under a dead recorded root.
        record.stop_requested_unix_ms = Some(requested_at);
    }
    record.write()?;
    // Drop the 0600 endpoint key file with the service, rather than stranding a
    // plaintext secret on disk for something that is no longer running.
    // Best-effort and idempotent: a missing key file is not an error, so a
    // loopback service that never had one, and a repeated stop, both pass
    // through here harmlessly.
    //
    // Gated, because an unconfirmed stop may have left the engine alive and
    // still enforcing that key — and this is the only copy. Discarding it would
    // lock the CLI's own probes, chat and service discovery out of a service
    // that is otherwise fine, with no way to re-mint it. The marker written
    // above defers the cleanup instead; see there for who performs it.
    if all_stopped {
        let _ = std::fs::remove_file(rocm_engine_protocol::endpoint_key_file_path(
            paths, service_id,
        ));
    }
    Ok(json!({
        "service": record,
        "signaled_pids": signaled_pids,
        "force_signaled_pids": force_signaled_pids,
        "skipped_pids": skipped_pids,
        "pid_outcomes": pid_outcomes,
        "engine_stop": engine_stop_report(engine_stop),
        "stopped": all_stopped,
    }))
}

/// The engine `Stop` outcome as it appears in a stop report; `null` when no
/// engine `Stop` was sent.
fn engine_stop_report(engine_stop: Option<Result<StopResponse>>) -> Value {
    match engine_stop {
        None => Value::Null,
        Some(Ok(response)) => json!({
            "attempted": true,
            "stopped": response.stopped,
            "graceful": response.graceful,
        }),
        Some(Err(error)) => json!({
            "attempted": true,
            "error": error.to_string(),
        }),
    }
}

/// Whether a [`stop_managed_service`] report confirms that every recorded
/// process is gone. A report without the verdict counts as unconfirmed, so a
/// caller can never mistake a missing field for a completed stop.
pub(crate) fn stop_report_confirmed(report: &Value) -> bool {
    report.get("stopped").and_then(Value::as_bool) == Some(true)
}

/// How long a stop waits for a recorded process tree to exit on its own before
/// escalating to a forced kill. Matches `rocm services stop`, which terminates
/// the same records.
const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// What a stop did about one PID recorded in a service manifest.
struct RecordedPidStop {
    pid: u32,
    /// Which recorded PID this is: `supervisor`, `engine`, or
    /// `supervisor_and_engine` when the record holds the same PID for both.
    role: &'static str,
    /// Stable label for what happened, from
    /// [`rocm_core::TerminationOutcome::as_str`], plus `self` for the stopping
    /// process's own PID.
    outcome: &'static str,
    signaled: bool,
    forced: bool,
    /// Whether the recorded process is confirmed to be no longer running.
    stopped: bool,
}

/// Terminate the processes a service manifest records, verifying identity first.
///
/// PIDs are recycled, so a persisted PID is not by itself evidence that the
/// recorded process is still the one holding it. Each PID is therefore paired
/// with **its own** start-time token — `supervisor_start_ticks` with
/// `supervisor_pid`, `engine_start_ticks` with `engine_pid` — and routed through
/// [`rocm_core::terminate_verified`], which signals nothing when the recorded
/// identity is refuted. `rocm services stop` reads the same records through the
/// same primitive, so neither command can decide differently about one.
///
/// Descendants are reached from the *verified* root rather than expanded out of
/// a live process listing keyed on a PID nobody checked — which, given a stale
/// root, enumerated a stranger's children and signalled those too. Where the
/// kernel start-time is readable, `terminate_verified` additionally binds each
/// child to its own identity before any forced kill; see
/// [`terminate_recorded_pid`] for what Windows can and cannot do here.
///
/// The stopping process's own PID is reported but never signalled, and does not
/// count towards "the recorded service is gone": `rocmd` records itself as the
/// supervisor of the services it launches, and it is the launcher, not the
/// service.
///
/// `terminate` performs the verified termination of one recorded PID;
/// production passes [`terminate_recorded_pid`].
fn terminate_recorded_service_pids(
    record: &ManagedServiceRecord,
    terminate: impl Fn(&rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
) -> Vec<RecordedPidStop> {
    // Build the (pid, role, own-token) work list, de-duplicating on PID and
    // preferring an entry that carries a verifiable start-time. The two PIDs can
    // coincide — a launcher that is also the server — and a token must never be
    // read across from the PID it does not belong to.
    let mut entries: Vec<(u32, &'static str, Option<u64>)> = Vec::new();
    for (pid, role, ticks) in [
        (
            Some(record.supervisor_pid),
            "supervisor",
            record.supervisor_start_ticks,
        ),
        (record.engine_pid, "engine", record.engine_start_ticks),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _, _)| *seen == pid) {
            // Only the engine entry can land here, aliasing the supervisor: the
            // one PID is both, and is reported as both.
            existing.1 = "supervisor_and_engine";
            if existing.2.is_none() {
                existing.2 = ticks;
            }
        } else {
            entries.push((pid, role, ticks));
        }
    }

    entries
        .into_iter()
        .map(|(pid, role, ticks)| {
            if pid == std::process::id() {
                return RecordedPidStop {
                    pid,
                    role,
                    outcome: "self",
                    signaled: false,
                    forced: false,
                    stopped: true,
                };
            }
            let outcome = terminate(&rocm_core::ProcessIdentity::new(pid, ticks));
            RecordedPidStop {
                pid,
                role,
                outcome: outcome.as_str(),
                // TimedOut belongs with the signalled: the signal went out, the
                // exit was just never observed. `stopped` carries that truth.
                signaled: matches!(
                    outcome,
                    rocm_core::TerminationOutcome::Graceful
                        | rocm_core::TerminationOutcome::Forced
                        | rocm_core::TerminationOutcome::TimedOut
                ),
                // "The stop escalated to a forced kill", not "the forced kill
                // worked": a forced stop only reaches `TimedOut` by way of that
                // escalation, and `stopped` is what says whether the process
                // actually went away.
                forced: matches!(
                    outcome,
                    rocm_core::TerminationOutcome::Forced | rocm_core::TerminationOutcome::TimedOut
                ),
                stopped: outcome.stopped(),
            }
        })
        .collect()
}

/// Terminate one recorded process after verifying its identity.
///
/// The stop is forced: its contract is definitive, so the command must not
/// report success while an engine worker still holds the GPU. Only a process
/// whose recorded identity is confirmed is ever reached by it.
fn terminate_recorded_pid(identity: &rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome {
    // Windows needs the descendants taken separately. `rocm_core` has no way to
    // walk a process tree there — `process_tree_pids` returns just the root — and
    // the engine runs one level below the recorded PID, because `rocmd` launches
    // it through an `__engine-serve-http` process. A root-only kill would leave
    // the engine running and still holding the device.
    //
    // Identity verification does not apply on Windows. `process_start_ticks`
    // has no `/proc` to read there and returns `None`, so no recorded identity
    // can be refuted and the gate below only ever turns away a PID that is not
    // running at all. A recorded PID that has been recycled by an unrelated
    // process still passes it, and `taskkill /T /F` then takes that process and
    // its whole subtree: the stale-PID hazard is unchanged on Windows, exactly
    // as it was before this gate existed. Closing it needs `rocm_core` to read a
    // Windows process creation time.
    #[cfg(windows)]
    if matches!(
        rocm_core::identity_state(identity),
        rocm_core::IdentityState::Matches
    ) {
        // Taken while the root is still alive: `taskkill /T` resolves children
        // through the live parent, so it cannot reach them once the root is gone.
        let _ = ProcessCommand::new("taskkill")
            .args(["/PID", &identity.pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        // `terminate_verified` still owns the bounded wait and the verdict, but
        // a root the tree kill already took now reads back as `AlreadyGone` —
        // which would deny a stop this command did perform.
        let outcome = rocm_core::terminate_verified(
            identity,
            rocm_core::KillScope::Tree,
            MANAGED_STOP_GRACE,
            true,
        );
        return if matches!(outcome, rocm_core::TerminationOutcome::AlreadyGone) {
            rocm_core::TerminationOutcome::Forced
        } else {
            outcome
        };
    }
    rocm_core::terminate_verified(
        identity,
        rocm_core::KillScope::Tree,
        MANAGED_STOP_GRACE,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::load_service_record;
    use crate::test_support::{UNCONFIRMED_STOP_PID, seed_keyed_service, temp_app_paths};
    use crate::watchers::stop_requested;
    use std::fs;
    #[cfg(unix)]
    use std::process::{Command as ProcessCommand, Stdio};

    #[test]
    fn stop_managed_service_removes_endpoint_key_file() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-removes-endpoint-key");
        paths.ensure()?;
        let current_pid = std::process::id();
        let service_id = "svc-endpoint-key-stop";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "0.0.0.0",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir())?;
        fs::write(&key_path, "secret-key")?;
        assert!(key_path.exists());

        let result = stop_managed_service(&paths, service_id);
        // Observe the real filesystem state before the blanket temp-dir cleanup,
        // otherwise remove_dir_all would delete the key file and mask a missing
        // production cleanup (the regression this test guards).
        let key_removed = !key_path.exists();
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value
                .get("service")
                .and_then(|service| service.get("status"))
                .and_then(Value::as_str),
            Some("stopped")
        );
        assert!(key_removed, "endpoint key file must be removed after stop");
        Ok(())
    }

    #[test]
    fn stop_managed_service_without_endpoint_key_file_succeeds() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-no-endpoint-key");
        paths.ensure()?;
        let current_pid = std::process::id();
        let service_id = "svc-no-endpoint-key-stop";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        // Loopback service: no endpoint key file was ever written for it.
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        assert!(!key_path.exists());

        let result = stop_managed_service(&paths, service_id);
        let reloaded = crate::persistence::load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value
                .get("service")
                .and_then(|service| service.get("status"))
                .and_then(Value::as_str),
            Some("stopped")
        );
        assert_eq!(reloaded?.status, "stopped");
        assert!(!key_path.exists());
        Ok(())
    }

    /// Look up the reported outcome for `pid` in a stop result.
    #[cfg(unix)]
    fn stop_outcome_for_pid(value: &Value, pid: u32) -> Option<String> {
        value
            .get("pid_outcomes")?
            .as_array()?
            .iter()
            .find(|entry| entry.get("pid").and_then(Value::as_u64) == Some(u64::from(pid)))?
            .get("outcome")?
            .as_str()
            .map(ToOwned::to_owned)
    }

    #[cfg(unix)]
    fn pids_in(value: &Value, key: &str) -> Vec<u64> {
        value
            .get(key)
            .and_then(Value::as_array)
            .map(|pids| pids.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default()
    }

    /// A PID whose recorded start-time no longer matches the kernel's belongs to
    /// a different process: the recorded service already exited and its PID was
    /// recycled. The stop must refuse it rather than signal a stranger.
    ///
    /// Linux-only: a refutable identity needs a readable start-time, and
    /// `rocm_core::process_start_ticks` reads it from `/proc`. On Windows it is
    /// always `None`, so no identity can be refuted there and the state this
    /// asserts on cannot be constructed.
    #[cfg(target_os = "linux")]
    #[test]
    fn stop_managed_service_refuses_a_pid_whose_recorded_identity_no_longer_matches() -> Result<()>
    {
        let (root, paths) = temp_app_paths("stop-stale-pid-identity");
        paths.ensure()?;

        let mut bystander = ProcessCommand::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let bystander_pid = bystander.id();
        let live_ticks =
            rocm_core::process_start_ticks(bystander_pid).context("read start ticks")?;
        let recorded_ticks = live_ticks.wrapping_add(1);

        let recorded_identity =
            rocm_core::ProcessIdentity::new(bystander_pid, Some(recorded_ticks));
        assert_eq!(
            rocm_core::identity_state(&recorded_identity),
            rocm_core::IdentityState::Recycled,
            "precondition: the recorded identity must be refutable"
        );

        let service_id = "svc-stale-pid-identity";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11439,
            "managed",
            bystander_pid,
            None,
            None,
            None,
        );
        record.supervisor_start_ticks = Some(recorded_ticks);
        record.status = "ready".to_owned();
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        let exited = bystander.try_wait()?;
        let _ = bystander.kill();
        let _ = bystander.wait();
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert!(
            !pids_in(&value, "signaled_pids").contains(&u64::from(bystander_pid)),
            "a refuted PID must never be signalled: {value}"
        );
        assert!(
            exited.is_none(),
            "the unrelated live process must survive the stop"
        );
        assert!(
            pids_in(&value, "skipped_pids").contains(&u64::from(bystander_pid)),
            "a refuted PID must be reported as skipped: {value}"
        );
        // Refuted is not the same as absent, and the result must say which.
        assert_eq!(
            stop_outcome_for_pid(&value, bystander_pid).as_deref(),
            Some("identity_mismatch"),
            "{value}"
        );
        Ok(())
    }

    /// The blast radius of a refuted PID is not limited to the PID itself: the
    /// stop used to expand each recorded PID into its *current* descendants, so
    /// a recycled PID dragged that process's whole subtree into the kill set.
    ///
    /// Linux-only for the same reason as the test above: the refuted identity it
    /// starts from cannot exist on a platform without `/proc` start-times.
    #[cfg(target_os = "linux")]
    #[test]
    fn stop_managed_service_leaves_the_subtree_of_a_refuted_pid_running() -> Result<()> {
        use std::io::{BufRead, BufReader};

        let (root, paths) = temp_app_paths("stop-stale-pid-subtree");
        paths.ensure()?;

        // A parent with a child of its own, standing in for any unrelated
        // process tree that happens to hold the recorded PID.
        let mut parent = ProcessCommand::new("sh")
            .args(["-c", "sleep 60 & echo $!; wait"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut line = String::new();
        BufReader::new(parent.stdout.take().context("piped stdout")?).read_line(&mut line)?;
        let child_pid: u32 = line.trim().parse().context("child pid")?;
        let parent_pid = parent.id();

        let recorded_ticks = rocm_core::process_start_ticks(parent_pid)
            .context("start ticks")?
            .wrapping_add(1);

        let service_id = "svc-stale-pid-subtree";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11440,
            "managed",
            parent_pid,
            None,
            None,
            None,
        );
        record.supervisor_start_ticks = Some(recorded_ticks);
        record.status = "ready".to_owned();
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        let child_running = rocm_core::process_is_running(child_pid);
        let _ = parent.kill();
        let _ = parent.wait();
        let _ = ProcessCommand::new("kill")
            .args(["-KILL", &child_pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert!(
            !pids_in(&value, "signaled_pids").contains(&u64::from(child_pid)),
            "a descendant of a refuted PID must never be signalled: {value}"
        );
        assert!(
            child_running,
            "the unrelated process subtree must survive the stop"
        );
        Ok(())
    }

    /// A PID that is simply not running is a different situation from one that
    /// is running as somebody else, and the result must let a caller tell them
    /// apart: nothing was signalled in either case, but only the refuted one
    /// means "this PID is now owned by an unrelated process".
    #[cfg(unix)]
    #[test]
    fn stop_managed_service_reports_an_absent_pid_as_already_gone() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-absent-pid");
        paths.ensure()?;

        // Far above any attainable pid_max, so it cannot be live.
        let absent_pid = 999_999_999;
        let service_id = "svc-absent-pid";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11441,
            "managed",
            absent_pid,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert!(
            !pids_in(&value, "signaled_pids").contains(&u64::from(absent_pid)),
            "{value}"
        );
        assert_eq!(
            stop_outcome_for_pid(&value, absent_pid).as_deref(),
            Some("already_gone"),
            "{value}"
        );
        Ok(())
    }

    /// A completed stop must not leave the PIDs it just terminated in the
    /// record. They are stale the instant the processes exit, and a second stop
    /// (or any later one) would re-signal whatever the kernel has since handed
    /// those numbers to.
    #[cfg(unix)]
    #[test]
    fn stop_managed_service_clears_recorded_pids_once_the_processes_are_gone() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-clears-recorded-pids");
        paths.ensure()?;

        let mut child = ProcessCommand::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let child_pid = child.id();

        // No start-ticks: a record written before identity was captured. These
        // are exactly the records for which a stale PID cannot be refuted, so
        // clearing the PIDs is the only thing standing between a repeat stop
        // and an unrelated process.
        let service_id = "svc-clears-recorded-pids";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11442,
            "managed",
            child_pid,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        let reloaded = load_service_record(&paths, service_id);
        let _ = child.kill();
        let _ = child.wait();
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert!(
            pids_in(&value, "signaled_pids").contains(&u64::from(child_pid)),
            "precondition: the recorded process must have been stopped: {value}"
        );
        assert_eq!(value.get("stopped").and_then(Value::as_bool), Some(true));

        let reloaded = reloaded?;
        assert_eq!(
            reloaded.supervisor_pid, 0,
            "a confirmed stop must clear the supervisor PID"
        );
        assert_eq!(
            reloaded.engine_pid, None,
            "a confirmed stop must clear the engine PID"
        );
        Ok(())
    }

    /// A confirmed stop closes out the pending-stop protocol it shares with
    /// `rocm services stop`: the deferred-cleanup marker is for a stop that
    /// could *not* be confirmed, so leaving one set behind a completed stop
    /// would keep asking `rocm`'s liveness refresh to finish work that is done.
    ///
    /// `rocmd` never wrote this field at all before it could tell a completed
    /// stop from an attempted one, so a marker left by `rocm` outlived a `rocmd`
    /// stop of the same service.
    #[cfg(unix)]
    #[test]
    fn a_confirmed_stop_clears_the_pending_stop_marker() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-clears-pending-marker");
        paths.ensure()?;

        let service_id = "svc-pending-marker";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11445,
            "managed",
            // Not running, so the stop is confirmed without signalling anything.
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.stop_requested_unix_ms = Some(1);
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(value.get("stopped").and_then(Value::as_bool), Some(true));
        let reloaded = reloaded?;
        assert_eq!(reloaded.status, "stopped");
        assert_eq!(
            reloaded.stop_requested_unix_ms, None,
            "a confirmed stop must clear the deferred-cleanup marker"
        );
        Ok(())
    }

    /// The stop request is on disk for the whole of the stop, not only once it
    /// ends: a stop can take tens of seconds, and recovery must not restart the
    /// service partway through it.
    #[test]
    fn the_stop_request_is_recorded_before_any_process_is_signalled() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-request-before-signal");
        paths.ensure()?;
        let service_id = "svc-stop-request-before-signal";
        seed_keyed_service(&paths, service_id, 11452)?;

        let seen_during_stop = std::cell::Cell::new(None);
        let result = stop_managed_service_with(&paths, service_id, |_| {
            seen_during_stop.set(Some(
                load_service_record(&paths, service_id).map(|record| stop_requested(&record)),
            ));
            rocm_core::TerminationOutcome::Graceful
        });
        fs::remove_dir_all(root).ok();

        result?;
        assert!(
            seen_during_stop
                .take()
                .context("the stop never reached a termination")??,
            "the stop request must be on disk while the processes are being stopped"
        );
        Ok(())
    }

    /// Seeds a keyed service and relabels it as belonging to `engine`.
    fn seed_service_for_engine(
        paths: &AppPaths,
        service_id: &str,
        port: u16,
        engine: &str,
    ) -> Result<()> {
        seed_keyed_service(paths, service_id, port)?;
        let mut record = load_service_record(paths, service_id)?;
        record.engine = engine.to_owned();
        record.write()
    }

    /// A container-backed service (`aim`) has no process whose exit proves it
    /// stopped, so a stop must send the engine a `Stop` and report its result.
    #[test]
    fn stopping_an_aim_service_sends_the_engine_a_forced_stop() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-aim-sends-engine-stop");
        paths.ensure()?;
        let service_id = "svc-stop-aim-sends-engine-stop";
        seed_service_for_engine(&paths, service_id, 11460, "aim")?;
        let requested = std::cell::RefCell::new(Vec::new());

        let result = stop_managed_service_via(
            &paths,
            service_id,
            |_| rocm_core::TerminationOutcome::Graceful,
            |_, record| {
                requested
                    .borrow_mut()
                    .push((record.engine.clone(), record.service_id.clone()));
                Ok(StopResponse {
                    stopped: true,
                    graceful: false,
                })
            },
        );
        fs::remove_dir_all(root).ok();

        let report = result?;
        assert_eq!(
            requested.into_inner(),
            vec![("aim".to_owned(), service_id.to_owned())]
        );
        assert_eq!(report["stopped"], true);
        assert_eq!(report["engine_stop"]["attempted"], true);
        assert_eq!(report["engine_stop"]["stopped"], true);
        Ok(())
    }

    /// If the container cannot be stopped the service is still running, however
    /// cleanly its supervisor process went away.
    #[test]
    fn a_failed_engine_stop_leaves_an_aim_service_unconfirmed() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-aim-engine-stop-fails");
        paths.ensure()?;
        let service_id = "svc-stop-aim-engine-stop-fails";
        seed_service_for_engine(&paths, service_id, 11461, "aim")?;

        let result = stop_managed_service_via(
            &paths,
            service_id,
            |_| rocm_core::TerminationOutcome::Graceful,
            |_, _| anyhow::bail!("docker kill failed"),
        );
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let report = result?;
        assert_eq!(report["stopped"], false);
        assert!(
            report["engine_stop"]["error"]
                .as_str()
                .is_some_and(|error| error.contains("docker kill failed")),
            "{report}"
        );
        let reloaded = reloaded?;
        assert_ne!(reloaded.status, "stopped");
        assert!(reloaded.stop_requested_unix_ms.is_some());
        Ok(())
    }

    /// Lemonade is unloaded over HTTP by `rocm services stop`, never sent an
    /// engine `Stop`; this path must not newly start doing so. A PID-supervised
    /// engine gets the `Stop` but its PIDs stay authoritative for the verdict.
    #[test]
    fn engine_stop_is_skipped_for_lemonade_and_advisory_for_pid_engines() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-engine-stop-policy");
        paths.ensure()?;
        seed_service_for_engine(&paths, "svc-stop-policy-lemonade", 11462, "lemonade")?;
        seed_service_for_engine(&paths, "svc-stop-policy-vllm", 11463, "vllm")?;

        let lemonade = stop_managed_service_via(
            &paths,
            "svc-stop-policy-lemonade",
            |_| rocm_core::TerminationOutcome::Graceful,
            |_, _| panic!("lemonade must not be sent an engine Stop"),
        );
        let vllm = stop_managed_service_via(
            &paths,
            "svc-stop-policy-vllm",
            |_| rocm_core::TerminationOutcome::Graceful,
            |_, _| anyhow::bail!("engine unreachable"),
        );
        fs::remove_dir_all(root).ok();

        let lemonade = lemonade?;
        assert_eq!(lemonade["stopped"], true);
        assert!(lemonade["engine_stop"].is_null());
        let vllm = vllm?;
        assert_eq!(
            vllm["stopped"], true,
            "the PIDs decide for a process engine"
        );
        assert!(vllm["engine_stop"]["error"].is_string());
        Ok(())
    }

    /// The stop applies its result to the record as it is *after* the stop, not
    /// to the copy it loaded before: anything another writer persisted in the
    /// meantime survives.
    #[test]
    fn a_stop_keeps_what_another_writer_persisted_while_it_ran() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-keeps-concurrent-write");
        paths.ensure()?;
        let service_id = "svc-stop-keeps-concurrent-write";
        let key_path = seed_keyed_service(&paths, service_id, 11453)?;

        let result = stop_managed_service_with(&paths, service_id, |_| {
            let mut on_disk = load_service_record(&paths, service_id).expect("record");
            on_disk.startup_phase = Some("loading".to_owned());
            on_disk.write().expect("concurrent write");
            rocm_core::TerminationOutcome::Graceful
        });
        let key_removed = !key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert_eq!(result?.get("stopped").and_then(Value::as_bool), Some(true));
        let reloaded = reloaded?;
        assert_eq!(reloaded.status, "stopped");
        assert_eq!(
            reloaded.startup_phase.as_deref(),
            Some("loading"),
            "a write made during the stop must not be discarded"
        );
        assert!(key_removed);
        Ok(())
    }

    /// If the record names different processes once the stop is over — a
    /// restart recorded fresh PIDs while it ran — those are live processes the
    /// stop never touched. It must not claim the service stopped, clear their
    /// PIDs, or drop the key they serve with.
    #[test]
    fn a_stop_does_not_clear_processes_recorded_while_it_ran() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-keeps-restarted-pids");
        paths.ensure()?;
        let service_id = "svc-stop-keeps-restarted-pids";
        let key_path = seed_keyed_service(&paths, service_id, 11454)?;
        let restarted_pid = UNCONFIRMED_STOP_PID - 1;

        let result = stop_managed_service_with(&paths, service_id, |_| {
            let mut on_disk = load_service_record(&paths, service_id).expect("record");
            on_disk.supervisor_pid = restarted_pid;
            on_disk.supervisor_start_ticks = Some(42);
            on_disk.write().expect("concurrent restart write");
            rocm_core::TerminationOutcome::Graceful
        });
        let key_kept = key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value.get("stopped").and_then(Value::as_bool),
            Some(false),
            "the processes now recorded were never stopped: {value}"
        );
        let reloaded = reloaded?;
        assert_eq!(reloaded.status, "ready");
        assert_eq!(reloaded.supervisor_pid, restarted_pid);
        assert_eq!(reloaded.supervisor_start_ticks, Some(42));
        assert!(reloaded.stop_requested_unix_ms.is_some());
        assert!(key_kept, "the restarted service's key must be kept");
        Ok(())
    }

    /// A launcher that is also the server is recorded under both roles. The
    /// stop handles that PID once, and must report it under both.
    #[test]
    fn a_pid_recorded_as_supervisor_and_engine_is_reported_under_both_roles() -> Result<()> {
        let (root, paths) = temp_app_paths("stop-aliased-pid-role");
        paths.ensure()?;
        // Not running, so the stop confirms it without signalling anything.
        let aliased_pid = 999_999_999;
        let service_id = "svc-aliased-pid-role";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11449,
            "managed",
            aliased_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(aliased_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let result = stop_managed_service(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        let outcomes = value
            .get("pid_outcomes")
            .and_then(Value::as_array)
            .context("pid_outcomes")?;
        assert_eq!(outcomes.len(), 1, "{value}");
        assert_eq!(
            outcomes[0].get("role").and_then(Value::as_str),
            Some("supervisor_and_engine"),
            "{value}"
        );
        Ok(())
    }
}
