// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Waiting on a launched or already-live managed service: the readiness budget,
//! liveness of the tracked process, how a wait ends, and the end-to-end
//! fault-injection seam that fabricates a launch on a host with no GPU — see
//! `e2e_simulate_oom_launch` for the modes it offers and what each one leaves
//! for the surface under test to compute.

// There is deliberately no `use std::fs`, `use std::thread` or
// `use rocm_core::process_is_running` here: this file names them by full path.
// Their use sites sit under different `cfg` combinations (the Linux-only zombie
// probe, the fault-injection seam, one Linux-only test), so any single import
// would be unused in some build, and a gate on it would have to be kept in step
// with every use site. Keep the full paths. The tests module imports `fs` for
// itself because its tests use it on every platform.
use super::launch_transaction::{
    RecordUpdate, live_and_no_stop_requested, update_managed_record_on_disk,
};
use super::spawn::ManagedLaunchReport;
#[cfg(feature = "e2e-oom-fault-injection")]
use crate::device_policy_name;
use crate::endpoint_keys;
use crate::engine_runtime_select::wait_for_service_http_ready_with_progress;
use crate::serve_progress;
use crate::service_hints;
use crate::services_text::read_optional_tail_lines;
use crate::status_for_readiness;
#[cfg(feature = "e2e-oom-fault-injection")]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "e2e-oom-fault-injection")]
use rocm_core::format_http_base_url;
use rocm_core::{AppPaths, EndpointReadiness, ManagedServiceRecord};
#[cfg(feature = "e2e-oom-fault-injection")]
use rocm_engine_protocol::{DevicePolicy, ResolveModelResponse};
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

/// How long one `--wait` readiness pass polls: the same budget after which a
/// piped `rocm serve` reports `starting`.
const READINESS_WAIT: Duration = crate::serve_progress::PIPED_SERVE_STATUS_BUDGET;

/// `rocm serve --wait`: keep waiting past [`READINESS_WAIT`] (a first start
/// downloads the model, which can take far longer) until the server answers or
/// stops being live. A server that exited ends the wait with the last reading
/// instead of blocking forever.
fn wait_past_readiness_budget(
    mut probe: impl FnMut() -> EndpointReadiness,
    mut still_live: impl FnMut() -> bool,
) -> EndpointReadiness {
    loop {
        let readiness = probe();
        if readiness == EndpointReadiness::Serving || !still_live() {
            return readiness;
        }
    }
}

/// Whether `pid` is a running process that is not a zombie. The serve
/// supervisor is an unreaped child of this process, so after it dies
/// `process_is_running` alone keeps reporting it alive.
pub(super) fn service_pid_is_alive(pid: u32) -> bool {
    rocm_core::process_is_running(pid) && !process_is_zombie(pid)
}

#[cfg(target_os = "linux")]
fn process_is_zombie(pid: u32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    // `comm` may contain spaces and parentheses; the state follows the last `)`.
    stat.rfind(')')
        .and_then(|end| stat.get(end + 1..))
        .and_then(|rest| rest.split_whitespace().next())
        == Some("Z")
}

#[cfg(not(target_os = "linux"))]
const fn process_is_zombie(_pid: u32) -> bool {
    false
}

/// Whether the record still describes a server that can finish starting: its
/// status is a live one *and* a tracked process is alive. The status alone is
/// not enough, since a killed engine never writes a terminal state.
fn record_still_live(record: &ManagedServiceRecord, pid_alive: impl Fn(u32) -> bool) -> bool {
    if !crate::managed_service_status::managed_service_is_live(record) {
        return false;
    }
    let pids = crate::managed_service_status::recorded_service_pids(record);
    pids.is_empty() || pids.into_iter().any(pid_alive)
}

/// [`record_still_live`] over the manifest on disk, so a `rocm services stop`
/// from another terminal (which writes only the manifest) ends the wait too.
fn service_still_live(paths: &AppPaths, service_id: &str, pid_alive: impl Fn(u32) -> bool) -> bool {
    let Ok(mut latest) = ManagedServiceRecord::read_manifest(paths, service_id) else {
        return false;
    };
    let _ = latest.refresh_from_engine_state();
    record_still_live(&latest, pid_alive)
}

/// What an exit nobody asked for is recorded as.
///
/// `failed` is overloaded: to a reader of `rocm services list` it means "this
/// died on its own", and to `rocmd` it is the restart trigger
/// (`manifest_service_recovery_reason` in
/// apps/rocmd/src/watchers/server_recover.rs recovers `failed`, and
/// `find_recoverable_service` only skips a record carrying a stop marker).
/// Exit points differ in which of the two they can afford, so each names its
/// own policy rather than inheriting one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum UnexplainedDeath {
    /// ROCMAI-541's exit points: the launch, the attached session and `--wait`
    /// reported the death only through the record, so a crash recorded as
    /// `stopped` was indistinguishable from an operator's own stop. Recording
    /// it as the crash it was is the whole fix, and `rocmd` restarting a
    /// crashed server afterwards is supervision working.
    IsACrash,
    /// The interactive startup wait, which hands the user a loud synchronous
    /// `ServeStartupFailed` instead, so the record is not the only thing
    /// saying the launch failed. Stamping `failed` here would additionally
    /// make `rocmd` restart the server behind the error just printed, so this
    /// path keeps the clean `stopped` it chose deliberately.
    IsRecordedAsStopped,
}

/// Record how a server that died mid-start died, leaving a manifest that a
/// `services stop` already settled untouched. A `failed` state the engine
/// reported is persisted rather than overwritten, so the reason survives.
///
/// The manifest is re-read here, after the caller decided the server ended.
/// If it now describes a live process again, `rocmd` restarted the service in
/// between (`restart_managed_service` in apps/rocmd/src/supervise.rs records
/// the new pid through `record_spawned_supervisor`), and that restart is left alone rather than written back to
/// `stopped`.
///
/// Returns whether the death was **asked for**, so the caller can say so. The
/// `--wait` message used to read "the server exited before it became ready"
/// even for a `rocm services stop` issued from another shell, which is a true
/// sentence told in a way that reads as a failure.
///
/// The verdict is read off the *same* snapshot that gets stamped below, through
/// [`record_shows_stop_was_requested`] rather than a second
/// [`managed_service_stop_was_requested`] call. It used to be its own
/// independent file read, taken a line before this one: a stop landing in the
/// gap pinned `deliberate` to `false` from the stale first read while the
/// second could already show the record live, and the two together stamped
/// `failed` on a death that had in fact just been asked for. One read cannot
/// disagree with itself.
fn settle_exited_service(
    paths: &AppPaths,
    service_id: &str,
    pid_alive: &dyn Fn(u32) -> bool,
    unexplained: UnexplainedDeath,
) -> bool {
    let Ok(mut latest) = ManagedServiceRecord::read_manifest(paths, service_id) else {
        // No snapshot to judge from and nothing to stamp. The standalone read
        // is sound *here* precisely because this branch writes nothing, so
        // there is no second read for it to contradict.
        return managed_service_stop_was_requested(paths, service_id);
    };
    let deliberate = record_shows_stop_was_requested(&latest);
    let on_disk = latest.status.clone();
    let _ = latest.refresh_from_engine_state();
    if crate::managed_service_status::managed_service_is_live(&latest) {
        // A recorded pid is alive again: `rocmd` restarted the service between
        // the caller observing the death and this read, so there is no death
        // here to characterise and the restart is left exactly as it is.
        // Checked before the verdict below, which would otherwise stamp a
        // live, freshly restarted record as `failed`.
        if crate::managed_service_status::recorded_service_pids(&latest)
            .into_iter()
            .any(pid_alive)
        {
            return deliberate;
        }
        // The record still reads live even after the death was observed, so
        // nothing has recorded *why* it died: the engine's own in-process
        // backstop (`record_engine_serve_http_failure`) never got to run. A
        // SIGKILL, an OOM kill or a segfault all land here — plausible for an
        // inference server, which is why the surrounding code already carries
        // OOM guidance — and calling any of them a clean stop is the ROCMAI-541
        // defect at the one exit point the launch/attach/restart fix did not
        // cover.
        //
        // An engine that *did* stamp `failed` itself is not reached: that record
        // no longer reads live, so it takes the branch below and keeps its own
        // reason.
        if deliberate || unexplained == UnexplainedDeath::IsRecordedAsStopped {
            latest.status = "stopped".to_owned();
        } else {
            latest.status = "failed".to_owned();
            // The engine state file is what ROCMAI-541's acceptance criterion
            // names, and `latest.write()` below only covers the manifest. A
            // merge, not a whole-file write, for the same reason
            // `mark_service_failed` uses one.
            let _ = rocm_core::merge_json_state_file(
                &latest.engine_state_path,
                &serde_json::json!({ "status": "failed" }),
            );
        }
    } else if latest.status == on_disk {
        // The refresh learned nothing, so the snapshot in hand says no more
        // than what is already on disk and must NOT be written back whole:
        // anything a concurrent writer put there after this read was taken
        // would go with it. That is what this early return protects, and
        // `a_wait_death_is_a_crash_unless_a_stop_asked_for_it`'s third arm
        // pins it by planting a key the record type does not model.
        //
        // But "nothing new to write" is not the same as "nothing to correct".
        // `managed_service_status`'s liveness reconciliation demotes any
        // record whose recorded PIDs are gone to `stopped`, unconditionally
        // and with no idea why they went — a `rocm services` or `rocm dash`
        // listing running alongside this wait is enough. When that lands
        // before the read above, `on_disk` is already `stopped`, the status
        // after the refresh matches it, and this return left the manifest
        // reading `stopped` for a death nothing asked for: the caller
        // correctly reports a crash while `rocm services list` shows a clean
        // stop. That is ROCMAI-541's own acceptance criterion failing through
        // the one interleaving the stamp above cannot reach, because by then
        // the record no longer reads live.
        //
        // Corrected through the same read-modify-write merge
        // `mark_service_failed` uses, so the fix does not reintroduce the
        // clobber this branch exists to prevent: only the `status` key is
        // replaced, under the state-file lock. A record already reading
        // `failed` is left alone — the engine's own reason is on it, and
        // rewriting it would be the invisible no-op write the guard above
        // forbids.
        if unexplained == UnexplainedDeath::IsACrash && !deliberate && latest.status != "failed" {
            mark_service_failed(&latest.manifest_path, &latest.engine_state_path);
        }
        return deliberate;
    }
    // Reaching here means the refresh changed the status, and the only way it
    // changes it to `failed` is the engine writing that itself. For a stop this
    // function has already proved was asked for, that verdict is the adapters'
    // known misreading of SIGTERM as an unsuccessful exit — and persisting it
    // unchanged would return `deliberate` to the caller, which prints "was
    // stopped", while leaving `rocm services list` and `rocmd`'s recovery scan
    // reading `failed` off the same record. The live-child branch above already
    // resolves exactly this tie in favour of the stop; this one did not.
    //
    // The recorded stop is the stronger evidence: it is a fact about what was
    // asked for, written by `stop_internal_managed_service_confirmed_with`
    // before anything was signalled, whereas `failed` here is an inference from
    // an exit code. A crash carries no such marker, so this cannot mask one.
    if deliberate && latest.status == "failed" {
        latest.status = "stopped".to_owned();
    }
    // A whole-record write, not the `merge_json_state_file` the branch above
    // uses, and deliberately so: reaching here means `refresh_from_engine_state`
    // adopted fields from the engine state file (`engine_pid`,
    // `engine_start_ticks`, `served_model_id`) alongside the status verdict, and
    // merging only `status` would drop them. The clobber window the other branch
    // guards against is the *opposite* case — nothing new learned, so writing the
    // stale snapshot back could only overwrite a concurrent writer. This is the
    // codebase's established convention for `ManagedServiceRecord`
    // (`stop_internal_managed_service_confirmed_with` does the same), so the
    // window here is the pre-existing one rather than a new hazard.
    let _ = latest.write();
    deliberate
}

/// The failure a `--wait` ends with when the server died before it became
/// ready, after recording the death in the manifest. `None` while the wait did
/// not apply, the endpoint is serving, or the server is still alive.
pub(super) fn died_during_wait(
    paths: &AppPaths,
    service_id: &str,
    wait_until_ready: bool,
    readiness: EndpointReadiness,
    pid_alive: impl Fn(u32) -> bool,
) -> Option<anyhow::Error> {
    if !wait_until_ready
        || readiness == EndpointReadiness::Serving
        || service_still_live(paths, service_id, &pid_alive)
    {
        return None;
    }
    if settle_exited_service(paths, service_id, &pid_alive, UnexplainedDeath::IsACrash) {
        // A `rocm services stop` from another shell ends the wait too, and it is
        // not a launch that failed. The wait still ends non-zero, because the
        // thing the caller asked to wait for did not happen — but it says which
        // of the two it was instead of reporting an operator's own stop as the
        // server failing to come up.
        return Some(anyhow!(
            "the server for service `{service_id}` was stopped before it became ready"
        ));
    }
    let message = format!(
        "the server exited before it became ready (vLLM and AIM give up after 5 minutes by default; `ROCM_CLI_VLLM_READY_TIMEOUT_SECS` / `ROCM_CLI_AIM_READY_TIMEOUT_SECS` allow longer); see `{}` for its output",
        crate::service_hints::service_logs_hint(service_id)
    );
    // The record is already settled above — the verdict it returned is what
    // chose the message — so only the audit line and the memory guidance are
    // left. Settling a second time would re-stamp a record that a concurrent
    // writer may have moved on since.
    Some(anyhow!(report_ended_launch(
        paths,
        service_id,
        "exited during --wait",
        message
    )))
}

/// Settle a launch whose server ended before it became ready, record it in
/// the audit log (`why` says how it ended), and return `message` with the
/// memory guidance a plain `serve` would print for it.
///
/// Shared by a `--wait` that saw the server exit and the interactive wait's
/// engine-failed or process-exited stop: the record is settled by
/// [`settle_exited_service`] (`stopped`, or the engine's own `failed`, never
/// over a stop already recorded), and a vLLM launch that ran out of VRAM gets
/// the same note either way. "starting" is the status the note keys on for
/// "never became ready", whatever the settled record now says.
pub(super) fn settle_ended_launch(
    paths: &AppPaths,
    service_id: &str,
    why: &str,
    message: String,
    pid_alive: &dyn Fn(u32) -> bool,
) -> String {
    settle_exited_service(
        paths,
        service_id,
        pid_alive,
        UnexplainedDeath::IsRecordedAsStopped,
    );
    report_ended_launch(paths, service_id, why, message)
}

/// The reporting half of [`settle_ended_launch`], for a caller that settled the
/// record itself because it needed [`settle_exited_service`]'s verdict —
/// whether the death was asked for — to choose `message` in the first place.
fn report_ended_launch(
    paths: &AppPaths,
    service_id: &str,
    why: &str,
    mut message: String,
) -> String {
    crate::record_cli_audit_event(
        paths,
        "service",
        "managed_service_launch",
        "error",
        format!("managed service ended before it became ready: {why}"),
        Some(service_id),
    );
    if let Ok(record) = ManagedServiceRecord::read_manifest(paths, service_id) {
        for note in crate::serve_summary::append_oom_serve_note(
            Vec::new(),
            &record.engine,
            false,
            "starting",
            Some(&record.log_path),
        ) {
            let _ = write!(message, "\n  note: {note}");
        }
    }
    message
}

/// Record the outcome of a finished wait on the manifest as it is now, changing
/// only what the wait learned. The copy read at spawn time can be long stale:
/// `rocmd` may have recorded a restart or a new supervisor pid, and `rocm logs`
/// refreshes the record, so writing it back whole would revert those.
///
/// The verdict is written only while the record still reads as live and no
/// stop has been requested for it. A `rocm services stop` made during the wait
/// keeps its marker and, once settled, its `stopped`, from the moment it writes
/// the marker rather than only once it has settled, and the wait fails saying
/// a stop was requested: reporting a readiness verdict (and exiting 0) for a
/// server somebody is stopping would be false. A record settled without a
/// stop marker means the server exited on its own (without `--wait`,
/// `died_during_wait` does not run, so this is where that is noticed), and the
/// failure points at its log instead. A record removed during the
/// wait (`rocm services remove`) is not re-created from the stale copy; that
/// fails too, since there is no service left to report.
pub(super) fn record_wait_outcome(
    paths: &AppPaths,
    held: &mut ManagedServiceRecord,
    status: &str,
    verified: bool,
) -> Result<()> {
    let written = update_managed_record_on_disk(
        paths,
        &held.service_id,
        // Looser than the publish's and the cleanup's `launch_still_owns`, on
        // purpose: no supervisor identity is compared. A restart during the wait
        // (`rocmd` records a new supervisor) leaves the same service at the same
        // endpoint, and the verdict is what that endpoint just answered, so it
        // still holds; the write keeps the restart's fields because it changes
        // only the status and the verification time.
        // `record_wait_outcome_keeps_what_changed_on_disk_during_the_wait` pins it.
        live_and_no_stop_requested,
        |latest| {
            status.clone_into(&mut latest.status);
            if verified {
                // Latch the verification the wait just performed, so the readiness
                // checks behind `services list` and chat read it instead of
                // re-probing.
                latest.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
            }
        },
    )?;
    // The record the guarded write refused is the one the message describes:
    // it is not read a second time, so nothing can change between the decision
    // and what is reported.
    match written {
        RecordUpdate::Written(latest) => {
            *held = *latest;
            Ok(())
        }
        RecordUpdate::NotOwned(settled) => {
            *held = *settled;
            // A stop leaves its marker; anything else that settled the record
            // (the liveness refresh demoting a dead engine, or the engine's own
            // `failed`) means the server exited on its own, and the user needs
            // its log rather than to hear somebody stopped it.
            if held.stop_requested_unix_ms.is_some() {
                // While the stop is still in flight the record reads live, and a
                // new `rocm serve` for the same model would reuse the service being
                // stopped (reuse goes by status alone, in `find_live_managed_service`
                // and `existing_live_managed_service_in` alike), so the advice is the
                // stop itself, which waits for the processes and reports whether it
                // could confirm them gone. Running it on a stop that already settled
                // is harmless.
                // `the_advised_stop_frees_the_model_a_stop_in_flight_still_holds`
                // proves both halves.
                let next_step = if crate::managed_service_status::managed_service_is_live(held) {
                    format!(
                        "finish the stop with `{}`, then start it again with `rocm serve`",
                        service_hints::service_stop_hint(&held.service_id)
                    )
                } else {
                    "start it again with `rocm serve`".to_owned()
                };
                bail!(
                    "a stop was requested for managed service `{}` while it was starting (its \
                     record now reads `{}`), so no readiness is reported for it; {next_step}",
                    held.service_id,
                    held.status
                );
            }
            bail!(
                "managed service `{}` exited while it was starting (its record now reads `{}`); \
                 see `{}` for its output",
                held.service_id,
                held.status,
                service_hints::service_logs_hint(&held.service_id)
            );
        }
        RecordUpdate::Gone => bail!(
            "managed service `{}` was removed while `rocm serve` waited for it, so there is \
             no record of it left to report",
            held.service_id
        ),
    }
}

/// Test-only fault-injection switch, off (`const false`) in shipped binaries:
/// the `e2e-oom-fault-injection` feature is enabled only by the `xtask e2e`
/// mock-lane build, never by the release build. When enabled and
/// `ROCM_E2E_SIMULATE_OOM_LAUNCH=1` is set on the process, `rocm serve` fabricates
/// the on-disk state of a managed vLLM launch that spawned, wrote an
/// out-of-memory traceback to the log it owns, and never became ready — so the
/// positive OOM-guidance summary path can be verified on a GPU-less CI host,
/// where a real launch cannot run. See [`simulate_oom_managed_launch`].
///
/// The other two values select the other two modes. Every value opens the same
/// seams (the GPU pre-flight waiver and the background-helper suppression).
///
/// - `ready`: the fabricated launch is a managed vLLM launch that *did* become
///   ready (nothing to blame on memory), so a scenario can point its port at a
///   mock engine and watch what the summary does when the post-launch smoke test
///   then fails.
/// - `extension-load`: the launch died the way a vLLM compiled extension that
///   failed to load kills it, and the fabrication stops at the log. What the
///   user then sees is built by the engine's own
///   `startup_log_context`/`extension_load_failure_hint`, reached through
///   [`rocm_engine_vllm::e2e_exited_before_ready_message`] — see
///   [`simulate_extension_load_launch_failure`].
#[cfg(feature = "e2e-oom-fault-injection")]
pub(crate) fn e2e_simulate_oom_launch() -> bool {
    std::env::var_os("ROCM_E2E_SIMULATE_OOM_LAUNCH")
        .is_some_and(|value| value == "1" || value == "ready" || value == "extension-load")
}

/// Whether the armed hook is in its `ready` mode (see [`e2e_simulate_oom_launch`]).
#[cfg(feature = "e2e-oom-fault-injection")]
fn e2e_simulate_ready_launch() -> bool {
    std::env::var_os("ROCM_E2E_SIMULATE_OOM_LAUNCH").is_some_and(|value| value == "ready")
}

/// Whether the armed hook is in its `extension-load` mode (see
/// [`e2e_simulate_oom_launch`]).
#[cfg(feature = "e2e-oom-fault-injection")]
fn e2e_simulate_extension_load_failure() -> bool {
    std::env::var_os("ROCM_E2E_SIMULATE_OOM_LAUNCH").is_some_and(|value| value == "extension-load")
}

/// Test-only (see [`e2e_simulate_oom_launch`]): the startup log a vLLM launch
/// leaves behind when one of its compiled extensions failed to load, as the
/// engine wrote it on the host in ROCMAI-1058 (runtime path shortened).
///
/// Deliberately *raw*: the import warning, the engine core's `AttributeError`,
/// and then enough API-server traceback to push both of them out of the echoed
/// tail — the shape that makes the hint read a wider window than it echoes.
/// None of rocm-cli's own wording is planted here, so every sentence the
/// scenario asserts has to have been derived from these lines by the engine.
#[cfg(feature = "e2e-oom-fault-injection")]
fn extension_load_startup_log() -> String {
    let mut lines = vec![
        "(APIServer pid=629241) INFO 11-16 11:48:20 [utils.py:253] non-default args".to_owned(),
        "WARNING 11-16 11:48:27 [rocm.py:53] Failed to import from vllm._C_stable_libtorch with \
         ImportError(\"/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required \
         by /opt/rt/site-packages/vllm/_C_stable_libtorch.abi3.so)\")"
            .to_owned(),
        "(EngineCore pid=629513) INFO 11-16 11:48:40 [core.py:98] Initializing engine".to_owned(),
        "(EngineCore pid=629513) ERROR 11-16 11:48:44 [core.py:1374] AttributeError: \
         '_OpNamespace' '_C' object has no attribute 'get_cuda_view_from_cpu_tensor'"
            .to_owned(),
    ];
    for frame in 0..(2 * rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES) {
        lines.push(format!(
            "(APIServer pid=629241)   File \"/opt/rt/site-packages/vllm/frame_{frame}.py\", line \
             {frame}, in run"
        ));
    }
    lines.push(
        "(APIServer pid=629241) RuntimeError: Engine core initialization failed. See root cause \
         above. Failed core proc(s): {}"
            .to_owned(),
    );
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

/// Test-only (see [`e2e_simulate_oom_launch`]): plant that log under the
/// service record and fail the launch with the engine's own message for it.
///
/// The two fabricated facts are the ones a GPU-less lane cannot produce: that
/// the vLLM process died, and what it had written when it did. Everything the
/// user then reads — the echoed tail, which line is named as the detected
/// failure, and the `rocm diagnose --symptom '…'` command — is computed by the
/// engine's production code from that log, which is what makes the scenario
/// able to fail when that code is wrong.
#[cfg(feature = "e2e-oom-fault-injection")]
fn simulate_extension_load_launch_failure(record: &mut ManagedServiceRecord) -> anyhow::Error {
    record.status = "starting".to_owned();
    if let Err(error) = record.write() {
        return error;
    }
    if let Err(error) = std::fs::write(&record.log_path, extension_load_startup_log()) {
        return anyhow!("failed to write {}: {error}", record.log_path.display());
    }
    anyhow!(
        "{}",
        rocm_engine_vllm::e2e_exited_before_ready_message("exit status: 1", &record.log_path)
    )
}

/// Release builds carry no fault-injection switch: the seam compiles out
/// entirely, so `rocm serve` behaves identically to a build without the feature.
#[cfg(not(feature = "e2e-oom-fault-injection"))]
pub(crate) const fn e2e_simulate_oom_launch() -> bool {
    false
}

/// Test-only switch for the mock lane, compiled in only with
/// `e2e-oom-fault-injection` and read only on Linux: a managed launch spawns a
/// stand-in engine that writes one log line and exits during startup, in place
/// of the real `rocm` serve child, so the interactive startup wait's
/// process-exited path runs end to end on a host with no GPU. Everything else
/// about the launch (record, log, wait, settling) is the production path.
#[cfg(all(feature = "e2e-oom-fault-injection", target_os = "linux"))]
pub(crate) fn e2e_simulate_startup_exit() -> bool {
    std::env::var_os(SIMULATE_STARTUP_EXIT_ENV).is_some_and(|value| value == "1")
}

/// Release builds, and hosts other than Linux, carry no such switch.
#[cfg(not(all(feature = "e2e-oom-fault-injection", target_os = "linux")))]
pub(crate) const fn e2e_simulate_startup_exit() -> bool {
    false
}

/// The variable [`e2e_simulate_startup_exit`] reads; the e2e step spells the
/// same name, since the shipped crate must not depend on the harness.
#[cfg(all(feature = "e2e-oom-fault-injection", target_os = "linux"))]
const SIMULATE_STARTUP_EXIT_ENV: &str = "ROCM_E2E_SIMULATE_STARTUP_EXIT";

/// What the stand-in engine runs: one meaningful log line, then an exit after
/// the spawn's immediate-exit check, so the launch is published and the
/// startup wait (not the spawn) is what sees the process end.
#[cfg(all(feature = "e2e-oom-fault-injection", target_os = "linux"))]
pub(super) const SIMULATED_STARTUP_EXIT_SCRIPT: &str = "echo 'e2e stand-in engine: loading model weights'; sleep 2; \
     echo 'e2e stand-in engine: exiting during startup'; exit 1";

/// Test-only (see [`e2e_simulate_oom_launch`]): fabricate the on-disk state of a
/// managed vLLM launch, then return the launch report the summary path would see
/// for it. `already_running` is false either way (this invocation launched it)
/// and no process is spawned. Compiled out of shipped binaries.
///
/// - Default mode (`ROCM_E2E_SIMULATE_OOM_LAUNCH=1`): the launch OOM'd and never
///   became ready — `status: "starting"` and a `log_path` whose contents carry a
///   real allocator OOM signature.
///   [`crate::serve_summary::append_oom_serve_note`] then renders the memory-knob
///   guidance.
/// - `ready` mode: the launch became ready — `status: "ready"`, the record marked
///   inference-verified, and `log_path: None` (nothing failed to start), so the
///   caller's smoke test runs against whatever answers on `host:port`.
/// - `extension-load` mode: the launch died on a vLLM compiled extension that
///   failed to load, and this returns an `Err` rather than a report, because
///   that is what the real readiness wait does. See
///   [`simulate_extension_load_launch_failure`].
#[cfg(feature = "e2e-oom-fault-injection")]
#[allow(clippy::too_many_arguments)]
pub(super) fn simulate_oom_managed_launch(
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
) -> Result<ManagedLaunchReport> {
    paths.ensure()?;
    std::fs::create_dir_all(paths.services_dir())?;
    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine,
        requested_model,
        resolve.canonical_model_id.clone(),
        host,
        port,
        "managed",
        std::process::id(),
        runtime_id.map(str::to_owned),
        env_id.map(str::to_owned),
        Some(device_policy_name(device_policy).to_owned()),
    );
    record.gpu_indices = gpu_indices.to_vec();
    if e2e_simulate_ready_launch() {
        // The `ready` mode: a launch that came up. The record is what the local
        // provider needs to send the smoke test to `host:port`; no log is written
        // because nothing here failed to start.
        record.status = "ready".to_owned();
        // Load-bearing: without it the smoke test's `/v1/models` listing probe
        // fails and the reason becomes "no ready managed service".
        record.inference_verified_at_unix_ms = Some(1);
        record.write()?;
        return Ok(ManagedLaunchReport {
            service_id: service_id.to_owned(),
            endpoint_url: format!("{}/v1", format_http_base_url(host, port)),
            status: "ready".to_owned(),
            already_running: false,
            child_pid: None,
            log_path: None,
            manifest_path: Some(record.manifest_path),
        });
    }
    if e2e_simulate_extension_load_failure() {
        // The `extension-load` mode: a launch that died the way a compiled
        // extension that failed to load kills it. Unlike the other two modes
        // this answers with an error, because the surface under test *is* the
        // engine's startup-failure message.
        return Err(simulate_extension_load_launch_failure(&mut record));
    }
    record.status = "starting".to_owned();
    record.write()?;
    // A real allocator OOM signature in the log this invocation owns, so the
    // summary renders memory guidance for a launch that actually OOM'd.
    //
    // The engine's own error block is reproduced below it, because that is what
    // a real managed launch leaves on disk, and leaving it out hid a bug this
    // scenario should have caught: the supervisor and the engine share one log,
    // so when vLLM OOMs the engine bails with its startup-log context — which
    // embeds rocm-cli's own OOM hint — and `exit_code_for` writes that whole
    // error back into the same file, *after* the real line. A fixture carrying
    // only the raw OOM line can never see the newest-first scan land on
    // rocm-cli's own sentence instead of the user's error.
    let real_line = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
    std::fs::write(
        &record.log_path,
        format!(
            "{real_line}\n\
             Error: vLLM server process exited before becoming ready (status: exit status: 1)\n\
             \n\
             Startup log, last {} lines as the terminal drew them:\n\
             {real_line}\n\
             \n\
             {}{real_line}). {}\n\
             For conditional remediation, run `{} 'vllm: {real_line}'`.\n",
            rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES,
            rocm_core::VLLM_OOM_DETECTED_PREFIX,
            rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT,
            rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
        ),
    )
    .with_context(|| format!("failed to write {}", record.log_path.display()))?;
    Ok(ManagedLaunchReport {
        service_id: service_id.to_owned(),
        endpoint_url: format!("{}/v1", format_http_base_url(host, port)),
        status: "starting".to_owned(),
        already_running: false,
        child_pid: None,
        log_path: Some(record.log_path),
        manifest_path: Some(record.manifest_path),
    })
}

/// Probe `record`'s endpoint until it answers (`--wait`), for as long as the
/// server stays alive. Each pass is one [`READINESS_WAIT`]-long readiness wait.
pub(super) fn wait_for_record_ready(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    endpoint_api_key: Option<&str>,
    on_wait_tick: &mut dyn FnMut(Duration),
) -> EndpointReadiness {
    let probe = |on_tick: &mut dyn FnMut(Duration)| {
        wait_for_service_http_ready_with_progress(
            &record.engine,
            &record.host,
            record.port,
            &|| {
                let mut latest = record.clone();
                let _ = latest.refresh_from_engine_state();
                latest.served_model_name().to_owned()
            },
            endpoint_api_key,
            READINESS_WAIT,
            on_tick,
        )
    };
    wait_past_readiness_budget(
        || probe(&mut *on_wait_tick),
        || service_still_live(paths, &record.service_id, service_pid_is_alive),
    )
}

/// One sample of a freshly launched service for the startup wait: the model
/// name its endpoint answers to and the engine/process signals.
///
/// Read from the manifest on disk, like [`service_still_live`], so a
/// `rocm services stop` from another terminal is seen. Liveness is
/// [`service_pid_is_alive`]: the supervisor is an unreaped child of this
/// process, so a dead one still passes `process_is_running`.
fn observe_launch(
    paths: &AppPaths,
    held: &ManagedServiceRecord,
    endpoint_api_key: Option<&str>,
    pid_alive: impl Fn(u32) -> bool,
) -> (String, serve_progress::ServeStartupSignals) {
    let mut latest = ManagedServiceRecord::read_manifest(paths, &held.service_id)
        .unwrap_or_else(|_| held.clone());
    let _ = latest.refresh_from_engine_state();
    let pids = crate::managed_service_status::recorded_service_pids(&latest);
    let signals = serve_progress::ServeStartupSignals {
        engine_failed: latest.status == "failed",
        process_alive: pids.is_empty() || pids.into_iter().any(pid_alive),
        log_line: serve_progress::latest_meaningful_log_line(&latest.log_path, endpoint_api_key),
    };
    (latest.served_model_name().to_owned(), signals)
}

/// Follow a fresh launch (without `--wait`) until it serves, the policy's
/// budget runs out, or the wait stops. See [`serve_progress`].
pub(super) fn wait_for_launch_startup(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    requested_model: &str,
    endpoint_api_key: Option<&str>,
    policy: &serve_progress::ServeStartupPolicy,
    on_view: &mut dyn FnMut(&serve_progress::ServeStartupView),
) -> Result<EndpointReadiness, serve_progress::ServeStartupStop> {
    serve_progress::wait_for_serve_startup(
        &record.engine,
        requested_model,
        &record.host,
        record.port,
        endpoint_api_key,
        policy,
        &mut || observe_launch(paths, record, endpoint_api_key, service_pid_is_alive),
        on_view,
    )
}

/// An equivalent service is already live. Without `--wait`, or once it is
/// serving, report it as is; otherwise wait for it like a fresh launch would.
pub(super) fn wait_on_existing_service(
    paths: &AppPaths,
    mut report: ManagedLaunchReport,
    wait_until_ready: bool,
    on_wait_begin: &mut dyn FnMut(&str, u16),
    on_wait_tick: &mut dyn FnMut(Duration),
) -> Result<ManagedLaunchReport> {
    if !wait_until_ready || report.status == "ready" {
        return Ok(report);
    }
    let Ok(mut record) = ManagedServiceRecord::read_manifest(paths, &report.service_id) else {
        return Ok(report);
    };
    // The id of the server being waited on, which is not the caller's freshly
    // generated one when it attached to an existing server.
    on_wait_begin(&report.service_id, record.port);
    // The key the caller generated belongs to a launch that did not happen; the
    // existing service answers to its own.
    let stored_key = endpoint_keys::endpoint_api_key(paths, &report.service_id);
    let readiness = wait_for_record_ready(paths, &record, stored_key.as_deref(), on_wait_tick);
    if let Some(error) = died_during_wait(
        paths,
        &report.service_id,
        true,
        readiness,
        service_pid_is_alive,
    ) {
        return Err(error);
    }
    report.status = status_for_readiness(readiness).to_owned();
    record_wait_outcome(
        paths,
        &mut record,
        &report.status,
        readiness == EndpointReadiness::Serving,
    )?;
    Ok(report)
}

/// Retire a crashed service: mark the manifest and the engine's own state file
/// `failed`. The engine process is gone and cannot stamp its own file, so this
/// does it on the engine's behalf.
///
/// Both writes are merges, never a whole-record `ManagedServiceRecord::write()`,
/// because every caller holds a copy of the record captured before the crash:
/// writing that back wholesale would erase a `stop_requested_unix_ms` a
/// concurrent stop may have written in the same window. Stamping the manifest
/// alone is not enough either — `refresh_from_engine_state` reads the engine
/// state file back into a newer manifest, so an unstamped state file would undo
/// the manifest write on the next refresh.
///
/// Best-effort by design: a crash report must not be lost because the
/// bookkeeping behind it could not be written.
pub(crate) fn mark_service_failed(manifest_path: &Path, engine_state_path: &Path) {
    let _ =
        rocm_core::merge_json_state_file(manifest_path, &serde_json::json!({ "status": "failed" }));
    let _ = rocm_core::merge_json_state_file(
        engine_state_path,
        &serde_json::json!({ "status": "failed" }),
    );
}

/// Render an unexpected managed-engine exit for the user, when the engine was
/// already running and then disappeared on its own — unlike
/// [`managed_engine_startup_failure_detail`], there is no [`ExitStatus`] to
/// quote here. Callers reach this only after [`managed_service_stop_was_requested`]
/// has already ruled out an intentional/external stop, so by the time this
/// renders, the exit really is unexplained — i.e. a crash. The service log is
/// the only record of why, since the child is detached and writes nowhere else
/// — platform-generic since ROCMAI-541 attaches it on Windows too (see
/// `spawn_detached_with_log_watching_startup`).
pub(crate) fn managed_engine_exit_failure_detail(service_id: &str, log_path: &Path) -> String {
    let recent_lines = read_optional_tail_lines(log_path, 80, "service log");
    if recent_lines.is_empty() {
        format!(
            "managed engine for service `{service_id}` exited unexpectedly; inspect {}",
            log_path.display()
        )
    } else {
        format!(
            "managed engine for service `{service_id}` exited unexpectedly; inspect {} \
             (or `rocm services logs {service_id}`)\n\nrecent service log output:\n{}",
            log_path.display(),
            recent_lines.join("\n")
        )
    }
}

/// Whether the on-disk manifest shows this exit was something other than a
/// crash: an operator-requested stop either still in flight
/// (`stop_requested_unix_ms`) or already confirmed
/// (`settled_stop_unix_ms`).
///
/// Two markers, and no status term. The record's `status` is deliberately NOT
/// evidence here - any concurrent reader can write `stopped` onto a record
/// naming dead pids, which is what relabelled a crash as a clean stop and
/// reproduced ROCMAI-541 inside its own fix. See the long account below.
///
/// `AttachOutcome::Stop` only ever fires from *this* attached session's own
/// keystroke reader (Ctrl-C) — it says nothing about a stop triggered any
/// other way. `rocm services stop <id>` run from another shell and `rocm
/// dash`'s stop action both kill the pid a watching session (this attach
/// loop, or a background launch/restart's readiness wait) cares about by
/// signalling it directly, after writing `stop_requested_unix_ms` first —
/// which is what lands here, in `ServerExited`, rather than `Stop`, and is
/// otherwise indistinguishable from a crash (ROCMAI-541). `rocmd`'s own
/// restart-recovery watcher (`restart_managed_service` in
/// `apps/rocmd/src/supervise.rs`) is a different case: it only ever fires once
/// the manifest or the endpoint already shows the service dead or
/// unreachable on its own, and reacts by clearing `stop_requested_unix_ms`
/// and re-exec'ing a fresh supervisor — it does not itself signal the old
/// engine, so that path is already a genuine crash by the time this check
/// runs, and correctly still reads as one (no marker was ever set for it).
///
/// Read fresh from disk — never a `record` some caller captured at spawn
/// time, which cannot reflect a stop another process requested afterward. A
/// manifest that cannot be read at all falls through to "not a requested
/// stop": that is itself anomalous, and silently treating it as a clean stop
/// would risk masking a real crash.
///
/// # Why neither term is the record's `status`
///
/// This used to read `status == "stopped"`, and that was never proof of an
/// intentional stop. `managed_service_status::refresh_managed_service_runtime_liveness`
/// reconciles *any* record whose tracked pids are all dead to `"stopped"`,
/// saying so in its own comment ("dead pids alone do not distinguish a stopped
/// service from a crashed one"), and writes no marker doing it. So an ordinary
/// concurrent `rocm services` / `rocm dash` / `rocmd` read landing between a
/// crash and this check relabelled the crash as a clean stop, and it was
/// reported as one — exit 0, "server process exited", the original ROCMAI-541
/// symptom reappearing inside its own fix.
///
/// The term could not simply be dropped, because a *confirmed* stop used to
/// clear its own marker and the status was then the only trace it left:
/// removing it turned `serve-scripted-external-stop-while-attached` (serve-65)
/// red, which also meant that scenario was passing on the unsound term and so
/// could not tell a stop from a crash at all.
///
/// Both halves are fixed by making the stop's own evidence durable instead:
/// `stop_internal_managed_service_confirmed_with` now *moves* its marker into
/// [`ManagedServiceRecord::settled_stop_unix_ms`] rather than dropping it. That
/// field already existed for the sibling case (a pending marker consumed by the
/// refresh), is written only by a stop, and is cleared by
/// [`ManagedServiceRecord::reset_for_restart`] — so a stop settled before a
/// relaunch cannot mask a crash after it. Nothing a crash can do sets either
/// field.
///
/// [`StopWatch`] still narrows the live window for a stop that is requested and
/// confirmed entirely between two polls.
pub(crate) fn managed_service_stop_was_requested(paths: &AppPaths, service_id: &str) -> bool {
    crate::services_text::read_managed_service_manifest(paths, service_id)
        .is_ok_and(|(record, _)| record_shows_stop_was_requested(&record))
}

/// The predicate [`managed_service_stop_was_requested`] is made of, over a
/// record a caller already holds — see that function for why these two fields
/// and not the status are the evidence.
///
/// Split out, rather than spelled a second time at the one call site that has
/// its snapshot in hand ([`settle_exited_service`]), so the two cannot drift:
/// the whole point of `settled_stop_unix_ms` is that it answers where the
/// short-lived `stop_requested_unix_ms` has already been consumed, and a call
/// site that re-tested only the marker would silently lose exactly that half
/// while still reading as if it asked the shared question.
pub(crate) const fn record_shows_stop_was_requested(record: &ManagedServiceRecord) -> bool {
    record.stop_requested_unix_ms.is_some() || record.settled_stop_unix_ms.is_some()
}

/// The live half of [`managed_service_stop_was_requested`]: the in-flight
/// marker only, without its confirmed-stop term (`settled_stop_unix_ms`).
///
/// For the one caller whose own precondition is a stop this process just
/// performed: a restart confirms a stop before it respawns, and that
/// confirmation *settles* the marker into `settled_stop_unix_ms`.
/// `record_restarted_child` does not publish `running` until the new child
/// clears its early-exit window, so inside that window the settled field is
/// still set from *this restart's own* stop — and the full predicate would
/// therefore fire unconditionally, reporting every early crash as a deliberate
/// stop, dropping the exit status and log tail and leaving the engine state
/// file unstamped.
///
/// The marker alone is sound evidence here precisely because that preceding
/// stop cleared it (`stop_internal_managed_service_confirmed_with`): anything
/// still set has been written since this restart began, so it belongs to a
/// *new* stop racing the restart rather than to the one the restart performed.
pub(crate) fn managed_service_stop_marker_present(paths: &AppPaths, service_id: &str) -> bool {
    crate::services_text::read_managed_service_manifest(paths, service_id)
        .is_ok_and(|(record, _)| record.stop_requested_unix_ms.is_some())
}

/// Latches an operator-requested stop seen at any point while a managed child
/// is being watched.
///
/// `stop_requested_unix_ms` is a stop's *live* evidence (see
/// [`managed_service_stop_was_requested`] for why the record's status is no
/// evidence at all), but it is deliberately short-lived:
/// `stop_internal_managed_service_confirmed_with` writes it before it signals
/// anything and takes it away again once the stop is confirmed. A watcher that
/// only reads it *after* noticing the child is gone can therefore arrive after
/// the marker is gone, and report a deliberate stop as a crash.
///
/// Sampling on every liveness poll catches any stop that was still in flight
/// while the child was being watched — in particular an *unconfirmed* stop,
/// which re-stamps the marker and deliberately leaves the status at
/// `ready`/`running`, since that stop did not happen and must not be claimed.
///
/// It is a narrowing, not a closure: a stop that is requested, signals, and
/// confirms entirely inside one poll interval takes its marker away before the
/// next sample, so the latch misses it. What catches that case is not this
/// latch and not the record's status, but the fact that a confirmed stop
/// *moves* its marker into [`ManagedServiceRecord::settled_stop_unix_ms`]
/// rather than dropping it — so the one-shot
/// [`managed_service_stop_was_requested`] at the end of a wait still finds the
/// evidence through its second term. (An earlier version of this comment
/// credited a `status == "stopped"` term in that predicate. There is no such
/// term; removing it as unsound is what this change is for, and making the
/// stop's own evidence durable is what replaced it.)
///
/// The residual gap is now narrow and named: `reset_for_restart` clears both
/// fields, on purpose, so a stop settled before a relaunch cannot mask a crash
/// after it — which also means a stop whose record is reset for relaunch
/// between the stop and the end of the wait leaves nothing for either check to
/// find. A crash never writes either field, so neither this latch nor that
/// predicate can produce a false "stopped" of its own.
#[derive(Default)]
pub(crate) struct StopWatch {
    seen: bool,
}

impl StopWatch {
    /// Check the marker once, keeping any previous sighting. Cheap enough to
    /// call on every poll: one small JSON read, and it stops reading entirely
    /// once latched.
    pub(crate) fn sample(&mut self, paths: &AppPaths, service_id: &str) {
        if !self.seen {
            // The MARKER alone, never `managed_service_stop_was_requested`.
            // Historically that was because the predicate also accepted
            // `status == "stopped"`, which liveness reconciliation (`rocmd`,
            // `dash`, `rocm services`) writes for a *crashed* server, so
            // latching on it turned a crash reconciled mid-wait into a clean
            // stop and exit 0 — the original bug. That term is gone, and the
            // narrower reason it leaves behind still holds: this samples
            // repeatedly and latches forever, so it may only fire on evidence
            // written *during* the watch. `stop_requested_unix_ms` is exactly
            // that — short-lived, and taken away again the moment a stop
            // confirms. `settled_stop_unix_ms` is durable on purpose and can
            // outlive the stop that wrote it, so latching on it could report a
            // death as asked-for on the strength of an earlier stop.
            //
            // The one-shot check at the end of a wait does take the durable
            // field, because it has no second chance to see a marker already
            // consumed; it accepts the mirror-image exposure in exchange, held
            // down by `reset_for_restart` clearing both fields on every
            // relaunch.
            self.seen = managed_service_stop_marker_present(paths, service_id);
        }
    }

    /// Whether a stop was requested at any point during the watch.
    pub(crate) const fn stop_was_requested(&self) -> bool {
        self.seen
    }
}

/// Whether the service's files on disk still name the supervisor the caller
/// spawned, read fresh rather than from a spawn-time snapshot.
///
/// `supervisor_pid`/`supervisor_start_ticks` are the right pair to compare on:
/// `refresh_from_engine_state` never touches them, and a legitimate restart
/// does rewrite them (`record_supervisor_identity` at the top of the spawn
/// path), so they distinguish "these files still describe the child I am
/// watching" from "a restart has already replaced it". The pid alone would not
/// — pids are recycled, which is what the start-time token rules out.
pub(super) fn still_watching_this_supervisor(
    paths: &AppPaths,
    service_id: &str,
    supervisor_pid: u32,
    supervisor_start_ticks: Option<u64>,
) -> bool {
    crate::services_text::read_managed_service_manifest(paths, service_id).is_ok_and(
        |(current, _)| {
            current.supervisor_pid == supervisor_pid
                && current.supervisor_start_ticks == supervisor_start_ticks
        },
    )
}

/// Decide a managed launch's final status from the HTTP readiness wait and the
/// child's liveness, and retire a child that died before reaching `Serving` as
/// a failure rather than quietly reporting whatever lower status the readiness
/// wait happened to see last.
///
/// A readiness wait that only times out (never reaches [`EndpointReadiness::Serving`])
/// is ambiguous on its own: the process could still be loading a large model, or
/// it could already be dead (ROCMAI-541 — this was previously indistinguishable,
/// so `rocm serve` reported `starting` and exited 0 either way). `child_is_alive`
/// resolves it and is injected so the dead-child branch is unit-testable without
/// spawning a real process — mirroring `autostart_spawn_in_flight`'s `pid_alive`
/// parameter elsewhere in this file. The real caller backs it with
/// [`rocm_core::ManagedChildLiveness`] rather than a bare `process_is_running`
/// pid probe: on Unix a signal probe alone succeeds for a zombie child this
/// process has not reaped yet (see that type's doc comment), which previously
/// left this exact check inert on Linux despite the readiness/liveness split
/// existing. `Serving` always succeeds regardless of the live check: inference
/// having already answered is a stronger readiness signal than a liveness check
/// taken afterward, which could itself lose a race against the same process
/// exiting right after serving the probe.
///
/// On the failure branch this also marks the engine's own state file `failed`
/// (best-effort): the engine process is gone and cannot do it for itself, and
/// without this the state file can be left reading `starting` indefinitely (see
/// `run_builtin_engine_serve_http`'s own, primary fix for the common case where
/// the engine process is alive long enough to run that logic itself).
///
/// A dead-and-not-`Serving` child is still ambiguous one level further: it
/// could be a crash, or it could be a `rocm services stop`/`rocm dash` stop
/// that landed during this readiness wait — see
/// [`managed_service_stop_was_requested`]'s doc comment (ROCMAI-541). That
/// check is read fresh from disk, never from `record`, which was captured at
/// spawn time and cannot reflect a stop requested since. When it fires,
/// `record` is refreshed from disk too — rather than left as that stale
/// spawn-time snapshot — so that a caller's own follow-up write of `record`
/// carries the stop marker forward instead of clobbering it. Returns `false`
/// in that case so the caller knows to skip any further bookkeeping that
/// would otherwise write back over it; `true` otherwise (the normal path,
/// unchanged).
///
/// Marks the manifest itself via a merge (`merge_json_state_file`), not a
/// whole-file `record.write()`: `record` is this function's own stale,
/// spawn-time snapshot, and writing it back wholesale would erase a
/// `stop_requested_unix_ms` a concurrent stop just wrote to disk — the same
/// hazard the merge above (and the attach path's equivalent write) already
/// guards against.
pub(super) fn finalize_managed_launch_readiness(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    readiness: EndpointReadiness,
    mut child_is_alive: impl FnMut() -> bool,
    stop_seen_while_waiting: bool,
) -> Result<bool> {
    if readiness != EndpointReadiness::Serving && !child_is_alive() {
        // The latch first, then a fresh read: a stop that began and finished
        // inside the readiness wait has already cleared its own marker, so the
        // point check below cannot see it (see `StopWatch`).
        if stop_seen_while_waiting || managed_service_stop_was_requested(paths, &record.service_id)
        {
            if let Ok((on_disk, _)) =
                crate::services_text::read_managed_service_manifest(paths, &record.service_id)
            {
                *record = on_disk;
            }
            return Ok(false);
        }
        // Only stamp the service's own files while they still describe the
        // child this invocation spawned. `child_is_alive` is bound to this
        // `rocm serve`'s own pid, but the files are named by `service_id`, so a
        // restart that lands inside this readiness wait leaves them describing
        // a different, possibly healthy child. `reset_for_restart` clears the
        // stop evidence the branch above looks for, so without this check the
        // two true facts "no stop evidence" and "my child is dead" combine into
        // `failed` on somebody else's record -- and stick, because
        // `refresh_managed_service_runtime_liveness_with` short-circuits on a
        // `failed` status and never re-probes to correct it.
        //
        // The attached path has guarded this since ROCMAI-541; this is the same
        // guard for the background launch, which is the more common one.
        if still_watching_this_supervisor(
            paths,
            &record.service_id,
            record.supervisor_pid,
            record.supervisor_start_ticks,
        ) {
            record.status = "failed".to_owned();
            mark_service_failed(&record.manifest_path, &record.engine_state_path);
        }
        bail!(
            "{}",
            managed_engine_exit_failure_detail(&record.service_id, &record.log_path)
        );
    }
    record.status = status_for_readiness(readiness).to_owned();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_launch::test_support::reusable_record;
    use crate::tests::test_paths;
    use std::fs;

    /// ROCMAI-541: the service log tail must reach the user, and a missing log
    /// must still name the service and where to look rather than being silent.
    #[test]
    fn exit_failure_detail_quotes_the_service_log_tail_when_present() {
        let (root, paths) = test_paths("exit-failure-detail-log-present");
        paths.ensure().expect("ensure paths");
        let log_path = paths.service_log_path("svc-dead-1");
        fs::create_dir_all(log_path.parent().expect("log parent")).expect("mkdir log dir");
        fs::write(
            &log_path,
            "Model pulled successfully\nFATAL: HOME is unset\n",
        )
        .expect("write log");

        let detail = managed_engine_exit_failure_detail("svc-dead-1", &log_path);

        assert!(detail.contains("svc-dead-1"), "{detail}");
        assert!(detail.contains("exited unexpectedly"), "{detail}");
        assert!(detail.contains("FATAL: HOME is unset"), "{detail}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn exit_failure_detail_names_the_service_even_without_a_log() {
        let (root, paths) = test_paths("exit-failure-detail-log-absent");
        let log_path = paths.service_log_path("svc-dead-2");
        // Deliberately never created.
        let detail = managed_engine_exit_failure_detail("svc-dead-2", &log_path);
        assert!(detail.contains("svc-dead-2"), "{detail}");
        let _ = fs::remove_dir_all(&root);
    }

    /// ROCMAI-541: `AttachOutcome::Stop` only ever fires from this
    /// session's own keystroke reader, never from `rocm services stop`
    /// another shell issued, `rocm dash`, or `rocmd`'s restart-recovery — all
    /// of which kill the watched pid directly and so land in `ServerExited`.
    /// The manifest's stop marker is what tells the two apart; this covers
    /// every state that marker can be in before `managed_service_stop_was_requested`
    /// is consulted.
    #[test]
    fn stop_was_requested_true_for_in_flight_or_confirmed_stop() {
        let (root, paths) = test_paths("stop-was-requested-true");
        paths.ensure().expect("ensure paths");
        let make_record = |service_id: &str| {
            ManagedServiceRecord::new(
                &paths,
                service_id,
                "lemonade",
                "qwen",
                "qwen-canonical",
                "127.0.0.1",
                11540,
                "managed",
                std::process::id(),
                None,
                None,
                None,
            )
        };

        // An unconfirmed stop: the marker is set but the status is left
        // `ready`/`running` until termination is confirmed (see
        // `stop_internal_managed_service_confirmed_with`).
        let mut in_flight = make_record("svc-stop-in-flight");
        in_flight.status = "running".to_owned();
        in_flight.stop_requested_unix_ms = Some(1);
        in_flight.write().expect("write in-flight record");
        assert!(managed_service_stop_was_requested(
            &paths,
            "svc-stop-in-flight"
        ));

        // A confirmed stop moves its marker into `settled_stop_unix_ms` rather
        // than dropping it. This used to assert that `status: "stopped"` alone
        // was enough, which is what let a reconciled crash pass as a stop — see
        // `a_reconciled_crash_is_not_evidence_of_a_stop_but_a_settled_one_is`.
        let mut confirmed = make_record("svc-stop-confirmed");
        confirmed.status = "stopped".to_owned();
        confirmed.settled_stop_unix_ms = Some(1);
        confirmed.write().expect("write confirmed-stop record");
        assert!(managed_service_stop_was_requested(
            &paths,
            "svc-stop-confirmed"
        ));

        let _ = fs::remove_dir_all(&root);
    }

    /// The complement: a plain running service with no stop marker and no
    /// `stopped` status must not be mistaken for an intentional stop — an
    /// unexplained exit from this state is exactly ROCMAI-541's crash case.
    #[test]
    fn stop_was_requested_false_for_a_live_service_or_a_missing_manifest() {
        let (root, paths) = test_paths("stop-was-requested-false");
        paths.ensure().expect("ensure paths");
        let mut running = ManagedServiceRecord::new(
            &paths,
            "svc-stop-not-requested",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11541,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        running.status = "running".to_owned();
        running.write().expect("write running record");
        assert!(!managed_service_stop_was_requested(
            &paths,
            "svc-stop-not-requested"
        ));

        // A manifest that cannot be read at all (never written here) must
        // fail closed too — see this function's doc comment.
        assert!(!managed_service_stop_was_requested(
            &paths,
            "svc-never-existed"
        ));

        let _ = fs::remove_dir_all(&root);
    }

    /// ROCMAI-541 acceptance criterion 2: a managed engine that died before
    /// ever answering HTTP must fail `rocm serve` with both a non-zero exit
    /// code and a message naming it, not just the unreadable "starting"
    /// status the readiness timeout alone used to leave behind. `pid_is_running`
    /// is injected (`|_| false`) rather than spawning and killing a real
    /// process, the same seam `autostart_spawn_in_flight` above uses for an
    /// identical reason: this repo cannot spawn and reap a real child inside a
    /// `cargo test` run portably across Linux and Windows.
    #[test]
    fn finalize_managed_launch_readiness_fails_closed_when_the_child_died() -> Result<()> {
        let (root, paths) = test_paths("finalize-readiness-dead-child");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-dead-3",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11530,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        fs::create_dir_all(record.log_path.parent().expect("log parent"))?;
        fs::write(&record.log_path, "Model pulled successfully\n")?;
        record.status = "running".to_owned();
        record.write()?;

        let result = finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Unreachable,
            || false,
            false,
        );

        let Err(error) = result else {
            let _ = fs::remove_dir_all(&root);
            panic!("a dead child must fail the launch, not report `starting`");
        };
        // ROCMAI-541 AC2: the failure must name the service and carry the log
        // tail, not just the unreadable "starting" status the old readiness
        // timeout alone left behind.
        //
        // This deliberately does not also assert
        // `crate::exit_code_for(Err(error)) == ExitCode::FAILURE`: every
        // `anyhow::Error` that is not a `FixExitCode`/`ClapExitCode` downcast
        // maps to `ExitCode::FAILURE` there (see `exit_code_for`), and
        // `bail!(...)` here can never produce either downcast — so that
        // assertion could not fail regardless of whether this function is
        // correct, and would misleadingly suggest the real `run()` call chain
        // (clap parsing, `start_managed_service`, `exit_code_for`) is exercised
        // here. It is not; this test calls `finalize_managed_launch_readiness`
        // directly.
        let message = error.to_string();
        assert!(message.contains("svc-dead-3"), "{message}");
        assert!(message.contains("Model pulled successfully"), "{message}");

        // Both the manifest and the engine's own state file must read `failed`
        // (AC3) — the manifest because this function already held it, and the
        // engine state file because the engine process that owns it is gone and
        // cannot write "failed" there itself.
        assert_eq!(record.status, "failed");
        let (reread, _) =
            crate::services_text::read_managed_service_manifest(&paths, "svc-dead-3")?;
        assert_eq!(reread.status, "failed");
        let engine_state: serde_json::Value =
            serde_json::from_slice(&fs::read(&record.engine_state_path)?)?;
        assert_eq!(engine_state["status"], "failed");

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// A restart that lands inside this invocation's readiness wait must not be
    /// stamped `failed` by it.
    ///
    /// The two facts this function decides on are both still true in that case:
    /// this invocation's own child really is dead, and `reset_for_restart` has
    /// already cleared the stop evidence the branch above looks for. What is no
    /// longer true is that the service's files describe *this* child — the
    /// restart rewrote `supervisor_pid`/`supervisor_start_ticks` — and stamping
    /// them would mislabel a service that may be serving. It sticks, too:
    /// `refresh_managed_service_runtime_liveness_with` short-circuits on a
    /// `failed` status and never re-probes to correct it.
    ///
    /// The launch still fails for the caller, because this invocation's child
    /// did die; what the guard changes is whose record pays for it. Removing
    /// the `still_watching_this_supervisor` call fails this test.
    #[test]
    fn finalize_managed_launch_readiness_does_not_stamp_a_restarted_service() -> Result<()> {
        let (root, paths) = test_paths("finalize-readiness-restarted");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-restarted-1",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11531,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        fs::create_dir_all(record.log_path.parent().expect("log parent"))?;
        fs::write(&record.log_path, "Model pulled successfully\n")?;
        record.status = "running".to_owned();
        record.write()?;

        // What a restart leaves behind: the same `service_id`, a different
        // supervisor. Written to disk only -- `record` keeps this invocation's
        // own identity, which is exactly the divergence the guard reads.
        let (mut restarted, _) =
            crate::services_text::read_managed_service_manifest(&paths, "svc-restarted-1")?;
        restarted.supervisor_pid = record.supervisor_pid.wrapping_add(1);
        restarted.supervisor_start_ticks = Some(4_242_424_242);
        restarted.status = "running".to_owned();
        restarted.write()?;

        let result = finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Unreachable,
            || false,
            false,
        );

        let Err(error) = result else {
            let _ = fs::remove_dir_all(&root);
            panic!("this invocation's child died, so its own launch must still fail");
        };
        assert!(error.to_string().contains("svc-restarted-1"));

        // The restarted service's record must be untouched by this invocation.
        let (reread, _) =
            crate::services_text::read_managed_service_manifest(&paths, "svc-restarted-1")?;
        assert_eq!(
            reread.status, "running",
            "a restarted service was stamped `failed` by an older invocation's readiness wait"
        );
        assert_ne!(record.status, "failed");

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The same readiness timeout must NOT be treated as a failure while the
    /// child is still alive — a large model can legitimately take longer than
    /// the readiness budget to load, and that must keep reporting `starting`
    /// with exit code 0, not regress into the failure path above.
    #[test]
    fn finalize_managed_launch_readiness_keeps_reporting_starting_while_alive() -> Result<()> {
        let (root, paths) = test_paths("finalize-readiness-alive-child");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-alive-1",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11531,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.write()?;

        finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Unreachable,
            || true,
            false,
        )?;

        assert_eq!(record.status, "starting");
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// A readiness wait that already confirmed `Serving` must not be
    /// second-guessed by a liveness check that could itself lose a race against
    /// the same process exiting right after answering the probe.
    #[test]
    fn finalize_managed_launch_readiness_trusts_a_confirmed_serving_result() -> Result<()> {
        let (root, paths) = test_paths("finalize-readiness-serving-raced");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-served-1",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11532,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.write()?;

        finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Serving,
            || false,
            false,
        )?;

        assert_eq!(record.status, "ready");
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The `stop_seen_while_waiting` latch, which nothing else exercised:
    /// every other caller test passes `false`, so deleting the latch left the
    /// whole suite green.
    ///
    /// The case it exists for is a stop that both begins and finishes inside
    /// the readiness wait. `stop_internal_managed_service_confirmed_with`
    /// clears its own marker once it confirms, so by the time the one-shot
    /// check runs there is nothing left on disk to find — only the latch,
    /// sampled while the marker was briefly up, still knows. Here the manifest
    /// is deliberately left with NO marker and a non-`stopped` status, so the
    /// point check cannot answer and the latch is the only thing that can.
    #[test]
    fn finalize_managed_launch_readiness_trusts_the_latch_after_the_marker_is_cleared() -> Result<()>
    {
        let (root, paths) = test_paths("finalize-readiness-latched-stop");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-latched-stop",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11534,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;

        // What `StopWatch` would have latched: sample while the marker is up,
        // then let the confirmed stop clear it again, as production does.
        let mut watch = StopWatch::default();
        let mut marked = record.clone();
        marked.stop_requested_unix_ms = Some(4_242);
        marked.write()?;
        watch.sample(&paths, "svc-latched-stop");
        marked.stop_requested_unix_ms = None;
        marked.write()?;
        assert!(
            watch.stop_was_requested(),
            "the watch must latch a marker it saw, and keep it after the clear"
        );
        assert!(
            !managed_service_stop_was_requested(&paths, "svc-latched-stop"),
            "the one-shot check must be blind here, or this test would pass \
             without the latch"
        );

        let proceed = finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Unreachable,
            || false,
            watch.stop_was_requested(),
        )?;

        let (on_disk, _) =
            crate::services_text::read_managed_service_manifest(&paths, "svc-latched-stop")?;
        let _ = fs::remove_dir_all(root);

        assert!(
            !proceed,
            "a latched stop must tell the caller to skip its own follow-up write"
        );
        assert_ne!(
            on_disk.status, "failed",
            "a stop the latch saw must not be recorded as a crash"
        );
        Ok(())
    }

    /// ROCMAI-541: a `rocm services stop`/`rocm dash` stop that lands
    /// during the 45s launch readiness wait must not be reported as a crash —
    /// no `failed` status, no non-zero exit — and the in-memory `record` must
    /// come back refreshed from disk so a caller's own follow-up write cannot
    /// clobber the stop marker the stop itself just wrote.
    #[test]
    fn finalize_managed_launch_readiness_treats_a_requested_stop_as_clean() -> Result<()> {
        let (root, paths) = test_paths("finalize-readiness-requested-stop");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-stopped-during-launch",
            "lemonade",
            "qwen",
            "qwen-canonical",
            "127.0.0.1",
            11533,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;

        // Simulate a concurrent `rocm services stop` landing mid-wait: it
        // writes the marker (and, once confirmed, `status: "stopped"`) to the
        // manifest on disk — never through this stale in-memory `record`.
        let mut stopped_on_disk = record.clone();
        stopped_on_disk.status = "stopped".to_owned();
        stopped_on_disk.stop_requested_unix_ms = Some(999);
        stopped_on_disk.write()?;

        let proceed = finalize_managed_launch_readiness(
            &paths,
            &mut record,
            EndpointReadiness::Unreachable,
            || false,
            false,
        )?;

        assert!(
            !proceed,
            "a requested stop must tell the caller to skip its own follow-up write"
        );
        assert_eq!(record.status, "stopped", "record is refreshed from disk");
        assert_eq!(record.stop_requested_unix_ms, Some(999));

        let (on_disk, _) = crate::services_text::read_managed_service_manifest(
            &paths,
            "svc-stopped-during-launch",
        )?;
        assert_eq!(
            on_disk.status, "stopped",
            "the manifest must not be overwritten with `failed`"
        );
        assert_eq!(on_disk.stop_requested_unix_ms, Some(999));

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The startup wait's sample comes from the manifest on disk: the
    /// engine's own `failed`, liveness by pid (supervisor or engine), and the
    /// name the endpoint answers to.
    #[test]
    fn observe_launch_reads_failure_and_liveness_from_the_manifest() -> anyhow::Result<()> {
        let (root, paths) = test_paths("observe-launch");
        paths.ensure()?;
        let mut held = reusable_record(&paths);
        held.supervisor_pid = 41;
        held.engine_pid = Some(42);
        held.write()?;
        let sample = |pid_alive: fn(u32) -> bool| observe_launch(&paths, &held, None, pid_alive);

        let (model, signals) = sample(|pid| pid == 42);
        assert_eq!(model, held.served_model_name());
        assert!(!signals.engine_failed);
        assert!(signals.process_alive, "a live engine pid is alive");
        assert!(
            sample(|pid| pid == 41).1.process_alive,
            "a live supervisor is alive"
        );
        assert!(
            !sample(|_| false).1.process_alive,
            "every recorded pid dead"
        );

        // Another writer's `failed` on disk is what the wait sees, not the
        // copy it holds.
        let mut on_disk = ManagedServiceRecord::read_manifest(&paths, &held.service_id)?;
        on_disk.status = "failed".to_owned();
        on_disk.write()?;
        assert!(sample(|_| true).1.engine_failed);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn wait_past_readiness_budget_keeps_probing_a_live_starting_server() {
        let mut readings = vec![
            EndpointReadiness::Unreachable,
            EndpointReadiness::Unreachable,
            EndpointReadiness::Serving,
        ]
        .into_iter();
        let mut probes = 0;
        let result = wait_past_readiness_budget(
            || {
                probes += 1;
                readings.next().expect("no probe after Serving")
            },
            || true,
        );
        assert_eq!(result, EndpointReadiness::Serving);
        assert_eq!(probes, 3, "must outlast the first (45s) budget");
    }

    #[test]
    fn wait_past_readiness_budget_stops_when_the_server_exits() {
        let mut probes = 0;
        let result = wait_past_readiness_budget(
            || {
                probes += 1;
                EndpointReadiness::Unreachable
            },
            || false,
        );
        assert_eq!(result, EndpointReadiness::Unreachable);
        assert_eq!(probes, 1);
    }

    #[test]
    fn wait_past_readiness_budget_returns_at_once_when_already_serving() {
        let result = wait_past_readiness_budget(
            || EndpointReadiness::Serving,
            || panic!("liveness is irrelevant once serving"),
        );
        assert_eq!(result, EndpointReadiness::Serving);
    }

    fn record_tracking_pid(paths: &AppPaths, pid: u32, status: &str) -> ManagedServiceRecord {
        let mut record = reusable_record(paths);
        record.supervisor_pid = pid;
        record.status = status.to_owned();
        record
    }

    #[cfg(unix)]
    fn exited_child_pid() -> u32 {
        let mut child = std::process::Command::new("true").spawn().expect("spawn");
        let pid = child.id();
        child.wait().expect("reap");
        pid
    }

    #[test]
    #[cfg(unix)]
    fn record_still_live_ends_the_wait_for_a_dead_pid_with_a_live_status() {
        // The engine was killed before writing any terminal state: the status
        // still says `starting`, but nothing tracked is running.
        let (root, paths) = test_paths("wait-dead-pid");
        paths.ensure().expect("paths");
        let dead = record_tracking_pid(&paths, exited_child_pid(), "starting");
        assert!(!record_still_live(&dead, service_pid_is_alive));

        let alive = record_tracking_pid(&paths, std::process::id(), "starting");
        assert!(record_still_live(&alive, service_pid_is_alive));

        // A settled status ends the wait even while the pid is still around.
        let stopped = record_tracking_pid(&paths, std::process::id(), "stopped");
        assert!(!record_still_live(&stopped, service_pid_is_alive));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn service_pid_is_alive_treats_an_unreaped_child_as_dead() {
        // The serve supervisor is our own unreaped child; once it dies it is a
        // zombie that `kill(pid, 0)` still reports as running.
        let mut child = std::process::Command::new("true").spawn().expect("spawn");
        let pid = child.id();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !process_is_zombie(pid) {
            assert!(std::time::Instant::now() < deadline, "child never exited");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rocm_core::process_is_running(pid),
            "a zombie still answers kill(pid, 0)"
        );
        assert!(!service_pid_is_alive(pid));
        child.wait().expect("reap");
    }

    #[test]
    fn service_still_live_follows_the_manifest_on_disk() {
        // `rocm services stop` from another terminal writes only the manifest.
        let (root, paths) = test_paths("wait-manifest");
        paths.ensure().expect("paths");
        let mut record = record_tracking_pid(&paths, std::process::id(), "starting");
        record.write().expect("write");
        assert!(service_still_live(&paths, &record.service_id, |_| true));

        record.status = "stopped".to_owned();
        record.write().expect("write");
        assert!(!service_still_live(&paths, &record.service_id, |_| true));
        assert!(!service_still_live(&paths, "no-such-service", |_| true));
        let _ = fs::remove_dir_all(root);
    }

    /// The residual ROCMAI-541 race, closed.
    ///
    /// `refresh_managed_service_runtime_liveness` reconciles any record with
    /// dead tracked pids to `"stopped"` without writing any marker, so when this
    /// predicate also accepted `status == "stopped"`, a crash that any
    /// concurrent `rocm services`/`rocm dash`/`rocmd` read happened to reconcile
    /// became indistinguishable from a deliberate stop — and was reported as
    /// one, exit 0.
    ///
    /// Both directions are pinned. The first arm is the regression itself; the
    /// second is what stopped the term being removable before, and would fail
    /// for a fix that simply deleted the status term without making the stop's
    /// own evidence durable.
    #[test]
    fn a_reconciled_crash_is_not_evidence_of_a_stop_but_a_settled_one_is() {
        let (root, paths) = test_paths("stop-evidence");
        paths.ensure().expect("paths");

        // What the liveness reconciliation leaves behind for a crash: status
        // `stopped`, and no marker of any kind.
        let mut crashed = record_tracking_pid(&paths, 0, "stopped");
        crashed.stop_requested_unix_ms = None;
        crashed.settled_stop_unix_ms = None;
        crashed.write().expect("write");
        assert!(
            !managed_service_stop_was_requested(&paths, &crashed.service_id),
            "a status nobody can vouch for must not pass as proof of a stop"
        );

        // What a confirmed stop leaves behind: the marker moved to the durable
        // field. Same status, opposite verdict - which is the whole point.
        let mut settled =
            ManagedServiceRecord::read_manifest(&paths, &crashed.service_id).expect("read");
        settled.settled_stop_unix_ms = Some(4_321);
        settled.write().expect("write");
        assert!(
            managed_service_stop_was_requested(&paths, &settled.service_id),
            "a stop that confirmed and cleared its own marker still has to be recognised"
        );

        // And a relaunch clears it, so an old stop cannot mask a later crash.
        let mut relaunched =
            ManagedServiceRecord::read_manifest(&paths, &crashed.service_id).expect("read");
        relaunched.reset_for_restart();
        relaunched.write().expect("write");
        assert!(
            !managed_service_stop_was_requested(&paths, &relaunched.service_id),
            "the stop before a relaunch says nothing about the run after it"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A `--wait` that ends in a death nobody asked for is a crash, and used to
    /// be written as `stopped` whatever happened. That is the same defect
    /// ROCMAI-541 fixes on the launch, attach and restart paths, left open at
    /// the one exit point none of them covers: a death that bypasses the
    /// engine's own backstop (SIGKILL, OOM kill, segfault) leaves the record
    /// still reading live, so there is nothing on disk saying why.
    ///
    /// All three arms are pinned together because they are one decision. Only
    /// asserting the crash arm would pass a version that stopped reading the
    /// stop marker entirely and called every `--wait` death a crash — the
    /// mirror-image defect, and the one that would start telling users a
    /// deliberate stop had failed.
    #[test]
    fn a_wait_death_is_a_crash_unless_a_stop_asked_for_it() {
        let (root, paths) = test_paths("wait-settle");
        paths.ensure().expect("paths");
        let mut record = record_tracking_pid(&paths, 4_242, "starting");
        record.write().expect("write");
        // A manifest whose recorded process is alive again was restarted by
        // `rocmd` after the caller decided the server ended: left alone, and
        // checked before the crash/stop verdict so a restart is never stamped.
        settle_exited_service(
            &paths,
            &record.service_id,
            &|_| true,
            UnexplainedDeath::IsACrash,
        );
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(after.status, "starting");

        // The stop arm, while nothing has stamped `failed` anywhere: the marker
        // is the sound evidence, so a stop that landed during the wait settles
        // as a clean stop.
        record.stop_requested_unix_ms = Some(1);
        record.status = "starting".to_owned();
        record.write().expect("write");
        assert!(
            settle_exited_service(
                &paths,
                &record.service_id,
                &|_| false,
                UnexplainedDeath::IsACrash,
            ),
            "it must report the death as deliberate, so `--wait` can say so"
        );
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(
            after.status, "stopped",
            "a stop that asked for this death must not be reported as a crash"
        );

        // Same service, same death, with nothing having asked for it.
        record.stop_requested_unix_ms = None;
        record.status = "starting".to_owned();
        record.write().expect("write");
        settle_exited_service(
            &paths,
            &record.service_id,
            &|_| false,
            UnexplainedDeath::IsACrash,
        );
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(
            after.status, "failed",
            "nothing asked this server to stop, so its death is a crash"
        );
        let engine_state = fs::read_to_string(&record.engine_state_path).expect("engine state");
        assert!(
            engine_state.contains("\"failed\""),
            "the engine state file is the one ROCMAI-541 names: {engine_state}"
        );

        // A record that already failed keeps its reason — but the status alone
        // cannot show that, because a `failed` -> `failed` rewrite is
        // invisible: delete the `latest.status == on_disk` early return and the
        // status assertion still passes. What that guard actually controls is
        // whether the stale snapshot is written back AT ALL, so that is what
        // this arm pins.
        //
        // The stand-in for "something a concurrent writer put on disk after
        // this snapshot was taken" is a key the record type does not know.
        // `ManagedServiceRecord` does not deny unknown fields, so a read drops
        // the key and `write()` re-serializes without it: the key survives
        // exactly when the early return skips the write, and is gone the moment
        // it does not. That makes this a deterministic proxy for the concurrent
        // overwrite the guard exists to avoid, with no race to win.
        record.status = "failed".to_owned();
        record.write().expect("write");
        let planted_manifest = fs::read_to_string(&record.manifest_path).expect("manifest");
        let mut planted: serde_json::Value =
            serde_json::from_str(&planted_manifest).expect("manifest json");
        planted["written_by_someone_else"] = serde_json::json!("concurrent");
        fs::write(
            &record.manifest_path,
            serde_json::to_vec_pretty(&planted).expect("serialize"),
        )
        .expect("plant a concurrent writer's field");

        settle_exited_service(
            &paths,
            &record.service_id,
            &|_| false,
            UnexplainedDeath::IsACrash,
        );

        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(after.status, "failed");
        let reread: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&record.manifest_path).expect("manifest"))
                .expect("manifest json");
        assert_eq!(
            reread
                .get("written_by_someone_else")
                .and_then(serde_json::Value::as_str),
            Some("concurrent"),
            "the refresh learned nothing, so the stale snapshot must not be written back over \
             whatever else reached disk meanwhile: that is what the `latest.status == on_disk` \
             early return protects, and what the status assertion above cannot see"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// ROCMAI-541 review: a crash must still be recorded as one when a
    /// concurrent listing has already reconciled the record to `stopped`.
    ///
    /// `managed_service_status`'s liveness refresh demotes any record whose
    /// recorded PIDs are gone to `stopped`, unconditionally and without
    /// knowing why they went — `rocm services` and `rocm dash` both run it. A
    /// listing that lands between the death and `settle_exited_service`'s own
    /// read therefore leaves `on_disk == "stopped"`, and the refresh from the
    /// engine state file (which a SIGKILL/OOM-kill never got to write) learns
    /// nothing, so `latest.status == on_disk` and the early return fires with
    /// `deliberate == false`. Before this fix that returned the right verdict
    /// to the caller — the `--wait` said the server exited — while leaving
    /// `rocm services list` showing a clean stop forever, which is the exact
    /// acceptance criterion this change set exists to meet.
    ///
    /// The planted key is the same stand-in the arm above uses, and is the
    /// other half of the assertion: the correction has to go through
    /// `mark_service_failed`'s read-modify-write merge, not `latest.write()`,
    /// or it fixes the status by reintroducing the clobber the early return is
    /// there to prevent. Both assertions have to hold at once; either one
    /// alone admits a wrong implementation.
    #[test]
    fn a_crash_is_still_recorded_when_a_listing_already_reconciled_it_to_stopped() {
        let (root, paths) = test_paths("wait-settle-after-reconcile");
        paths.ensure().expect("paths");
        // The record exactly as a concurrent listing's liveness refresh leaves
        // it: demoted to `stopped` because the PIDs are gone, with neither
        // half of the stop evidence set, because nobody asked for this death.
        let mut record = record_tracking_pid(&paths, 0, "stopped");
        record.stop_requested_unix_ms = None;
        record.settled_stop_unix_ms = None;
        record.write().expect("write");
        let planted_manifest = fs::read_to_string(&record.manifest_path).expect("manifest");
        let mut planted: serde_json::Value =
            serde_json::from_str(&planted_manifest).expect("manifest json");
        planted["written_by_someone_else"] = serde_json::json!("concurrent");
        fs::write(
            &record.manifest_path,
            serde_json::to_vec_pretty(&planted).expect("serialize"),
        )
        .expect("plant a concurrent writer's field");

        let deliberate = settle_exited_service(
            &paths,
            &record.service_id,
            &|_| false,
            UnexplainedDeath::IsACrash,
        );

        assert!(
            !deliberate,
            "a status a liveness refresh wrote is not evidence anyone asked for this death"
        );
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(
            after.status, "failed",
            "the death was reported as a crash, so the manifest must not be left reading \
             `stopped` by a listing that happened to run first"
        );
        let engine_state = fs::read_to_string(&record.engine_state_path).expect("engine state");
        assert!(
            engine_state.contains("\"failed\""),
            "the engine state file is the one ROCMAI-541 names: {engine_state}"
        );
        let reread: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&record.manifest_path).expect("manifest"))
                .expect("manifest json");
        assert_eq!(
            reread
                .get("written_by_someone_else")
                .and_then(serde_json::Value::as_str),
            Some("concurrent"),
            "the correction must be a merge of the `status` key, not a write-back of the \
             stale snapshot: fixing the status by clobbering the concurrent writer trades \
             one defect for the one the early return exists to prevent"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A stop this function proves was asked for must not be persisted as
    /// `failed` by the engine's own verdict, on the one path where the two can
    /// both be true at once.
    ///
    /// The adapters read a SIGTERM'd server as an unsuccessful exit and write
    /// `failed` to their own state file. When that lands beside a real stop
    /// marker, `refresh_from_engine_state` adopts it, the record is no longer
    /// live, and the status now differs from what was on disk — so neither the
    /// live-child branch nor the `status == on_disk` early return applies and
    /// execution reaches the final write. Before the reconciliation there, that
    /// write persisted `failed` while the function returned `deliberate`,
    /// i.e. the caller printed "was stopped" over a record saying `failed`.
    ///
    /// `died_during_wait_reports_a_stop_during_the_wait_as_a_stop` asserts the
    /// same invariant but deletes the engine state file first, specifically to
    /// avoid this interleaving — so it cannot catch this. This one plants both
    /// halves on purpose.
    #[test]
    fn a_stop_is_not_rewritten_as_failed_by_the_engines_own_verdict() {
        let (root, paths) = test_paths("wait-settle-stop-vs-engine-failed");
        paths.ensure().expect("paths");
        // Not live (pid 0), status still `starting` on disk, and a genuine stop
        // marker: `stop_internal_managed_service_confirmed_with` writes this
        // before it signals anything.
        let mut record = record_tracking_pid(&paths, 0, "starting");
        record.stop_requested_unix_ms = Some(4_242);
        record.settled_stop_unix_ms = None;
        record.write().expect("write");
        // The adapter's own misreading of the SIGTERM, in the place it writes
        // it. This is what the refresh adopts.
        let _ = rocm_core::merge_json_state_file(
            &record.engine_state_path,
            &serde_json::json!({ "status": "failed" }),
        );

        let deliberate = settle_exited_service(
            &paths,
            &record.service_id,
            &|_| false,
            UnexplainedDeath::IsACrash,
        );

        assert!(
            deliberate,
            "a recorded stop is a fact about what was asked for; an exit code is an \
             inference, and must not overturn it"
        );
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_ne!(
            after.status, "failed",
            "the verdict said this death was asked for, so the record must not be left \
             reading `failed` for `rocm services list` and rocmd's recovery scan"
        );
        assert_eq!(after.status, "stopped");
        assert_eq!(
            after.stop_requested_unix_ms,
            Some(4_242),
            "the evidence the verdict rests on survives the write"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// The other side of the correction above: a record that already reads
    /// `failed` has nothing to correct, so the early return must still write
    /// nothing at all.
    ///
    /// Without the `!= "failed"` term the correction would run here too. The
    /// manifest status would not visibly change — a `failed` -> `failed` merge
    /// is invisible, which is why the sibling test's planted key cannot see it
    /// — but `mark_service_failed` also stamps the engine state file, and
    /// *that* is observable: a record whose engine never wrote one would come
    /// back from a no-op settle with a state file this process invented. That
    /// is what pins the guard.
    #[test]
    fn a_record_that_already_failed_is_left_entirely_alone() {
        let (root, paths) = test_paths("wait-settle-already-failed");
        paths.ensure().expect("paths");
        // The engine stamped the manifest itself and got no further: its own
        // reason is on the record, and no engine state file exists.
        let mut record = record_tracking_pid(&paths, 0, "failed");
        record.stop_requested_unix_ms = None;
        record.settled_stop_unix_ms = None;
        record.write().expect("write");
        let _ = fs::remove_file(&record.engine_state_path);
        let planted_manifest = fs::read_to_string(&record.manifest_path).expect("manifest");
        let mut planted: serde_json::Value =
            serde_json::from_str(&planted_manifest).expect("manifest json");
        planted["written_by_someone_else"] = serde_json::json!("concurrent");
        fs::write(
            &record.manifest_path,
            serde_json::to_vec_pretty(&planted).expect("serialize"),
        )
        .expect("plant a concurrent writer's field");

        assert!(!settle_exited_service(
            &paths,
            &record.service_id,
            &|_| false,
            UnexplainedDeath::IsACrash,
        ));

        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(after.status, "failed");
        let reread: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&record.manifest_path).expect("manifest"))
                .expect("manifest json");
        assert_eq!(
            reread
                .get("written_by_someone_else")
                .and_then(serde_json::Value::as_str),
            Some("concurrent"),
            "nothing changed, so nothing may be written back"
        );
        assert!(
            !record.engine_state_path.exists(),
            "a settle with nothing to correct must not invent an engine state file the \
             engine never wrote"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// ROCMAI-541 review: `settle_exited_service`'s verdict and the record it
    /// stamps now come from one snapshot, asked through the same predicate
    /// [`managed_service_stop_was_requested`] uses.
    ///
    /// The case that pins it is a stop whose *marker* has already been consumed
    /// — a concurrent `rocm services`/`dash` listing saw the signalled PIDs gone
    /// first and moved `stop_requested_unix_ms` into `settled_stop_unix_ms`,
    /// which is the whole reason that durable field exists. A call site that
    /// re-spelled the test inline as "the marker is set" would find nothing and
    /// stamp `failed` on a death the operator asked for, while still reading as
    /// though it asked the shared question. Deleting either term of
    /// `record_shows_stop_was_requested` fails one of these two arms.
    ///
    /// What this does **not** cover is the interleaving itself. The defect was
    /// two independent reads with a stop landing between them; the fix is that
    /// there is now only one read, which is a structural property, not one a
    /// single-threaded test can falsify. No read-injection seam exists here to
    /// drive it, and none is added for this.
    #[test]
    fn a_wait_death_is_deliberate_on_either_half_of_the_stop_evidence() {
        for consumed_by_a_listing in [false, true] {
            let (root, paths) =
                test_paths(&format!("wait-settle-evidence-{consumed_by_a_listing}"));
            paths.ensure().expect("paths");
            let mut record = record_tracking_pid(&paths, 0, "starting");
            if consumed_by_a_listing {
                record.stop_requested_unix_ms = None;
                record.settled_stop_unix_ms = Some(7);
            } else {
                record.stop_requested_unix_ms = Some(7);
                record.settled_stop_unix_ms = None;
            }
            record.write().expect("write");

            let deliberate = settle_exited_service(
                &paths,
                &record.service_id,
                &|_| false,
                UnexplainedDeath::IsACrash,
            );
            let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();

            assert!(
                deliberate,
                "a stop is a stop whether or not a listing has already settled \
                 its marker (consumed={consumed_by_a_listing})"
            );
            assert_eq!(
                after.status, "stopped",
                "and the record it stamps must agree with that verdict, not \
                 call the operator's own stop a crash (consumed={consumed_by_a_listing})"
            );
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn wait_on_existing_service_only_waits_for_one_that_is_not_serving_yet() {
        // The probing path itself blocks for the readiness budget, so only the
        // cases that must return at once are driven here; a manifest planted
        // as `starting` would hang a wrong guard for 45 seconds and fail it.
        let (root, paths) = test_paths("wait-existing");
        paths.ensure().expect("paths");
        let mut record = record_tracking_pid(&paths, 0, "starting");
        record.write().expect("write");
        let report = |status: &str| ManagedLaunchReport {
            service_id: record.service_id.clone(),
            endpoint_url: record.endpoint_url.clone(),
            status: status.to_owned(),
            already_running: true,
            child_pid: None,
            log_path: None,
            manifest_path: None,
        };
        let mut no_tick = |_: Duration| panic!("must not probe");

        let untouched = wait_on_existing_service(
            &paths,
            report("starting"),
            false,
            &mut |_, _| panic!("no --wait, no banner"),
            &mut no_tick,
        )
        .expect("no --wait");
        assert_eq!(untouched.status, "starting");
        let ready = wait_on_existing_service(
            &paths,
            report("ready"),
            true,
            &mut |_, _| panic!("already ready, no banner"),
            &mut no_tick,
        )
        .expect("already ready");
        assert_eq!(ready.status, "ready");
        record.status = "starting".to_owned();
        assert_eq!(
            ManagedServiceRecord::read_manifest(&paths, &record.service_id)
                .unwrap()
                .status,
            "starting"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn died_during_wait_fails_with_a_logs_pointer_and_records_the_death() {
        let (root, paths) = test_paths("wait-died");
        paths.ensure().expect("paths");
        let record = record_tracking_pid(&paths, std::process::id(), "starting");
        record.write().expect("write");
        let id = record.service_id;
        let status = || {
            ManagedServiceRecord::read_manifest(&paths, &id)
                .unwrap()
                .status
        };

        let launch_errors = || {
            rocm_core::load_recent_audit_events(&paths, 20)
                .unwrap_or_default()
                .into_iter()
                .filter(|event| {
                    event.action == "managed_service_launch"
                        && event.level == "error"
                        && event.service_id.as_deref() == Some(id.as_str())
                })
                .collect::<Vec<_>>()
        };

        // Not waiting, already serving, or still alive: no failure, no change.
        let pending = EndpointReadiness::Unreachable;
        assert!(died_during_wait(&paths, &id, false, pending, |_| false).is_none());
        assert!(
            died_during_wait(&paths, &id, true, EndpointReadiness::Serving, |_| false).is_none()
        );
        assert!(died_during_wait(&paths, &id, true, pending, |_| true).is_none());
        assert_eq!(status(), "starting");
        assert!(launch_errors().is_empty(), "{:?}", launch_errors());

        // Dead server: the error names the logs command and the record is settled.
        let error = died_during_wait(&paths, &id, true, pending, |_| false).expect("died");
        let message = error.to_string();
        assert!(
            message.contains("exited before it became ready"),
            "{message}"
        );
        // The hinted command parses with these flags:
        // `advised_commands::every_advised_command_parses` reads the literal in
        // `service_hints::service_logs_hint`.
        assert!(
            message.contains(&format!(
                "see `{}` for its output",
                service_hints::service_logs_hint(&id)
            )),
            "{message}"
        );
        // `failed`, not `stopped`: nothing asked this server to stop, so the
        // death is a crash. This assertion used to read "stopped", which was
        // the ROCMAI-541 defect surviving at the `--wait` exit point — see
        // `a_wait_death_is_a_crash_unless_a_stop_asked_for_it`.
        assert_eq!(status(), "failed");

        // The `--wait` death is audited like the interactive wait's.
        let errors = launch_errors();
        assert!(
            errors
                .iter()
                .any(|event| event.message.contains("exited during --wait")),
            "{errors:?}"
        );

        // And a stop issued from another shell during the wait gets its own
        // sentence, rather than being reported as the launch failing to come
        // up. Re-plant the record live with a stop marker on it.
        let mut stopped = ManagedServiceRecord::read_manifest(&paths, &id).expect("read");
        stopped.status = "starting".to_owned();
        stopped.supervisor_pid = std::process::id();
        stopped.stop_requested_unix_ms = Some(1);
        stopped.write().expect("write");
        // The crash arm above stamped the *engine state file* `failed`, and
        // `settle_exited_service` refreshes the record from it — so without
        // clearing it this arm would re-adopt the previous arm's verdict and
        // the state assertion below would fail on fixture carry-over rather
        // than on the behaviour. Found by adding that assertion.
        let _ = fs::remove_file(&stopped.engine_state_path);
        let error =
            died_during_wait(&paths, &id, true, pending, |_| false).expect("the wait still ends");
        let message = error.to_string();
        assert!(
            message.contains("was stopped before it became ready"),
            "an operator's own stop must not read as the server failing to start: {message}"
        );
        assert!(
            !message.contains("exited before it became ready"),
            "{message}"
        );

        // The message and the state it leaves, pinned together. Asserting only
        // the wording would stay green if the same stop were still recorded as
        // a crash — the user would read "stopped" and `rocm services list`
        // would say `failed`.
        let after = ManagedServiceRecord::read_manifest(&paths, &id).expect("read");
        assert_ne!(
            after.status, "failed",
            "a stop reported as a stop must not also be recorded as a crash"
        );
        let engine_state = fs::read_to_string(&after.engine_state_path).unwrap_or_default();
        assert!(
            !engine_state.contains("\"failed\""),
            "nor may its engine state file say failed: {engine_state}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn died_during_wait_adds_the_oom_guidance_for_a_vllm_launch_only() {
        let (root, paths) = test_paths("wait-died-oom");
        paths.ensure().expect("paths");
        let mut record = record_tracking_pid(&paths, std::process::id(), "starting");
        record.engine = "vllm".to_owned();
        record.write().expect("write");
        fs::write(
            &record.log_path,
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.\n",
        )
        .expect("log");
        let id = record.service_id;
        let pending = EndpointReadiness::Unreachable;
        let message = died_during_wait(&paths, &id, true, pending, |_| false)
            .expect("died")
            .to_string();
        assert!(message.contains("--gpu-memory-utilization"), "{message}");

        // Same log, another engine: that engine has no such flag to suggest.
        let mut record = ManagedServiceRecord::read_manifest(&paths, &id).unwrap();
        record.engine = "lemonade".to_owned();
        "starting".clone_into(&mut record.status);
        record.write().expect("write");
        let message = died_during_wait(&paths, &id, true, pending, |_| false)
            .expect("died")
            .to_string();
        assert!(!message.contains("--gpu-memory-utilization"), "{message}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn record_wait_outcome_keeps_what_changed_on_disk_during_the_wait() {
        let (root, paths) = test_paths("wait-outcome");
        paths.ensure().expect("paths");
        let mut held = record_tracking_pid(&paths, 111, "starting");
        held.write().expect("write");
        // While the wait ran, `rocmd` recorded a restart and a new supervisor.
        let mut elsewhere = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
        elsewhere.restart_count = 3;
        elsewhere.supervisor_pid = 222;
        elsewhere.write().expect("write");

        record_wait_outcome(&paths, &mut held, "ready", true).expect("outcome");
        let after = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
        assert_eq!(after.status, "ready");
        assert!(after.inference_verified_at_unix_ms.is_some());
        assert_eq!(after.restart_count, 3, "the restart count was reverted");
        assert_eq!(after.supervisor_pid, 222, "the supervisor pid was reverted");

        record_wait_outcome(&paths, &mut held, "starting", false).expect("outcome");
        let after = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
        assert_eq!(after.status, "starting");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn record_wait_outcome_does_not_recreate_a_record_removed_during_the_wait() {
        let (root, paths) = test_paths("wait-outcome-removed");
        paths.ensure().expect("paths");
        let mut held = record_tracking_pid(&paths, 111, "starting");
        held.write().expect("write");
        // `rocm services remove` ran while `rocm serve` waited.
        fs::remove_file(&held.manifest_path).expect("remove manifest");

        let error = record_wait_outcome(&paths, &mut held, "ready", true)
            .expect_err("a removed service has nothing left to report");
        let recreated = held.manifest_path.exists();
        let _ = fs::remove_dir_all(root);
        assert!(
            error.to_string().contains("was removed while"),
            "the failure must say what happened: {error:#}"
        );
        assert!(!recreated, "the stale copy must not re-create the record");
    }

    #[test]
    fn an_engine_that_exited_during_the_wait_is_reported_as_exited_not_stopped() {
        // Without `--wait`, `died_during_wait` does not run, so a record the
        // liveness refresh demoted (or the engine marked `failed`) reaches the
        // verdict write. No stop marker: nobody stopped it, it exited.
        for settled_status in ["stopped", "failed"] {
            let (root, paths) = test_paths(&format!("wait-outcome-exited-{settled_status}"));
            paths.ensure().expect("paths");
            let mut held = record_tracking_pid(&paths, 111, "starting");
            held.write().expect("write");
            let mut exited = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
            exited.status = settled_status.to_owned();
            exited.write().expect("write");

            let error = record_wait_outcome(&paths, &mut held, "starting", false)
                .expect_err("a server that exited must not be reported as launched");
            let after = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
            let _ = fs::remove_dir_all(root);
            let message = error.to_string();
            assert!(
                message.contains("exited while it was starting"),
                "{message}"
            );
            assert!(!message.contains("was stopped"), "{message}");
            // The hinted command parses with these flags:
            // `advised_commands::every_advised_command_parses` reads the literal
            // in `service_hints::service_logs_hint`.
            assert!(
                message.contains(&format!(
                    "see `{}` for its output",
                    service_hints::service_logs_hint(&held.service_id)
                )),
                "{message}"
            );
            assert_eq!(
                after.status, settled_status,
                "the settled record was overwritten"
            );
            assert_eq!(after.stop_requested_unix_ms, None);
        }
    }

    #[test]
    fn a_stop_requested_during_the_wait_fails_the_serve_and_keeps_the_stop() {
        // `rocm services stop` writes its marker first and `stopped` only once
        // the processes are gone, so the wait can end on either state: a stop
        // still in flight (the marker on a record that still reads live) or one
        // that settled. Both are a stop the user asked for.
        for settled_status in [None, Some("stopped")] {
            let label = settled_status.unwrap_or("in-flight");
            let (root, paths) = test_paths(&format!("wait-outcome-stop-{label}"));
            paths.ensure().expect("paths");
            let mut held = record_tracking_pid(&paths, 111, "starting");
            held.write().expect("write");
            let mut stopping =
                ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
            stopping.stop_requested_unix_ms = Some(1_234);
            if let Some(status) = settled_status {
                stopping.status = status.to_owned();
            }
            stopping.write().expect("write");
            let status_before = stopping.status.clone();

            // An `Err` here is what `rocm serve` turns into a non-zero exit: the
            // launch path propagates it with `?` and prints nothing else.
            let error = record_wait_outcome(&paths, &mut held, "ready", true)
                .expect_err("a server stopped during its wait must not be reported as launched");
            let after = ManagedServiceRecord::read_manifest(&paths, &held.service_id).unwrap();
            let _ = fs::remove_dir_all(root);
            let message = error.to_string();
            assert!(
                message.contains("a stop was requested for managed service"),
                "{label}: {message}"
            );
            assert!(message.contains(&held.service_id), "{label}: {message}");
            // The advice matches the state: `rocm serve` reuses a record that
            // still reads live, so while the stop is in flight it is told to
            // finish the stop first.
            // Pins the wording only; that the advised stop is what frees the
            // model is proved by
            // `the_advised_stop_frees_the_model_a_stop_in_flight_still_holds`.
            let advises_stop = message.contains(&format!(
                "finish the stop with `{}`",
                service_hints::service_stop_hint(&held.service_id)
            ));
            assert_eq!(advises_stop, settled_status.is_none(), "{label}: {message}");
            assert!(
                message.contains("start it again with `rocm serve`"),
                "{label}: {message}"
            );
            // The message and the state it describes, together: the record keeps
            // the stop's status and marker, and carries no readiness verdict.
            assert_eq!(
                after.status, status_before,
                "{label}: the stop was overwritten"
            );
            assert_eq!(after.stop_requested_unix_ms, Some(1_234), "{label}");
            assert_eq!(after.inference_verified_at_unix_ms, None, "{label}");
        }
    }

    /// The behaviour behind the in-flight advice: while a stop is in flight the
    /// record still reads live, so a new `rocm serve` for the same model would
    /// reuse the service being stopped; the stop the failure names, run through
    /// the real stop path, settles it and leaves nothing for reuse to find.
    #[test]
    fn the_advised_stop_frees_the_model_a_stop_in_flight_still_holds() -> Result<()> {
        let (root, paths) = test_paths("wait-outcome-advised-stop");
        paths.ensure()?;
        // A closed port, so the Lemonade unload the stop sends fails at once.
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))?
            .local_addr()?
            .port();
        let mut held = ManagedServiceRecord::new(
            &paths,
            "lemonade-advised-stop",
            "lemonade",
            "model-ref",
            "canonical/advised-stop",
            "127.0.0.1",
            port,
            "managed",
            // A pid that is alive for the whole test (this test process), so
            // the liveness refresh keeps the record live. The stop below runs
            // the real stop path with a stub termination that reports the pid
            // gone, so nothing is signalled; an unconfirmed stop is not covered
            // here (it keeps the record live, and says so).
            std::process::id(),
            None,
            None,
            None,
        );
        held.status = "starting".to_owned();
        held.write()?;
        let mut stopping = ManagedServiceRecord::read_manifest(&paths, &held.service_id)?;
        stopping.stop_requested_unix_ms = Some(1_234);
        stopping.write()?;
        let reusable = |paths: &AppPaths| -> Result<bool> {
            let records = crate::managed_service_status::load_managed_services(paths)?;
            Ok(crate::managed_service_status::find_live_managed_service(
                &records,
                "lemonade",
                "canonical/advised-stop",
            )
            .is_some())
        };

        let error = record_wait_outcome(&paths, &mut held, "ready", true)
            .expect_err("a stop in flight must fail the serve");
        let reused_while_stopping = reusable(&paths)?;
        crate::service_control::stop_internal_managed_service_with(
            &paths,
            &held.service_id,
            |record| {
                rocm_core::terminate_recorded_service_pids(record, |_| {
                    rocm_core::TerminationOutcome::Graceful
                })
            },
        )?;
        let reused_after_the_stop = reusable(&paths)?;
        let after = ManagedServiceRecord::read_manifest(&paths, &held.service_id);
        let _ = fs::remove_dir_all(root);
        let message = error.to_string();
        assert!(
            message.contains(&service_hints::service_stop_hint(&held.service_id)),
            "{message}"
        );
        assert!(
            reused_while_stopping,
            "the advice to finish the stop rests on reuse taking the record being stopped"
        );
        assert_eq!(after?.status, "stopped");
        assert!(
            !reused_after_the_stop,
            "after the advised stop, `rocm serve` must launch anew rather than reuse"
        );
        Ok(())
    }
}
