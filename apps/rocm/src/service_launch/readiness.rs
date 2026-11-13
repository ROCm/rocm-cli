// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Waiting on a launched or already-live managed service: the readiness budget,
//! liveness of the tracked process, how a wait ends, and the end-to-end OOM
//! fault-injection seam that fabricates a launch that never became ready.

// There is deliberately no `use std::fs`, `use std::thread` or
// `use rocm_core::process_is_running` here: this file names them by full path.
// Their use sites sit under different `cfg` combinations (the Linux-only zombie
// probe, the OOM fault-injection seam, one Linux-only test), so any single import
// would be unused in some build, and a gate on it would have to be kept in step
// with every use site. Keep the full paths. The tests module imports `fs` for
// itself because its tests use it on every platform.
use super::spawn::ManagedLaunchReport;
#[cfg(feature = "e2e-oom-fault-injection")]
use crate::device_policy_name;
use crate::endpoint_keys;
use crate::engine_runtime_select::wait_for_service_http_ready_with_progress;
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

/// How long one readiness wait polls before `rocm serve` reports `starting`.
const READINESS_WAIT: Duration = Duration::from_secs(45);

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

/// Record how a server that died mid-start died, leaving a manifest that a
/// `services stop` already settled untouched. A `failed` state the engine
/// reported is persisted rather than overwritten, so the reason survives.
///
/// Returns whether the death was **asked for**, so the caller can say so. The
/// `--wait` message used to read "the server exited before it became ready"
/// even for a `rocm services stop` issued from another shell, which is a true
/// sentence told in a way that reads as a failure.
fn settle_exited_service(paths: &AppPaths, service_id: &str) -> bool {
    let deliberate = managed_service_stop_was_requested(paths, service_id);
    let Ok(mut latest) = ManagedServiceRecord::read_manifest(paths, service_id) else {
        return deliberate;
    };
    let on_disk = latest.status.clone();
    let _ = latest.refresh_from_engine_state();
    if crate::managed_service_status::managed_service_is_live(&latest) {
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
        if deliberate {
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
        return deliberate;
    }
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
        || service_still_live(paths, service_id, pid_alive)
    {
        return None;
    }
    if settle_exited_service(paths, service_id) {
        // A `rocm services stop` from another shell ends the wait too, and it is
        // not a launch that failed. The wait still ends non-zero, because the
        // thing the caller asked to wait for did not happen — but it says which
        // of the two it was instead of reporting an operator's own stop as the
        // server failing to come up.
        return Some(anyhow!(
            "the server for service `{service_id}` was stopped before it became ready"
        ));
    }
    let mut message = format!(
        "the server exited before it became ready (vLLM and AIM give up after 5 minutes by default; `ROCM_CLI_VLLM_READY_TIMEOUT_SECS` / `ROCM_CLI_AIM_READY_TIMEOUT_SECS` allow longer); see `rocm logs --service {service_id}` for its output"
    );
    // A plain `serve` adds the memory guidance for a vLLM launch that ran out of
    // VRAM; the wait's failure is the same launch, so it gets the same note.
    // "starting" is the status the note keys on for "never became ready", which is
    // what a `--wait` death is, whatever the settled record now says.
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
    Some(anyhow!(message))
}

/// Record the outcome of a finished wait on the manifest as it is now, changing
/// only what the wait learned. The copy read at spawn time can be long stale:
/// `rocmd` may have recorded a restart or a new supervisor pid, and `rocm logs`
/// refreshes the record, so writing it back whole would revert those.
pub(super) fn record_wait_outcome(
    paths: &AppPaths,
    held: &mut ManagedServiceRecord,
    status: &str,
    verified: bool,
) -> Result<()> {
    let mut latest = ManagedServiceRecord::read_manifest(paths, &held.service_id)
        .unwrap_or_else(|_| held.clone());
    status.clone_into(&mut latest.status);
    if verified {
        // Latch the verification the wait just performed, so the readiness checks
        // behind `services list` and chat read it instead of re-probing.
        latest.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
    }
    latest.write()?;
    *held = latest;
    Ok(())
}

/// Test-only fault-injection switch, off (`const false`) in shipped binaries:
/// the `e2e-oom-fault-injection` feature is enabled only by the `xtask e2e`
/// mock-lane build, never by the release build. When enabled and
/// `ROCM_E2E_SIMULATE_OOM_LAUNCH=1` is set on the process, `rocm serve` fabricates
/// the on-disk state of a managed vLLM launch that spawned, wrote an
/// out-of-memory traceback to the log it owns, and never became ready — so the
/// positive OOM-guidance summary path can be verified on a GPU-less CI host,
/// where a real launch cannot run. See [`simulate_oom_managed_launch`].
#[cfg(feature = "e2e-oom-fault-injection")]
pub(crate) fn e2e_simulate_oom_launch() -> bool {
    std::env::var_os("ROCM_E2E_SIMULATE_OOM_LAUNCH").is_some_and(|value| value == "1")
}

/// Release builds carry no fault-injection switch: the seam compiles out
/// entirely, so `rocm serve` behaves identically to a build without the feature.
#[cfg(not(feature = "e2e-oom-fault-injection"))]
pub(crate) const fn e2e_simulate_oom_launch() -> bool {
    false
}

/// Test-only (see [`e2e_simulate_oom_launch`]): fabricate the on-disk state of a
/// managed vLLM launch that spawned, OOM'd, and never became ready, then return
/// the launch report the summary path would see for it — `already_running: false`
/// (this invocation launched it), `status: "starting"` (it failed to become
/// ready), and a `log_path` whose contents carry a real allocator OOM signature.
/// [`crate::serve_summary::append_oom_serve_note`] then renders the memory-knob guidance. No process is
/// spawned. Compiled out of shipped binaries.
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

/// Probe `record`'s endpoint until it answers: once within the readiness
/// budget, or (with `wait_until_ready`) for as long as the server stays alive.
pub(super) fn wait_for_record_ready(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    wait_until_ready: bool,
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
    if wait_until_ready {
        wait_past_readiness_budget(
            || probe(&mut *on_wait_tick),
            || service_still_live(paths, &record.service_id, service_pid_is_alive),
        )
    } else {
        probe(on_wait_tick)
    }
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
    let readiness =
        wait_for_record_ready(paths, &record, true, stored_key.as_deref(), on_wait_tick);
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
/// `spawn_detached_with_log`).
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
/// crash: an operator-requested stop in flight or already confirmed
/// (`stop_requested_unix_ms`), or a status that already reads `stopped`.
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
/// `apps/rocmd/src/watchers.rs`) is a different case: it only ever fires once
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
/// removing it turned `serve-scripted-external-stop-while-attached` (serve-57)
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
    crate::services_text::read_managed_service_manifest(paths, service_id).is_ok_and(
        |(record, _)| {
            record.stop_requested_unix_ms.is_some() || record.settled_stop_unix_ms.is_some()
        },
    )
}

/// The marker half of [`managed_service_stop_was_requested`], without its
/// `status == "stopped"` term.
///
/// For the one caller whose own precondition is a `stopped` record: a restart
/// completes a confirmed stop before it respawns, and `record_restarted_child`
/// does not publish `running` until after the new child clears its early-exit
/// window. Inside that window the manifest still reads `stopped` because *this
/// restart* stopped it, so the full predicate fires unconditionally and reports
/// every early crash as a deliberate stop — dropping the exit status and log
/// tail, and leaving the engine state file unstamped.
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
/// `stop_requested_unix_ms` is the only sound evidence that an exit was
/// intentional (see [`managed_service_stop_was_requested`] for why the record's
/// status is not), but it is deliberately short-lived:
/// `stop_internal_managed_service_confirmed_with` writes it before it signals
/// anything and clears it again once the stop is confirmed. A watcher that only
/// reads it *after* noticing the child is gone can therefore arrive after the
/// marker has been cleared, and report a deliberate stop as a crash.
///
/// Sampling on every liveness poll catches any stop that was still in flight
/// while the child was being watched — in particular an *unconfirmed* stop,
/// which re-stamps the marker and leaves the status alone, so
/// [`managed_service_stop_was_requested`]'s status term would not fire for it
/// either.
///
/// It is a narrowing, not a closure: a stop that is requested, signals, and
/// confirms entirely inside one poll interval clears its marker before the next
/// sample, so the latch misses it. That case is caught by the status term in
/// [`managed_service_stop_was_requested`] — see the residual race documented
/// there. A crash never writes the marker, so the latch cannot produce a false
/// "stopped" of its own.
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
            // The MARKER alone, never `managed_service_stop_was_requested`: its
            // `status == "stopped"` term is what liveness reconciliation
            // (`rocmd`, `dash`, `rocm services`) writes for a *crashed* server,
            // so latching on it would turn a crash reconciled mid-wait into a
            // clean stop and exit 0 — the original bug. The marker is only ever
            // written by a stop, which is what makes this latch sound and the
            // type's doc above true. The status term still backs the one-shot
            // check at the end of a wait, where no latch is available.
            self.seen = managed_service_stop_marker_present(paths, service_id);
        }
    }

    /// Whether a stop was requested at any point during the watch.
    pub(crate) const fn stop_was_requested(&self) -> bool {
        self.seen
    }
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
        // The stop arm runs first, while nothing has stamped `failed` anywhere:
        // the marker is the sound evidence, so a stop that landed during the
        // wait settles as a clean stop.
        let mut record = record_tracking_pid(&paths, 0, "starting");
        record.stop_requested_unix_ms = Some(1);
        record.write().expect("write");
        settle_exited_service(&paths, &record.service_id);
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(
            after.status, "stopped",
            "a stop that asked for this death must not be reported as a crash"
        );
        assert!(
            settle_exited_service(&paths, &record.service_id),
            "and it must report the death as deliberate, so `--wait` can say so"
        );

        // Same service, same death, with nothing having asked for it.
        record.stop_requested_unix_ms = None;
        record.status = "starting".to_owned();
        record.write().expect("write");
        settle_exited_service(&paths, &record.service_id);
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

        // A record that already failed keeps its reason: the guard, not the
        // overwrite, is what spares it (a `failed` -> `failed` rewrite would
        // be invisible).
        record.status = "failed".to_owned();
        record.write().expect("write");
        settle_exited_service(&paths, &record.service_id);
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(after.status, "failed");
        let _ = fs::remove_dir_all(root);
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

        // Not waiting, already serving, or still alive: no failure, no change.
        let pending = EndpointReadiness::Unreachable;
        assert!(died_during_wait(&paths, &id, false, pending, |_| false).is_none());
        assert!(
            died_during_wait(&paths, &id, true, EndpointReadiness::Serving, |_| false).is_none()
        );
        assert!(died_during_wait(&paths, &id, true, pending, |_| true).is_none());
        assert_eq!(status(), "starting");

        // Dead server: the error names the logs command and the record is settled.
        let error = died_during_wait(&paths, &id, true, pending, |_| false).expect("died");
        let message = error.to_string();
        assert!(
            message.contains("exited before it became ready"),
            "{message}"
        );
        assert!(
            message.contains(&format!("rocm logs --service {id}")),
            "{message}"
        );
        // `failed`, not `stopped`: nothing asked this server to stop, so the
        // death is a crash. This assertion used to read "stopped", which was
        // the ROCMAI-541 defect surviving at the `--wait` exit point — see
        // `a_wait_death_is_a_crash_unless_a_stop_asked_for_it`.
        assert_eq!(status(), "failed");

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
}
