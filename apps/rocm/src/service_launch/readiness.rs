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
use crate::serve_progress;
use crate::status_for_readiness;
#[cfg(feature = "e2e-oom-fault-injection")]
use anyhow::Context;
use anyhow::{Result, anyhow};
#[cfg(feature = "e2e-oom-fault-injection")]
use rocm_core::format_http_base_url;
use rocm_core::{AppPaths, EndpointReadiness, ManagedServiceRecord};
#[cfg(feature = "e2e-oom-fault-injection")]
use rocm_engine_protocol::{DevicePolicy, ResolveModelResponse};
use std::fmt::Write as _;
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

/// Record a server that died mid-start as stopped, leaving a manifest that a
/// `services stop` already settled untouched. A `failed` state the engine
/// reported is persisted rather than overwritten, so the reason survives.
fn settle_exited_service(paths: &AppPaths, service_id: &str) {
    let Ok(mut latest) = ManagedServiceRecord::read_manifest(paths, service_id) else {
        return;
    };
    let on_disk = latest.status.clone();
    let _ = latest.refresh_from_engine_state();
    if crate::managed_service_status::managed_service_is_live(&latest) {
        latest.status = "stopped".to_owned();
    } else if latest.status == on_disk {
        return;
    }
    let _ = latest.write();
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
    settle_exited_service(paths, service_id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_launch::test_support::reusable_record;
    use crate::tests::test_paths;
    use std::fs;

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

    #[test]
    fn settle_exited_service_demotes_a_live_manifest_and_spares_a_settled_one() {
        let (root, paths) = test_paths("wait-settle");
        paths.ensure().expect("paths");
        let mut record = record_tracking_pid(&paths, 0, "starting");
        record.write().expect("write");
        settle_exited_service(&paths, &record.service_id);
        let after = ManagedServiceRecord::read_manifest(&paths, &record.service_id).unwrap();
        assert_eq!(after.status, "stopped");

        // A record that already failed keeps its reason: the guard, not the
        // overwrite, is what spares it (a `stopped` -> `stopped` rewrite would
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
        assert_eq!(status(), "stopped");
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
