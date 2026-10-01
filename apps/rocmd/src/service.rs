// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, ManagedServiceRecord, RocmCliConfig, WatcherRuntimeSnapshot,
    builtin_watchers, daemon_binary_path, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

pub(crate) fn stop_managed_service(paths: &AppPaths, service_id: &str) -> Result<Value> {
    let mut record = crate::persistence::load_managed_services(paths)?
        .into_iter()
        .find(|record| record.service_id == service_id)
        .with_context(|| format!("managed service `{service_id}` not found"))?;
    let stops = terminate_recorded_service_pids(&record);
    let pids_where = |keep: fn(&RecordedPidStop) -> bool| -> Vec<u32> {
        stops
            .iter()
            .filter(|stop| keep(stop))
            .map(|s| s.pid)
            .collect()
    };
    let signaled_pids = pids_where(|stop| stop.signaled);
    let force_signaled_pids = pids_where(|stop| stop.forced);
    // Everything that was deliberately not signalled: the stopping process
    // itself, a PID that was already gone, and — the case this list used not to
    // be able to express — a PID that is live but provably belongs to somebody
    // else now. `pid_outcomes` says which.
    let skipped_pids = pids_where(|stop| !stop.signaled);
    let all_stopped = stops.iter().all(|stop| stop.stopped);
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
    // The status records that a stop was carried out, as it always has. Whether
    // every process is confirmed gone is a separate question, answered by
    // `stopped` in the result rather than by overloading the status string.
    record.status = "stopped".to_owned();
    // Recorded PIDs go stale the instant their processes exit, and a stale PID
    // is exactly what the identity check exists to catch — so do not leave one
    // behind for the next stop to find. Cleared only once every recorded process
    // is confirmed gone: an unconfirmed survivor is still the service's, and a
    // later stop has to be able to reach it.
    if all_stopped {
        record.supervisor_pid = 0;
        record.supervisor_start_ticks = None;
        record.engine_pid = None;
        record.engine_start_ticks = None;
    }
    record.write()?;
    // Best-effort and idempotent: a missing key file is not an error, so this
    // is safe to call unconditionally on every stop (including loopback
    // services that never had a key, and repeated stops of an already-stopped
    // service). Leaving the 0600 key file behind after stop would strand a
    // plaintext secret on disk for a service that no longer exists.
    let _ = std::fs::remove_file(rocm_engine_protocol::endpoint_key_file_path(
        paths, service_id,
    ));
    Ok(json!({
        "service": record,
        "signaled_pids": signaled_pids,
        "force_signaled_pids": force_signaled_pids,
        "skipped_pids": skipped_pids,
        "pid_outcomes": pid_outcomes,
        "stopped": all_stopped,
    }))
}

/// How long a stop waits for a recorded process tree to exit on its own before
/// escalating to a forced kill. Matches `rocm services stop`, which terminates
/// the same records.
const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// What a stop did about one PID recorded in a service manifest.
struct RecordedPidStop {
    pid: u32,
    /// Which recorded PID this is: `supervisor` or `engine`.
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
fn terminate_recorded_service_pids(record: &ManagedServiceRecord) -> Vec<RecordedPidStop> {
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
            let outcome = terminate_recorded_pid(&rocm_core::ProcessIdentity::new(pid, ticks));
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
    // This is the same reach the stop has always had on Windows, and it costs no
    // safety: `process_start_ticks` has no `/proc` to read there, so a Windows
    // identity can never be refuted in the first place. It is still gated on the
    // same verdict `terminate_verified` reaches, so a refuted root's subtree
    // stays untouched wherever identity *is* verifiable.
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

pub(crate) async fn run_daemon(
    paths: &AppPaths,
    automations_enabled: bool,
    local_webhook_port: Option<u16>,
) -> Result<()> {
    if local_webhook_port.is_some() && !automations_enabled {
        bail!("local webhook source requires --automations-enabled");
    }

    let config = RocmCliConfig::load(paths)?;
    let mut state = build_runtime_state(&config, automations_enabled);
    let local_webhook = if let Some(port) = local_webhook_port {
        Some(crate::webhook::start_local_webhook_source(port).await?)
    } else {
        None
    };
    let (local_webhook_endpoint, mut local_webhook_receiver, local_webhook_task) =
        match local_webhook {
            Some(source) => (
                Some(source.endpoint),
                Some(source.receiver),
                Some(source.task),
            ),
            None => (None, None, None),
        };
    state.local_webhook_endpoint = local_webhook_endpoint.clone();

    println!("rocmd run");
    println!("  automations enabled: {automations_enabled}");
    println!(
        "  lifecycle: {}",
        if automations_enabled {
            "persistent"
        } else {
            "on-demand"
        }
    );
    println!("  config: {}", paths.config_path().display());
    println!("  state: {}", paths.automation_state_path().display());
    println!(
        "  local_webhook_endpoint: {}",
        local_webhook_endpoint.as_deref().unwrap_or("disabled")
    );
    let enabled_count = state
        .active_watchers
        .iter()
        .filter(|watcher| watcher.enabled)
        .count();
    println!("  enabled watchers: {enabled_count}");
    // This banner is the foreground-loop readiness contract used by callers and
    // integration tests. Flush it before any persistent work so piped stdout on
    // Windows cannot retain the line in a userspace buffer indefinitely.
    io::stdout()
        .flush()
        .context("failed to flush rocmd run banner")?;

    if !automations_enabled {
        println!(
            "  note: rerun with --automations-enabled to keep rocmd alive for watcher execution"
        );
        return Ok(());
    }

    paths.ensure()?;
    state.write(paths)?;
    crate::persistence::record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    crate::watchers::evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let mut ticker = time::interval(crate::WATCHER_TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let config = RocmCliConfig::load(paths)?;
                crate::watchers::reconcile_watcher_snapshots(&config, &mut state);
                crate::watchers::evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            event = crate::webhook::receive_local_webhook_event(&mut local_webhook_receiver) => {
                if let Some(event) = event {
                    let config = RocmCliConfig::load(paths)?;
                    crate::watchers::reconcile_watcher_snapshots(&config, &mut state);
                    crate::persistence::record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "info",
                        "local_webhook_event",
                        &format!(
                            "received local webhook event kind={} watcher_hint={}; dispatching through existing watcher policy; webhook payload grants no new action",
                            event.kind,
                            event.watcher_hint.as_deref().unwrap_or("<none>")
                        ),
                        event.service_id.clone(),
                    )?;
                    if let Err(error) =
                        crate::watchers::evaluate_watchers_for_events(paths, &config, &mut state, &[event])
                    {
                        crate::persistence::record_event(
                            paths,
                            &mut state,
                            "rocmd",
                            "error",
                            "local_webhook_dispatch_failed",
                            &format!(
                                "local webhook event could not be dispatched through watcher policy: {error}"
                            ),
                            None,
                        )?;
                    }
                    state.last_tick_unix_ms = unix_time_millis();
                    state.write(paths)?;
                } else {
                    local_webhook_receiver = None;
                    state.local_webhook_endpoint = None;
                    crate::persistence::record_event(
                        paths,
                        &mut state,
                        "rocmd",
                        "warn",
                        "local_webhook_stopped",
                        "local webhook source stopped; automation daemon continues without webhook ingestion",
                        None,
                    )?;
                    state.write(paths)?;
                }
            }
            () = &mut shutdown => {
                break;
            }
        }
    }

    state.running = false;
    state.last_tick_unix_ms = unix_time_millis();
    state.local_webhook_endpoint = None;
    crate::persistence::record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_stop",
        "rocmd automation supervisor stopped",
        None,
    )?;
    state.write(paths)?;
    if let Some(task) = local_webhook_task {
        task.abort();
    }
    Ok(())
}

pub(crate) fn print_status(paths: &AppPaths) -> Result<()> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  policy: on-demand by default, persistent only with background features");
    println!(
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match AutomationRuntimeState::load(paths)? {
        Some(state) => {
            println!(
                "  automations runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
            println!(
                "  local_webhook_endpoint: {}",
                state
                    .local_webhook_endpoint
                    .as_deref()
                    .unwrap_or("disabled")
            );
            for watcher in state
                .active_watchers
                .into_iter()
                .filter(|watcher| watcher.enabled)
            {
                println!(
                    "  watcher {} mode={} last_event={}",
                    watcher.id,
                    watcher.mode.as_str(),
                    watcher.last_event.as_deref().unwrap_or("<none>")
                );
            }
        }
        None => println!("  automations runtime: inactive"),
    }
    println!(
        "  automation events: {}",
        paths.automation_events_path().display()
    );
    println!("  audit events: {}", paths.audit_events_path().display());

    let records = crate::persistence::load_managed_services(paths)?;
    if records.is_empty() {
        println!("  services: none");
        return Ok(());
    }

    for record in records {
        println!(
            "  service {} engine={} status={} endpoint={}",
            record.service_id, record.engine, record.status, record.endpoint_url
        );
    }

    Ok(())
}

fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    let Some(raw) = value else {
        return Ok(Vec::new());
    };
    match rocm_engine_protocol::GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg)? {
        rocm_engine_protocol::GpuSelection::Auto => Ok(Vec::new()),
        rocm_engine_protocol::GpuSelection::Index(index) => Ok(vec![index]),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn supervise_service(
    paths: &AppPaths,
    service_id: String,
    engine: String,
    model_ref: String,
    canonical_model_id: String,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    device_policy: String,
    gpu: Option<String>,
    engine_recipe_json: Option<String>,
) -> Result<()> {
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(&engine))?;
    fs::create_dir_all(paths.engine_state_dir(&engine))?;
    fs::create_dir_all(paths.services_dir())?;

    let gpu_indices = parse_gpu_indices_arg(gpu.as_deref())?;
    let _ = daemon_binary_path();

    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine.clone(),
        model_ref,
        canonical_model_id.clone(),
        host,
        port,
        "managed",
        std::process::id(),
        runtime_id.clone(),
        env_id.clone(),
        Some(device_policy.clone()),
    );
    // Pair the recorded supervisor PID with the kernel's start-time for it.
    // Without that token a later stop has no way to tell this process from an
    // unrelated one that inherited the PID, and falls back to signalling blind.
    record.supervisor_start_ticks = rocm_core::process_start_ticks(std::process::id());
    record.gpu_indices = gpu_indices;
    record.engine_recipe_json = engine_recipe_json.clone();
    // Carried over from whatever is on disk. `ManagedServiceRecord::new` starts
    // this false, so rebuilding a record here without restoring it would not
    // just skip the check now — it would write the weakened record back and
    // disarm every later `rocm services restart` as well.
    //
    // Propagated, not defaulted. This read arms the guard below, so it is not
    // best-effort the way an identical-looking call feeding a printed warning
    // would be. `load_managed_services` already *skips* unparseable records, so
    // an `Err` here is a real I/O failure — and a missing directory is `Ok`
    // anyway. Swallowing it would say "no service ever required a key", the
    // key-file fallback is false precisely when a service has been stopped, and
    // the weakened record would then be written back at the bottom of this
    // function. That is the outcome the comment above says must not happen.
    let previously_required = crate::persistence::load_managed_services(paths)
        .context(
            "could not read the service registry to check whether this service requires an \
             endpoint API key; refusing to recover it rather than assume it does not",
        )?
        .iter()
        .any(|existing| existing.service_id == record.service_id && existing.requires_api_key);
    // Only what the registry recorded. The `|| key-file-is-present` clause that
    // used to be here re-derived the flag the same way `spawn_managed_engine_child`
    // did, and was wrong for the same reason: a public bind always has a key file
    // whether or not auth was ever demanded, so recovery re-armed this on services
    // that never asked for it and refused them with the wrong remediation.
    record.requires_api_key = previously_required;
    // Refuse a keyless public respawn before the manifest write, so a refused
    // attempt leaves the recorded restart_count and timestamps intact instead of
    // clobbering them with a record no live process will ever back. The spawn
    // site below re-checks against the key actually threaded onto the command.
    crate::common::ensure_public_service_has_endpoint_key(
        &record.host,
        rocm_engine_protocol::endpoint_key_file_if_present(paths, &record.service_id)
            .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
            .is_some(),
        record.requires_api_key,
    )?;
    record.write()?;

    let log_file = fs::File::create(&record.log_path)
        .with_context(|| format!("failed to create {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    let rocm_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let mut command = ProcessCommand::new(rocm_binary);
    command
        .args(engine_serve_http_args(
            &engine,
            &record.service_id,
            &canonical_model_id,
            &record.host,
            record.port,
            &device_policy,
            &record.gpu_indices,
            runtime_id.as_deref(),
            env_id.as_deref(),
            engine_recipe_json.as_deref(),
            &record.engine_state_path,
        ))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err));
    // Re-thread the endpoint key file (public bind only) onto the engine child,
    // same as the initial `rocm serve` spawn. This path also runs on daemon
    // recovery (`restart_managed_service` re-execs `rocmd supervise`), so
    // without this a previously-authenticated public service would come back
    // up anonymous after a crash/recover cycle.
    // If the key is gone the child would listen on the recorded public host with
    // no auth, so fail closed instead — an unreachable service is recoverable,
    // an anonymous public one is not.
    let endpoint_key_applied =
        crate::common::apply_endpoint_key_env(&mut command, paths, &record.service_id);
    crate::common::ensure_public_service_has_endpoint_key(
        &record.host,
        endpoint_key_applied,
        record.requires_api_key,
    )?;
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn engine supervisor child for {engine}"))?;

    record.engine_pid = Some(child.id());
    // Captured while the child is known alive, so a later stop verifies this
    // exact process instead of whatever has since inherited its PID.
    record.engine_start_ticks = rocm_core::process_start_ticks(child.id());
    record.status = "running".to_owned();
    record.write()?;

    // Clone the fields the poller reads so the `on_phase` closure can borrow
    // `record` mutably to persist each startup-phase transition to disk.
    let ready_engine = record.engine.clone();
    let ready_service_id = record.service_id.clone();
    let ready_log_path = record.log_path.clone();
    let became_ready = wait_for_service_ready(
        paths,
        &ready_engine,
        &ready_service_id,
        &ready_log_path,
        Duration::from_mins(3),
        |phase| {
            record.startup_phase = Some(phase.to_owned());
            let _ = record.write();
        },
    );
    if became_ready {
        record.status = "ready".to_owned();
        // The phase only describes the coming-up window; clear it once ready.
        record.startup_phase = None;
        record.write()?;
    }

    let exit_status = child.wait().context("failed waiting for engine child")?;
    record.status = if exit_status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;

    if exit_status.success() {
        Ok(())
    } else {
        std::process::exit(exit_status.code().unwrap_or(1));
    }
}

#[allow(clippy::too_many_arguments)]
fn engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &str,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe_json: Option<&str>,
    state_path: &Path,
) -> Vec<String> {
    let mut args = vec![
        "__engine-serve-http".to_owned(),
        engine.to_owned(),
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--device-policy".to_owned(),
        device_policy.to_owned(),
    ];
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(crate::common::optional_arg("--runtime-id", runtime_id));
    args.extend(crate::common::optional_arg("--env-id", env_id));
    args.extend(crate::common::optional_arg(
        "--engine-recipe-json",
        engine_recipe_json,
    ));
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    args
}

fn build_runtime_state(
    config: &RocmCliConfig,
    automations_enabled: bool,
) -> AutomationRuntimeState {
    let now = unix_time_millis();
    let active_watchers = builtin_watchers()
        .iter()
        .map(|watcher| WatcherRuntimeSnapshot {
            id: watcher.id.to_owned(),
            enabled: config.watcher_enabled(watcher),
            mode: config.effective_watcher_mode(watcher),
            summary: watcher.summary.to_owned(),
            last_event: None,
            last_event_unix_ms: None,
        })
        .collect();
    AutomationRuntimeState {
        running: automations_enabled,
        automations_enabled,
        daemon_pid: std::process::id(),
        started_at_unix_ms: now,
        last_tick_unix_ms: now,
        local_webhook_endpoint: None,
        active_watchers,
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

/// Keep only the final visible segment of a `\r`-redrawn progress line.
///
/// Progress tools (pip, tqdm, Hugging Face) redraw a line in place with a bare
/// carriage return and no newline, so the segment after the last `\r` is its
/// final visible state. Lines without `\r` pass through unchanged. (Same
/// collapse rule the dashboard job console applies to streamed job output.)
fn last_cr_segment(line: &str) -> &str {
    line.rsplit('\r').next().unwrap_or(line)
}

/// Classify a single serve-log line into a coarse startup phase token
/// (`downloading`/`loading`/`warmup`), or `None` when the line carries no phase
/// signal. Case-insensitive substring match over the common vLLM / llama.cpp /
/// Hugging Face startup vocabulary. Checked warmup → loading → downloading so
/// the latest lifecycle stage a line mentions wins.
fn classify_startup_phase(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("capturing cuda graph")
        || lower.contains("capturing the model")
        || lower.contains("warming up")
        || lower.contains("warmup")
    {
        Some("warmup")
    } else if lower.contains("loading weights")
        || lower.contains("loading model")
        || lower.contains("load_tensors")
        || lower.contains("llama_model_loader")
        || lower.contains("model loading took")
    {
        Some("loading")
    } else if lower.contains("downloading") || lower.contains("fetching") {
        Some("downloading")
    } else {
        None
    }
}

/// Read log bytes appended since `*pos`, advance `*pos`, and return the most
/// recent recognizable startup phase in that new output (later lines win, so a
/// download → load → warmup progression advances naturally).
///
/// Best-effort: any I/O error (file not created yet, transient read) yields
/// `None`. A shrunk file (rotation/truncation) resets the cursor to the top.
fn read_new_log_phase(log_path: &Path, pos: &mut u64) -> Option<&'static str> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(log_path).ok()?;
    let len = file.metadata().ok()?.len();
    if len < *pos {
        *pos = 0;
    }
    if len == *pos {
        return None;
    }
    file.seek(SeekFrom::Start(*pos)).ok()?;
    let mut bytes = Vec::new();
    let read = file.read_to_end(&mut bytes).ok()?;
    *pos += read as u64;
    let text = String::from_utf8_lossy(&bytes);
    let mut phase = None;
    for line in text.lines() {
        if let Some(found) = classify_startup_phase(last_cr_segment(line)) {
            phase = Some(found);
        }
    }
    phase
}

fn engine_healthcheck_ready(paths: &AppPaths, engine: &str, service_id: &str) -> Result<bool> {
    Ok(crate::common::healthcheck_response_ready(
        &crate::common::engine_healthcheck_response(paths, engine, service_id)?,
    ))
}

/// Poll a freshly-spawned service until its healthcheck reports ready (or the
/// timeout elapses), tailing its log file meanwhile and reporting each coarse
/// startup phase transition via `on_phase`.
fn wait_for_service_ready(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    log_path: &Path,
    timeout: Duration,
    mut on_phase: impl FnMut(&str),
) -> bool {
    let start = std::time::Instant::now();
    let mut log_pos: u64 = 0;
    let mut last_phase: Option<&'static str> = None;
    while start.elapsed() < timeout {
        if let Some(phase) = read_new_log_phase(log_path, &mut log_pos)
            && last_phase != Some(phase)
        {
            last_phase = Some(phase);
            on_phase(phase);
        }
        if engine_healthcheck_ready(paths, engine, service_id).unwrap_or(false) {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_app_paths, unique_test_root};
    #[cfg(unix)]
    use crate::watchers::load_service_record;

    /// Drive `supervise_service` far enough to reach the key guard, and return
    /// what it did.
    ///
    /// The guard sits before the manifest write and well before any spawn, so a
    /// refusal returns without starting a process — which is what makes the real
    /// call site testable at all. The arguments below are the shape a recovery
    /// re-exec passes: a loopback bind, no GPU, no recipe.
    ///
    /// This exists because testing `ensure_public_service_has_endpoint_key`
    /// directly with literal arguments cannot catch the defect that actually
    /// happened twice in this crate's history — the guard being *wired up* with
    /// the wrong value at its call site.
    ///
    /// Bounded, and the bound is the assertion. A guard that fails to refuse
    /// does not return an error — it falls through to the engine spawn and
    /// supervises a child that never exits, so an unbounded call would hang the
    /// suite instead of failing it. Both callers below are regression tests for
    /// a fail-*open*, which is exactly the shape that turns into a hang.
    fn supervise_at_the_guard(paths: &AppPaths, service_id: &str) -> Result<()> {
        let paths = paths.clone();
        let service_id = service_id.to_owned();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = supervise_service(
                &paths,
                service_id,
                "llamacpp".to_owned(),
                "a-model".to_owned(),
                "a-model".to_owned(),
                None,
                None,
                "127.0.0.1".to_owned(),
                11434,
                "gpu_required".to_owned(),
                None,
                None,
            );
            let _ = sender.send(outcome.map_err(|error| format!("{error:#}")));
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .unwrap_or_else(|_| {
                panic!(
                    "supervise_service did not return within 30s: the key guard let the call \
                     through and it reached the engine spawn, which is the fail-open this test \
                     exists to catch"
                )
            })
            .map_err(anyhow::Error::msg)
    }

    /// Write a service record into the registry the way a live service would
    /// have left it behind.
    fn seed_registry(paths: &AppPaths, service_id: &str, requires_api_key: bool) {
        fs::create_dir_all(paths.services_dir()).unwrap();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id.to_owned(),
            "llamacpp".to_owned(),
            "a-model".to_owned(),
            "a-model".to_owned(),
            "127.0.0.1".to_owned(),
            11434,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.requires_api_key = requires_api_key;
        record.write().unwrap();
    }

    /// Drive `supervise_service` and report whether the key guard let it past.
    ///
    /// Decided on what the call *returns*, not on any file. The obvious
    /// observable — the manifest appearing — is useless here, because
    /// `seed_registry` has already written one, so polling for it passes
    /// whatever the guard does. That mistake was made first and caught by
    /// mutating the code the test claims to protect.
    ///
    /// A call the guard admits does not return: it carries on to the engine
    /// spawn. So the guard's refusal is the only thing that comes back quickly,
    /// and it is identified by its message rather than by the mere fact of an
    /// error — a later, unrelated failure must not read as a refusal.
    fn guard_admits(paths: &AppPaths, service_id: &str) -> bool {
        let owned_paths = paths.clone();
        let owned_id = service_id.to_owned();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = supervise_service(
                &owned_paths,
                owned_id,
                "llamacpp".to_owned(),
                "a-model".to_owned(),
                "a-model".to_owned(),
                None,
                None,
                "127.0.0.1".to_owned(),
                11434,
                "gpu_required".to_owned(),
                None,
                None,
            );
            let _ = sender.send(outcome.map_err(|error| format!("{error:#}")));
        });

        match receiver.recv_timeout(std::time::Duration::from_secs(10)) {
            // The key guard refused, by its own words.
            Ok(Err(rendered)) if rendered.contains("without authentication") => false,
            // Anything else means it got past the guard: it either finished, or
            // failed later for a reason that is not this guard, or is still
            // running because it reached the spawn.
            _ => true,
        }
    }

    #[test]
    fn a_service_that_never_required_a_key_is_not_refused_for_lacking_one() {
        // The other direction of the guard, and the one no test covered.
        // Hardcoding `record.requires_api_key = true` at the restore site passes
        // every other test in this crate, because they all seed a service that
        // *does* require a key. This is the case that catches it.
        //
        // Two records are seeded, not one: with a single record the
        // `existing.service_id == record.service_id` half of the lookup does
        // nothing, so dropping that comparison would go unnoticed and one
        // service's requirement would leak onto another's.
        let (root, paths) = temp_app_paths("supervise-no-key-needed");
        seed_registry(&paths, "svc-needs-key", true);
        seed_registry(&paths, "svc-plain", false);

        assert!(
            guard_admits(&paths, "svc-plain"),
            "a loopback service that never asked for a key must not be refused for lacking one"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn supervising_a_service_that_required_a_key_refuses_when_the_key_is_gone() {
        // The real call site, not the guard in isolation. `supervise_service`
        // rebuilds the record with `ManagedServiceRecord::new`, which starts
        // `requires_api_key` false, and restores it from the registry. Passing
        // the wrong value here — a literal, or the freshly-built field before it
        // is restored — is exactly the miswiring that shipped twice in this
        // crate and that a literal-argument unit test cannot see.
        let (root, paths) = temp_app_paths("supervise-requires-key");
        seed_registry(&paths, "svc-needs-key", true);

        let error = supervise_at_the_guard(&paths, "svc-needs-key")
            .expect_err("a service that required a key must not be recovered without one");
        assert!(
            format!("{error:#}").contains("without authentication"),
            "{error:#}"
        );

        // And the refusal must not have weakened what is on disk. The guard runs
        // before `record.write()` precisely so a refused attempt leaves the
        // recorded requirement armed for the next attempt.
        let stored = crate::persistence::load_managed_services(&paths).unwrap();
        let stored = stored
            .iter()
            .find(|candidate| candidate.service_id == "svc-needs-key")
            .expect("the seeded record must survive a refused recovery");
        assert!(
            stored.requires_api_key,
            "a refused recovery must not disarm the requirement"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_registry_that_cannot_be_read_refuses_recovery_rather_than_assuming_no_key() {
        // `load_managed_services` already skips records it cannot parse, so an
        // `Err` from it is a real I/O failure — and a missing directory is `Ok`.
        // Defaulting it away therefore says "no service ever required a key",
        // which is fail-open on an auth gate and, worse, gets written back.
        //
        // The failure is provoked portably: a directory named like a record
        // makes the `fs::read` inside the loop fail rather than the read_dir.
        let (root, paths) = temp_app_paths("supervise-unreadable-registry");
        fs::create_dir_all(paths.services_dir().join("not-a-record.json")).unwrap();

        let error = supervise_at_the_guard(&paths, "svc-unknown")
            .expect_err("an unreadable registry must refuse, not assume no key was required");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("could not read the service registry"),
            "{rendered}"
        );

        // Nothing was written: a registry we could not read is not a registry we
        // may add a weakened record to.
        assert!(
            !paths.service_manifest_path("svc-unknown").exists(),
            "a refused recovery must not persist a record"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn last_cr_segment_keeps_final_progress_redraw() {
        // A tqdm/HF-style in-place redraw collapses to its last segment.
        assert_eq!(
            last_cr_segment("Downloading:  10%\rDownloading:  55%\rDownloading: 100%"),
            "Downloading: 100%"
        );
        // A plain line is unchanged.
        assert_eq!(
            last_cr_segment("Loading model weights"),
            "Loading model weights"
        );
    }

    #[test]
    fn classify_startup_phase_maps_engine_vocabulary() {
        assert_eq!(
            classify_startup_phase("Downloading shards: 100%"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("Fetching 12 files"),
            Some("downloading")
        );
        assert_eq!(
            classify_startup_phase("INFO: Loading model weights took 4.2s"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("llama_model_loader: loaded meta data"),
            Some("loading")
        );
        assert_eq!(
            classify_startup_phase("Capturing CUDA graph shapes"),
            Some("warmup")
        );
        assert_eq!(
            classify_startup_phase("Warming up the engine"),
            Some("warmup")
        );
        // Ordinary chatter carries no phase signal.
        assert_eq!(
            classify_startup_phase("Uvicorn running on http://..."),
            None
        );
    }

    #[test]
    fn classify_startup_phase_emits_only_dashboard_known_tokens() {
        // These tokens are the wire contract with the dashboard's
        // `StartupPhase::from_token` (rocm-dash-core); emitting anything else
        // would be silently dropped there. rocmd can't link that crate, so the
        // contract is pinned here by literal.
        for line in [
            "Downloading shards",
            "Loading model weights",
            "Capturing CUDA graph",
        ] {
            let token = classify_startup_phase(line).expect("line is a phase signal");
            assert!(
                matches!(token, "downloading" | "loading" | "warmup"),
                "token {token:?} must be one the dashboard understands"
            );
        }
    }

    #[test]
    fn read_new_log_phase_advances_and_tracks_latest() {
        use std::io::Write as _;
        // Workspace-local test root (rooted at CARGO_MANIFEST_DIR, not the
        // ambient temp dir) — same helper the other rocmd tests use.
        let dir = unique_test_root(&format!("rocmd-phase-{}", std::process::id()));
        let log = dir.join("svc.log");
        std::fs::write(&log, "boot\nDownloading shards: 100%\n").unwrap();

        let mut pos = 0_u64;
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("downloading"));
        // No new bytes → no phase, cursor unchanged.
        let after_first = pos;
        assert_eq!(read_new_log_phase(&log, &mut pos), None);
        assert_eq!(pos, after_first);

        // Appending a later stage advances the phase.
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "Loading model weights took 3s").unwrap();
        assert_eq!(read_new_log_phase(&log, &mut pos), Some("loading"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_serve_http_args_forward_engine_recipe_json() {
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            Some("therock-release:gfx120X-all"),
            Some("env-1"),
            Some(engine_recipe_json),
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2).any(|pair| {
                pair[0] == "--runtime-id" && pair[1] == "therock-release:gfx120X-all"
            })
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--state-path" && pair[1] == "state.json")
        );
    }

    #[test]
    fn engine_serve_http_args_emit_gpu_indices_when_pinned() {
        let args = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[1],
            None,
            None,
            None,
            Path::new("state.json"),
        );

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--gpu" && pair[1] == "1")
        );

        let auto = engine_serve_http_args(
            "vllm",
            "svc-1",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "gpu_required",
            &[],
            None,
            None,
            None,
            Path::new("state.json"),
        );
        assert!(!auto.iter().any(|arg| arg == "--gpu"));
    }

    #[tokio::test]
    async fn local_webhook_requires_enabled_automation_loop() {
        let (_root, paths) = temp_app_paths("local-webhook-requires-loop");
        let error = run_daemon(&paths, false, Some(0)).await.unwrap_err();

        assert!(error.to_string().contains("requires --automations-enabled"));
    }

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
        let reloaded = crate::watchers::load_service_record(&paths, service_id);
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
}
