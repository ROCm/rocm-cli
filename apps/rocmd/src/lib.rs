// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#![allow(clippy::items_after_test_module)]

mod cli;
mod common;
mod mcp;
mod persistence;
mod sandbox;
#[cfg(test)]
mod test_support;
mod webhook;

pub use cli::{run_bin_cli, run_from_args};

use anyhow::{Context, Result, bail};
#[cfg(test)]
use rocm_core::AuditEventRecord;
#[cfg(test)]
use rocm_core::AutomationEventRecord;
use rocm_core::{
    AppPaths, AutomationProposalRecord, AutomationRuntimeState, AutomationTriggerEvent,
    CodexBridgeGpuSnapshot, ManagedServiceRecord, RocmCliConfig, WatcherMode,
    WatcherRuntimeSnapshot, append_automation_proposal, builtin_watchers, daemon_binary_path,
    resolve_model_recipe_artifact, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const SERVER_TRANSIENT_STALE_MS: u128 = 5 * 60 * 1_000;
const ENDPOINT_HEALTH_TIMEOUT: Duration = Duration::from_millis(250);
const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;
const GPU_METRICS_INTERVAL_MS: u128 = 60 * 1000;
const GPU_THERMAL_HOTSPOT_PRESSURE_C: f64 = 95.0;
const GPU_THERMAL_MEMORY_PRESSURE_C: f64 = 95.0;
const GPU_MEMORY_VRAM_PRESSURE_PERCENT: f64 = 95.0;
const ARTIFACT_PREFETCH_TIMEOUT: Duration = Duration::from_mins(10);

fn stop_managed_service(paths: &AppPaths, service_id: &str) -> Result<Value> {
    let mut record = persistence::load_managed_services(paths)?
        .into_iter()
        .find(|record| record.service_id == service_id)
        .with_context(|| format!("managed service `{service_id}` not found"))?;
    let mut signaled_pids = Vec::new();
    let mut skipped_pids = Vec::new();
    let mut root_pids = Vec::new();
    if let Some(engine_pid) = record.engine_pid
        && engine_pid != 0
    {
        if engine_pid == std::process::id() {
            skipped_pids.push(engine_pid);
        } else {
            root_pids.push(engine_pid);
        }
    }
    if record.supervisor_pid != 0
        && record.supervisor_pid != std::process::id()
        && Some(record.supervisor_pid) != record.engine_pid
    {
        root_pids.push(record.supervisor_pid);
    }
    let mut pids_to_signal = descendant_pids_for_roots(&root_pids)?;
    pids_to_signal.extend(root_pids);
    let mut seen_pids = HashSet::new();
    for pid in pids_to_signal {
        if !seen_pids.insert(pid) {
            continue;
        }
        if terminate_process(pid)? {
            signaled_pids.push(pid);
        } else {
            skipped_pids.push(pid);
        }
    }
    let force_signaled_pids = force_terminate_remaining_processes(&signaled_pids)?;
    record.status = "stopped".to_owned();
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
    }))
}

#[cfg(unix)]
fn descendant_pids_for_roots(root_pids: &[u32]) -> Result<Vec<u32>> {
    if root_pids.is_empty() {
        return Ok(Vec::new());
    }
    let output = ProcessCommand::new("ps")
        .args(["-eo", "pid=,ppid="])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to list process tree with ps")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        bail!(
            "failed to list process tree: {}",
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }
    let output = String::from_utf8_lossy(&output.stdout);
    Ok(descendant_pids_from_ps_output(&output, root_pids))
}

#[cfg(not(unix))]
fn descendant_pids_for_roots(_root_pids: &[u32]) -> Result<Vec<u32>> {
    Ok(Vec::new())
}

#[cfg(any(unix, test))]
fn descendant_pids_from_ps_output(output: &str, root_pids: &[u32]) -> Vec<u32> {
    fn append_descendants(
        parent: u32,
        processes: &[(u32, u32)],
        seen: &mut HashSet<u32>,
        output: &mut Vec<u32>,
    ) {
        for (pid, ppid) in processes {
            if *ppid != parent || *pid == parent || !seen.insert(*pid) {
                continue;
            }
            append_descendants(*pid, processes, seen, output);
            output.push(*pid);
        }
    }

    let processes = output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let pid = parts.next()?.parse::<u32>().ok()?;
            let ppid = parts.next()?.parse::<u32>().ok()?;
            Some((pid, ppid))
        })
        .collect::<Vec<_>>();
    let mut seen = root_pids.iter().copied().collect::<HashSet<_>>();
    let mut descendants = Vec::new();
    for root in root_pids {
        append_descendants(*root, &processes, &mut seen, &mut descendants);
    }
    descendants
}

fn terminate_process(pid: u32) -> Result<bool> {
    if pid == std::process::id() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        let output = ProcessCommand::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch kill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if stderr.contains("No such process") || stdout.contains("No such process") {
                return Ok(false);
            }
            bail!(
                "failed to signal pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
    #[cfg(windows)]
    {
        let output = ProcessCommand::new("taskkill")
            .arg("/PID")
            .arg(pid.to_string())
            .arg("/T")
            .arg("/F")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch taskkill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if stderr.contains("not found") || stdout.contains("not found") {
                return Ok(false);
            }
            bail!(
                "failed to stop pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
}

#[cfg(unix)]
fn force_terminate_remaining_processes(pids: &[u32]) -> Result<Vec<u32>> {
    if pids.is_empty() {
        return Ok(Vec::new());
    }
    thread::sleep(Duration::from_millis(750));
    let mut force_signaled = Vec::new();
    for pid in pids {
        if *pid == std::process::id() || !process_is_running(*pid)? {
            continue;
        }
        let output = ProcessCommand::new("kill")
            .arg("-KILL")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch kill -KILL for pid {pid}"))?;
        if output.status.success() {
            force_signaled.push(*pid);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "failed to force stop pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            );
        }
    }
    Ok(force_signaled)
}

#[cfg(not(unix))]
fn force_terminate_remaining_processes(_pids: &[u32]) -> Result<Vec<u32>> {
    Ok(Vec::new())
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> Result<bool> {
    let output = ProcessCommand::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch kill -0 for pid {pid}"))?;
    Ok(output.status.success())
}

async fn run_daemon(
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
        Some(webhook::start_local_webhook_source(port).await?)
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
    persistence::record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let mut ticker = time::interval(WATCHER_TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let config = RocmCliConfig::load(paths)?;
                reconcile_watcher_snapshots(&config, &mut state);
                evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            event = webhook::receive_local_webhook_event(&mut local_webhook_receiver) => {
                if let Some(event) = event {
                    let config = RocmCliConfig::load(paths)?;
                    reconcile_watcher_snapshots(&config, &mut state);
                    persistence::record_event(
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
                        evaluate_watchers_for_events(paths, &config, &mut state, &[event])
                    {
                        persistence::record_event(
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
                    persistence::record_event(
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
    persistence::record_event(
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

fn print_status(paths: &AppPaths) -> Result<()> {
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

    let records = persistence::load_managed_services(paths)?;
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
fn supervise_service(
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
    let previously_required = persistence::load_managed_services(paths)
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
    common::ensure_public_service_has_endpoint_key(
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
        common::apply_endpoint_key_env(&mut command, paths, &record.service_id);
    common::ensure_public_service_has_endpoint_key(
        &record.host,
        endpoint_key_applied,
        record.requires_api_key,
    )?;
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn engine supervisor child for {engine}"))?;

    record.engine_pid = Some(child.id());
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
    args.extend(common::optional_arg("--runtime-id", runtime_id));
    args.extend(common::optional_arg("--env-id", env_id));
    args.extend(common::optional_arg(
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

fn reconcile_watcher_snapshots(config: &RocmCliConfig, state: &mut AutomationRuntimeState) {
    for watcher in builtin_watchers() {
        match state.watcher_mut(watcher.id) {
            Some(snapshot) => {
                snapshot.enabled = config.watcher_enabled(watcher);
                snapshot.mode = config.effective_watcher_mode(watcher);
                snapshot.summary = watcher.summary.to_owned();
            }
            None => state.active_watchers.push(WatcherRuntimeSnapshot {
                id: watcher.id.to_owned(),
                enabled: config.watcher_enabled(watcher),
                mode: config.effective_watcher_mode(watcher),
                summary: watcher.summary.to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }),
        }
    }
}

fn evaluate_watchers(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let events = collect_automation_events(paths, config, state)?;
    evaluate_watchers_for_events(paths, config, state, &events)
}

fn collect_automation_events(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &AutomationRuntimeState,
) -> Result<Vec<AutomationTriggerEvent>> {
    collect_automation_events_with_gpu_snapshot(paths, state, || {
        common::gather_gpu_snapshot_for_config(config)
    })
}

fn collect_automation_events_with_gpu_snapshot<F>(
    paths: &AppPaths,
    state: &AutomationRuntimeState,
    gpu_snapshot: F,
) -> Result<Vec<AutomationTriggerEvent>>
where
    F: FnMut() -> CodexBridgeGpuSnapshot,
{
    let now = unix_time_millis();
    let mut events = Vec::new();

    if therock_update_due(state, now) {
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({
                "interval_ms": THEROCK_UPDATE_INTERVAL_MS,
            }),
        });
    }

    if server_recover_due(state, now)
        && let Some((record, recovery_reason)) = find_recoverable_service(paths)?
    {
        let kind = service_recovery_event_kind(&recovery_reason);
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: kind.to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some(recovery_reason.clone()),
            payload: json!({
                "engine": record.engine,
                "status": record.status,
                "endpoint": record.endpoint_url,
                "recovery_reason": recovery_reason,
            }),
        });
    }

    let gpu_metrics_due_now = gpu_metrics_due(state, now);
    let gpu_thermal_protect_due_now = gpu_thermal_protect_due(state, now);
    let snapshot = (gpu_metrics_due_now || gpu_thermal_protect_due_now).then(gpu_snapshot);

    if gpu_metrics_due_now {
        let snapshot = snapshot
            .as_ref()
            .expect("GPU snapshot should be collected for due metrics");
        let available = snapshot.amd_smi_available && snapshot.monitor_snapshot.is_some();
        events.push(AutomationTriggerEvent {
            at_unix_ms: now,
            kind: if available {
                "gpu.metrics".to_owned()
            } else {
                "gpu.metrics_unavailable".to_owned()
            },
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-metrics".to_owned()),
            service_id: None,
            reason: if available {
                Some("amd_smi_snapshot_available".to_owned())
            } else {
                snapshot
                    .note
                    .clone()
                    .or_else(|| Some("amd_smi_snapshot_unavailable".to_owned()))
            },
            payload: json!({
                "amd_smi_available": snapshot.amd_smi_available,
                "static_available": snapshot.static_snapshot.is_some(),
                "monitor_available": snapshot.monitor_snapshot.is_some(),
                "summary": gpu_snapshot_summary(snapshot),
                "interval_ms": GPU_METRICS_INTERVAL_MS,
            }),
        });
    }

    if gpu_thermal_protect_due_now && let Some(snapshot) = snapshot.as_ref() {
        events.extend(gpu_pressure_events(now, snapshot));
    }

    Ok(events)
}

fn evaluate_watchers_for_events(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
    events: &[AutomationTriggerEvent],
) -> Result<()> {
    for watcher in builtin_watchers() {
        if !config.watcher_enabled(watcher) {
            continue;
        }
        let mode = config.effective_watcher_mode(watcher);
        match watcher.id {
            "therock-update" => {
                for event in events_for_watcher(events, watcher.id, "schedule.tick") {
                    handle_therock_update_event(paths, mode, state, event)?;
                }
            }
            "server-recover" => {
                for event in events_for_watcher(events, watcher.id, "service.") {
                    handle_server_recover_event(paths, mode, state, event)?;
                }
            }
            "gpu-metrics" => {
                for event in events_for_watcher(events, watcher.id, "gpu.") {
                    handle_gpu_metrics_event(paths, mode, state, event)?;
                }
            }
            "gpu-thermal-protect" => {
                for event in
                    events_for_watcher_exact(events, watcher.id, "gpu.thermal_pressure").chain(
                        events_for_watcher_exact(events, watcher.id, "gpu.memory_pressure"),
                    )
                {
                    handle_gpu_thermal_protect_event(paths, mode, state, event)?;
                }
            }
            "cache-warm" => {
                for event in events_for_watcher_exact(events, watcher.id, "cache.warm") {
                    handle_cache_warm_event(paths, mode, state, event)?;
                }
            }
            "driver-upgrade" => {
                for event in events_for_watcher_exact(events, watcher.id, "update.available") {
                    handle_driver_upgrade_event(paths, mode, state, event)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn events_for_watcher<'a>(
    events: &'a [AutomationTriggerEvent],
    watcher_id: &str,
    kind_prefix: &str,
) -> impl Iterator<Item = &'a AutomationTriggerEvent> {
    events.iter().filter(move |event| {
        event.watcher_hint.as_deref() == Some(watcher_id) && event.kind.starts_with(kind_prefix)
    })
}

fn events_for_watcher_exact<'a>(
    events: &'a [AutomationTriggerEvent],
    watcher_id: &str,
    kind: &'static str,
) -> impl Iterator<Item = &'a AutomationTriggerEvent> {
    events.iter().filter(move |event| {
        event.watcher_hint.as_deref() == Some(watcher_id) && event.kind == kind
    })
}

fn handle_therock_update_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_therock_update_event_with_runner(paths, mode, state, event, |paths| {
        sandbox::run_sandbox_tool(
            paths,
            cli::SandboxToolArg::CheckUpdates,
            None,
            None,
            None,
            cli::SandboxToolPolicy::default(),
        )
    })
}

fn handle_therock_update_event_with_runner<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    _event: &AutomationTriggerEvent,
    update_runner: F,
) -> Result<()>
where
    F: FnOnce(&AppPaths) -> Result<Value>,
{
    let policy = watcher_policy_action("therock-update", mode);
    let action = match policy {
        WatcherPolicyAction::Observe => "observe_schedule",
        WatcherPolicyAction::QueueProposal => "queue_update_proposal",
        WatcherPolicyAction::RunContained => "run_update_check",
    };
    let message = match policy {
        WatcherPolicyAction::Observe => {
            "scheduled TheRock update check reminder emitted; run `rocm update` to inspect the selected channel"
        }
        WatcherPolicyAction::QueueProposal => {
            "scheduled TheRock update check reminder emitted; queueing read-only update-check proposal for review"
        }
        WatcherPolicyAction::RunContained => {
            "scheduled TheRock update check is approved for contained read-only execution"
        }
    };
    match policy {
        WatcherPolicyAction::Observe | WatcherPolicyAction::QueueProposal => {
            persistence::record_event(
                paths,
                state,
                "therock-update",
                "info",
                action,
                message,
                None,
            )?;
            if policy == WatcherPolicyAction::QueueProposal {
                queue_proposal(
                    paths,
                    "therock-update",
                    action,
                    "Check TheRock updates",
                    "Run `rocm update` to inspect available CLI, runtime, engine, and recipe updates before applying changes.",
                    None,
                )?;
            }
        }
        WatcherPolicyAction::RunContained => match update_runner(paths) {
            Ok(output) => match restricted_check_updates_result(&output) {
                Ok(result) => {
                    persistence::record_event(
                        paths,
                        state,
                        "therock-update",
                        if result.exit_status == 0 {
                            "info"
                        } else {
                            "error"
                        },
                        action,
                        &format!(
                            "{message}; restricted check_updates status={}; {}",
                            result.status,
                            common::update_check_message(result.status)
                        ),
                        None,
                    )?;
                    if result.update_available {
                        record_update_available_notification(paths, state, result.status)?;
                    }
                }
                Err(error) => {
                    persistence::record_event(
                        paths,
                        state,
                        "therock-update",
                        "error",
                        "update_check_failed",
                        &format!(
                            "scheduled TheRock update check failed during contained restricted execution: {error}; no updates were applied"
                        ),
                        None,
                    )?;
                }
            },
            Err(error) => {
                persistence::record_event(
                    paths,
                    state,
                    "therock-update",
                    "error",
                    "update_check_failed",
                    &format!(
                        "scheduled TheRock update check failed during contained read-only execution: {error}; no updates were applied"
                    ),
                    None,
                )?;
            }
        },
    }
    Ok(())
}

struct RestrictedCheckUpdatesResult<'a> {
    status: &'a str,
    update_available: bool,
    exit_status: i64,
}

fn restricted_check_updates_result(value: &Value) -> Result<RestrictedCheckUpdatesResult<'_>> {
    let tool = value
        .get("tool")
        .and_then(Value::as_str)
        .context("restricted update check did not report a tool name")?;
    if tool != cli::SandboxToolArg::CheckUpdates.as_cli_value() {
        bail!("restricted update check returned `{tool}`, expected `check_updates`");
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("checked");
    let update_available = value
        .get("update_available")
        .and_then(Value::as_bool)
        .unwrap_or(matches!(status, "update_available" | "repair_available"));
    let exit_status = value
        .get("exit_status")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| i64::from(status == "error"));
    Ok(RestrictedCheckUpdatesResult {
        status,
        update_available,
        exit_status,
    })
}

fn record_update_available_notification(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
    status: &str,
) -> Result<()> {
    let message = if status == "repair_available" {
        "A ROCm runtime repair is available because its package composition changed. Preview it before applying. No updates were applied."
    } else {
        "A ROCm runtime update is available. Preview it before applying. No updates were applied."
    };
    persistence::record_event(
        paths,
        state,
        "therock-update",
        "info",
        "notify_if_newer",
        message,
        None,
    )?;
    sandbox::record_notification_audit(
        paths,
        "watcher:therock-update",
        "notify_if_newer",
        Some("therock-update"),
        message,
    )
}

fn therock_update_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "therock-update" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= THEROCK_UPDATE_INTERVAL_MS)
}

fn gpu_metrics_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "gpu-metrics" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= GPU_METRICS_INTERVAL_MS)
}

fn gpu_thermal_protect_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "gpu-thermal-protect" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= GPU_METRICS_INTERVAL_MS)
}

fn handle_gpu_metrics_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let summary = event
        .payload
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("summary unavailable");
    let level = if event.kind == "gpu.metrics" {
        "info"
    } else {
        "warn"
    };
    let mode_note = match mode {
        WatcherMode::Observe => "observe mode records telemetry only",
        WatcherMode::Propose => {
            "propose mode has no GPU mutation policy yet, so telemetry is recorded only"
        }
        WatcherMode::Contained => {
            "contained mode has no GPU mutation policy yet, so telemetry is recorded only"
        }
    };
    let reason = event
        .reason
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("no detail");
    let source = match event.source.as_str() {
        "gpu_telemetry" => "local amd-smi telemetry",
        "local_webhook" => "local webhook",
        other => other,
    };

    persistence::record_event(
        paths,
        state,
        "gpu-metrics",
        level,
        "record_gpu_metrics",
        &format!(
            "GPU metrics event from {source}: {summary}; reason={reason}; {mode_note}; no mutating action was taken"
        ),
        None,
    )
}

fn gpu_snapshot_summary(snapshot: &CodexBridgeGpuSnapshot) -> String {
    let mut parts = Vec::new();
    parts.push(format!("amd_smi_available={}", snapshot.amd_smi_available));
    parts.push(format!(
        "static_snapshot={}",
        if snapshot.static_snapshot.is_some() {
            "available"
        } else {
            "missing"
        }
    ));
    parts.push(format!(
        "monitor_snapshot={}",
        if snapshot.monitor_snapshot.is_some() {
            "available"
        } else {
            "missing"
        }
    ));
    if let Some(count) = snapshot.static_snapshot.as_ref().and_then(gpu_data_count) {
        parts.push(format!("gpu_count={count}"));
    }
    if let Some(note) = snapshot.note.as_deref()
        && !note.trim().is_empty()
    {
        parts.push(format!("note={note}"));
    }
    parts.join(" ")
}

fn gpu_data_count(value: &Value) -> Option<usize> {
    value
        .get("gpu_data")
        .and_then(Value::as_array)
        .map(Vec::len)
}

#[derive(Debug, Clone, Copy)]
struct GpuPressureReading {
    gpu_index: Option<u64>,
    hotspot_temperature_c: Option<f64>,
    memory_temperature_c: Option<f64>,
    vram_percent: Option<f64>,
}

fn gpu_pressure_events(
    now: u128,
    snapshot: &CodexBridgeGpuSnapshot,
) -> Vec<AutomationTriggerEvent> {
    let Some(monitor_snapshot) = snapshot.monitor_snapshot.as_ref() else {
        return Vec::new();
    };
    monitor_entries(monitor_snapshot)
        .into_iter()
        .filter_map(gpu_pressure_reading)
        .filter_map(|reading| gpu_pressure_event(now, reading))
        .collect()
}

fn gpu_pressure_event(now: u128, reading: GpuPressureReading) -> Option<AutomationTriggerEvent> {
    let (kind, reason, metric_label, value, threshold) = if let Some(value) =
        reading.hotspot_temperature_c
        && value >= GPU_THERMAL_HOTSPOT_PRESSURE_C
    {
        (
            "gpu.thermal_pressure",
            "hotspot_temperature_threshold",
            "hotspot temperature",
            value,
            GPU_THERMAL_HOTSPOT_PRESSURE_C,
        )
    } else if let Some(value) = reading.memory_temperature_c
        && value >= GPU_THERMAL_MEMORY_PRESSURE_C
    {
        (
            "gpu.thermal_pressure",
            "memory_temperature_threshold",
            "memory temperature",
            value,
            GPU_THERMAL_MEMORY_PRESSURE_C,
        )
    } else if let Some(value) = reading.vram_percent
        && value >= GPU_MEMORY_VRAM_PRESSURE_PERCENT
    {
        (
            "gpu.memory_pressure",
            "vram_pressure_threshold",
            "VRAM use",
            value,
            GPU_MEMORY_VRAM_PRESSURE_PERCENT,
        )
    } else {
        return None;
    };
    let gpu_label = reading
        .gpu_index
        .map_or_else(|| "the GPU".to_owned(), |gpu| format!("GPU {gpu}"));
    let unit = if metric_label == "VRAM use" {
        "%"
    } else {
        " C"
    };
    let summary = format!(
        "{gpu_label} {metric_label} is {}{} (limit {}{})",
        display_metric(value),
        unit,
        display_metric(threshold),
        unit
    );
    Some(AutomationTriggerEvent {
        at_unix_ms: now,
        kind: kind.to_owned(),
        source: "gpu_telemetry".to_owned(),
        watcher_hint: Some("gpu-thermal-protect".to_owned()),
        service_id: None,
        reason: Some(reason.to_owned()),
        payload: json!({
            "gpu": reading.gpu_index,
            "hotspot_temperature_c": reading.hotspot_temperature_c,
            "memory_temperature_c": reading.memory_temperature_c,
            "vram_percent": reading.vram_percent,
            "hotspot_threshold_c": GPU_THERMAL_HOTSPOT_PRESSURE_C,
            "memory_temperature_threshold_c": GPU_THERMAL_MEMORY_PRESSURE_C,
            "vram_threshold_percent": GPU_MEMORY_VRAM_PRESSURE_PERCENT,
            "recommended_action": "stop_serving_load",
            "summary": summary,
        }),
    })
}

fn monitor_entries(value: &Value) -> Vec<&Value> {
    if let Some(entries) = value.as_array() {
        return entries.iter().collect();
    }
    value
        .get("gpu_data")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().collect())
        .unwrap_or_default()
}

fn gpu_pressure_reading(entry: &Value) -> Option<GpuPressureReading> {
    let reading = GpuPressureReading {
        gpu_index: metric_u64(entry, &["gpu", "gpu_id", "gpu_index"]),
        hotspot_temperature_c: metric_f64(
            entry,
            &[
                "hotspot_temperature",
                "hotspot_temperature_c",
                "temperature_hotspot",
            ],
        ),
        memory_temperature_c: metric_f64(
            entry,
            &[
                "memory_temperature",
                "memory_temperature_c",
                "temperature_memory",
            ],
        ),
        vram_percent: metric_f64(
            entry,
            &["vram_percent", "vram_usage_percent", "vram_used_percent"],
        ),
    };
    (reading.hotspot_temperature_c.is_some()
        || reading.memory_temperature_c.is_some()
        || reading.vram_percent.is_some())
    .then_some(reading)
}

fn metric_f64(entry: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| entry.get(*key).and_then(value_as_metric_f64))
}

fn metric_u64(entry: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| entry.get(*key).and_then(value_as_metric_u64))
}

fn value_as_metric_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        Value::Object(map) => map
            .get("value")
            .or_else(|| map.get("val"))
            .and_then(value_as_metric_f64),
        _ => None,
    }
}

fn value_as_metric_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        Value::Object(map) => map
            .get("value")
            .or_else(|| map.get("val"))
            .and_then(value_as_metric_u64),
        _ => None,
    }
}

fn display_metric(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn handle_gpu_thermal_protect_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let summary = payload_string(&event.payload, "summary")
        .unwrap_or_else(|| "GPU pressure is high".to_owned());
    let reason = event.reason.as_deref().unwrap_or("gpu_pressure_threshold");

    if matches!(mode, WatcherMode::Observe) {
        return persistence::record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "warn",
            "observe_gpu_pressure",
            &format!(
                "{summary}; observe mode records this only and does not stop any model server"
            ),
            event.service_id.clone(),
        );
    }

    let Some(record) = resolve_gpu_pressure_service_target(paths, event)? else {
        return persistence::record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "warn",
            "gpu_pressure_no_clear_target",
            &format!(
                "{summary}; rocm-cli did not choose a model server to stop because there was no single clear running managed server"
            ),
            None,
        );
    };

    if pending_stop_proposal_exists(paths, &record.service_id)? {
        return persistence::record_event(
            paths,
            state,
            "gpu-thermal-protect",
            "info",
            "stop_proposal_already_pending",
            &format!(
                "{summary}; a reviewed stop request is already waiting for {}",
                record.service_id
            ),
            Some(record.service_id),
        );
    }

    let action = "queue_stop_server_proposal";
    let mode_note = if matches!(mode, WatcherMode::Contained) {
        "contained mode still asks before stopping anything"
    } else {
        "asking before stopping anything"
    };
    let message = format!(
        "{summary}; {mode_note}; selected managed server {} ({})",
        record.service_id, record.endpoint_url
    );
    persistence::record_event(
        paths,
        state,
        "gpu-thermal-protect",
        "warn",
        action,
        &message,
        Some(record.service_id.clone()),
    )?;
    queue_proposal_with_arguments(
        paths,
        "gpu-thermal-protect",
        action,
        "Review GPU pressure stop",
        &message,
        Some(record.service_id.clone()),
        json!({
            "service_id": record.service_id,
            "model_ref": record.model_ref,
            "canonical_model_id": record.canonical_model_id,
            "endpoint_url": record.endpoint_url,
            "engine": record.engine,
            "pressure_kind": event.kind,
            "pressure_reason": reason,
            "pressure_summary": summary,
            "gpu": event.payload.get("gpu").cloned().unwrap_or(Value::Null),
            "hotspot_temperature_c": event.payload.get("hotspot_temperature_c").cloned().unwrap_or(Value::Null),
            "memory_temperature_c": event.payload.get("memory_temperature_c").cloned().unwrap_or(Value::Null),
            "vram_percent": event.payload.get("vram_percent").cloned().unwrap_or(Value::Null),
            "hotspot_threshold_c": GPU_THERMAL_HOTSPOT_PRESSURE_C,
            "memory_temperature_threshold_c": GPU_THERMAL_MEMORY_PRESSURE_C,
            "vram_threshold_percent": GPU_MEMORY_VRAM_PRESSURE_PERCENT,
        }),
    )
}

fn resolve_gpu_pressure_service_target(
    paths: &AppPaths,
    event: &AutomationTriggerEvent,
) -> Result<Option<ManagedServiceRecord>> {
    if let Some(service_id) = event
        .service_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let record = load_service_record(paths, service_id)?;
        return Ok(active_pressure_target(record));
    }

    let active = persistence::load_managed_services(paths)?
        .into_iter()
        .filter_map(active_pressure_target)
        .collect::<Vec<_>>();
    if active.len() == 1 {
        Ok(active.into_iter().next())
    } else {
        Ok(None)
    }
}

fn active_pressure_target(record: ManagedServiceRecord) -> Option<ManagedServiceRecord> {
    (record.mode == "managed" && matches!(record.status.as_str(), "ready" | "running"))
        .then_some(record)
}

fn pending_stop_proposal_exists(paths: &AppPaths, service_id: &str) -> Result<bool> {
    Ok(rocm_core::load_recent_automation_proposals(paths, 100)?
        .into_iter()
        .any(|proposal| {
            proposal.status == "pending"
                && proposal.watcher_id == "gpu-thermal-protect"
                && proposal.service_id.as_deref() == Some(service_id)
                && (proposal.action == "queue_stop_server_proposal"
                    || proposal.tool.as_deref() == Some("stop_server"))
        }))
}

fn handle_cache_warm_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_cache_warm_event_with_resolver(paths, mode, state, event, |artifact_ref| {
        resolve_model_recipe_artifact(artifact_ref).map(|resolved| resolved.is_some())
    })
}

fn handle_cache_warm_event_with_resolver<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    mut artifact_exists: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<bool>,
{
    let Some(artifact_ref) = payload_string(&event.payload, "artifact_ref") else {
        return persistence::record_event(
            paths,
            state,
            "cache-warm",
            "warn",
            "cache_warm_missing_artifact",
            "cache warm event did not include artifact_ref; no prefetch proposal was queued",
            None,
        );
    };
    match artifact_exists(&artifact_ref) {
        Ok(true) => {}
        Ok(false) => {
            return persistence::record_event(
                paths,
                state,
                "cache-warm",
                "warn",
                "cache_warm_unknown_artifact",
                &format!(
                    "cache warm requested unknown registry artifact {artifact_ref}; no prefetch proposal was queued"
                ),
                None,
            );
        }
        Err(error) => {
            return persistence::record_event(
                paths,
                state,
                "cache-warm",
                "error",
                "cache_warm_registry_error",
                &format!(
                    "cache warm could not verify registry artifact {artifact_ref}: {error}; no prefetch proposal was queued"
                ),
                None,
            );
        }
    }
    match mode {
        WatcherMode::Observe => persistence::record_event(
            paths,
            state,
            "cache-warm",
            "info",
            "observe_cache_warm_request",
            &format!(
                "observed cache warm request for {artifact_ref}; observe mode does not queue or download artifacts"
            ),
            None,
        ),
        WatcherMode::Propose | WatcherMode::Contained => {
            let action = "queue_prefetch_proposal";
            let message = if matches!(mode, WatcherMode::Contained) {
                format!(
                    "cache warm requested for {artifact_ref}; contained mode still queues a review because artifact downloads require explicit source-policy approval"
                )
            } else {
                format!(
                    "cache warm requested for {artifact_ref}; queueing a reviewed prefetch proposal"
                )
            };
            persistence::record_event(paths, state, "cache-warm", "info", action, &message, None)?;
            queue_proposal_with_arguments(
                paths,
                "cache-warm",
                action,
                "Prefetch model artifact",
                &message,
                None,
                json!({
                    "artifact_ref": artifact_ref,
                }),
            )
        }
    }
}

fn handle_driver_upgrade_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_driver_upgrade_event_with_runner(paths, mode, state, event, |paths| {
        sandbox::run_sandbox_tool(
            paths,
            cli::SandboxToolArg::DriverPlan,
            None,
            None,
            None,
            cli::SandboxToolPolicy::default(),
        )
    })
}

fn handle_driver_upgrade_event_with_runner<F>(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    driver_plan_runner: F,
) -> Result<()>
where
    F: FnOnce(&AppPaths) -> Result<Value>,
{
    if payload_string(&event.payload, "component").as_deref() != Some("driver") {
        return persistence::record_event(
            paths,
            state,
            "driver-upgrade",
            "warn",
            "driver_upgrade_ignored_component",
            "driver-upgrade event did not include payload.component=driver; no driver plan proposal was queued",
            None,
        );
    }

    match mode {
        WatcherMode::Observe => persistence::record_event(
            paths,
            state,
            "driver-upgrade",
            "info",
            "observe_driver_update",
            "observed local driver update signal; observe mode does not queue or run a driver plan",
            None,
        ),
        WatcherMode::Propose => {
            let action = "prepare_driver_plan";
            let message =
                "local driver update signal received; queueing a reviewed read-only driver plan";
            persistence::record_event(
                paths,
                state,
                "driver-upgrade",
                "info",
                action,
                message,
                None,
            )?;
            queue_proposal(
                paths,
                "driver-upgrade",
                action,
                "Review driver install plan",
                message,
                None,
            )
        }
        WatcherMode::Contained => match driver_plan_runner(paths) {
            Ok(output) => match restricted_driver_plan_result(&output) {
                Ok(result) => persistence::record_event(
                    paths,
                    state,
                    "driver-upgrade",
                    if result.exit_status == 0 {
                        "info"
                    } else {
                        "error"
                    },
                    "run_driver_plan",
                    &format!(
                        "local driver update signal received; contained restricted driver_plan status={}; no driver commands were executed",
                        result.status
                    ),
                    None,
                ),
                Err(error) => persistence::record_event(
                    paths,
                    state,
                    "driver-upgrade",
                    "error",
                    "driver_plan_failed",
                    &format!(
                        "local driver update signal received, but contained restricted driver_plan failed: {error}; no driver commands were executed"
                    ),
                    None,
                ),
            },
            Err(error) => persistence::record_event(
                paths,
                state,
                "driver-upgrade",
                "error",
                "driver_plan_failed",
                &format!(
                    "local driver update signal received, but contained restricted driver_plan failed: {error}; no driver commands were executed"
                ),
                None,
            ),
        },
    }
}

struct RestrictedDriverPlanResult<'a> {
    status: &'a str,
    exit_status: i64,
}

fn restricted_driver_plan_result(value: &Value) -> Result<RestrictedDriverPlanResult<'_>> {
    let tool = value
        .get("tool")
        .and_then(Value::as_str)
        .context("restricted driver plan did not report a tool name")?;
    if tool != cli::SandboxToolArg::DriverPlan.as_cli_value() {
        bail!("restricted driver plan returned `{tool}`, expected `driver_plan`");
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("planned");
    let exit_status = value
        .get("exit_status")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| i64::from(status == "error"));
    Ok(RestrictedDriverPlanResult {
        status,
        exit_status,
    })
}

pub(crate) fn payload_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
fn evaluate_server_recover(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let now = unix_time_millis();
    if !server_recover_due(state, now) {
        return Ok(());
    }

    let Some((mut record, recovery_reason)) = find_recoverable_service(paths)? else {
        return Ok(());
    };
    let kind = service_recovery_event_kind(&recovery_reason);
    let event = AutomationTriggerEvent {
        at_unix_ms: now,
        kind: kind.to_owned(),
        source: "managed_service".to_owned(),
        watcher_hint: Some("server-recover".to_owned()),
        service_id: Some(record.service_id.clone()),
        reason: Some(recovery_reason),
        payload: json!({
            "engine": record.engine.clone(),
            "status": record.status.clone(),
            "endpoint": record.endpoint_url.clone(),
        }),
    };
    handle_server_recover_event_with_record(paths, mode, state, &event, &mut record)
}

fn handle_server_recover_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    let service_id = event
        .service_id
        .as_deref()
        .context("server-recover event is missing service_id")?;
    let mut record = load_service_record(paths, service_id)?;
    if !service_record_matches_recovery_event(paths, &record, event) {
        persistence::record_event(
            paths,
            state,
            "server-recover",
            "info",
            "ignore_nonrecoverable_service",
            &format!(
                "managed service {} does not currently need recovery; restart not attempted",
                record.service_id
            ),
            Some(record.service_id.clone()),
        )?;
        return Ok(());
    }
    handle_server_recover_event_with_record(paths, mode, state, event, &mut record)
}

fn service_record_matches_recovery_event(
    paths: &AppPaths,
    record: &ManagedServiceRecord,
    event: &AutomationTriggerEvent,
) -> bool {
    match event.kind.as_str() {
        "service.manifest_recoverable" => {
            manifest_service_recovery_reason(record, unix_time_millis()).is_some()
        }
        "service.endpoint_recoverable" => endpoint_service_recovery_reason(record).is_some(),
        "service.healthcheck_recoverable" => {
            common::engine_healthcheck_response(paths, &record.engine, &record.service_id)
                .is_ok_and(|healthcheck| common::healthcheck_response_recoverable(&healthcheck))
        }
        _ => false,
    }
}

fn handle_server_recover_event_with_record(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
    record: &mut ManagedServiceRecord,
) -> Result<()> {
    let now = unix_time_millis();
    let recovery_reason = event.reason.as_deref().unwrap_or("recoverable_event");
    let recovery_reason_display = display_recovery_reason(recovery_reason);

    match watcher_policy_action("server-recover", mode) {
        WatcherPolicyAction::Observe => persistence::record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "observe_failure",
            &format!(
                "observed managed service {} needing recovery ({recovery_reason_display}); restart not attempted in observe mode",
                record.service_id,
            ),
            Some(record.service_id.clone()),
        ),
        WatcherPolicyAction::QueueProposal => {
            let message = format!(
                "managed service {} needs recovery ({recovery_reason_display}); queueing restart proposal",
                record.service_id,
            );
            persistence::record_event(
                paths,
                state,
                "server-recover",
                "warn",
                "queue_restart_proposal",
                &message,
                Some(record.service_id.clone()),
            )?;
            queue_proposal(
                paths,
                "server-recover",
                "queue_restart_proposal",
                "Restart managed service",
                &message,
                Some(record.service_id.clone()),
            )
        }
        WatcherPolicyAction::RunContained => {
            if let Some(last_restart) = record.last_restart_unix_ms
                && now.saturating_sub(last_restart) < SERVER_RECOVER_BACKOFF_MS
            {
                return Ok(());
            }
            // A public service whose endpoint key is gone can never be recovered:
            // the respawn guard in `supervise_service` refuses it by design.
            // Report it and stop, rather than letting a permanent failure
            // propagate out of `evaluate_watchers` and take the whole daemon —
            // and every other watcher — down on each 30s recovery tick.
            if let Err(error) = common::ensure_public_service_has_endpoint_key(
                &record.host,
                rocm_engine_protocol::endpoint_key_file_if_present(paths, &record.service_id)
                    .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
                    .is_some(),
                record.requires_api_key,
            ) {
                return persistence::record_event(
                    paths,
                    state,
                    "server-recover",
                    "error",
                    "restart_managed_service_refused",
                    &format!(
                        "cannot recover managed service {} on {}:{} after \
                         {recovery_reason_display}: {error}",
                        record.service_id, record.host, record.port
                    ),
                    Some(record.service_id.clone()),
                );
            }
            restart_managed_service(paths, &mut *record)?;
            persistence::record_event(
                paths,
                state,
                "server-recover",
                "info",
                "restart_managed_service",
                &format!(
                    "restarted managed service {} on {}:{} after {recovery_reason_display}",
                    record.service_id, record.host, record.port
                ),
                Some(record.service_id.clone()),
            )
        }
    }
}

fn display_recovery_reason(reason: &str) -> String {
    match reason {
        "manifest_status_failed" => "manifest reports failed".to_owned(),
        "manifest_status_exited" => "manifest reports exited".to_owned(),
        "manifest_status_unreachable" => "manifest reports unreachable".to_owned(),
        "manifest_status_starting_stale" => "service has been starting for too long".to_owned(),
        "manifest_status_recovering_stale" => "service has been recovering for too long".to_owned(),
        "endpoint_status_unreachable" => "endpoint port is unreachable".to_owned(),
        other if other.starts_with("healthcheck_status_") => format!(
            "engine healthcheck reports {}",
            other.trim_start_matches("healthcheck_status_")
        ),
        other => other.replace('_', " "),
    }
}

fn service_recovery_event_kind(recovery_reason: &str) -> &'static str {
    if recovery_reason.starts_with("healthcheck_status_") {
        "service.healthcheck_recoverable"
    } else if recovery_reason.starts_with("endpoint_status_") {
        "service.endpoint_recoverable"
    } else {
        "service.manifest_recoverable"
    }
}

fn server_recover_due(state: &AutomationRuntimeState, now: u128) -> bool {
    let Some(snapshot) = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "server-recover" && watcher.enabled)
    else {
        return false;
    };
    snapshot
        .last_event_unix_ms
        .is_none_or(|last_event| now.saturating_sub(last_event) >= SERVER_RECOVER_BACKOFF_MS)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WatcherPolicyAction {
    Observe,
    QueueProposal,
    RunContained,
}

const fn watcher_policy_action(watcher_id: &str, mode: WatcherMode) -> WatcherPolicyAction {
    match (watcher_id, mode) {
        (_, WatcherMode::Observe) => WatcherPolicyAction::Observe,
        (_, WatcherMode::Propose) => WatcherPolicyAction::QueueProposal,
        (_, WatcherMode::Contained) => WatcherPolicyAction::RunContained,
    }
}

fn find_recoverable_service(paths: &AppPaths) -> Result<Option<(ManagedServiceRecord, String)>> {
    let now = unix_time_millis();
    for record in persistence::load_managed_services(paths)? {
        if record.mode != "managed" {
            continue;
        }
        if let Some(reason) = manifest_service_recovery_reason(&record, now) {
            return Ok(Some((record, reason)));
        }
        if matches!(record.status.as_str(), "ready" | "running") {
            let Ok(healthcheck) =
                common::engine_healthcheck_response(paths, &record.engine, &record.service_id)
            else {
                if let Some(reason) = endpoint_service_recovery_reason(&record) {
                    return Ok(Some((record, reason)));
                }
                continue;
            };
            if common::healthcheck_response_recoverable(&healthcheck) {
                return Ok(Some((
                    record,
                    format!("healthcheck_status_{}", healthcheck.status),
                )));
            }
            if let Some(reason) = endpoint_service_recovery_reason(&record) {
                return Ok(Some((record, reason)));
            }
        }
    }
    Ok(None)
}

fn endpoint_service_recovery_reason(record: &ManagedServiceRecord) -> Option<String> {
    (!common::wait_for_port(&record.host, record.port, ENDPOINT_HEALTH_TIMEOUT))
        .then(|| "endpoint_status_unreachable".to_owned())
}

fn load_service_record(paths: &AppPaths, service_id: &str) -> Result<ManagedServiceRecord> {
    rocm_core::ServiceId::new(service_id)
        .with_context(|| format!("invalid managed service id `{service_id}`"))?;
    let manifest_path = paths.service_manifest_path(service_id);
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "managed service `{service_id}` not found at {}",
            manifest_path.display()
        )
    })?;
    let record = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if record.service_id != service_id {
        bail!(
            "managed service manifest {} contains service_id `{}`, expected `{service_id}`",
            manifest_path.display(),
            record.service_id
        );
    }
    Ok(record)
}

fn manifest_service_recovery_reason(
    record: &ManagedServiceRecord,
    now_unix_ms: u128,
) -> Option<String> {
    match record.status.as_str() {
        "failed" | "exited" | "unreachable" => Some(format!("manifest_status_{}", record.status)),
        "starting" | "recovering" => {
            let started_at = record
                .last_restart_unix_ms
                .unwrap_or(record.created_at_unix_ms);
            (now_unix_ms.saturating_sub(started_at) >= SERVER_TRANSIENT_STALE_MS)
                .then(|| format!("manifest_status_{}_stale", record.status))
        }
        _ => None,
    }
}

fn restart_managed_service(_paths: &AppPaths, record: &mut ManagedServiceRecord) -> Result<()> {
    let rocmd_binary =
        std::env::current_exe().context("failed to resolve current rocmd executable path")?;
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    record.status = "recovering".to_owned();
    // Counts the restart and drops the previous run's inference verification.
    // The respawned child writes a fresh record of its own, and "recovering" is
    // outside the statuses that probe, so a stale verdict would not currently be
    // acted on — but this record is written again below, after the spawn, and
    // that write can land after the child's. Clearing here keeps the invariant
    // true at the one site that reuses a record across restarts.
    record.reset_for_restart();
    record.supervisor_pid = std::process::id();
    record.write()?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .args(recovery_supervise_args(record))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    record.supervisor_pid = child.id();
    record.write()?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        record.status = "failed".to_owned();
        record.write()?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

fn recovery_supervise_args(record: &ManagedServiceRecord) -> Vec<String> {
    let mut args = vec![
        "supervise".to_owned(),
        record.service_id.clone(),
        "--engine".to_owned(),
        record.engine.clone(),
        "--model-ref".to_owned(),
        record.model_ref.clone(),
        "--canonical-model-id".to_owned(),
        record.canonical_model_id.clone(),
        "--host".to_owned(),
        record.host.clone(),
        "--port".to_owned(),
        record.port.to_string(),
        "--device-policy".to_owned(),
        record
            .device_policy
            .as_deref()
            .unwrap_or("gpu_required")
            .to_owned(),
    ];
    args.extend(common::optional_arg(
        "--runtime-id",
        record.runtime_id.as_deref(),
    ));
    args.extend(common::optional_arg("--env-id", record.env_id.as_deref()));
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&record.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(common::optional_arg(
        "--engine-recipe-json",
        record.engine_recipe_json.as_deref(),
    ));
    args
}

fn queue_proposal(
    paths: &AppPaths,
    watcher_id: &str,
    action: &str,
    title: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    queue_proposal_with_arguments(
        paths,
        watcher_id,
        action,
        title,
        message,
        service_id.clone(),
        proposal_arguments_for_action(action, service_id.as_deref()),
    )
}

fn queue_proposal_with_arguments(
    paths: &AppPaths,
    watcher_id: &str,
    action: &str,
    title: &str,
    message: &str,
    service_id: Option<String>,
    arguments: Value,
) -> Result<()> {
    append_automation_proposal(
        paths,
        &AutomationProposalRecord {
            at_unix_ms: unix_time_millis(),
            proposal_id: String::new(),
            watcher_id: watcher_id.to_owned(),
            action: action.to_owned(),
            title: title.to_owned(),
            message: message.to_owned(),
            status: "pending".to_owned(),
            service_id,
            tool: proposal_tool_for_action(action).map(str::to_owned),
            arguments,
            reviewed_at_unix_ms: None,
        },
    )
}

fn proposal_tool_for_action(action: &str) -> Option<&'static str> {
    match action {
        "queue_restart_proposal" => Some("restart_server"),
        "queue_stop_server_proposal" => Some("stop_server"),
        "queue_update_proposal" => Some("check_updates"),
        "queue_prefetch_proposal" => Some("prefetch_artifact"),
        "prepare_driver_plan" => Some("driver_plan"),
        _ => None,
    }
}

fn proposal_arguments_for_action(action: &str, service_id: Option<&str>) -> Value {
    match action {
        "queue_restart_proposal" => json!({
            "service_id": service_id,
        }),
        "queue_stop_server_proposal" => json!({
            "service_id": service_id,
        }),
        "queue_update_proposal" => json!({}),
        "prepare_driver_plan" => json!({}),
        _ => Value::Null,
    }
}

#[cfg(unix)]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("setsid");
    command.arg(rocmd_binary);
    command
}

#[cfg(not(unix))]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    ProcessCommand::new(rocmd_binary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_app_paths, unique_test_root};

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
        let stored = persistence::load_managed_services(&paths).unwrap();
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

    #[test]
    fn recovery_supervise_args_preserve_engine_recipe_json() {
        let (_root, paths) = temp_app_paths("recovery-engine-recipe");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5-4B",
            "127.0.0.1",
            11435,
            "managed",
            123,
            Some("therock-release:gfx120X-all".to_owned()),
            Some("env-1".to_owned()),
            Some("gpu_required".to_owned()),
        );
        let engine_recipe_json = r#"{"contract_version":"0.1.0","engine":"vllm","required_flags":["--enable-auto-tool-choice"]}"#;
        record.engine_recipe_json = Some(engine_recipe_json.to_owned());

        let args = recovery_supervise_args(&record);

        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--engine-recipe-json" && pair[1] == engine_recipe_json)
        );
        assert!(
            args.windows(2)
                .any(|pair| { pair[0] == "--canonical-model-id" && pair[1] == "Qwen/Qwen3.5-4B" })
        );
    }

    #[test]
    fn watcher_policy_maps_modes_to_decisions() {
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Observe),
            WatcherPolicyAction::Observe
        );
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Propose),
            WatcherPolicyAction::QueueProposal
        );
        assert_eq!(
            watcher_policy_action("server-recover", WatcherMode::Contained),
            WatcherPolicyAction::RunContained
        );
        assert_eq!(
            watcher_policy_action("therock-update", WatcherMode::Contained),
            WatcherPolicyAction::RunContained
        );
    }

    #[tokio::test]
    async fn local_webhook_requires_enabled_automation_loop() {
        let (_root, paths) = temp_app_paths("local-webhook-requires-loop");
        let error = run_daemon(&paths, false, Some(0)).await.unwrap_err();

        assert!(error.to_string().contains("requires --automations-enabled"));
    }

    #[test]
    fn event_collector_emits_schedule_tick_for_due_update() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-schedule");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "therock-update",
            WatcherMode::Observe,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("therock-update"))
            .expect("schedule tick event should be emitted");
        assert_eq!(event.kind, "schedule.tick");
        assert_eq!(event.source, "scheduler");
        assert_eq!(event.reason.as_deref(), Some("therock_update_interval_due"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_recoverable_service_event() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-service");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("server-recover"))
            .expect("recoverable service event should be emitted");
        assert_eq!(event.kind, "service.manifest_recoverable");
        assert_eq!(event.service_id.as_deref(), Some("svc-failed"));
        assert_eq!(event.reason.as_deref(), Some("manifest_status_failed"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_endpoint_recoverable_service_event() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-endpoint");
        paths.ensure()?;
        let mut service = ManagedServiceRecord::new(
            &paths,
            "svc-endpoint",
            "missing-engine",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            1,
            "managed",
            123,
            None,
            None,
            None,
        );
        service.status = "ready".to_owned();
        service.write()?;
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events(&paths, &RocmCliConfig::default(), &state)?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("server-recover"))
            .expect("endpoint recoverable service event should be emitted");
        assert_eq!(event.kind, "service.endpoint_recoverable");
        assert_eq!(event.service_id.as_deref(), Some("svc-endpoint"));
        assert_eq!(event.reason.as_deref(), Some("endpoint_status_unreachable"));
        Ok(())
    }

    #[test]
    fn event_collector_emits_gpu_metrics_event_when_enabled() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-metrics");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Observe,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: Some(json!({
                    "gpu_data": [
                        { "gpu": 0, "asic": { "market_name": "AMD Radeon Test" } }
                    ]
                })),
                monitor_snapshot: Some(json!({ "gpu_data": [] })),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("gpu-metrics"))
            .expect("gpu metrics event should be emitted");
        assert_eq!(event.kind, "gpu.metrics");
        assert_eq!(event.source, "gpu_telemetry");
        assert_eq!(event.reason.as_deref(), Some("amd_smi_snapshot_available"));
        assert_eq!(
            event
                .payload
                .get("monitor_available")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            event
                .payload
                .get("summary")
                .and_then(Value::as_str)
                .is_some_and(|summary| summary.contains("gpu_count=1"))
        );
        Ok(())
    }

    #[test]
    fn event_collector_emits_gpu_thermal_pressure_event_when_enabled() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-thermal-pressure");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: None,
                monitor_snapshot: Some(json!({
                    "gpu_data": [
                        {
                            "gpu": 0,
                            "hotspot_temperature": { "value": 96.0 },
                            "memory_temperature": { "value": 88.0 },
                            "vram_percent": { "value": 72.0 }
                        }
                    ]
                })),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        let event = events
            .iter()
            .find(|event| event.watcher_hint.as_deref() == Some("gpu-thermal-protect"))
            .expect("thermal pressure event should be emitted");
        assert_eq!(event.kind, "gpu.thermal_pressure");
        assert_eq!(event.source, "gpu_telemetry");
        assert_eq!(
            event.reason.as_deref(),
            Some("hotspot_temperature_threshold")
        );
        assert!(
            event
                .payload
                .get("summary")
                .and_then(Value::as_str)
                .is_some_and(|summary| summary.contains("GPU 0 hotspot temperature is 96 C"))
        );
        Ok(())
    }

    #[test]
    fn event_collector_skips_gpu_pressure_below_thresholds() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-gpu-pressure-cool");
        let state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);

        let events = collect_automation_events_with_gpu_snapshot(&paths, &state, || {
            CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot: None,
                monitor_snapshot: Some(json!([
                    {
                        "gpu": 0,
                        "hotspot_temperature": 80.0,
                        "memory_temperature": 82.0,
                        "vram_percent": 50.0
                    }
                ])),
                note: None,
            }
        })?;
        fs::remove_dir_all(root).ok();

        assert!(
            !events
                .iter()
                .any(|event| event.watcher_hint.as_deref() == Some("gpu-thermal-protect"))
        );
        Ok(())
    }

    #[test]
    fn gpu_metrics_event_records_read_only_status_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-metrics-record");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Contained,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        let events = vec![AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.metrics_unavailable".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-metrics".to_owned()),
            service_id: None,
            reason: Some("amd-smi missing".to_owned()),
            payload: json!({
                "summary": "amd_smi_available=false static_snapshot=missing monitor_snapshot=missing",
            }),
        }];

        evaluate_watchers_for_events(&paths, &config, &mut state, &events)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-metrics");
        assert_eq!(event.action, "record_gpu_metrics");
        assert!(event.message.contains("telemetry is recorded only"));
        assert!(event.message.contains("amd-smi missing"));
        assert!(event.message.contains("no mutating action was taken"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_propose_queues_reviewed_stop_for_one_running_service() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-propose");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-thermal-protect");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "gpu": 0,
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
                "hotspot_temperature_c": 96.0,
            }),
        };

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let saved = load_service_record(&paths, "svc-hot")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-thermal-protect");
        assert_eq!(event.action, "queue_stop_server_proposal");
        assert_eq!(event.service_id.as_deref(), Some("svc-hot"));
        assert!(event.message.contains("asking before stopping anything"));
        assert_eq!(saved.status, "ready");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-hot"));
        assert_eq!(
            proposals[0]
                .arguments
                .get("pressure_reason")
                .and_then(Value::as_str),
            Some("hotspot_temperature_threshold")
        );
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_contained_still_queues_reviewed_stop() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-contained");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "qwen",
            "Qwen/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "running".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.memory_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("vram_pressure_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 VRAM use is 96% (limit 95%)",
                "vram_percent": 96.0,
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Contained, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let saved = load_service_record(&paths, "svc-hot")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "queue_stop_server_proposal");
        assert!(event.message.contains("contained mode still asks"));
        assert_eq!(saved.status, "running");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_observe_records_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-observe");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Observe,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("memory_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU memory temperature is high",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Observe, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "observe_gpu_pressure");
        assert!(event.message.contains("does not stop any model server"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_ambiguous_services_records_no_action() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-ambiguous");
        paths.ensure()?;
        for service_id in ["svc-a", "svc-b"] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Test",
                "127.0.0.1",
                11435,
                "managed",
                123,
                None,
                None,
                None,
            );
            record.status = "ready".to_owned();
            record.write()?;
        }
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: None,
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "gpu_pressure_no_clear_target");
        assert!(event.message.contains("did not choose a model server"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn gpu_thermal_protect_does_not_duplicate_pending_stop_proposals() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-dedupe");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-hot",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-thermal-protect",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "gpu.thermal_pressure".to_owned(),
            source: "gpu_telemetry".to_owned(),
            watcher_hint: Some("gpu-thermal-protect".to_owned()),
            service_id: Some("svc-hot".to_owned()),
            reason: Some("hotspot_temperature_threshold".to_owned()),
            payload: json!({
                "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)",
            }),
        };

        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let events = rocm_core::load_recent_automation_events(&paths, 2)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 10)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("stop_server"));
        assert!(
            events
                .iter()
                .any(|event| event.action == "stop_proposal_already_pending")
        );
        Ok(())
    }

    #[test]
    fn local_webhook_gpu_metrics_event_uses_existing_read_only_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("local-webhook-gpu-metrics");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "gpu-metrics",
            WatcherMode::Contained,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        let event = webhook::local_webhook_event_from_request(webhook::LocalWebhookEventRequest {
            watcher_hint: "gpu-metrics".to_owned(),
            kind: "gpu.metrics".to_owned(),
            service_id: None,
            reason: Some("manual smoke".to_owned()),
            payload: json!({
                "summary": "manual webhook probe",
                "action": "restart_server",
                "mode": "contained",
            }),
        })?;

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "gpu-metrics");
        assert_eq!(event.action, "record_gpu_metrics");
        assert!(event.message.contains("from local webhook"));
        assert!(event.message.contains("manual smoke"));
        assert!(event.message.contains("no mutating action was taken"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn cache_warm_propose_mode_queues_prefetch_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-propose");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Propose,
            None,
        )]);
        let event = webhook::local_webhook_event_from_request(webhook::LocalWebhookEventRequest {
            watcher_hint: "cache-warm".to_owned(),
            kind: "cache.warm".to_owned(),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
                "tool": "restart_server",
                "allow_artifact_download": true,
                "artifact_max_bytes": 1024,
            }),
        })?;

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            |_| Ok(true),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "cache-warm");
        assert_eq!(event.action, "queue_prefetch_proposal");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "cache-warm");
        assert_eq!(proposals[0].tool.as_deref(), Some("prefetch_artifact"));
        assert_eq!(
            proposals[0]
                .arguments
                .get("artifact_ref")
                .and_then(Value::as_str),
            Some("Qwen/Test-1B#hf-main")
        );
        assert!(
            proposals[0].arguments.get("tool").is_none(),
            "webhook payload must not grant arbitrary tool choice"
        );
        assert!(
            proposals[0]
                .arguments
                .get("allow_artifact_download")
                .is_none(),
            "webhook payload must not grant source-policy approval"
        );
        assert!(
            proposals[0].arguments.get("artifact_max_bytes").is_none(),
            "webhook payload must not grant download byte-limit approval"
        );
        Ok(())
    }

    #[test]
    fn cache_warm_unknown_artifact_does_not_queue_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-unknown");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "cache.warm".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("cache-warm".to_owned()),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "missing#artifact",
            }),
        };

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            |_| Ok(false),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "cache_warm_unknown_artifact");
        assert!(event.message.contains("unknown registry artifact"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn cache_warm_contained_mode_still_requires_reviewed_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("cache-warm-contained");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "cache-warm",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "cache.warm".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("cache-warm".to_owned()),
            service_id: None,
            reason: Some("idle window".to_owned()),
            payload: json!({
                "artifact_ref": "Qwen/Test-1B#hf-main",
            }),
        };

        handle_cache_warm_event_with_resolver(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_| Ok(true),
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "queue_prefetch_proposal");
        assert!(event.message.contains("explicit source-policy approval"));
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("prefetch_artifact"));
        Ok(())
    }

    #[test]
    fn driver_upgrade_propose_mode_queues_driver_plan_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-propose");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Propose,
            None,
        )]);
        let event = webhook::local_webhook_event_from_request(webhook::LocalWebhookEventRequest {
            watcher_hint: "driver-upgrade".to_owned(),
            kind: "update.available".to_owned(),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
                "tool": "restart_server",
            }),
        })?;

        handle_driver_upgrade_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "driver-upgrade");
        assert_eq!(event.action, "prepare_driver_plan");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "driver-upgrade");
        assert_eq!(proposals[0].tool.as_deref(), Some("driver_plan"));
        assert!(
            proposals[0].arguments.get("tool").is_none(),
            "webhook payload must not grant arbitrary tool choice"
        );
        Ok(())
    }

    #[test]
    fn driver_upgrade_contained_mode_runs_restricted_driver_plan() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-contained");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox::sandbox_driver_plan_value(common::CommandCapture {
                    argv: vec![
                        "rocm".to_owned(),
                        "install".to_owned(),
                        "driver".to_owned(),
                        "--dkms".to_owned(),
                        "--dry-run".to_owned(),
                    ],
                    exit_status: 0,
                    stdout: "driver install plan\n  supported: true\n".to_owned(),
                    stderr: String::new(),
                }))
            },
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "run_driver_plan");
        assert!(
            event
                .message
                .contains("contained restricted driver_plan status=planned")
        );
        assert!(event.message.contains("no driver commands were executed"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_contained_mode_requires_restricted_driver_plan_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-contained-tool");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(json!({
                    "tool": "check_updates",
                    "status": "checked",
                    "mutating": false,
                }))
            },
        )?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "driver_plan_failed");
        assert!(event.message.contains("expected `driver_plan`"));
        assert!(event.message.contains("no driver commands were executed"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_observe_mode_records_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-observe");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Observe,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("driver version is newer".to_owned()),
            payload: json!({
                "component": "driver",
            }),
        };

        handle_driver_upgrade_event(&paths, WatcherMode::Observe, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "observe_driver_update");
        assert!(event.message.contains("does not queue or run"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn driver_upgrade_ignores_non_driver_component_without_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("driver-upgrade-wrong-component");
        paths.ensure()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "driver-upgrade",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "update.available".to_owned(),
            source: "test".to_owned(),
            watcher_hint: Some("driver-upgrade".to_owned()),
            service_id: None,
            reason: Some("runtime version is newer".to_owned()),
            payload: json!({
                "component": "runtime",
            }),
        };

        handle_driver_upgrade_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "driver_upgrade_ignored_component");
        assert!(event.message.contains("payload.component=driver"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn event_dispatcher_preserves_server_recover_proposal_behavior() -> Result<()> {
        let (root, paths) = temp_app_paths("event-bus-dispatch");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("server-recover");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let events = vec![AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some("svc-failed".to_owned()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        }];

        evaluate_watchers_for_events(&paths, &config, &mut state, &events)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-failed"));
        assert_eq!(proposals[0].tool.as_deref(), Some("restart_server"));
        assert!(proposals[0].message.contains("manifest reports failed"));
        assert!(!proposals[0].message.contains("manifest_status_failed"));
        Ok(())
    }

    #[test]
    fn server_recover_local_webhook_does_not_restart_healthy_service() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-healthy-webhook");
        paths.ensure()?;
        let mut healthy = ManagedServiceRecord::new(
            &paths,
            "svc-healthy",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        healthy.status = "ready".to_owned();
        healthy.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("server-recover");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Propose);
        let event = webhook::local_webhook_event_from_request(webhook::LocalWebhookEventRequest {
            watcher_hint: "server-recover".to_owned(),
            kind: "service.manifest_recoverable".to_owned(),
            service_id: Some("svc-healthy".to_owned()),
            reason: Some("manual recovery smoke".to_owned()),
            payload: json!({}),
        })?;

        evaluate_watchers_for_events(&paths, &config, &mut state, &[event])?;
        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        let reloaded = load_service_record(&paths, "svc-healthy")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.action, "ignore_nonrecoverable_service");
        assert!(event.message.contains("does not currently need recovery"));
        assert!(proposals.is_empty());
        assert_eq!(reloaded.status, "ready");
        Ok(())
    }

    #[test]
    fn recovery_reason_display_avoids_raw_status_tokens() {
        assert_eq!(
            display_recovery_reason("manifest_status_starting_stale"),
            "service has been starting for too long"
        );
        assert_eq!(
            display_recovery_reason("healthcheck_status_unreachable"),
            "engine healthcheck reports unreachable"
        );
        assert_eq!(
            display_recovery_reason("endpoint_status_unreachable"),
            "endpoint port is unreachable"
        );
    }

    #[test]
    fn manifest_recovery_policy_covers_terminal_and_stale_transient_states() {
        let (root, paths) = temp_app_paths("manifest-recovery-policy");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-stale",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.created_at_unix_ms = 1_000;

        record.status = "exited".to_owned();
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_001).as_deref(),
            Some("manifest_status_exited")
        );

        record.status = "unreachable".to_owned();
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_001).as_deref(),
            Some("manifest_status_unreachable")
        );

        record.status = "starting".to_owned();
        assert_eq!(manifest_service_recovery_reason(&record, 2_000), None);
        assert_eq!(
            manifest_service_recovery_reason(&record, 1_000 + SERVER_TRANSIENT_STALE_MS).as_deref(),
            Some("manifest_status_starting_stale")
        );

        record.status = "recovering".to_owned();
        record.last_restart_unix_ms = Some(5_000);
        assert_eq!(manifest_service_recovery_reason(&record, 6_000), None);
        assert_eq!(
            manifest_service_recovery_reason(&record, 5_000 + SERVER_TRANSIENT_STALE_MS).as_deref(),
            Some("manifest_status_recovering_stale")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn find_recoverable_service_prefers_failed_managed_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("recoverable-service");
        paths.ensure()?;
        let mut failed = ManagedServiceRecord::new(
            &paths,
            "svc-failed",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        failed.status = "failed".to_owned();
        failed.write()?;

        let found = find_recoverable_service(&paths)?.expect("failed service should be found");
        fs::remove_dir_all(root).ok();
        assert_eq!(found.0.service_id, "svc-failed");
        assert_eq!(found.1, "manifest_status_failed");
        Ok(())
    }

    #[test]
    fn find_recoverable_service_detects_stale_starting_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("recoverable-stale-starting");
        paths.ensure()?;
        let mut stale = ManagedServiceRecord::new(
            &paths,
            "svc-starting",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        stale.status = "starting".to_owned();
        stale.created_at_unix_ms = 0;
        stale.write()?;

        let found =
            find_recoverable_service(&paths)?.expect("stale starting service should recover");
        fs::remove_dir_all(root).ok();
        assert_eq!(found.0.service_id, "svc-starting");
        assert_eq!(found.1, "manifest_status_starting_stale");
        Ok(())
    }

    #[test]
    fn server_recover_propose_mode_queues_restart_proposal() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-proposal");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "server-recover".to_owned(),
                enabled: true,
                mode: WatcherMode::Propose,
                summary: "recover".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        evaluate_server_recover(&paths, WatcherMode::Propose, &mut state)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].action, "queue_restart_proposal");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-1"));
        assert_eq!(proposals[0].status, "pending");
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_runs_read_only_check_without_queueing() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox::sandbox_check_updates_value(
                    common::CommandCapture {
                        argv: vec!["rocm".to_owned(), "update".to_owned()],
                        exit_status: 0,
                        stdout: "update\n  managed runtimes: none\n".to_owned(),
                        stderr: String::new(),
                    },
                ))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "therock-update");
        assert_eq!(event.action, "run_update_check");
        assert!(event.message.contains("contained read-only execution"));
        assert!(
            event
                .message
                .contains("restricted check_updates status=checked")
        );
        assert!(event.message.contains("no updates were applied"));
        assert!(!event.message.contains("fallback"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_records_update_available_without_applying() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained-available");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(sandbox::sandbox_check_updates_value(common::CommandCapture {
                    argv: vec!["rocm".to_owned(), "update".to_owned()],
                    exit_status: 0,
                    stdout: "update\n  runtime release-pip-gfx120x-all status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
                    stderr: String::new(),
            }))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let events = event_text
            .lines()
            .map(serde_json::from_str::<AutomationEventRecord>)
            .collect::<Result<Vec<_>, _>>()?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        let update_check = events
            .iter()
            .find(|event| event.action == "run_update_check")
            .expect("update check event should be recorded");
        assert_eq!(update_check.watcher_id, "therock-update");
        assert!(
            update_check
                .message
                .contains("restricted check_updates status=update_available")
        );
        assert!(
            update_check
                .message
                .contains("a ROCm runtime update is available")
        );
        assert!(update_check.message.contains("no updates were applied"));
        assert!(!update_check.message.contains("fallback"));
        let notification = events
            .iter()
            .find(|event| event.action == "notify_if_newer")
            .expect("notify-if-newer event should be recorded");
        assert_eq!(notification.watcher_id, "therock-update");
        assert!(
            notification
                .message
                .contains("ROCm runtime update is available")
        );
        assert!(notification.message.contains("No updates were applied"));
        assert!(audit_text.contains("\"category\":\"notification\""));
        assert!(audit_text.contains("\"action\":\"notify_if_newer\""));
        assert!(audit_text.contains("ROCm runtime update is available"));
        assert!(proposals.is_empty());
        Ok(())
    }

    #[test]
    fn therock_update_contained_mode_uses_restricted_check_updates_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-contained-tool");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };
        let event = AutomationTriggerEvent {
            at_unix_ms: 42,
            kind: "schedule.tick".to_owned(),
            source: "scheduler".to_owned(),
            watcher_hint: Some("therock-update".to_owned()),
            service_id: None,
            reason: Some("therock_update_interval_due".to_owned()),
            payload: json!({ "interval_ms": THEROCK_UPDATE_INTERVAL_MS }),
        };

        handle_therock_update_event_with_runner(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            |_paths| {
                Ok(json!({
                    "tool": "examine_snapshot",
                    "status": "captured",
                    "mutating": false,
                }))
            },
        )?;

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let event = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(event.watcher_id, "therock-update");
        assert_eq!(event.action, "update_check_failed");
        assert!(event.message.contains("expected `check_updates`"));
        assert!(event.message.contains("no updates were applied"));
        Ok(())
    }

    #[test]
    fn therock_update_notify_if_newer_uses_restricted_notification_contract() -> Result<()> {
        let (root, paths) = temp_app_paths("therock-update-notify-contract");
        paths.ensure()?;
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: vec![WatcherRuntimeSnapshot {
                id: "therock-update".to_owned(),
                enabled: true,
                mode: WatcherMode::Contained,
                summary: "check updates".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        record_update_available_notification(&paths, &mut state, "update_available")?;

        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit = audit_text
            .lines()
            .map(serde_json::from_str::<AuditEventRecord>)
            .collect::<Result<Vec<_>, _>>()?;
        fs::remove_dir_all(root).ok();

        let notification = audit
            .iter()
            .find(|event| event.category == "notification" && event.action == "notify_if_newer")
            .expect("notify_if_newer audit should be recorded");
        assert_eq!(notification.category, "notification");
        assert_eq!(notification.actor, "watcher:therock-update");
        assert_eq!(notification.watcher_id.as_deref(), Some("therock-update"));
        assert!(
            notification
                .message
                .contains("ROCm runtime update is available")
        );
        assert!(notification.message.contains("No updates were applied"));
        Ok(())
    }

    #[test]
    fn sandbox_tool_stop_server_updates_manifest_and_skips_current_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-current-pid");
        paths.ensure()?;
        let current_pid = std::process::id();
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-current",
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

        let value = sandbox::run_sandbox_tool(
            &paths,
            cli::SandboxToolArg::StopServer,
            Some("svc-current".to_owned()),
            None,
            None,
            cli::SandboxToolPolicy::default(),
        )?;
        let reloaded = load_service_record(&paths, "svc-current")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("stopped"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(reloaded.status, "stopped");
        assert!(
            value
                .get("result")
                .and_then(|result| result.get("skipped_pids"))
                .and_then(Value::as_array)
                .is_some_and(|pids| pids
                    .iter()
                    .any(|pid| pid.as_u64() == Some(u64::from(current_pid))))
        );
        Ok(())
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
        let reloaded = load_service_record(&paths, service_id);
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

    #[test]
    fn stop_server_process_tree_discovers_descendants_before_parents() {
        let output = "\
10 1
11 10
12 11
13 10
20 1
21 20
";

        assert_eq!(
            descendant_pids_from_ps_output(output, &[10]),
            vec![12, 11, 13]
        );
        assert_eq!(
            descendant_pids_from_ps_output(output, &[10, 20]),
            vec![12, 11, 13, 21]
        );
    }

    fn test_watcher_snapshot(
        id: &str,
        mode: WatcherMode,
        last_event_unix_ms: Option<u128>,
    ) -> WatcherRuntimeSnapshot {
        WatcherRuntimeSnapshot {
            id: id.to_owned(),
            enabled: true,
            mode,
            summary: "test watcher".to_owned(),
            last_event: None,
            last_event_unix_ms,
        }
    }

    fn test_runtime_state(active_watchers: Vec<WatcherRuntimeSnapshot>) -> AutomationRuntimeState {
        AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers,
        }
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
    Ok(common::healthcheck_response_ready(
        &common::engine_healthcheck_response(paths, engine, service_id)?,
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
