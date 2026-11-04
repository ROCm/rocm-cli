// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, RocmCliConfig, WatcherRuntimeSnapshot, builtin_watchers,
    unix_time_millis,
};
use std::io::{self, Write};
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);

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

    let mut ticker = time::interval(WATCHER_TICK_INTERVAL);
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
        // Captured while this process is alive so a later `rocm uninstall` can
        // tell this daemon from an unrelated process that inherited the PID.
        daemon_start_ticks: rocm_core::ProcessIdentity::capture(std::process::id()).start_ticks,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;

    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn a_fresh_runtime_state_records_this_daemons_start_time() {
        // `rocm uninstall` refuses to signal a daemon pid it cannot prove is
        // rocmd, and the proof is this field: `None` here would make every
        // record look pre-upgrade and abort every uninstall that finds a daemon.
        let state = build_runtime_state(&RocmCliConfig::default(), true);
        assert_eq!(state.daemon_pid, std::process::id());
        let recorded = state
            .daemon_start_ticks
            .expect("a live daemon must record its own start-time");
        assert_eq!(
            Some(recorded),
            rocm_core::process_start_ticks(std::process::id())
        );
    }

    #[tokio::test]
    async fn local_webhook_requires_enabled_automation_loop() {
        let (_root, paths) = temp_app_paths("local-webhook-requires-loop");
        let error = run_daemon(&paths, false, Some(0)).await.unwrap_err();

        assert!(error.to_string().contains("requires --automations-enabled"));
    }
}
