// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationProposalRecord, AutomationRuntimeState, AutomationTriggerEvent,
    CodexBridgeGpuSnapshot, ManagedServiceRecord, RocmCliConfig, WatcherMode,
    WatcherRuntimeSnapshot, append_automation_proposal, builtin_watchers,
    resolve_model_recipe_artifact, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
use std::fs;
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const SERVER_TRANSIENT_STALE_MS: u128 = 5 * 60 * 1_000;
const ENDPOINT_HEALTH_TIMEOUT: Duration = Duration::from_millis(250);
const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;
const GPU_METRICS_INTERVAL_MS: u128 = 60 * 1000;
const GPU_THERMAL_HOTSPOT_PRESSURE_C: f64 = 95.0;
const GPU_THERMAL_MEMORY_PRESSURE_C: f64 = 95.0;
const GPU_MEMORY_VRAM_PRESSURE_PERCENT: f64 = 95.0;

pub(crate) fn reconcile_watcher_snapshots(
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
) {
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

pub(crate) fn evaluate_watchers(
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
        crate::common::gather_gpu_snapshot_for_config(config)
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

pub(crate) fn evaluate_watchers_for_events(
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
        crate::sandbox::run_sandbox_tool(
            paths,
            crate::cli::SandboxToolArg::CheckUpdates,
            None,
            None,
            None,
            crate::cli::SandboxToolPolicy::default(),
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
            crate::persistence::record_event(
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
                    crate::persistence::record_event(
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
                            crate::common::update_check_message(result.status)
                        ),
                        None,
                    )?;
                    if result.update_available {
                        record_update_available_notification(paths, state, result.status)?;
                    }
                }
                Err(error) => {
                    crate::persistence::record_event(
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
                crate::persistence::record_event(
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
    if tool != crate::cli::SandboxToolArg::CheckUpdates.as_cli_value() {
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
    crate::persistence::record_event(
        paths,
        state,
        "therock-update",
        "info",
        "notify_if_newer",
        message,
        None,
    )?;
    crate::sandbox::record_notification_audit(
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

    crate::persistence::record_event(
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
        return crate::persistence::record_event(
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
        return crate::persistence::record_event(
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
        return crate::persistence::record_event(
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
    crate::persistence::record_event(
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

    let active = crate::persistence::load_managed_services(paths)?
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
        return crate::persistence::record_event(
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
            return crate::persistence::record_event(
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
            return crate::persistence::record_event(
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
        WatcherMode::Observe => crate::persistence::record_event(
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
            crate::persistence::record_event(
                paths,
                state,
                "cache-warm",
                "info",
                action,
                &message,
                None,
            )?;
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
        crate::sandbox::run_sandbox_tool(
            paths,
            crate::cli::SandboxToolArg::DriverPlan,
            None,
            None,
            None,
            crate::cli::SandboxToolPolicy::default(),
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
        return crate::persistence::record_event(
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
        WatcherMode::Observe => crate::persistence::record_event(
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
            crate::persistence::record_event(
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
                Ok(result) => crate::persistence::record_event(
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
                Err(error) => crate::persistence::record_event(
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
            Err(error) => crate::persistence::record_event(
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
    if tool != crate::cli::SandboxToolArg::DriverPlan.as_cli_value() {
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
        crate::persistence::record_event(
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
    // Re-checked here as well as in the scan: a stop can be requested between
    // the event being raised and it being handled.
    if stop_requested(record) {
        return false;
    }
    match event.kind.as_str() {
        "service.manifest_recoverable" => {
            manifest_service_recovery_reason(record, unix_time_millis()).is_some()
        }
        "service.endpoint_recoverable" => endpoint_service_recovery_reason(record).is_some(),
        "service.healthcheck_recoverable" => {
            crate::common::engine_healthcheck_response(paths, &record.engine, &record.service_id)
                .is_ok_and(|healthcheck| {
                    crate::common::healthcheck_response_recoverable(&healthcheck)
                })
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
        WatcherPolicyAction::Observe => crate::persistence::record_event(
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
            crate::persistence::record_event(
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
            if let Err(error) = crate::common::ensure_public_service_has_endpoint_key(
                &record.host,
                rocm_engine_protocol::endpoint_key_file_if_present(paths, &record.service_id)
                    .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
                    .is_some(),
                record.requires_api_key,
            ) {
                return crate::persistence::record_event(
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
            // A contended allocation lock or an occupied recorded port is a
            // recoverable, per-service condition — report it and move on rather
            // than letting one service take the whole 30s watcher tick (and the
            // daemon) down.
            if let Err(error) = restart_managed_service(paths, &mut *record) {
                // `restart_managed_service` can fail before it ever writes the
                // record (e.g. the recorded port is still occupied), so without
                // this stamp the backoff guard above never sees a prior attempt
                // and the next 30s tick retries immediately, forever. Best
                // effort: a write failure here just means the next tick retries
                // sooner, not a worse outcome than today.
                let _ = crate::service::update_supervised_record(paths, record, |current| {
                    current.last_restart_unix_ms = Some(now);
                });
                return crate::persistence::record_event(
                    paths,
                    state,
                    "server-recover",
                    "error",
                    "restart_managed_service_failed",
                    &format!(
                        "failed to recover managed service {} on {}:{} after \
                         {recovery_reason_display}: {error:#}",
                        record.service_id, record.host, record.port
                    ),
                    Some(record.service_id.clone()),
                );
            }
            crate::persistence::record_event(
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

pub(crate) fn find_recoverable_service(
    paths: &AppPaths,
) -> Result<Option<(ManagedServiceRecord, String)>> {
    let now = unix_time_millis();
    for record in crate::persistence::load_managed_services(paths)? {
        if record.mode != "managed" || stop_requested(&record) {
            continue;
        }
        if let Some(reason) = manifest_service_recovery_reason(&record, now) {
            return Ok(Some((record, reason)));
        }
        if matches!(record.status.as_str(), "ready" | "running") {
            let Ok(healthcheck) = crate::common::engine_healthcheck_response(
                paths,
                &record.engine,
                &record.service_id,
            ) else {
                if let Some(reason) = endpoint_service_recovery_reason(&record) {
                    return Ok(Some((record, reason)));
                }
                continue;
            };
            if crate::common::healthcheck_response_recoverable(&healthcheck) {
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

/// Whether an operator asked for this service to stop and that stop is still
/// standing.
///
/// Such a service is never a recovery candidate, whatever its status says. A
/// stop that could not confirm every recorded process gone deliberately leaves
/// the status at `ready`/`running` — it did not happen, so it must not be
/// claimed — and a stop still in flight has not reached its verdict yet. Either
/// way the endpoint has stopped answering, which on its own is exactly what
/// recovery restarts. `stop_requested_unix_ms` is what tells the operator's
/// stop apart from a crash; a confirmed stop, and any fresh launch, clear it.
pub(crate) const fn stop_requested(record: &ManagedServiceRecord) -> bool {
    record.stop_requested_unix_ms.is_some()
}

fn endpoint_service_recovery_reason(record: &ManagedServiceRecord) -> Option<String> {
    (!crate::common::wait_for_port(&record.host, record.port, ENDPOINT_HEALTH_TIMEOUT))
        .then(|| "endpoint_status_unreachable".to_owned())
}

pub(crate) fn load_service_record(
    paths: &AppPaths,
    service_id: &str,
) -> Result<ManagedServiceRecord> {
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

/// Respawn the supervisor for a managed service that needs recovery.
///
/// Recovery re-claims the concrete port already written in the record under
/// [`rocm_core::MANAGED_LAUNCH_LOCK_TIMEOUT`]-bounded acquisition of the same
/// cross-process launch lock a `rocm serve` launch holds (unbounded, via
/// [`rocm_core::FileLock::acquire`]). The two still can't publish competing
/// records for this port at the same instant — the lock is exclusive either
/// way. A concurrent `rocm serve` launch does still block for as long as
/// recovery holds the lock; what the bound buys is the other direction —
/// recovery never waits out an interactive launch indefinitely. If a `rocm
/// serve` call is already mid-transaction when this bounded acquire times
/// out, recovery is deferred (not retried in a busy loop) rather than
/// queuing behind the CLI.
pub(crate) fn restart_managed_service(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
) -> Result<()> {
    let _allocation = rocm_core::lock_service_allocation(paths)?;

    // Re-check against the disk, now that the lock excludes a concurrent
    // `rocm serve` publish: the caller's `record` was read before this lock was
    // acquired, so a fresh launch could have claimed this exact (host, port)
    // in the meantime (the engine itself binds the port only after the model
    // loads, so a raw OS bind probe alone would not catch that — see
    // `crate::persistence::load_managed_services`'s callers for the shared
    // `is_live` reservation this mirrors). Refuse rather than spawn a
    // competitor; the next watcher tick reconsiders this service.
    let on_disk = crate::persistence::load_managed_services(paths)
        .context("failed to re-read managed services under the launch lock")?;
    if let Some(collision) = on_disk
        .iter()
        .find(|other| other.service_id != record.service_id && other.blocks_port(record.port))
    {
        anyhow::bail!(
            "cannot recover managed service {} on {}:{}: {} claimed that port while recovery \
             was waiting for the launch lock",
            record.service_id,
            record.host,
            record.port,
            collision.service_id
        );
    }

    // On the canonical loopback host, prove the recorded port is genuinely free
    // before publishing `recovering` or spawning anything: a supervisor that
    // cannot bind would replace a truthful record with a lie. Custom hosts keep
    // the existing engine-owned bind behavior — the record transition is still
    // serialized, but the bind is not preflighted.
    let lease = if rocm_core::is_canonical_loopback_host(&record.host) {
        Some(rocm_core::lease_loopback_port_or_bind_error(
            record.port,
            format_args!(
                "cannot recover managed service {} on {}:{}: its recorded port is in use",
                record.service_id, record.host, record.port
            ),
        )?)
    } else {
        None
    };

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
    // No supervisor exists until the spawn below succeeds, so none is recorded.
    // Recording this daemon's own PID here, as a stand-in, left the manifest
    // naming the daemon as the service's supervisor whenever the spawn failed:
    // a stop run from any other process (`rocmd sandbox-tool stop_server`,
    // `rocm services stop`) verifies that identity, finds it live, and
    // terminates the daemon's whole process tree. The spawned supervisor's
    // identity is recorded right after the spawn.
    record.supervisor_pid = 0;
    record.supervisor_start_ticks = None;
    // The engine PID belonged to the run being replaced, and nothing here
    // starts an engine: the supervisor spawned below records its own, together
    // with its token. Carried forward, the old PID would sit beside a
    // supervisor it no longer matches — and on a record `rocm` wrote, with no
    // token at all, so the next stop could only signal it blind.
    record.engine_pid = None;
    record.engine_start_ticks = None;
    // Restarting is a request to run, so it supersedes any earlier stop request,
    // as `rocm services restart` treats it too.
    record.stop_requested_unix_ms = None;
    record.write()?;

    // Hand the port to the supervisor: close the probe socket immediately before
    // the spawn, still holding the allocation lock.
    if let Some(lease) = lease {
        lease.release();
    }

    let mut child = detached_rocmd_command(&rocmd_binary)
        .args(recovery_supervise_args(record))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    if let Err(error) = record_spawned_supervisor(paths, record, child.id()) {
        // The supervisor is alive but unrecorded: it would hold the port while
        // being invisible to every stop/recovery path.
        terminate_recovery_supervisor(&mut child);
        return Err(error);
    }

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        // Not a write-back of `record`: a stop issued since the spawn has
        // persisted its marker, and a `failed` record without one is what
        // recovery restarts. Same discipline as the supervisor's own writes.
        crate::service::update_supervised_record(paths, record, |current| {
            current.status = "failed".to_owned();
        })?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

/// Record the supervisor [`restart_managed_service`] just spawned, as `pid`.
///
/// Grace given to a recovery supervisor that must be terminated because its
/// record could not be published. Short: it is milliseconds old.
const RECOVERY_CLEANUP_GRACE: Duration = Duration::from_secs(2);

/// Terminate and reap a recovery supervisor that could not be recorded.
///
/// Tree scope, forced: the supervisor may already have started an engine child,
/// and leaving either alive would keep the recorded port occupied by a process
/// no stop or recovery path can find. Reaping afterwards keeps the daemon free
/// of zombies.
fn terminate_recovery_supervisor(child: &mut std::process::Child) {
    let _outcome = rocm_core::terminate_verified(
        &rocm_core::ProcessIdentity::capture(child.id()),
        rocm_core::KillScope::Tree,
        RECOVERY_CLEANUP_GRACE,
        true,
    );
    let _ = child.wait();
}

/// Between the restart's first write — which clears the recorded PIDs, since
/// no supervisor exists yet — and this one, the record names no process at
/// all. A stop that starts in that window persists its marker first and then
/// finds nothing to signal; writing the restart's whole copy back here would
/// erase that marker while the new supervisor comes up, and leave the stop's
/// own final write comparing against a record that no longer shows the
/// process it has to answer for. So only the supervisor's identity is applied,
/// to the record as it is on disk now, through the same ownership guard as the
/// supervisor's own writes: against the copy from before this assignment, so
/// it is the cleared PID that has to still be there. If the new supervisor has
/// already recorded itself, or the record was removed, nothing is written.
fn record_spawned_supervisor(
    paths: &AppPaths,
    record: &mut ManagedServiceRecord,
    pid: u32,
) -> Result<()> {
    let unrecorded = record.clone();
    record.record_supervisor_identity(pid);
    let (supervisor_pid, supervisor_start_ticks) =
        (record.supervisor_pid, record.supervisor_start_ticks);
    crate::service::update_supervised_record(paths, &unrecorded, |current| {
        current.supervisor_pid = supervisor_pid;
        current.supervisor_start_ticks = supervisor_start_ticks;
    })
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
    args.extend(crate::common::optional_arg(
        "--runtime-id",
        record.runtime_id.as_deref(),
    ));
    args.extend(crate::common::optional_arg(
        "--env-id",
        record.env_id.as_deref(),
    ));
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&record.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    args.extend(crate::common::optional_arg(
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
    use crate::test_support::identity_probe_record;
    use crate::test_support::{
        UNCONFIRMED_STOP_PID, seed_keyed_service, stop_with_outcome, temp_app_paths,
    };
    use rocm_core::{AuditEventRecord, AutomationEventRecord};

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
        let event = crate::webhook::local_webhook_event_from_request(
            crate::webhook::LocalWebhookEventRequest {
                watcher_hint: "gpu-metrics".to_owned(),
                kind: "gpu.metrics".to_owned(),
                service_id: None,
                reason: Some("manual smoke".to_owned()),
                payload: json!({
                    "summary": "manual webhook probe",
                    "action": "restart_server",
                    "mode": "contained",
                }),
            },
        )?;

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
        let event = crate::webhook::local_webhook_event_from_request(
            crate::webhook::LocalWebhookEventRequest {
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
            },
        )?;

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
        let event = crate::webhook::local_webhook_event_from_request(
            crate::webhook::LocalWebhookEventRequest {
                watcher_hint: "driver-upgrade".to_owned(),
                kind: "update.available".to_owned(),
                service_id: None,
                reason: Some("driver version is newer".to_owned()),
                payload: json!({
                    "component": "driver",
                    "tool": "restart_server",
                }),
            },
        )?;

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
                Ok(crate::sandbox::sandbox_driver_plan_value(
                    crate::common::CommandCapture {
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
                    },
                ))
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
        let event = crate::webhook::local_webhook_event_from_request(
            crate::webhook::LocalWebhookEventRequest {
                watcher_hint: "server-recover".to_owned(),
                kind: "service.manifest_recoverable".to_owned(),
                service_id: Some("svc-healthy".to_owned()),
                reason: Some("manual recovery smoke".to_owned()),
                payload: json!({}),
            },
        )?;

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
                Ok(crate::sandbox::sandbox_check_updates_value(
                    crate::common::CommandCapture {
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
                Ok(crate::sandbox::sandbox_check_updates_value(crate::common::CommandCapture {
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

    #[test]
    fn recovery_refuses_a_port_a_launch_claimed_while_it_waited_for_the_lock() -> Result<()> {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        // The gap Finding #2 closes: a crashed service's port is simultaneously
        // "free" to a brand-new `rocm serve --gpu auto` launch (which only
        // checks *live* records) and "needs recovery" to this function (which
        // used to only probe the raw OS bind). Neither process's engine binds
        // the port until its model finishes loading, so a probe alone cannot
        // see a `starting` competitor that has not bound yet. Recreate that
        // exact disk state under the lock and require recovery to refuse,
        // rather than racing a second engine onto the same port.
        let (_root, paths) = temp_app_paths("recovery-launch-claimed-port");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let port = 21_777_u16;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-recover-3",
            "lemonade",
            "qwen",
            "qwen-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        let guard = rocm_core::lock_service_allocation(&paths)?;
        let finished = Arc::new(AtomicBool::new(false));
        let handle = {
            let paths = paths.clone();
            let finished = Arc::clone(&finished);
            thread::spawn(move || {
                let result = restart_managed_service(&paths, &mut record);
                finished.store(true, Ordering::SeqCst);
                result
            })
        };

        thread::sleep(Duration::from_millis(250));
        assert!(
            !finished.load(Ordering::SeqCst),
            "recovery must serialize on the shared allocation lock"
        );

        // Simulate a `rocm serve --gpu auto` publishing a brand-new service on
        // this exact port while recovery was still waiting for the lock it
        // itself now holds — the one window the launch lock is supposed to
        // close.
        let mut competitor = ManagedServiceRecord::new(
            &paths,
            "lemonade-fresh-launch",
            "lemonade",
            "other-model",
            "other-model-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        competitor.status = "starting".to_owned();
        competitor.write()?;
        drop(guard);

        let error = handle
            .join()
            .expect("recovery thread")
            .expect_err("a live competitor on the same port must refuse recovery");
        let message = format!("{error:#}");
        assert!(message.contains("lemonade-fresh-launch"), "{message}");
        assert!(message.contains("claimed that port"), "{message}");

        let on_disk = crate::persistence::load_managed_services(&paths)?
            .into_iter()
            .find(|candidate| candidate.service_id == "lemonade-recover-3")
            .expect("record still present");
        assert_eq!(
            on_disk.status, "failed",
            "refused recovery must not overwrite the truthful record with `recovering`"
        );
        Ok(())
    }

    #[test]
    fn contained_recovery_survives_a_failed_restart_and_backs_off() -> Result<()> {
        use std::net::TcpListener;

        let (root, paths) = temp_app_paths("recovery-contained-port-collision");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let occupant = TcpListener::bind(("127.0.0.1", 0))?;
        let port = occupant.local_addr()?.port();

        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-recover-2",
            "lemonade",
            "qwen",
            "qwen-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Contained,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "manifest_scan".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };

        // The watcher must not propagate a per-service failure: that would take
        // the whole 30s tick (and every other watcher) down with it.
        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Contained,
            &mut state,
            &event,
            &mut record,
        )
        .expect("a failed restart must be caught and reported, not propagated");

        let event_text = fs::read_to_string(paths.automation_events_path())?;
        let logged = serde_json::from_str::<AutomationEventRecord>(event_text.trim())?;
        let reloaded = load_service_record(&paths, "lemonade-recover-2")?;
        fs::remove_dir_all(root).ok();
        drop(occupant);

        assert_eq!(logged.level, "error");
        assert_eq!(logged.action, "restart_managed_service_failed");
        assert!(
            logged.message.contains("recorded port is in use"),
            "{}",
            logged.message
        );
        // Unchanged other than the backoff stamp: the occupied port must not
        // have been reported as recovered.
        assert_eq!(reloaded.status, "failed");
        // Stamped despite `restart_managed_service` never writing the record on
        // this failure path, so the next 30s tick's backoff guard sees a prior
        // attempt instead of retrying immediately forever.
        assert!(reloaded.last_restart_unix_ms.is_some());
        Ok(())
    }

    #[test]
    fn recovery_refuses_when_the_registry_cannot_be_re_read_leaving_the_prior_record_intact()
    -> Result<()> {
        use std::net::TcpListener;

        let (_root, paths) = temp_app_paths("recovery-publication-failure");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // A free recorded port, so a lease was never the obstacle under test.
        let probe = TcpListener::bind(("127.0.0.1", 0))?;
        let port = probe.local_addr()?.port();
        drop(probe);

        let mut record = ManagedServiceRecord::new(
            &paths,
            "lemonade-recover-2",
            "lemonade",
            "qwen",
            "qwen-canonical".to_owned(),
            rocm_core::DEFAULT_LOCAL_HOST,
            port,
            "managed",
            4242,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        // Replace the manifest with a directory. `load_managed_services` scans
        // every `.json` entry under `services_dir()` to re-check for a
        // collision before publishing `recovering`, and this record's own
        // manifest is one of those entries — so the corruption trips that
        // re-read, not the publish a few lines later. That is the more
        // conservative of the two and still proves the invariant this test is
        // for: recovery must fail closed, not spawn a competitor, when it
        // cannot trust what else is live.
        fs::remove_file(&record.manifest_path)?;
        fs::create_dir_all(&record.manifest_path)?;

        let mut attempt = record.clone();
        let error = restart_managed_service(&paths, &mut attempt)
            .expect_err("an unreadable registry must refuse, not assume no collision");
        let message = format!("{error:#}");
        assert!(
            message.contains("failed to re-read managed services under the launch lock"),
            "{message}"
        );
        // Nothing was spawned and the on-disk row was never replaced, so the
        // error must NOT claim a failed rollback: that phrasing is reserved for
        // the post-spawn case where the record really is left mid-transition.
        assert!(!message.contains("rollback also failed"), "{message}");

        // The caller's record must not claim a recovery that never happened.
        assert_eq!(attempt.status, "failed");
        assert_eq!(attempt.restart_count, record.restart_count);
        assert_eq!(attempt.supervisor_pid, record.supervisor_pid);
        fs::remove_dir_all(&record.manifest_path)?;
        Ok(())
    }

    /// The same guarantee for the daemon's recovery path, which records a PID of
    /// its own. See `supervise_service_persists_the_supervisor_identity_token`
    /// for why an unrecorded token is the defect rather than a cosmetic gap.
    ///
    /// The restart re-execs `rocmd supervise`, which here is the test binary:
    /// it rejects those arguments and exits at once, so the restart reports
    /// failure — after it has written the record this test reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_managed_service_persists_the_supervisor_identity_token() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-records-identity");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-identity";
        let mut record = identity_probe_record(&paths, service_id, 11444);
        record.status = "failed".to_owned();
        record.write()?;
        assert_eq!(
            load_service_record(&paths, service_id)?.supervisor_start_ticks,
            None,
            "precondition: the seeded record carries no token"
        );

        let _ = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "restart_managed_service must persist the supervisor's start-time token beside its PID"
        );
        Ok(())
    }

    /// An unconfirmed stop leaves the status at `ready` on purpose, which is
    /// exactly what the daemon's recovery scan looks for. The stop request it
    /// records is the only thing telling it apart from a crash, so recovery
    /// must honour it — otherwise the `server-recover` watcher, which acts by
    /// default, restarts the service the operator just asked to stop.
    #[test]
    fn recovery_does_not_restart_a_service_whose_stop_is_unconfirmed() -> Result<()> {
        let (root, paths) = temp_app_paths("recover-skips-unconfirmed-stop");
        paths.ensure()?;
        let service_id = "svc-recover-skips-unconfirmed-stop";
        seed_keyed_service(&paths, service_id, 11450)?;

        let stopped =
            stop_with_outcome(&paths, service_id, rocm_core::TerminationOutcome::TimedOut);
        let found = find_recoverable_service(&paths);
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert_eq!(
            stopped?.get("stopped").and_then(Value::as_bool),
            Some(false),
            "precondition: the stop must be unconfirmed"
        );
        let reloaded = reloaded?;
        assert_eq!(
            reloaded.status, "ready",
            "precondition: status left as it was"
        );
        assert!(
            found?.is_none(),
            "a service with a standing stop request must not be recovered"
        );
        Ok(())
    }

    /// The same gate on a record recovery would otherwise pick up from its
    /// status alone, without any endpoint probe: with a stop request it is
    /// skipped, and the identical record without one is still recovered.
    #[test]
    fn recovery_skips_a_failed_service_only_while_its_stop_request_stands() -> Result<()> {
        let (root, paths) = temp_app_paths("recover-honours-stop-request");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-recover-honours-stop-request",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11451,
            "managed",
            UNCONFIRMED_STOP_PID,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.stop_requested_unix_ms = Some(1);
        record.write()?;
        let event = AutomationTriggerEvent {
            at_unix_ms: unix_time_millis(),
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };
        let with_request = find_recoverable_service(&paths);
        let event_with_request = service_record_matches_recovery_event(&paths, &record, &event);

        record.stop_requested_unix_ms = None;
        record.write()?;
        let without_request = find_recoverable_service(&paths);
        let event_without_request = service_record_matches_recovery_event(&paths, &record, &event);
        fs::remove_dir_all(root).ok();

        assert!(
            with_request?.is_none(),
            "a standing stop request must block recovery"
        );
        assert!(
            !event_with_request,
            "a queued recovery event must not act on a service since asked to stop"
        );
        assert_eq!(
            without_request?.map(|(found, reason)| (found.service_id, reason)),
            Some((
                "svc-recover-honours-stop-request".to_owned(),
                "manifest_status_failed".to_owned()
            )),
            "without a stop request the same record must still be recovered"
        );
        assert!(event_without_request);
        Ok(())
    }

    /// A restart moves the supervisor PID; the engine PID it carried belonged
    /// to the run being replaced. On a record `rocm serve --background` wrote,
    /// that PID has no token, so carrying it past the point where the two PIDs
    /// diverge leaves the next stop an entry it can only signal blind.
    ///
    /// Linux-only for the same reason as the test above: the restart re-execs
    /// the test binary, which is only known to reject those arguments there.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_managed_service_drops_the_replaced_runs_engine_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-drops-engine-pid");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-drops-engine-pid";
        let mut record = identity_probe_record(&paths, service_id, 11455);
        // The shape `rocm serve --background` writes: one PID in both roles.
        record.supervisor_pid = UNCONFIRMED_STOP_PID;
        record.engine_pid = Some(UNCONFIRMED_STOP_PID);
        record.engine_start_ticks = None;
        record.stop_requested_unix_ms = Some(1);
        record.status = "failed".to_owned();
        record.write()?;

        let _ = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_ne!(persisted.supervisor_pid, UNCONFIRMED_STOP_PID);
        assert_eq!(
            persisted.engine_pid, None,
            "the replaced run's engine PID must not outlive the restart"
        );
        assert_eq!(persisted.engine_start_ticks, None);
        assert_eq!(
            persisted.stop_requested_unix_ms, None,
            "a restart supersedes an earlier stop request"
        );
        Ok(())
    }

    /// Recovery re-claims the concrete port already written in the record, so
    /// it must serialize on the same cross-process allocation lock a fresh
    /// `rocm serve` launch takes, and must refuse a recorded port something
    /// else already holds rather than publish a `recovering` record that
    /// lies.
    #[test]
    fn recovery_waits_for_the_allocation_lock_and_refuses_an_occupied_port() -> Result<()> {
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let (root, paths) = temp_app_paths("recovery-port-collision");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        // Something else already owns the port this service recorded.
        let occupant = TcpListener::bind(("127.0.0.1", 0))?;
        let port = occupant.local_addr()?.port();

        let mut record = identity_probe_record(&paths, "svc-recovery-port-collision", port);
        record.status = "failed".to_owned();
        record.write()?;

        let guard = rocm_core::lock_service_allocation(&paths)?;
        let finished = Arc::new(AtomicBool::new(false));
        let handle = {
            let paths = paths.clone();
            let finished = Arc::clone(&finished);
            let mut record = record.clone();
            thread::spawn(move || {
                let result = restart_managed_service(&paths, &mut record);
                finished.store(true, Ordering::SeqCst);
                result
            })
        };

        thread::sleep(Duration::from_millis(250));
        assert!(
            !finished.load(Ordering::SeqCst),
            "recovery must serialize on the shared allocation lock"
        );
        drop(guard);

        let error = handle
            .join()
            .expect("recovery thread")
            .expect_err("an occupied recorded port must fail recovery");
        let message = format!("{error:#}");
        assert!(message.contains("recorded port is in use"), "{message}");

        // Nothing was spawned and the prior truthful record is untouched: it must
        // not have been rewritten as `recovering`.
        let on_disk = crate::persistence::load_managed_services(&paths)?
            .into_iter()
            .find(|candidate| candidate.service_id == "svc-recovery-port-collision")
            .expect("record still present");
        fs::remove_dir_all(root).ok();
        drop(occupant);

        assert_eq!(on_disk.status, "failed");
        assert_eq!(on_disk.restart_count, record.restart_count);
        Ok(())
    }

    #[test]
    #[ignore = "spawned as a child process by terminate_recovery_supervisor_kills_and_reaps_the_child"]
    fn recovery_cleanup_sleeper_child() {
        if std::env::var_os("ROCMD_TEST_SLEEPER_CHILD").is_none() {
            return;
        }
        thread::sleep(Duration::from_mins(2));
    }

    #[test]
    fn terminate_recovery_supervisor_kills_and_reaps_the_child() {
        // A real long-lived process stands in for a supervisor that was spawned
        // but whose record could not be published.
        let mut child = ProcessCommand::new(std::env::current_exe().expect("test binary"))
            .arg("watchers::tests::recovery_cleanup_sleeper_child")
            .arg("--exact")
            .arg("--ignored")
            .env("ROCMD_TEST_SLEEPER_CHILD", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleeper child");
        let child_pid = child.id();

        terminate_recovery_supervisor(&mut child);

        // The helper reaps, so the pid must be gone rather than a zombie.
        assert!(
            !rocm_core::process_is_running(child_pid),
            "an unrecordable supervisor must be terminated and reaped"
        );
        // The helper already reaped it; wait again so the child is provably
        // handled on every path in this function (and reaping is idempotent).
        assert!(child.wait().is_ok());
    }

    /// If publishing the `recovering` transition fails, nothing has been
    /// spawned yet: the rollback at that first write must undo the in-memory
    /// transition and leave the on-disk record exactly as it was, not a
    /// half-applied `recovering` row. The manifest write goes through
    /// [`crate::write_atomic`][rocm_core_write_atomic]-style write-then-rename,
    /// so a read-only *file* no longer blocks it (rename ignores the
    /// destination's own permission bits) — a read-only *directory* still does,
    /// since it can't hold the new temp file the rename needs.
    ///
    /// [rocm_core_write_atomic]: rocm_core
    ///
    /// Unix only and skipped as root, where a read-only mode is not enforced
    /// and the write would simply succeed.
    #[cfg(unix)]
    #[test]
    fn restart_rolls_back_the_in_memory_record_when_the_first_write_fails() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        if rocm_core::openmpi::running_as_root() {
            return Ok(());
        }

        let (root, paths) = temp_app_paths("restart-rollback-first-write");
        paths.ensure()?;
        let service_id = "svc-restart-rollback-first-write";
        let mut record = identity_probe_record(&paths, service_id, 11459);
        record.status = "failed".to_owned();
        record.write()?;

        let services_dir = paths.services_dir();
        // The launch lock and service log files are both created lazily, on
        // first acquire and on first open respectively; pre-create them so
        // making the directory read-only below blocks only the manifest's
        // write-then-rename, not those earlier opens.
        fs::File::create(paths.managed_launch_lock_path())?;
        fs::File::create(&record.log_path)?;
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o555))?;

        let result = restart_managed_service(&paths, &mut record);

        // Restore write access so the rest of the write (if any landed) can be
        // read back, and so cleanup can remove the directory.
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o755))?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let error = result.expect_err("a read-only manifest must fail the first write");
        assert!(
            format!("{error:#}").contains("failed to publish the recovering service record"),
            "{error:#}"
        );
        assert_eq!(
            record.status, "failed",
            "a failed first write must restore the in-memory record rather than leave it \
             \"recovering\""
        );
        assert_eq!(
            persisted?.status, "failed",
            "nothing must reach disk when the first write itself fails"
        );
        Ok(())
    }

    /// The first-write rollback above covers a failure before anything is
    /// spawned; this one covers the second write, after the supervisor exists.
    /// A thread races the restart: as soon as the first write lands
    /// (`recovering`, no supervisor yet), it corrupts the manifest's content so
    /// `record_spawned_supervisor`'s read fails. That must kill the spawned
    /// supervisor and restore the prior record — and the rollback write itself
    /// must still land, since corrupting *content* leaves the file a writable
    /// regular file.
    #[test]
    fn restart_rolls_back_and_kills_the_supervisor_when_the_second_write_cannot_be_read()
    -> Result<()> {
        let (root, paths) = temp_app_paths("restart-rollback-second-write-unreadable");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-rollback-second-write-unreadable";
        let mut record = identity_probe_record(&paths, service_id, 11461);
        record.status = "failed".to_owned();
        record.write()?;
        let prior = record.clone();

        let manifest_path = paths.service_manifest_path(service_id);
        let corrupt = thread::spawn(move || -> bool {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while std::time::Instant::now() < deadline {
                if let Ok(bytes) = fs::read(&manifest_path)
                    && let Ok(on_disk) = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
                    && on_disk.status == "recovering"
                    && on_disk.supervisor_pid == 0
                {
                    let _ = fs::write(&manifest_path, b"not json");
                    return true;
                }
                // Busy-spin: the window this races (between the restart's
                // first write and its spawned child's own write) is only as
                // long as a `fork`+`exec`, far shorter than any sleep tick
                // could reliably catch.
                std::hint::spin_loop();
            }
            false
        });
        let outcome = restart_managed_service(&paths, &mut record);
        let corrupted_in_time = corrupt.join().expect("corrupt thread");
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            corrupted_in_time,
            "precondition: the race won before the second write ran"
        );
        let error = outcome.expect_err("an unreadable second write must fail the restart");
        let message = format!("{error:#}");
        assert!(
            message.contains("failed to publish the recovering service record"),
            "{message}"
        );
        assert!(
            !message.contains("rollback also failed"),
            "corrupting content, not permissions, leaves the rollback write itself able to \
             land: {message}"
        );
        assert_eq!(
            record.status, prior.status,
            "the in-memory record must be restored to what it was before this restart"
        );
        let persisted = persisted?;
        assert_eq!(
            persisted.status, prior.status,
            "the rollback write must overwrite the corrupted content with the prior record"
        );
        assert_eq!(persisted.supervisor_pid, prior.supervisor_pid);
        Ok(())
    }

    /// As above, but the rollback write itself also fails: a manifest made
    /// read-only after the first write still lets `record_spawned_supervisor`
    /// *read* it (ownership check passes), but both its write and the
    /// subsequent rollback write hit the same permission error. That must be
    /// reported as a rollback failure, not swallowed as the original one.
    ///
    /// The manifest writes go through write-then-rename, so blocking the
    /// *directory* (not the file) is what actually denies them.
    ///
    /// Unix only and skipped as root, where a read-only mode is not enforced.
    #[cfg(unix)]
    #[test]
    fn restart_reports_when_the_rollback_write_also_fails() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        if rocm_core::openmpi::running_as_root() {
            return Ok(());
        }

        let (root, paths) = temp_app_paths("restart-rollback-also-fails");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-rollback-also-fails";
        let mut record = identity_probe_record(&paths, service_id, 11462);
        record.status = "failed".to_owned();
        record.write()?;

        let manifest_path = paths.service_manifest_path(service_id);
        let services_dir = paths.services_dir();
        let lock = {
            let services_dir = services_dir.clone();
            thread::spawn(move || -> bool {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while std::time::Instant::now() < deadline {
                    if let Ok(bytes) = fs::read(&manifest_path)
                        && let Ok(on_disk) = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
                        && on_disk.status == "recovering"
                        && on_disk.supervisor_pid == 0
                        && fs::set_permissions(
                            &services_dir,
                            std::fs::Permissions::from_mode(0o555),
                        )
                        .is_ok()
                    {
                        return true;
                    }
                    std::hint::spin_loop();
                }
                false
            })
        };
        let outcome = restart_managed_service(&paths, &mut record);
        let locked_in_time = lock.join().expect("lock thread");
        fs::set_permissions(&services_dir, std::fs::Permissions::from_mode(0o755))?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            locked_in_time,
            "precondition: the race won before the second write ran"
        );
        let error = outcome.expect_err("a read-only manifest must fail both writes");
        let message = format!("{error:#}");
        assert!(
            message.contains("rollback also failed"),
            "both the second write and its rollback hit the same permission error: {message}"
        );
        assert!(
            message.contains(service_id),
            "the rollback-also-failed message must name the affected service: {message}"
        );
        // Left exactly as the second write's failure found it: `recovering`
        // with no supervisor, matching the "no supervisor" the error names.
        assert_eq!(persisted?.status, "recovering");
        Ok(())
    }

    /// The lease branch above proves a bound loopback port refuses recovery;
    /// the doc comment on `restart_managed_service`'s lease also claims custom
    /// hosts skip that preflight entirely, relying on the engine's own bind
    /// error instead. Binding the exact recorded port on a custom host and
    /// still reaching a spawned supervisor is what proves that claim, rather
    /// than just trusting the comment.
    ///
    /// Linux-only and ignores the restart's own outcome for the same reason as
    /// `restart_managed_service_persists_the_supervisor_identity_token`: the
    /// restart re-execs the test binary, which rejects the `supervise`
    /// arguments and exits at once, so the restart itself still reports
    /// failure — after writing the supervisor identity this test reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_on_a_custom_host_skips_the_loopback_lease() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-custom-host-skips-lease");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-custom-host";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "llamacpp",
            "a-model",
            "a-model",
            "0.0.0.0",
            11460,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "failed".to_owned();
        record.write()?;

        let _blocker = std::net::TcpListener::bind(("0.0.0.0", 11460))?;

        let outcome = restart_managed_service(&paths, &mut record);
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let message = outcome.err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(
            !message.contains("its recorded port is in use"),
            "a custom host must not preflight the port with a loopback lease: {message}"
        );
        assert!(
            persisted?.supervisor_start_ticks.is_some(),
            "recovery must still reach a spawned supervisor despite the bound port"
        );
        Ok(())
    }

    /// A stop that lands while a restart waits to see whether its supervisor
    /// survived startup must outlive the restart's `failed` write: a `failed`
    /// record without the marker is exactly what recovery restarts.
    ///
    /// The restarted supervisor is the test binary, which rejects the
    /// `supervise` arguments and exits at once, so the restart reaches that
    /// write. A thread plays the stop: once the restart has recorded the
    /// spawned supervisor, it persists a marker, as a stop does first.
    #[cfg(target_os = "linux")]
    #[test]
    fn restart_failure_write_keeps_a_stop_marker_set_meanwhile() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-failure-keeps-marker");
        paths.ensure()?;
        fs::create_dir_all(paths.services_dir())?;

        let service_id = "svc-restart-failure-marker";
        let mut record = identity_probe_record(&paths, service_id, 11456);
        record.status = "failed".to_owned();
        record.write()?;

        let stop = {
            let paths = paths.clone();
            thread::spawn(move || -> Result<bool> {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while std::time::Instant::now() < deadline {
                    // `write` is not atomic, so a read can land mid-write and
                    // see a truncated file; just look again.
                    let Ok(mut on_disk) = load_service_record(&paths, service_id) else {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    };
                    if on_disk.supervisor_pid != 0 && on_disk.supervisor_pid != std::process::id() {
                        on_disk.stop_requested_unix_ms = Some(1);
                        on_disk.write()?;
                        return Ok(true);
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(false)
            })
        };
        let outcome = restart_managed_service(&paths, &mut record);
        let stop_landed = stop.join().expect("stop thread");
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        assert!(
            stop_landed?,
            "precondition: the stop ran during the restart"
        );
        assert!(outcome.is_err(), "the test binary is not a supervisor");
        let persisted = persisted?;
        assert_eq!(persisted.status, "failed");
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1),
            "the restart's failure write must not withdraw a stop issued meanwhile"
        );
        Ok(())
    }

    /// Between a restart's first write, which clears the recorded PIDs, and the
    /// one recording the supervisor it spawned, a stop can persist its marker.
    /// Writing the restart's own copy back would erase it while the new
    /// supervisor comes up — so only the identity is applied.
    #[test]
    fn restart_records_its_supervisor_without_erasing_a_stop_marker() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-spawn-keeps-marker");
        paths.ensure()?;
        let service_id = "svc-restart-spawn-marker";
        // The restart's copy after its first write: no supervisor, no marker.
        let mut restarting = identity_probe_record(&paths, service_id, 11457);
        restarting.status = "recovering".to_owned();
        restarting.supervisor_pid = 0;
        restarting.supervisor_start_ticks = None;
        // A stop started since: same (empty) processes, marker set.
        let mut on_disk = restarting.clone();
        on_disk.stop_requested_unix_ms = Some(1);
        on_disk.write()?;

        record_spawned_supervisor(&paths, &mut restarting, std::process::id())?;
        let persisted = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let persisted = persisted?;
        assert_eq!(restarting.supervisor_pid, std::process::id());
        assert_eq!(persisted.supervisor_pid, std::process::id());
        assert_eq!(
            persisted.supervisor_start_ticks,
            restarting.supervisor_start_ticks
        );
        assert_eq!(
            persisted.stop_requested_unix_ms,
            Some(1),
            "recording the spawned supervisor must not withdraw a standing stop"
        );
        Ok(())
    }

    /// The spawned supervisor may record itself before the restart does, and
    /// the record may be removed outright; either way the restart's write is
    /// not the one to land.
    #[test]
    fn restart_leaves_a_record_naming_another_supervisor_or_none() -> Result<()> {
        let (root, paths) = temp_app_paths("restart-spawn-not-owner");
        paths.ensure()?;
        let service_id = "svc-restart-spawn-not-owner";
        let mut restarting = identity_probe_record(&paths, service_id, 11458);
        restarting.supervisor_pid = 0;
        restarting.supervisor_start_ticks = None;
        let mut on_disk = restarting.clone();
        on_disk.supervisor_pid = 4_242_424;
        on_disk.write()?;

        record_spawned_supervisor(&paths, &mut restarting.clone(), std::process::id())?;
        let after_supervisor = load_service_record(&paths, service_id);

        fs::remove_file(paths.service_manifest_path(service_id))?;
        record_spawned_supervisor(&paths, &mut restarting, std::process::id())?;
        let resurrected = paths.service_manifest_path(service_id).exists();
        fs::remove_dir_all(root).ok();

        assert_eq!(after_supervisor?.supervisor_pid, 4_242_424);
        assert!(!resurrected, "a removed record must not be written back");
        Ok(())
    }
}
