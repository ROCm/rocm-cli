// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `gpu-metrics` watcher: turning an amd-smi snapshot into telemetry and
//! GPU-pressure events, and recording the telemetry events.

use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationRuntimeState, AutomationTriggerEvent, CodexBridgeGpuSnapshot, WatcherMode,
};
use serde_json::{Value, json};

pub(super) const GPU_METRICS_INTERVAL_MS: u128 = 60 * 1000;
pub(super) const GPU_THERMAL_HOTSPOT_PRESSURE_C: f64 = 95.0;
pub(super) const GPU_THERMAL_MEMORY_PRESSURE_C: f64 = 95.0;
pub(super) const GPU_MEMORY_VRAM_PRESSURE_PERCENT: f64 = 95.0;

pub(super) fn gpu_metrics_due(state: &AutomationRuntimeState, now: u128) -> bool {
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

pub(super) fn handle_gpu_metrics_event(
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

pub(super) fn gpu_snapshot_summary(snapshot: &CodexBridgeGpuSnapshot) -> String {
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

pub(super) fn gpu_pressure_events(
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
