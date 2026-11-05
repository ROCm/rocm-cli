// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationRuntimeState, AutomationTriggerEvent, CodexBridgeGpuSnapshot,
    RocmCliConfig, WatcherRuntimeSnapshot, builtin_watchers, unix_time_millis,
};
use serde_json::json;

mod cache_warm;
mod driver_upgrade;
mod gpu_metrics;
mod gpu_thermal_protect;
mod policy;
mod server_recover;
mod therock_update;

use cache_warm::handle_cache_warm_event;
use driver_upgrade::handle_driver_upgrade_event;
use gpu_metrics::{
    GPU_METRICS_INTERVAL_MS, gpu_metrics_due, gpu_pressure_events, gpu_snapshot_summary,
    handle_gpu_metrics_event,
};
use gpu_thermal_protect::{gpu_thermal_protect_due, handle_gpu_thermal_protect_event};
pub(crate) use server_recover::find_recoverable_service;
#[cfg(test)]
pub(crate) use server_recover::stop_requested;
use server_recover::{
    handle_server_recover_event, server_recover_due, service_recovery_event_kind,
};
use therock_update::{THEROCK_UPDATE_INTERVAL_MS, handle_therock_update_event, therock_update_due};

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
        crate::host_snapshot::gather_gpu_snapshot_for_config(config)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::load_service_record;
    use crate::test_support::{temp_app_paths, test_runtime_state, test_watcher_snapshot};
    use rocm_core::{AutomationEventRecord, ManagedServiceRecord, WatcherMode};
    use serde_json::Value;
    use std::fs;

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
}
