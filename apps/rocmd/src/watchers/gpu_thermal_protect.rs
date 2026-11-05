// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `gpu-thermal-protect` watcher: choosing the one running managed
//! service to relieve when the GPU is under thermal or memory pressure, and
//! queueing a reviewed stop request for it.

use super::gpu_metrics::{
    GPU_MEMORY_VRAM_PRESSURE_PERCENT, GPU_METRICS_INTERVAL_MS, GPU_THERMAL_HOTSPOT_PRESSURE_C,
    GPU_THERMAL_MEMORY_PRESSURE_C,
};
use super::policy::{pending_proposal_exists, queue_proposal_with_arguments};
use crate::common::payload_string;
use crate::persistence::load_service_record;
use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationRuntimeState, AutomationTriggerEvent, ManagedServiceRecord, WatcherMode,
};
use serde_json::{Value, json};

pub(super) fn gpu_thermal_protect_due(state: &AutomationRuntimeState, now: u128) -> bool {
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

pub(super) fn handle_gpu_thermal_protect_event(
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
    pending_proposal_exists(
        paths,
        "gpu-thermal-protect",
        service_id,
        "queue_stop_server_proposal",
        "stop_server",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        proposals_beyond_a_recent_window, temp_app_paths, test_runtime_state, test_watcher_snapshot,
    };
    use rocm_core::AutomationEventRecord;
    use std::fs;

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
    fn gpu_thermal_protect_dedupe_sees_a_pending_stop_older_than_the_recent_window() -> Result<()> {
        let (root, paths) = temp_app_paths("gpu-thermal-protect-dedupe-old");
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
        // More newer proposals than the store keeps reviewed ones (all pending
        // here, so none is compacted away), which puts the stop proposal beyond
        // a recent-window read the size of that cap. A pending proposal is
        // never dropped from the store, so it is still waiting for review.
        for index in 0..proposals_beyond_a_recent_window() {
            queue_proposal_with_arguments(
                &paths,
                "gpu-thermal-protect",
                "queue_stop_server_proposal",
                "Stop another server",
                "unrelated proposal",
                Some(format!("svc-other-{index}")),
                json!({}),
            )?;
        }
        handle_gpu_thermal_protect_event(&paths, WatcherMode::Propose, &mut state, &event)?;
        let stops_for_hot_server = rocm_core::load_automation_proposals(&paths)?
            .into_iter()
            .filter(|proposal| {
                proposal.tool.as_deref() == Some("stop_server")
                    && proposal.service_id.as_deref() == Some("svc-hot")
                    && proposal.status == "pending"
            })
            .count();
        let events = rocm_core::load_recent_automation_events(&paths, 2)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(stops_for_hot_server, 1);
        assert!(
            events
                .iter()
                .any(|event| event.action == "stop_proposal_already_pending")
        );
        Ok(())
    }
}
