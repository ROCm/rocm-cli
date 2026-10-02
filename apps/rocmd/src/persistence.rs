// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, AuditEventRecord, AutomationEventRecord, AutomationRuntimeState,
    ManagedServiceRecord, append_audit_event, append_automation_event, unix_time_millis,
};
use std::fs;

pub(crate) fn record_event(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
    watcher_id: &str,
    level: &str,
    action: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    let now = unix_time_millis();
    if let Some(snapshot) = state.watcher_mut(watcher_id) {
        snapshot.last_event = Some(message.to_owned());
        snapshot.last_event_unix_ms = Some(now);
    }
    let event = AutomationEventRecord {
        at_unix_ms: now,
        watcher_id: watcher_id.to_owned(),
        level: level.to_owned(),
        action: action.to_owned(),
        message: message.to_owned(),
        service_id,
    };
    append_automation_event(paths, &event)?;

    let audit_watcher_id = (watcher_id != "rocmd").then(|| watcher_id.to_owned());
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: now,
            source: "rocmd".to_owned(),
            category: "automation".to_owned(),
            actor: audit_watcher_id
                .as_deref()
                .map_or_else(|| "rocmd".to_owned(), |id| format!("watcher:{id}")),
            level: level.to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            watcher_id: audit_watcher_id,
            service_id: event.service_id,
        },
    )
}

pub(crate) fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) {
            records.push(record);
        }
    }

    records.sort_by_key(|record| std::cmp::Reverse(record.created_at_unix_ms));
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use rocm_core::{WatcherMode, WatcherRuntimeSnapshot};

    #[test]
    fn record_event_mirrors_watcher_actions_to_audit_log() -> Result<()> {
        let (root, paths) = temp_app_paths("record-event-audit");
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
                mode: WatcherMode::Contained,
                summary: "recover failed managed services".to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }],
        };

        record_event(
            &paths,
            &mut state,
            "server-recover",
            "info",
            "restart_managed_service",
            "restarted failed managed service svc-1",
            Some("svc-1".to_owned()),
        )?;

        let automation_text = fs::read_to_string(paths.automation_events_path())?;
        let automation_event =
            serde_json::from_str::<AutomationEventRecord>(automation_text.trim())?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        let audit_event = serde_json::from_str::<AuditEventRecord>(audit_text.trim())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(automation_event.watcher_id, "server-recover");
        assert_eq!(automation_event.action, "restart_managed_service");
        assert_eq!(audit_event.category, "automation");
        assert_eq!(audit_event.actor, "watcher:server-recover");
        assert_eq!(audit_event.watcher_id.as_deref(), Some("server-recover"));
        assert_eq!(audit_event.service_id.as_deref(), Some("svc-1"));
        Ok(())
    }
}
