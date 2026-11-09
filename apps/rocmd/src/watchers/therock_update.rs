// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `therock-update` watcher: the scheduled update check, run read-only in
//! contained mode, and the notification it records when an update is available.

use super::policy::{WatcherPolicyAction, queue_proposal, watcher_policy_action};
use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, AutomationRuntimeState, AutomationTriggerEvent, WatcherMode};
use serde_json::Value;

pub(super) const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;

pub(super) fn handle_therock_update_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_therock_update_event_with_runner(paths, mode, state, event, |paths| {
        crate::sandbox::run_sandbox_tool(
            paths,
            crate::sandbox_tool::SandboxToolArg::CheckUpdates,
            None,
            None,
            None,
            crate::sandbox_tool::SandboxToolPolicy::default(),
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
    if tool != crate::sandbox_tool::SandboxToolArg::CheckUpdates.as_cli_value() {
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

pub(super) fn therock_update_due(state: &AutomationRuntimeState, now: u128) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use rocm_core::{AuditEventRecord, AutomationEventRecord, WatcherRuntimeSnapshot};
    use serde_json::json;
    use std::fs;

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
                    rocm_core::CommandCapture {
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
                Ok(crate::sandbox::sandbox_check_updates_value(rocm_core::CommandCapture {
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
}
