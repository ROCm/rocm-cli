// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `driver-upgrade` watcher: turning an available driver update into a
//! reviewed driver-plan proposal, or a read-only plan in contained mode.

use super::policy::queue_proposal;
use crate::common::payload_string;
use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, AutomationRuntimeState, AutomationTriggerEvent, WatcherMode};
use serde_json::Value;

pub(super) fn handle_driver_upgrade_event(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
    event: &AutomationTriggerEvent,
) -> Result<()> {
    handle_driver_upgrade_event_with_runner(paths, mode, state, event, |paths| {
        crate::sandbox::run_sandbox_tool(
            paths,
            crate::sandbox_tool::SandboxToolArg::DriverPlan,
            None,
            None,
            None,
            crate::sandbox_tool::SandboxToolPolicy::default(),
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
    if tool != crate::sandbox_tool::SandboxToolArg::DriverPlan.as_cli_value() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_app_paths, test_runtime_state, test_watcher_snapshot};
    use rocm_core::AutomationEventRecord;
    use serde_json::json;
    use std::fs;

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
                    rocm_core::CommandCapture {
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
}
