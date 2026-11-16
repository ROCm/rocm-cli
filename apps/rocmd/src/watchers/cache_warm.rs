// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `cache-warm` watcher: checking that a recipe artifact is known and
//! queueing a reviewed prefetch proposal for it.

use super::policy::queue_proposal_with_arguments;
use crate::common::payload_string;
use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationRuntimeState, AutomationTriggerEvent, WatcherMode,
    resolve_model_recipe_artifact,
};
use serde_json::json;

pub(super) fn handle_cache_warm_event(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_app_paths, test_runtime_state, test_watcher_snapshot};
    use rocm_core::AutomationEventRecord;
    use serde_json::Value;
    use std::fs;

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
}
