// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The watcher policy: what a watcher does in each mode, and the automation
//! proposals it queues for review.

use anyhow::Result;
use rocm_core::{
    AppPaths, AutomationProposalRecord, WatcherMode, append_automation_proposal, unix_time_millis,
};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum WatcherPolicyAction {
    Observe,
    QueueProposal,
    RunContained,
}

pub(super) const fn watcher_policy_action(
    watcher_id: &str,
    mode: WatcherMode,
) -> WatcherPolicyAction {
    match (watcher_id, mode) {
        (_, WatcherMode::Observe) => WatcherPolicyAction::Observe,
        (_, WatcherMode::Propose) => WatcherPolicyAction::QueueProposal,
        (_, WatcherMode::Contained) => WatcherPolicyAction::RunContained,
    }
}

pub(super) fn queue_proposal(
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

pub(super) fn queue_proposal_with_arguments(
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

/// Whether `watcher_id` already has a pending proposal for `service_id`, either
/// queued under `action` or carrying `tool`.
///
/// Reads the whole proposals store rather than a recent window: a pending
/// proposal is never dropped, so it can be arbitrarily old, and a window that
/// missed it would let the watcher queue a duplicate once the original aged
/// out of that window. Compaction bounds the reviewed proposals to a fixed
/// count; pending ones are kept.
pub(super) fn pending_proposal_exists(
    paths: &AppPaths,
    watcher_id: &str,
    service_id: &str,
    action: &str,
    tool: &str,
) -> Result<bool> {
    Ok(rocm_core::load_automation_proposals(paths)?
        .into_iter()
        .any(|proposal| {
            proposal.status == "pending"
                && proposal.watcher_id == watcher_id
                && proposal.service_id.as_deref() == Some(service_id)
                && (proposal.action == action || proposal.tool.as_deref() == Some(tool))
        }))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
