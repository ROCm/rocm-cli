// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `server-recover` watcher: finding a managed service that needs
//! recovery, and restarting it or queueing a restart proposal.

use super::policy::{
    WatcherPolicyAction, pending_proposal_exists, queue_proposal, watcher_policy_action,
};
use crate::persistence::load_service_record;
use crate::supervise::restart_managed_service;
use anyhow::{Context, Result};
use rocm_core::{
    AppPaths, AutomationRuntimeState, AutomationTriggerEvent, ManagedServiceRecord, WatcherMode,
    format_host_port, unix_time_millis,
};
#[cfg(test)]
use serde_json::json;
use std::net::{SocketAddr, TcpStream};
use std::thread;
use std::time::Duration;

const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const SERVER_TRANSIENT_STALE_MS: u128 = 5 * 60 * 1_000;
const ENDPOINT_HEALTH_TIMEOUT: Duration = Duration::from_millis(250);

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

pub(super) fn handle_server_recover_event(
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
        "service.healthcheck_recoverable" => crate::engine_ipc::engine_healthcheck_response(
            paths,
            &record.engine,
            &record.service_id,
        )
        .is_ok_and(|healthcheck| crate::engine_ipc::healthcheck_response_recoverable(&healthcheck)),
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
            // An unreadable proposals store is reported and skipped, never
            // propagated: the daemon loop exits on an error out of
            // `evaluate_watchers`, so a store that stays corrupt would take it
            // down on every recovery tick.
            match pending_restart_proposal_exists(paths, &record.service_id) {
                Ok(true) => {
                    return crate::persistence::record_event(
                        paths,
                        state,
                        "server-recover",
                        "info",
                        "restart_proposal_already_pending",
                        &format!(
                            "managed service {} needs recovery ({recovery_reason_display}); a restart request awaiting review is already queued",
                            record.service_id,
                        ),
                        Some(record.service_id.clone()),
                    );
                }
                Ok(false) => {}
                Err(error) => {
                    return crate::persistence::record_event(
                        paths,
                        state,
                        "server-recover",
                        "error",
                        "restart_proposal_check_failed",
                        &format!(
                            "managed service {} needs recovery ({recovery_reason_display}); \
                             not queueing a restart request because the proposals store could \
                             not be checked for one already pending: {error:#}",
                            record.service_id,
                        ),
                        Some(record.service_id.clone()),
                    );
                }
            }
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
            if let Err(error) = crate::engine_ipc::ensure_public_service_has_endpoint_key(
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
            restart_managed_service(paths, &mut *record)?;
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

pub(super) fn service_recovery_event_kind(recovery_reason: &str) -> &'static str {
    if recovery_reason.starts_with("healthcheck_status_") {
        "service.healthcheck_recoverable"
    } else if recovery_reason.starts_with("endpoint_status_") {
        "service.endpoint_recoverable"
    } else {
        "service.manifest_recoverable"
    }
}

pub(super) fn server_recover_due(state: &AutomationRuntimeState, now: u128) -> bool {
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
            let Ok(healthcheck) = crate::engine_ipc::engine_healthcheck_response(
                paths,
                &record.engine,
                &record.service_id,
            ) else {
                if let Some(reason) = endpoint_service_recovery_reason(&record) {
                    return Ok(Some((record, reason)));
                }
                continue;
            };
            if crate::engine_ipc::healthcheck_response_recoverable(&healthcheck) {
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
    (!wait_for_port(&record.host, record.port, ENDPOINT_HEALTH_TIMEOUT))
        .then(|| "endpoint_status_unreachable".to_owned())
}

fn wait_for_port(host: &str, port: u16, timeout: Duration) -> bool {
    let address: SocketAddr = match format_host_port(host, port).parse() {
        Ok(value) => value,
        Err(_) => return false,
    };

    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
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

fn pending_restart_proposal_exists(paths: &AppPaths, service_id: &str) -> Result<bool> {
    pending_proposal_exists(
        paths,
        "server-recover",
        service_id,
        "queue_restart_proposal",
        "restart_server",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        UNCONFIRMED_STOP_PID, proposals_beyond_a_recent_window, seed_keyed_service,
        stop_with_outcome, temp_app_paths, test_runtime_state, test_watcher_snapshot,
    };
    use crate::watchers::policy::queue_proposal_with_arguments;
    use rocm_core::{AutomationProposalRecord, WatcherRuntimeSnapshot, append_automation_proposal};
    use serde_json::Value;
    use std::fs;

    #[test]
    fn server_recover_does_not_duplicate_pending_restart_proposals() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-dedupe");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-unhealthy",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "unreachable".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };

        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        )?;
        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        )?;
        let events = rocm_core::load_recent_automation_events(&paths, 2)?;
        let proposals = rocm_core::load_recent_automation_proposals(&paths, 10)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].tool.as_deref(), Some("restart_server"));
        // The wording describes a proposal still awaiting review; the single
        // pending proposal asserted here is what makes that true.
        assert_eq!(proposals[0].status, "pending");
        assert!(events.iter().any(|event| {
            event.action == "restart_proposal_already_pending"
                && event.message.contains("a restart request awaiting review")
        }));
        Ok(())
    }

    #[test]
    fn server_recover_dedupe_sees_a_pending_restart_older_than_the_recent_window() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-dedupe-old");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-unhealthy",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "unreachable".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };

        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        )?;
        // More newer proposals than the store keeps reviewed ones (all pending
        // here, so none is compacted away), which puts the restart proposal
        // beyond a recent-window read the size of that cap.
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
        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        )?;
        let restarts = rocm_core::load_automation_proposals(&paths)?
            .into_iter()
            .filter(|proposal| proposal.tool.as_deref() == Some("restart_server"))
            .count();
        fs::remove_dir_all(root).ok();

        assert_eq!(restarts, 1);
        Ok(())
    }

    #[test]
    fn server_recover_dedupe_ignores_proposals_that_do_not_block_a_restart() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-dedupe-filters");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-unhealthy",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "unreachable".to_owned();
        record.write()?;
        // Each of these matches the restart check on all but one field, so
        // none of them may suppress a new restart for `svc-unhealthy`.
        for (watcher_id, service_id, status) in [
            ("server-recover", "svc-unhealthy", "approved"),
            ("server-recover", "svc-other", "pending"),
            ("another-watcher", "svc-unhealthy", "pending"),
        ] {
            append_automation_proposal(
                &paths,
                &AutomationProposalRecord {
                    at_unix_ms: unix_time_millis(),
                    proposal_id: String::new(),
                    watcher_id: watcher_id.to_owned(),
                    action: "queue_restart_proposal".to_owned(),
                    title: "Restart".to_owned(),
                    message: "unrelated".to_owned(),
                    status: status.to_owned(),
                    service_id: Some(service_id.to_owned()),
                    tool: Some("restart_server".to_owned()),
                    arguments: json!({}),
                    reviewed_at_unix_ms: None,
                },
            )?;
        }
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };

        handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        )?;
        let queued = rocm_core::load_automation_proposals(&paths)?
            .into_iter()
            .filter(|proposal| {
                proposal.watcher_id == "server-recover"
                    && proposal.status == "pending"
                    && proposal.service_id.as_deref() == Some("svc-unhealthy")
            })
            .count();
        fs::remove_dir_all(root).ok();

        assert_eq!(queued, 1);
        Ok(())
    }

    #[test]
    fn server_recover_reports_an_unreadable_proposals_store_instead_of_failing() -> Result<()> {
        let (root, paths) = temp_app_paths("server-recover-corrupt-store");
        paths.ensure()?;
        fs::write(paths.automation_proposals_path(), "not a proposal\n")?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-unhealthy",
            "vllm",
            "tiny",
            "Tiny/Test",
            "127.0.0.1",
            11436,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.status = "unreachable".to_owned();
        record.write()?;
        let mut state = test_runtime_state(vec![test_watcher_snapshot(
            "server-recover",
            WatcherMode::Propose,
            None,
        )]);
        let event = AutomationTriggerEvent {
            at_unix_ms: 1,
            kind: "service.manifest_recoverable".to_owned(),
            source: "managed_service".to_owned(),
            watcher_hint: Some("server-recover".to_owned()),
            service_id: Some(record.service_id.clone()),
            reason: Some("manifest_status_failed".to_owned()),
            payload: json!({}),
        };

        let outcome = handle_server_recover_event_with_record(
            &paths,
            WatcherMode::Propose,
            &mut state,
            &event,
            &mut record,
        );
        let events = rocm_core::load_recent_automation_events(&paths, 1)?;
        let stored = fs::read_to_string(paths.automation_proposals_path())?;
        fs::remove_dir_all(root).ok();

        // An error out of a watcher ends the daemon loop, so a corrupt store
        // must surface as an event and a skipped tick, not as `Err`.
        outcome?;
        assert_eq!(stored, "not a proposal\n", "nothing may be queued");
        assert!(events.iter().any(|event| {
            event.level == "error" && event.action == "restart_proposal_check_failed"
        }));
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
}
