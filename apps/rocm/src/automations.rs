// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm automations` command handler, the automation-status/proposal
//! rendering it prints, and the `rocm daemon --status` background-helper
//! summary.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays `automations(command)` (re-imported via
//! `use crate::automations::automations;`). As of Phase 6f (ROCMAI-210, see
//! `docs/architecture.md`) this module also owns the rendering/classification
//! helpers behind `render_automations_text` (the `rocm automations` status
//! report: watcher status/policy lines, recent automation activity, and
//! plain-language proposal/audit summaries) and `render_daemon_text` (the
//! `rocm daemon --status` summary) — both folded in from `main.rs` alongside
//! the `update_cmd.rs` split in the same PR.
//!
//! This module owns: `render_automations_text`, the `ProposalKind`
//! classification it renders through (`proposal_kind`,
//! `fallback_tool_for_automation_action`, `proposal_plain_summary`,
//! `proposal_plain_reason`), the small plain-language label helpers
//! (`proposal_status_label`, `proposal_bool_argument`,
//! `automation_event_plain_summary`, `audit_event_plain_summary`,
//! `watcher_mode_plain_label`, `watcher_last_check`, `watcher_plain_name`,
//! `watcher_plain_trigger`, `watcher_plain_action`, `watcher_policy_note`),
//! and `render_daemon_text`. `render_automations_text` and
//! `render_daemon_text` are `pub(crate)`: `main.rs`'s `dispatch()` calls
//! `render_daemon_text` directly (re-imported via
//! `use crate::automations::render_daemon_text;`), and `chat.rs`'s
//! deterministic tool summaries call both. Everything else here has no
//! caller outside this module (including `watcher_policy_note`, now that the
//! `automations()` handler above reaches it locally rather than through a
//! crate-root re-import) and stays private.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching the shared `test_paths` fixture through `crate::tests::` the
//! same way other mechanically relocated modules do.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use rocm_core::{
    AppPaths, AuditEventRecord, AutomationEventRecord, AutomationProposalRecord,
    AutomationRuntimeState, RocmCliConfig, WatcherMode, builtin_watcher, builtin_watchers,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
};

use crate::service_launch::ensure_background_helper_running;
use crate::{AutomationsCommand, format_bytes};

pub(crate) fn automations(command: Option<AutomationsCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(AutomationsCommand::List) {
        AutomationsCommand::List => {
            print!("{}", render_automations_text(&paths, &config)?);
        }
        AutomationsCommand::Enable { watcher, mode } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = true;
            if let Some(mode) = mode {
                entry.mode = Some(mode.into());
            }
            config.automations.daemon_enabled = true;
            config.save(&paths)?;
            println!("automation watcher enabled");
            println!("  watcher: {}", spec.id);
            println!("  mode: {}", config.effective_watcher_mode(spec).as_str());
            println!("  trigger: {}", spec.trigger);
            if let Some(note) = watcher_policy_note(spec.id) {
                println!("  policy: {note}");
            }
            println!("  config: {}", paths.config_path().display());
            ensure_background_helper_running()?;
        }
        AutomationsCommand::Disable { watcher } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = false;
            if !config
                .automations
                .watchers
                .values()
                .any(|watcher| watcher.enabled)
            {
                config.automations.daemon_enabled = false;
            }
            config.save(&paths)?;
            println!("automation watcher disabled");
            println!("  watcher: {}", spec.id);
            println!("  config: {}", paths.config_path().display());
        }
    }
    Ok(())
}

pub(crate) fn render_automations_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let recent_events = load_recent_automation_events(paths, 5).unwrap_or_default();
    let recent_proposals = load_recent_automation_proposals(paths, 5).unwrap_or_default();
    let recent_audit_events = load_recent_audit_events(paths, 5).unwrap_or_default();
    let mut output = String::new();
    let _ = writeln!(output, "automation checks");
    let _ = writeln!(output, "  config: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  background checks: {}",
        if config.automation_daemon_enabled() {
            "on"
        } else {
            "off"
        }
    );
    if let Some(state) = runtime_state.as_ref() {
        let _ = writeln!(
            output,
            "  background service: {}",
            if state.running { "running" } else { "stopped" }
        );
        let _ = writeln!(
            output,
            "  local event intake: {}",
            state
                .local_webhook_endpoint
                .as_deref()
                .unwrap_or("disabled")
        );
    } else {
        let _ = writeln!(output, "  background service: not running");
        let _ = writeln!(output, "  local event intake: disabled");
    }
    for watcher in builtin_watchers() {
        let runtime_snapshot = runtime_state.as_ref().and_then(|state| {
            state
                .active_watchers
                .iter()
                .find(|item| item.id == watcher.id)
        });
        let _ = writeln!(
            output,
            "  {} ({})",
            watcher_plain_name(watcher.id),
            if config.watcher_enabled(watcher) {
                "on"
            } else {
                "off"
            }
        );
        let _ = writeln!(
            output,
            "    setting: {}",
            watcher_mode_plain_label(config.effective_watcher_mode(watcher))
        );
        let _ = writeln!(
            output,
            "    listens for: {}",
            watcher_plain_trigger(watcher.id)
        );
        let _ = writeln!(output, "    does: {}", watcher_plain_action(watcher.id));
        if let Some(note) = watcher_policy_note(watcher.id) {
            let _ = writeln!(output, "    policy: {note}");
        }
        if let Some(snapshot) = runtime_snapshot {
            let _ = writeln!(output, "    last check: {}", watcher_last_check(snapshot));
        }
    }
    if !recent_events.is_empty() {
        let _ = writeln!(output, "  recent automation activity:");
        for event in recent_events {
            let _ = writeln!(output, "    {}", automation_event_plain_summary(&event));
            if let Some(service_id) = event.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
        }
    }
    if !recent_proposals.is_empty() {
        let _ = writeln!(output, "  recent review requests:");
        for proposal in recent_proposals {
            let _ = writeln!(
                output,
                "    {} [{}] {}",
                proposal.proposal_id,
                proposal_status_label(&proposal.status),
                proposal_plain_summary(&proposal)
            );
            let _ = writeln!(output, "      why: {}", proposal_plain_reason(&proposal));
            if let Some(service_id) = proposal.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
            if let Some(artifact_ref) = proposal
                .arguments
                .get("artifact_ref")
                .and_then(serde_json::Value::as_str)
            {
                let _ = writeln!(output, "      model file: {artifact_ref}");
            }
            if proposal_bool_argument(&proposal, "allow_artifact_download") {
                let limit = proposal
                    .arguments
                    .get("artifact_max_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .map_or_else(|| "not set".to_owned(), format_bytes);
                let _ = writeln!(output, "      download: approved up to {limit}");
            } else if proposal_kind(&proposal) == ProposalKind::PrefetchArtifact {
                let _ = writeln!(output, "      download: not approved yet");
            }
            if proposal_kind(&proposal) == ProposalKind::DriverPlan {
                let _ = writeln!(
                    output,
                    "      effect: show a driver plan only; no driver install"
                );
            }
            if proposal.status == "pending" {
                let _ = writeln!(
                    output,
                    "      controls: /automations approve {} | /automations reject {}",
                    proposal.proposal_id, proposal.proposal_id
                );
            }
        }
    }
    if !recent_audit_events.is_empty() {
        let _ = writeln!(output, "  recent background activity:");
        for event in recent_audit_events {
            let _ = writeln!(output, "    {}", audit_event_plain_summary(&event));
            if let Some(service_id) = event.service_id.as_deref() {
                let _ = writeln!(output, "      server: {service_id}");
            }
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalKind {
    RestartServer,
    StopServer,
    CheckUpdates,
    PrefetchArtifact,
    DriverPlan,
    Other,
}

fn proposal_kind(proposal: &AutomationProposalRecord) -> ProposalKind {
    match proposal
        .tool
        .as_deref()
        .or_else(|| fallback_tool_for_automation_action(&proposal.action))
    {
        Some("restart_server") => ProposalKind::RestartServer,
        Some("stop_server") => ProposalKind::StopServer,
        Some("check_updates") => ProposalKind::CheckUpdates,
        Some("prefetch_artifact") => ProposalKind::PrefetchArtifact,
        Some("driver_plan") => ProposalKind::DriverPlan,
        _ => ProposalKind::Other,
    }
}

fn fallback_tool_for_automation_action(action: &str) -> Option<&'static str> {
    match action {
        "queue_restart_proposal" => Some("restart_server"),
        "queue_stop_server_proposal" => Some("stop_server"),
        "queue_update_proposal" => Some("check_updates"),
        "queue_prefetch_proposal" => Some("prefetch_artifact"),
        "prepare_driver_plan" => Some("driver_plan"),
        _ => None,
    }
}

fn proposal_plain_summary(proposal: &AutomationProposalRecord) -> &'static str {
    match proposal_kind(proposal) {
        ProposalKind::RestartServer => "Restart a model server",
        ProposalKind::StopServer => "Stop a model server",
        ProposalKind::CheckUpdates => "Check for ROCm updates",
        ProposalKind::PrefetchArtifact => "Prepare a model file",
        ProposalKind::DriverPlan => "Show a driver plan",
        ProposalKind::Other => "Review an automation request",
    }
}

fn proposal_plain_reason(proposal: &AutomationProposalRecord) -> &'static str {
    match proposal_kind(proposal) {
        ProposalKind::RestartServer => "A managed server looks stopped or unhealthy.",
        ProposalKind::StopServer => "GPU pressure is high and serving should be reviewed.",
        ProposalKind::CheckUpdates => "A scheduled update check is due.",
        ProposalKind::PrefetchArtifact => "rocm-cli was asked to prepare this model file.",
        ProposalKind::DriverPlan => "A driver update signal was received.",
        ProposalKind::Other => "An enabled automation asked for review.",
    }
}

fn proposal_status_label(status: &str) -> &'static str {
    match status {
        "pending" => "waiting for review",
        "approved" => "approved",
        "completed" => "done",
        "rejected" => "rejected",
        "failed" => "failed",
        _ => "status unknown",
    }
}

fn proposal_bool_argument(proposal: &AutomationProposalRecord, key: &str) -> bool {
    proposal
        .arguments
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn automation_event_plain_summary(event: &AutomationEventRecord) -> &'static str {
    match event.watcher_id.as_str() {
        "therock-update" => "ROCm update check recorded.",
        "server-recover" => "Server recovery check recorded.",
        "gpu-metrics" => "GPU status check recorded.",
        "gpu-thermal-protect" => "GPU pressure check recorded.",
        "cache-warm" => "Model file preparation request recorded.",
        "driver-upgrade" => "Driver update review recorded.",
        _ => "Automation activity recorded.",
    }
}

fn audit_event_plain_summary(event: &AuditEventRecord) -> &'static str {
    match event.category.as_str() {
        "automation" => "Automation activity was recorded.",
        "proposal" => "A review request changed status.",
        "provider" => "Provider request completed.",
        "service" => "Managed server activity was recorded.",
        "install" => "Install activity was recorded.",
        "update" => "Update activity was recorded.",
        "runtime" => "ROCm runtime activity was recorded.",
        "engine" => "Engine activity was recorded.",
        _ => "Background activity was recorded.",
    }
}

const fn watcher_mode_plain_label(mode: WatcherMode) -> &'static str {
    match mode {
        WatcherMode::Observe => "record only",
        WatcherMode::Propose => "ask before taking action",
        WatcherMode::Contained => "ask before changes; keep actions limited",
    }
}

fn watcher_last_check(snapshot: &rocm_core::WatcherRuntimeSnapshot) -> &'static str {
    let Some(event) = snapshot.last_event.as_deref() else {
        return "not yet";
    };
    match event {
        "remind_update_check" => "ROCm update check recorded",
        "queue_update_proposal" => "ROCm update review requested",
        "collect_failure_snapshot" => "server health check recorded",
        "restart_managed_service" | "queue_restart_proposal" => "server restart review requested",
        "record_gpu_metrics" => "GPU status check recorded",
        "queue_stop_server_proposal" => "GPU pressure review requested",
        "queue_prefetch_proposal" => "model file review requested",
        "prepare_driver_plan" => "driver plan review requested",
        _ => "activity recorded",
    }
}

fn watcher_plain_name(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "ROCm update checks",
        "server-recover" => "Server recovery",
        "gpu-metrics" => "GPU status checks",
        "gpu-thermal-protect" => "GPU pressure protection",
        "cache-warm" => "Model file preparation",
        "driver-upgrade" => "Driver update review",
        _ => "Automation check",
    }
}

fn watcher_plain_trigger(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "a scheduled check",
        "server-recover" => "a server that stops or becomes unhealthy",
        "gpu-metrics" => "local GPU status updates",
        "gpu-thermal-protect" => "high GPU temperature or memory pressure",
        "cache-warm" => "a request to prepare a model file",
        "driver-upgrade" => "a driver update signal",
        _ => "local automation activity",
    }
}

fn watcher_plain_action(watcher_id: &str) -> &'static str {
    match watcher_id {
        "therock-update" => "checks for ROCm updates without installing them",
        "server-recover" => "asks before restarting a managed server",
        "gpu-metrics" => "records local GPU status only",
        "gpu-thermal-protect" => "asks before stopping a managed server",
        "cache-warm" => "asks before preparing or downloading a model file",
        "driver-upgrade" => "shows a driver plan only",
        _ => "records the event",
    }
}

fn watcher_policy_note(watcher_id: &str) -> Option<&'static str> {
    match watcher_id {
        "gpu-metrics" => Some(
            "read-only telemetry; propose/contained modes record events only and do not create review requests or mutate services",
        ),
        "gpu-thermal-protect" => Some(
            "GPU pressure protection is review-gated; propose/contained modes queue a stop review only and never stop servers automatically",
        ),
        "cache-warm" => Some(
            "artifact prefetch stays review-gated; contained mode creates a review request instead of downloading without reviewed source policy",
        ),
        "driver-upgrade" => Some(
            "local driver update signals stay review-gated; review requests run a read-only driver plan and never install drivers automatically",
        ),
        _ => None,
    }
}

pub(crate) fn render_daemon_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let mut output = String::new();
    let _ = writeln!(output, "Background helper");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Status: {}",
        match runtime_state {
            Some(ref state) if state.running => "running",
            _ => "not running",
        }
    );
    let _ = writeln!(output, "Startup: on demand");
    let _ = writeln!(
        output,
        "Why it runs: automation checks and local model servers use it when needed."
    );
    let _ = writeln!(
        output,
        "Automation checks: {}",
        if config.automation_daemon_enabled() {
            "on"
        } else {
            "off"
        }
    );
    let _ = writeln!(output, "Saved state: kept on this computer");
    let _ = writeln!(output);
    let _ = writeln!(output, "Choose Automations to review background checks.");
    let _ = writeln!(output, "Choose Local servers to see local model servers.");
    output
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;

    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn render_automations_text_marks_gpu_metrics_read_only() -> Result<()> {
        let (root, paths) = test_paths("automation-status-policy");
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-metrics");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        config.automations.daemon_enabled = true;

        let rendered = render_automations_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("GPU status checks (on)"));
        assert!(rendered.contains("setting: ask before changes; keep actions limited"));
        assert!(rendered.contains("listens for: local GPU status updates"));
        assert!(rendered.contains("does: records local GPU status only"));
        assert!(rendered.contains("policy: read-only telemetry"));
        assert!(rendered.contains("do not create review requests or mutate services"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("gpu-metrics"));
        Ok(())
    }

    #[test]
    fn render_automations_text_marks_gpu_thermal_protect_review_gated() -> Result<()> {
        let (root, paths) = test_paths("automation-gpu-thermal-policy");
        let mut config = RocmCliConfig::default();
        let watcher = config.watcher_config_mut("gpu-thermal-protect");
        watcher.enabled = true;
        watcher.mode = Some(WatcherMode::Contained);
        config.automations.daemon_enabled = true;

        let rendered = render_automations_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("GPU pressure protection (on)"));
        assert!(rendered.contains("setting: ask before changes; keep actions limited"));
        assert!(rendered.contains("listens for: high GPU temperature or memory pressure"));
        assert!(rendered.contains("does: asks before stopping a managed server"));
        assert!(rendered.contains("policy: GPU pressure protection is review-gated"));
        assert!(rendered.contains("never stop servers automatically"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("gpu-thermal-protect"));
        Ok(())
    }

    #[test]
    fn render_automations_text_surfaces_local_webhook_endpoint() -> Result<()> {
        let (root, paths) = test_paths("automation-local-webhook");
        AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 123,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 2,
            local_webhook_endpoint: Some("http://127.0.0.1:19191/automation-events".to_owned()),
            active_watchers: vec![rocm_core::WatcherRuntimeSnapshot {
                id: "gpu-metrics".to_owned(),
                enabled: true,
                mode: WatcherMode::Observe,
                summary: "record metrics".to_owned(),
                last_event: Some("record_gpu_metrics".to_owned()),
                last_event_unix_ms: Some(2),
            }],
        }
        .write(&paths)?;

        let rendered = render_automations_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("background service: running"));
        assert!(rendered.contains("local event intake: http://127.0.0.1:19191/automation-events"));
        assert!(rendered.contains("last check: GPU status check recorded"));
        assert!(!rendered.contains("pid=123"));
        assert!(!rendered.contains("last_tick_unix_ms"));
        assert!(!rendered.contains("local_webhook_endpoint"));
        assert!(!rendered.contains("record_gpu_metrics"));
        assert!(!rendered.contains("gpu-metrics"));
        Ok(())
    }

    #[test]
    fn render_automations_text_uses_plain_proposal_history() -> Result<()> {
        let (root, paths) = test_paths("automation-plain-proposal-history");
        rocm_core::append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 42,
                proposal_id: "proposal-plain".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "queue restart proposal".to_owned(),
                message: "backend-authored recovery message".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-plain".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-plain" }),
                reviewed_at_unix_ms: None,
            },
        )?;
        rocm_core::append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 43,
                proposal_id: "proposal-file".to_owned(),
                watcher_id: "cache-warm".to_owned(),
                action: "queue_prefetch_proposal".to_owned(),
                title: "queue prefetch proposal".to_owned(),
                message: "cache warm requested internal artifact prefetch".to_owned(),
                status: "pending".to_owned(),
                service_id: None,
                tool: Some("prefetch_artifact".to_owned()),
                arguments: serde_json::json!({
                    "artifact_ref": "tiny/model#gguf",
                    "allow_artifact_download": true,
                    // One byte under 1 MiB: `{:.1}` rounds it up to a full
                    // 1024 KiB, so the approved cap must read as "1.0 MiB",
                    // not "1024.0 KiB". It must also read as `MiB`, not `MB`
                    // — this is base-1024 math, not base-1000 (issue #190).
                    "artifact_max_bytes": 1_048_575
                }),
                reviewed_at_unix_ms: None,
            },
        )?;
        rocm_core::append_automation_event(
            &paths,
            &rocm_core::AutomationEventRecord {
                at_unix_ms: 44,
                watcher_id: "gpu-metrics".to_owned(),
                level: "info".to_owned(),
                action: "record_gpu_metrics".to_owned(),
                message: "raw amd-smi telemetry event".to_owned(),
                service_id: None,
            },
        )?;
        rocm_core::append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 45,
                source: "test".to_owned(),
                category: "proposal".to_owned(),
                actor: "watcher:server-recover".to_owned(),
                level: "info".to_owned(),
                action: "proposal_approved".to_owned(),
                message: "approved proposal-plain with backend detail".to_owned(),
                watcher_id: Some("server-recover".to_owned()),
                service_id: Some("svc-plain".to_owned()),
            },
        )?;

        let rendered = render_automations_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("recent automation activity:"));
        assert!(rendered.contains("GPU status check recorded."));
        assert!(rendered.contains("recent review requests:"));
        assert!(rendered.contains("proposal-plain [waiting for review] Restart a model server"));
        assert!(rendered.contains("why: A managed server looks stopped or unhealthy."));
        assert!(rendered.contains("server: svc-plain"));
        assert!(rendered.contains("proposal-file [waiting for review] Prepare a model file"));
        assert!(rendered.contains("model file: tiny/model#gguf"));
        assert!(rendered.contains("download: approved up to 1.0 MiB"));
        assert!(rendered.contains(
            "controls: /automations approve proposal-plain | /automations reject proposal-plain"
        ));
        assert!(rendered.contains("recent background activity:"));
        assert!(rendered.contains("A review request changed status."));
        assert!(!rendered.contains("queue_restart_proposal"));
        assert!(!rendered.contains("restart_server"));
        assert!(!rendered.contains("queue restart proposal"));
        assert!(!rendered.contains("backend-authored recovery message"));
        assert!(!rendered.contains("record_gpu_metrics"));
        assert!(!rendered.contains("raw amd-smi telemetry event"));
        assert!(!rendered.contains("proposal_approved"));
        assert!(!rendered.contains("approved proposal-plain with backend detail"));
        assert!(!rendered.contains("created_unix_ms"));
        assert!(!rendered.contains("last_tick_unix_ms"));
        assert!(!rendered.contains("local_webhook_endpoint"));
        assert!(!rendered.contains("command id:"));
        assert!(!rendered.contains("mode:"));
        assert!(!rendered.contains("1048576 bytes"));
        assert!(!rendered.contains("recent events:"));
        assert!(!rendered.contains("recent audit:"));
        assert!(!rendered.contains("tool:"));
        Ok(())
    }
}
