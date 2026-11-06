// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Automation-daemon runtime state and its on-disk records.
//!
//! The running daemon's live snapshot (`AutomationRuntimeState`), the
//! logically append-only JSONL stores for watcher action events
//! (`AutomationEventRecord`), audit log entries, and review proposals, and
//! the accessors used to read them back. The built-in watcher catalog
//! (`BuiltinWatcherSpec`/`builtin_watcher(s)`) lives in `config.rs`, next to
//! `WatcherMode`, which it and this module both read.
//!
//! The `rocm` CLI and `rocmd` write these stores concurrently. The machinery
//! that makes that safe — the per-store `.lock` file (`jsonl_lock_path`),
//! single-line appends, torn-tail repair, rotation and the tail-first readers
//! — is the generic JSONL engine in `jsonl_store.rs`; this module owns which
//! stores exist, their record types and the policy applied to each.
//!
//! The events and audit stores are additionally capped by size-based rotation
//! (see [`EVENT_LOG_ROTATE_BYTES`]): an append that finds the live file at the
//! cap rotates it aside first, and the recent-record readers span the rotated
//! generations. The proposals store is never rotated, because a pending
//! proposal must stay findable by id until it is reviewed; instead the status
//! rewrite compacts it, dropping all but the newest
//! [`REVIEWED_PROPOSALS_RETAINED`] reviewed proposals.
//!
//! `AutomationTriggerEvent` is also defined here, but this module persists
//! none of it: it is the in-memory trigger type passed between `rocmd`'s
//! watcher and webhook plumbing (`apps/rocmd/src/watchers.rs`,
//! `apps/rocmd/src/webhook.rs`), never serialized.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppPaths;
use crate::FileLock;
use crate::WatcherMode;
use crate::jsonl_store::{
    JsonlRotation, append_jsonl_record, jsonl_lock_path, load_recent_jsonl_records,
    parse_jsonl_records,
};
use crate::require_nonempty;
use crate::unix_time_millis;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherRuntimeSnapshot {
    pub id: String,
    pub enabled: bool,
    pub mode: WatcherMode,
    pub summary: String,
    #[serde(default)]
    pub last_event: Option<String>,
    #[serde(default)]
    pub last_event_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRuntimeState {
    pub running: bool,
    pub automations_enabled: bool,
    pub daemon_pid: u32,
    pub started_at_unix_ms: u128,
    pub last_tick_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_webhook_endpoint: Option<String>,
    pub active_watchers: Vec<WatcherRuntimeSnapshot>,
}

impl AutomationRuntimeState {
    pub fn load(paths: &AppPaths) -> Result<Option<Self>> {
        let path = paths.automation_state_path();
        if !path.is_file() {
            return Ok(None);
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let state = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(state))
    }

    pub fn write(&self, paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        let path = paths.automation_state_path();
        fs::write(
            &path,
            serde_json::to_vec_pretty(self)
                .context("failed to serialize automation runtime state")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn watcher_mut(&mut self, watcher_id: &str) -> Option<&mut WatcherRuntimeSnapshot> {
        self.active_watchers
            .iter_mut()
            .find(|watcher| watcher.id == watcher_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEventRecord {
    pub at_unix_ms: u128,
    pub watcher_id: String,
    pub level: String,
    pub action: String,
    pub message: String,
    #[serde(default)]
    pub service_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutomationTriggerEvent {
    pub at_unix_ms: u128,
    pub kind: String,
    pub source: String,
    #[serde(default)]
    pub watcher_hint: Option<String>,
    #[serde(default)]
    pub service_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationProposalRecord {
    pub at_unix_ms: u128,
    #[serde(default)]
    pub proposal_id: String,
    pub watcher_id: String,
    pub action: String,
    pub title: String,
    pub message: String,
    pub status: String,
    #[serde(default)]
    pub service_id: Option<String>,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub arguments: serde_json::Value,
    #[serde(default)]
    pub reviewed_at_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEventRecord {
    pub at_unix_ms: u128,
    pub source: String,
    pub category: String,
    pub actor: String,
    pub level: String,
    pub action: String,
    pub message: String,
    #[serde(default)]
    pub watcher_id: Option<String>,
    #[serde(default)]
    pub service_id: Option<String>,
}

/// Size at which the automation events store and the audit store are rotated:
/// the next append renames the live file to `<store>.1` and starts a new one.
///
/// With the `gpu-metrics` watcher enabled, the busiest writer, `rocmd` adds a
/// record a minute to each store — about 600 KB a day to the audit log — so
/// one generation holds roughly two weeks of history; without it, months.
/// Small enough that a rotated file can still be opened in an editor or read
/// whole by a script.
pub const EVENT_LOG_ROTATE_BYTES: u64 = 8 * 1024 * 1024;

/// Rotated generations kept per store besides the live file.
///
/// `<store>.1` is the newest, `<store>.4` the oldest; the oldest is dropped by
/// the next rotation. Caps each store at about 40 MiB — about two months of
/// history at the `gpu-metrics` rate — where it previously grew without bound.
/// A store that grew past the cap before rotation existed is rotated whole, so
/// it only falls under the cap once that file is itself dropped.
pub const EVENT_LOG_ROTATED_GENERATIONS: usize = 4;

/// Reviewed proposals kept by the proposals store's compaction.
///
/// "Reviewed" is any status but `pending` (approved, rejected, ...); the most
/// recently reviewed are kept, and pending proposals always are. Reviewed
/// proposals only come from the status rewrite that compacts, so this bounds
/// them exactly.
pub const REVIEWED_PROPOSALS_RETAINED: usize = 200;

/// The policy for the automation events store and the audit store.
const EVENT_LOG_ROTATION: JsonlRotation = JsonlRotation {
    max_bytes: EVENT_LOG_ROTATE_BYTES,
    generations: EVENT_LOG_ROTATED_GENERATIONS,
};

pub fn append_automation_event(paths: &AppPaths, event: &AutomationEventRecord) -> Result<()> {
    paths.ensure()?;
    append_jsonl_record(
        &paths.automation_events_path(),
        event,
        Some(EVENT_LOG_ROTATION),
        "automation event record",
    )
}

pub fn append_automation_proposal(
    paths: &AppPaths,
    proposal: &AutomationProposalRecord,
) -> Result<()> {
    paths.ensure()?;
    let mut proposal = proposal.clone();
    if proposal.proposal_id.is_empty() {
        proposal.proposal_id = generate_proposal_id(&proposal.watcher_id);
    }
    // Never rotated: a pending proposal must stay findable by id until it is
    // reviewed. The status rewrite compacts the store instead.
    append_jsonl_record(
        &paths.automation_proposals_path(),
        &proposal,
        None,
        "automation proposal record",
    )
}

pub fn append_audit_event(paths: &AppPaths, event: &AuditEventRecord) -> Result<()> {
    paths.ensure()?;
    append_jsonl_record(
        &paths.audit_events_path(),
        event,
        Some(EVENT_LOG_ROTATION),
        "audit event record",
    )
}

pub fn load_automation_proposals(paths: &AppPaths) -> Result<Vec<AutomationProposalRecord>> {
    let path = paths.automation_proposals_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(
        parse_jsonl_records::<AutomationProposalRecord>(&bytes, &path, "proposal record")?
            .into_iter()
            .map(|(index, mut proposal)| {
                normalize_proposal_identity(&mut proposal, index);
                proposal
            })
            .collect(),
    )
}

/// The most recent `limit` proposals, newest first.
///
/// Reads only the end of the store (see [`load_recent_jsonl_records`]), unless
/// that window holds a legacy proposal written before proposals had ids: its
/// id derives from its line number in the whole file, so then the whole store
/// is read. The first status update rewrites every legacy proposal with its
/// derived id (see [`update_automation_proposal_status`]), so this fallback
/// only lasts until then.
pub fn load_recent_automation_proposals(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationProposalRecord>> {
    let mut proposals: Vec<AutomationProposalRecord> = load_recent_jsonl_records(
        &paths.automation_proposals_path(),
        0,
        limit,
        "proposal record",
    )?;
    if proposals
        .iter()
        .any(|proposal| proposal.proposal_id.is_empty())
    {
        proposals = load_automation_proposals(paths)?;
        let skip = proposals.len().saturating_sub(limit);
        proposals.drain(..skip);
    }
    proposals.reverse();
    Ok(proposals)
}

/// Find a proposal by id. Reads the whole store: a pending proposal can be
/// arbitrarily old. Compaction keeps that bounded by the pending proposals
/// plus [`REVIEWED_PROPOSALS_RETAINED`].
pub fn find_automation_proposal(
    paths: &AppPaths,
    proposal_id: &str,
) -> Result<AutomationProposalRecord> {
    load_automation_proposals(paths)?
        .into_iter()
        .find(|proposal| proposal.proposal_id == proposal_id)
        .with_context(|| format!("automation proposal `{proposal_id}` not found"))
}

/// Set one proposal's `status` by rewriting the proposals store, compacting
/// it on the way (see [`compact_reviewed_proposals`]).
///
/// The whole read-modify-write runs under the store's [`jsonl_lock_path`]
/// lock, so a proposal appended (or another status set) meanwhile is neither
/// dropped nor overwritten.
///
/// The rewrite also migrates legacy proposals, written before proposals had
/// ids: each is written back with the id it was read under, derived from its
/// line number *before* compaction removed any line above it, so its id never
/// changes — and once written, reads no longer need the whole file to derive it.
pub fn update_automation_proposal_status(
    paths: &AppPaths,
    proposal_id: &str,
    status: &str,
) -> Result<AutomationProposalRecord> {
    require_nonempty(proposal_id, "proposal_id")?;
    require_nonempty(status, "status")?;
    paths.ensure()?;
    let _lock = FileLock::acquire(jsonl_lock_path(&paths.automation_proposals_path()))?;
    let mut proposals = load_automation_proposals(paths)?;
    let Some(proposal) = proposals
        .iter_mut()
        .find(|proposal| proposal.proposal_id == proposal_id)
    else {
        bail!("automation proposal `{proposal_id}` not found");
    };
    status.clone_into(&mut proposal.status);
    if status != "pending" {
        proposal.reviewed_at_unix_ms = Some(unix_time_millis());
    }
    let updated = proposal.clone();
    compact_reviewed_proposals(&mut proposals, REVIEWED_PROPOSALS_RETAINED, proposal_id);
    write_automation_proposals(paths, &proposals)?;
    Ok(updated)
}

/// Drop all but the `keep` most recently reviewed proposals, keeping every
/// pending one and the on-disk order of the rest.
///
/// "Most recently reviewed" goes by `reviewed_at_unix_ms`, except that
/// `just_reviewed`, the proposal the calling status update reviewed, always
/// counts as the newest — even if it was queued long ago or the clock has
/// moved backwards since other reviews — so a review never removes the
/// proposal it just recorded. A reviewed proposal with no review time (older
/// records) counts as the oldest; ties go by position in the store.
fn compact_reviewed_proposals(
    proposals: &mut Vec<AutomationProposalRecord>,
    keep: usize,
    just_reviewed: &str,
) {
    let mut reviewed: Vec<(bool, u128, usize)> = proposals
        .iter()
        .enumerate()
        .filter(|(_, proposal)| proposal.status != "pending")
        .map(|(index, proposal)| {
            (
                proposal.proposal_id == just_reviewed,
                proposal.reviewed_at_unix_ms.unwrap_or(0),
                index,
            )
        })
        .collect();
    let Some(excess) = reviewed
        .len()
        .checked_sub(keep)
        .filter(|&excess| excess > 0)
    else {
        return;
    };
    reviewed.sort_unstable();
    let dropped: std::collections::HashSet<usize> = reviewed[..excess]
        .iter()
        .map(|&(_, _, index)| index)
        .collect();
    let mut index = 0;
    proposals.retain(|_| {
        let kept = !dropped.contains(&index);
        index += 1;
        kept
    });
}

pub fn load_recent_audit_events(paths: &AppPaths, limit: usize) -> Result<Vec<AuditEventRecord>> {
    load_recent_jsonl_records(
        &paths.audit_events_path(),
        EVENT_LOG_ROTATED_GENERATIONS,
        limit,
        "audit event",
    )
}

/// Replace the proposals store with `proposals`. The caller must hold the
/// store's [`jsonl_lock_path`] lock.
fn write_automation_proposals(
    paths: &AppPaths,
    proposals: &[AutomationProposalRecord],
) -> Result<()> {
    let path = paths.automation_proposals_path();
    let mut text = String::new();
    for proposal in proposals {
        text.push_str(
            &serde_json::to_string(proposal)
                .context("failed to serialize automation proposal record")?,
        );
        text.push('\n');
    }
    crate::write_atomic(&path, text.as_bytes(), "jsonl")
}

fn normalize_proposal_identity(proposal: &mut AutomationProposalRecord, index: usize) {
    if proposal.proposal_id.is_empty() {
        proposal.proposal_id = format!("legacy-{}-{index}", proposal.at_unix_ms);
    }
}

pub fn generate_proposal_id(prefix: &str) -> String {
    let normalized_prefix = prefix
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    let prefix = if normalized_prefix.is_empty() {
        "proposal"
    } else {
        normalized_prefix.as_str()
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{prefix}-{nanos}")
}

pub fn load_recent_automation_events(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationEventRecord>> {
    load_recent_jsonl_records(
        &paths.automation_events_path(),
        EVENT_LOG_ROTATED_GENERATIONS,
        limit,
        "event",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonl_store::jsonl_generation_path;
    use crate::test_support::{
        audit_line, audit_messages, audit_record, seed_audit_store, temp_app_paths,
    };
    use std::path::Path;

    #[test]
    fn append_audit_event_writes_jsonl_record() -> Result<()> {
        let (root, paths) = temp_app_paths("append-audit");
        let event = AuditEventRecord {
            at_unix_ms: 123,
            source: "rocmd".to_owned(),
            category: "automation".to_owned(),
            actor: "watcher:server-recover".to_owned(),
            level: "info".to_owned(),
            action: "restart_managed_service".to_owned(),
            message: "restarted failed managed service svc-1".to_owned(),
            watcher_id: Some("server-recover".to_owned()),
            service_id: Some("svc-1".to_owned()),
        };

        append_audit_event(&paths, &event)?;

        let text = fs::read_to_string(paths.audit_events_path())?;
        let parsed = serde_json::from_str::<AuditEventRecord>(text.trim())?;
        fs::remove_dir_all(root).ok();
        assert_eq!(parsed.category, "automation");
        assert_eq!(parsed.watcher_id.as_deref(), Some("server-recover"));
        assert_eq!(parsed.service_id.as_deref(), Some("svc-1"));
        Ok(())
    }

    #[test]
    fn append_and_load_recent_automation_proposals() -> Result<()> {
        let (root, paths) = temp_app_paths("append-proposal");
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 1,
                proposal_id: "proposal-1".to_owned(),
                watcher_id: "therock-update".to_owned(),
                action: "queue_update_proposal".to_owned(),
                title: "Check TheRock updates".to_owned(),
                message: "run rocm update".to_owned(),
                status: "pending".to_owned(),
                service_id: None,
                tool: Some("check_updates".to_owned()),
                arguments: serde_json::json!({}),
                reviewed_at_unix_ms: None,
            },
        )?;
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 2,
                proposal_id: "proposal-2".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "Restart service".to_owned(),
                message: "restart svc-1".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-1".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-1" }),
                reviewed_at_unix_ms: None,
            },
        )?;

        let proposals = load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].proposal_id, "proposal-2");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-1"));
        Ok(())
    }

    #[test]
    fn proposal_status_update_rewrites_record() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-status");
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 1,
                proposal_id: "proposal-1".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "Restart service".to_owned(),
                message: "restart svc-1".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-1".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-1" }),
                reviewed_at_unix_ms: None,
            },
        )?;

        let updated = update_automation_proposal_status(&paths, "proposal-1", "rejected")?;
        let loaded = find_automation_proposal(&paths, "proposal-1")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(updated.status, "rejected");
        assert_eq!(loaded.status, "rejected");
        assert!(loaded.reviewed_at_unix_ms.is_some());
        Ok(())
    }

    #[test]
    fn load_recent_audit_events_returns_tail() -> Result<()> {
        let (root, paths) = temp_app_paths("audit-tail");
        append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 1,
                source: "rocm".to_owned(),
                category: "proposal".to_owned(),
                actor: "tui".to_owned(),
                level: "info".to_owned(),
                action: "proposal_approved".to_owned(),
                message: "approved proposal-1".to_owned(),
                watcher_id: None,
                service_id: None,
            },
        )?;
        append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 2,
                source: "rocm".to_owned(),
                category: "proposal".to_owned(),
                actor: "tui".to_owned(),
                level: "info".to_owned(),
                action: "proposal_rejected".to_owned(),
                message: "rejected proposal-2".to_owned(),
                watcher_id: None,
                service_id: None,
            },
        )?;

        let events = load_recent_audit_events(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, "proposal_rejected");
        Ok(())
    }

    fn event_record(tag: &str) -> AutomationEventRecord {
        AutomationEventRecord {
            at_unix_ms: 1,
            watcher_id: "gpu-metrics".to_owned(),
            level: "info".to_owned(),
            action: "record_gpu_metrics".to_owned(),
            message: tag.to_owned(),
            service_id: None,
        }
    }

    fn proposal_record(tag: &str) -> AutomationProposalRecord {
        AutomationProposalRecord {
            at_unix_ms: 1,
            proposal_id: tag.to_owned(),
            watcher_id: "cache-warm".to_owned(),
            action: "queue_prefetch_proposal".to_owned(),
            title: "Prefetch".to_owned(),
            message: tag.to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: None,
            arguments: serde_json::json!({}),
            reviewed_at_unix_ms: None,
        }
    }

    /// Append one record tagged `tag` to each of the three JSONL stores.
    fn append_to_every_store(paths: &AppPaths, tag: &str) -> Result<()> {
        append_audit_event(paths, &audit_record(tag))?;
        append_automation_event(paths, &event_record(tag))?;
        append_automation_proposal(paths, &proposal_record(tag))
    }

    /// The tags in each store, in on-disk order, read straight from the files
    /// so a lost record cannot hide behind any loader's filtering.
    fn tags_in_every_store(paths: &AppPaths) -> Result<[Vec<String>; 3]> {
        let audit = fs::read_to_string(paths.audit_events_path())?
            .lines()
            .map(|line| serde_json::from_str::<AuditEventRecord>(line).map(|r| r.message))
            .collect::<Result<Vec<_>, _>>()?;
        let events = fs::read_to_string(paths.automation_events_path())?
            .lines()
            .map(|line| serde_json::from_str::<AutomationEventRecord>(line).map(|r| r.message))
            .collect::<Result<Vec<_>, _>>()?;
        let proposals = fs::read_to_string(paths.automation_proposals_path())?
            .lines()
            .map(|line| {
                serde_json::from_str::<AutomationProposalRecord>(line).map(|r| r.proposal_id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok([audit, events, proposals])
    }

    fn assert_every_store_holds_exactly(stores: [Vec<String>; 3], expected: &[String]) {
        let mut expected = expected.to_vec();
        expected.sort();
        for (store, mut tags) in ["audit", "events", "proposals"].into_iter().zip(stores) {
            tags.sort();
            assert_eq!(
                tags.len(),
                expected.len(),
                "{store} store lost or duplicated records under concurrent writers"
            );
            assert_eq!(
                tags, expected,
                "{store} store holds the wrong set of records"
            );
        }
    }

    /// Run `writers` threads, each appending `appends_per_writer` tagged
    /// records (`t<writer>-<n>`) to all three stores concurrently (a
    /// `Barrier` lines them up so the appends actually overlap rather than
    /// running one at a time), then return what each store actually holds
    /// afterward. Shared by the two tests immediately below, which differ
    /// only in what (if anything) was in the stores before this runs and in
    /// what tags they expect alongside `t<writer>-<n>`.
    fn run_concurrent_thread_appends(
        paths: &AppPaths,
        writers: usize,
        appends_per_writer: usize,
    ) -> Result<[Vec<String>; 3]> {
        let barrier = std::sync::Barrier::new(writers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..writers)
                .map(|writer| {
                    let (paths, barrier) = (paths, &barrier);
                    scope.spawn(move || -> Result<()> {
                        barrier.wait();
                        for n in 0..appends_per_writer {
                            append_to_every_store(paths, &format!("t{writer}-{n}"))?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("writer thread panicked")?;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        tags_in_every_store(paths)
    }

    // Regression test for concurrent writers silently dropping each other's
    // records: every append used to read the whole store, add one line in
    // memory, and rename a rewritten copy over the original, so two writers
    // that overlapped kept only the last rename (and two threads of one
    // process even shared a temp-file name).
    #[test]
    fn concurrent_appends_from_many_threads_lose_no_records() -> Result<()> {
        const WRITERS: usize = 8;
        const APPENDS_PER_WRITER: usize = 25;
        let (root, paths) = temp_app_paths("jsonl-concurrent-threads");
        paths.ensure()?;
        let stores = run_concurrent_thread_appends(&paths, WRITERS, APPENDS_PER_WRITER);

        let expected: Vec<String> = (0..WRITERS)
            .flat_map(|writer| (0..APPENDS_PER_WRITER).map(move |n| format!("t{writer}-{n}")))
            .collect();
        fs::remove_dir_all(root).ok();
        assert_every_store_holds_exactly(stores?, &expected);
        Ok(())
    }

    /// Seed each of the three stores with one clean record followed by the
    /// same shape of torn tail a killed writer leaves: a cut-off JSON
    /// fragment with no trailing newline.
    fn seed_torn_tail_in_every_store(paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        let torn_audit = &audit_line("torn")[..20];
        fs::write(
            paths.audit_events_path(),
            format!("{}\n{torn_audit}", audit_line("seed")),
        )?;
        let torn_event = &serde_json::to_string(&event_record("torn"))?[..20];
        fs::write(
            paths.automation_events_path(),
            format!(
                "{}\n{torn_event}",
                serde_json::to_string(&event_record("seed"))?
            ),
        )?;
        let torn_proposal = &serde_json::to_string(&proposal_record("torn"))?[..20];
        fs::write(
            paths.automation_proposals_path(),
            format!(
                "{}\n{torn_proposal}",
                serde_json::to_string(&proposal_record("seed"))?
            ),
        )?;
        Ok(())
    }

    // Regression test for the exact hazard jsonl_lock_path's own doc comment
    // gives as its reason to exist: a torn-tail repair (the atomic-rename
    // replace) racing a concurrent appender that already opened the
    // pre-repair file. Neither existing concurrency test starts from a torn
    // tail, so the Torn arm's rewrite path was never exercised concurrently
    // until now.
    #[test]
    fn concurrent_appends_after_a_torn_tail_lose_no_records() -> Result<()> {
        const WRITERS: usize = 8;
        const APPENDS_PER_WRITER: usize = 25;
        let (root, paths) = temp_app_paths("jsonl-torn-tail-concurrent");
        seed_torn_tail_in_every_store(&paths)?;
        let stores = run_concurrent_thread_appends(&paths, WRITERS, APPENDS_PER_WRITER);

        let mut expected: Vec<String> = vec!["seed".to_owned()];
        expected.extend(
            (0..WRITERS)
                .flat_map(|writer| (0..APPENDS_PER_WRITER).map(move |n| format!("t{writer}-{n}"))),
        );
        fs::remove_dir_all(root).ok();
        assert_every_store_holds_exactly(stores?, &expected);
        Ok(())
    }

    /// Env var that turns a re-exec of this test binary into one writer of
    /// [`concurrent_appends_from_many_processes_lose_no_records`]: its value is
    /// `<data_dir>|<writer index>`.
    const CHILD_WRITER_ENV: &str = "ROCM_CORE_TEST_JSONL_CHILD_WRITER";
    const PROCESS_APPENDS_PER_WRITER: usize = 20;

    // Same regression as the thread test, across processes — the shape the
    // stores actually see (the `rocm` CLI and `rocmd` both write the audit
    // store, and both touch the proposals store).
    #[test]
    fn concurrent_appends_from_many_processes_lose_no_records() -> Result<()> {
        const WRITERS: usize = 6;
        if let Some(spec) = std::env::var_os(CHILD_WRITER_ENV) {
            let spec = spec.to_string_lossy().into_owned();
            let (data_dir, writer) = spec.split_once('|').context("malformed writer spec")?;
            let paths = AppPaths {
                config_dir: Path::new(data_dir).join("config"),
                data_dir: Path::new(data_dir).to_path_buf(),
                cache_dir: Path::new(data_dir).join("cache"),
            };
            for n in 0..PROCESS_APPENDS_PER_WRITER {
                append_to_every_store(&paths, &format!("p{writer}-{n}"))?;
            }
            return Ok(());
        }

        let (root, paths) = temp_app_paths("jsonl-concurrent-processes");
        paths.ensure()?;
        let exe = std::env::current_exe().context("current test executable")?;
        let children = (0..WRITERS)
            .map(|writer| {
                std::process::Command::new(&exe)
                    .args([
                        "automation::tests::concurrent_appends_from_many_processes_lose_no_records",
                        "--exact",
                        "--quiet",
                    ])
                    .env(
                        CHILD_WRITER_ENV,
                        format!("{}|{writer}", paths.data_dir.display()),
                    )
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .context("spawn writer process")
            })
            .collect::<Result<Vec<_>>>()?;
        for mut child in children {
            let status = child.wait()?;
            assert!(status.success(), "writer process failed: {status:?}");
        }

        let expected: Vec<String> = (0..WRITERS)
            .flat_map(|writer| {
                (0..PROCESS_APPENDS_PER_WRITER).map(move |n| format!("p{writer}-{n}"))
            })
            .collect();
        let stores = tags_in_every_store(&paths);
        fs::remove_dir_all(root).ok();
        assert_every_store_holds_exactly(stores?, &expected);
        Ok(())
    }

    /// Run `body` with `RLIMIT_FSIZE`'s soft limit temporarily lowered to
    /// `limit_bytes`, restoring the original soft limit — leaving the hard
    /// limit untouched throughout — before returning, so a coverage build's
    /// own `.profraw` write at process exit (after this returns, outside
    /// `body`'s control) isn't cut short by the same rlimit. `SIGXFSZ` is
    /// ignored for the duration, so a write that would exceed the limit
    /// returns `Err(EFBIG)` instead of killing the process.
    #[cfg(unix)]
    #[allow(unsafe_code)] // libc FFI: lower/restore RLIMIT_FSIZE and ignore SIGXFSZ for the duration
    fn with_lowered_fsize_limit(limit_bytes: u64, body: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut original = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe {
            if libc::getrlimit(libc::RLIMIT_FSIZE, &raw mut original) != 0 {
                bail!("getrlimit failed: {}", std::io::Error::last_os_error());
            }
        }
        let lowered = libc::rlimit {
            rlim_cur: limit_bytes as libc::rlim_t,
            rlim_max: original.rlim_max,
        };
        // The signal handler and the rlimit are restored unconditionally
        // below, regardless of whether lowering the limit here succeeded or
        // `body` even ran — this process stays around for the rest of the
        // (child) test process's life, so a skipped restore on an early
        // `bail!` would leave it running with SIGXFSZ permanently ignored.
        let previous_handler = unsafe { libc::signal(libc::SIGXFSZ, libc::SIG_IGN) };
        let lowered_ok = unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &raw const lowered) == 0 };
        let result = if lowered_ok {
            body()
        } else {
            Err(anyhow::anyhow!(
                "setrlimit failed: {}",
                std::io::Error::last_os_error()
            ))
        };
        let restored_ok = unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &raw const original) == 0 };
        unsafe {
            libc::signal(libc::SIGXFSZ, previous_handler);
        }
        if !restored_ok {
            bail!(
                "restoring rlimit failed: {}",
                std::io::Error::last_os_error()
            );
        }
        result
    }

    // Regression test for append_or_tolerate_a_missing_newline's glue around
    // a real failed write — not just the pure landed_despite_the_error
    // predicate it calls, tested in isolation (in `jsonl_store.rs`) with
    // hand-built byte slices. An RLIMIT_FSIZE one byte under the record's full length
    // reproduces, deterministically, the exact case that glue exists for.
    // `write_all` loops over `write(2)`: its first call lands every byte up
    // to the limit as a successful short write; its next call, with the one
    // remaining byte (the trailing `\n`) now past the limit, is the one that
    // raises SIGXFSZ (ignored) and returns `Err(EFBIG)`. The append must
    // still report `Ok`, and the store must hold exactly the record that
    // landed — not zero (treated as a real failure), not duplicated, and
    // genuinely missing its `\n` rather than having silently written in
    // full (the assertion below on the file's own last byte is what tells
    // those two apart). A first, ordinary append before the limit is
    // installed puts this at a non-zero `start` (`TailState::Clean` with an
    // existing record ahead of it), not only the empty-file case.
    #[cfg(unix)]
    #[test]
    fn append_tolerates_a_short_write_from_an_fsize_limit() -> Result<()> {
        const SHORT_WRITE_CHILD_ENV: &str = "ROCM_CORE_TEST_JSONL_SHORT_WRITE_CHILD";

        if let Some(data_dir) = std::env::var_os(SHORT_WRITE_CHILD_ENV) {
            let data_dir = data_dir.to_string_lossy().into_owned();
            let paths = AppPaths {
                config_dir: Path::new(&data_dir).join("config"),
                data_dir: Path::new(&data_dir).to_path_buf(),
                cache_dir: Path::new(&data_dir).join("cache"),
            };
            append_audit_event(&paths, &audit_record("first"))?;
            let record = audit_record("short-write");
            let mut line = serde_json::to_vec(&record).context("serialize audit record")?;
            line.push(b'\n');
            let existing_len = fs::metadata(paths.audit_events_path())
                .context("stat seeded store")?
                .len();
            with_lowered_fsize_limit(existing_len + line.len() as u64 - 1, || {
                append_audit_event(&paths, &record)
            })?;
            return Ok(());
        }

        let (root, paths) = temp_app_paths("jsonl-short-write");
        paths.ensure()?;
        let exe = std::env::current_exe().context("current test executable")?;
        let output = std::process::Command::new(&exe)
            .args([
                "automation::tests::append_tolerates_a_short_write_from_an_fsize_limit",
                "--exact",
                "--quiet",
                "--nocapture",
            ])
            .env(SHORT_WRITE_CHILD_ENV, paths.data_dir.display().to_string())
            .output()
            .context("spawn short-write child")?;
        let text = fs::read_to_string(paths.audit_events_path());
        fs::remove_dir_all(root).ok();
        assert!(
            output.status.success(),
            "child process failed: {:?}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let text = text?;
        assert!(
            !text.ends_with('\n'),
            "the short write must have genuinely landed one byte short of its own \
             trailing newline — a file ending in `\\n` here means the rlimit never \
             actually cut the write, and this test proved nothing: {text:?}"
        );
        let records: Vec<AuditEventRecord> = text
            .lines()
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|error| panic!("not a complete record: {error}: {line:?}"))
            })
            .collect();
        assert_eq!(
            audit_messages(&records),
            ["first", "short-write"],
            "the write that failed but landed must not be duplicated or dropped: {text:?}"
        );
        Ok(())
    }

    // Sibling of the test above, for the other side of the same glue: a
    // write that fails and lands *nothing* (the limit leaves no room even
    // for the first byte) must surface as a genuine `Err`, not be folded
    // into a false `Ok` the way a landed-but-incomplete write is.
    // `landed_despite_the_error` itself already has a direct unit test (in
    // `jsonl_store.rs`) for "nothing landing is not a complete record" — what has no test without
    // this one is the `match` in `append_or_tolerate_a_missing_newline` that
    // acts on that verdict: a mutation collapsing its `Ok(landed) if
    // landed_despite_the_error(..) => Ok(())` / `Ok(_) => Err(..)` arms into
    // an unconditional `Ok(())` would pass every other test in this file.
    #[cfg(unix)]
    #[test]
    fn append_reports_a_real_failure_when_nothing_landed() -> Result<()> {
        const NOTHING_LANDS_CHILD_ENV: &str = "ROCM_CORE_TEST_JSONL_NOTHING_LANDS_CHILD";

        if let Some(data_dir) = std::env::var_os(NOTHING_LANDS_CHILD_ENV) {
            let data_dir = data_dir.to_string_lossy().into_owned();
            let paths = AppPaths {
                config_dir: Path::new(&data_dir).join("config"),
                data_dir: Path::new(&data_dir).to_path_buf(),
                cache_dir: Path::new(&data_dir).join("cache"),
            };
            append_audit_event(&paths, &audit_record("first"))?;
            let existing_len = fs::metadata(paths.audit_events_path())
                .context("stat seeded store")?
                .len();
            // Zero slack: the write offset already sits at the limit, so
            // `write(2)`'s very first call returns `Err(EFBIG)` having
            // written nothing at all — not even a partial record.
            let result = with_lowered_fsize_limit(existing_len, || {
                append_audit_event(&paths, &audit_record("lost"))
            });
            // Report the child's own verdict on stdout for the parent to
            // read, rather than via exit status: an `Err` here is the
            // *correct* outcome for this test, and letting it propagate as
            // `?` would make the child process itself look like a failure
            // for the wrong reason. The message itself (not just ok/err) is
            // what tells a genuine write failure apart from some unrelated
            // error (e.g. a lock or `paths.ensure()` failure) that would
            // also make `is_err()` true without this test having proven
            // anything.
            match result {
                Ok(()) => println!("ok"),
                Err(error) => println!("err: {error:#}"),
            }
            return Ok(());
        }

        let (root, paths) = temp_app_paths("jsonl-nothing-lands");
        paths.ensure()?;
        let exe = std::env::current_exe().context("current test executable")?;
        let output = std::process::Command::new(&exe)
            .args([
                "automation::tests::append_reports_a_real_failure_when_nothing_landed",
                "--exact",
                "--quiet",
                "--nocapture",
            ])
            .env(
                NOTHING_LANDS_CHILD_ENV,
                paths.data_dir.display().to_string(),
            )
            .output()
            .context("spawn nothing-lands child")?;
        let text = fs::read_to_string(paths.audit_events_path());
        fs::remove_dir_all(root).ok();
        assert!(
            output.status.success(),
            "child process failed: {:?}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("err: failed to append"),
            "a write that lands nothing must be reported as a real failure from the append \
             itself (not folded into a false success, and not some unrelated error): \
             child stdout was {stdout:?}"
        );
        assert_eq!(
            audit_messages(&[serde_json::from_str(text?.trim()).context("parse seeded record")?]),
            ["first"],
            "the store must be left exactly as the seed append wrote it — no partial or \
             lost record from the write that failed with nothing landing"
        );
        Ok(())
    }

    // A proposal-status update rewrites the whole proposals store, so it must
    // neither drop proposals appended while it runs nor be undone by them.
    #[test]
    fn proposal_status_updates_racing_appends_lose_neither() -> Result<()> {
        const APPENDS: usize = 100;
        const UPDATES: usize = 50;
        let (root, paths) = temp_app_paths("proposal-status-race");
        append_automation_proposal(&paths, &proposal_record("reviewed"))?;
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let appender = scope.spawn(|| -> Result<()> {
                barrier.wait();
                for n in 0..APPENDS {
                    append_automation_proposal(&paths, &proposal_record(&format!("new-{n}")))?;
                }
                Ok(())
            });
            let reviewer = scope.spawn(|| -> Result<()> {
                barrier.wait();
                for n in 0..UPDATES {
                    let status = if n % 2 == 0 { "approved" } else { "rejected" };
                    update_automation_proposal_status(&paths, "reviewed", status)?;
                }
                Ok(())
            });
            appender.join().expect("appender panicked")?;
            reviewer.join().expect("reviewer panicked")?;
            Ok::<_, anyhow::Error>(())
        })?;

        let proposals = load_automation_proposals(&paths)?;
        fs::remove_dir_all(root).ok();
        let mut ids: Vec<&str> = proposals.iter().map(|p| p.proposal_id.as_str()).collect();
        ids.sort_unstable();
        let mut expected: Vec<String> = (0..APPENDS).map(|n| format!("new-{n}")).collect();
        expected.push("reviewed".to_owned());
        expected.sort();
        assert_eq!(
            ids, expected,
            "a status rewrite dropped an appended proposal"
        );
        let reviewed = proposals
            .iter()
            .find(|p| p.proposal_id == "reviewed")
            .expect("reviewed proposal present");
        // UPDATES is even, so the last update (n = UPDATES - 1) is odd: rejected.
        assert_eq!(
            reviewed.status, "rejected",
            "the last status update was lost"
        );
        Ok(())
    }

    // Smoke test, not a race-pinning one: these stores stay well under one
    // scan chunk, so on Linux a small append is essentially never actually
    // observed half-written here, and this can mostly only fail via a
    // propagated read/write error or the final count assertion — not by
    // actually exercising the torn-line skip under real contention. That
    // skip is pinned deterministically instead by
    // `readers_skip_an_unterminated_last_line_that_does_not_parse`. What
    // this test is still worth keeping for: a reader that started failing
    // outright under concurrent load (rather than just reading stale data,
    // which this can't detect either) would still show up here.
    #[test]
    fn concurrent_reads_and_appends_never_error() -> Result<()> {
        const APPENDS: usize = 200;
        let (root, paths) = temp_app_paths("jsonl-read-race");
        paths.ensure()?;
        let done = std::sync::atomic::AtomicBool::new(false);
        let outcome = std::thread::scope(|scope| {
            let reader = scope.spawn(|| -> Result<usize> {
                let mut reads = 0;
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    load_recent_audit_events(&paths, 5)?;
                    load_recent_automation_events(&paths, 5)?;
                    load_automation_proposals(&paths)?;
                    reads += 1;
                }
                Ok(reads)
            });
            let written =
                (0..APPENDS).try_for_each(|n| append_to_every_store(&paths, &format!("r{n}")));
            done.store(true, std::sync::atomic::Ordering::Release);
            let reads = reader.join().expect("reader panicked");
            written.and(reads)
        });
        let loaded = load_recent_audit_events(&paths, APPENDS);
        fs::remove_dir_all(root).ok();
        outcome?;
        assert_eq!(loaded?.len(), APPENDS);
        Ok(())
    }

    #[test]
    fn load_recent_stops_before_corrupt_lines_older_than_the_window() -> Result<()> {
        let (root, paths) = temp_app_paths("recent-window");
        seed_audit_store(
            &paths,
            &format!(
                "not json\n{}\n\n{}\n{}\n",
                audit_line("a"),
                audit_line("b"),
                audit_line("c")
            ),
        )?;
        let recent = load_recent_audit_events(&paths, 3);
        let too_far = load_recent_audit_events(&paths, 4);
        fs::remove_dir_all(root).ok();
        assert_eq!(audit_messages(&recent?), ["a", "b", "c"]);
        // Reaching the corrupt line is still an error, as before.
        assert!(
            too_far.is_err(),
            "a corrupt line inside the window must fail"
        );
        Ok(())
    }

    #[test]
    fn readers_skip_an_unterminated_last_line_that_does_not_parse() -> Result<()> {
        let (root, paths) = temp_app_paths("unterminated-torn");
        let torn = &audit_line("torn")[..20];
        seed_audit_store(&paths, &format!("{}\n{torn}", audit_line("a")))?;
        let torn_proposal = &serde_json::to_string(&proposal_record("torn"))?[..20];
        fs::write(
            paths.automation_proposals_path(),
            format!(
                "{}\n{torn_proposal}",
                serde_json::to_string(&proposal_record("a"))?
            ),
        )?;
        let audit = load_recent_audit_events(&paths, 5);
        let proposals = load_automation_proposals(&paths);
        fs::remove_dir_all(root).ok();
        assert_eq!(audit_messages(&audit?), ["a"]);
        let ids: Vec<String> = proposals?.into_iter().map(|p| p.proposal_id).collect();
        assert_eq!(ids, ["a"]);
        Ok(())
    }

    // `serde_json` writes non-ASCII text as raw UTF-8, so a record cut
    // mid-write can end inside a character; that must not fail a whole-file
    // UTF-8 decode and block every proposal read, approve and reject.
    #[test]
    fn proposal_reads_skip_a_torn_line_cut_inside_a_utf8_character() -> Result<()> {
        let (root, paths) = temp_app_paths("unterminated-utf8");
        paths.ensure()?;
        let mut torn = proposal_record("torn");
        torn.message = "ü".repeat(8);
        let torn = serde_json::to_vec(&torn)?;
        let cut = torn
            .iter()
            .position(|&byte| byte == 0xC3)
            .expect("message is raw UTF-8")
            + 1;
        let mut contents = serde_json::to_vec(&proposal_record("a"))?;
        contents.push(b'\n');
        contents.extend_from_slice(&torn[..cut]);
        fs::write(paths.automation_proposals_path(), contents)?;
        let proposals = load_automation_proposals(&paths);
        let updated = update_automation_proposal_status(&paths, "a", "approved");
        fs::remove_dir_all(root).ok();
        let ids: Vec<String> = proposals?.into_iter().map(|p| p.proposal_id).collect();
        assert_eq!(ids, ["a"]);
        assert_eq!(updated?.status, "approved");
        Ok(())
    }

    // Complete JSON that is not a valid record is real data, not an append
    // caught mid-write: reads still report it, and a status rewrite must fail
    // rather than silently drop it from disk.
    #[test]
    fn an_unterminated_last_line_of_the_wrong_shape_is_reported_not_dropped() -> Result<()> {
        let (root, paths) = temp_app_paths("unterminated-wrong-shape");
        paths.ensure()?;
        let contents = format!(
            "{}\n{{\"unexpected\":true}}",
            serde_json::to_string(&proposal_record("a"))?
        );
        fs::write(paths.automation_proposals_path(), &contents)?;
        seed_audit_store(
            &paths,
            &format!("{}\n{{\"unexpected\":true}}", audit_line("a")),
        )?;
        let proposals = load_automation_proposals(&paths);
        let updated = update_automation_proposal_status(&paths, "a", "approved");
        let on_disk = fs::read_to_string(paths.automation_proposals_path());
        let audit = load_recent_audit_events(&paths, 5);
        fs::remove_dir_all(root).ok();
        assert!(proposals.is_err(), "a wrong-shape record must be reported");
        assert!(
            updated.is_err(),
            "the rewrite must refuse, not drop the line"
        );
        assert_eq!(on_disk?, contents);
        assert!(audit.is_err(), "a wrong-shape record must be reported");
        Ok(())
    }

    // Mirrors the test above, but for the position half of
    // parse_jsonl_records's guard rather than its shape half: a line that
    // is not valid JSON (so `is_torn_record` is true for it, same as a
    // genuine torn tail) but is NOT the store's last line must still be
    // reported as an error, not silently skipped. Without the
    // `index + 1 == line_count` half of that guard, this line would be
    // indistinguishable from torn debris and silently dropped by
    // update_automation_proposal_status's whole-file rewrite.
    #[test]
    fn a_corrupt_line_before_the_last_is_reported_not_dropped() -> Result<()> {
        let (root, paths) = temp_app_paths("corrupt-line-not-last");
        paths.ensure()?;
        let corrupt = &serde_json::to_string(&proposal_record("corrupt"))?[..10];
        // No trailing `\n` after "b": the buffer as a whole must be
        // unterminated too, so the corrupt line is excluded from the
        // tolerance window by the position check specifically
        // (`index + 1 == line_count`) and not merely because the whole
        // buffer happens to already be terminated.
        let contents = format!(
            "{}\n{corrupt}\n{}",
            serde_json::to_string(&proposal_record("a"))?,
            serde_json::to_string(&proposal_record("b"))?
        );
        fs::write(paths.automation_proposals_path(), &contents)?;
        let proposals = load_automation_proposals(&paths);
        let updated = update_automation_proposal_status(&paths, "a", "approved");
        let on_disk = fs::read_to_string(paths.automation_proposals_path());
        fs::remove_dir_all(root).ok();
        assert!(
            proposals.is_err(),
            "a corrupt line that isn't the store's last line must be reported, not silently skipped"
        );
        assert!(
            updated.is_err(),
            "the rewrite must refuse, not drop the line"
        );
        assert_eq!(
            on_disk?, contents,
            "a refused rewrite must leave the file untouched"
        );
        Ok(())
    }

    #[test]
    fn readers_keep_an_unterminated_last_line_that_parses() -> Result<()> {
        let (root, paths) = temp_app_paths("unterminated-complete");
        seed_audit_store(&paths, &format!("{}\n{}", audit_line("a"), audit_line("b")))?;
        let audit = load_recent_audit_events(&paths, 5);
        fs::remove_dir_all(root).ok();
        assert_eq!(audit_messages(&audit?), ["a", "b"]);
        Ok(())
    }

    #[test]
    fn append_cuts_a_torn_last_line_left_by_a_killed_writer() -> Result<()> {
        let (root, paths) = temp_app_paths("append-torn-tail");
        let torn = &audit_line("torn")[..20];
        seed_audit_store(&paths, &format!("{}\n{torn}", audit_line("a")))?;
        append_audit_event(&paths, &audit_record("b"))?;
        let text = fs::read_to_string(paths.audit_events_path());
        fs::remove_dir_all(root).ok();
        assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        Ok(())
    }

    #[test]
    fn append_terminates_a_complete_unterminated_last_line() -> Result<()> {
        let (root, paths) = temp_app_paths("append-unterminated");
        seed_audit_store(&paths, &audit_line("a"))?;
        append_audit_event(&paths, &audit_record("b"))?;
        let text = fs::read_to_string(paths.audit_events_path());
        fs::remove_dir_all(root).ok();
        assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        Ok(())
    }

    // A whitespace-only unterminated tail is not valid JSON either, so
    // without `is_blank` in `inspect_tail`'s check it would be classified
    // Torn and cut by the next append's whole-store rewrite instead of being
    // kept and terminated in place, as TailState::ValidUnterminated's own
    // doc comment documents it should be.
    #[test]
    fn append_terminates_a_blank_unterminated_last_line_without_cutting_it() -> Result<()> {
        let (root, paths) = temp_app_paths("append-blank-unterminated");
        seed_audit_store(&paths, &format!("{}\n ", audit_line("a")))?;
        append_audit_event(&paths, &audit_record("b"))?;
        let text = fs::read_to_string(paths.audit_events_path());
        fs::remove_dir_all(root).ok();
        assert_eq!(
            text?,
            format!("{}\n \n{}\n", audit_line("a"), audit_line("b")),
            "a blank unterminated tail must be kept and terminated, not cut as torn debris"
        );
        Ok(())
    }

    #[test]
    fn append_leaves_a_torn_line_longer_than_one_scan_chunk_cut_cleanly() -> Result<()> {
        let (root, paths) = temp_app_paths("append-long-torn-tail");
        let long_torn = format!("{{\"message\":\"{}", "x".repeat(3 * 64 * 1024));
        seed_audit_store(&paths, &format!("{}\n{long_torn}", audit_line("a")))?;
        append_audit_event(&paths, &audit_record("b"))?;
        let recent = load_recent_audit_events(&paths, 5);
        fs::remove_dir_all(root).ok();
        assert_eq!(audit_messages(&recent?), ["a", "b"]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn append_of_a_torn_tail_replaces_rather_than_truncates_the_file_in_place() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        // The inode is the observable difference between the two ways to
        // repair a torn tail: truncating it in place keeps rewriting the
        // SAME file, so a lock-free reader mid-scan can see it briefly
        // shorter than before the repair and then longer again once the new
        // record lands — the shrink-and-regrow window a reverse scan can
        // misread as a record nobody wrote (see `append_jsonl_record`). A
        // rename publishes a DIFFERENT file over the name, so a reader only
        // ever sees the pre-repair length or the post-append length, never
        // one in between.
        //
        // `append_cuts_a_torn_last_line_left_by_a_killed_writer` above only
        // checks the final bytes, which holds just as well for an in-place
        // truncate-then-write, so it cannot tell the two apart.
        //
        // The `#[cfg(windows)]` function of the same name below asserts the
        // same guarantee through `GetFileInformationByHandle`.
        let (root, paths) = temp_app_paths("append-torn-tail-replaces");
        let torn = &audit_line("torn")[..20];
        seed_audit_store(&paths, &format!("{}\n{torn}", audit_line("a")))?;
        let path = paths.audit_events_path();
        let before = fs::metadata(&path)?.ino();

        append_audit_event(&paths, &audit_record("b"))?;

        let after = fs::metadata(&path)?.ino();
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        assert_ne!(
            before, after,
            "repairing a torn tail kept the same inode, so it truncated the \
             live file in place instead of replacing it"
        );
        assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn append_of_a_torn_tail_replaces_rather_than_truncates_the_file_in_place() -> Result<()> {
        use crate::test_support::windows_file_identity;

        // The Windows counterpart of the Unix test above. The file index
        // together with the volume serial number is this platform's file
        // identity, the way an inode is on Unix: an in-place truncate keeps
        // that pair, while a repair-by-rename gives the live path a
        // different one.
        let (root, paths) = temp_app_paths("append-torn-tail-replaces");
        let torn = &audit_line("torn")[..20];
        seed_audit_store(&paths, &format!("{}\n{torn}", audit_line("a")))?;
        let path = paths.audit_events_path();
        let before = windows_file_identity(&path)?;

        append_audit_event(&paths, &audit_record("b"))?;

        let after = windows_file_identity(&path)?;
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        assert_ne!(
            before, after,
            "repairing a torn tail kept the same file identity, so it \
             truncated the live file in place instead of replacing it"
        );
        assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn append_to_a_clean_or_unterminated_store_keeps_the_same_file() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        // The counterpart of the torn-tail-repair test above, for the other
        // two `TailState` arms: with no torn tail to repair — the store either
        // already ends on a line boundary, or ends in a complete record
        // missing just its `\n` — an append must stay a plain append-mode
        // write on the SAME file, not the atomic replace the torn-tail case
        // takes. An inode change here would mean every append costs a
        // full-file rewrite, which is exactly the per-append cost this
        // store's design exists to avoid.
        for seed in [
            format!("{}\n", audit_line("a")), // Clean: ends on a line boundary.
            audit_line("a"),                  // ValidUnterminated: no trailing `\n`.
        ] {
            let (root, paths) = temp_app_paths("append-clean-same-file");
            seed_audit_store(&paths, &seed)?;
            let path = paths.audit_events_path();
            let before = fs::metadata(&path)?.ino();

            append_audit_event(&paths, &audit_record("b"))?;

            let after = fs::metadata(&path)?.ino();
            let text = fs::read_to_string(&path);
            fs::remove_dir_all(root).ok();
            assert_eq!(
                before, after,
                "an append with no torn tail to repair replaced the file \
                 instead of appending to it in place, seed {seed:?}"
            );
            assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        }
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn append_to_a_clean_or_unterminated_store_keeps_the_same_file() -> Result<()> {
        use crate::test_support::windows_file_identity;

        // The Windows counterpart of the Unix test above.
        for seed in [
            format!("{}\n", audit_line("a")), // Clean: ends on a line boundary.
            audit_line("a"),                  // ValidUnterminated: no trailing `\n`.
        ] {
            let (root, paths) = temp_app_paths("append-clean-same-file");
            seed_audit_store(&paths, &seed)?;
            let path = paths.audit_events_path();
            let before = windows_file_identity(&path)?;

            append_audit_event(&paths, &audit_record("b"))?;

            let after = windows_file_identity(&path)?;
            let text = fs::read_to_string(&path);
            fs::remove_dir_all(root).ok();
            assert_eq!(
                before, after,
                "an append with no torn tail to repair replaced the file \
                 instead of appending to it in place, seed {seed:?}"
            );
            assert_eq!(text?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        }
        Ok(())
    }

    fn event_line(tag: &str) -> String {
        serde_json::to_string(&event_record(tag)).expect("serialize event record")
    }

    /// Write `contents` as the automation events store, creating its directory.
    fn seed_events_store(paths: &AppPaths, contents: &str) -> Result<()> {
        paths.ensure()?;
        fs::write(paths.automation_events_path(), contents)?;
        Ok(())
    }

    // The rotation tests in `jsonl_store.rs` all drive `append_jsonl_record` and
    // `load_recent_jsonl_records` directly with a custom `JsonlRotation`,
    // which proves the mechanism works but never exercises the wiring that
    // connects it to the real stores: `append_automation_event` and
    // `append_audit_event` passing `Some(EVENT_LOG_ROTATION)`, and
    // `load_recent_automation_events`/`load_recent_audit_events` passing
    // `EVENT_LOG_ROTATED_GENERATIONS`. Flip either side back to `None`/`0`
    // and every one of those still passes. These two tests close that gap by
    // going through the public wrappers only, at the real
    // `EVENT_LOG_ROTATE_BYTES` cap — seeding the live file directly (fast)
    // rather than looping a real append to size, since it's the one real
    // append afterward that is under test, not how the filler got there.
    #[test]
    fn events_store_rotates_through_the_public_wrapper() -> Result<()> {
        let (root, paths) = temp_app_paths("events-public-wrapper-rotation");
        let filler = event_line("filler") + "\n";
        let repeats = (EVENT_LOG_ROTATE_BYTES as usize).div_ceil(filler.len());
        seed_events_store(&paths, &filler.repeat(repeats))?;

        append_automation_event(&paths, &event_record("new"))?;
        let rotated = jsonl_generation_path(&paths.automation_events_path(), 1).exists();
        // `limit: 2` so the read must actually cross into the rotated `.1` to
        // satisfy it — `limit: 1` would be answered by the live file alone and
        // never exercise `load_recent_automation_events`'s own
        // `EVENT_LOG_ROTATED_GENERATIONS` wiring, only the append side's.
        let recent = load_recent_automation_events(&paths, 2);
        fs::remove_dir_all(root).ok();

        assert!(
            rotated,
            "the live file was at the cap and should have rotated"
        );
        assert_eq!(
            recent?
                .into_iter()
                .map(|event| event.message)
                .collect::<Vec<_>>(),
            ["filler", "new"]
        );
        Ok(())
    }

    #[test]
    fn audit_log_rotates_through_the_public_wrapper() -> Result<()> {
        let (root, paths) = temp_app_paths("audit-public-wrapper-rotation");
        let filler = audit_line("filler") + "\n";
        let repeats = (EVENT_LOG_ROTATE_BYTES as usize).div_ceil(filler.len());
        seed_audit_store(&paths, &filler.repeat(repeats))?;

        append_audit_event(&paths, &audit_record("new"))?;
        let rotated = jsonl_generation_path(&paths.audit_events_path(), 1).exists();
        // `limit: 2`, for the same reason as the events-store twin above: the
        // read must actually cross into `.1` to exercise
        // `load_recent_audit_events`'s own generations wiring.
        let recent = load_recent_audit_events(&paths, 2);
        fs::remove_dir_all(root).ok();

        assert!(
            rotated,
            "the live file was at the cap and should have rotated"
        );
        assert_eq!(
            recent?
                .into_iter()
                .map(|event| event.message)
                .collect::<Vec<_>>(),
            ["filler", "new"]
        );
        Ok(())
    }

    fn reviewed_proposal(tag: &str, reviewed_at: Option<u128>) -> AutomationProposalRecord {
        AutomationProposalRecord {
            status: "approved".to_owned(),
            reviewed_at_unix_ms: reviewed_at,
            ..proposal_record(tag)
        }
    }

    fn proposal_ids(proposals: &[AutomationProposalRecord]) -> Vec<&str> {
        proposals.iter().map(|p| p.proposal_id.as_str()).collect()
    }

    #[test]
    fn compaction_keeps_every_pending_proposal_and_the_most_recently_reviewed() {
        let mut proposals = vec![
            proposal_record("pending-old"),
            reviewed_proposal("reviewed-at-10", Some(10)),
            reviewed_proposal("reviewed-at-30", Some(30)),
            proposal_record("pending-new"),
            reviewed_proposal("reviewed-at-20", Some(20)),
            reviewed_proposal("reviewed-never-timed", None),
        ];
        compact_reviewed_proposals(&mut proposals, 2, "pending-new");
        assert_eq!(
            proposal_ids(&proposals),
            [
                "pending-old",
                "reviewed-at-30",
                "pending-new",
                "reviewed-at-20"
            ]
        );
        compact_reviewed_proposals(&mut proposals, 2, "pending-new");
        assert_eq!(
            proposals.len(),
            4,
            "already within the cap: nothing dropped"
        );
        // The proposal just reviewed is kept even when its review time is the
        // oldest, as after the clock moved backwards.
        compact_reviewed_proposals(&mut proposals, 1, "reviewed-at-20");
        assert_eq!(
            proposal_ids(&proposals),
            ["pending-old", "pending-new", "reviewed-at-20"]
        );
    }

    /// Write `proposals` as the proposals store, one JSON line each.
    fn seed_proposals_store(paths: &AppPaths, lines: &[String]) -> Result<()> {
        paths.ensure()?;
        fs::write(paths.automation_proposals_path(), lines.join("\n") + "\n")?;
        Ok(())
    }

    fn proposal_line(proposal: &AutomationProposalRecord) -> String {
        serde_json::to_string(proposal).expect("serialize proposal")
    }

    // A pending proposal must stay findable by id until it is reviewed, however
    // many proposals were reviewed after it was queued.
    #[test]
    fn a_status_update_compacts_reviewed_proposals_but_keeps_pending_ones() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-compaction");
        let reviewed = REVIEWED_PROPOSALS_RETAINED + 5;
        let mut lines = vec![proposal_line(&proposal_record("pending-old"))];
        lines.extend((0..reviewed).map(|n| {
            proposal_line(&reviewed_proposal(
                &format!("r{n}"),
                Some(1_000 + n as u128),
            ))
        }));
        lines.push(proposal_line(&proposal_record("to-review")));
        seed_proposals_store(&paths, &lines)?;

        let updated = update_automation_proposal_status(&paths, "to-review", "rejected")?;
        let proposals = load_automation_proposals(&paths);
        let pending_old = find_automation_proposal(&paths, "pending-old");
        fs::remove_dir_all(root).ok();

        let proposals = proposals?;
        assert_eq!(updated.status, "rejected");
        assert_eq!(pending_old?.status, "pending");
        let reviewed_ids: Vec<&str> = proposals
            .iter()
            .filter(|p| p.status != "pending")
            .map(|p| p.proposal_id.as_str())
            .collect();
        assert_eq!(reviewed_ids.len(), REVIEWED_PROPOSALS_RETAINED);
        // The just-reviewed proposal is the newest review, so it is kept; the
        // six oldest reviews make room for it.
        assert_eq!(reviewed_ids.last(), Some(&"to-review"));
        for n in 0..6 {
            assert!(
                !reviewed_ids.contains(&format!("r{n}").as_str()),
                "r{n} kept"
            );
        }
        assert_eq!(reviewed_ids[0], "r6");
        assert_eq!(proposals[0].proposal_id, "pending-old");
        Ok(())
    }

    // A legacy proposal's id derives from its line number, so compaction
    // removing lines above it must not change the id it is known by — and
    // the rewrite must persist that id so later reads need no line number.
    #[test]
    fn legacy_proposal_ids_survive_compaction_and_are_persisted() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-legacy-compaction");
        let reviewed = REVIEWED_PROPOSALS_RETAINED + 3;
        let mut lines: Vec<String> = (0..reviewed)
            .map(|n| proposal_line(&reviewed_proposal(&format!("r{n}"), Some(n as u128 + 1))))
            .collect();
        let legacy = AutomationProposalRecord {
            at_unix_ms: 77,
            ..proposal_record("")
        };
        lines.push(proposal_line(&legacy));
        lines.push(proposal_line(&proposal_record("to-review")));
        seed_proposals_store(&paths, &lines)?;
        let legacy_id = format!("legacy-77-{reviewed}");

        let before = find_automation_proposal(&paths, &legacy_id);
        let update = update_automation_proposal_status(&paths, "to-review", "approved");
        let after = find_automation_proposal(&paths, &legacy_id);
        let recent = load_recent_automation_proposals(&paths, 2);
        let on_disk = fs::read_to_string(paths.automation_proposals_path());
        let approve_legacy = update_automation_proposal_status(&paths, &legacy_id, "approved");
        fs::remove_dir_all(root).ok();

        assert_eq!(before?.status, "pending");
        update?;
        assert_eq!(after?.status, "pending", "the legacy id still resolves");
        assert_eq!(proposal_ids(&recent?), ["to-review", legacy_id.as_str()]);
        assert!(
            on_disk?.contains(&format!("\"proposal_id\":\"{legacy_id}\"")),
            "the rewrite persists the legacy id"
        );
        assert_eq!(approve_legacy?.status, "approved");
        Ok(())
    }

    #[test]
    fn recent_proposals_read_only_the_tail_but_fall_back_for_legacy_ids() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-tail");
        // A corrupt line older than the window fails a whole-file read, so a
        // read that succeeds proves only the tail was read.
        seed_proposals_store(
            &paths,
            &[
                "not json".to_owned(),
                proposal_line(&proposal_record("a")),
                proposal_line(&proposal_record("b")),
            ],
        )?;
        let tail = load_recent_automation_proposals(&paths, 2);
        let whole = load_automation_proposals(&paths);
        let legacy = AutomationProposalRecord {
            at_unix_ms: 5,
            ..proposal_record("")
        };
        seed_proposals_store(
            &paths,
            &[
                proposal_line(&proposal_record("a")),
                String::new(),
                proposal_line(&legacy),
                proposal_line(&proposal_record("b")),
            ],
        )?;
        let with_legacy = load_recent_automation_proposals(&paths, 2);
        let full = load_automation_proposals(&paths);
        fs::remove_dir_all(root).ok();
        assert_eq!(proposal_ids(&tail?), ["b", "a"]);
        assert!(
            whole.is_err(),
            "the whole-file read reports the corrupt line"
        );
        // Same ids as the whole-file read, line index and all.
        assert_eq!(proposal_ids(&with_legacy?), ["b", "legacy-5-2"]);
        assert_eq!(proposal_ids(&full?), ["a", "legacy-5-2", "b"]);
        Ok(())
    }
}
