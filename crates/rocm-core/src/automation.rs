// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Automation-daemon runtime state and its on-disk records.
//!
//! The built-in watcher catalog (`BuiltinWatcherSpec`/`builtin_watcher(s)`),
//! the running daemon's live snapshot (`AutomationRuntimeState`), the
//! logically append-only JSONL stores for watcher action events
//! (`AutomationEventRecord`), audit log entries, and review proposals, and
//! the accessors used to read them back.
//!
//! The `rocm` CLI and `rocmd` write these stores concurrently, so every
//! mutation of a store holds its sibling `.lock` file (see `jsonl_lock_path`).
//! Appending a record is one append-mode write of one line — its cost does not
//! grow with the store — rolled back if the write fails; a proposal status
//! update rewrites the store via `write_atomic`. Readers take no lock, and an
//! unterminated last line that is not valid JSON — an append caught mid-write —
//! is skipped rather than reported as corruption. The stores are not rotated
//! or capped. `AutomationTriggerEvent` is also defined here, but this
//! module persists none of it: it is the in-memory trigger type passed
//! between `rocmd`'s watcher and webhook plumbing (`apps/rocmd/src/watchers.rs`,
//! `apps/rocmd/src/webhook.rs`), never serialized.

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppPaths;
use crate::FileLock;
use crate::WatcherMode;
use crate::require_nonempty;
use crate::unix_time_millis;

#[derive(Debug, Clone, Copy)]
pub struct BuiltinWatcherSpec {
    pub id: &'static str,
    pub summary: &'static str,
    pub trigger: &'static str,
    pub default_mode: WatcherMode,
    pub actions: &'static [&'static str],
}

const BUILTIN_WATCHERS: &[BuiltinWatcherSpec] = &[
    BuiltinWatcherSpec {
        id: "therock-update",
        summary: "Emit scheduled TheRock update reminders and proposals.",
        trigger: "schedule: every 6h",
        default_mode: WatcherMode::Observe,
        actions: &["remind_update_check", "queue_update_proposal"],
    },
    BuiltinWatcherSpec {
        id: "server-recover",
        summary: "Observe or restart failed managed services when restart metadata exists.",
        trigger: "event: managed_service_failed",
        default_mode: WatcherMode::Contained,
        actions: &["collect_failure_snapshot", "restart_managed_service"],
    },
    BuiltinWatcherSpec {
        id: "gpu-metrics",
        summary: "Record read-only local amd-smi GPU telemetry availability; no proposals or mutations.",
        trigger: "event: gpu.metrics availability/unavailability",
        default_mode: WatcherMode::Observe,
        actions: &["record_gpu_metrics"],
    },
    BuiltinWatcherSpec {
        id: "cache-warm",
        summary: "Queue reviewed artifact prefetch proposals for registry model artifacts.",
        trigger: "event: cache.warm",
        default_mode: WatcherMode::Propose,
        actions: &["queue_prefetch_proposal"],
    },
    BuiltinWatcherSpec {
        id: "driver-upgrade",
        summary: "Queue reviewed read-only driver install plans when a local driver update signal is received.",
        trigger: "event: update.available component=driver",
        default_mode: WatcherMode::Propose,
        actions: &["prepare_driver_plan"],
    },
    BuiltinWatcherSpec {
        id: "gpu-thermal-protect",
        summary: "Queue reviewed stop-serving proposals when GPU temperature or memory pressure is high.",
        trigger: "event: gpu.thermal_pressure or gpu.memory_pressure",
        default_mode: WatcherMode::Propose,
        actions: &["queue_stop_server_proposal"],
    },
];

pub const fn builtin_watchers() -> &'static [BuiltinWatcherSpec] {
    BUILTIN_WATCHERS
}

pub fn builtin_watcher(id: &str) -> Option<&'static BuiltinWatcherSpec> {
    builtin_watchers().iter().find(|watcher| watcher.id == id)
}

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

/// Lock file serializing every mutation of the JSONL store at `path` — a
/// sibling named `<store file name>.lock` (e.g. `events.jsonl.lock`).
///
/// Appends and the proposal-status rewrite both take it. The rewrite needs it
/// for its read-modify-write; appends need it too, because the rewrite
/// replaces the store by renaming a new file over it, and an append that had
/// already opened the old file would land in the replaced copy and vanish.
fn jsonl_lock_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    path.with_file_name(name)
}

/// How many bytes the backward scans below read per step.
const JSONL_SCAN_CHUNK: u64 = 64 * 1024;

/// Append one JSONL `record` to `path`, creating the file if needed.
///
/// Holds the store's [`jsonl_lock_path`] lock and writes the line with a
/// single append-mode write, so the cost no longer grows with the store and
/// a concurrent writer can never drop this record (or have its own record
/// dropped by this one), across threads and processes alike.
///
/// A failed write is rolled back to the pre-append length, so a full disk
/// never leaves a torn line behind. A torn line left by a writer that was
/// killed mid-write is cleaned up by the next append: see
/// [`repair_unterminated_tail`].
fn append_jsonl_record<T: Serialize>(path: &Path, record: &T, context_label: &str) -> Result<()> {
    let mut line = serde_json::to_vec(record)
        .with_context(|| format!("failed to serialize {context_label}"))?;
    line.push(b'\n');

    let _lock = FileLock::acquire(jsonl_lock_path(path))?;
    let file = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let (start, needs_separator) = repair_unterminated_tail(path, &file)?;
    if needs_separator {
        line.insert(0, b'\n');
    }
    if let Err(error) = (&file).write_all(&line) {
        // Best effort: if this fails too, the torn tail is still repaired by
        // the next append, and readers already skip it.
        let _ = truncate_file(path, start);
        return Err(error).with_context(|| format!("failed to append to {}", path.display()));
    }
    Ok(())
}

/// Make the store at `path` end on a line boundary before an append, returning
/// the length to roll back to if the append fails and whether the new line
/// needs a leading `\n`.
///
/// Must be called with the store's lock held, so no other writer is mid-write
/// and an unterminated last line is either crash debris (a writer killed
/// mid-write: not valid JSON, cut off) or a complete record whose newline was
/// lost to a hand edit (valid JSON, kept and terminated).
fn repair_unterminated_tail(path: &Path, file: &fs::File) -> Result<(u64, bool)> {
    let len = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    // The common case — empty, or ending on a line boundary — costs one byte.
    if len == 0 || read_range(path, file, len - 1, len)? == b"\n" {
        return Ok((len, false));
    }
    let mut tail = Vec::new();
    let mut tail_start = len;
    while tail_start > 0 {
        let chunk_start = tail_start.saturating_sub(JSONL_SCAN_CHUNK);
        let chunk = read_range(path, file, chunk_start, tail_start)?;
        if let Some(newline) = chunk.iter().rposition(|&byte| byte == b'\n') {
            tail.splice(0..0, chunk[newline + 1..].iter().copied());
            tail_start = chunk_start + newline as u64 + 1;
            break;
        }
        tail.splice(0..0, chunk);
        tail_start = chunk_start;
    }
    if tail.is_empty() {
        return Ok((len, false));
    }
    if is_blank(&tail) || !is_torn_record(&tail) {
        return Ok((len, true));
    }
    truncate_file(path, tail_start)?;
    Ok((tail_start, false))
}

/// Read bytes `[start, end)` of `file`.
///
/// Returns fewer bytes if the file now ends before `end`: a lock-free reader
/// can race a writer cutting a torn last line (see
/// [`repair_unterminated_tail`]), and the bytes cut are never a complete
/// record, so what is still there is all the caller needs.
fn read_range(path: &Path, file: &fs::File, start: u64, end: u64) -> Result<Vec<u8>> {
    let mut reader = file;
    reader
        .seek(SeekFrom::Start(start))
        .with_context(|| format!("failed to seek in {}", path.display()))?;
    let mut bytes = Vec::new();
    reader
        .take(end - start)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(bytes)
}

/// Cut `path` back to `len` bytes. Opens its own write handle: the append
/// handle cannot be relied on to truncate on Windows, where append-only
/// access does not include the right to move end-of-file backwards.
fn truncate_file(path: &Path, len: u64) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_len(len))
        .with_context(|| format!("failed to truncate {}", path.display()))
}

/// Parse a whole JSONL store, pairing each record with its line index.
///
/// Readers take no lock, so one may catch an append mid-write: a last line
/// with no terminating `\n` that is not valid JSON is that in-flight (or
/// crash-torn) record, and is skipped rather than failing the whole read. Any
/// other unparseable line is still an error.
///
/// Works on bytes, not a `String`: a record cut mid-write can end inside a
/// multi-byte UTF-8 character, which would fail a whole-file UTF-8 decode
/// before the skip could apply.
fn parse_jsonl_records<T: DeserializeOwned>(
    bytes: &[u8],
    path: &Path,
    context_label: &str,
) -> Result<Vec<(usize, T)>> {
    let unterminated = !bytes.is_empty() && !bytes.ends_with(b"\n");
    let line_count = bytes.split(|&byte| byte == b'\n').count();
    let mut records = Vec::new();
    for (index, line) in bytes.split(|&byte| byte == b'\n').enumerate() {
        if is_blank(line) {
            continue;
        }
        match serde_json::from_slice::<T>(line) {
            Ok(record) => records.push((index, record)),
            Err(_) if unterminated && index + 1 == line_count && is_torn_record(line) => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to parse {context_label} in {}", path.display())
                });
            }
        }
    }
    Ok(records)
}

/// Read the most recent `limit` JSONL records from `path`, in on-disk order.
/// Returns an empty vec when `path` doesn't exist yet or `limit` is `0`.
///
/// Scans backwards from the end of the file and stops once it has `limit`
/// records, so the cost tracks `limit`, not the size of the store; a corrupt
/// line older than those records is therefore not reported. As in
/// [`parse_jsonl_records`], an unterminated last line that is not valid JSON is
/// an append caught mid-write and is skipped.
///
/// The scan takes no lock, so a writer may cut a torn last line (see
/// [`repair_unterminated_tail`]) and append again while it runs. A scan that
/// fails while the file's length moved under it is retried from the new end,
/// so that race never surfaces as an error.
fn load_recent_jsonl_records<T: DeserializeOwned>(
    path: &Path,
    limit: usize,
    context_label: &str,
) -> Result<Vec<T>> {
    const ATTEMPTS: usize = 3;
    if !path.is_file() || limit == 0 {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let file_len = || {
        file.metadata()
            .map(|metadata| metadata.len())
            .with_context(|| format!("failed to stat {}", path.display()))
    };
    let mut attempt = 1;
    loop {
        let len = file_len()?;
        match scan_recent_jsonl_records(path, &file, len, limit, context_label) {
            Err(_) if attempt < ATTEMPTS && file_len()? != len => attempt += 1,
            outcome => return outcome,
        }
    }
}

/// One backward scan of [`load_recent_jsonl_records`], over the first `len`
/// bytes of `file`.
fn scan_recent_jsonl_records<T: DeserializeOwned>(
    path: &Path,
    file: &fs::File,
    len: u64,
    limit: usize,
    context_label: &str,
) -> Result<Vec<T>> {
    let mut lines = ReverseLines::new(path, file, len, JSONL_SCAN_CHUNK);
    let parse = |line: &[u8]| {
        serde_json::from_slice::<T>(line)
            .with_context(|| format!("failed to parse {context_label} in {}", path.display()))
    };
    let mut records = Vec::new();
    // The first segment is whatever follows the last `\n`: empty when the
    // file ends on a line boundary, else an unterminated last line.
    if let Some(tail) = lines.next_line()?
        && !is_blank(&tail)
        && !is_torn_record(&tail)
    {
        records.push(parse(&tail)?);
    }
    while records.len() < limit {
        let Some(line) = lines.next_line()? else {
            break;
        };
        if !is_blank(&line) {
            records.push(parse(&line)?);
        }
    }
    records.truncate(limit);
    records.reverse();
    Ok(records)
}

fn is_blank(line: &[u8]) -> bool {
    line.iter().all(u8::is_ascii_whitespace)
}

/// Whether an unterminated last line is a record cut off mid-write, as
/// opposed to a complete one that merely lost its `\n` (e.g. to a hand edit).
///
/// The test is "not valid JSON", not "not a valid record": a cut-off JSON
/// object never parses, while complete JSON of the wrong shape is real data —
/// readers report it as before, and writers keep it rather than cut it.
fn is_torn_record(line: &[u8]) -> bool {
    serde_json::from_slice::<serde::de::IgnoredAny>(line).is_err()
}

/// Yields the `\n`-separated segments of a file from last to first, reading
/// it backwards in fixed-size chunks. The first segment yielded is the one
/// after the final `\n` (empty when the file ends with one).
struct ReverseLines<'a> {
    path: &'a Path,
    file: &'a fs::File,
    chunk: u64,
    /// File offset of `pending[0]`; everything before it is still unread.
    pos: u64,
    /// Bytes read but not yet yielded, i.e. file bytes `[pos, pos + len)`.
    pending: Vec<u8>,
    done: bool,
}

impl<'a> ReverseLines<'a> {
    /// Read the first `len` bytes of `file` backwards, `chunk` bytes a step.
    const fn new(path: &'a Path, file: &'a fs::File, len: u64, chunk: u64) -> Self {
        Self {
            path,
            file,
            chunk: if chunk == 0 { 1 } else { chunk },
            pos: len,
            pending: Vec::new(),
            done: false,
        }
    }

    fn next_line(&mut self) -> Result<Option<Vec<u8>>> {
        if self.done {
            return Ok(None);
        }
        loop {
            // Scans from the end, so it stops at the nearest `\n`: a line is
            // rescanned only when it is longer than one chunk.
            if let Some(newline) = self.pending.iter().rposition(|&byte| byte == b'\n') {
                let line = self.pending.split_off(newline + 1);
                self.pending.truncate(newline);
                return Ok(Some(line));
            }
            if self.pos == 0 {
                self.done = true;
                return Ok(Some(std::mem::take(&mut self.pending)));
            }
            let start = self.pos.saturating_sub(self.chunk);
            let mut bytes = read_range(self.path, self.file, start, self.pos)?;
            // A short first read just means a torn tail was cut (see
            // `read_range`). Short after that, the bytes already read no longer
            // continue from these, so splicing them would invent a line.
            if bytes.len() as u64 != self.pos - start && !self.pending.is_empty() {
                bail!("{} shrank while it was being read", self.path.display());
            }
            bytes.append(&mut self.pending);
            self.pending = bytes;
            self.pos = start;
        }
    }
}

pub fn append_automation_event(paths: &AppPaths, event: &AutomationEventRecord) -> Result<()> {
    paths.ensure()?;
    append_jsonl_record(
        &paths.automation_events_path(),
        event,
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
    append_jsonl_record(
        &paths.automation_proposals_path(),
        &proposal,
        "automation proposal record",
    )
}

pub fn append_audit_event(paths: &AppPaths, event: &AuditEventRecord) -> Result<()> {
    paths.ensure()?;
    append_jsonl_record(&paths.audit_events_path(), event, "audit event record")
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

pub fn load_recent_automation_proposals(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationProposalRecord>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut proposals = load_automation_proposals(paths)?;
    proposals.reverse();
    proposals.truncate(limit);
    Ok(proposals)
}

pub fn find_automation_proposal(
    paths: &AppPaths,
    proposal_id: &str,
) -> Result<AutomationProposalRecord> {
    load_automation_proposals(paths)?
        .into_iter()
        .find(|proposal| proposal.proposal_id == proposal_id)
        .with_context(|| format!("automation proposal `{proposal_id}` not found"))
}

/// Set one proposal's `status` by rewriting the proposals store.
///
/// The whole read-modify-write runs under the store's [`jsonl_lock_path`]
/// lock, so a proposal appended (or another status set) meanwhile is neither
/// dropped nor overwritten.
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
    write_automation_proposals(paths, &proposals)?;
    Ok(updated)
}

pub fn load_recent_audit_events(paths: &AppPaths, limit: usize) -> Result<Vec<AuditEventRecord>> {
    load_recent_jsonl_records(&paths.audit_events_path(), limit, "audit event")
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
    load_recent_jsonl_records(&paths.automation_events_path(), limit, "event")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;

    #[test]
    fn builtin_watchers_include_read_only_gpu_metrics() {
        let watcher = builtin_watcher("gpu-metrics").expect("gpu-metrics watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Observe);
        assert!(watcher.trigger.contains("gpu.metrics"));
        assert_eq!(watcher.actions, &["record_gpu_metrics"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_cache_warm() {
        let watcher = builtin_watcher("cache-warm").expect("cache-warm watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("cache.warm"));
        assert_eq!(watcher.actions, &["queue_prefetch_proposal"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_driver_upgrade() {
        let watcher =
            builtin_watcher("driver-upgrade").expect("driver-upgrade watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("update.available"));
        assert!(watcher.trigger.contains("component=driver"));
        assert_eq!(watcher.actions, &["prepare_driver_plan"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_gpu_thermal_protect() {
        let watcher = builtin_watcher("gpu-thermal-protect")
            .expect("gpu-thermal-protect watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("gpu.thermal_pressure"));
        assert!(watcher.trigger.contains("gpu.memory_pressure"));
        assert_eq!(watcher.actions, &["queue_stop_server_proposal"]);
    }

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

    fn audit_record(tag: &str) -> AuditEventRecord {
        AuditEventRecord {
            at_unix_ms: 1,
            source: "test".to_owned(),
            category: "concurrency".to_owned(),
            actor: "writer".to_owned(),
            level: "info".to_owned(),
            action: "append".to_owned(),
            message: tag.to_owned(),
            watcher_id: None,
            service_id: None,
        }
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
        let barrier = std::sync::Barrier::new(WRITERS);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WRITERS)
                .map(|writer| {
                    let (paths, barrier) = (&paths, &barrier);
                    scope.spawn(move || -> Result<()> {
                        barrier.wait();
                        for n in 0..APPENDS_PER_WRITER {
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

        let expected: Vec<String> = (0..WRITERS)
            .flat_map(|writer| (0..APPENDS_PER_WRITER).map(move |n| format!("t{writer}-{n}")))
            .collect();
        let stores = tags_in_every_store(&paths);
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

    // Readers take no lock, so they can catch an append mid-write; that must
    // never surface as a parse error from `rocm status` or the dashboard.
    // This is a smoke test: on Linux a small append is rarely seen half-done,
    // so the skip itself is pinned deterministically by
    // `readers_skip_an_unterminated_last_line_that_does_not_parse`.
    #[test]
    fn reads_racing_appends_never_fail() -> Result<()> {
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

    /// Write `contents` as the audit store, creating its directory.
    fn seed_audit_store(paths: &AppPaths, contents: &str) -> Result<()> {
        paths.ensure()?;
        fs::write(paths.audit_events_path(), contents)?;
        Ok(())
    }

    fn audit_line(tag: &str) -> String {
        serde_json::to_string(&audit_record(tag)).expect("serialize audit record")
    }

    fn audit_messages(records: &[AuditEventRecord]) -> Vec<&str> {
        records.iter().map(|r| r.message.as_str()).collect()
    }

    #[test]
    fn reverse_lines_matches_a_forward_split_at_every_chunk_size() -> Result<()> {
        let (root, _paths) = temp_app_paths("reverse-lines");
        fs::create_dir_all(&root)?;
        let path = root.join("lines.jsonl");
        for contents in [
            "",
            "\n",
            "a",
            "a\n",
            "a\nbb\n",
            "a\nbb\nccc",
            "\n\nx\n\n",
            "long-line-crossing-chunks\nshort\n",
        ] {
            fs::write(&path, contents)?;
            let file = fs::File::open(&path)?;
            let mut expected: Vec<&[u8]> = contents.as_bytes().split(|&b| b == b'\n').collect();
            expected.reverse();
            for chunk in 1..=8 {
                let mut lines = ReverseLines::new(&path, &file, file.metadata()?.len(), chunk);
                let mut got = Vec::new();
                while let Some(line) = lines.next_line()? {
                    got.push(line);
                }
                assert_eq!(got, expected, "contents {contents:?}, chunk {chunk}");
            }
        }
        fs::remove_dir_all(root).ok();
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

    // A lock-free reader that read the length just before a writer cut a torn
    // tail sees the file end early; that is not an error.
    #[test]
    fn a_scan_that_finds_the_file_shorter_than_it_was_still_succeeds() -> Result<()> {
        let (root, paths) = temp_app_paths("scan-shrunk");
        seed_audit_store(
            &paths,
            &format!("{}\n{}\n", audit_line("a"), audit_line("b")),
        )?;
        let path = paths.audit_events_path();
        let file = fs::File::open(&path)?;
        let stale_len = file.metadata()?.len() + 40;
        let recent =
            scan_recent_jsonl_records::<AuditEventRecord>(&path, &file, stale_len, 5, "audit");
        fs::remove_dir_all(root).ok();
        assert_eq!(audit_messages(&recent?), ["a", "b"]);
        Ok(())
    }

    #[test]
    fn reverse_lines_refuses_to_splice_across_a_mid_scan_shrink() -> Result<()> {
        let (root, _paths) = temp_app_paths("reverse-lines-shrink");
        fs::create_dir_all(&root)?;
        let path = root.join("lines.jsonl");
        fs::write(&path, "aaaa\nbbbb\ncccc")?;
        let file = fs::File::open(&path)?;
        let mut lines = ReverseLines::new(&path, &file, 14, 4);
        // As if the first step had read the last chunk, "cccc" (no `\n` yet),
        // and the file then shrank below the next chunk's end.
        lines.pending = b"cccc".to_vec();
        lines.pos = 10;
        let shrunk = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .and_then(|f| f.set_len(7));
        let next = lines.next_line();
        fs::remove_dir_all(root).ok();
        shrunk?;
        assert!(
            next.is_err(),
            "a short read after the first must not be spliced"
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

    #[test]
    fn the_store_lock_is_a_sibling_of_the_store() {
        let paths = AppPaths {
            config_dir: PathBuf::from("c"),
            data_dir: PathBuf::from("d"),
            cache_dir: PathBuf::from("k"),
        };
        assert_eq!(
            jsonl_lock_path(&paths.audit_events_path()),
            paths.audit_dir().join("events.jsonl.lock")
        );
        assert_eq!(
            jsonl_lock_path(&paths.automation_proposals_path()),
            paths.automations_dir().join("proposals.jsonl.lock")
        );
    }
}
