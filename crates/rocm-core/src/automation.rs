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
//! Appending a record is, in the common case, one append-mode write of one
//! line — its cost does not grow with the store. A torn tail — crash debris,
//! or left by a write that failed partway — is cut by the next append rather
//! than truncated in place: that append writes the store's clean prefix plus
//! its own new record as a single rewrite via
//! [`crate::atomic_write::write_file_atomically`], so a lock-free reader never
//! observes a file shrunk mid-repair. That rewrite costs O(store) rather than
//! O(record), the same as a proposal status update, which rewrites the whole
//! store the same way. Nothing in this module ever shrinks a store in place.
//! Readers take no lock, and an unterminated last
//! line that is not valid JSON — an append caught mid-write — is skipped
//! rather than reported as corruption.
//!
//! The events and audit stores are additionally capped by size-based rotation
//! (see [`EVENT_LOG_ROTATE_BYTES`]): the append that finds the live file at
//! the cap renames it whole to `<store>.1` first, under the same lock —
//! before any torn tail in it would otherwise be repaired, so an at-cap store
//! is never paid for with an atomic rewrite only to be rotated away
//! immediately after — and the recent-record readers span the rotated
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
use same_file::Handle;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
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
/// Appends and the proposal-status rewrite both take it. The proposal
/// rewrite needs it for its read-modify-write; an append needs it both to
/// serialize against other appends and because it can itself replace or
/// rename the live file out from under a concurrent writer — a torn tail is
/// repaired by renaming a new file over the store (see [`TailState::Torn`]),
/// and a rotating events or audit store is renamed aside to `<store>.1` (see
/// [`rotate_jsonl_generations`]) — and an append that had already opened the
/// old file would otherwise land in the replaced or rotated-away copy and
/// vanish.
fn jsonl_lock_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    path.with_file_name(name)
}

/// How many bytes the backward scans below read per step.
const JSONL_SCAN_CHUNK: u64 = 64 * 1024;

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

/// Size-based rotation policy for a JSONL store.
#[derive(Debug, Clone, Copy)]
struct JsonlRotation {
    /// Rotate before an append once the live file is at least this long.
    max_bytes: u64,
    /// Rotated generations to keep besides the live file; at least 1.
    generations: usize,
}

/// The policy for the automation events store and the audit store.
const EVENT_LOG_ROTATION: JsonlRotation = JsonlRotation {
    max_bytes: EVENT_LOG_ROTATE_BYTES,
    generations: EVENT_LOG_ROTATED_GENERATIONS,
};

/// Path of generation `generation` of the store at `path`: the live file for
/// `0`, else `<store file name>.<generation>` beside it (`events.jsonl.1`).
fn jsonl_generation_path(path: &Path, generation: usize) -> PathBuf {
    if generation == 0 {
        return path.to_path_buf();
    }
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{generation}"));
    path.with_file_name(name)
}

/// Rotate the live file to `.1`, first shifting the generations in its way
/// up by one — `.{n-1}` over `.{n}` (dropping the oldest), …, `.1` to `.2` —
/// and leaving no live file for the caller's append to create. The caller must
/// hold the store's [`jsonl_lock_path`] lock, so no append can land in a file
/// being renamed.
///
/// Only the unbroken run of generations from `.1` up is shifted: a missing
/// generation is a free slot, so nothing older than it has to move, and
/// nothing is dropped. That matters after a rotation that stopped part-way
/// (shifted `.1` away, then failed to rename the live file, leaving a gap at
/// `.1`): the next rotation then only fills the gap, instead of shifting — and
/// dropping — one more generation each time it runs.
///
/// Renames newest-last, so at every instant the files present, read live
/// first, are still in order with no file under two names; a lock-free reader
/// relies on that (see [`open_jsonl_generations`]). Stops at the first failed
/// rename: going on could rename a file over one that was never shifted away.
fn rotate_jsonl_generations(path: &Path, generations: usize) -> Result<()> {
    let mut run = 0;
    while run < generations && fs::symlink_metadata(jsonl_generation_path(path, run + 1)).is_ok() {
        run += 1;
    }
    // When every generation exists, the oldest is renamed over, not moved.
    for generation in (1..=run.min(generations - 1)).rev() {
        let from = jsonl_generation_path(path, generation);
        let to = jsonl_generation_path(path, generation + 1);
        fs::rename(&from, &to)
            .with_context(|| format!("failed to rotate {} to {}", from.display(), to.display()))?;
    }
    let first = jsonl_generation_path(path, 1);
    fs::rename(path, &first)
        .with_context(|| format!("failed to rotate {} to {}", path.display(), first.display()))
}

fn open_jsonl_for_append(path: &Path) -> Result<fs::File> {
    fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))
}

/// Append one JSONL `record` to `path`, creating the file if needed.
///
/// Holds the store's [`jsonl_lock_path`] lock. The common case — the store
/// already ends on a line boundary — is a single append-mode write of one
/// line, so the cost no longer grows with the store and a concurrent writer
/// can never drop this record (or have its own record dropped by this one),
/// across threads and processes alike.
///
/// A write that fails partway leaves whatever it already wrote as a new torn
/// tail — the same debris a killed writer leaves — rather than rolling it
/// back, and the debris itself is discarded by the *next* append rather than
/// being cut first as a separate step — see [`inspect_tail`] and
/// [`TailState::Torn`]. A reader already skips the debris in the meantime.
/// Nothing in this function ever truncates the live file, so a lock-free
/// reader never observes it shrink.
///
/// [`append_or_tolerate_a_missing_newline`] checks whether a failed write
/// landed anyway before reporting it, so a write that fails but still lands
/// a complete record is reported as `Ok`, not `Err`. Seeing `Err` here
/// usually does mean the record did not land — with one case where that's
/// not knowable rather than false: if the write itself fails *and* the
/// follow-up check of what actually landed also fails (a second I/O fault
/// on the same unhealthy disk, say), the original write failure is reported
/// without knowing whether the record landed anyway. A caller that retries
/// on `Err` in that specific, low-likelihood case could write a duplicate
/// the store has no way to detect or remove.
///
/// With a `rotation` policy, a live file already at `max_bytes` is rotated
/// aside (see [`rotate_jsonl_generations`]) *before* [`inspect_tail`] runs —
/// whole, debris and all, rather than repairing a torn tail first and
/// rotating the repaired copy away a moment later. Rotation is best effort:
/// if it fails, the record is still appended to the unrotated file via the
/// normal `inspect_tail` path below, and the next append tries again — losing
/// a record to a failed rename would be worse than one oversized file.
fn append_jsonl_record<T: Serialize>(
    path: &Path,
    record: &T,
    rotation: Option<JsonlRotation>,
    context_label: &str,
) -> Result<()> {
    let mut line = serde_json::to_vec(record)
        .with_context(|| format!("failed to serialize {context_label}"))?;
    line.push(b'\n');

    let _lock = FileLock::acquire(jsonl_lock_path(path))?;
    let mut file = open_jsonl_for_append(path)?;
    if let Some(rotation) = rotation {
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        if len >= rotation.max_bytes {
            // Close the live file before renaming it, so no write handle of
            // ours is held across the rename and the reopen below gets the
            // new file.
            drop(file);
            if let Err(error) = rotate_jsonl_generations(path, rotation.generations)
                && let Some(warning) = rotation_failure_warning(path, &error)
            {
                eprintln!("{warning}");
            }
            file = open_jsonl_for_append(path)?;
        }
    }
    match inspect_tail(path, &file)? {
        TailState::Torn(tail_start) => {
            // A torn tail is debris — crash or a failed write — that must be
            // cut before this record lands. Cutting it in place and then
            // appending as a second step would briefly shrink the file for
            // any lock-free reader mid-scan, and regrowing it afterward can
            // make that scan splice bytes read at two different moments into
            // something that parses as a record nobody wrote (the hazard
            // `ReverseLines::next_line`'s short-read guard exists for).
            // Replacing the whole file atomically instead means a reader
            // only ever sees the pre-repair length or this append's, never a
            // length in between.
            let mut content = read_range(path, &file, 0, tail_start)?;
            content.extend_from_slice(&line);
            // Done with the pre-repair file's handle before replacing it,
            // rather than leaving that to whatever happens to still have
            // `file` open at function return. The replace already succeeds
            // with this handle open regardless — std opens files with
            // `FILE_SHARE_DELETE` by default on Windows, so it isn't blocked
            // by it (see the `atomic_write` module doc for how the publish
            // treats a destination another process holds open) — but this
            // states the intent rather than leaning on that default.
            drop(file);
            crate::atomic_write::write_file_atomically(path, &content)?;
        }
        TailState::ValidUnterminated(start) => {
            line.insert(0, b'\n');
            append_or_tolerate_a_missing_newline(path, &file, &line, start)?;
        }
        TailState::Clean(start) => {
            append_or_tolerate_a_missing_newline(path, &file, &line, start)?;
        }
    }
    Ok(())
}

/// The warning to print that rotating `path` failed, or `None` after the first
/// one for that store in this process, so a rename that keeps failing (e.g.
/// another program holds the live file open without delete sharing on
/// Windows) is observable without a line on every single append while the
/// store stays stuck over its cap. Keyed by path so one store's failure does
/// not silence another's.
fn rotation_failure_warning(path: &Path, error: &anyhow::Error) -> Option<String> {
    static WARNED: Mutex<BTreeSet<PathBuf>> = Mutex::new(BTreeSet::new());
    WARNED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(path.to_path_buf())
        .then(|| format!("warning: failed to rotate {}: {error:#}", path.display()))
}

/// Append-mode write of `line`, starting at offset `start`.
///
/// A write can fail after landing every byte of `line` except its final
/// `\n` (e.g. the disk fills up one byte short). That is not torn debris:
/// it is a complete, valid record, and readers already treat a complete
/// record missing only its trailing newline as fully stored (see
/// `readers_keep_an_unterminated_last_line_that_parses`). Reporting that as
/// a failure would tell a caller that retries on error to write the same
/// record again, leaving a duplicate the store has no way to detect or
/// remove — so this checks what actually landed before deciding.
///
/// That check can itself fail (stat or read-back hits its own I/O error),
/// in which case this reports the original write failure without knowing
/// whether the record landed anyway — the one case where `Err` here does
/// not guarantee the record is absent (see [`append_jsonl_record`]'s doc).
fn append_or_tolerate_a_missing_newline(
    path: &Path,
    file: &fs::File,
    line: &[u8],
    start: u64,
) -> Result<()> {
    let mut file = file;
    let Err(error) = file.write_all(line) else {
        return Ok(());
    };
    let landed = (|| -> Result<Vec<u8>> {
        let end = file
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len()
            .max(start);
        read_range(path, file, start, end)
    })();
    match landed {
        Ok(landed) if landed_despite_the_error(&landed, line) => Ok(()),
        Ok(_) => Err(error).with_context(|| format!("failed to append to {}", path.display())),
        // The write itself is still the more informative failure (e.g. the
        // disk is full); checking whether it landed anyway is this
        // function's own extra step, so losing that check is a detail
        // folded into the same report rather than the headline error.
        Err(check_error) => Err(error).with_context(|| {
            format!(
                "failed to append to {} ({check_error:#} while checking whether it landed anyway)",
                path.display()
            )
        }),
    }
}

/// Whether `landed` — the bytes a failed append-mode write actually put on
/// disk, read back from where it started — show `line` arrived intact
/// except possibly for its own trailing `\n`. Split out from
/// [`append_or_tolerate_a_missing_newline`] so the decision itself can be
/// tested without inducing a real write failure.
fn landed_despite_the_error(landed: &[u8], line: &[u8]) -> bool {
    let without_trailing_newline = &line[..line.len().saturating_sub(1)];
    landed.len() >= without_trailing_newline.len()
        && landed[..without_trailing_newline.len()] == *without_trailing_newline
}

/// How [`inspect_tail`] found the store's current last line, ahead of an
/// append. The `Clean`/`ValidUnterminated` payload is the offset this
/// append's own bytes will start at, used by
/// [`append_or_tolerate_a_missing_newline`] to tell a complete record that
/// merely failed to land its own trailing newline from one that is actually
/// missing.
enum TailState {
    /// Empty, or already ends on a line boundary.
    Clean(u64),
    /// A complete record, or a blank (whitespace-only) line, lost its
    /// trailing `\n` — to a hand edit, or to a write that landed every byte
    /// except that final one; the new line needs a leading `\n`.
    ValidUnterminated(u64),
    /// Debris — crash, or a write that failed partway through a genuinely
    /// incomplete record — starting at this offset. Must be cut before the
    /// next record lands, but never by truncating in place — see
    /// [`append_jsonl_record`].
    Torn(u64),
}

/// Classify the store's current last line ahead of an append.
///
/// Must be called with the store's lock held, so no other writer is
/// mid-write and an unterminated last line is either debris (not valid JSON,
/// cut off — a writer killed mid-write, or one whose own append failed
/// partway) or a complete record or blank line whose newline was lost to a
/// hand edit or a write that failed one byte short (kept and terminated).
fn inspect_tail(path: &Path, file: &fs::File) -> Result<TailState> {
    let len = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    // The common case — empty, or ending on a line boundary — costs one byte.
    if len == 0 || read_range(path, file, len - 1, len)? == b"\n" {
        return Ok(TailState::Clean(len));
    }
    // The segment after the last `\n` is exactly `ReverseLines`'s first line,
    // so reuse its backward scan rather than re-implementing it here.
    let (tail_start, tail) = ReverseLines::new(path, file, len, JSONL_SCAN_CHUNK)
        .next_line()?
        .unwrap_or_default();
    if tail.is_empty() {
        return Ok(TailState::Clean(len));
    }
    if is_blank(&tail) || !is_torn_record(&tail) {
        return Ok(TailState::ValidUnterminated(len));
    }
    Ok(TailState::Torn(tail_start))
}

/// Read bytes `[start, end)` of `file`.
///
/// Returns fewer bytes if the file now ends before `end`. Nothing this
/// module's own writers do can shrink a store in place — an append only ever
/// grows it, and a torn-tail repair replaces it wholesale via the atomic writer
/// rather than truncating the live file (see [`inspect_tail`] and
/// [`TailState::Torn`]) — so a short read here means the store was edited or
/// truncated out-of-band, a case this module does not try to recover from
/// cleanly. Most callers treat what is still there the same as any other
/// unexpectedly-ending tail; [`ReverseLines`] is the one exception, and
/// reports an error instead once it would have to splice such a read onto
/// bytes it already read, rather than risk inventing a line.
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
    // Redundant with "this is the last split segment and it's non-blank"
    // below (splitting on `\n` makes the final segment empty exactly when
    // `bytes` ends in one, so a non-blank final segment can only occur when
    // it doesn't) — kept spelled out because it states directly what that
    // arm is for, rather than leaning on the split/is_blank interaction to
    // prove it.
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

/// Read the most recent `limit` JSONL records from the store at `path` and
/// its first `generations` rotated generations (see
/// [`rotate_jsonl_generations`]), in on-disk order, oldest first. Returns an
/// empty vec when no generation exists yet or `limit` is `0`.
///
/// Reads the live file first and moves on to `.1`, `.2`, … only while it is
/// still short of `limit`, so a read just after a rotation still returns the
/// records that were rotated aside. Each file is scanned backwards from its
/// end, stopping once it has `limit` records, so the cost tracks `limit`, not
/// the size of the store; a corrupt line older than those records is
/// therefore not reported. As in [`parse_jsonl_records`], an unterminated
/// last line that is not valid JSON is an append caught mid-write and is
/// skipped.
fn load_recent_jsonl_records<T: DeserializeOwned>(
    path: &Path,
    generations: usize,
    limit: usize,
    context_label: &str,
) -> Result<Vec<T>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut newest_first = Vec::new();
    let mut found = 0;
    for (generation_path, handle) in open_jsonl_generations(path, generations)? {
        let records = load_recent_jsonl_records_from(
            &generation_path,
            handle.as_file(),
            limit - found,
            context_label,
        )?;
        found += records.len();
        newest_first.push(records);
        if found >= limit {
            break;
        }
    }
    Ok(newest_first.into_iter().rev().flatten().collect())
}

/// Open the store's live file and its first `generations` rotated
/// generations, newest first, skipping any that do not exist — as one
/// consistent snapshot, so the readers can take no lock.
///
/// A rotation can rename files between two of the opens: opening the live
/// file and then `.1` across one would open the same file twice (and read its
/// records twice), and opening `.1` and then the live file across one would
/// skip a generation. A torn-tail repair (see [`TailState::Torn`]) also
/// replaces the live file, by renaming a new one over it — not a rotation,
/// but the same hazard for this snapshot: a path opened before the repair
/// would still read the pre-repair file by handle, yet a path opened after
/// would see different content at the same name. So after opening them all,
/// every path is opened again and must still name the same file, or still be
/// missing; if any moved, the whole set is reopened — the identity check does
/// not need to tell a rotation from a repair apart, it only needs one retry
/// either way. A rotation only ever moves a file to a higher generation, and
/// a new live file (whether from rotation or repair) is a new file, so a path
/// naming the same file at both opens named it throughout: the handles are
/// what every path named at one instant between the two passes, which
/// [`rotate_jsonl_generations`] keeps in order with no file under two names.
/// The handles stay open, so a dropped generation is still readable and its
/// file identity cannot be reused meanwhile.
///
/// Giving up after eight retries returns an error naming `path`, but some
/// callers (the dashboard's status view, among others) read recent records
/// through `.unwrap_or_default()` and would present that as an empty
/// activity list rather than surface it. Accepted: reaching the retry limit
/// needs the file to keep moving under every one of eight consecutive opens,
/// far more contention than a rotation once per `EVENT_LOG_ROTATE_BYTES` of
/// appends produces, and the very next read tries again from scratch.
fn open_jsonl_generations(path: &Path, generations: usize) -> Result<Vec<(PathBuf, Handle)>> {
    // A rotation happens once per `EVENT_LOG_ROTATE_BYTES` of appends, so
    // even one retry is rare; running out means something else is renaming
    // the files.
    const ATTEMPTS: usize = 8;
    let open = |path: &Path| -> Result<Option<Handle>> {
        match fs::File::open(path) {
            Ok(file) => Handle::from_file(file)
                .map(Some)
                .with_context(|| format!("failed to identify {}", path.display())),
            // Windows cannot open a directory this way; like a missing file,
            // it holds no records (a directory opened on Linux is skipped
            // below).
            Err(error) if error.kind() == ErrorKind::NotFound || path.is_dir() => Ok(None),
            Err(error) => Err(error).with_context(|| format!("failed to open {}", path.display())),
        }
    };
    let paths: Vec<PathBuf> = (0..=generations)
        .map(|generation| jsonl_generation_path(path, generation))
        .collect();
    for _ in 0..ATTEMPTS {
        let handles = paths
            .iter()
            .map(|path| open(path))
            .collect::<Result<Vec<_>>>()?;
        // A single file cannot be read twice or skipped.
        let mut stable = generations == 0;
        if !stable {
            stable = true;
            for (path, handle) in paths.iter().zip(&handles) {
                if open(path)?.as_ref() != handle.as_ref() {
                    stable = false;
                    break;
                }
            }
        }
        if !stable {
            continue;
        }
        let mut files = Vec::new();
        for (path, handle) in paths.into_iter().zip(handles) {
            let Some(handle) = handle else { continue };
            let metadata = handle
                .as_file()
                .metadata()
                .with_context(|| format!("failed to stat {}", path.display()))?;
            if metadata.is_file() {
                files.push((path, handle));
            }
        }
        return Ok(files);
    }
    bail!("{} kept being rotated while it was read", path.display())
}

/// [`load_recent_jsonl_records`] over one file, `file`, opened from `path`.
///
/// The scan takes no lock. A concurrent append only ever grows the file (see
/// [`append_jsonl_record`]); nothing this module's own writers do shrinks it
/// in place. The retry below is for a store edited or truncated out-of-band
/// — by a person, not a concurrent writer — while a scan is in flight: a
/// scan that fails while the file's length moved under it is retried from
/// the new end, so that still never surfaces as an error.
fn load_recent_jsonl_records_from<T: DeserializeOwned>(
    path: &Path,
    file: &fs::File,
    limit: usize,
    context_label: &str,
) -> Result<Vec<T>> {
    const ATTEMPTS: usize = 3;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let file_len = || {
        file.metadata()
            .map(|metadata| metadata.len())
            .with_context(|| format!("failed to stat {}", path.display()))
    };
    let mut attempt = 1;
    loop {
        let len = file_len()?;
        match scan_recent_jsonl_records(path, file, len, limit, context_label) {
            Err(_) if attempt < ATTEMPTS && file_len()? != len => attempt += 1,
            outcome => return outcome,
        }
    }
}

/// One backward scan of [`load_recent_jsonl_records_from`], over the first
/// `len` bytes of `file`.
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
    if let Some((_, tail)) = lines.next_line()?
        && !is_blank(&tail)
        && !is_torn_record(&tail)
    {
        records.push(parse(&tail)?);
    }
    while records.len() < limit {
        let Some((_, line)) = lines.next_line()? else {
            break;
        };
        if !is_blank(&line) {
            records.push(parse(&line)?);
        }
    }
    // No-op today: the loop above never pushes past `limit`, and the one
    // production caller (`load_recent_jsonl_records`) already refuses
    // `limit == 0` before calling in. Kept as a guard against a future
    // direct caller that skips that guard — the tail-line push above the
    // loop isn't itself gated by `limit`, so it would return one record for
    // `limit == 0` without this.
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
///
/// How many chunks get read depends on the file's `\n` layout, not just its
/// length: a single line longer than `chunk` is rescanned from its own start
/// on every subsequent chunk read until its leading `\n` is found (each read
/// re-prepends onto the same growing `pending`), so one pathologically long
/// line costs O(length² / chunk) rather than O(length). Fine for this
/// module's records and the deliberately oversized test cases below; a
/// store whose lines could grow unbounded would need a different scan.
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

    /// Yields `(start, bytes)`: `start` is the returned line's absolute file
    /// offset, derived from `self.pos` (which tracks real, current file
    /// offsets as actually read) rather than from a length measured earlier
    /// by the caller — which may be stale by the time a scan reaches it.
    fn next_line(&mut self) -> Result<Option<(u64, Vec<u8>)>> {
        if self.done {
            return Ok(None);
        }
        loop {
            // Scans from the end, so it stops at the nearest `\n`: a line is
            // rescanned only when it is longer than one chunk.
            if let Some(newline) = self.pending.iter().rposition(|&byte| byte == b'\n') {
                let line = self.pending.split_off(newline + 1);
                self.pending.truncate(newline);
                let start = self.pos + newline as u64 + 1;
                return Ok(Some((start, line)));
            }
            if self.pos == 0 {
                self.done = true;
                return Ok(Some((0, std::mem::take(&mut self.pending))));
            }
            let start = self.pos.saturating_sub(self.chunk);
            #[cfg(test)]
            test_hooks::before_scan_read(self.path);
            let mut bytes = read_range(self.path, self.file, start, self.pos)?;
            // A short read while `pending` is empty — nothing accumulated
            // yet, whether from an earlier call or from this one — just
            // means the store was edited or truncated out-of-band (see
            // `read_range`); nothing a concurrent writer does can shrink it
            // in place. Once `pending` holds any bytes, a short read means
            // they no longer continue from it, so splicing them would
            // invent a line.
            if bytes.len() as u64 != self.pos - start && !self.pending.is_empty() {
                bail!("{} shrank while it was being read", self.path.display());
            }
            bytes.append(&mut self.pending);
            self.pending = bytes;
            self.pos = start;
        }
    }
}

/// A seam for forcing an out-of-band shrink between two of `ReverseLines`'s
/// own chunked reads, so a test can drive `load_recent_jsonl_records`'s
/// retry-after-a-failed-scan path deterministically. Mirrors the
/// path-keyed-callback pattern in `uv.rs`'s `test_hooks`, but each hook is
/// one-shot and keyed by how many reads to let pass first: unlike an install
/// download's single commit point, a scan's vulnerable read (the one where
/// `pending` already holds bytes from an earlier read) is not the first one.
#[cfg(test)]
mod test_hooks {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, LazyLock, Mutex, PoisonError};

    type Hook = Arc<dyn Fn() + Send + Sync>;

    /// Keyed by path, value is (reads left to skip, hook to run at zero).
    static BEFORE_SCAN_READ: LazyLock<Mutex<HashMap<PathBuf, (usize, Hook)>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Run `hook` once, immediately before the `skip`-th subsequent physical
    /// chunk read `ReverseLines` performs against `path` (`0` = the very next
    /// read). Removed after it fires, so later reads of the same scan (or a
    /// retried one) proceed unhooked.
    pub(super) fn set_before_scan_read(path: &Path, skip: usize, hook: Hook) {
        BEFORE_SCAN_READ
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(path.to_path_buf(), (skip, hook));
    }

    pub(super) fn before_scan_read(path: &Path) {
        let mut hooks = BEFORE_SCAN_READ
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some((skip, hook)) = hooks.get_mut(path) else {
            return;
        };
        if *skip > 0 {
            *skip -= 1;
            return;
        }
        let hook = Arc::clone(hook);
        hooks.remove(path);
        drop(hooks);
        hook();
    }
}

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
    crate::atomic_write::write_file_atomically(&path, text.as_bytes())
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
    use crate::test_support::{
        assert_write_spares_foreign_temp_files, retired_temp_path, temp_app_paths,
    };

    #[test]
    fn a_rotation_failure_is_reported_once_per_store() {
        // Shaped like the real stores, which share a file name and differ
        // only by directory: keying on the file name alone must not pass.
        let events = Path::new("rotation-warning-test/automations/events.jsonl");
        let audit = Path::new("rotation-warning-test/audit/events.jsonl");
        let error = anyhow::anyhow!("rename blocked");

        let first = rotation_failure_warning(events, &error)
            .expect("the first failure for a store is reported");
        assert!(first.contains("rotation-warning-test/automations/events.jsonl"));
        assert!(first.contains("rename blocked"));
        assert_eq!(rotation_failure_warning(events, &error), None);
        let other = rotation_failure_warning(audit, &error)
            .expect("one store's failure must not silence another store's warning");
        assert!(other.contains("rotation-warning-test/audit/events.jsonl"));
        assert_eq!(rotation_failure_warning(audit, &error), None);
    }

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

    /// A status update never writes through a file another writer already
    /// staged next to the proposals store. The occupied names are the ones the
    /// retired clock-and-pid temp naming would have picked, so a rewrite that
    /// still used one would truncate a foreign file and rename it into place.
    /// `proposal_status_update_rewrites_record` would pass for either writer;
    /// the foreign files are what tell them apart.
    #[test]
    fn proposal_status_update_never_reuses_another_writers_temp_file() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-status-foreign-temp");
        append_automation_proposal(&paths, &proposal_record("proposal-1"))?;
        let store = paths.automation_proposals_path();

        assert_write_spares_foreign_temp_files(
            |millis| retired_temp_path(&store, millis),
            || update_automation_proposal_status(&paths, "proposal-1", "rejected").map(|_| ()),
        )?;

        let loaded = find_automation_proposal(&paths, "proposal-1")?;
        let _ = fs::remove_dir_all(&root);
        assert_eq!(loaded.status, "rejected");
        Ok(())
    }

    /// The torn-tail repair, the other place a store is replaced wholesale,
    /// spares foreign staged files for the same reason. Each attempt seeds the
    /// torn tail again, since a repair only happens once.
    #[test]
    fn torn_tail_repair_never_reuses_another_writers_temp_file() -> Result<()> {
        let (root, paths) = temp_app_paths("torn-tail-foreign-temp");
        // Creates the store directory the foreign files go in.
        seed_torn_tail_in_every_store(&paths)?;
        let store = paths.automation_events_path();

        assert_write_spares_foreign_temp_files(
            |millis| retired_temp_path(&store, millis),
            || {
                seed_torn_tail_in_every_store(&paths)?;
                append_automation_event(&paths, &event_record("after-repair"))
            },
        )?;

        // Only the events store was appended to; the other two still hold the
        // torn tails the seeding left in them.
        let tags = fs::read_to_string(&store)?
            .lines()
            .map(|line| serde_json::from_str::<AutomationEventRecord>(line).map(|r| r.message))
            .collect::<Result<Vec<_>, _>>()?;
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            tags,
            vec!["seed".to_owned(), "after-repair".to_owned()],
            "the repair must keep the clean prefix, drop the torn tail and add the new record"
        );
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
    // predicate it calls, tested below in isolation with hand-built byte
    // slices. An RLIMIT_FSIZE one byte under the record's full length
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
    // `landed_despite_the_error` itself already has a direct unit test for
    // "nothing landing is not a complete record" — what has no test without
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
                let mut starts = Vec::new();
                while let Some((start, line)) = lines.next_line()? {
                    starts.push(start);
                    got.push(line);
                }
                assert_eq!(got, expected, "contents {contents:?}, chunk {chunk}");
                // Each line's reported start plus its own length must land
                // exactly one byte before the previous (later-in-file) line's
                // start — the byte in between is the `\n` that separated
                // them — except the first line yielded here (the file's
                // *last* line, with nothing after it to separate from),
                // which must instead land exactly on the file's own length.
                let mut expected_next = contents.len() as u64;
                for (&start, line) in starts.iter().zip(&got) {
                    assert_eq!(
                        start + line.len() as u64,
                        expected_next,
                        "contents {contents:?}, chunk {chunk}: wrong start for {line:?}"
                    );
                    expected_next = start.saturating_sub(1);
                }
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

    // A lock-free reader that measured a stale, larger length — because the
    // store was edited out-of-band, not because a concurrent writer shrank
    // it — sees the file end early; that is not an error.
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

    // Nothing in this module's own writers shrinks a store, but an
    // out-of-band edit might, and load_recent_jsonl_records is supposed to
    // retry from the new length rather than fail. Driving that needs the
    // shrink to land strictly between two of ReverseLines's own chunked
    // reads (see the `test_hooks` module doc) — which in turn
    // needs a scan that takes more than one physical read in the first
    // place, hence the pathologically long first record here (same
    // technique as append_leaves_a_torn_line_longer_than_one_scan_chunk_cut_cleanly
    // below, just complete JSON rather than torn).
    #[test]
    fn load_recent_retries_past_a_shrink_mid_scan_and_succeeds() -> Result<()> {
        let (root, paths) = temp_app_paths("scan-retry-after-shrink");
        let seed = format!("{}\n", audit_line("seed"));
        let giant = audit_line(&"x".repeat(200_000));
        seed_audit_store(&paths, &format!("{seed}{giant}"))?;
        let path = paths.audit_events_path();
        let seed_len = seed.len() as u64;
        let hook_path = path.clone();
        test_hooks::set_before_scan_read(
            &path,
            1,
            std::sync::Arc::new(move || {
                fs::OpenOptions::new()
                    .write(true)
                    .open(&hook_path)
                    .and_then(|f| f.set_len(seed_len))
                    .expect("truncate mid-scan");
            }),
        );
        let recent = load_recent_audit_events(&paths, 5);
        fs::remove_dir_all(root).ok();
        assert_eq!(
            audit_messages(&recent?),
            ["seed"],
            "a shrink strictly between two chunked reads must be retried, not failed"
        );
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

    // Unlike the shrink above, a shrink before any chunk has been read yet
    // isn't guarded (there is no prior `pending` to misalign against), so it
    // must not error — but the line it finds, and the offset it reports,
    // still have to match where the content actually is now, not where the
    // `len` passed to `new` (now stale) said it would be.
    #[test]
    fn reverse_lines_tolerates_a_shrink_before_the_first_read() -> Result<()> {
        let (root, _paths) = temp_app_paths("reverse-lines-shrink-first-read");
        fs::create_dir_all(&root)?;
        let path = root.join("lines.jsonl");
        fs::write(&path, "aaaa\nbbbb\ncccc")?;
        let file = fs::File::open(&path)?;
        // As if this had measured the file at its original 14 bytes, then the
        // file shrank to 9 ("aaaa\nbbbb", no trailing "\ncccc") before this
        // scan's first read.
        fs::OpenOptions::new().write(true).open(&path)?.set_len(9)?;
        let mut lines = ReverseLines::new(&path, &file, 14, 4);
        let next = lines.next_line();
        fs::remove_dir_all(root).ok();
        let (start, line) = next?.expect("one line");
        assert_eq!(line, b"bbbb", "must read what is really there now");
        assert_eq!(
            start, 5,
            "offset must match the real content, not the stale length"
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

    #[test]
    fn landed_despite_the_error_tolerates_only_a_missing_trailing_newline() {
        let line = b"{\"a\":1}\n";
        assert!(
            landed_despite_the_error(&line[..line.len() - 1], line),
            "a write that failed one byte short of the trailing newline must \
             still count as landed: the record is complete and valid, and a \
             caller that reports this as a failure and retries would write \
             it twice"
        );
        assert!(
            landed_despite_the_error(line, line),
            "the full line landing must also count, even if write_all \
             somehow still reported an error for it"
        );
        assert!(
            !landed_despite_the_error(&line[..line.len() - 2], line),
            "missing more than just the trailing newline is genuine torn \
             debris, not a complete record"
        );
        assert!(
            !landed_despite_the_error(b"", line),
            "nothing landing at all is not a complete record"
        );
        assert!(
            !landed_despite_the_error(b"{\"a\":2}\n", line),
            "different bytes landing at the same length must not be \
             mistaken for this record having landed"
        );
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

    #[test]
    fn rotated_generations_are_numbered_siblings_of_the_store() {
        let path = PathBuf::from("d").join("audit").join("events.jsonl");
        assert_eq!(jsonl_generation_path(&path, 0), path);
        assert_eq!(
            jsonl_generation_path(&path, 1),
            PathBuf::from("d").join("audit").join("events.jsonl.1")
        );
        assert_eq!(
            jsonl_generation_path(&path, 4),
            PathBuf::from("d").join("audit").join("events.jsonl.4")
        );
    }

    /// Append an audit record tagged `tag` to `path` under `rotation`.
    fn append_rotating(path: &Path, tag: &str, rotation: JsonlRotation) -> Result<()> {
        append_jsonl_record(
            path,
            &audit_record(tag),
            Some(rotation),
            "audit event record",
        )
    }

    /// The tags in every generation of the audit-shaped store at `path`,
    /// oldest first, read straight from the files.
    fn tags_in_every_generation(path: &Path, generations: usize) -> Result<Vec<String>> {
        let mut tags = Vec::new();
        for generation in (0..=generations).rev() {
            let generation_path = jsonl_generation_path(path, generation);
            if !generation_path.exists() {
                continue;
            }
            for line in fs::read_to_string(&generation_path)?.lines() {
                tags.push(serde_json::from_str::<AuditEventRecord>(line)?.message);
            }
        }
        Ok(tags)
    }

    fn recent_tags(path: &Path, generations: usize, limit: usize) -> Result<Vec<String>> {
        Ok(
            load_recent_jsonl_records::<AuditEventRecord>(path, generations, limit, "audit")?
                .into_iter()
                .map(|record| record.message)
                .collect(),
        )
    }

    #[test]
    fn rotation_keeps_the_live_file_and_the_configured_generations() -> Result<()> {
        let (root, paths) = temp_app_paths("rotation-generations");
        paths.ensure()?;
        let path = paths.audit_events_path();
        // One record per file: every append after the first rotates.
        let rotation = JsonlRotation {
            max_bytes: 1,
            generations: 2,
        };
        for tag in ["a", "b", "c", "d"] {
            append_rotating(&path, tag, rotation)?;
        }
        let on_disk = tags_in_every_generation(&path, 3);
        let third = jsonl_generation_path(&path, 3).exists();
        let lock = jsonl_lock_path(&path).is_file();
        let all = recent_tags(&path, 2, 10);
        let two = recent_tags(&path, 2, 2);
        fs::remove_dir_all(root).ok();
        // `a` was dropped by the rotation that would have made it `.3`.
        assert_eq!(on_disk?, ["b", "c", "d"]);
        assert!(!third, "only `generations` rotated files are kept");
        assert!(lock, "rotation leaves the store's lock file alone");
        assert_eq!(all?, ["b", "c", "d"]);
        assert_eq!(two?, ["c", "d"]);
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

    // The rotation tests above all drive `append_jsonl_record` and
    // `load_recent_jsonl_records` directly with a custom `JsonlRotation`,
    // which proves the mechanism works but never exercises the wiring that
    // connects it to the real stores: `append_automation_event` and
    // `append_audit_event` passing `Some(EVENT_LOG_ROTATION)`, and
    // `load_recent_automation_events`/`load_recent_audit_events` passing
    // `EVENT_LOG_ROTATED_GENERATIONS`. Flip either side back to `None`/`0`
    // and every test above still passes. These two tests close that gap by
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

    // Right after a rollover the live file holds fewer records than a reader
    // asks for; the rest must come from the rotated generations, in order.
    #[test]
    fn load_recent_spans_a_rollover() -> Result<()> {
        let (root, paths) = temp_app_paths("rotation-spans");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let line_len = audit_line("00").len() as u64 + 1;
        // Rotate once a file holds five records.
        let rotation = JsonlRotation {
            max_bytes: 5 * line_len,
            generations: 4,
        };
        for n in 0..12 {
            append_rotating(&path, &format!("{n:02}"), rotation)?;
        }
        let live = fs::read_to_string(&path)?.lines().count();
        let recent = recent_tags(&path, rotation.generations, 4);
        let spanning = recent_tags(&path, rotation.generations, 9);
        let everything = recent_tags(&path, rotation.generations, 100);
        let live_only = recent_tags(&path, 0, 100);
        fs::remove_dir_all(root).ok();
        assert_eq!(live, 2, "12 records rotate after the 5th and 10th");
        assert_eq!(recent?, ["08", "09", "10", "11"]);
        assert_eq!(
            spanning?,
            ["03", "04", "05", "06", "07", "08", "09", "10", "11"]
        );
        let expected: Vec<String> = (0..12).map(|n| format!("{n:02}")).collect();
        assert_eq!(everything?, expected);
        assert_eq!(live_only?, ["10", "11"]);
        Ok(())
    }

    // A reader can see the store mid-rotation: no live file yet, or a gap
    // where `.1` was shifted to `.2` but the live file not yet renamed.
    #[test]
    fn load_recent_reads_a_store_caught_mid_rotation_in_order() -> Result<()> {
        let (root, paths) = temp_app_paths("rotation-mid");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let write = |generation: usize, tags: &[&str]| -> Result<()> {
            let text: String = tags.iter().map(|tag| audit_line(tag) + "\n").collect();
            fs::write(jsonl_generation_path(&path, generation), text)?;
            Ok(())
        };
        write(1, &["c", "d"])?;
        write(2, &["a", "b"])?;
        let no_live = recent_tags(&path, 4, 3);
        write(0, &["e"])?;
        fs::remove_file(jsonl_generation_path(&path, 1))?;
        write(3, &["a"])?;
        write(2, &["b", "c", "d"])?;
        let gap = recent_tags(&path, 4, 10);
        fs::remove_dir_all(root).ok();
        assert_eq!(no_live?, ["b", "c", "d"]);
        assert_eq!(gap?, ["a", "b", "c", "d", "e"]);
        Ok(())
    }

    // Rotation is best effort: a rename that fails must not cost the record,
    // nor rename the live file over a `.1` that was never shifted away.
    #[test]
    fn a_rotation_that_fails_still_appends_the_record() -> Result<()> {
        let (root, paths) = temp_app_paths("rotation-fails");
        paths.ensure()?;
        let path = paths.audit_events_path();
        fs::write(&path, audit_line("a") + "\n")?;
        fs::write(jsonl_generation_path(&path, 1), audit_line("old") + "\n")?;
        // `.1` cannot be renamed over a non-empty directory.
        let blocker = jsonl_generation_path(&path, 2);
        fs::create_dir_all(&blocker)?;
        fs::write(blocker.join("keep"), "x")?;
        let rotation = JsonlRotation {
            max_bytes: 1,
            generations: 2,
        };
        let appended = append_rotating(&path, "b", rotation);
        let live = fs::read_to_string(&path);
        let first = fs::read_to_string(jsonl_generation_path(&path, 1));
        let recent = recent_tags(&path, 2, 10);
        fs::remove_dir_all(root).ok();
        appended?;
        assert_eq!(live?, format!("{}\n{}\n", audit_line("a"), audit_line("b")));
        assert_eq!(first?, audit_line("old") + "\n");
        assert_eq!(recent?, ["old", "a", "b"]);
        Ok(())
    }

    // `rotation_failure_warning` is unit-tested on its own; this pins that
    // `append_jsonl_record` actually prints it, once for a store that keeps
    // failing to rotate. Run in a child process to read its stderr, which the
    // test harness would otherwise swallow.
    #[test]
    fn a_failed_rotation_prints_its_warning_once_to_stderr() -> Result<()> {
        const WARNING_CHILD_ENV: &str = "ROCM_CORE_TEST_ROTATION_WARNING_CHILD";

        if let Some(path) = std::env::var_os(WARNING_CHILD_ENV) {
            let rotation = JsonlRotation {
                max_bytes: 1,
                generations: 2,
            };
            append_rotating(Path::new(&path), "b", rotation)?;
            append_rotating(Path::new(&path), "c", rotation)?;
            return Ok(());
        }

        let (root, paths) = temp_app_paths("rotation-warning");
        paths.ensure()?;
        let path = paths.audit_events_path();
        fs::write(&path, audit_line("a") + "\n")?;
        fs::write(jsonl_generation_path(&path, 1), audit_line("old") + "\n")?;
        // `.1` cannot be renamed over a non-empty directory.
        let blocker = jsonl_generation_path(&path, 2);
        fs::create_dir_all(&blocker)?;
        fs::write(blocker.join("keep"), "x")?;
        let exe = std::env::current_exe().context("current test executable")?;
        let output = std::process::Command::new(&exe)
            .args([
                "automation::tests::a_failed_rotation_prints_its_warning_once_to_stderr",
                "--exact",
                "--quiet",
                "--nocapture",
            ])
            .env(WARNING_CHILD_ENV, &path)
            .output()
            .context("spawn rotation-warning child")?;
        fs::remove_dir_all(root).ok();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child process failed: {:?}\nstdout: {}\nstderr: {stderr}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
        );
        assert_eq!(
            stderr.matches("warning: failed to rotate").count(),
            1,
            "two appends to a store stuck at its cap must print one warning: {stderr}"
        );
        assert!(
            stderr.contains("events.jsonl"),
            "the warning names the store: {stderr}"
        );
        Ok(())
    }

    // A rotation that shifted `.1` away but then failed to rename the live
    // file leaves a gap at `.1`. The next rotation must only fill it: shifting
    // the older generations again would drop one more each time a rotation
    // keeps failing that way (e.g. on Windows, while another program holds the
    // live file open without delete sharing).
    #[test]
    fn a_rotation_after_one_left_part_way_drops_no_generation() -> Result<()> {
        let (root, paths) = temp_app_paths("rotation-part-way");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let write = |generation: usize, tag: &str| -> Result<()> {
            fs::write(
                jsonl_generation_path(&path, generation),
                audit_line(tag) + "\n",
            )?;
            Ok(())
        };
        // `.1` = a was shifted to `.2`; the live file `e` was not renamed.
        write(0, "e")?;
        write(2, "a")?;
        write(3, "b")?;
        write(4, "c")?;
        let rotation = JsonlRotation {
            max_bytes: 1,
            generations: 4,
        };
        append_rotating(&path, "f", rotation)?;
        let on_disk = tags_in_every_generation(&path, 5);
        let fifth = jsonl_generation_path(&path, 5).exists();
        fs::remove_dir_all(root).ok();
        assert_eq!(on_disk?, ["c", "b", "a", "e", "f"]);
        assert!(!fifth, "rotation never writes past `generations`");
        Ok(())
    }

    fn assert_holds_exactly_in_writer_order(tags: &[String], expected: &[String]) {
        let mut sorted = tags.to_vec();
        sorted.sort();
        let mut expected = expected.to_vec();
        expected.sort();
        assert_eq!(
            sorted, expected,
            "rotation lost or duplicated records under concurrent writers"
        );
        // Each writer appends in order, so its records must read back in order
        // across the generations, oldest first.
        let mut last_seen = std::collections::HashMap::new();
        for tag in tags {
            let (writer, n) = tag.rsplit_once('-').expect("tag is <writer>-<n>");
            let n: usize = n.parse().expect("numeric sequence");
            if let Some(previous) = last_seen.insert(writer.to_owned(), n) {
                assert!(
                    previous < n,
                    "{writer}'s records are out of order: {tags:?}"
                );
            }
        }
    }

    /// Small enough that the concurrency tests below rotate every ~14
    /// records, with enough generations that none is dropped.
    const TEST_ROTATION: JsonlRotation = JsonlRotation {
        max_bytes: 2048,
        generations: 64,
    };

    // Rotation renames the live file, so it must run under the store lock:
    // an append that had already opened the file being renamed would land in
    // the rotated copy (or be lost), and two writers rotating at once would
    // shift the generations twice.
    #[test]
    fn rotation_under_concurrent_threads_loses_no_records() -> Result<()> {
        const WRITERS: usize = 8;
        const APPENDS_PER_WRITER: usize = 25;
        let (root, paths) = temp_app_paths("rotation-threads");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let barrier = std::sync::Barrier::new(WRITERS);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WRITERS)
                .map(|writer| {
                    let (path, barrier) = (&path, &barrier);
                    scope.spawn(move || -> Result<()> {
                        barrier.wait();
                        for n in 0..APPENDS_PER_WRITER {
                            append_rotating(path, &format!("t{writer}-{n}"), TEST_ROTATION)?;
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
        let tags = tags_in_every_generation(&path, TEST_ROTATION.generations);
        let rotated = jsonl_generation_path(&path, 2).exists();
        fs::remove_dir_all(root).ok();
        assert!(rotated, "the test must actually rotate more than once");
        assert_holds_exactly_in_writer_order(&tags?, &expected);
        Ok(())
    }

    /// Env var that turns a re-exec of this test binary into one writer of
    /// [`rotation_under_concurrent_processes_loses_no_records`]: its value is
    /// `<store path>|<writer index>`.
    const CHILD_ROTATING_WRITER_ENV: &str = "ROCM_CORE_TEST_JSONL_CHILD_ROTATING_WRITER";

    #[test]
    fn rotation_under_concurrent_processes_loses_no_records() -> Result<()> {
        const WRITERS: usize = 6;
        const APPENDS_PER_WRITER: usize = 20;
        if let Some(spec) = std::env::var_os(CHILD_ROTATING_WRITER_ENV) {
            let spec = spec.to_string_lossy().into_owned();
            let (path, writer) = spec.split_once('|').context("malformed writer spec")?;
            for n in 0..APPENDS_PER_WRITER {
                append_rotating(Path::new(path), &format!("p{writer}-{n}"), TEST_ROTATION)?;
            }
            return Ok(());
        }

        let (root, paths) = temp_app_paths("rotation-processes");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let exe = std::env::current_exe().context("current test executable")?;
        let children = (0..WRITERS)
            .map(|writer| {
                std::process::Command::new(&exe)
                    .args([
                        "automation::tests::rotation_under_concurrent_processes_loses_no_records",
                        "--exact",
                        "--quiet",
                    ])
                    .env(
                        CHILD_ROTATING_WRITER_ENV,
                        format!("{}|{writer}", path.display()),
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
            .flat_map(|writer| (0..APPENDS_PER_WRITER).map(move |n| format!("p{writer}-{n}")))
            .collect();
        let tags = tags_in_every_generation(&path, TEST_ROTATION.generations);
        let rotated = jsonl_generation_path(&path, 2).exists();
        fs::remove_dir_all(root).ok();
        assert!(rotated, "the test must actually rotate more than once");
        assert_holds_exactly_in_writer_order(&tags?, &expected);
        Ok(())
    }

    // Readers take no lock, so a rotation can rename the files they are
    // opening. One sequential writer makes every correct read a run of
    // consecutive numbers: a file read twice or a generation skipped breaks
    // the run.
    #[test]
    fn reads_racing_rotations_see_each_record_once_and_in_order() -> Result<()> {
        const APPENDS: usize = 4000;
        const LIMIT: usize = 20;
        let rotation = JsonlRotation {
            max_bytes: 2048,
            generations: 3,
        };
        let (root, paths) = temp_app_paths("rotation-read-race");
        paths.ensure()?;
        let path = paths.audit_events_path();
        let done = std::sync::atomic::AtomicBool::new(false);
        let outcome = std::thread::scope(|scope| {
            let reader = scope.spawn(|| -> Result<usize> {
                let mut reads = 0;
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    let numbers: Vec<usize> = recent_tags(&path, rotation.generations, LIMIT)?
                        .iter()
                        .map(|tag| tag.parse().expect("numeric tag"))
                        .collect();
                    if numbers.windows(2).any(|pair| pair[1] != pair[0] + 1) {
                        bail!("read {numbers:?}: a record was read twice or skipped");
                    }
                    // Three full generations always hold more than LIMIT.
                    if numbers.last().is_some_and(|&last| last >= 100) && numbers.len() != LIMIT {
                        bail!("read {numbers:?}: a generation was skipped");
                    }
                    reads += 1;
                }
                Ok(reads)
            });
            let written =
                (0..APPENDS).try_for_each(|n| append_rotating(&path, &n.to_string(), rotation));
            done.store(true, std::sync::atomic::Ordering::Release);
            let reads = reader.join().expect("reader panicked");
            written.and(reads)
        });
        fs::remove_dir_all(root).ok();
        assert!(outcome? > 0, "the reader must have run");
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
