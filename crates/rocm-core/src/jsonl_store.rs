// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The JSONL store engine behind the automation stores.
//!
//! Logically append-only newline-delimited JSON files that the `rocm` CLI and
//! `rocmd` write concurrently, so every mutation of a store holds its sibling
//! `.lock` file (see [`jsonl_lock_path`]). Appending a record is, in the
//! common case, one append-mode write of one line — its cost does not grow
//! with the store. A torn tail — crash debris, or left by a write that failed
//! partway — is cut by the next append rather than truncated in place: that
//! append writes the store's clean prefix plus its own new record as a single
//! rewrite via `write_atomic`, so a lock-free reader never observes a file
//! shrunk mid-repair. That rewrite costs O(store) rather than O(record), the
//! same as a proposal status update, which rewrites the whole store via
//! `write_atomic` too. Only when that rewrite itself fails (a full disk, or on
//! Windows another program holding the store open without delete sharing) does
//! the append cut the debris in place instead, so a store cannot be wedged by
//! a failed repair; otherwise nothing in this module ever shrinks a store in
//! place. Readers take no lock, and an unterminated last line that is not
//! valid JSON — an append caught mid-write — is skipped rather than reported
//! as corruption.
//!
//! A store can additionally be capped by size-based rotation (see
//! [`JsonlRotation`]): the append that finds the live file at the cap renames
//! it whole to `<store>.1` first, under the same lock — before any torn tail
//! in it would otherwise be repaired, so an at-cap store is never paid for
//! with a `write_atomic` rewrite only to be rotated away immediately after —
//! and the recent-record readers span the rotated generations.
//!
//! The engine is generic over the record type. Which stores exist, their
//! record types, their rotation policy and their record-level accessors
//! belong to `automation.rs`.

use anyhow::{Context, Result, bail};
use same_file::Handle;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::BTreeSet;
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use crate::FileLock;

/// Lock file serializing every mutation of the JSONL store at `path` — a
/// sibling named `<store file name>.lock` (e.g. `events.jsonl.lock`).
///
/// Appends and the proposal-status rewrite both take it. The proposal
/// rewrite needs it for its read-modify-write; an append needs it both to
/// serialize against other appends and because it can itself replace or
/// rename the live file out from under a concurrent writer — a torn tail is
/// repaired by renaming a new file over the store, or by truncating it when
/// that fails (see `TailState::Torn`), and a rotating events or audit store is
/// renamed aside to `<store>.1` (see `rotate_jsonl_generations`) — and an
/// append that had already opened the old file would otherwise land in the
/// replaced or rotated-away copy and vanish.
pub fn jsonl_lock_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    path.with_file_name(name)
}

/// How many bytes the backward scans below read per step.
const JSONL_SCAN_CHUNK: u64 = 64 * 1024;

/// Size-based rotation policy for a JSONL store.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JsonlRotation {
    /// Rotate before an append once the live file is at least this long.
    pub(crate) max_bytes: u64,
    /// Rotated generations to keep besides the live file; at least 1.
    pub(crate) generations: usize,
}

/// Path of generation `generation` of the store at `path`: the live file for
/// `0`, else `<store file name>.<generation>` beside it (`events.jsonl.1`).
pub(crate) fn jsonl_generation_path(path: &Path, generation: usize) -> PathBuf {
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
/// This function truncates the live file only as the last resort when that
/// replacement itself fails (see [`repair_torn_tail_and_append`]), so a
/// lock-free reader can observe it shrink only then, never on the normal path.
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
pub(crate) fn append_jsonl_record<T: Serialize>(
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
        TailState::Torn(tail_start) => repair_torn_tail_and_append(
            path,
            file,
            tail_start,
            &line,
            |path, content| crate::write_atomic(path, content, "jsonl"),
            truncate_jsonl_in_place,
        )?,
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

/// Discard the torn tail of the store at `path`, which starts at `tail_start`,
/// and land `line` in its place. The caller must hold the store's
/// [`jsonl_lock_path`] lock; `file` is its handle on the pre-repair store.
///
/// The repair publishes the clean prefix plus `line` as one `replace`, so a
/// lock-free reader only ever sees the pre-repair length or the post-append
/// length (see [`append_jsonl_record`]). That rewrite can itself fail — no
/// room for the copy on a full disk, or on Windows another program holding
/// the store open without delete sharing — and leaving it there would wedge
/// the store: every later append would hit the same debris and the same
/// failure. Only then does this fall back to `truncate`, cutting the debris
/// in place and appending `line` as a plain append. A truncate frees space
/// rather than needing it and renames nothing, so it can succeed where
/// `replace` cannot. Like a repair by `replace`, the fallback prints nothing.
///
/// The fallback reopens, on this already-failing path only, the shrink a
/// lock-free reader can observe, and the reader's guards narrow that window
/// without closing it: a scan that comes back short is refused or retried
/// (see [`ReverseLines`] and `load_recent_jsonl_records_from`), but one that
/// already read part of a debris line longer than a scan chunk and then finds
/// the file regrown past the offset the scan has reached (not its old length:
/// the check compares each read with that offset, which only shrinks) reads on
/// undetected, and can splice the two reads into a record nobody wrote. That
/// is the cost of not wedging the store, accepted for a path that needs a
/// failing replace, a debris line that long and a reader mid-scan all at once.
///
/// If `truncate` fails too, the error names both causes and the store is left
/// as it was; readers keep skipping the debris. If the append after a
/// successful truncate fails, the error carries the replace failure too: it is
/// the root cause.
///
/// `replace` and `truncate` are parameters so a test can make either fail on
/// demand.
fn repair_torn_tail_and_append(
    path: &Path,
    file: fs::File,
    tail_start: u64,
    line: &[u8],
    replace: impl FnOnce(&Path, &[u8]) -> Result<()>,
    truncate: impl FnOnce(&Path, u64) -> Result<()>,
) -> Result<()> {
    let mut content = read_range(path, &file, 0, tail_start)?;
    content.extend_from_slice(line);
    // Done with the pre-repair file's handle before replacing it, rather than
    // leaving that to whatever happens to still have `file` open at function
    // return. The rename already succeeds with this handle open regardless —
    // std opens files with `FILE_SHARE_DELETE` by default on Windows, so a
    // rename isn't blocked by it (see `write_atomic`'s doc for the case that
    // *isn't* covered: another process holding the destination open without
    // delete-sharing) — but this states the intent rather than leaning on that
    // default.
    drop(file);
    let replace_error = match replace(path, &content) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    drop(content);
    if let Err(truncate_error) = truncate(path, tail_start) {
        bail!(
            "failed to repair the torn tail of {}: replacing it failed ({replace_error:#}); \
             truncating it failed ({truncate_error:#})",
            path.display()
        );
    }
    open_jsonl_for_append(path)
        .and_then(|file| append_or_tolerate_a_missing_newline(path, &file, line, tail_start))
        .with_context(|| {
            format!(
                "failed to append to {} after cutting its torn tail \
                 (replacing the store had failed: {replace_error:#})",
                path.display()
            )
        })
}

/// Cut the store at `path` down to `len` bytes through a fresh write handle.
///
/// Not the append handle: on Windows an append-only handle lacks
/// `FILE_WRITE_DATA`, which `set_len` needs.
fn truncate_jsonl_in_place(path: &Path, len: u64) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open {} for writing", path.display()))?
        .set_len(len)
        .with_context(|| format!("failed to truncate {}", path.display()))
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
    /// next record lands, by replacing the store rather than truncating it in
    /// place — except as the fallback when that replacement fails; see
    /// [`append_jsonl_record`] and [`repair_torn_tail_and_append`].
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
/// module's own writers do shrinks a store in place — an append only ever
/// grows it, and a torn-tail repair replaces it wholesale via `write_atomic`
/// (see [`inspect_tail`] and [`TailState::Torn`]) — except the repair's
/// fallback, taken only when that replacement fails and only under the store's
/// lock (see [`repair_torn_tail_and_append`]); otherwise a short read here
/// means the store was edited or truncated out-of-band, a case this module
/// does not try to recover from cleanly. Most callers treat what is still
/// there the same as any other unexpectedly-ending tail; [`ReverseLines`] is
/// the one exception, and reports an error instead once it would have to
/// splice such a read onto bytes it already read, rather than risk inventing
/// a line.
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
pub(crate) fn parse_jsonl_records<T: DeserializeOwned>(
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
pub(crate) fn load_recent_jsonl_records<T: DeserializeOwned>(
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
/// skip a generation. A torn-tail repair (see [`TailState::Torn`]) normally
/// also replaces the live file, by renaming a new one over it — not a rotation,
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
/// in place except the torn-tail repair's fallback, taken only when replacing
/// the store fails (see [`repair_torn_tail_and_append`]). The retry below is
/// for a store edited or truncated out-of-band — by a person — or shrunk by
/// that rare fallback while a scan is in flight: a scan that fails while the
/// file's length moved under it is retried from the new end, so a shrink the
/// scan's reads catch never surfaces as an error. The one shrink they cannot
/// catch is described on [`repair_torn_tail_and_append`].
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
            // `read_range`); no concurrent writer shrinks it in place short of
            // the torn-tail repair's rare fallback. Once `pending` holds any
            // bytes, a short read means they no longer continue from it, so
            // splicing them would invent a line.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use serde::Deserialize;

    /// The record these tests store. The engine is generic over its record
    /// type, so its tests carry their own rather than borrow one of its users':
    /// `automation.rs` covers the stores it builds on this engine.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Entry {
        tag: String,
        /// Padding, so a line is about as long as a real store's. The
        /// rotation tests size their byte caps, and the torn-tail tests cut
        /// their debris, against that length.
        detail: String,
    }

    fn entry(tag: &str) -> Entry {
        Entry {
            tag: tag.to_owned(),
            detail: "d".repeat(110),
        }
    }

    /// One serialized record, without its newline.
    fn entry_line(tag: &str) -> String {
        serde_json::to_string(&entry(tag)).expect("serialize entry")
    }

    fn entry_tags(entries: &[Entry]) -> Vec<&str> {
        entries.iter().map(|entry| entry.tag.as_str()).collect()
    }

    /// A throwaway root, and the path of a store inside it, whose directory
    /// exists.
    fn temp_store(name: &str) -> Result<(PathBuf, PathBuf)> {
        let (root, _paths) = temp_app_paths(name);
        let path = root.join("store").join("events.jsonl");
        fs::create_dir_all(path.parent().context("a store has a directory")?)?;
        Ok((root, path))
    }

    /// Write `contents` as the store, as an out-of-band edit or a killed
    /// writer would have left it.
    fn seed_store(path: &Path, contents: &str) -> Result<()> {
        fs::write(path, contents)?;
        Ok(())
    }

    /// Append the record tagged `tag` to `path`, with no rotation.
    fn append_entry(path: &Path, tag: &str) -> Result<()> {
        append_jsonl_record(path, &entry(tag), None, "entry record")
    }

    /// The most recent `limit` records of the store at `path`, reading up to
    /// the generations a real store keeps.
    fn load_recent_entries(path: &Path, limit: usize) -> Result<Vec<Entry>> {
        load_recent_jsonl_records(path, 4, limit, "entry")
    }

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

    // A lock-free reader that measured a stale, larger length — because the
    // store was edited out-of-band, not because a concurrent writer shrank
    // it — sees the file end early; that is not an error.
    #[test]
    fn a_scan_that_finds_the_file_shorter_than_it_was_still_succeeds() -> Result<()> {
        let (root, path) = temp_store("scan-shrunk")?;
        seed_store(
            &path,
            &format!("{}\n{}\n", entry_line("a"), entry_line("b")),
        )?;
        let file = fs::File::open(&path)?;
        let stale_len = file.metadata()?.len() + 40;
        let recent = scan_recent_jsonl_records::<Entry>(&path, &file, stale_len, 5, "entry");
        fs::remove_dir_all(root).ok();
        assert_eq!(entry_tags(&recent?), ["a", "b"]);
        Ok(())
    }

    // Nothing in this module's own writers shrinks a store (bar the torn-tail
    // repair's rare fallback), but an out-of-band edit might, and
    // load_recent_jsonl_records is supposed to retry from the new length
    // rather than fail. Driving that needs the shrink to land strictly
    // between two of ReverseLines's own chunked reads (see the `test_hooks`
    // module doc) — which in turn needs a scan that takes more than one
    // physical read in the first place, hence the pathologically long first
    // record here (same technique as
    // append_leaves_a_torn_line_longer_than_one_scan_chunk_cut_cleanly in
    // `automation.rs`, just complete JSON rather than torn).
    #[test]
    fn load_recent_retries_past_a_shrink_mid_scan_and_succeeds() -> Result<()> {
        let (root, path) = temp_store("scan-retry-after-shrink")?;
        let seed = format!("{}\n", entry_line("seed"));
        let giant = entry_line(&"x".repeat(200_000));
        seed_store(&path, &format!("{seed}{giant}"))?;
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
        let recent = load_recent_entries(&path, 5);
        fs::remove_dir_all(root).ok();
        assert_eq!(
            entry_tags(&recent?),
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

    /// The store's file identity, as a string so one assertion covers the
    /// inode on Unix and the volume serial plus file index on Windows.
    #[cfg(unix)]
    fn file_identity(path: &Path) -> Result<String> {
        use std::os::unix::fs::MetadataExt;
        Ok(fs::metadata(path)?.ino().to_string())
    }

    #[cfg(windows)]
    fn file_identity(path: &Path) -> Result<String> {
        Ok(format!(
            "{:?}",
            crate::test_support::windows_file_identity(path)?
        ))
    }

    /// Run the torn-tail repair on the seeded store at `path` with the replace
    /// and truncate steps supplied by the test, appending the record `b`.
    fn repair_store_with(
        path: &Path,
        replace: impl FnOnce(&Path, &[u8]) -> Result<()>,
        truncate: impl FnOnce(&Path, u64) -> Result<()>,
    ) -> Result<()> {
        let file = open_jsonl_for_append(path)?;
        let TailState::Torn(tail_start) = inspect_tail(path, &file)? else {
            bail!("the seeded store has no torn tail");
        };
        let mut line = entry_line("b").into_bytes();
        line.push(b'\n');
        repair_torn_tail_and_append(path, file, tail_start, &line, replace, truncate)
    }

    // The fallback the whole-store rewrite lacked: when it cannot complete
    // (a full disk, a sharing violation), the debris is cut in place instead
    // of wedging every later append on it. The same file identity afterwards
    // is what tells that apart from the replace that
    // append_of_a_torn_tail_replaces_rather_than_truncates_the_file_in_place
    // in `automation.rs` pins.
    #[test]
    fn a_torn_tail_is_cut_in_place_when_replacing_the_store_fails() -> Result<()> {
        let (root, path) = temp_store("torn-tail-fallback-truncates")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;
        let before = file_identity(&path)?;
        let mut attempted = Vec::new();

        repair_store_with(
            &path,
            |_, content| {
                attempted = content.to_vec();
                bail!("no space left on device")
            },
            truncate_jsonl_in_place,
        )?;

        let after = file_identity(&path)?;
        let text = fs::read_to_string(&path);
        let recent = load_recent_entries(&path, 5);
        fs::remove_dir_all(root).ok();
        let expected = format!("{}\n{}\n", entry_line("a"), entry_line("b"));
        assert_eq!(
            String::from_utf8(attempted)?,
            expected,
            "the whole-store replace must be tried first, with the full repaired store"
        );
        assert_eq!(text?, expected);
        assert_eq!(before, after, "the fallback must cut the debris in place");
        assert_eq!(entry_tags(&recent?), ["a", "b"]);
        Ok(())
    }

    // The fallback is a last resort: a replace that works must not be
    // followed by a truncate, or the shrink window it closes would be back on
    // every repair.
    #[test]
    fn a_torn_tail_is_not_truncated_when_replacing_the_store_works() -> Result<()> {
        let (root, path) = temp_store("torn-tail-fallback-unused")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;
        let before = file_identity(&path)?;

        repair_store_with(
            &path,
            |path, content| crate::write_atomic(path, content, "jsonl"),
            |_, _| bail!("truncate must not run when the replace succeeded"),
        )?;

        let after = file_identity(&path)?;
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        assert_eq!(text?, format!("{}\n{}\n", entry_line("a"), entry_line("b")));
        assert_ne!(before, after, "a working replace publishes a new file");
        Ok(())
    }

    #[test]
    fn a_torn_tail_that_can_be_neither_replaced_nor_cut_reports_both_causes() -> Result<()> {
        let (root, path) = temp_store("torn-tail-fallback-both-fail")?;
        let torn = &entry_line("torn")[..20];
        let seeded = format!("{}\n{torn}", entry_line("a"));
        seed_store(&path, &seeded)?;

        // Layered like the real steps (`write_atomic` and
        // `truncate_jsonl_in_place` both wrap the OS error in a context), so
        // the assertion fails if either cause is formatted without its chain.
        let result = repair_store_with(
            &path,
            |_, _| {
                Err(anyhow::anyhow!("no space left on device"))
                    .context("failed to replace the store")
            },
            |_, _| {
                Err(anyhow::anyhow!("permission denied")).context("failed to truncate the store")
            },
        );

        let text = fs::read_to_string(&path);
        let recent = load_recent_entries(&path, 5);
        fs::remove_dir_all(root).ok();
        assert_eq!(
            format!("{:#}", result.expect_err("both steps failed")),
            format!(
                "failed to repair the torn tail of {}: replacing it failed (failed to replace \
                 the store: no space left on device); truncating it failed (failed to truncate \
                 the store: permission denied)",
                path.display()
            )
        );
        assert_eq!(
            text?, seeded,
            "a failed repair must leave the store as it was"
        );
        assert_eq!(entry_tags(&recent?), ["a"]);
        Ok(())
    }

    // The replace failure is the root cause, so it must survive into the error
    // when the cut itself works but the append after it does not. The truncate
    // stub swaps the store for a directory, which makes reopening it fail.
    #[test]
    fn a_failed_append_after_the_cut_still_reports_why_the_replace_failed() -> Result<()> {
        let (root, path) = temp_store("torn-tail-fallback-append-fails")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;

        let result = repair_store_with(
            &path,
            |_, _| {
                Err(anyhow::anyhow!("no space left on device"))
                    .context("failed to replace the store")
            },
            |path, _| {
                fs::remove_file(path)?;
                fs::create_dir(path)?;
                Ok(())
            },
        );

        fs::remove_dir_all(root).ok();
        let message = format!("{:#}", result.expect_err("the append after the cut failed"));
        assert!(
            message.starts_with(&format!(
                "failed to append to {} after cutting its torn tail (replacing the store had \
                 failed: failed to replace the store: no space left on device): ",
                path.display()
            )),
            "unexpected error: {message}"
        );
        Ok(())
    }

    // After the fallback the store is clean, so the next append is the plain
    // one-line append: no second repair, no new file.
    #[test]
    fn appends_after_a_fallback_repair_are_plain_appends() -> Result<()> {
        let (root, path) = temp_store("torn-tail-fallback-then-append")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;
        repair_store_with(
            &path,
            |_, _| bail!("no space left on device"),
            truncate_jsonl_in_place,
        )?;
        let repaired = file_identity(&path)?;

        append_entry(&path, "c")?;

        let after = file_identity(&path)?;
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        assert_eq!(
            text?,
            format!(
                "{}\n{}\n{}\n",
                entry_line("a"),
                entry_line("b"),
                entry_line("c")
            )
        );
        assert_eq!(repaired, after);
        Ok(())
    }

    // The real trigger rather than an injected one: a store directory that
    // takes no new files, so the temp copy of the whole-store replace cannot
    // be created, while the store file itself can still be written.
    //
    // This is the only test that drives the production `replace`/`truncate`
    // wiring in `append_jsonl_record` rather than injected steps, and it passes
    // without testing anything for a user the permission bits do not bind
    // (root). CI runs it as a regular user.
    #[cfg(unix)]
    #[test]
    fn append_repairs_a_torn_tail_in_a_directory_that_takes_no_new_files() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let (root, path) = temp_store("torn-tail-read-only-dir")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;
        let dir = path.parent().context("the store has a directory")?;
        // Create the lock file now: the append reuses an existing one, but
        // could not create it inside the read-only directory.
        drop(FileLock::acquire(jsonl_lock_path(&path))?);
        let before = file_identity(&path)?;

        fs::set_permissions(dir, fs::Permissions::from_mode(0o555))?;
        let read_only_holds = fs::File::create(dir.join("probe")).is_err();
        let result = read_only_holds.then(|| append_entry(&path, "b"));
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755))?;

        let after = file_identity(&path)?;
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        let Some(result) = result else {
            // Running as a user the permission bits do not bind (root).
            eprintln!("skipped: a read-only directory still takes new files for this user");
            return Ok(());
        };
        result?;
        assert_eq!(text?, format!("{}\n{}\n", entry_line("a"), entry_line("b")));
        assert_eq!(before, after);
        Ok(())
    }

    // Windows' real trigger: another program holds the store open without
    // delete sharing (a virus scanner or indexer, say), so the rename over it
    // is refused while the store itself can still be written.
    #[cfg(windows)]
    #[test]
    fn append_repairs_a_torn_tail_while_another_handle_blocks_replacing_the_store() -> Result<()> {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_SHARE_READ_WRITE: u32 = 0x1 | 0x2;
        let (root, path) = temp_store("torn-tail-share-violation")?;
        let torn = &entry_line("torn")[..20];
        seed_store(&path, &format!("{}\n{torn}", entry_line("a")))?;
        let before = file_identity(&path)?;
        let blocker = fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ_WRITE)
            .open(&path)?;

        let result = append_entry(&path, "b");

        drop(blocker);
        let after = file_identity(&path)?;
        let text = fs::read_to_string(&path);
        fs::remove_dir_all(root).ok();
        result?;
        assert_eq!(text?, format!("{}\n{}\n", entry_line("a"), entry_line("b")));
        assert_eq!(before, after);
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
        let events = PathBuf::from("d").join("first").join("events.jsonl");
        let proposals = PathBuf::from("d").join("second").join("proposals.jsonl");
        assert_eq!(
            jsonl_lock_path(&events),
            PathBuf::from("d").join("first").join("events.jsonl.lock")
        );
        assert_eq!(
            jsonl_lock_path(&proposals),
            PathBuf::from("d")
                .join("second")
                .join("proposals.jsonl.lock")
        );
    }

    #[test]
    fn rotated_generations_are_numbered_siblings_of_the_store() {
        let path = PathBuf::from("d").join("store").join("events.jsonl");
        assert_eq!(jsonl_generation_path(&path, 0), path);
        assert_eq!(
            jsonl_generation_path(&path, 1),
            PathBuf::from("d").join("store").join("events.jsonl.1")
        );
        assert_eq!(
            jsonl_generation_path(&path, 4),
            PathBuf::from("d").join("store").join("events.jsonl.4")
        );
    }

    /// Append the record tagged `tag` to `path` under `rotation`.
    fn append_rotating(path: &Path, tag: &str, rotation: JsonlRotation) -> Result<()> {
        append_jsonl_record(path, &entry(tag), Some(rotation), "entry record")
    }

    /// The tags in every generation of the store at `path`, oldest first, read
    /// straight from the files.
    fn tags_in_every_generation(path: &Path, generations: usize) -> Result<Vec<String>> {
        let mut tags = Vec::new();
        for generation in (0..=generations).rev() {
            let generation_path = jsonl_generation_path(path, generation);
            if !generation_path.exists() {
                continue;
            }
            for line in fs::read_to_string(&generation_path)?.lines() {
                tags.push(serde_json::from_str::<Entry>(line)?.tag);
            }
        }
        Ok(tags)
    }

    fn recent_tags(path: &Path, generations: usize, limit: usize) -> Result<Vec<String>> {
        Ok(
            load_recent_jsonl_records::<Entry>(path, generations, limit, "entry")?
                .into_iter()
                .map(|record| record.tag)
                .collect(),
        )
    }

    #[test]
    fn rotation_keeps_the_live_file_and_the_configured_generations() -> Result<()> {
        let (root, path) = temp_store("rotation-generations")?;
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

    // Right after a rollover the live file holds fewer records than a reader
    // asks for; the rest must come from the rotated generations, in order.
    #[test]
    fn load_recent_spans_a_rollover() -> Result<()> {
        let (root, path) = temp_store("rotation-spans")?;
        let line_len = entry_line("00").len() as u64 + 1;
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
        let (root, path) = temp_store("rotation-mid")?;
        let write = |generation: usize, tags: &[&str]| -> Result<()> {
            let text: String = tags.iter().map(|tag| entry_line(tag) + "\n").collect();
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
        let (root, path) = temp_store("rotation-fails")?;
        fs::write(&path, entry_line("a") + "\n")?;
        fs::write(jsonl_generation_path(&path, 1), entry_line("old") + "\n")?;
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
        assert_eq!(live?, format!("{}\n{}\n", entry_line("a"), entry_line("b")));
        assert_eq!(first?, entry_line("old") + "\n");
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

        let (root, path) = temp_store("rotation-warning")?;
        fs::write(&path, entry_line("a") + "\n")?;
        fs::write(jsonl_generation_path(&path, 1), entry_line("old") + "\n")?;
        // `.1` cannot be renamed over a non-empty directory.
        let blocker = jsonl_generation_path(&path, 2);
        fs::create_dir_all(&blocker)?;
        fs::write(blocker.join("keep"), "x")?;
        let exe = std::env::current_exe().context("current test executable")?;
        let output = std::process::Command::new(&exe)
            .args([
                "jsonl_store::tests::a_failed_rotation_prints_its_warning_once_to_stderr",
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
        let (root, path) = temp_store("rotation-part-way")?;
        let write = |generation: usize, tag: &str| -> Result<()> {
            fs::write(
                jsonl_generation_path(&path, generation),
                entry_line(tag) + "\n",
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
        let (root, path) = temp_store("rotation-threads")?;
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

        let (root, path) = temp_store("rotation-processes")?;
        let exe = std::env::current_exe().context("current test executable")?;
        let children = (0..WRITERS)
            .map(|writer| {
                std::process::Command::new(&exe)
                    .args([
                        "jsonl_store::tests::rotation_under_concurrent_processes_loses_no_records",
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
        let (root, path) = temp_store("rotation-read-race")?;
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
}
