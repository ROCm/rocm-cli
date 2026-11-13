// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Atomic file replacement, shared by the `rocm` and `rocmd` apps, `rocm-core`'s
//! own config, automation and service-record writers, and the Lemonade engine.
//!
//! [`write_file_atomically`] stages the bytes in a fresh sibling temp file and
//! then publishes it over the destination in one step, so a reader sees either
//! the previous complete file or the new complete file, never a partial one,
//! and the destination is not removed before the new file takes its place.
//!
//! The temp file is named `{file name}.tmp-{pid}-{millis}-{attempt}` and
//! reserved with `create_new`, so two writers — threads of one process or
//! separate processes — can never write through the same temp file, even when
//! their pid and millisecond collide; a collision just moves on to the next
//! attempt. A failed write or publish removes its temp file, with the one
//! exception described below.
//!
//! On Windows the publish tries `rename` first. `std::fs::rename` replaces an
//! existing file in one step, so a lock-free reader (`RocmCliConfig::load`, the
//! automation stores) finds the old or the new file and never an absent one.
//! Only when that rename fails and the destination exists, which is the case of
//! another process holding it open without delete sharing, does the publish
//! fall back to `ReplaceFileW`. The concurrency tests below run on every
//! platform and publish by `rename`; two `ReplaceFileW` publishes of one file in
//! flight at once are not exercised by any test.
//!
//! A publish by `rename` leaves the destination with the staged file's
//! attributes and the ACL its directory hands down, where `ReplaceFileW` carries
//! over those of the file it replaces. No caller of [`write_file_atomically`]
//! sets a hidden or read-only flag or a custom ACL on a file it writes, so
//! nothing is lost today. A file that must keep such settings across a rewrite
//! should not rely on this writer to carry them over.
//!
//! [`write_owner_only_file_atomically`] is the variant for secrets. It stages
//! the bytes in a file only the current user can open (see
//! [`crate::owner_only_file`]) instead of copying the old file's permissions,
//! and publishes by `rename` alone on every platform, never `ReplaceFileW`,
//! because `ReplaceFileW` would carry the old file's ACL — possibly a loose
//! one — over to the new contents. A destination another process holds open
//! without delete sharing therefore makes it fail rather than fall back. It
//! does not resolve a symbolic link at the destination either: the rename
//! replaces the link with the new file instead of writing the secret through
//! it to wherever it points.
//!
//! The fallback is the one path that can leave the destination absent:
//! Microsoft documents that `ReplaceFileW`, called without a backup name, can
//! fail after it has already moved the old file off the destination name
//! (`ERROR_UNABLE_TO_MOVE_REPLACEMENT` and `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2`).
//! The publish then retries once as a plain rename. If that fails too, the
//! temp file is kept rather than removed, since it holds the only copy of the
//! new contents, and the error is a [`KeptReplacement`] naming it. The
//! decision rests on the error code the replace returned, not on whether the
//! destination existed beforehand, so a second writer that created the
//! destination in between cannot turn it into a deletion.
//!
//! Durable as well as atomic: the temp file is flushed to disk before it is
//! published, so the new name never points at data still in the cache. On
//! Unix the directory is flushed after the rename too, so a power loss leaves
//! the old file or the new one rather than an empty one. Windows has no
//! directory flush here; there the rename's durability is the filesystem's.
//!
//! Replacing a file with [`write_file_atomically`] must not quietly change what
//! the user set up:
//!
//! - on Unix the new file keeps the old one's permission bits, applied to the
//!   temp file before any byte is written into it (on Windows nothing is
//!   copied; see the note on attributes above);
//! - a destination that is a symbolic link — a dotfiles checkout linking
//!   `config.json`, say — has its *target* replaced through
//!   [`write_file_atomically`], so the link survives. A link loop is an error.
//!
//! This makes each file individually atomic. It does not serialize a
//! read-modify-write across processes: two writers that each read, change and
//! write the same file are still last-writer-wins.

use crate::disk_space;
use crate::unix_time_millis;
use anyhow::{Context, Result, bail};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

/// How many symbolic links [`write_file_atomically`] follows before giving up,
/// matching the usual kernel limit (`MAXSYMLINKS` on Linux).
const MAX_SYMLINK_HOPS: usize = 40;

/// The raw values of Windows' `ERROR_UNABLE_TO_MOVE_REPLACEMENT` (1176) and
/// `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2` (1177), the two codes with which
/// `ReplaceFileW` reports that it moved the old file off the destination name
/// and then failed. Plain numbers so the decision built on them can be tested
/// on every platform; a Windows-only test ties them to `windows-sys`.
#[cfg(any(windows, test))]
const PARTIAL_REPLACE_ERROR_CODES: [i32; 2] = [1176, 1177];

/// Whether a failed `ReplaceFileW` left the destination name empty.
#[cfg(any(windows, test))]
fn is_partial_replace(error: &io::Error) -> bool {
    error
        .raw_os_error()
        .is_some_and(|code| PARTIAL_REPLACE_ERROR_CODES.contains(&code))
}

/// A publish that took the old destination away and could not put the new
/// file in its place: the replace failed part-way, and so did the rename
/// retried after it.
#[derive(Debug)]
struct DestinationRemoved {
    replace: io::Error,
    retry: io::Error,
}

impl fmt::Display for DestinationRemoved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}; retrying as a plain rename also failed: {}",
            self.replace, self.retry
        )
    }
}

impl std::error::Error for DestinationRemoved {}

/// Test seam: the error a publish returns when it took the old destination
/// away and could not put the new file in its place, which is what Windows'
/// partial `ReplaceFileW` followed by a failed rename produces.
///
/// `pub` only so tests in other crates can drive that path through
/// [`write_file_atomically_with_publish`] on any platform; production code
/// never builds one.
#[doc(hidden)]
pub fn destination_removed_error(replace: io::Error, retry: io::Error) -> io::Error {
    io::Error::new(retry.kind(), DestinationRemoved { replace, retry })
}

fn removed_the_destination(error: &io::Error) -> bool {
    error
        .get_ref()
        .is_some_and(<dyn std::error::Error + Send + Sync>::is::<DestinationRemoved>)
}

/// The error of a publish that took the old file away and kept the new
/// contents in the temp file, which then holds the only copy.
///
/// A caller that knows better than to want those contents back — one that has
/// already put the previous file back, say — can downcast to this to find the
/// kept file and remove it.
#[derive(Debug)]
pub struct KeptReplacement {
    destination: PathBuf,
    kept: PathBuf,
    cause: io::Error,
}

impl KeptReplacement {
    /// The path the publish was meant to replace, which is now missing.
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// The temp file holding the new contents.
    pub fn kept(&self) -> &Path {
        &self.kept
    }

    /// Why the publish failed.
    pub const fn cause(&self) -> &io::Error {
        &self.cause
    }
}

impl fmt::Display for KeptReplacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to publish {}, and the replace took the old file away; the new contents \
             are kept in {} - rename that file to {} to restore it",
            self.destination.display(),
            self.kept.display(),
            self.destination.display()
        )
    }
}

impl std::error::Error for KeptReplacement {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// What to do after `ReplaceFileW` failed with `error`: give up when the
/// destination is untouched, and otherwise put the replacement in place with
/// `rename`, which is a plain rename in production and injected by tests.
#[cfg(any(windows, test))]
fn recover_partial_replace<R>(
    error: io::Error,
    replacement: &Path,
    path: &Path,
    rename: R,
) -> io::Result<()>
where
    R: FnOnce(&Path, &Path) -> io::Result<()>,
{
    if !is_partial_replace(&error) {
        return Err(error);
    }
    rename(replacement, path).map_err(|retry| destination_removed_error(error, retry))
}

/// What separates a staged file's name from the name of the file it will
/// replace: `{file name}.tmp-{suffix}`.
const STAGED_NAME_MARKER: &str = ".tmp-";

/// A temp path next to `path`, preserving the full file name so a
/// multi-extension artifact keeps its extensions (`sdk.tar.gz` becomes
/// `sdk.tar.gz.tmp-<suffix>`, where `with_extension` would drop `.gz`).
fn temp_sibling_path(path: &Path, suffix: &OsStr) -> Result<PathBuf> {
    let parent = path.parent().context("file path has no parent directory")?;
    let mut file_name = path
        .file_name()
        .context("file path has no file name")?
        .to_os_string();
    file_name.push(STAGED_NAME_MARKER);
    file_name.push(suffix);
    Ok(parent.join(file_name))
}

/// The name of the file a staged file named `file_name` was going to replace,
/// or `None` when `file_name` is not a staged file's name.
///
/// A writer killed between staging and publishing leaves its staged file
/// behind, and for a secret written with [`write_owner_only_file_atomically`]
/// that file holds the secret. This lets a cleanup that removes the
/// destination recognise those leftovers as belonging to it.
///
/// Only the suffix this module's own writers produce is recognised —
/// `{pid}-{millis}-{attempt}`, three runs of digits — so a file whose name
/// merely contains `.tmp-` is not taken for a staged one. That matters where
/// names are chosen by someone else: the service id `a.endpoint-key.tmp-x`
/// gives a key file named `a.endpoint-key.tmp-x.endpoint-key`, which must not
/// read as a staged copy of service `a`'s key. Names are compared as text, so
/// a staged file whose name is not Unicode is not recognised.
pub fn staged_destination_name(file_name: &str) -> Option<&str> {
    let (destination, suffix) = file_name.rsplit_once(STAGED_NAME_MARKER)?;
    let mut parts = suffix.split('-');
    let well_formed = (0..3).all(|_| {
        parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    }) && parts.next().is_none();
    (!destination.is_empty() && well_formed).then_some(destination)
}

/// Every staged file next to `path` that was going to replace it — what a
/// writer killed before publishing leaves behind. Empty when there are none or
/// the directory cannot be read.
///
/// A writer that is still running has its staged file listed too, so remove
/// these only when the destination itself is being removed. The result is a
/// snapshot taken when the directory is read; it does not exclude a concurrent
/// writer.
pub fn staged_siblings(path: &Path) -> Vec<PathBuf> {
    let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return Vec::new();
    };
    let mut staged: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate
                .file_name()
                .and_then(OsStr::to_str)
                .and_then(staged_destination_name)
                .is_some_and(|destination| OsStr::new(destination) == name)
        })
        .collect();
    staged.sort();
    staged
}

fn process_temp_id() -> String {
    format!("{}-{}", std::process::id(), unix_time_millis())
}

/// How a staged file gets its permissions before any byte is written into it.
#[derive(Clone, Copy, Debug)]
enum StagingMode {
    /// Copy the permission bits of the file it will replace, if any (Unix
    /// only; see `keep_existing_permissions`).
    KeepExisting,
    /// Create it readable and writable by the current user only, whatever the
    /// file it replaces allowed.
    OwnerOnly,
}

/// Replace `path` with `bytes` atomically, creating missing parent directories.
pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_id = process_temp_id();
    write_file_atomically_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        || {},
    )
}

/// [`write_file_atomically`] with test seams for the temp name and publish point.
///
/// `suffix_for_attempt` names the temp file for each attempt, and
/// `before_publish` runs after the temp file is fully written and before it is
/// published. Both exist so tests can force a temp-name collision and hold
/// writers at the publish point; production code calls
/// [`write_file_atomically`].
fn write_file_atomically_with<S, P>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    before_publish: P,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
{
    write_file_atomically_with_publish(
        path,
        bytes,
        suffix_for_attempt,
        before_publish,
        publish_temp_file,
    )
}

/// Test seam: [`write_file_atomically`] with the temp name, a hook before the
/// publish, and the publish step itself injected.
///
/// Tests use it to force a temp-name collision, to hold writers at the publish
/// point, to observe the staged file and the destination at the moment of
/// publication, or to make the publication fail. Pass [`publish_temp_file`]
/// for the real publish. Production code calls [`write_file_atomically`].
///
/// `pub` only because `apps/rocm`'s metadata-cache tests drive it across the
/// crate boundary; hidden from the docs because it is not an API to build on.
#[doc(hidden)]
pub fn write_file_atomically_with_publish<S, P, F>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    before_publish: P,
    publish: F,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    let path = resolve_symlink_target(path)?;
    let tmp = stage_file_for_atomic_publish_with(
        &path,
        bytes,
        suffix_for_attempt,
        StagingMode::KeepExisting,
    )?;
    before_publish();
    publish_staged_file_with(&tmp, &path, publish)
}

/// Replace `path` with `bytes` atomically, as a file only the current user can open.
///
/// Mode `0600` on Unix, a protected DACL granting only the current user on
/// Windows. Creates missing parent directories.
///
/// Unlike [`write_file_atomically`], nothing of the file being replaced is
/// kept: a key file someone loosened to `0644`, or one that inherited a
/// readable ACL, comes back owner-only. The publish is a plain rename on every
/// platform, so it fails, leaving the old file in place, when another process
/// holds the destination open without delete sharing. A symbolic link at
/// `path` is replaced, not followed. See the module doc.
pub fn write_owner_only_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = stage_owner_only_file(path, bytes)?;
    publish_staged_file_with(&tmp, path, rename_file)
}

/// The staging half of [`write_owner_only_file_atomically`].
fn stage_owner_only_file(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = process_temp_id();
    stage_file_for_atomic_publish_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        StagingMode::OwnerOnly,
    )
}

/// Test support: what a [`write_owner_only_file_atomically`] killed between
/// staging and publishing leaves behind — `bytes` in an owner-only staged file
/// next to `path`, never published. Returns the staged file's path.
///
/// Compiled only for this crate's tests and under the `test-support` feature,
/// so the tests of the cleanups that must remove such leftovers start from the
/// file a real store leaves rather than a hand-made look-alike.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn stage_owner_only_file_unpublished(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    stage_owner_only_file(path, bytes)
}

/// The file a write to `path` should replace: `path` itself, or — when it is
/// a symbolic link — the file the link chain ends at, so the link survives.
/// A dangling link resolves to where its target would be, which is created.
fn resolve_symlink_target(path: &Path) -> Result<PathBuf> {
    let mut current = path.to_path_buf();
    for _ in 0..MAX_SYMLINK_HOPS {
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let target = fs::read_link(&current)
                    .with_context(|| format!("failed to read link {}", current.display()))?;
                current = if target.is_absolute() {
                    target
                } else {
                    current
                        .parent()
                        .context("link path has no parent directory")?
                        .join(target)
                };
            }
            _ => return Ok(current),
        }
    }
    bail!(
        "too many levels of symbolic links resolving {}",
        path.display()
    )
}

/// Write `bytes` to a fresh, uniquely named sibling of `path` and return the
/// staged file's path without publishing it.
///
/// The caller owns the staged file and must remove it once done with it; on a
/// staging failure nothing is left behind. To *replace* `path` with the bytes,
/// call [`write_file_atomically`] instead, which also removes the staged file
/// when the publish fails.
///
/// Unlike [`write_file_atomically`], `path` is taken as given: a symbolic link
/// there is not resolved, so the temp file sits next to the link, and
/// publishing it over that `path` replaces the link with a regular file.
pub fn stage_file_for_atomic_publish(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = process_temp_id();
    stage_file_for_atomic_publish_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        StagingMode::KeepExisting,
    )
}

fn stage_file_for_atomic_publish_with<S>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    mode: StagingMode,
) -> Result<PathBuf>
where
    S: FnMut(u32) -> OsString,
{
    // Flushed before it can be published: a rename that reaches the disk
    // before the data would leave an empty file under the real name after a
    // crash. A full disk can surface in the flush rather than in `write_all`.
    stage_file_for_atomic_publish_with_writer(
        path,
        bytes,
        suffix_for_attempt,
        mode,
        |file, bytes| write_then_flush(file, bytes, fs::File::sync_all),
    )
}

/// [`stage_file_for_atomic_publish_with`] with the write into the reserved
/// temp file injected, so a test can make it fail the way a full disk does
/// without filling one.
fn stage_file_for_atomic_publish_with_writer<S, W>(
    path: &Path,
    bytes: &[u8],
    mut suffix_for_attempt: S,
    mode: StagingMode,
    write: W,
) -> Result<PathBuf>
where
    S: FnMut(u32) -> OsString,
    W: FnOnce(&mut fs::File, &[u8]) -> io::Result<()>,
{
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let mut reserved = None;
    for attempt in 0..ATOMIC_WRITE_TEMP_ATTEMPTS {
        let suffix = suffix_for_attempt(attempt);
        let tmp = temp_sibling_path(path, &suffix)?;
        let opened = match mode {
            StagingMode::KeepExisting => fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp),
            StagingMode::OwnerOnly => crate::owner_only_file::open_new_owner_only(&tmp),
        };
        match opened {
            Ok(file) => {
                reserved = Some((tmp, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", tmp.display()));
            }
        }
    }
    let Some((tmp, mut file)) = reserved else {
        bail!(
            "failed to reserve a temporary file next to {} after {} attempts",
            path.display(),
            ATOMIC_WRITE_TEMP_ATTEMPTS
        );
    };

    // Before any byte is written, so a private file's new contents are never
    // readable under the wider default mode, even for a moment. An owner-only
    // file was created narrow and must not be widened to the old file's mode.
    let kept = match mode {
        StagingMode::KeepExisting => keep_existing_permissions(path, &tmp),
        StagingMode::OwnerOnly => Ok(()),
    };
    if let Err(error) = kept {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = write(&mut file, bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, path));
    }
    drop(file);
    Ok(tmp)
}

/// Write `bytes` into the staged file, then flush it to disk with `flush`, so
/// the data reaches the disk before the rename can. A flush error fails the
/// write like any other, so the caller removes the temp file. `flush` is
/// injected so a test can see that it runs, and after the bytes; production
/// passes `File::sync_all`.
fn write_then_flush<F>(file: &mut fs::File, bytes: &[u8], flush: F) -> io::Result<()>
where
    F: FnOnce(&fs::File) -> io::Result<()>,
{
    file.write_all(bytes)?;
    flush(file)
}

/// Give the staged file the permission bits of the file it will replace, so a
/// `0600` file is not republished as `0644`. Nothing to keep for a new file.
#[cfg(unix)]
fn keep_existing_permissions(path: &Path, tmp: &Path) -> Result<()> {
    match fs::metadata(path) {
        Ok(existing) => fs::set_permissions(tmp, existing.permissions())
            .with_context(|| format!("failed to copy the permissions of {}", path.display())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to read the permissions of {}", path.display())),
    }
}

/// Nothing is copied on Windows: a publish by rename leaves the new file with
/// the staged file's attributes and the ACL its directory hands down (see the
/// module doc), and no caller sets a mode-like attribute it needs kept.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn keep_existing_permissions(_path: &Path, _tmp: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
fn publish_staged_file(tmp: &Path, path: &Path) -> Result<()> {
    publish_staged_file_with(tmp, path, publish_temp_file)
}

/// Publish a staged file over `path`, removing the temp file if publishing
/// fails — unless the failed publish took the existing file away.
///
/// A rename either replaces the destination or leaves it alone, but Windows'
/// `ReplaceFileW` can fail after moving the old file off the destination name
/// (see `recover_partial_replace`). Removing the temp file then would leave no
/// copy of the contents under any name the user can find, so it is kept and
/// the error is a [`KeptReplacement`] that says where. The publish step itself
/// reports that case, so it is told apart by the error rather than by a look
/// at the destination taken before the publish, which another writer can make
/// stale.
fn publish_staged_file_with<F>(tmp: &Path, path: &Path, publish: F) -> Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    publish_staged_file_with_dir_sync(tmp, path, publish, sync_parent_dir)
}

/// [`publish_staged_file_with`] with the directory flush injected, so a test
/// can see that a successful publish flushes the destination's directory and
/// a failed one does not; production passes `sync_parent_dir`.
fn publish_staged_file_with_dir_sync<F, D>(
    tmp: &Path,
    path: &Path,
    publish: F,
    sync_dir: D,
) -> Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
    D: FnOnce(&Path),
{
    let Err(error) = publish(tmp, path) else {
        sync_dir(path);
        return Ok(());
    };
    if removed_the_destination(&error) && tmp.exists() {
        return Err(anyhow::Error::new(KeptReplacement {
            destination: path.to_path_buf(),
            kept: tmp.to_path_buf(),
            cause: error,
        }));
    }
    let _ = fs::remove_file(tmp);
    Err(error).with_context(|| format!("failed to publish {}", path.display()))
}

/// Flush the directory entry the rename just changed, so the new name — not
/// the old file, or no file — is what a crash leaves behind.
///
/// Best-effort: the file is already published, and some filesystems refuse to
/// sync a directory. Reporting failure here would claim the write failed when
/// it did not.
#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    // A bare relative name has the empty path as its parent, which cannot be
    // opened; its folder is the current one.
    let parent = match path.parent() {
        Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
        Some(parent) => parent,
        None => return,
    };
    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }
}

/// Not done on Windows: the rename's durability is left to the filesystem.
#[cfg(not(unix))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn sync_parent_dir(_path: &Path) {}

/// Test seam: move the staged file `tmp` over `path` in one step, replacing
/// any existing file without first removing it.
///
/// The real publish step to pass to [`write_file_atomically_with_publish`].
/// `pub` only for that cross-crate test use; it does not remove `tmp` on
/// failure, so production code calls [`write_file_atomically`] instead.
#[cfg(not(windows))]
#[doc(hidden)]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    fs::rename(tmp, path)
}

/// Test seam: move the staged file `tmp` over `path` in one step, replacing
/// any existing file without first removing it.
///
/// The real publish step to pass to [`write_file_atomically_with_publish`].
/// `pub` only for that cross-crate test use; it does not remove `tmp` on
/// failure, so production code calls [`write_file_atomically`] instead.
#[cfg(windows)]
#[doc(hidden)]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    publish_rename_first(tmp, path, rename_file, replace_file_windows)
}

/// The Windows publish order, with both steps passed in so a test can check it
/// on any host: `rename` first, because it replaces in one step and never
/// leaves the destination absent; `replace` only when the rename failed and the
/// destination exists. A failed rename onto a destination that does not exist is
/// returned as it is, since `replace` has nothing to replace.
#[cfg(any(windows, test))]
fn publish_rename_first<R, P>(tmp: &Path, path: &Path, rename: R, replace: P) -> io::Result<()>
where
    R: FnOnce(&Path, &Path) -> io::Result<()>,
    P: FnOnce(&Path, &Path) -> io::Result<()>,
{
    match rename(tmp, path) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            if path.try_exists()? {
                replace(path, tmp)
            } else {
                Err(rename_error)
            }
        }
    }
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn replace_file_windows(path: &Path, replacement: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;

    let path_wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement_wide: Vec<u16> = replacement
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();

    // SAFETY: both path buffers are valid, NUL-terminated UTF-16 strings and
    // remain alive for the duration of the synchronous Windows API call. The
    // optional backup, exclude, and reserved pointers are intentionally null.
    let replaced = unsafe {
        ReplaceFileW(
            path_wide.as_ptr(),
            replacement_wide.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if replaced != 0 {
        return Ok(());
    }
    // Microsoft documents two partial outcomes for a call without a backup
    // name, and in both the replacement is still under its own name:
    // `ERROR_UNABLE_TO_MOVE_REPLACEMENT` ("the replaced file no longer exists
    // and the replacement file exists under its original name") and
    // `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2` ("the file to be replaced still
    // exists with a different name"). Either way nothing is at `path` any
    // more. One plain rename puts the new contents there; if that fails too,
    // `publish_staged_file_with` keeps the replacement file.
    recover_partial_replace(io::Error::last_os_error(), replacement, path, rename_file)
}

/// `fs::rename` with both paths borrowed, the shape `recover_partial_replace`,
/// `publish_rename_first` and `publish_staged_file_with` take. `fs::rename`
/// itself is generic over `AsRef<Path>`, so passed directly it is pinned to one
/// lifetime per argument and does not satisfy their `FnOnce(&Path, &Path)`
/// bounds.
fn rename_file(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rocm-core-{label}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ))
    }

    /// A write that runs out of space while staging says so in words, and
    /// leaves no temp file behind to keep the disk full. Drives the real
    /// staging path — the one both [`write_file_atomically`] and
    /// [`stage_file_for_atomic_publish`] go through — with only the byte write
    /// into the reserved temp file swapped for one that fails like a full
    /// disk, so the mapping is checked where it is applied rather than on its
    /// own.
    #[test]
    fn staging_into_a_full_disk_reports_it_clearly_and_removes_the_temp_file() {
        let root = scratch_dir("atomic-full-disk");
        let destination = root.join("rocm.tar.gz");

        let error = stage_file_for_atomic_publish_with_writer(
            &destination,
            b"payload",
            |attempt| OsString::from(format!("full-disk-{attempt}")),
            StagingMode::KeepExisting,
            |_, _| Err(io::Error::from(io::ErrorKind::StorageFull)),
        )
        .expect_err("a write that hits a full disk must fail");

        let leftovers: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let _ = fs::remove_dir_all(&root);
        let text = format!("{error:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
        // The user knows the file they asked for, not the temp file that was
        // already removed, so the message names the destination.
        assert!(
            text.contains(&destination.display().to_string()) && !text.contains(".tmp-"),
            "{text}"
        );
        assert!(
            leftovers.is_empty(),
            "the failed staging left files behind: {leftovers:?}"
        );
    }

    /// When every temp name the writer tries is taken, it gives up after a
    /// bounded number of tries with an error that says so, rather than
    /// looping or writing through a file another writer reserved.
    #[test]
    fn staging_gives_up_when_every_temp_name_is_taken() {
        let root = scratch_dir("atomic-names-exhausted");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("active.json");
        let taken = temp_sibling_path(&destination, OsStr::new("taken")).unwrap();
        fs::write(&taken, b"another writer's staged file").unwrap();
        let mut tries = 0;

        let error = stage_file_for_atomic_publish_with(
            &destination,
            b"payload",
            |_| {
                tries += 1;
                OsString::from("taken")
            },
            StagingMode::KeepExisting,
        )
        .expect_err("a writer that finds every name taken must fail");

        let foreign = fs::read(&taken);
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);
        let text = format!("{error:#}");
        assert_eq!(tries, ATOMIC_WRITE_TEMP_ATTEMPTS);
        assert!(
            text.contains(&format!("after {ATOMIC_WRITE_TEMP_ATTEMPTS} attempts")),
            "{text}"
        );
        assert_eq!(foreign.unwrap(), b"another writer's staged file");
        assert_eq!(leftovers, vec![OsString::from("active.json.tmp-taken")]);
    }

    /// The temp name keeps every extension, so a cleanup sweep over a cache
    /// directory can still tell what a leftover was going to be.
    #[test]
    fn temp_sibling_path_preserves_multi_dot_file_names() {
        let temp =
            temp_sibling_path(Path::new("/tmp/cache/sdk.tar.gz"), OsStr::new("test")).unwrap();
        let name = temp.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(name, "sdk.tar.gz.tmp-test");
        assert_eq!(temp.parent().unwrap(), Path::new("/tmp/cache"));
    }

    /// Same property end to end: the temp file a write stages, observed at the
    /// moment it is published, keeps the destination's full name, and the
    /// publish leaves only the destination behind.
    #[test]
    fn write_file_atomically_temp_name_preserves_multi_dot_file_names() {
        let root = scratch_dir("atomic-name");
        let destination = root.join("artifact.tar.gz");
        let temp_id = process_temp_id();
        let mut staged_name = None;

        write_file_atomically_with_publish(
            &destination,
            b"payload",
            |attempt| OsString::from(format!("{temp_id}-{attempt}")),
            || {},
            |tmp, path| {
                staged_name = tmp.file_name().map(OsStr::to_os_string);
                publish_temp_file(tmp, path)
            },
        )
        .unwrap();

        let names: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let published = fs::read(&destination);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            staged_name,
            Some(OsString::from(format!("artifact.tar.gz.tmp-{temp_id}-0")))
        );
        assert_eq!(published.unwrap(), b"payload");
        assert_eq!(names, vec!["artifact.tar.gz".to_owned()]);
    }

    /// The Windows publish order: `rename` is tried first so the destination is
    /// never absent, and `ReplaceFileW` is reached only after a failed rename
    /// onto a destination that exists. Runs on every host because the order is
    /// a plain function of the two injected steps.
    #[test]
    fn windows_publish_renames_first_and_skips_the_replace_when_it_succeeds() {
        let root = scratch_dir("publish-rename-first");
        fs::create_dir_all(&root).unwrap();
        // The destination must exist: with it absent, no ordering of the two
        // steps would reach the replace, so the test could not tell them apart.
        let existing = root.join("existing");
        fs::write(&existing, b"old").unwrap();
        let replaced = std::cell::Cell::new(false);
        let renamed = std::cell::Cell::new(false);
        publish_rename_first(
            Path::new("tmp"),
            &existing,
            |_, _| {
                renamed.set(true);
                Ok(())
            },
            |_, _| {
                replaced.set(true);
                Ok(())
            },
        )
        .expect("a successful rename is the whole publish");
        let _ = fs::remove_dir_all(&root);
        assert!(renamed.get(), "the rename must be tried first");
        assert!(
            !replaced.get(),
            "ReplaceFileW must not run after a rename that worked"
        );
    }

    #[test]
    fn windows_publish_falls_back_to_the_replace_only_for_an_existing_destination() {
        let root = scratch_dir("publish-fallback");
        fs::create_dir_all(&root).unwrap();
        let existing = root.join("existing");
        fs::write(&existing, b"old").unwrap();
        let sharing = || Err(io::Error::from(io::ErrorKind::PermissionDenied));

        let replaced = std::cell::Cell::new(None);
        publish_rename_first(
            Path::new("tmp"),
            &existing,
            |_, _| sharing(),
            |path, tmp| {
                replaced.set(Some((path.to_path_buf(), tmp.to_path_buf())));
                Ok(())
            },
        )
        .expect("the fallback result is the publish result");
        assert_eq!(
            replaced.take(),
            Some((existing, PathBuf::from("tmp"))),
            "the fallback replaces the destination with the staged file"
        );

        let missing = root.join("missing");
        let called = std::cell::Cell::new(false);
        let error = publish_rename_first(
            Path::new("tmp"),
            &missing,
            |_, _| sharing(),
            |_, _| {
                called.set(true);
                Ok(())
            },
        )
        .expect_err("a failed rename onto a missing destination is an error");
        let _ = fs::remove_dir_all(&root);
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(
            !called.get(),
            "there is nothing for ReplaceFileW to replace"
        );
    }

    /// A reader that polls the destination while a writer replaces it over and
    /// over must never find it missing: that is what a lock-free reader of the
    /// config or an automation store would read as an empty store. A transient
    /// permission error while a replace is in flight is tolerated; an absent
    /// destination is not.
    ///
    /// This is a Windows-lane check. On Linux `rename` replaces atomically, so
    /// the test passes with either publish order and would not notice the order
    /// being reversed; it only discriminates on Windows, where a publish that
    /// removes the destination before putting the new file in place leaves a
    /// window the reader can fall into. What it still does not cover is the
    /// `ReplaceFileW` fallback, which runs only when the rename fails and is the
    /// one path that can leave the destination absent: it is reached by no test
    /// that races a reader, and its order is pinned only by the injected-step
    /// tests above.
    #[test]
    fn a_reader_never_finds_the_destination_absent_while_it_is_replaced() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        /// Tells the reader to stop when dropped, so a writer that panics
        /// through `unwrap` still ends the polling thread instead of leaving it
        /// to spin until the process exits.
        struct StopReader(Arc<AtomicBool>);
        impl Drop for StopReader {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }

        let root = scratch_dir("atomic-reader-gap");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("store.json");
        write_file_atomically(&destination, b"0").unwrap();

        let done = Arc::new(AtomicBool::new(false));
        let (first_read, reader_started) = std::sync::mpsc::channel::<()>();
        let reader = {
            let destination = destination.clone();
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut reads = 0u32;
                while !done.load(Ordering::Relaxed) {
                    match fs::read(&destination) {
                        Ok(_) => {
                            reads += 1;
                            if reads == 1 {
                                let _ = first_read.send(());
                            }
                        }
                        Err(error) if error.kind() == io::ErrorKind::NotFound => {
                            return Err(error);
                        }
                        Err(_) => {}
                    }
                }
                Ok(reads)
            })
        };
        // Start writing only once the reader is polling, so a late-scheduled
        // reader cannot miss every replacement and fail on timing alone. A
        // reader that exits before its first read (the destination was absent)
        // disconnects the channel; that is reported from its outcome below.
        let stop_reader = StopReader(Arc::clone(&done));
        let started = reader_started.recv_timeout(std::time::Duration::from_secs(30));
        if started.is_ok() {
            for round in 1..=300u32 {
                write_file_atomically(&destination, round.to_string().as_bytes()).unwrap();
            }
        }
        drop(stop_reader);
        let outcome = reader.join().unwrap();
        let _ = fs::remove_dir_all(&root);
        let reads = outcome.expect("the destination was absent while it was being replaced");
        assert!(
            started.is_ok() && reads > 0,
            "the reader never read the destination at all: {started:?}"
        );
    }

    #[test]
    fn concurrent_atomic_writes_do_not_remove_a_published_destination() {
        let root = scratch_dir("atomic-collision");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("sdk.tar.gz");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .enumerate()
            .map(|(writer, bytes)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    write_file_atomically_with(
                        &destination,
                        bytes,
                        |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        },
                        || {
                            before_publish.wait();
                        },
                    )
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a writer must remain published");
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
    }

    #[test]
    fn concurrent_cached_publications_use_distinct_staging_files() {
        let root = scratch_dir("cache-publish-collision");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.body");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .enumerate()
            .map(|(writer, bytes)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    // Both threads ask for the same first name, as two writers
                    // sharing a pid and a millisecond would, so the collision
                    // happens every run rather than when the clock lines up.
                    let staged = stage_file_for_atomic_publish_with(
                        &destination,
                        bytes,
                        |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        },
                        StagingMode::KeepExisting,
                    )?;
                    before_publish.wait();
                    publish_staged_file(&staged, &destination)
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a cache writer must remain published");
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
        assert!(
            leftovers.is_empty(),
            "staged cache files leaked: {leftovers:?}"
        );
    }

    #[test]
    fn failed_cached_publication_preserves_destination_and_cleans_staging_file() {
        let root = scratch_dir("cache-publish-failure");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.body");
        fs::write(&destination, b"published").unwrap();
        let staged = stage_file_for_atomic_publish(&destination, b"replacement").unwrap();

        publish_staged_file_with(&staged, &destination, |_, _| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "simulated cache publication failure",
            ))
        })
        .expect_err("simulated cache publication failure must be returned");

        assert_eq!(fs::read(&destination).unwrap(), b"published");
        assert!(
            !staged.exists(),
            "failed publication leaked its staging file"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_atomic_replace_preserves_destination_and_cleans_temp() {
        let root = scratch_dir("atomic-replace-failure");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("sdk.tar.gz");
        fs::write(&destination, b"published").unwrap();

        write_file_atomically_with_publish(
            &destination,
            b"replacement",
            |attempt| OsString::from(format!("replace-failure-{attempt}")),
            || {},
            |tmp, path| {
                assert_eq!(fs::read(tmp).unwrap(), b"replacement");
                assert_eq!(path, destination);
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "simulated atomic replacement failure",
                ))
            },
        )
        .expect_err("simulated replacement failure must be returned");

        assert_eq!(fs::read(&destination).unwrap(), b"published");
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(leftovers, vec![OsString::from("sdk.tar.gz")]);
    }

    fn simulated_partial_replace() -> io::Error {
        destination_removed_error(
            io::Error::from_raw_os_error(PARTIAL_REPLACE_ERROR_CODES[0]),
            io::Error::from(io::ErrorKind::PermissionDenied),
        )
    }

    /// A publish that fails after taking the old file away — what Windows'
    /// `ReplaceFileW` documents for `ERROR_UNABLE_TO_MOVE_REPLACEMENT` without
    /// a backup name, when the rename retried after it fails too — must not
    /// also delete the temp file, which then holds the only copy. The error
    /// names it so the user can put it back, and a caller can find it.
    #[test]
    fn a_publish_that_removes_the_destination_keeps_the_new_contents() {
        let root = scratch_dir("replace-partial");
        let destination = root.join("active.json");
        fs::create_dir_all(&root).unwrap();
        fs::write(&destination, b"old").unwrap();

        let error = write_file_atomically_with_publish(
            &destination,
            b"new",
            |attempt| OsString::from(format!("replace-partial-{attempt}")),
            || {},
            |_tmp, path| {
                fs::remove_file(path).unwrap();
                Err(simulated_partial_replace())
            },
        )
        .expect_err("the failed publish must be reported");
        let message = format!("{error:#}");
        let reported = error
            .downcast_ref::<KeptReplacement>()
            .map(|kept| (kept.kept().to_path_buf(), kept.destination().to_path_buf()));

        let kept = temp_sibling_path(&destination, OsStr::new("replace-partial-0")).unwrap();
        let kept_bytes = fs::read(&kept);
        let destination_exists = destination.exists();
        let _ = fs::remove_dir_all(&root);
        assert!(!destination_exists);
        assert_eq!(kept_bytes.unwrap(), b"new", "the new contents were deleted");
        assert!(
            message.contains(&kept.display().to_string()),
            "the error does not say where the contents are: {message}"
        );
        assert_eq!(reported, Some((kept, destination)));
    }

    /// The race a pre-publish look at the destination gets wrong: no
    /// destination when this writer starts, another writer publishes one, and
    /// then this writer's replace takes it away and fails. The temp file still
    /// holds the only copy, so it is still kept.
    #[test]
    fn a_publish_that_removes_a_destination_another_writer_created_keeps_the_new_contents() {
        let root = scratch_dir("replace-partial-race");
        let destination = root.join("active.json");
        fs::create_dir_all(&root).unwrap();

        let error = write_file_atomically_with_publish(
            &destination,
            b"second",
            |attempt| OsString::from(format!("replace-partial-race-{attempt}")),
            || {},
            |_tmp, path| {
                fs::write(path, b"first").unwrap();
                fs::remove_file(path).unwrap();
                Err(simulated_partial_replace())
            },
        )
        .expect_err("the failed publish must be reported");

        let kept = temp_sibling_path(&destination, OsStr::new("replace-partial-race-0")).unwrap();
        let kept_bytes = fs::read(&kept);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            kept_bytes.unwrap(),
            b"second",
            "the new contents were deleted"
        );
        assert!(
            error.downcast_ref::<KeptReplacement>().is_some(),
            "{error:#}"
        );
    }

    /// When the temp file is gone too there is nothing to keep, so the error
    /// must not point the user at it.
    #[test]
    fn a_partial_replace_that_lost_the_temp_file_does_not_claim_to_keep_it() {
        let root = scratch_dir("replace-partial-no-temp");
        let destination = root.join("active.json");
        fs::create_dir_all(&root).unwrap();
        fs::write(&destination, b"old").unwrap();

        let error = write_file_atomically_with_publish(
            &destination,
            b"new",
            |attempt| OsString::from(format!("replace-partial-no-temp-{attempt}")),
            || {},
            |tmp, path| {
                fs::remove_file(path).unwrap();
                fs::remove_file(tmp).unwrap();
                Err(simulated_partial_replace())
            },
        )
        .expect_err("the failed publish must be reported");
        let _ = fs::remove_dir_all(&root);
        let message = format!("{error:#}");
        assert!(
            error.downcast_ref::<KeptReplacement>().is_none(),
            "{message}"
        );
        assert!(!message.contains("kept in"), "{message}");
    }

    /// Any other failed replace leaves the destination alone, so nothing is
    /// retried and the error comes back as it was.
    #[test]
    fn a_replace_that_left_the_destination_alone_is_not_retried() {
        let mut renamed = false;
        let error = recover_partial_replace(
            io::Error::from_raw_os_error(5),
            Path::new("staged"),
            Path::new("destination"),
            |_, _| {
                renamed = true;
                Ok(())
            },
        )
        .expect_err("an ordinary failure is returned");
        assert!(!renamed);
        assert_eq!(error.raw_os_error(), Some(5));
        assert!(!removed_the_destination(&error));
    }

    /// Both documented partial codes retry once as a rename of the staged
    /// file onto the destination, and a successful retry is a success.
    #[test]
    fn a_partial_replace_is_retried_as_a_rename() {
        for code in PARTIAL_REPLACE_ERROR_CODES {
            let mut renamed = None;
            recover_partial_replace(
                io::Error::from_raw_os_error(code),
                Path::new("staged"),
                Path::new("destination"),
                |from, to| {
                    renamed = Some((from.to_path_buf(), to.to_path_buf()));
                    Ok(())
                },
            )
            .unwrap_or_else(|error| panic!("code {code}: {error}"));
            assert_eq!(
                renamed,
                Some((PathBuf::from("staged"), PathBuf::from("destination"))),
                "code {code}"
            );
        }
    }

    /// A retry that fails too is reported as the case that keeps the temp file.
    #[test]
    fn a_partial_replace_whose_retry_fails_reports_the_destination_removed() {
        for code in PARTIAL_REPLACE_ERROR_CODES {
            let error = recover_partial_replace(
                io::Error::from_raw_os_error(code),
                Path::new("staged"),
                Path::new("destination"),
                |_, _| Err(io::Error::from(io::ErrorKind::PermissionDenied)),
            )
            .expect_err("a failed retry is returned");
            assert!(removed_the_destination(&error), "code {code}: {error}");
            assert!(error.to_string().contains("retrying as a plain rename"));
        }
    }

    /// The plain numbers the partial-replace decision uses are the codes
    /// `ReplaceFileW` actually returns.
    #[cfg(windows)]
    #[test]
    fn partial_replace_codes_match_the_windows_constants() {
        use windows_sys::Win32::Foundation::{
            ERROR_UNABLE_TO_MOVE_REPLACEMENT, ERROR_UNABLE_TO_MOVE_REPLACEMENT_2,
        };
        assert_eq!(
            PARTIAL_REPLACE_ERROR_CODES.map(|code| u32::try_from(code).unwrap()),
            [
                ERROR_UNABLE_TO_MOVE_REPLACEMENT,
                ERROR_UNABLE_TO_MOVE_REPLACEMENT_2
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_temp_name_preserves_non_unicode_file_name_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let file_name = OsString::from_vec(b"sdk-\xff.tar.gz".to_vec());
        let destination = Path::new("/tmp").join(&file_name);
        let temp = temp_sibling_path(&destination, OsStr::new("collision")).unwrap();

        let mut expected = file_name.into_vec();
        expected.extend_from_slice(b".tmp-collision");
        assert_eq!(temp.file_name().unwrap().as_bytes(), expected);
    }

    /// Regression: a failed write must not leave a `.tmp-*` scratch file
    /// behind. The name is unique per attempt, so before this an orphan
    /// accumulated per retry — and when the failure is a full disk, those
    /// orphans are exactly what keeps it full.
    ///
    /// Provokes the failure by pointing the destination at a non-empty
    /// directory: the temp file is written, then neither the rename nor the
    /// replace fallback can succeed. Portable, unlike an out-of-space test.
    #[test]
    fn write_file_atomically_cleans_up_temp_when_the_rename_fails() {
        let root = scratch_dir("atomic");
        let occupied = root.join("sdk.tar.gz");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        write_file_atomically(&occupied, b"payload")
            .expect_err("renaming onto a non-empty directory should fail");

        let leftovers: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert!(
            leftovers.is_empty(),
            "failed write left temp files behind: {leftovers:?}"
        );
    }

    /// Mirrors the `/dev/shm` reproduction from the original report: a genuine
    /// ENOSPC, not a rename failure standing in for one.
    ///
    /// Ignored by default because it fills `/dev/shm`, which is shared with
    /// anything else on the host, so it is not safe to run concurrently. Run
    /// with `cargo test -p rocm-core -- --ignored write_file_atomically_cleans_up`.
    #[test]
    #[ignore = "fills /dev/shm to provoke ENOSPC; not safe to run concurrently"]
    fn write_file_atomically_cleans_up_temp_on_write_failure() {
        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        }
        let dir = shm.join(format!("rocm-core-enospc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("artifact.tar.gz");
        // Larger than the tmpfs, so the write is guaranteed to hit ENOSPC.
        let payload = vec![0u8; 256 * 1024 * 1024];

        let mut failures = Vec::new();
        for _ in 0..2 {
            write_file_atomically(&dest, &payload)
                .expect_err("writing past the end of the filesystem should fail");
            failures.push(
                fs::read_dir(&dir)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            );
        }
        let destination_exists = dest.exists();
        let _ = fs::remove_dir_all(&dir);

        for leftovers in &failures {
            assert!(
                leftovers.is_empty(),
                "failed write left files behind: {leftovers:?}"
            );
        }
        assert!(
            !destination_exists,
            "destination must not exist after failure"
        );
    }

    /// The names in `dir`, sorted.
    fn names_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// Replacing a file keeps its permission bits: a `0600` file must not be
    /// republished as `0644`, and an unusual mode survives as set.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_keeps_the_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let root = scratch_dir("permissions");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("config.json");
        for mode in [0o600, 0o640, 0o604] {
            fs::write(&destination, b"old").unwrap();
            fs::set_permissions(&destination, fs::Permissions::from_mode(mode)).unwrap();

            write_file_atomically(&destination, b"new").unwrap();

            let after = fs::metadata(&destination).unwrap().permissions().mode() & 0o777;
            assert_eq!(after, mode, "mode {mode:o} was not kept");
            assert_eq!(fs::read(&destination).unwrap(), b"new");
        }
        let _ = fs::remove_dir_all(&root);
    }

    /// The kept mode is on the temp file before the first byte reaches it, so
    /// a private file's new contents are never readable under the default mode.
    #[cfg(unix)]
    #[test]
    fn staging_applies_the_kept_mode_before_writing_any_byte() {
        use std::os::unix::fs::PermissionsExt;

        // 0o604 is a mode no usual umask gives a new file (022 gives 0o644,
        // 077 gives 0o600), so seeing it on the temp file means it was set.
        for mode in [0o600, 0o604] {
            let root = scratch_dir("permissions-before-write");
            fs::create_dir_all(&root).unwrap();
            let destination = root.join("config.json");
            fs::write(&destination, b"old").unwrap();
            fs::set_permissions(&destination, fs::Permissions::from_mode(mode)).unwrap();

            let mut mode_while_writing = None;
            stage_file_for_atomic_publish_with_writer(
                &destination,
                b"secret",
                |attempt| OsString::from(format!("mode-{attempt}")),
                StagingMode::KeepExisting,
                |file, bytes| {
                    mode_while_writing = Some(file.metadata()?.permissions().mode() & 0o777);
                    file.write_all(bytes)
                },
            )
            .unwrap();
            let _ = fs::remove_dir_all(&root);

            assert_eq!(mode_while_writing, Some(mode), "mode {mode:o}");
        }
    }

    /// When the existing file's permissions cannot be read, staging fails
    /// with that error and removes its temp file. A symlink loop at the
    /// destination makes the read fail on Unix without root or chmod;
    /// `stage_file_for_atomic_publish` takes the path as given, so the loop
    /// reaches the permission read rather than the link resolution.
    #[cfg(unix)]
    #[test]
    fn staging_that_cannot_read_the_existing_permissions_removes_its_temp_file() {
        use std::os::unix::fs::symlink;

        let root = scratch_dir("permissions-unreadable");
        fs::create_dir_all(&root).unwrap();
        let a = root.join("a.json");
        let b = root.join("b.json");
        symlink(&b, &a).unwrap();
        symlink(&a, &b).unwrap();

        let error = stage_file_for_atomic_publish(&a, b"x").expect_err("the read must fail");
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(
            format!("{error:#}").contains("failed to read the permissions of"),
            "{error:#}"
        );
        assert_eq!(names, ["a.json", "b.json"], "the temp file was left behind");
    }

    /// The staged bytes are flushed, after they are written, and a flush
    /// error fails the write. Whether the flush reaches the disk cannot be
    /// observed from a test; that it is asked for can.
    #[test]
    fn staged_bytes_are_flushed_after_they_are_written() {
        let root = scratch_dir("flush-order");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("staged");
        let mut file = fs::File::create(&path).unwrap();

        let mut seen_at_flush = None;
        write_then_flush(&mut file, b"payload", |_| {
            seen_at_flush = Some(fs::read(&path).unwrap());
            Ok(())
        })
        .unwrap();
        let failed = write_then_flush(&mut file, b"more", |_| {
            Err(io::Error::other("flush refused"))
        });
        drop(file);
        let _ = fs::remove_dir_all(&root);

        assert_eq!(seen_at_flush.as_deref(), Some(&b"payload"[..]));
        assert_eq!(failed.unwrap_err().to_string(), "flush refused");
    }

    /// A publish that succeeds flushes the destination's directory once; one
    /// that fails does not.
    #[test]
    fn a_successful_publish_flushes_the_directory_and_a_failed_one_does_not() {
        let root = scratch_dir("dir-flush");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("target.json");
        let tmp = root.join("target.json.tmp-1");
        fs::write(&tmp, b"new").unwrap();

        let mut flushed = Vec::new();
        publish_staged_file_with_dir_sync(&tmp, &destination, publish_temp_file, |path| {
            flushed.push(path.to_path_buf());
        })
        .unwrap();
        fs::write(&tmp, b"newer").unwrap();
        let mut flushed_after_failure = 0;
        publish_staged_file_with_dir_sync(
            &tmp,
            &destination,
            |_, _| Err(io::Error::other("publish refused")),
            |_| flushed_after_failure += 1,
        )
        .expect_err("the publish was refused");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(flushed, [destination]);
        assert_eq!(flushed_after_failure, 0);
    }

    /// A destination that is a symbolic link keeps being one: the file it
    /// points at is replaced, through a chain of links and a relative link
    /// alike, and nothing is left next to either end.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_replaces_a_symlinks_target_and_keeps_the_link() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = scratch_dir("symlink");
        let dotfiles = root.join("dotfiles");
        let config_dir = root.join("config");
        fs::create_dir_all(&dotfiles).unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let target = dotfiles.join("rocm-config.json");
        fs::write(&target, b"old").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
        // config/config.json -> links/deeper/hop.json
        //   -> ../../dotfiles/rocm-config.json, a relative hop in a directory
        // two levels deep, so resolving it against the entry link's directory
        // instead of the hop's own would miss the target.
        let hop_dir = root.join("links").join("deeper");
        fs::create_dir_all(&hop_dir).unwrap();
        let hop = hop_dir.join("hop.json");
        symlink(Path::new("../../dotfiles/rocm-config.json"), &hop).unwrap();
        let link = config_dir.join("config.json");
        symlink(&hop, &link).unwrap();

        write_file_atomically(&link, b"new").unwrap();

        let link_kept = fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink();
        let hop_kept = fs::symlink_metadata(&hop).unwrap().file_type().is_symlink();
        let through_link = fs::read(&link).unwrap();
        let target_contents = fs::read(&target).unwrap();
        let target_mode = fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        let config_names = names_in(&config_dir);
        let hop_names = names_in(&hop_dir);
        let dotfiles_names = names_in(&dotfiles);
        let _ = fs::remove_dir_all(&root);

        assert!(link_kept && hop_kept, "a link was replaced by a file");
        assert_eq!(through_link, b"new");
        assert_eq!(target_contents, b"new");
        assert_eq!(target_mode, 0o600);
        assert_eq!(config_names, ["config.json"]);
        assert_eq!(hop_names, ["hop.json"]);
        assert_eq!(dotfiles_names, ["rocm-config.json"]);
    }

    /// A link loop is an error, not a hang, and writes nothing.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_refuses_a_symlink_loop() {
        use std::os::unix::fs::symlink;

        let root = scratch_dir("symlink-loop");
        fs::create_dir_all(&root).unwrap();
        let a = root.join("a.json");
        let b = root.join("b.json");
        symlink(&b, &a).unwrap();
        symlink(&a, &b).unwrap();

        let error = write_file_atomically(&a, b"x").expect_err("a link loop must fail");
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(
            format!("{error:#}").contains("too many levels of symbolic links"),
            "{error:#}"
        );
        assert_eq!(names, ["a.json", "b.json"]);
    }

    /// What a writer killed before publishing leaves — a file the real staging
    /// path names — is found as belonging to its destination, and nothing
    /// else is: not the destination, not another file's staged copy whose
    /// name merely starts the same way.
    #[test]
    fn staged_siblings_finds_the_files_staging_left_for_that_destination_only() {
        let root = scratch_dir("staged-siblings");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("svc.endpoint-key");
        fs::write(&destination, b"published").unwrap();
        let staged = stage_owner_only_file_unpublished(&destination, b"secret").unwrap();
        for unrelated in [
            "svc.endpoint-key2.tmp-1-2-0",
            "other-svc.endpoint-key.tmp-1-2-0",
            "svc.endpoint-key.tmp-",
            "svc.log",
        ] {
            fs::write(root.join(unrelated), b"x").unwrap();
        }

        let found = staged_siblings(&destination);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(found, [staged]);
    }

    /// The name the real owner-only staging writes is read back as its
    /// destination's; a name that merely contains the marker is not.
    #[test]
    fn staged_destination_name_reads_back_only_what_staging_writes() {
        let root = scratch_dir("staged-name");
        let destination = root.join("svc-1.endpoint-key");
        let staged = stage_owner_only_file_unpublished(&destination, b"secret").unwrap();
        let name = staged.file_name().unwrap().to_str().unwrap().to_owned();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(staged_destination_name(&name), Some("svc-1.endpoint-key"));
        assert_eq!(staged_destination_name("svc-1.endpoint-key"), None);
        assert_eq!(staged_destination_name(".tmp-1-2-0"), None);
        assert_eq!(staged_destination_name("svc-1.endpoint-key.tmp-"), None);
        assert_eq!(staged_destination_name("svc-1.endpoint-key.tmp-1-2"), None);
        assert_eq!(
            staged_destination_name("svc-1.endpoint-key.tmp-1-2-0-3"),
            None
        );
        assert_eq!(
            staged_destination_name("svc-1.endpoint-key.tmp-1-x-0"),
            None
        );
        // The key file of a service whose id itself contains the marker.
        assert_eq!(
            staged_destination_name("a.endpoint-key.tmp-x.endpoint-key"),
            None
        );
        assert_eq!(
            staged_destination_name("a.endpoint-key.tmp-1-2-3.endpoint-key"),
            None
        );
    }

    /// Cleaning up service `a`'s key must not take the live key of a service
    /// whose id was chosen to look like a staged copy of it.
    #[test]
    fn staged_siblings_ignores_a_key_named_like_a_staged_copy() {
        let root = scratch_dir("staged-adversarial");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("a.endpoint-key");
        fs::write(&destination, b"a's key").unwrap();
        for id in ["a.endpoint-key.tmp-x", "a.endpoint-key.tmp-1-2-3"] {
            fs::write(root.join(format!("{id}.endpoint-key")), b"another key").unwrap();
        }

        let found = staged_siblings(&destination);
        let _ = fs::remove_dir_all(&root);
        assert!(found.is_empty(), "{found:?}");
    }

    /// A secret file someone loosened comes back owner-only: unlike
    /// [`write_file_atomically`], the owner-only writer does not keep the old
    /// mode, and does not leave a temp file behind.
    #[cfg(unix)]
    #[test]
    fn write_owner_only_file_atomically_tightens_a_loosened_file() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = scratch_dir("owner-only-loosened");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("svc.endpoint-key");
        fs::write(&destination, b"old").unwrap();
        fs::set_permissions(&destination, fs::Permissions::from_mode(0o644)).unwrap();

        write_owner_only_file_atomically(&destination, b"new").unwrap();

        let mode = fs::metadata(&destination).unwrap().permissions().mode() & 0o777;
        let contents = fs::read(&destination).unwrap();
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(mode, 0o600, "mode was {mode:o}");
        assert_eq!(contents, b"new");
        assert_eq!(names, ["svc.endpoint-key"]);
    }

    /// A new file, in a directory that does not exist yet, is created 0600.
    #[cfg(unix)]
    #[test]
    fn write_owner_only_file_atomically_creates_a_new_file_0600() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = scratch_dir("owner-only-new");
        let destination = root.join("services").join("svc.endpoint-key");

        write_owner_only_file_atomically(&destination, b"key").unwrap();

        let mode = fs::metadata(&destination).unwrap().permissions().mode() & 0o777;
        let contents = fs::read(&destination).unwrap();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(mode, 0o600, "mode was {mode:o}");
        assert_eq!(contents, b"key");
    }

    /// A symbolic link at the destination is replaced by the new file, not
    /// followed: the secret must not be written to wherever the link points,
    /// and the link's target keeps its contents and mode.
    #[cfg(unix)]
    #[test]
    fn write_owner_only_file_atomically_replaces_a_symlink_instead_of_following_it() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let root = scratch_dir("owner-only-symlink");
        fs::create_dir_all(&root).unwrap();
        let target = root.join("elsewhere");
        fs::write(&target, b"untouched").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
        let destination = root.join("svc.endpoint-key");
        symlink(&target, &destination).unwrap();

        write_owner_only_file_atomically(&destination, b"secret").unwrap();

        let is_link = fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_symlink();
        let contents = fs::read(&destination).unwrap();
        let mode = fs::metadata(&destination).unwrap().permissions().mode() & 0o777;
        let target_contents = fs::read(&target).unwrap();
        let target_mode = fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        let _ = fs::remove_dir_all(&root);
        assert!(!is_link, "the link was kept and followed");
        assert_eq!(contents, b"secret");
        assert_eq!(mode, 0o600, "mode was {mode:o}");
        assert_eq!(target_contents, b"untouched");
        assert_eq!(target_mode, 0o644);
    }

    /// A publish that cannot rename over the destination fails, leaves the
    /// destination alone and removes the staged file, so no stray copy of the
    /// secret stays on disk.
    #[test]
    fn write_owner_only_file_atomically_cleans_up_when_the_rename_fails() {
        let root = scratch_dir("owner-only-rename-fails");
        let destination = root.join("svc.endpoint-key");
        // A non-empty directory cannot be replaced by a rename on any platform.
        fs::create_dir_all(destination.join("occupied")).unwrap();

        let error = write_owner_only_file_atomically(&destination, b"secret")
            .expect_err("renaming a file over a directory must fail");

        let names = names_in(&root);
        let still_a_dir = destination.join("occupied").is_dir();
        let _ = fs::remove_dir_all(&root);
        assert!(
            format!("{error:#}").contains("failed to publish"),
            "{error:#}"
        );
        assert_eq!(names, ["svc.endpoint-key"]);
        assert!(still_a_dir);
    }

    /// On Windows a destination that inherited or was given a loose DACL comes
    /// back with a protected DACL granting only the current user, because the
    /// publish is a rename of a file created owner-only, never a
    /// `ReplaceFileW` that would carry the old DACL over.
    #[cfg(windows)]
    #[test]
    fn write_owner_only_file_atomically_replaces_a_loose_dacl_with_an_owner_only_one() {
        use crate::owner_only_file::owner_only_sddl;
        use crate::owner_only_file::win32::test_support::{
            canonical_sddl, dacl_sddl, flags_and_entries,
        };
        use crate::owner_only_file::win32::{create_new_with_sddl, current_user_sid};

        let root = scratch_dir("owner-only-dacl");
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("svc.endpoint-key");
        let sid = current_user_sid().unwrap();
        // Everyone (`WD`) may read and write it, and it inherits too.
        let loose = format!("D:(A;;FA;;;WD)(A;;FA;;;{sid})");
        let mut file = create_new_with_sddl(&destination, &loose).unwrap();
        file.write_all(b"old").unwrap();
        drop(file);
        let before = dacl_sddl(&destination).unwrap();

        write_owner_only_file_atomically(&destination, b"new").unwrap();

        let after = dacl_sddl(&destination).unwrap();
        let expected = canonical_sddl(&owner_only_sddl(&sid)).unwrap();
        let contents = fs::read(&destination).unwrap();
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);

        assert!(
            before.contains(";;;WD)"),
            "the setup DACL is not loose: {before}"
        );
        let (flags, entries) = flags_and_entries(&after);
        let (_, expected_entries) = flags_and_entries(&expected);
        assert!(flags.contains('P'), "the DACL is not protected: {after}");
        assert_eq!(
            entries, expected_entries,
            "the DACL grants more than the user"
        );
        assert_eq!(contents, b"new");
        assert_eq!(names, ["svc.endpoint-key"]);
    }
}
