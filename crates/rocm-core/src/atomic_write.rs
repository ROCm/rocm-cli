// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Atomic file replacement, shared by `rocm` and `rocmd`.
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
//! On Windows, when the destination already exists, the publish uses
//! `ReplaceFileW` rather than `rename`. `std::fs::rename` does replace an
//! existing file there, but as `MoveFileExW` it fails with a sharing violation
//! while another process holds the destination open, which is the case
//! `ReplaceFileW` is used for. The concurrency tests below run on every
//! platform, but each starts with no destination, so on Windows at most one of
//! their publishes goes through `ReplaceFileW`: two `ReplaceFileW` publishes of
//! one file in flight at once are not exercised by any test.
//!
//! One failure leaves the destination absent: Microsoft documents that
//! `ReplaceFileW`, called without a backup name, can fail after it has already
//! moved the old file off the destination name
//! (`ERROR_UNABLE_TO_MOVE_REPLACEMENT` and `ERROR_UNABLE_TO_MOVE_REPLACEMENT_2`).
//! The publish then retries once as a plain rename. If that fails too, the
//! temp file is kept rather than removed, since it holds the only copy of the
//! new contents, and the error is a [`KeptReplacement`] naming it. The
//! decision rests on the error code the replace returned, not on whether the
//! destination existed beforehand, so a second writer that created the
//! destination in between cannot turn it into a deletion.
//!
//! Atomic, not durable: neither the temp file nor its directory is fsynced
//! before the publish. That survives a crashed process and a concurrent
//! writer, but after a power loss a filesystem that does not order data before
//! the rename can still show an empty or stale destination.
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

/// A temp path next to `path`, preserving the full file name so a
/// multi-extension artifact keeps its extensions (`sdk.tar.gz` becomes
/// `sdk.tar.gz.tmp-<suffix>`, where `with_extension` would drop `.gz`).
fn temp_sibling_path(path: &Path, suffix: &OsStr) -> Result<PathBuf> {
    let parent = path.parent().context("file path has no parent directory")?;
    let mut file_name = path
        .file_name()
        .context("file path has no file name")?
        .to_os_string();
    file_name.push(".tmp-");
    file_name.push(suffix);
    Ok(parent.join(file_name))
}

fn process_temp_id() -> String {
    format!("{}-{}", std::process::id(), unix_time_millis())
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
    let tmp = stage_file_for_atomic_publish_with(path, bytes, suffix_for_attempt)?;
    before_publish();
    publish_staged_file_with(&tmp, path, publish)
}

/// Write `bytes` to a fresh, uniquely named sibling of `path` and return the
/// staged file's path without publishing it.
///
/// The caller owns the staged file and must remove it once done with it; on a
/// staging failure nothing is left behind. To *replace* `path` with the bytes,
/// call [`write_file_atomically`] instead, which also removes the staged file
/// when the publish fails.
pub fn stage_file_for_atomic_publish(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = process_temp_id();
    stage_file_for_atomic_publish_with(path, bytes, |attempt| {
        OsString::from(format!("{temp_id}-{attempt}"))
    })
}

fn stage_file_for_atomic_publish_with<S>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
) -> Result<PathBuf>
where
    S: FnMut(u32) -> OsString,
{
    stage_file_for_atomic_publish_with_writer(path, bytes, suffix_for_attempt, |file, bytes| {
        file.write_all(bytes)
    })
}

/// [`stage_file_for_atomic_publish_with`] with the write into the reserved
/// temp file injected, so a test can make it fail the way a full disk does
/// without filling one.
fn stage_file_for_atomic_publish_with_writer<S, W>(
    path: &Path,
    bytes: &[u8],
    mut suffix_for_attempt: S,
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
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
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

    if let Err(error) = write(&mut file, bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, path));
    }
    drop(file);
    Ok(tmp)
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
    let Err(error) = publish(tmp, path) else {
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
    if path.try_exists()? {
        return replace_file_windows(path, tmp);
    }

    match fs::rename(tmp, path) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            if path.try_exists()? {
                replace_file_windows(path, tmp)
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

/// `fs::rename` with both paths borrowed, the shape `recover_partial_replace`
/// takes.
#[cfg(windows)]
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

        let error = stage_file_for_atomic_publish_with(&destination, b"payload", |_| {
            tries += 1;
            OsString::from("taken")
        })
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
                    let staged =
                        stage_file_for_atomic_publish_with(&destination, bytes, |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        })?;
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
}
