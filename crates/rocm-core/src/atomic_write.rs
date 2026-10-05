// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Write a file so a reader sees either the old contents or the new ones,
//! never a truncated mix — including after a crash or power loss.
//!
//! The bytes go to a uniquely named sibling first, are flushed to disk, and
//! the sibling is then renamed over the destination (`ReplaceFileW` on
//! Windows, where a plain rename cannot replace an existing file). On Unix the
//! directory is flushed after the rename too, so the new name survives a
//! crash. A failure while writing — a full disk is the common one, reported
//! through [`crate::disk_space::map_write_error`] — leaves the destination
//! untouched and removes the temporary file.
//!
//! Replacing a file this way must not quietly change what the user set up:
//!
//! - on Unix the new file keeps the old one's permission bits (on Windows,
//!   `ReplaceFileW` keeps the replaced file's attributes and ACL itself);
//! - a destination that is a symbolic link — a dotfiles checkout linking
//!   `config.json`, say — has its *target* replaced, so the link survives.
//!
//! Use this for any file whose loss costs the user something: a plain
//! `fs::write` truncates first, so an interrupted write leaves an empty or
//! partial file behind.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::{disk_space, unix_time_millis};

const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

/// How many symbolic links [`write_file_atomically`] follows before giving up,
/// matching the usual kernel limit (`MAXSYMLINKS` on Linux).
const MAX_SYMLINK_HOPS: usize = 40;

/// A unique temp path next to `path`.
///
/// Keeps the full file name so a multi-extension artifact keeps its extensions
/// (`sdk.tar.gz` becomes `sdk.tar.gz.tmp-<id>`, where `with_extension` would
/// drop `.gz`).
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

/// Replace `path` with `bytes` atomically, creating its parent if needed.
pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    write_file_atomically_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        || {},
    )
}

/// [`write_file_atomically`] with the temp-name source and a hook that runs
/// between staging and publishing injected, so tests can force collisions and
/// observe the window before the rename.
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

/// [`write_file_atomically`] with the temp-name source, a hook between
/// staging and publishing, and the publish step itself injected.
///
/// A test seam, public only because the `rocm` binary's metadata-cache tests
/// drive their own commit through it to make the final rename fail or race.
/// Production code calls [`write_file_atomically`].
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
    let tmp = stage_file_for_atomic_publish_with(&path, bytes, suffix_for_attempt)?;
    before_publish();
    publish_staged_file_with(&tmp, &path, publish)
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

/// Write `bytes` to a fresh temp sibling of `path` and return it, for a caller
/// that must do more with the file than publish it.
pub fn stage_file_for_atomic_publish(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    stage_file_for_atomic_publish_with(path, bytes, |attempt| {
        OsString::from(format!("{temp_id}-{attempt}"))
    })
}

fn stage_file_for_atomic_publish_with<S>(
    path: &Path,
    bytes: &[u8],
    mut suffix_for_attempt: S,
) -> Result<PathBuf>
where
    S: FnMut(u32) -> OsString,
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

    // Flushed before it can be published: a rename that reaches the disk
    // before the data would leave an empty file under the real name after a
    // crash. A full disk can surface here rather than in `write_all`.
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, &tmp));
    }
    drop(file);
    if let Err(error) = keep_existing_permissions(path, &tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(tmp)
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

/// `ReplaceFileW` already carries the replaced file's attributes and ACL over
/// to the replacement, so there is nothing to copy by hand.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn keep_existing_permissions(_path: &Path, _tmp: &Path) -> Result<()> {
    Ok(())
}

/// Publish a staged file over `path`, removing the temp file if publishing
/// fails.
fn publish_staged_file_with<F>(tmp: &Path, path: &Path, publish: F) -> Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    publish(tmp, path)
        .inspect_err(|_| {
            let _ = fs::remove_file(tmp);
        })
        .with_context(|| format!("failed to publish {}", path.display()))?;
    sync_parent_dir(path);
    Ok(())
}

/// Flush the directory entry the rename just changed, so the new name — not
/// the old file, or no file — is what a crash leaves behind.
///
/// Best-effort: the file is already published, and some filesystems refuse to
/// sync a directory. Reporting failure here would claim the write failed when
/// it did not.
#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent()
        && let Ok(dir) = fs::File::open(parent)
    {
        let _ = dir.sync_all();
    }
}

/// Windows has no directory handle to flush; `ReplaceFileW` and `MoveFileExW`
/// update the directory through the journaling filesystem.
#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) {}

/// Rename `tmp` over `path`, replacing an existing file.
#[cfg(not(windows))]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    fs::rename(tmp, path)
}

/// Rename `tmp` over `path`, replacing an existing file.
#[cfg(windows)]
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
#[allow(unsafe_code)]
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
    // No flags: REPLACEFILE_WRITE_THROUGH is documented as unsupported, and the
    // replacement's data was already flushed with `sync_all` when it was staged.
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
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rocm-core-atomic-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn names_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
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

    #[cfg(unix)]
    #[test]
    fn temp_sibling_path_preserves_non_unicode_file_name_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let file_name = OsString::from_vec(b"sdk-\xff.tar.gz".to_vec());
        let destination = Path::new("/tmp").join(&file_name);
        let temp = temp_sibling_path(&destination, OsStr::new("collision")).unwrap();

        let mut expected = file_name.into_vec();
        expected.extend_from_slice(b".tmp-collision");
        assert_eq!(temp.file_name().unwrap().as_bytes(), expected);
    }

    #[test]
    fn concurrent_atomic_writes_do_not_remove_a_published_destination() {
        let root = scratch_dir("collision");
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
    fn concurrent_staged_publications_use_distinct_staging_files() {
        let root = scratch_dir("staged-collision");
        let destination = root.join("index.body");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|bytes| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    let staged = stage_file_for_atomic_publish(&destination, bytes)?;
                    before_publish.wait();
                    publish_staged_file_with(&staged, &destination, publish_temp_file)
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a writer must remain published");
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
        assert_eq!(names, ["index.body"], "staged files leaked");
    }

    #[test]
    fn failed_staged_publication_preserves_destination_and_cleans_staging_file() {
        let root = scratch_dir("staged-failure");
        let destination = root.join("index.body");
        fs::write(&destination, b"published").unwrap();
        let staged = stage_file_for_atomic_publish(&destination, b"replacement").unwrap();

        publish_staged_file_with(&staged, &destination, |_, _| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "simulated publication failure",
            ))
        })
        .expect_err("simulated publication failure must be returned");

        let published = fs::read(&destination).unwrap();
        let staged_left = staged.exists();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(published, b"published");
        assert!(!staged_left, "failed publication leaked its staging file");
    }

    #[test]
    fn failed_atomic_replace_preserves_destination_and_cleans_temp() {
        let root = scratch_dir("replace-failure");
        let destination = root.join("manifest.json");
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

        let published = fs::read(&destination).unwrap();
        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(published, b"published");
        assert_eq!(names, ["manifest.json"]);
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
        let root = scratch_dir("rename-failure");
        let occupied = root.join("artifact.tar.gz");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        // Fails after the temp file exists, so a leak would show its real name.
        write_file_atomically(&occupied, b"payload")
            .expect_err("renaming onto a non-empty directory should fail");

        let names = names_in(&root);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            names,
            ["artifact.tar.gz"],
            "failed write left temp files behind"
        );
    }

    /// Mirrors the `/dev/shm` reproduction from the original report: a genuine
    /// ENOSPC, not a rename failure standing in for one.
    ///
    /// Ignored by default because it fills `/dev/shm`, which is shared with
    /// anything else on the host, so it is not safe to run concurrently. Run
    /// with `cargo test -p rocm-core --lib -- --ignored atomic_write`.
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
            let error = write_file_atomically(&dest, &payload)
                .expect_err("writing past the end of the filesystem should fail");
            assert!(
                format!("{error:#}").contains("ran out of disk space"),
                "{error:#}"
            );
            failures.push(names_in(&dir));
        }
        let destination_exists = dest.exists();
        let _ = fs::remove_dir_all(&dir);

        for leftovers in &failures {
            assert!(
                leftovers.is_empty(),
                "failed write left files behind: {leftovers:?}"
            );
        }
        assert!(!destination_exists);
    }

    /// Replacing a file keeps its permission bits: a `0600` file must not be
    /// republished as `0644`, and an unusual mode survives as set.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_keeps_the_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let root = scratch_dir("permissions");
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
        // config/config.json -> config/hop.json -> ../dotfiles/rocm-config.json
        let hop = config_dir.join("hop.json");
        symlink(Path::new("../dotfiles/rocm-config.json"), &hop).unwrap();
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
        let dotfiles_names = names_in(&dotfiles);
        let _ = fs::remove_dir_all(&root);

        assert!(link_kept && hop_kept, "a link was replaced by a file");
        assert_eq!(through_link, b"new");
        assert_eq!(target_contents, b"new");
        assert_eq!(target_mode, 0o600);
        assert_eq!(config_names, ["config.json", "hop.json"]);
        assert_eq!(dotfiles_names, ["rocm-config.json"]);
    }

    /// A link loop is an error, not a hang, and writes nothing.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_refuses_a_symlink_loop() {
        use std::os::unix::fs::symlink;

        let root = scratch_dir("symlink-loop");
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
}
