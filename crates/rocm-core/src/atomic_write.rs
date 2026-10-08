// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Crash- and race-safe file replacement.
//!
//! [`write_file_atomically`] stages the bytes in a fresh sibling temp file and
//! then publishes it over the destination in one step, so a reader sees either
//! the previous complete file or the new complete file, never a partial one,
//! and the destination is never absent while it is being replaced.
//!
//! The temp file is named `{file name}.tmp-{pid}-{millis}-{attempt}` and
//! reserved with `create_new`, so two writers — threads of one process or
//! separate processes — can never write through the same temp file, even when
//! their pid and millisecond collide; a collision just moves on to the next
//! attempt. A failed write or publish removes its temp file. On Windows the
//! publish uses `ReplaceFileW` when the destination already exists, because
//! `rename` onto an existing file is not a replace there.
//!
//! This makes each file individually atomic. It does not serialize a
//! read-modify-write across processes: two writers that each read, change and
//! write the same file are still last-writer-wins.

use crate::disk_space;
use crate::unix_time_millis;
use anyhow::{Context, Result, bail};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

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

/// [`write_file_atomically`] with the temp-name suffix for each attempt and a
/// hook that runs after the temp file is fully written and before it is
/// published. Both exist so tests can force a temp-name collision and hold
/// writers at the publish point; production code calls
/// [`write_file_atomically`].
pub fn write_file_atomically_with<S, P>(
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

/// [`write_file_atomically_with`] with the publish step injected as well, so
/// tests can observe the staged file and the destination at the moment of
/// publication, or make the publication fail. Pass [`publish_temp_file`] for
/// the real one.
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
/// staged file's path without publishing it. The caller owns the staged file:
/// it either publishes it or removes it.
pub fn stage_file_for_atomic_publish(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = process_temp_id();
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

    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, &tmp));
    }
    drop(file);
    Ok(tmp)
}

#[cfg(test)]
fn publish_staged_file(tmp: &Path, path: &Path) -> Result<()> {
    publish_staged_file_with(tmp, path, publish_temp_file)
}

fn publish_staged_file_with<F>(tmp: &Path, path: &Path, publish: F) -> Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    publish(tmp, path)
        .inspect_err(|_| {
            let _ = fs::remove_file(tmp);
        })
        .with_context(|| format!("failed to publish {}", path.display()))
}

/// Move the staged file `tmp` over `path` in one step, replacing any existing
/// file without first removing it.
#[cfg(not(windows))]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    fs::rename(tmp, path)
}

/// Move the staged file `tmp` over `path` in one step, replacing any existing
/// file without first removing it.
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
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
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

    #[test]
    fn write_file_atomically_reports_a_full_disk_clearly() {
        // Exercise the mapping the write path uses, without filling a disk.
        let error = disk_space::map_write_error(
            std::io::Error::from(std::io::ErrorKind::StorageFull),
            Path::new("/cache/rocm.tar.gz.tmp"),
        );
        let text = format!("{error:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
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

    /// Same property end to end: the temp file the real write creates (and
    /// then cleans up) leaves only the destination behind.
    #[test]
    fn write_file_atomically_temp_name_preserves_multi_dot_file_names() {
        let root = scratch_dir("atomic-name");
        let occupied = root.join("artifact.tar.gz");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        // Fails after the temp file exists, so the observed name is the real one.
        write_file_atomically(&occupied, b"payload").expect_err("rename should fail");

        let names: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            names,
            vec!["artifact.tar.gz".to_owned()],
            "only the occupied destination should remain"
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
            .map(|bytes| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    let staged = stage_file_for_atomic_publish(&destination, bytes)?;
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
