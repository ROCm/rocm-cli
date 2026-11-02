// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::time::Duration;

/// Archive paths currently being validated, published, or extracted by this
/// process, with a condvar to wake waiters. Keyed per archive so an install of
/// one archive does not serialize behind an unrelated one — the download window
/// is minutes long, and the cross-process guarantee comes from the per-archive
/// file lock, not from this set.
static ARCHIVE_CACHE_BUSY: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
static ARCHIVE_CACHE_BUSY_SIGNAL: std::sync::Condvar = std::sync::Condvar::new();

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for verification", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        bail!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

fn publish_verified_archive(
    temporary: tempfile::NamedTempFile,
    archive: &Path,
    expected_sha256: &str,
) -> Result<()> {
    match temporary.persist_noclobber(archive) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            let publication_error = error.error;
            verify_sha256(archive, expected_sha256).with_context(|| {
                format!(
                    "concurrent archive publication at {} failed verification after publish conflict ({publication_error})",
                    archive.display()
                )
            })
        }
        Err(error) => Err(error.error)
            .with_context(|| format!("failed to publish verified archive {}", archive.display())),
    }
}

pub(crate) fn ensure_cached_archive<F>(
    url: &str,
    archive: &Path,
    expected_sha256: &str,
    download: F,
) -> Result<ArchiveCacheGuard>
where
    F: FnMut(&str, &Path) -> Result<()>,
{
    ensure_cached_archive_after_invalid(url, archive, expected_sha256, download, || {})
}

#[derive(Debug)]
struct ArchiveCacheLock {
    file: fs::File,
}

impl Drop for ArchiveCacheLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// In-process exclusion for one archive path.
#[derive(Debug)]
struct ArchiveCacheProcessLock {
    key: PathBuf,
}

impl Drop for ArchiveCacheProcessLock {
    fn drop(&mut self) {
        let mut busy = ARCHIVE_CACHE_BUSY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        busy.remove(&self.key);
        drop(busy);
        ARCHIVE_CACHE_BUSY_SIGNAL.notify_all();
    }
}

fn lock_archive_cache_in_process(archive: &Path) -> ArchiveCacheProcessLock {
    let key = archive.to_path_buf();
    let mut busy = ARCHIVE_CACHE_BUSY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if busy.contains(&key) {
        // The holder may be inside a download window minutes long; without this
        // the command sits silent and looks hung.
        eprintln!(
            "Waiting for another install in this process to finish with {}...",
            archive.display()
        );
    }
    while busy.contains(&key) {
        busy = ARCHIVE_CACHE_BUSY_SIGNAL
            .wait(busy)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    busy.insert(key.clone());
    ArchiveCacheProcessLock { key }
}

/// Exclusive access to one cached archive, held by the caller for as long as
/// the verified bytes are relied upon.
///
/// Verification and use have to happen under the same lock: releasing it at the
/// end of verification would leave a window in which a co-resident writer could
/// swap the archive before it is read again for extraction.
#[derive(Debug)]
pub(crate) struct ArchiveCacheGuard {
    _process: ArchiveCacheProcessLock,
    _file: ArchiveCacheLock,
}

fn archive_lock_path(archive: &Path) -> Result<PathBuf> {
    let parent = archive
        .parent()
        .with_context(|| format!("archive path has no parent: {}", archive.display()))?;
    let file_name = archive
        .file_name()
        .with_context(|| format!("archive path has no file name: {}", archive.display()))?;
    let mut lock_name = OsString::from(".");
    lock_name.push(file_name);
    lock_name.push(".lock");
    Ok(parent.join(lock_name))
}

#[cfg(test)]
fn mark_archive_lock_test_stage(stage: &str) {
    const ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    if let (Ok(role), Some(dir)) = (std::env::var(ROLE), std::env::var_os(DIR)) {
        fs::write(PathBuf::from(dir).join(format!("{role}-{stage}")), b"")
            .expect("write archive lock test marker");
    }
}

#[cfg(test)]
fn start_archive_lock_test_heartbeat() -> Option<std::thread::JoinHandle<()>> {
    const ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    if std::env::var(ROLE).as_deref() != Ok("recoverer") {
        return None;
    }
    let dir = std::env::var_os(DIR)?;
    Some(std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        fs::write(PathBuf::from(dir).join("recoverer-heartbeat"), b"")
            .expect("write archive lock test heartbeat");
    }))
}

fn lock_archive_cache(archive: &Path) -> Result<ArchiveCacheLock> {
    let lock_path = archive_lock_path(archive)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open archive cache lock {}", lock_path.display()))?;
    #[cfg(test)]
    mark_archive_lock_test_stage("before-lock");
    #[cfg(test)]
    let heartbeat = start_archive_lock_test_heartbeat();
    // `lock` blocks with no timeout, and the holder may be minutes into a
    // download, so announce the wait instead of leaving the command silent.
    // Probing first keeps the common uncontended case quiet.
    if file.try_lock().is_err() {
        eprintln!(
            "Waiting for another process to finish with {}...",
            archive.display()
        );
        file.lock().with_context(|| {
            format!(
                "failed to acquire archive cache lock {}",
                lock_path.display()
            )
        })?;
    }
    #[cfg(test)]
    {
        mark_archive_lock_test_stage("after-lock");
        if let Some(heartbeat) = heartbeat {
            heartbeat
                .join()
                .expect("archive lock test heartbeat panicked");
        }
    }
    Ok(ArchiveCacheLock { file })
}

fn ensure_cached_archive_after_invalid<F, I>(
    url: &str,
    archive: &Path,
    expected_sha256: &str,
    mut download: F,
    after_invalid: I,
) -> Result<ArchiveCacheGuard>
where
    F: FnMut(&str, &Path) -> Result<()>,
    I: FnOnce(),
{
    let parent = archive
        .parent()
        .with_context(|| format!("archive path has no parent: {}", archive.display()))?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create archive cache directory {}",
            parent.display()
        )
    })?;
    let process_guard = lock_archive_cache_in_process(archive);
    let archive_guard = lock_archive_cache(archive)?;
    let guard = ArchiveCacheGuard {
        _process: process_guard,
        _file: archive_guard,
    };

    match fs::symlink_metadata(archive) {
        Ok(metadata) if metadata.file_type().is_file() => {
            if verify_sha256(archive, expected_sha256).is_ok() {
                eprintln!("Using verified cached {}.", archive.display());
                return Ok(guard);
            }
            after_invalid();
            match fs::remove_file(archive) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to remove invalid {}", archive.display())
                    });
                }
            }
        }
        Ok(_) => bail!(
            "archive cache path is not a regular file: {}",
            archive.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let temporary = tempfile::Builder::new()
        .prefix(".lemonade-download-")
        .tempfile_in(parent)
        .with_context(|| {
            format!(
                // Owner-only permissions are a Unix guarantee: `tempfile`
                // creates with mode 0600 there, while on Windows the file
                // inherits the parent directory's DACL.
                "failed to create temporary download beside {}",
                archive.display()
            )
        })?;
    let temporary_path = temporary.path().to_path_buf();
    eprintln!("Downloading {}...", archive.display());
    download(url, &temporary_path)?;
    verify_sha256(&temporary_path, expected_sha256)?;
    publish_verified_archive(temporary, archive, expected_sha256)?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::io::Write;
    use std::process::{Command as ProcessCommand, Stdio};

    #[test]
    fn sha256_verification_accepts_matching_content_and_rejects_mismatch() {
        let dir = scratch_dir("archive-digest");
        let archive = dir.join("archive");
        fs::write(&archive, b"verified archive").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"verified archive"));

        verify_sha256(&archive, &expected).unwrap();
        let error = verify_sha256(&archive, &"0".repeat(64)).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repeated_valid_archive_publication_is_idempotent() {
        let dir = scratch_dir("valid-publication-race");
        let archive = dir.join("archive");
        let content = b"verified archive";
        let expected = format!("{:x}", Sha256::digest(content));

        for _ in 0..2 {
            let mut temporary = tempfile::NamedTempFile::new_in(&dir).unwrap();
            temporary.write_all(content).unwrap();
            verify_sha256(temporary.path(), &expected).unwrap();
            publish_verified_archive(temporary, &archive, &expected).unwrap();
        }

        assert_eq!(fs::read(&archive).unwrap(), content);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_publication_conflict_with_foreign_bytes_fails_closed() {
        let dir = scratch_dir("invalid-publication-race");
        let archive = dir.join("archive");
        let content = b"verified archive";
        let expected = format!("{:x}", Sha256::digest(content));
        let mut temporary = tempfile::NamedTempFile::new_in(&dir).unwrap();
        temporary.write_all(content).unwrap();
        verify_sha256(temporary.path(), &expected).unwrap();

        fs::write(&archive, b"unverified concurrent archive").unwrap();
        let error = publish_verified_archive(temporary, &archive, &expected).unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("concurrent archive publication"));
        assert!(error.contains("failed verification after publish conflict"));
        assert!(error.contains("SHA-256 mismatch"));
        assert_eq!(
            fs::read(&archive).unwrap(),
            b"unverified concurrent archive"
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poisoned_cached_archive_is_removed_and_redownloaded() {
        let dir = scratch_dir("poisoned-cache");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));
        let mut downloads = 0;

        ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
            downloads += 1;
            fs::write(destination, b"fresh archive")?;
            Ok(())
        })
        .unwrap();

        assert_eq!(downloads, 1);
        assert_eq!(fs::read(&archive).unwrap(), b"fresh archive");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_with_wrong_content_is_rejected_and_leaves_no_archive() {
        let dir = scratch_dir("bad-content-download");
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        // A download that succeeds at the transport level but returns the wrong
        // bytes must not be published, even though nothing errored.
        let error = ensure_cached_archive("test://archive", &archive, &expected, |_, path| {
            fs::write(path, b"attacker archive")?;
            Ok(())
        })
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(error.contains("SHA-256 mismatch"), "{error}");
        assert!(!archive.exists(), "unverified bytes were published");
        // Only the lock file may remain; the temporary download must be gone.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name != ".archive.lock")
            .collect();
        assert!(leftovers.is_empty(), "leftovers: {leftovers:?}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_guard_outlives_verification_and_is_scoped_to_one_archive() {
        let dir = scratch_dir("archive-guard-scope");
        let archive = dir.join("archive");
        let other = dir.join("other-archive");
        let content = b"fresh archive";
        let expected = format!("{:x}", Sha256::digest(content));

        // The guard returned by verification is what callers hold across
        // extraction; while it is alive the same archive must not be lockable.
        let guard = ensure_cached_archive("test://archive", &archive, &expected, |_, path| {
            fs::write(path, content)?;
            Ok(())
        })
        .unwrap();

        let blocked = std::thread::spawn(move || {
            let _second = lock_archive_cache_in_process(&archive);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !blocked.is_finished(),
            "a second in-process lock on the held archive was granted"
        );

        // A different archive must not queue behind it.
        let unrelated = std::thread::spawn(move || {
            let _other = lock_archive_cache_in_process(&other);
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !unrelated.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "an unrelated archive serialized behind the held one"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        unrelated.join().unwrap();

        drop(guard);
        blocked.join().unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poisoned_process_mutex_is_recovered_before_digest_validation() {
        let _ = std::thread::spawn(|| {
            let _guard = ARCHIVE_CACHE_BUSY.lock().unwrap();
            panic!("poison archive cache process mutex");
        })
        .join();
        let dir = scratch_dir("poisoned-process-mutex");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
            fs::write(destination, b"fresh archive")?;
            Ok(())
        })
        .unwrap();

        verify_sha256(&archive, &expected).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    const ARCHIVE_LOCK_TEST_ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const ARCHIVE_LOCK_TEST_DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    const ARCHIVE_LOCK_TEST_TIMEOUT: Duration = Duration::from_secs(10);

    struct ChildGuard(Option<std::process::Child>);

    impl ChildGuard {
        fn spawn(test_exe: &Path, dir: &Path, role: &str) -> Self {
            let child = ProcessCommand::new(test_exe)
                .args([
                    "--exact",
                    "archive_cache::tests::archive_cache_subprocess_helper",
                    "--nocapture",
                ])
                .env(ARCHIVE_LOCK_TEST_ROLE, role)
                .env(ARCHIVE_LOCK_TEST_DIR, dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            Self(Some(child))
        }

        fn is_running(&mut self) -> bool {
            self.0.as_mut().unwrap().try_wait().unwrap().is_none()
        }

        fn wait_until_exit(&mut self, timeout: Duration) -> std::process::ExitStatus {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                if let Some(status) = self.0.as_mut().unwrap().try_wait().unwrap() {
                    self.0.take();
                    return status;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "timed out waiting for child process {}",
                    self.0.as_ref().unwrap().id()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                if child.try_wait().ok().flatten().is_none() {
                    let _ = child.kill();
                }
                let _ = child.wait();
            }
        }
    }

    #[test]
    fn archive_cache_subprocess_helper() {
        let Ok(role) = std::env::var(ARCHIVE_LOCK_TEST_ROLE) else {
            return;
        };
        let dir = PathBuf::from(std::env::var_os(ARCHIVE_LOCK_TEST_DIR).unwrap());
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        ensure_cached_archive_after_invalid(
            "test://archive",
            &archive,
            &expected,
            |_, destination| {
                fs::write(destination, b"fresh archive")?;
                Ok(())
            },
            || {
                if role == "validator" {
                    fs::write(dir.join("validator-paused"), b"").unwrap();
                    let deadline = std::time::Instant::now() + ARCHIVE_LOCK_TEST_TIMEOUT;
                    while !dir.join("release-validator").is_file() {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "timed out waiting to release validator"
                        );
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            },
        )
        .unwrap();
        verify_sha256(&archive, &expected).unwrap();
    }

    fn wait_for_test_file(path: &Path, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while !path.is_file() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn interprocess_lock_serializes_poisoned_validation_and_recovery() {
        let dir = scratch_dir("interprocess-archive-lock");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let test_exe = std::env::current_exe().unwrap();

        let mut validator = ChildGuard::spawn(&test_exe, &dir, "validator");
        wait_for_test_file(&dir.join("validator-paused"), ARCHIVE_LOCK_TEST_TIMEOUT);
        let mut recoverer = ChildGuard::spawn(&test_exe, &dir, "recoverer");
        wait_for_test_file(
            &dir.join("recoverer-before-lock"),
            ARCHIVE_LOCK_TEST_TIMEOUT,
        );
        wait_for_test_file(&dir.join("recoverer-heartbeat"), ARCHIVE_LOCK_TEST_TIMEOUT);
        assert!(
            !dir.join("recoverer-after-lock").is_file(),
            "recoverer acquired the inter-process lock before release"
        );
        assert!(
            recoverer.is_running(),
            "recoverer exited while waiting for the inter-process lock"
        );

        fs::write(dir.join("release-validator"), b"").unwrap();
        assert!(
            validator
                .wait_until_exit(ARCHIVE_LOCK_TEST_TIMEOUT)
                .success(),
            "validator subprocess failed"
        );
        assert!(
            recoverer
                .wait_until_exit(ARCHIVE_LOCK_TEST_TIMEOUT)
                .success(),
            "recoverer subprocess failed"
        );
        assert!(dir.join("recoverer-after-lock").is_file());
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));
        verify_sha256(&archive, &expected).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn failed_download_removes_private_partial_file() {
        let dir = scratch_dir("partial-download");
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"complete"));

        let error =
            ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
                fs::write(destination, b"partial")?;
                bail!("download interrupted")
            })
            .unwrap_err();

        assert!(error.to_string().contains("download interrupted"));
        assert!(!archive.exists());
        assert_eq!(
            fs::read_dir(&dir)
                .unwrap()
                .filter(
                    |entry| entry.as_ref().unwrap().path() != archive_lock_path(&archive).unwrap()
                )
                .count(),
            0
        );
        fs::remove_dir_all(&dir).ok();
    }
}
