// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Writing a stand-in executable that this same process is about to run.
//!
//! Tests across the workspace plant fake `python`, `uv`, `amd-smi`, ... shell
//! scripts and then run them, while other tests on parallel threads of the
//! same process are spawning children of their own. Writing such a script with
//! `std::fs::write` and then executing it fails intermittently with
//! `Text file busy (os error 26)` (`ETXTBSY`) on Linux:
//!
//! 1. thread A opens the script for writing (the fd is `O_CLOEXEC`);
//! 2. thread B spawns a child, which starts with a copy of every open fd —
//!    including A's write fd — and keeps it until its own `exec`;
//! 3. thread A closes its fd and runs the script, but the kernel refuses to
//!    execute a file that any process still has open for writing, and B's
//!    child has not reached `exec` yet.
//!
//! `O_CLOEXEC` only closes the copy at `exec`, and Linux has no close-on-fork,
//! so nothing a writer in this process can do with its own fd closes the
//! window. [`write_fake_executable`] therefore never opens the file for
//! writing in this process at all: a short-lived single-threaded child writes
//! it, and that child has exited — taking the only write fd with it — before
//! this function returns.
//!
//! The module is plain `pub`, not `#[cfg(test)]` and not behind a feature
//! such as `e2e-test-hooks`, because every caller is test code in *another*
//! crate (`apps/rocm`, the engines, `rocm-dash-daemon`, `xtask`, the e2e
//! harness): those crates cannot see rocm-core's `cfg(test)` items, and a
//! feature gate would have to be switched on in each consumer's
//! `[dev-dependencies]` entry for rocm-core.

use std::io;
use std::path::Path;

/// Write `contents` to `path` and mark it executable (`0o755`).
///
/// This process never holds a write fd on `path`, so a concurrent spawn on
/// another thread cannot make the first run of it fail with `ETXTBSY`. See
/// the module docs for the race.
///
/// Meant for tests that plant a fake binary and then run it. Replaces any
/// existing file at `path`; does not create parent directories.
///
/// # Errors
///
/// Fails when the writer cannot be spawned, cannot create `path`, or the
/// permissions cannot be set.
#[cfg(unix)]
pub fn write_fake_executable(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt as _;
    use std::process::{Command, Stdio};

    // The contents travel over a pipe. Our end of it can leak into another
    // thread's child exactly like a file fd can, but that only delays EOF for
    // `cat` until that child execs; it never makes `path` busy.
    let mut writer = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#"exec cat > "$1""#)
        .arg("sh")
        .arg(path)
        // Absolute, because some tests narrow `PATH` for the code under test.
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| step_error(path, "spawn the writer", &error))?;
    let write_result = writer
        .stdin
        .take()
        .expect("stdin was requested as a pipe")
        .write_all(contents.as_ref());
    let output = writer
        .wait_with_output()
        .map_err(|error| step_error(path, "wait for the writer", &error))?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "failed to write {}: writer exited with {}: {}",
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    write_result.map_err(|error| step_error(path, "send the contents", &error))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|error| step_error(path, "mark it executable", &error))
}

/// Name the path and the step that failed, keeping the original error kind,
/// so a caller's one `.expect("plant the fake …")` still says whether the
/// write or the `chmod` went wrong.
#[cfg(unix)]
fn step_error(path: &Path, step: &str, error: &io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("failed to write {}: {step}: {error}", path.display()),
    )
}

/// Off unix there is no `ETXTBSY` to avoid, and no mode bit to set: a plain
/// write is enough.
///
/// # Errors
///
/// Fails when `path` cannot be written.
#[cfg(not(unix))]
pub fn write_fake_executable(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    std::fs::write(path, contents)
}

#[cfg(all(test, unix))]
mod tests {
    use super::write_fake_executable;
    use std::os::unix::fs::PermissionsExt as _;
    use std::process::Command;

    #[test]
    fn writes_the_contents_and_marks_the_file_executable() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("fake");
        write_fake_executable(&path, "#!/bin/sh\necho first\n").expect("first write");
        // A second write replaces the first, as `std::fs::write` would.
        write_fake_executable(&path, "#!/bin/sh\necho second\n").expect("rewrite");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "#!/bin/sh\necho second\n"
        );
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o755);
        let output = Command::new(&path).output().expect("run the fake");
        assert_eq!(output.stdout, b"second\n");
    }

    #[test]
    fn reports_a_path_it_cannot_create() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("missing-parent").join("fake");
        let error = write_fake_executable(&path, "#!/bin/sh\n").expect_err("no parent dir");
        assert!(
            error.to_string().contains(&path.display().to_string()),
            "the error must name the path: {error}"
        );
    }

    /// The write succeeds but the `chmod` cannot: `path` is a symlink to
    /// `/dev/null`, which an unprivileged user may write but not chmod. The
    /// error must say which step failed, not just pass the bare `EPERM` up.
    #[test]
    fn a_failed_chmod_names_the_path_and_the_step() {
        if crate::openmpi::running_as_root() {
            // Root may chmod `/dev/null`; there is no failure to observe.
            return;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("fake");
        std::os::unix::fs::symlink("/dev/null", &path).expect("symlink to /dev/null");
        let error = write_fake_executable(&path, "#!/bin/sh\n").expect_err("chmod /dev/null");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        let message = error.to_string();
        assert!(message.contains(&path.display().to_string()), "{message}");
        assert!(message.contains("mark it executable"), "{message}");
    }

    /// The regression this helper exists for: many threads each plant a fake
    /// and run it straight away, so every run races the other threads' spawns.
    /// Writing the file in-process (`std::fs::write` + `set_permissions`) fails
    /// this with `Text file busy` on Linux.
    #[test]
    fn a_fake_runs_first_time_while_other_threads_spawn() {
        const THREADS: usize = 16;
        const PER_THREAD: usize = 40;
        let dir = tempfile::tempdir().expect("temp dir");
        let start = std::sync::Barrier::new(THREADS);
        let failures: Vec<String> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..THREADS)
                .map(|thread| {
                    let (dir, start) = (dir.path(), &start);
                    scope.spawn(move || {
                        start.wait();
                        let mut failures = Vec::new();
                        for run in 0..PER_THREAD {
                            let path = dir.join(format!("fake-{thread}-{run}"));
                            write_fake_executable(&path, "#!/bin/sh\necho ok\n")
                                .expect("write the fake");
                            match Command::new(&path).output() {
                                Ok(output) if output.stdout == b"ok\n" => {}
                                Ok(output) => failures.push(format!("{output:?}")),
                                Err(error) => failures.push(error.to_string()),
                            }
                        }
                        failures
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("a worker panicked"))
                .collect()
        });
        assert!(
            failures.is_empty(),
            "{} of {} first runs failed: {:?}",
            failures.len(),
            THREADS * PER_THREAD,
            &failures[..failures.len().min(5)]
        );
    }
}
