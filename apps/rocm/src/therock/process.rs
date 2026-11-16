// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Subprocess and progress-output helpers behind TheRock installs: running
//! and capturing child commands (with the Windows temp-file variants),
//! bounded waits, the progress-line output with its `--json` suppression
//! guard, and unpacking the SDK tarball.
//!
//! Depends on the sibling `io` module for scratch directories; nothing in
//! `io` depends on this one.

use super::io::{linux_temp_dir, windows_temp_dir};
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, disk_space, runtime_is_windows, runtime_path_for_windows_child, uv_command_env,
};
use std::cell::Cell;
use std::fs;
use std::io::Read;
#[cfg(not(test))]
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

thread_local! {
    /// Set while rendering `--json` output, whose "single compact JSON line"
    /// contract [`progress_line`] and the managed-Python installer would
    /// otherwise break by writing extra lines to stdout ahead of the JSON.
    static SUPPRESS_PROGRESS_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that silences [`progress_line`] and redirects managed-Python
/// installer output away from stdout for its lifetime, restoring the prior
/// state on drop (so nested callers compose correctly).
pub(super) struct SuppressProgressOutput {
    previous: bool,
}

impl SuppressProgressOutput {
    pub(super) fn new() -> Self {
        let previous = SUPPRESS_PROGRESS_OUTPUT.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for SuppressProgressOutput {
    fn drop(&mut self) {
        let previous = self.previous;
        SUPPRESS_PROGRESS_OUTPUT.with(|flag| flag.set(previous));
    }
}

pub(super) fn progress_output_suppressed() -> bool {
    SUPPRESS_PROGRESS_OUTPUT.with(Cell::get)
}

pub(super) fn progress_line(message: impl AsRef<str>) {
    if progress_output_suppressed() {
        return;
    }
    emit_progress_line(message.as_ref());
}

#[cfg(not(test))]
fn emit_progress_line(message: &str) {
    println!("{message}");
    let _ = std::io::stdout().flush();
}

// In tests, write through a thread-local buffer instead of stdout so a test can
// assert on `progress_line`'s actual output (in particular, that a
// `SuppressProgressOutput` guard held around a call really does silence it),
// rather than only on the suppression flag's own bookkeeping.
#[cfg(test)]
thread_local! {
    pub(super) static PROGRESS_LINE_SINK: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn emit_progress_line(message: &str) {
    PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().push(message.to_owned()));
}

fn capture_command_output(program: &Path, args: &[&str]) -> Result<Output> {
    if runtime_is_windows() {
        return capture_command_output_with_temp_files(program, args);
    }
    Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to launch {}", program.display()))
}

fn capture_command_output_with_temp_files(program: &Path, args: &[&str]) -> Result<Output> {
    let temp_dir = windows_temp_dir("rocm-cli-command")?;
    let stdout_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    let stdout = fs::read(&stdout_path).unwrap_or_default();
    let stderr = fs::read(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// Run a probe script with extra environment set and return its stdout.
///
/// The script goes to a temp file rather than `python -c`, because the loader
/// search path this exists to set must reach the child through its environment,
/// and a file keeps the invocation identical on every platform.
pub(super) fn capture_python_stdout_with_env(
    python_executable: &Path,
    script: &str,
    env: &[(String, String)],
    context_text: &str,
) -> Result<String> {
    let temp_root = if runtime_is_windows() {
        windows_temp_dir("rocm-cli-python-probe")?
    } else {
        linux_temp_dir("rocm-cli-python-probe")?
    };
    let script_path = temp_root.join("probe.py");
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write {}", script_path.display()))?;

    let mut command = Command::new(python_executable);
    command.arg(&script_path);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch {}", python_executable.display()));
    let _ = fs::remove_dir_all(&temp_root);
    let output = output?;

    if !output.status.success() {
        bail!(
            "{context_text}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("{context_text}: failed to decode Python output"))
}

pub(super) fn capture_python_stdout(
    python_executable: &Path,
    script: &str,
    context_text: &str,
) -> Result<String> {
    if !runtime_is_windows() {
        let output = capture_command_output(python_executable, &["-c", script])?;
        if !output.status.success() {
            bail!(
                "{context_text}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        return String::from_utf8(output.stdout)
            .with_context(|| format!("{context_text}: failed to decode Python output"));
    }

    let temp_dir = windows_temp_dir("rocm-cli-python")?;
    let script_path = temp_dir.join("probe.py");
    let wrapper_path = temp_dir.join("wrapper.py");
    let output_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write {}", script_path.display()))?;
    fs::write(
        &wrapper_path,
        r#"import contextlib
import pathlib
import runpy
import sys

out = pathlib.Path(sys.argv[1])
script = pathlib.Path(sys.argv[2])
with out.open("w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        runpy.run_path(str(script), run_name="__main__")
"#,
    )
    .with_context(|| format!("failed to write {}", wrapper_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = Command::new(python_executable)
        .arg(windows_child_path(&wrapper_path))
        .arg(windows_child_path(&output_path))
        .arg(windows_child_path(&script_path))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch {}", python_executable.display()))?;
    let text = fs::read_to_string(&output_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(&temp_dir);
    if status.success() {
        Ok(text)
    } else {
        let stderr = stderr.trim().to_owned();
        let detail = if stderr.is_empty() {
            format!("command exited with status {status}")
        } else {
            stderr
        };
        bail!("{context_text}: {detail}")
    }
}

fn windows_child_path(path: &Path) -> String {
    runtime_path_for_windows_child(path)
}

pub(super) fn run_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    if runtime_is_windows() {
        let status = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to launch {}", program.display()))?;
        if status.success() {
            return Ok(());
        }
        bail!("{context_text}: command exited with status {status}");
    }

    let output = capture_command_output(program, args)?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("command exited with status {}", output.status)
    };
    bail!("{context_text}: {detail}")
}

#[allow(dead_code)]
fn run_progress_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: command exited with status {status}");
}

pub(super) fn run_command_with_env(
    program: &Path,
    args: &[&str],
    env: &[(String, String)],
    context_text: &str,
) -> Result<()> {
    let mut command = Command::new(program);
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let detail = if stderr.is_empty() {
        format!("command exited with status {}", output.status)
    } else {
        stderr
    };
    bail!("{context_text}: {detail}")
}

pub(super) fn run_uv_progress_command(
    paths: &AppPaths,
    uv: &Path,
    args: &[&str],
    context_text: &str,
) -> Result<()> {
    let mut command = Command::new(uv);
    command.args(args);
    for (key, value) in &uv_command_env(paths) {
        command.env(key, value);
    }
    let status = command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| "failed to launch uv".to_string())?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: uv exited with status {status}");
}

/// Waits for `child` to exit, killing it and failing once `timeout` elapses.
///
/// `None` waits unbounded, matching an explicit user-invoked install with no
/// budget to respect. Reads stdout/stderr on background threads throughout the
/// wait so a slow or silent child can't deadlock the poll on a full pipe.
pub(super) fn wait_with_output_bounded(
    mut child: Child,
    timeout: Option<Duration>,
) -> Result<Output> {
    let Some(timeout) = timeout else {
        return child
            .wait_with_output()
            .context("failed to wait for child process");
    };
    let stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });
    let stderr_reader = child.stderr.take().map(|mut stderr| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stderr.read_to_end(&mut bytes);
            bytes
        })
    });
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to poll child process status")?
        {
            let stdout = stdout_reader
                .map(|reader| reader.join().unwrap_or_default())
                .unwrap_or_default();
            let stderr = stderr_reader
                .map(|reader| reader.join().unwrap_or_default())
                .unwrap_or_default();
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "timed out after {}s waiting for child process",
                timeout.as_secs()
            );
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn extract_tarball(archive_path: &Path, target_dir: &Path) -> Result<()> {
    run_command(
        Path::new("tar"),
        &[
            "-xf",
            archive_path.to_string_lossy().as_ref(),
            "-C",
            target_dir.to_string_lossy().as_ref(),
        ],
        "extract TheRock tarball artifact",
    )
    .map_err(|error| {
        // The extraction preflight only warns, because the extracted size is an
        // estimate. When that warning turns out to be right, the failure arrives
        // as `tar` stderr rather than an `io::Error`, so it never reaches
        // `map_write_error` — without this the user gets the raw
        // "tar: ...: No space left on device" this feature exists to replace.
        disk_space::subprocess_full_disk_error(&format!("{error:#}"), target_dir).unwrap_or(error)
    })
}

/// Unpack the SDK archive and then delete it.
///
/// Only the extracted tree is used from here on, so keeping the archive would
/// double the disk cost of every installed version. This mirrors the cleanup
/// `ensure_uv_binary` already performs after unpacking its own download.
///
/// Removing the archive is best-effort: the install has already succeeded by
/// this point, so a cleanup failure is non-fatal. The message is returned
/// rather than printed directly, so callers running a progress spinner over
/// this call can drop it first and avoid interleaving spinner frames with
/// the report.
pub(super) fn extract_tarball_and_discard_archive(
    archive_path: &Path,
    target_dir: &Path,
) -> Result<Option<String>> {
    extract_tarball(archive_path, target_dir)?;
    if let Err(error) = fs::remove_file(archive_path) {
        return Ok(Some(format!(
            "Could not remove the downloaded archive {}: {error}",
            archive_path.display()
        )));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use std::path::PathBuf;

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("therock")
            .join(format!(
                "rocm-cli-process-test-{name}-{}",
                crate::test_support::unique_suffix()
            ));
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    #[test]
    fn suppress_progress_output_contract() {
        // `SUPPRESS_PROGRESS_OUTPUT` is thread-local, so this doesn't race with
        // other tests' threads.
        assert!(!progress_output_suppressed());
        {
            let _outer = SuppressProgressOutput::new();
            assert!(progress_output_suppressed());
            {
                let _inner = SuppressProgressOutput::new();
                assert!(
                    progress_output_suppressed(),
                    "a nested guard must still suppress"
                );
            }
            assert!(
                progress_output_suppressed(),
                "dropping the inner guard must not lift the outer guard's suppression"
            );
        }
        assert!(
            !progress_output_suppressed(),
            "dropping the outer guard must restore the pre-guard state"
        );
    }

    #[test]
    fn progress_line_is_suppressed_while_guard_is_held() {
        // Unlike `suppress_progress_output_contract` (which only checks the
        // flag `SuppressProgressOutput` flips), this asserts on `progress_line`'s
        // actual output via the `#[cfg(test)]` sink, so it fails if the guard
        // is ever wired up but `progress_line` stops consulting it.
        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        progress_line("before guard");
        {
            let _guard = SuppressProgressOutput::new();
            progress_line("during guard");
        }
        progress_line("after guard");
        let captured = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
        assert_eq!(
            captured,
            vec!["before guard".to_owned(), "after guard".to_owned()]
        );
    }

    /// Takes the lock that serializes tests which replace process-wide
    /// environment variables. The program is started by name, so it is found
    /// through `PATH`, and another test replaces `PATH` for the duration of its
    /// body; a spawn that lands in that window fails with "program not found"
    /// and the test that wanted a sleeping child panics.
    fn lock_process_env_for_spawn() -> std::sync::MutexGuard<'static, ()> {
        PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(not(windows))]
    fn spawn_test_sleep(seconds: u64) -> std::process::Child {
        let _env = lock_process_env_for_spawn();
        Command::new("sleep")
            .arg(seconds.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep")
    }

    // `timeout.exe` refuses to run at all with its stdin redirected (it demands
    // a real console even with `/nobreak`), exiting immediately instead of
    // sleeping — which used to make both tests below pass or fail for the
    // wrong reason. `Start-Sleep` has no such requirement.
    #[cfg(windows)]
    fn spawn_test_sleep(seconds: u64) -> std::process::Child {
        let _env = lock_process_env_for_spawn();
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("Start-Sleep -Seconds {seconds}"),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn Start-Sleep")
    }

    /// The sleeping child is found through `PATH`, so starting it while another
    /// test has replaced `PATH` must wait for that test rather than fail. This
    /// holds the environment lock with `PATH` emptied, starts the child from
    /// another thread, and only then puts `PATH` back: without the lock in
    /// `spawn_test_sleep` the spawn runs at once and panics.
    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn spawning_the_test_sleep_waits_while_another_test_has_replaced_path() {
        let holder = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let old_path = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("PATH", "");
        }
        let spawner = std::thread::spawn(|| spawn_test_sleep(30));
        std::thread::sleep(Duration::from_millis(300));
        unsafe {
            match old_path {
                Some(value) => std::env::set_var("PATH", value),
                None => std::env::remove_var("PATH"),
            }
        }
        drop(holder);

        let mut child = spawner
            .join()
            .expect("the spawn must wait for the lock instead of failing while PATH is replaced");
        child.kill().ok();
        child.wait().ok();
    }

    /// A budget-constrained caller (the startup update check) must get its
    /// child back, not blocked past the budget it asked for.
    #[test]
    fn wait_with_output_bounded_kills_a_slow_child_at_the_deadline() {
        let child = spawn_test_sleep(30);
        let started = Instant::now();

        let error = wait_with_output_bounded(child, Some(Duration::from_millis(200)))
            .expect_err("a 30s sleep must not complete inside a 200ms budget");

        assert!(
            started.elapsed() < Duration::from_secs(5),
            "kill was not prompt"
        );
        assert!(error.to_string().contains("timed out"), "{error}");
    }

    /// An unbounded wait (an explicit user-invoked install, not a budgeted
    /// startup check) still returns the child's real output.
    #[test]
    fn wait_with_output_bounded_waits_unbounded_when_no_timeout_is_given() {
        let child = spawn_test_sleep(1);

        let output = wait_with_output_bounded(child, None).expect("child should exit");

        assert!(output.status.success(), "{output:?}");
    }

    /// The downloaded archive is removed once it has been unpacked; keeping it
    /// would double the disk cost of every installed SDK version.
    #[test]
    fn extracting_the_sdk_archive_removes_it() -> Result<()> {
        // Extraction spawns bare-name `tar` via `PATH`; hold the shared env lock so a
        // concurrent test that temporarily narrows `PATH` cannot make it fail to launch.
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, _paths) = test_paths("discard-archive");
        let cache = root.join("cache");
        let payload_dir = root.join("payload");
        fs::create_dir_all(&cache)?;
        fs::create_dir_all(&payload_dir)?;
        fs::write(payload_dir.join("marker.txt"), b"sdk")?;

        let archive = cache.join("therock-sdk.tar.gz");
        let tar = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&payload_dir)
            .arg("marker.txt")
            .status()?;
        if !tar.success() {
            eprintln!("skipping: tar unavailable on this host");
            let _ = fs::remove_dir_all(&root);
            return Ok(());
        }
        assert!(archive.is_file(), "archive fixture should exist");

        let target = root.join("install");
        fs::create_dir_all(&target)?;
        let cleanup_warning = extract_tarball_and_discard_archive(&archive, &target)?;
        assert!(
            cleanup_warning.is_none(),
            "archive cleanup should succeed: {cleanup_warning:?}"
        );

        assert!(
            target.join("marker.txt").is_file(),
            "the archive contents should have been extracted"
        );
        assert!(
            !archive.exists(),
            "the archive should be removed once unpacked, found {}",
            archive.display()
        );

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// If the archive can't be removed after a successful extraction, the
    /// extraction result still succeeds and callers receive a warning message
    /// describing the cleanup failure instead of a raised error.
    #[cfg(unix)]
    #[test]
    fn extracting_the_sdk_archive_reports_cleanup_failure() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, _paths) = test_paths("discard-archive-cleanup-failure");
        let cache = root.join("cache");
        let payload_dir = root.join("payload");
        fs::create_dir_all(&cache)?;
        fs::create_dir_all(&payload_dir)?;
        fs::write(payload_dir.join("marker.txt"), b"sdk")?;

        let archive = cache.join("therock-sdk.tar.gz");
        let tar = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&payload_dir)
            .arg("marker.txt")
            .status()?;
        if !tar.success() {
            eprintln!("skipping: tar unavailable on this host");
            let _ = fs::remove_dir_all(&root);
            return Ok(());
        }

        // Root can unlink a file regardless of its parent directory's write
        // permission bit, so the `chmod 0o555` below would not actually block
        // the removal and `cleanup_warning` would come back `None`, failing
        // the `.expect(...)` below on a mismatched assumption rather than the
        // behavior under test.
        #[allow(unsafe_code)] // libc FFI
        let euid = unsafe { libc::geteuid() };
        if euid == 0 {
            eprintln!("skipping: test requires a non-root user to enforce permissions");
            let _ = fs::remove_dir_all(&root);
            return Ok(());
        }

        let target = root.join("install");
        fs::create_dir_all(&target)?;

        // Removing the archive requires write access to its parent directory;
        // strip that so `fs::remove_file` fails after a successful extraction.
        let cache_perms = fs::metadata(&cache)?.permissions();
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o555))?;
        let result = extract_tarball_and_discard_archive(&archive, &target);
        fs::set_permissions(&cache, cache_perms)?;

        let cleanup_warning = result?;
        assert!(
            target.join("marker.txt").is_file(),
            "the archive contents should still be extracted"
        );
        assert!(
            archive.is_file(),
            "archive removal should have failed, leaving it in place"
        );
        let message = cleanup_warning.expect("a cleanup failure should produce a warning message");
        assert!(
            message.contains("Could not remove the downloaded archive"),
            "{message}"
        );

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn windows_child_path_maps_ape_drive_paths() {
        assert_eq!(
            windows_child_path(Path::new("/D/path/to/rocm-cli/file.ps1")),
            r"D:\path\to\rocm-cli\file.ps1"
        );
        assert_eq!(windows_child_path(Path::new("/c")), r"C:\");
    }
}

/// Property tests for the SDK tarball unpack, judged on the filesystem by
/// `crate::archive_props`. Extraction is the system `tar` resolved from
/// `PATH` (GNU tar on the Linux CI runners), so these pin what that `tar`
/// guarantees through `extract_tarball_and_discard_archive`.
#[cfg(all(test, unix))]
mod archive_properties {
    use super::extract_tarball_and_discard_archive;
    use crate::archive_props::{
        Expect, Modes, entries, note_mode_scope, run_case, run_property, run_round_trip,
        tar_gz_bytes,
    };
    use crate::test_support::PROCESS_ENV_TEST_LOCK;

    /// No escape, and no setuid/setgid or world-writable entry. The modes are
    /// what the system `tar` gives an unprivileged user, so they are not
    /// judged when this runs as root (see `Modes`).
    #[test]
    fn sdk_tarball_unpack_never_writes_outside_the_install_root() {
        // Every generated case below extracts through `extract_tarball_and_
        // discard_archive`, which spawns bare-name `tar` via `PATH`; hold the
        // lock for the whole property run so a concurrent test that narrows
        // `PATH` cannot make any case fail to launch.
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let expect = Expect {
            modes: Modes::SafeUnlessRoot,
            no_outward_links: false,
        };
        note_mode_scope("therock", expect.modes);
        run_property("therock", 256, entries(), Some(5), |layout, entries| {
            run_case(
                layout,
                "sdk.tar.gz",
                &tar_gz_bytes(entries),
                &[&layout.dest],
                &layout.dest,
                |layout, archive| {
                    extract_tarball_and_discard_archive(archive, &layout.dest).map(|_| ())
                },
            )
            .judge(expect)
        })
        .unwrap();
    }

    /// The other direction: a well-formed tarball is extracted in full, into
    /// the install root it was given. An unpack that wrote somewhere else
    /// entirely escapes the oracle above; this catches it.
    #[test]
    fn sdk_tarball_unpack_extracts_a_benign_tarball_into_the_install_root() {
        // Same bare-name `tar` spawn as above; same reason to hold the lock.
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        run_round_trip("therock-benign", |layout, entries| {
            let archive = layout.base.join("sdk.tar.gz");
            std::fs::write(&archive, tar_gz_bytes(entries))?;
            extract_tarball_and_discard_archive(&archive, &layout.dest)?;
            Ok(layout.dest.join("top"))
        })
        .unwrap();
    }
}
