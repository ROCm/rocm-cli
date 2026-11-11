// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Process-spawn, -run and -terminate primitives.
//!
//! Hidden-console/detached spawn and exit-wait on Windows, process/process-tree
//! termination on both platforms, and the exclusive `FileLock` used to
//! serialize check-then-act sequences over shared on-disk state across
//! processes — e.g. the daemon autostart path, the managed-launch GPU claim,
//! a `uv` install, and each JSONL store's append/rewrite.
//!
//! The bounded subprocess executor — [`run_with_timeout`], which runs a command
//! to completion with both streams captured and a real time bound, and
//! [`run_bounded`], the same executor returning a typed capture — lives in the
//! private `bounded` submodule, split out to keep this file under the module
//! size limit, and is re-exported from here under its original paths.

use crate::DEFAULT_LOCAL_HOST;
use anyhow::{Context, Result, bail};
#[cfg(windows)]
use std::collections::BTreeMap;
#[cfg(windows)]
use std::ffi::{OsStr, OsString};
use std::fs;
use std::net::TcpListener;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT,
    CreateProcessW, DETACHED_PROCESS, GetExitCodeProcess, INFINITE, OpenProcess,
    PROCESS_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    STARTF_USESHOWWINDOW, STARTF_USESTDHANDLES, STARTUPINFOW, TerminateProcess,
    WaitForSingleObject,
};

mod bounded;
pub use bounded::run_with_timeout;
pub(crate) use bounded::{MAX_QUOTED_OUTPUT_CHARS, run_bounded};

#[cfg(windows)]
pub fn spawn_detached_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
) -> Result<u32> {
    spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        false,
        None,
    )
}

#[cfg(windows)]
pub fn spawn_hidden_console_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
) -> Result<u32> {
    spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
        true,
        None,
    )
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn spawn_hidden_console_with_log(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
    log_path: &Path,
) -> Result<u32> {
    use std::os::windows::io::AsRawHandle;
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{
        CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let current_process = unsafe { GetCurrentProcess() };
    let source = log_file.as_raw_handle() as HANDLE;
    let mut stdout_handle: HANDLE = null_mut();
    let mut stderr_handle: HANDLE = null_mut();
    unsafe {
        if DuplicateHandle(
            current_process,
            source,
            current_process,
            &raw mut stdout_handle,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        ) == 0
        {
            bail!(
                "failed to duplicate stdout log handle for {}: {}",
                log_path.display(),
                std::io::Error::last_os_error()
            );
        }
        if DuplicateHandle(
            current_process,
            source,
            current_process,
            &raw mut stderr_handle,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        ) == 0
        {
            CloseHandle(stdout_handle);
            bail!(
                "failed to duplicate stderr log handle for {}: {}",
                log_path.display(),
                std::io::Error::last_os_error()
            );
        }
    }
    let result = spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
        true,
        Some((stdout_handle, stderr_handle)),
    );
    unsafe {
        CloseHandle(stdout_handle);
        CloseHandle(stderr_handle);
    }
    result
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn wait_for_process_exit(pid: u32) -> Result<u32> {
    use windows_sys::Win32::Foundation::CloseHandle;

    let handle = unsafe {
        OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        )
    };
    if handle.is_null() {
        bail!(
            "failed to open process {pid} for wait: {}",
            std::io::Error::last_os_error()
        );
    }
    unsafe {
        WaitForSingleObject(handle, INFINITE);
        let mut exit_code = 0;
        if GetExitCodeProcess(handle, &raw mut exit_code) == 0 {
            CloseHandle(handle);
            bail!(
                "failed to read process {pid} exit code: {}",
                std::io::Error::last_os_error()
            );
        }
        CloseHandle(handle);
        Ok(exit_code)
    }
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn terminate_process(pid: u32) -> Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;

    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
    if handle.is_null() {
        bail!(
            "failed to open process {pid} for termination: {}",
            std::io::Error::last_os_error()
        );
    }
    let terminated = unsafe { TerminateProcess(handle, 1) };
    unsafe {
        CloseHandle(handle);
    }
    if terminated == 0 {
        bail!(
            "failed to terminate process {pid}: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn terminate_process(pid: u32) -> Result<()> {
    let status = unsafe { libc::kill(pid.cast_signed(), libc::SIGTERM) };
    if status == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to terminate process {pid}"))
    }
}

/// Terminate `pid` together with every transitive child process.
///
/// Long-running engines such as vLLM spawn helper subprocesses (for example the
/// `EngineCore` worker that holds the GPU allocation). Signalling only the
/// launcher PID leaves those workers reparented to init, where they keep the
/// model resident and the device memory pinned. Walking the descendant tree and
/// signalling each process avoids that leak.
#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn terminate_process_tree(pid: u32) -> Result<()> {
    let mut last_error: Option<(u32, std::io::Error)> = None;
    for target in collect_process_tree(pid) {
        let status = unsafe { libc::kill(target.cast_signed(), libc::SIGTERM) };
        if status != 0 {
            let error = std::io::Error::last_os_error();
            // A process that already exited (ESRCH) is not a failure here.
            if error.raw_os_error() != Some(libc::ESRCH) {
                last_error = Some((target, error));
            }
        }
    }
    if let Some((target, error)) = last_error {
        return Err(error).with_context(|| format!("failed to terminate process {target}"));
    }
    Ok(())
}

/// Send `signal` to `pid`, optionally extending to its transitive children.
///
/// Delivery to a process that has already exited (`ESRCH`) counts as success:
/// the goal — that process no longer running — is already met. Returns `false`
/// only when a signal could not be delivered for another reason (for example
/// `EPERM`). Used by the verified-termination logic in [`proc_lifecycle`](crate::proc_lifecycle).
#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub(crate) fn signal_process_scope(pid: u32, signal: i32, include_tree: bool) -> bool {
    let targets = if include_tree {
        collect_process_tree(pid)
    } else {
        vec![pid]
    };
    let mut delivered = true;
    for target in targets {
        let status = unsafe { libc::kill(target.cast_signed(), signal) };
        if status != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
            delivered = false;
        }
    }
    delivered
}

/// Snapshot `root` plus its transitive descendants as a flat PID list.
///
/// Used by [`proc_lifecycle`](crate::proc_lifecycle) to bind a tree termination to the exact processes
/// present when the stop began. On platforms without `/proc` only `root` is
/// returned.
#[cfg(not(windows))]
pub(crate) fn process_tree_pids(root: u32) -> Vec<u32> {
    collect_process_tree(root)
}

#[cfg(windows)]
pub(crate) fn process_tree_pids(root: u32) -> Vec<u32> {
    vec![root]
}

/// Collect `root` plus all of its transitive descendants by reading `/proc`.
///
/// On platforms without `/proc` (for example macOS) only `root` is returned, so
/// callers degrade to single-process termination rather than failing.
#[cfg(not(windows))]
fn collect_process_tree(root: u32) -> Vec<u32> {
    let mut children: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    if let Ok(entries) = fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if let Some(ppid) = read_parent_pid(pid) {
                children.entry(ppid).or_default().push(pid);
            }
        }
    }

    walk_process_tree(root, &children)
}

/// Walk `root` plus its transitive descendants over an already-built
/// parent→children table.
///
/// `root` comes first, every other PID comes after its parent, and no PID is
/// listed twice, so a cycle in the table ends the walk instead of looping it.
///
/// Split out of [`collect_process_tree`] so the traversal — the part that
/// decides which PIDs get signalled — can be exercised against a synthetic
/// table (cycles, self-parents, shared parents) without touching `/proc`.
#[cfg(not(windows))]
fn walk_process_tree(root: u32, children: &std::collections::HashMap<u32, Vec<u32>>) -> Vec<u32> {
    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        order.push(pid);
        if let Some(kids) = children.get(&pid) {
            stack.extend(kids.iter().copied());
        }
    }
    order
}

/// Read the parent PID of `pid` from `/proc/<pid>/stat`.
///
/// The `comm` field can contain spaces and parentheses, so the parent PID is
/// parsed from the text after the final `)`.
#[cfg(not(windows))]
fn read_parent_pid(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.get(stat.rfind(')')? + 1..)?;
    let mut fields = after_comm.split_whitespace();
    let _state = fields.next()?;
    fields.next()?.parse::<u32>().ok()
}

/// Terminate `pid` together with every transitive child process.
///
/// The Windows implementation falls back to terminating the single process; the
/// engines that rely on descendant cleanup are Unix-only.
#[cfg(windows)]
pub fn terminate_process_tree(pid: u32) -> Result<()> {
    terminate_process(pid)
}

/// What one look at a PID could establish.
///
/// [`process_is_running`] answers a plain yes or no, so a process that is alive
/// but cannot be queried reads the same as one that is gone. A stop that is about
/// to report a process as exited must not make that mistake, so this keeps the
/// third answer apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessProbe {
    /// The PID names a process that has not exited.
    Running,
    /// The PID names no process, or one that has exited.
    Gone,
    /// The PID could not be queried, so it can be neither confirmed running nor
    /// confirmed gone. On Windows that includes a live process the caller may not
    /// open (`ERROR_ACCESS_DENIED`).
    ///
    /// The cost of reading this as "may be running" is that a stopped service's
    /// recorded PID, taken since by a process the caller cannot open, keeps its
    /// record live: the record never reconciles to `stopped`, `services remove`
    /// and `prune` refuse it, and a stop reports it unverified. That is the safe
    /// direction (the alternative deletes the key and record under a process that
    /// may be serving), and there is no better evidence to be had: the creation
    /// time that would tell the two processes apart is read with the same access
    /// right, so a process that cannot be opened has no readable token to
    /// compare. `rocm services remove <id> --force --yes` clears such a record
    /// (see `docs/manual-testing.md`, "A record that stays unverified"); it
    /// applies only when no recorded process is confirmed running, one could not
    /// be queried, and a stop was requested.
    Unqueryable,
}

impl ProcessProbe {
    /// Whether the process may still be running: everything but a confirmed
    /// exit. The answer for a decision that must not treat a process it could
    /// not look at as gone, such as dropping a service's endpoint key, marking
    /// its record stopped or deleting it.
    #[must_use]
    pub const fn may_be_running(self) -> bool {
        !matches!(self, Self::Gone)
    }
}

/// Windows `ERROR_INVALID_PARAMETER`: what `OpenProcess` sets for a PID that
/// names no process.
#[cfg(any(windows, test))]
const ERROR_INVALID_PARAMETER: i32 = 87;

/// Windows `STILL_ACTIVE`: the exit code `GetExitCodeProcess` reports for a
/// process that has not exited.
#[cfg(any(windows, test))]
const STILL_ACTIVE: u32 = 259;

/// What `GetExitCodeProcess` on a handle that opened says about the process:
/// still active is running, any other exit code is gone, and a failed query is
/// neither. Split out of the FFI, with the open-failure mapping above, so a test
/// runs the whole decision on every host.
#[cfg(any(windows, test))]
pub(crate) const fn classify_exit_query(queried: bool, exit_code: u32) -> ProcessProbe {
    match (queried, exit_code) {
        (false, _) => ProcessProbe::Unqueryable,
        (true, STILL_ACTIVE) => ProcessProbe::Running,
        (true, _) => ProcessProbe::Gone,
    }
}

/// What a failed `OpenProcess` says about the PID, from its `GetLastError` code.
///
/// Only the code for "no such process" counts as gone. `ERROR_ACCESS_DENIED`
/// (5), which a live process the caller may not open sets, and any other or
/// missing code are [`ProcessProbe::Unqueryable`]: a failure to look is not an
/// exit. Split out of the FFI so a test runs it on every host.
#[cfg(any(windows, test))]
pub(crate) const fn classify_open_failure(code: Option<i32>) -> ProcessProbe {
    match code {
        Some(ERROR_INVALID_PARAMETER) => ProcessProbe::Gone,
        _ => ProcessProbe::Unqueryable,
    }
}

/// The path of a Windows system tool in the system directory (`System32`).
///
/// The directory comes from `GetSystemDirectoryW`, not from the environment, so
/// neither a directory earlier on `PATH` nor a `SystemRoot` the caller controls
/// can stand in for it; the bare name when the directory cannot be read (and off
/// Windows). `name` includes the extension (`taskkill.exe`).
#[must_use]
pub fn windows_system_tool(name: &str) -> PathBuf {
    system_tool_path(windows_system_directory(), name)
}

/// [`windows_system_tool`] for a given system directory, so a test builds the
/// path on every host.
fn system_tool_path(system_dir: Option<PathBuf>, name: &str) -> PathBuf {
    match system_dir.filter(|dir| !dir.as_os_str().is_empty()) {
        Some(dir) => dir.join(name),
        None => PathBuf::from(name),
    }
}

/// The Windows system directory, read with `GetSystemDirectoryW`.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn windows_system_directory() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    const MAX_PATH: u32 = 260;
    let mut buffer = [0_u16; MAX_PATH as usize];
    // SAFETY: `buffer` is a live local valid for `MAX_PATH` wide characters, and
    // that is the size passed, so the call cannot write past it.
    let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), MAX_PATH) } as usize;
    // Zero is a failure; a length that does not fit means the path was too long
    // for the buffer, which is not a directory to trust either.
    (length > 0 && length < buffer.len())
        .then(|| PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length])))
}

#[cfg(not(windows))]
#[allow(clippy::missing_const_for_fn)] // keeps the signature of the Windows arm
fn windows_system_directory() -> Option<PathBuf> {
    None
}

/// E2E hook: `ROCM_E2E_UNQUERYABLE_PIDS`, a comma-separated list of PIDs, makes
/// [`probe_process`] answer [`ProcessProbe::Unqueryable`] for each of them.
///
/// A live process this account may not open cannot be staged from outside the
/// binary, and without it the stop, the liveness refresh and `services remove`
/// have no live coverage for that case off Windows. Compiled only into the
/// `e2e-test-hooks` build; production builds never read the variable.
#[cfg(feature = "e2e-test-hooks")]
fn scripted_probe(pid: u32) -> Option<ProcessProbe> {
    let listed = std::env::var("ROCM_E2E_UNQUERYABLE_PIDS").ok()?;
    listed
        .split(',')
        .filter_map(|entry| entry.trim().parse::<u32>().ok())
        .any(|listed| listed == pid)
        .then_some(ProcessProbe::Unqueryable)
}

/// Look at `pid`, keeping a process that cannot be queried apart from one that
/// is gone. See [`ProcessProbe`].
#[cfg(windows)]
#[must_use]
#[allow(unsafe_code)] // Win32 FFI
pub fn probe_process(pid: u32) -> ProcessProbe {
    use windows_sys::Win32::Foundation::CloseHandle;

    #[cfg(feature = "e2e-test-hooks")]
    if let Some(scripted) = scripted_probe(pid) {
        return scripted;
    }
    if pid == 0 {
        return ProcessProbe::Gone;
    }
    // SAFETY: `OpenProcess` takes only plain values and returns a handle or null.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return classify_open_failure(std::io::Error::last_os_error().raw_os_error());
    }
    let mut exit_code = 0;
    // SAFETY: `handle` is the non-null handle opened above, still open, and
    // `exit_code` is a live local that is valid for this one call to write to.
    let queried = unsafe { GetExitCodeProcess(handle, &raw mut exit_code) != 0 };
    // SAFETY: `handle` is the handle opened above, closed here exactly once and
    // not used afterwards.
    unsafe {
        CloseHandle(handle);
    }
    classify_exit_query(queried, exit_code)
}

/// Look at `pid`. `kill(pid, 0)` already tells a live process the caller may
/// not signal (`EPERM`, which counts as running) from one that is gone, so there
/// is no third answer to keep apart here.
#[cfg(not(windows))]
#[must_use]
pub fn probe_process(pid: u32) -> ProcessProbe {
    #[cfg(feature = "e2e-test-hooks")]
    if let Some(scripted) = scripted_probe(pid) {
        return scripted;
    }
    if process_is_running(pid) {
        ProcessProbe::Running
    } else {
        ProcessProbe::Gone
    }
}

/// Whether `pid` names a running process.
///
/// On Windows a live process this one may not open reads as not running here; a
/// decision that must not take it for gone (dropping a key, marking a record
/// stopped, deleting it) asks [`probe_process`] instead and keeps
/// [`ProcessProbe::Unqueryable`] apart.
#[cfg(windows)]
pub fn process_is_running(pid: u32) -> bool {
    probe_process(pid) == ProcessProbe::Running
}

#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    let status = unsafe { libc::kill(pid, 0) };
    if status == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// An advisory, cross-process exclusive lock backed by a lock file.
///
/// Wraps the standard-library file lock (`std::fs::File::lock`), so the exclusion
/// holds between *separate `rocm` processes*, not just threads: each caller opens
/// the same lock-file path and only one can hold the lock at a time. It exists to
/// serialize check-then-act sequences over shared on-disk state — e.g. the
/// daemon autostart decision, the managed-serve GPU select-then-claim, a
/// `uv` install, and each JSONL store's append-or-repair and read-modify-write
/// (see `jsonl_lock_path` in `jsonl_store.rs`) — so two concurrent invocations
/// cannot both pass the same check, or interleave a write with another's.
///
/// The lock is released when the guard is dropped, and by the OS if the process
/// exits while holding it (so a crashed holder never wedges the next caller).
#[derive(Debug)]
pub struct FileLock {
    file: fs::File,
    path: PathBuf,
}

impl FileLock {
    /// Acquire an exclusive lock on `path`, creating the lock file and any
    /// missing parent directories first. Blocks until the lock is available.
    ///
    /// The lock file itself carries no data; it is a rendezvous point, so an
    /// existing file is reused (never truncated) and its contents are ignored.
    pub fn acquire(path: impl Into<PathBuf>) -> Result<Self> {
        let (file, path) = Self::open(path.into())?;
        file.lock()
            .with_context(|| format!("failed to acquire lock {}", path.display()))?;
        Ok(Self { file, path })
    }

    /// As [`acquire`](Self::acquire), but never blocks: `Ok(None)` means another
    /// holder has the lock right now. For callers that must not park a thread
    /// (an async loop) and retry on their own schedule.
    pub fn try_acquire(path: impl Into<PathBuf>) -> Result<Option<Self>> {
        let (file, path) = Self::open(path.into())?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { file, path })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => {
                Err(error).with_context(|| format!("failed to try-acquire lock {}", path.display()))
            }
        }
    }

    fn open(path: PathBuf) -> Result<(fs::File, PathBuf)> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create lock directory {}", parent.display()))?;
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("failed to open lock file {}", path.display()))?;
        Ok((file, path))
    }

    /// The lock file backing this guard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Like [`Self::acquire`], but fails with an actionable error instead of
    /// blocking forever once `timeout` has elapsed.
    ///
    /// Used where an indefinite wait would be a worse failure mode than a
    /// clear timeout message — e.g. `rocmd`'s automatic recovery, which must
    /// not hang forever behind a `rocm serve` launch holding the same lock, and
    /// `rocm services restart`, which has already stopped the service when it
    /// waits here.
    /// A `rocm serve` launch itself uses the unbounded [`Self::acquire`]: its
    /// own worst-case hold time (the already-running probe inside the launch
    /// transaction) can exceed [`MANAGED_LAUNCH_LOCK_TIMEOUT`], so bounding it
    /// would trade a real, slow-but-working launch for a spurious failure.
    ///
    /// # Errors
    /// Fails when the lock file cannot be created/locked, or when another
    /// holder keeps it for the whole timeout.
    pub fn acquire_bounded(path: impl Into<PathBuf>, timeout: Duration) -> Result<Self> {
        let (file, path) = Self::open(path.into())?;
        let deadline = Instant::now() + timeout;
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Self { file, path }),
                Err(std::fs::TryLockError::WouldBlock) => {}
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(anyhow::Error::new(error)
                        .context(format!("failed to lock {}", path.display())));
                }
            }
            let now = Instant::now();
            if now >= deadline {
                bail!(
                    "timed out after {timeout:?} waiting for the lock at {path}; another \
                     `rocm serve` launch or `rocmd` recovery is mid-transaction. Retry in a \
                     moment, and check `rocm services` for a stuck launch.",
                    path = path.display()
                );
            }
            thread::sleep(FILE_LOCK_BOUNDED_POLL.min(deadline - now));
        }
    }
}

/// Polling interval used while waiting on [`FileLock::acquire_bounded`].
const FILE_LOCK_BOUNDED_POLL: Duration = Duration::from_millis(25);

/// Total time `rocmd`'s automatic recovery waits for the launch lock.
///
/// Bounds contention on [`crate::paths::AppPaths::managed_launch_lock_path`] during the
/// port-allocation window, failing with an actionable error instead of hanging. Only
/// `rocmd`'s recovery/restart path and `rocm services restart` (whose stop has
/// already happened by then) are bounded this way; a `rocm serve` launch holds the
/// same lock with the unbounded [`FileLock::acquire`] (see its doc comment on
/// [`FileLock::acquire_bounded`] for why).
pub const MANAGED_LAUNCH_LOCK_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Managed-service port leasing.
//
// Both `rocm serve` and `rocmd` recovery contend for the same local TCP ports
// and the same service records, so the primitives that arbitrate them live here
// rather than in either binary. The transaction itself is serialized by the
// existing [`crate::paths::AppPaths::managed_launch_lock_path`] / [`FileLock`] — see
// [`FileLock::acquire_bounded`] for the bounded variant this contention uses.
// ---------------------------------------------------------------------------

/// True only for the exact canonical default loopback host.
///
/// Proactive OS collision detection (leasing a real socket before launch) is
/// claimed only for this one host. `localhost`, `::1`, `0.0.0.0`, and other
/// IPv4 addresses are "custom hosts": they require an explicit port and keep the
/// existing engine-owned bind/error behavior, because an IPv4-loopback probe
/// would not prove anything about the address the engine will actually bind.
#[must_use]
pub fn is_canonical_loopback_host(host: &str) -> bool {
    host.trim() == DEFAULT_LOCAL_HOST
}

/// Validate the optional `port` argument of a `launch_server` tool call.
///
/// `None` and JSON `null` mean "not supplied" (automatic selection). Anything
/// else must be an integer in 1-65535: a malformed value must not be read as
/// "not supplied" and silently fall through to automatic port selection.
/// Shared by the chat assistant and the MCP server so the two cannot drift.
///
/// # Errors
/// Returns the reason, to be prefixed with the caller's own argument name.
pub fn launch_port_argument(
    value: Option<&serde_json::Value>,
) -> Result<Option<u16>, &'static str> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let port = value.as_u64().ok_or("must be an integer")?;
    match u16::try_from(port) {
        Ok(port) if port != 0 => Ok(Some(port)),
        _ => Err("must be between 1 and 65535"),
    }
}

/// A real, held `127.0.0.1:<port>` listener proving the port is bindable now.
///
/// The lease is the collision check: it is acquired before any manifest, log, or
/// child process exists, and [`LoopbackPortLease::release`] drops the socket
/// immediately before the engine child is spawned so the child can bind it. The
/// window between release and the child's bind is unavoidable without engine
/// socket-passing; it is far narrower than the previous "bind and hope" launch.
#[derive(Debug)]
pub struct LoopbackPortLease {
    port: u16,
    listener: Option<TcpListener>,
}

impl LoopbackPortLease {
    #[must_use]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// True while this lease still holds the socket open.
    ///
    /// No production caller needs this: `release` is the only thing that ever
    /// changes it, and callers already know whether they called `release`. It
    /// stays `pub` because downstream crates' tests assert on it directly
    /// (e.g. `apps/rocm`'s `serve_port` tests), where `#[cfg(test)]` here
    /// would not apply.
    #[must_use]
    pub const fn is_held(&self) -> bool {
        self.listener.is_some()
    }

    /// Close the socket and hand the port to the caller. Call this immediately
    /// before spawning the process that will bind it.
    pub fn release(mut self) -> u16 {
        self.listener = None;
        self.port
    }

    /// A lease over a port this process never probed, for custom hosts where the
    /// engine, not the CLI, owns bind success or failure.
    #[must_use]
    pub const fn engine_owned(port: u16) -> Self {
        Self {
            port,
            listener: None,
        }
    }
}

/// Try to take an exclusive IPv4-loopback lease on `port`.
///
/// Errors are returned verbatim so callers can distinguish "occupied"
/// ([`std::io::ErrorKind::AddrInUse`], the only condition that should advance an
/// automatic scan) from a privileged or unavailable address, which must fail
/// immediately rather than silently choosing a different port.
pub fn lease_loopback_port(port: u16) -> std::io::Result<LoopbackPortLease> {
    let listener = TcpListener::bind(std::net::SocketAddrV4::new(
        std::net::Ipv4Addr::LOCALHOST,
        port,
    ))?;
    Ok(LoopbackPortLease {
        port,
        listener: Some(listener),
    })
}

/// [`lease_loopback_port`], but mapped to an `anyhow::Error`.
///
/// Only uses `in_use_message` for a genuine collision
/// ([`std::io::ErrorKind::AddrInUse`]). Any other bind failure (permission
/// denied, address unavailable, ...) gets its own generic message instead —
/// mislabelling those as "in use" sends the user to stop a service that
/// isn't the actual problem. Mirrors `rocm serve`'s own
/// `explicit_bind_error`; shared by the CLI's restart re-check and
/// `rocmd`'s recovery re-check so neither duplicates (or drifts from) that
/// distinction.
pub fn lease_loopback_port_or_bind_error(
    port: u16,
    in_use_message: impl std::fmt::Display,
) -> anyhow::Result<LoopbackPortLease> {
    lease_loopback_port(port).map_err(|error| bind_error(port, &error, in_use_message))
}

/// Pure error-mapping half of [`lease_loopback_port_or_bind_error`], split out
/// so the `AddrInUse`-vs-other-error distinction is tested against a
/// synthetic `io::Error` rather than needing a real bind failure (a
/// privileged port, an unavailable address) that is not reliably
/// reproducible in CI.
fn bind_error(
    port: u16,
    error: &std::io::Error,
    in_use_message: impl std::fmt::Display,
) -> anyhow::Error {
    if error.kind() == std::io::ErrorKind::AddrInUse {
        anyhow::anyhow!("{in_use_message}")
    } else {
        anyhow::anyhow!("failed to reserve port {port}: {error}")
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Best-effort: an unlock failure only means the OS releases it slightly
        // later (at the latest when the file handle closes), never a lost lock.
        let _ = self.file.unlock();
    }
}

#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI (pre_exec/setsid)
pub fn detach_command_session(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
pub fn detach_command_session(_command: &mut Command) {}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn spawn_windows_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
    creation_flags: u32,
    hide_window: bool,
    std_handles: Option<(
        windows_sys::Win32::Foundation::HANDLE,
        windows_sys::Win32::Foundation::HANDLE,
    )>,
) -> Result<u32> {
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::CloseHandle;

    let mut command_line = windows_command_line(program.as_os_str(), args);
    let application_name = nul_terminated_wide(program.as_os_str());
    let mut environment = windows_environment_block(env_overrides);
    let mut startup_info = STARTUPINFOW {
        cb: std::mem::size_of::<STARTUPINFOW>() as u32,
        ..Default::default()
    };
    if hide_window {
        const SW_HIDE: u16 = 0;
        startup_info.dwFlags |= STARTF_USESHOWWINDOW;
        startup_info.wShowWindow = SW_HIDE;
    }
    if let Some((stdout_handle, stderr_handle)) = std_handles {
        startup_info.dwFlags |= STARTF_USESTDHANDLES;
        startup_info.hStdInput = null_mut();
        startup_info.hStdOutput = stdout_handle;
        startup_info.hStdError = stderr_handle;
    }
    let mut process_info = PROCESS_INFORMATION::default();
    let created = unsafe {
        CreateProcessW(
            application_name.as_ptr(),
            command_line.as_mut_ptr(),
            null(),
            null(),
            i32::from(std_handles.is_some()),
            creation_flags,
            environment.as_mut_ptr().cast(),
            null(),
            &raw const startup_info,
            &raw mut process_info,
        )
    };
    if created == 0 {
        bail!(
            "failed to launch detached process {}: {}",
            program.display(),
            std::io::Error::last_os_error()
        );
    }
    unsafe {
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
    }
    Ok(process_info.dwProcessId)
}

#[cfg(windows)]
fn nul_terminated_wide(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn windows_command_line(program: &OsStr, args: &[String]) -> Vec<u16> {
    let mut command = quote_windows_arg(&program.to_string_lossy());
    for arg in args {
        command.push(' ');
        command.push_str(&quote_windows_arg(arg));
    }
    OsStr::new(&command)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(windows)]
fn quote_windows_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r' | '"'))
    {
        return arg.to_owned();
    }
    let mut quoted = String::from("\"");
    let mut backslashes = 0usize;
    for ch in arg.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(std::iter::repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(std::iter::repeat_n('\\', backslashes));
                backslashes = 0;
                quoted.push(ch);
            }
        }
    }
    quoted.extend(std::iter::repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
}

#[cfg(windows)]
fn windows_environment_block(env_overrides: &[(&str, &Path)]) -> Vec<u16> {
    let mut env = BTreeMap::<String, OsString>::new();
    for (key, value) in std::env::vars_os() {
        let key_string = key.to_string_lossy().to_string();
        env.insert(
            key_string.to_ascii_uppercase(),
            OsString::from(format!("{}={}", key_string, value.to_string_lossy())),
        );
    }
    for (key, value) in env_overrides {
        env.insert(
            key.to_ascii_uppercase(),
            OsString::from(format!("{}={}", key, value.display())),
        );
    }
    let mut block = Vec::new();
    for entry in env.values() {
        block.extend(entry.encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

/// Property tests for [`walk_process_tree`]; a child module so it can reach
/// the walk while it stays private.
#[cfg(all(test, not(windows)))]
#[path = "process_tree_properties.rs"]
mod tree_properties;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::workspace_test_artifact_dir;
    use std::process::Stdio;

    /// A system tool is started from the system directory, never by a bare name
    /// that `PATH` could resolve to something else; with no directory it falls
    /// back to the bare name.
    #[test]
    fn a_system_tool_is_resolved_under_the_system_directory() {
        let dir = PathBuf::from("C:\\Windows").join("System32");
        let path = system_tool_path(Some(dir.clone()), "taskkill.exe");
        assert_eq!(path, dir.join("taskkill.exe"));
        assert!(path.ends_with(Path::new("System32").join("taskkill.exe")));
        assert!(!path.parent().unwrap().as_os_str().is_empty());

        assert_eq!(
            system_tool_path(None, "tasklist.exe"),
            PathBuf::from("tasklist.exe")
        );
        assert_eq!(
            system_tool_path(Some(PathBuf::new()), "tasklist.exe"),
            PathBuf::from("tasklist.exe"),
            "an empty directory is no directory"
        );
    }

    /// A failed `OpenProcess` is a gone process only for the code that means
    /// "no such process". Access denied, which a live process the caller may not
    /// open sets, and any code that is not known are not an exit.
    #[test]
    fn a_failed_open_is_gone_only_for_no_such_process() {
        const ERROR_ACCESS_DENIED: i32 = 5;
        assert_eq!(classify_open_failure(Some(87)), ProcessProbe::Gone);
        assert_eq!(
            classify_open_failure(Some(ERROR_ACCESS_DENIED)),
            ProcessProbe::Unqueryable
        );
        // The codes around the two that matter, and no code at all.
        for code in [Some(0), Some(6), Some(31), Some(299), Some(-1), None] {
            assert_eq!(
                classify_open_failure(code),
                ProcessProbe::Unqueryable,
                "{code:?}"
            );
        }
    }

    /// An opened handle's exit code: still active is running, any other code is
    /// gone, and a failed query (whatever the code it left behind) is neither.
    #[test]
    fn an_exit_code_query_is_running_gone_or_unqueryable() {
        assert_eq!(classify_exit_query(true, 259), ProcessProbe::Running);
        assert_eq!(classify_exit_query(true, 0), ProcessProbe::Gone);
        assert_eq!(classify_exit_query(true, 1), ProcessProbe::Gone);
        assert_eq!(classify_exit_query(false, 0), ProcessProbe::Unqueryable);
        assert_eq!(classify_exit_query(false, 259), ProcessProbe::Unqueryable);
    }

    /// Off Windows, a PID that is running (including one the caller may not
    /// signal) probes as running and an absent one as gone, matching
    /// [`process_is_running`].
    #[cfg(not(windows))]
    #[test]
    fn the_probe_agrees_with_process_is_running_off_windows() {
        let own = std::process::id();
        assert_eq!(probe_process(own), ProcessProbe::Running);
        assert_eq!(probe_process(0), ProcessProbe::Gone);
        assert_eq!(probe_process(999_999_998), ProcessProbe::Gone);
    }

    const LOCK_CHILD_PATH_ENV: &str = "ROCM_CORE_TEST_LOCK_PATH";
    const LOCK_CHILD_READY_ENV: &str = "ROCM_CORE_TEST_LOCK_READY";
    const HANDOFF_PORT_ENV: &str = "ROCM_CORE_TEST_HANDOFF_PORT";

    fn lock_test_dir(label: &str) -> PathBuf {
        let dir = workspace_test_artifact_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        fs::create_dir_all(&dir).expect("create lock test dir");
        dir
    }

    /// Re-run this test binary, executing exactly one `#[ignore]`d helper test in
    /// a genuinely separate process. Cross-process locking and socket handoff
    /// cannot be proven inside one process.
    fn spawn_helper_test(name: &str, envs: &[(&str, String)]) -> std::process::Child {
        let exe = std::env::current_exe().expect("test binary path");
        let mut command = Command::new(exe);
        command
            .arg(name)
            .arg("--exact")
            .arg("--ignored")
            .arg("--test-threads")
            .arg("1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (key, value) in envs {
            command.env(key, value);
        }
        command.spawn().expect("spawn helper test process")
    }

    #[test]
    fn canonical_loopback_host_is_only_the_exact_default_address() {
        assert!(is_canonical_loopback_host(DEFAULT_LOCAL_HOST));
        assert!(is_canonical_loopback_host("  127.0.0.1  "));
        // Everything else is a "custom host": explicit port, engine-owned bind.
        for host in [
            "localhost",
            "::1",
            "0.0.0.0",
            "127.0.0.2",
            "192.168.1.4",
            "",
        ] {
            assert!(
                !is_canonical_loopback_host(host),
                "{host} must not be treated as the canonical loopback host"
            );
        }
    }

    #[test]
    fn bounded_file_lock_excludes_independent_handles_and_fails_closed_on_timeout() {
        let path = lock_test_dir("core-bounded-lock-exclusive").join("launch.lock");
        let held = FileLock::acquire_bounded(&path, Duration::from_secs(5)).expect("first holder");
        assert_eq!(held.path(), path.as_path());

        // A second, independent file handle must contend on the OS lock rather
        // than sail through, and must give up with an actionable error.
        let started = Instant::now();
        let error = FileLock::acquire_bounded(&path, Duration::from_millis(200))
            .expect_err("a held lock must not be acquired twice");
        assert!(
            started.elapsed() >= Duration::from_millis(200),
            "waiter returned before its deadline"
        );
        let message = format!("{error:#}");
        assert!(message.contains("timed out"), "{message}");

        drop(held);
        FileLock::acquire_bounded(&path, Duration::from_secs(5))
            .expect("lock is free once the guard is dropped");
    }

    #[test]
    #[ignore = "spawned as a child process by bounded_file_lock_is_released_when_the_holder_process_exits"]
    fn bounded_file_lock_child_holder() {
        let Ok(path) = std::env::var(LOCK_CHILD_PATH_ENV) else {
            return;
        };
        let ready = std::env::var(LOCK_CHILD_READY_ENV).expect("ready-file path");
        let guard = FileLock::acquire_bounded(Path::new(&path), Duration::from_secs(10))
            .expect("child acquires the shared lock");
        fs::write(&ready, "held").expect("signal that the child holds the lock");
        thread::sleep(Duration::from_millis(300));
        // Leak the guard so the lock is released by process exit alone — that is
        // the property a crashed launcher depends on.
        std::mem::forget(guard);
    }

    #[test]
    fn bounded_file_lock_is_released_when_the_holder_process_exits() {
        let dir = lock_test_dir("core-bounded-lock-crossproc");
        let path = dir.join("launch.lock");
        let ready = dir.join("child-ready");
        let mut child = spawn_helper_test(
            "process::tests::bounded_file_lock_child_holder",
            &[
                (LOCK_CHILD_PATH_ENV, path.display().to_string()),
                (LOCK_CHILD_READY_ENV, ready.display().to_string()),
            ],
        );

        let deadline = Instant::now() + Duration::from_mins(1);
        while !ready.exists() {
            assert!(
                Instant::now() < deadline,
                "child never reported holding the lock"
            );
            thread::sleep(Duration::from_millis(25));
        }

        let busy = FileLock::acquire_bounded(&path, Duration::from_millis(50))
            .expect_err("the lock must be exclusive across processes");
        assert!(format!("{busy:#}").contains("timed out"));

        let status = child.wait().expect("child holder exits");
        assert!(status.success(), "child holder failed: {status}");
        FileLock::acquire_bounded(&path, Duration::from_secs(10))
            .expect("the OS releases the lock when the holding process exits");
    }

    #[test]
    fn loopback_lease_blocks_a_second_bind_until_released() {
        let ephemeral = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral bind");
        let port = ephemeral.local_addr().expect("local addr").port();
        drop(ephemeral);

        let lease = lease_loopback_port(port).expect("free port leases");
        assert_eq!(lease.port(), port);
        assert!(lease.is_held());
        let conflict = lease_loopback_port(port).expect_err("a held port must not lease twice");
        assert_eq!(conflict.kind(), std::io::ErrorKind::AddrInUse);

        assert_eq!(lease.release(), port);
        lease_loopback_port(port).expect("the port is bindable again after release");
    }

    #[test]
    fn bind_error_labels_only_a_genuine_collision_as_in_use() {
        let in_use = bind_error(
            9000,
            &std::io::Error::from(std::io::ErrorKind::AddrInUse),
            "its port is in use",
        );
        assert_eq!(in_use.to_string(), "its port is in use");

        let other = bind_error(
            9000,
            &std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            "its port is in use",
        );
        assert_ne!(other.to_string(), "its port is in use");
        assert!(other.to_string().contains("9000"));
    }

    #[test]
    fn engine_owned_lease_holds_no_socket() {
        // Custom hosts get a port carrier, never a loopback probe.
        let lease = LoopbackPortLease::engine_owned(8000);
        assert_eq!(lease.port(), 8000);
        assert!(!lease.is_held());
    }

    #[test]
    #[ignore = "spawned as a child process by released_loopback_lease_is_immediately_bindable_by_another_process"]
    fn loopback_handoff_child() {
        let Ok(port) = std::env::var(HANDOFF_PORT_ENV) else {
            return;
        };
        let port: u16 = port.parse().expect("handoff port");
        let lease = lease_loopback_port(port).expect("child binds the released port");
        assert_eq!(lease.port(), port);
    }

    #[test]
    fn released_loopback_lease_is_immediately_bindable_by_another_process() {
        let ephemeral = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral bind");
        let port = ephemeral.local_addr().expect("local addr").port();
        drop(ephemeral);
        let lease = lease_loopback_port(port).expect("parent leases the port");
        assert_eq!(lease.release(), port);

        let status = spawn_helper_test(
            "process::tests::loopback_handoff_child",
            &[(HANDOFF_PORT_ENV, port.to_string())],
        )
        .wait()
        .expect("handoff child exits");
        assert!(
            status.success(),
            "a separate process must bind the port immediately after the lease is released"
        );
    }

    #[test]
    fn file_lock_creates_missing_parent_dirs_and_lock_file() {
        let dir =
            std::env::temp_dir().join(format!("rocm-core-filelock-create-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let lock_path = dir.join("nested").join("child").join("guard.lock");
        assert!(!lock_path.exists(), "precondition: lock file absent");

        let guard = FileLock::acquire(&lock_path).expect("acquire creates parents");
        assert!(lock_path.is_file(), "lock file is created on acquire");
        assert_eq!(guard.path(), lock_path.as_path());
        drop(guard);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_lock_serializes_concurrent_holders() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir =
            std::env::temp_dir().join(format!("rocm-core-filelock-excl-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let lock_path = dir.join("guard.lock");

        // First holder takes the lock and keeps it until we explicitly release it.
        let held = FileLock::acquire(&lock_path).expect("first acquire");

        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let thread_path = lock_path;
        let handle = std::thread::spawn(move || {
            started_tx.send(()).expect("signal about-to-acquire");
            // Blocks until the main thread drops `held`.
            let _guard = FileLock::acquire(&thread_path).expect("second acquire");
            acquired_tx.send(()).expect("signal acquired");
        });

        // Ensure the contender has reached its acquire call before we assert it
        // is blocked, so the negative check below is about the lock, not
        // scheduling latency.
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("contender started");
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(300))
                .is_err(),
            "second acquire must block while the first lock is still held"
        );

        // Releasing the first lock lets the contender proceed promptly.
        drop(held);
        acquired_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second acquire proceeds once the first lock is released");
        handle.join().expect("contender thread joins");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_lock_distinct_paths_do_not_contend() {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-filelock-distinct-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);

        // Two different lock files are independent; holding one must not block the
        // other in the same process.
        let a = FileLock::acquire(dir.join("a.lock")).expect("acquire a");
        let b = FileLock::acquire(dir.join("b.lock")).expect("acquire b");
        drop(a);
        drop(b);

        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod launch_port_argument_tests {
    use super::launch_port_argument;
    use serde_json::json;

    #[test]
    fn absent_or_null_means_automatic() {
        assert_eq!(launch_port_argument(None), Ok(None));
        assert_eq!(launch_port_argument(Some(&json!(null))), Ok(None));
    }

    #[test]
    fn accepts_the_full_valid_range() {
        assert_eq!(launch_port_argument(Some(&json!(1))), Ok(Some(1)));
        assert_eq!(launch_port_argument(Some(&json!(65535))), Ok(Some(65535)));
    }

    #[test]
    fn rejects_malformed_values_instead_of_treating_them_as_absent() {
        for value in [
            json!("8000"),
            json!(8000.5),
            json!(-1),
            json!(true),
            json!([8000]),
        ] {
            assert_eq!(
                launch_port_argument(Some(&value)),
                Err("must be an integer"),
                "{value}"
            );
        }
        for value in [json!(0), json!(65536), json!(u64::MAX)] {
            assert_eq!(
                launch_port_argument(Some(&value)),
                Err("must be between 1 and 65535"),
                "{value}"
            );
        }
    }
}
