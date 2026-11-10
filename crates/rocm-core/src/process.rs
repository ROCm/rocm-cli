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
//! [`run_with_timeout`] is the bulk of the file: run a command to completion
//! with both streams captured and a real time bound. A thread per pipe drains
//! stdout and stderr from the moment the child starts, so output size cannot
//! stall a command into its own timeout, and a reader that cannot finish is
//! abandoned rather than joined, so a grandchild holding the pipe's write end
//! open cannot outlast the deadline either. The timeout error quotes a bounded
//! head of the child's own output, because that text reaches persisted
//! automation logs. `Capture`, `settle`, `run_captured`, `wait_bounded`,
//! `Waitable` and `assemble_output` exist to serve it.
//!
//! `run_bounded` is the same executor for a caller that wants the run as a
//! [`CommandCapture`] with a typed [`CommandOutcome`] and the raw bytes, a
//! timeout included, instead of an `Err`. The host probes' `run` and
//! `capture_optional_command*` are thin adapters over it. It is not yet the only
//! such helper in the tree; converging the others onto it is tracked in
//! [#623](https://github.com/ROCm/rocm-cli/issues/623).

use crate::DEFAULT_LOCAL_HOST;
use crate::command_capture::{self, CommandCapture, CommandOutcome};
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
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT,
    CreateProcessW, DETACHED_PROCESS, GetExitCodeProcess, INFINITE, OpenProcess,
    PROCESS_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    STARTF_USESHOWWINDOW, STARTF_USESTDHANDLES, STARTUPINFOW, TerminateProcess,
    WaitForSingleObject,
};

use crate::probe_exec::truncate_to_chars;
use std::io::{ErrorKind, Read};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn process_is_running(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;

    if pid == 0 {
        return false;
    }
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return false;
    }
    let mut exit_code = 0;
    let ok = unsafe { GetExitCodeProcess(handle, &raw mut exit_code) != 0 };
    unsafe {
        CloseHandle(handle);
    }
    ok && exit_code == 259
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

/// The longest the wait loop sleeps between polls for the child's exit.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The first sleep of the wait loop, before [`next_poll_interval`] backs it off.
///
/// A host probe is usually a child that exits within a few milliseconds, and a
/// flat 50 ms sleep made every such call take 50 ms: the probes that older
/// helpers polled every 20 to 25 ms would then each have got slower for taking
/// this executor on, and `rocm examine` runs dozens of them one after another.
const FIRST_POLL: Duration = Duration::from_millis(5);

/// The sleep after `current`: doubled, up to [`POLL_INTERVAL`].
const fn next_poll_interval(current: Duration) -> Duration {
    let doubled = current.saturating_mul(2);
    if doubled.as_millis() > POLL_INTERVAL.as_millis() {
        POLL_INTERVAL
    } else {
        doubled
    }
}

/// How much a reader accumulates before publishing it to the shared buffer.
const READ_CHUNK: usize = 16 * 1024;

/// How long the readers get to finish once the child is no longer running.
///
/// Reaching this means something other than the child still holds the pipe's
/// write end — a grandchild the child left behind — so there is no EOF coming
/// and waiting longer would make `timeout` meaningless.
///
/// For a child that genuinely finished the grace has only the tail to absorb: a
/// child cannot exit before writing everything, and the readers are started
/// before the wait loop. What they are not is *scheduled* before it —
/// `wait_bounded` polls `try_wait` once with no sleep in front of it, so a child
/// fast enough to exit between spawn and that first poll can be observed exited
/// while neither reader thread has run a single instruction. So the grace covers
/// reader start-up as well as the last buffer or two. 200ms absorbs both with
/// room to spare, including on the no-GPU e2e lane's 64 concurrent scenarios.
const DRAIN_GRACE: Duration = Duration::from_millis(200);

/// How much of a timed-out child's own output the timeout error quotes.
///
/// Bounded here rather than at any one caller because the quote is assembled
/// here: the timeout message is this function's own text, and a caller sees it
/// only as an opaque error, so this is the only place that can still tell which
/// of its characters came from the child. All four callers inherit it.
///
/// It bounds this message by itself. A caller that quotes the `Output` handed
/// back to it is quoting raw child output of its own, persisted the same way,
/// so it has to apply a bound too -- which is why this is `pub(crate)` rather
/// than private: [`crate::amd_smi::amd_smi_json`] bounds its non-zero-exit
/// quote (the one #623 tracked) with *this* constant instead of declaring a
/// second one free to drift from it. The daemon's
/// `parse_sandbox_child_output` still predates both and is outside this scope.
///
/// The bound matters because the message does not stop at a terminal. `rocmd`
/// renders the whole error chain into the bridge snapshot's `note`, and the
/// GPU-metrics watcher copies that note into the automation-event and audit
/// JSONL files once a minute — stores that are never rotated or capped.
/// Unbounded, a child that prints a megabyte and then stalls past its
/// timeout grows both logs by that much per minute, forever. A 151K-character
/// JSONL line was reproduced from exactly that poller.
///
/// Sized to still hold a real stderr backtrace: the point is to stop a
/// payload-sized quote, not to shorten diagnostics.
pub(crate) const MAX_QUOTED_OUTPUT_CHARS: usize = 2000;

/// One captured stream: the bytes so far, whether its reader saw EOF, and the
/// read error that stopped it, if one did.
///
/// # Lifetime of the reader thread
///
/// The thread outlives the call that started it whenever the drain grace runs
/// out — that is the point of the grace — and it then lives until the pipe's
/// last write end closes, which is the abandoned grandchild's business, not
/// ours. What it must *not* do is keep growing: `take()` marks the capture
/// abandoned, and from that moment the reader drains to EOF and discards, so a
/// buffer nobody will read again cannot grow without bound in a daemon that
/// probes every 60 seconds.
///
/// What that bounds is the memory, and only the memory. The *number* of these
/// threads is not bounded here: each abandoned capture leaves one parked in
/// `read` until the grandchild lets go, so every probe that meets one costs two
/// until it does. The trade is deliberate — a parked reader holds a stack and
/// no CPU, and the alternative is the one below.
///
/// Discarding rather than closing the read end is deliberate. Dropping the pipe
/// would send the still-writing grandchild `EPIPE`/`SIGPIPE`, which for the
/// background helpers a sandboxed installer legitimately leaves behind would
/// turn a leak whose cost we know into a behaviour change we cannot predict.
#[derive(Clone)]
struct Capture {
    buffer: Arc<Mutex<Vec<u8>>>,
    finished: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<std::io::Error>>>,
    /// Set by `take()`: nothing will read this capture's bytes again.
    ///
    /// Read and written under `buffer`'s mutex, so "emptied and marked" is one
    /// step as far as the reader is concerned. Checking it outside the lock
    /// would leave the window where a reader that has just passed the check
    /// appends to the buffer `take()` is in the middle of emptying.
    abandoned: Arc<AtomicBool>,
}

impl Capture {
    /// Start draining `pipe` on its own thread.
    ///
    /// Publishing in chunks as they arrive, rather than returning the whole
    /// buffer at EOF, is what lets the caller take partial output from a reader
    /// that is still blocked.
    fn draining<R: Read + Send + 'static>(pipe: Option<R>) -> Self {
        let capture = Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
            finished: Arc::new(AtomicBool::new(false)),
            failure: Arc::new(Mutex::new(None)),
            abandoned: Arc::new(AtomicBool::new(false)),
        };
        let Some(mut pipe) = pipe else {
            capture.finished.store(true, Ordering::Release);
            return capture;
        };
        let sink = capture.clone();
        thread::spawn(move || {
            let mut chunk = [0_u8; READ_CHUNK];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        // Keep draining once abandoned, but stop accumulating:
                        // see the type's doc. The check shares the buffer's
                        // lock with `take()`, so after `take()` returns this
                        // buffer cannot grow by another byte.
                        let mut buffer = sink.lock();
                        if !sink.abandoned.load(Ordering::Acquire) {
                            buffer.extend_from_slice(&chunk[..read]);
                        }
                    }
                    // `read_to_end` — what `wait_with_output` used before this
                    // function existed — retries an interrupted read and
                    // returns every other error. Treating either as EOF would
                    // silently shorten the output instead, and the callers
                    // parse these bytes as JSON, where that surfaces as a
                    // parse error pointing at the wrong thing.
                    // Round again rather than break: not the end of anything.
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) => {
                        // Recorded only while someone is still listening. Once
                        // `take()` has fully returned this is merely tidy —
                        // nothing calls `take()` twice, so a failure stored
                        // afterwards would just sit unread.
                        //
                        // It is load-bearing for one window, and that window is
                        // inside `take()` itself: `take()` marks `abandoned`
                        // and empties the buffer under `buffer`'s lock, then
                        // releases it and drains the failure slot under a
                        // *different* lock. A read error that lands in between
                        // would otherwise be handed back attached to bytes the
                        // caller has already been given, and `assemble_output`
                        // turns any failure on a non-timed-out run into a hard
                        // `failed to read the output of ...` — a cleanly
                        // finished command reported as a read failure. Taking
                        // `buffer`'s lock before the check is what closes it;
                        // a bare load could pass the check, then have `take()`
                        // run the whole sequence before the store.
                        let _buffer = sink.lock();
                        if !sink.abandoned.load(Ordering::Acquire) {
                            *sink.failure() = Some(error);
                        }
                        break;
                    }
                }
            }
            sink.finished.store(true, Ordering::Release);
        });
        capture
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn failure(&self) -> std::sync::MutexGuard<'_, Option<std::io::Error>> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Take what has been captured, leaving a still-running reader unblocked.
    ///
    /// The bytes come back alongside any read error rather than instead of it,
    /// so a caller already reporting something else — a timeout — can still
    /// quote what it managed to collect.
    ///
    /// This is also the point of no return for the reader: whatever it collects
    /// after this is discarded, because there is no second caller to hand it to.
    fn take(&self) -> (Vec<u8>, Option<std::io::Error>) {
        let bytes = {
            // Marked while the buffer is held, so the reader cannot append
            // between the mark and the emptying and strand bytes in a buffer
            // that is never read again.
            let mut buffer = self.lock();
            self.abandoned.store(true, Ordering::Release);
            std::mem::take(&mut *buffer)
        };
        (bytes, self.failure().take())
    }
}

/// Run `command` to completion with both output streams captured, giving up
/// after `timeout`.
///
/// A thread per pipe drains stdout and stderr, and both are started *before*
/// the wait loop. That ordering is the whole point. Waiting for the child to
/// exit and only then calling `wait_with_output` — the shape this replaced —
/// deadlocks: the pipes are bounded by the OS (commonly ~64KiB, less on
/// Windows), so a child that writes more than that blocks in `write`, which
/// means it never exits, which means the drain is never reached. The only way
/// out was the timeout, so it measured how much the child printed rather than
/// how long it took, and killed successful commands mid-flight.
///
/// Both pipes need their own reader for the same reason: draining stdout to
/// completion and only then reading stderr stalls on a child that fills stderr
/// first.
///
/// The call returns within roughly `timeout` either way: once the child is no
/// longer running the readers get `DRAIN_GRACE` to finish and are then abandoned
/// rather than joined. Joining unconditionally would hand the schedule to a
/// grandchild the child left behind, which holds the pipe's write end open so no
/// EOF ever arrives — that is how an abandoned `sleep` comes to decide how long
/// a 2-second probe takes.
///
/// The stdio configuration is this function's own: it sets null stdin and piped
/// stdout/stderr on `command`, overriding whatever the caller set, because
/// capturing both streams is the whole contract. Callers need not — and should
/// not — set them. The stdin half is a promise in its own right rather than
/// housekeeping: a child that reads from an inherited stdin waits for input
/// that is never coming, so it is the deadline that ends it. Pinned by
/// `a_child_that_reads_stdin_is_given_an_empty_one`, on both platforms.
///
/// `label` names the subject in error messages (e.g. `"process"`,
/// `"sandbox process"`).
pub fn run_with_timeout(mut command: Command, timeout: Duration, label: &str) -> Result<Output> {
    let (child, stdout, stderr) =
        spawn_captured(&mut command).with_context(|| format!("failed to spawn {label}"))?;
    run_captured(child, stdout, stderr, timeout, label)
}

/// [`run_with_timeout`] for a caller that wants to know how the run ended
/// rather than to have a timeout or a failure turned into an error: the same
/// executor, with the result as a [`CommandCapture`].
///
/// Nothing is an `Err` here. A program that cannot be started, a timeout and a
/// failure to wait are each an [`CommandOutcome`] of the capture, and a timeout
/// keeps what the child had written, so a caller that reads `stdout_raw` after
/// one gets the partial output [`run_with_timeout`] only quotes. The bounds are
/// [`run_with_timeout`]'s: the readers are abandoned after [`DRAIN_GRACE`], so
/// a descendant that keeps a pipe open does not extend the call.
pub(crate) fn run_bounded(mut command: Command, timeout: Duration, label: &str) -> CommandCapture {
    let argv = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|part| part.to_string_lossy().into_owned())
        .collect();
    match spawn_captured(&mut command) {
        Err(error) => CommandCapture::from_run(
            argv,
            CommandOutcome::from_spawn_error(error.kind()),
            None,
            Vec::new(),
            Vec::new(),
            Some(error.to_string()),
        ),
        Ok((child, stdout, stderr)) => {
            capture_of(argv, settle(child, stdout, stderr, timeout, label))
        }
    }
}

/// Start `command` with null stdin and both output pipes drained from the first
/// moment, which is what lets the wait loop be bounded at all.
fn spawn_captured(command: &mut Command) -> std::io::Result<(Child, Capture, Capture)> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // Taking the pipes here is what obliges us to assemble `Output` by hand
    // later: `wait_with_output` reads the handles it owns, and these are gone.
    let stdout = Capture::draining(child.stdout.take());
    let stderr = Capture::draining(child.stderr.take());
    Ok((child, stdout, stderr))
}

/// How a run ended and what it left behind, before it is turned into either a
/// `Result<Output>` or a [`CommandCapture`].
enum Settled {
    /// The child was waited for: it exited, or the deadline killed it.
    Waited {
        status: ExitStatus,
        timed_out: bool,
        /// Whether both readers reached end of output before the drain grace
        /// ended. When not, something other than the child still holds a pipe
        /// and the bytes are what arrived by then.
        drained: bool,
        stdout: (Vec<u8>, Option<std::io::Error>),
        stderr: (Vec<u8>, Option<std::io::Error>),
    },
    /// Polling or reaping the child failed, and it was killed.
    WaitFailed {
        error: anyhow::Error,
        stdout: (Vec<u8>, Option<std::io::Error>),
        stderr: (Vec<u8>, Option<std::io::Error>),
    },
}

/// The wait, the drain grace and the `take()` of both captures: everything
/// between a child being spawned and its result being formed.
///
/// Generic over [`Waitable`] rather than taking a `Child` so that the wait
/// loop's two failure paths — which are the only ways out of this function that
/// skip the drain grace — are reachable from a test.
fn settle<C: Waitable>(
    mut child: C,
    stdout: Capture,
    stderr: Capture,
    timeout: Duration,
    label: &str,
) -> Settled {
    // The single error path between `Capture::draining` and the `take()` pair
    // below, and it must take both captures before it returns. `take()` is
    // what marks a reader abandoned, and an abandoned reader is what stops a
    // buffer nobody will read again from growing for as long as a grandchild
    // holds the pipe open. Returning straight out of the wait loop — the shape
    // this replaced — left both readers appending forever, in a daemon that
    // probes every 60 seconds.
    let (status, timed_out) = match wait_bounded(&mut child, timeout, label, thread::sleep) {
        Ok(outcome) => outcome,
        Err(error) => {
            return Settled::WaitFailed {
                error,
                stdout: stdout.take(),
                stderr: stderr.take(),
            };
        }
    };

    let drained = wait_for_readers(&stdout, &stderr, DRAIN_GRACE, thread::sleep);
    Settled::Waited {
        status,
        timed_out,
        drained,
        stdout: stdout.take(),
        stderr: stderr.take(),
    }
}

/// Give both readers up to `grace` to reach end of output, looking again after
/// each `sleep` of the back-off schedule, and say whether they did.
///
/// `sleep` is a parameter so a test can see the schedule the loop really
/// follows instead of inferring it from [`next_poll_interval`] alone.
fn wait_for_readers(
    stdout: &Capture,
    stderr: &Capture,
    grace: Duration,
    mut sleep: impl FnMut(Duration),
) -> bool {
    let drained_by = Instant::now() + grace;
    let mut pause = FIRST_POLL;
    while !(stdout.is_finished() && stderr.is_finished()) && Instant::now() < drained_by {
        sleep(pause);
        pause = next_poll_interval(pause);
    }
    stdout.is_finished() && stderr.is_finished()
}

/// The rest of [`run_with_timeout`] once the child is spawned and both readers
/// are running.
fn run_captured<C: Waitable>(
    child: C,
    stdout: Capture,
    stderr: Capture,
    timeout: Duration,
    label: &str,
) -> Result<Output> {
    match settle(child, stdout, stderr, timeout, label) {
        Settled::WaitFailed { error, .. } => Err(error),
        Settled::Waited {
            status,
            timed_out,
            stdout,
            stderr,
            ..
        } => assemble_output(status, timed_out, stdout, stderr, label, timeout),
    }
}

/// A [`Settled`] run as a [`CommandCapture`], keeping the bytes a timeout or a
/// read error would otherwise have turned into an error.
fn capture_of(argv: Vec<String>, settled: Settled) -> CommandCapture {
    match settled {
        Settled::WaitFailed {
            error,
            stdout,
            stderr,
        } => CommandCapture::from_run(
            argv,
            CommandOutcome::WaitFailed,
            None,
            stdout.0,
            stderr.0,
            Some(format!("{error:#}")),
        ),
        Settled::Waited {
            status,
            timed_out,
            drained,
            stdout: (stdout, stdout_failure),
            stderr: (stderr, stderr_failure),
        } => {
            let outcome = if timed_out {
                CommandOutcome::TimedOut
            } else {
                command_capture::outcome_of_status(status)
            };
            // What cut the output short, if anything: a pipe that failed to
            // read, and/or a descendant still writing when the grace ended. The
            // outcome stays what the child did, and the bytes are what arrived.
            let notes: Vec<String> = [
                stdout_failure.map(|error| format!("failed to read stdout: {error}")),
                stderr_failure.map(|error| format!("failed to read stderr: {error}")),
                (!drained).then(|| STILL_ARRIVING.to_owned()),
            ]
            .into_iter()
            .flatten()
            .collect();
            let detail = (!notes.is_empty()).then(|| notes.join("; "));
            CommandCapture::from_run(argv, outcome, Some(status), stdout, stderr, detail)
        }
    }
}

/// Wait for `child` to exit, killing it once `timeout` has passed.
///
/// Returns the status and whether the deadline is what ended the wait. Kept
/// free of the captures so that it has exactly one caller to get the
/// abandon-on-error handling right, rather than one per `?`. `sleep` is what
/// the loop waits with between polls, a parameter so a test can see the
/// back-off schedule it follows.
fn wait_bounded<C: Waitable>(
    child: &mut C,
    timeout: Duration,
    label: &str,
    mut sleep: impl FnMut(Duration),
) -> Result<(ExitStatus, bool)> {
    let deadline = Instant::now() + timeout;
    let mut pause = FIRST_POLL;
    loop {
        let polled = child.try_wait();
        if polled.is_err() {
            // The only exit from this function that is not already followed by
            // a kill. Whatever stopped us polling, we are about to stop
            // waiting, and the caller drops the child without ever reaping it;
            // leaving it running would hand back a process nobody is watching.
            // `let _` to match the timeout branch: the kill is best-effort, and
            // the poll error is the one worth reporting.
            let _ = child.kill();
        }
        if let Some(status) = polled.with_context(|| format!("failed to poll {label}"))? {
            return Ok((status, false));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("failed to reap timed-out {label}"))?;
            return Ok((status, true));
        }
        sleep(pause);
        pause = next_poll_interval(pause);
    }
}

/// What a capture's `detail` says when the drain grace ended with a reader still
/// waiting for end of output, so the caller can tell a complete capture from a
/// short one.
const STILL_ARRIVING: &str = "output was still arriving when the drain grace ended";

/// The three child operations [`wait_bounded`] needs.
///
/// A trait only so a test can fail `try_wait` or the post-kill `wait` on
/// demand: both are error paths that skip the drain grace, and provoking either
/// from a real child needs the kernel to refuse to reap a process.
trait Waitable {
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>>;
    fn kill(&mut self) -> std::io::Result<()>;
    fn wait(&mut self) -> std::io::Result<ExitStatus>;
}

// `Child::` rather than `Self::` on purpose: inside this impl `Self` is
// `Child`, so `Self::try_wait` reads as the trait method being defined
// here rather than the inherent one being forwarded to.
#[allow(clippy::use_self)]
impl Waitable for Child {
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        Child::try_wait(self)
    }

    fn kill(&mut self) -> std::io::Result<()> {
        Child::kill(self)
    }

    fn wait(&mut self) -> std::io::Result<ExitStatus> {
        Child::wait(self)
    }
}

/// Turn what the wait loop and the two readers produced into the call's result.
///
/// Split out from [`run_with_timeout`] because it is the only way to test it.
/// Every branch here is plain logic over values already in hand — which error
/// wins, which stream gets quoted — but reaching it through a real child means
/// making a pipe fail on demand, which needs a kernel having a bad day. Given
/// the two `(bytes, error)` pairs directly, each branch is an ordinary unit
/// test; see the cases at the bottom of this file.
fn assemble_output(
    status: ExitStatus,
    timed_out: bool,
    (stdout, stdout_failure): (Vec<u8>, Option<std::io::Error>),
    (stderr, stderr_failure): (Vec<u8>, Option<std::io::Error>),
    label: &str,
    timeout: Duration,
) -> Result<Output> {
    let output = Output {
        status,
        stdout,
        stderr,
    };

    if timed_out {
        // `command_failure_detail` is the crate's one "which stream explains
        // this failure" rule — stderr, else stdout, else "no output" — shared
        // since #584 so callers cannot quietly disagree about the order.
        //
        // Truncated, not dropped: what the child managed to say before it
        // stalled is usually the whole explanation, and the timeout is named
        // first either way, so a reader who only sees the head still learns
        // which command gave up and why.
        bail!(
            "{label} exceeded {}s timeout: {}",
            timeout.as_secs(),
            truncate_to_chars(
                crate::command_failure_detail(&output),
                MAX_QUOTED_OUTPUT_CHARS,
            )
        );
    }

    // Only once the timeout has had its say: a killed child whose pipe then
    // fails to read is a timeout, and reporting it as a read error would name
    // the symptom instead of the cause.
    if let Some(error) = stdout_failure.or(stderr_failure) {
        return Err(anyhow::Error::new(error))
            .with_context(|| format!("failed to read the output of {label}"));
    }

    Ok(output)
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

    /// The shells every grandchild case below is run against.
    ///
    /// Whether a shell forks or execs decides whether a grandchild exists at
    /// all, so it decides whether these tests test anything: dash forks `sleep`
    /// in `sh -c "sleep 30"`, bash execs it into the shell's own process and no
    /// grandchild is left holding the pipe. A test written against one is dead
    /// weight on a host running the other, and which one `/bin/sh` is varies
    /// across the distributions this ships on. So the scripts below are written
    /// to fork under either, and each case runs under every shell present to
    /// keep that true.
    #[cfg(unix)]
    fn grandchild_shells() -> Vec<&'static str> {
        let shells: Vec<&'static str> = ["/bin/sh", "/bin/bash", "/bin/dash"]
            .into_iter()
            .filter(|shell| std::path::Path::new(shell).exists())
            .collect();
        assert!(
            !shells.is_empty(),
            "no shell to run the grandchild cases in"
        );
        shells
    }

    #[test]
    fn a_command_that_succeeds_reports_its_output() {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            // No space around `&`: cmd would carry a trailing one into the
            // echoed text, which `trim` hides on stdout but is easier not to
            // write than to explain.
            command.args(["/C", "echo hello&echo oops 1>&2"]);
        } else {
            command.args(["-c", "echo hello; echo oops >&2"]);
        }

        let output = run_with_timeout(command, Duration::from_secs(10), "process").expect("ran");

        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hello");
        // Asserted on Windows too: `1>&2` is a cmd redirection like any other,
        // and skipping it there left the second pipe — the one whose separate
        // reader this function exists to provide — unchecked on that platform.
        assert_eq!(String::from_utf8_lossy(&output.stderr).trim(), "oops");
    }

    /// What a stdin-reading fixture prints when its read reaches EOF.
    const STDIN_EOF_MARKER: &str = "eof";

    /// The assertions shared by both platforms' empty-stdin cases.
    ///
    /// `command` must run a child that reads one line from stdin and prints
    /// [`STDIN_EOF_MARKER`] if that read reaches EOF instead.
    ///
    /// The stdin set on `command` here is the read end of a pipe whose write end
    /// this function holds open across the whole call, so it never reaches EOF
    /// of its own accord. That is what makes the assertion two-sided rather than
    /// a tautology: were `run_with_timeout` to stop setting null stdin, the
    /// child would sit in `read` until the deadline killed it — exactly how a
    /// child that prompts behaves against an inherited terminal — and the call
    /// would come back as a timeout rather than with the marker.
    ///
    /// Setting stdin on the caller's side is also the contract under test: this
    /// function's own stdio overrides whatever the caller set, which is what
    /// lets all four callers hand over a bare `Command` and say nothing about
    /// stdin at all.
    ///
    /// One thing it does not pin, said plainly rather than left to be
    /// discovered: `null` against `inherit`. Telling those two apart needs this
    /// test process's own fd 0 to be something other than EOF, and arranging
    /// that is a `dup2` racing every other test in the binary. An `inherit`
    /// mutant was run and does pass here. What is caught is the regression that
    /// can actually happen -- no stdin setting at all, which is the line this
    /// function took over from its callers.
    ///
    /// The race is avoidable by re-execing this test binary under a stdin pipe
    /// the outer test holds open and asserting in that child, which would fail
    /// the `inherit` mutant too. Not done: it buys one mutant that no plausible
    /// edit produces -- nothing would replace `null` with `inherit`, the stdio
    /// mode this helper exists to stop callers from being handed -- at the cost
    /// of a self-re-execing test whose own harness plumbing is then the thing
    /// most likely to break.
    fn assert_the_child_is_given_an_empty_stdin(mut command: Command) {
        let (stdin_reader, stdin_writer) =
            std::io::pipe().expect("failed to create a pipe for the child's stdin");
        command.stdin(Stdio::from(stdin_reader));

        let started = Instant::now();
        let output = run_with_timeout(command, Duration::from_secs(5), "process").expect(
            "a child given an empty stdin must read EOF rather than block until the deadline",
        );
        let elapsed = started.elapsed();
        // Dropped only here: releasing the write end any earlier would close the
        // pipe, so the child would reach EOF through the caller's stdin and the
        // test would pass whatever this function set.
        drop(stdin_writer);

        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            STDIN_EOF_MARKER,
            "the child must be given an empty stdin rather than the one the caller set"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "the read must hit EOF at once rather than waiting out the deadline, but the call \
             took {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_child_that_reads_stdin_is_given_an_empty_one() {
        // `read` reports EOF by returning non-zero, so the `||` branch is the
        // one an empty stdin takes. A stdin left open takes neither.
        let script = format!("read line || echo {STDIN_EOF_MARKER}");
        let mut command = Command::new("sh");
        command.args(["-c", script.as_str()]);

        assert_the_child_is_given_an_empty_stdin(command);
    }

    #[cfg(unix)]
    #[test]
    fn a_genuinely_slow_command_times_out_on_time() {
        for shell in grandchild_shells() {
            // `sleep 30 & wait` rather than plain `sleep 30`: a lone final
            // command is exec-ed into the shell itself by bash, which would
            // leave nothing behind to hold the pipe and quietly turn this into
            // a test of the easy case. Backgrounding forces a real grandchild
            // under every shell.
            //
            // Killing the shell at the deadline does not kill that grandchild,
            // so it keeps the inherited write end of both pipes open and no EOF
            // arrives. An implementation that joins its readers therefore waits
            // out the full 30 seconds and the timeout bounds nothing — which
            // for the 2-second amd-smi probe is the difference between a snappy
            // telemetry read and a stalled one.
            let mut command = Command::new(shell);
            command.args(["-c", "sleep 30 & wait"]);

            let started = Instant::now();
            let error = run_with_timeout(command, Duration::from_secs(1), "process")
                .expect_err("a command that outlives its timeout must fail");
            let elapsed = started.elapsed();

            assert!(
                error.to_string().contains("exceeded 1s timeout"),
                "the timeout must be reported as one, under {shell}: {error}"
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "the call must return at its deadline rather than waiting for an abandoned \
                 grandchild to exit, but under {shell} it took {elapsed:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_grandchild_holding_the_pipe_does_not_hold_a_successful_call() {
        for shell in grandchild_shells() {
            // The timeout case above covers a child that is killed. This is the
            // same hazard on the path that has no timeout to blame: the child
            // exits 0 straight away, having left a background `sleep` holding
            // the write end of both pipes, so there is no EOF for another five
            // seconds. Joining the readers — or giving them an unbounded grace
            // — hands the schedule to that `sleep` and a command that finished
            // instantly takes 5s. `DRAIN_GRACE` is what bounds it.
            let mut command = Command::new(shell);
            command.args(["-c", "echo ready; sleep 5 &"]);

            let started = Instant::now();
            let output = run_with_timeout(command, Duration::from_secs(20), "process")
                .expect("a child that exits cleanly must not be reported as a timeout");
            let elapsed = started.elapsed();

            assert!(output.status.success(), "under {shell}: {output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout).trim(),
                "ready",
                "what the child wrote before exiting must still arrive, under {shell}"
            );
            // Generous against a loaded machine, and still a third of the
            // grandchild's lifetime: the regression this guards takes the full
            // five seconds, not a few hundred milliseconds more than the grace.
            assert!(
                elapsed < Duration::from_secs(3),
                "the call must not wait for the grandchild, but under {shell} it took {elapsed:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn output_arriving_after_the_child_exits_is_still_collected() {
        for shell in grandchild_shells() {
            // What `DRAIN_GRACE` is for, and the only way to ask for it on
            // demand. A child that writes and exits normally cannot pin it: its
            // reader is already running and drains the pipe microseconds later,
            // long before the wait loop's next poll even notices the exit, so
            // taking the buffer the instant the child is seen to have exited
            // still collects everything. Here a grandchild writes once the
            // child is gone instead.
            //
            // The 0.1s sits between the two bounds that decide the outcome:
            // past `POLL_INTERVAL` (50ms), the longest the wait loop can take
            // to notice the exit, so taking the output there misses
            // `late-canary` outright; and well inside `DRAIN_GRACE` (200ms), so
            // draining collects it. Both margins are ~2x, and the grandchild
            // exits immediately after writing, so a working implementation sees
            // EOF and returns rather than sitting out the whole grace.
            let mut command = Command::new(shell);
            command.args(["-c", "echo early-canary; (sleep 0.1; echo late-canary) &"]);

            let started = Instant::now();
            let output = run_with_timeout(command, Duration::from_secs(20), "process")
                .expect("a child that exits cleanly must not be reported as a timeout");
            let elapsed = started.elapsed();
            let stdout = String::from_utf8_lossy(&output.stdout);

            assert!(
                stdout.contains("early-canary"),
                "under {shell}, the child's own output is missing: {stdout:?}"
            );
            assert!(
                stdout.contains("late-canary"),
                "under {shell}, output still in flight when the child exited was dropped \
                 instead of drained: {stdout:?}"
            );
            assert!(
                elapsed < Duration::from_secs(3),
                "draining must end at EOF rather than running long, but under {shell} the call \
                 took {elapsed:?}"
            );
        }
    }

    #[test]
    fn an_interrupted_read_is_retried_and_a_failed_one_is_reported() {
        // Neither branch is reachable through a real child here — EINTR needs a
        // signal to land mid-read, and a pipe read failing outright needs the
        // kernel to be having a bad day — so the reader is fed a scripted
        // stream directly. The property is the one `read_to_end` gave before
        // this function replaced it: an interrupted read is not the end of the
        // stream, and a failed one is not a successful short read.
        struct ScriptedPipe(std::collections::VecDeque<std::io::Result<&'static [u8]>>);

        impl Read for ScriptedPipe {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                match self.0.pop_front() {
                    Some(Ok(bytes)) => {
                        out[..bytes.len()].copy_from_slice(bytes);
                        Ok(bytes.len())
                    }
                    Some(Err(error)) => Err(error),
                    None => Ok(0),
                }
            }
        }

        fn drain(script: Vec<std::io::Result<&'static [u8]>>) -> (Vec<u8>, Option<std::io::Error>) {
            let capture = Capture::draining(Some(ScriptedPipe(script.into())));
            let deadline = Instant::now() + Duration::from_secs(10);
            while !capture.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(capture.is_finished(), "the reader never finished");
            capture.take()
        }

        let (bytes, failure) = drain(vec![
            Ok(b"before"),
            Err(std::io::Error::from(ErrorKind::Interrupted)),
            Ok(b"after"),
            Ok(b""),
        ]);
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            "beforeafter",
            "an interrupted read must be retried, not taken for the end of the stream"
        );
        assert!(failure.is_none(), "{failure:?}");

        let (bytes, failure) = drain(vec![
            Ok(b"partial"),
            Err(std::io::Error::from(ErrorKind::BrokenPipe)),
            Ok(b"unreachable"),
        ]);
        assert_eq!(String::from_utf8_lossy(&bytes), "partial");
        assert_eq!(
            failure.map(|error| error.kind()),
            Some(ErrorKind::BrokenPipe),
            "a read that failed must be reported rather than passed off as EOF"
        );
    }

    /// A pipe that never ends until the test says so.
    ///
    /// Stoppable rather than genuinely endless so the reader thread finishes
    /// with the test instead of spinning for the life of the test binary.
    struct EndlessPipe {
        reads: Arc<std::sync::atomic::AtomicUsize>,
        stop: Arc<AtomicBool>,
    }

    impl Read for EndlessPipe {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.stop.load(Ordering::Acquire) {
                return Ok(0);
            }
            // Paced so the test can interleave with the reader rather than
            // racing a tight loop, and so a regression accumulates a size the
            // assertion can see without the test allocating gigabytes.
            thread::sleep(Duration::from_millis(1));
            let filled = out.len().min(4096);
            out[..filled].fill(b'x');
            self.reads.fetch_add(1, Ordering::Release);
            Ok(filled)
        }
    }

    #[test]
    fn an_abandoned_reader_stops_filling_a_buffer_nobody_will_read() {
        // The shape this guards: the child exited but a grandchild still holds
        // the write end, so `DRAIN_GRACE` expires, `take()` hands the caller
        // what there is and the reader is left running. In `rocmd` — a daemon
        // that probes amd-smi every 60s and runs MCP captures besides — a
        // reader that kept appending would grow a `Vec` that nothing will ever
        // read, for as long as the grandchild lives. There is no way to ask
        // `run_with_timeout` for that state on demand, so the capture is driven
        // directly, which is also where the invariant lives.
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let capture = Capture::draining(Some(EndlessPipe {
            reads: Arc::clone(&reads),
            stop: Arc::clone(&stop),
        }));

        let wait_for_reads = |target: usize| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while reads.load(Ordering::Acquire) < target && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                reads.load(Ordering::Acquire) >= target,
                "the reader stalled before {target} reads"
            );
        };

        wait_for_reads(4);
        let (bytes, failure) = capture.take();
        // Establishes that the reader really was accumulating up to this point,
        // so the emptiness asserted below is the flag working rather than a
        // reader that never ran.
        assert!(
            !bytes.is_empty(),
            "the reader must collect while someone is still listening"
        );
        assert!(failure.is_none(), "{failure:?}");

        // Well past what fits in anything the implementation might buffer
        // internally: 50 more 4KiB reads is ~200KB, which a regression leaves
        // sitting in the buffer and a correct implementation discards.
        wait_for_reads(reads.load(Ordering::Acquire) + 50);
        let leaked = capture.lock().len();
        assert_eq!(
            leaked, 0,
            "after `take()` nothing will read this buffer again, so a reader that keeps \
             appending grows it without bound for the life of the daemon; {leaked} bytes \
             accumulated"
        );

        stop.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !capture.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(capture.is_finished(), "the reader never finished");
    }

    /// A child that never exits on its own and fails whichever of the wait
    /// loop's two calls the test asks it to.
    struct FailingChild {
        /// `true`: `try_wait` reports an error rather than "still running", so
        /// the loop's first `?` fires. `false`: it stays running until the
        /// deadline, and the post-kill `wait` is what fails.
        fail_poll: bool,
        /// Raised by `kill`, so the test can tell a wait loop that leaves the
        /// child running from one that does not. This is the whole reason
        /// `kill` is on the trait rather than called on a concrete `Child`.
        killed: Arc<AtomicBool>,
    }

    impl Waitable for FailingChild {
        fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
            if self.fail_poll {
                return Err(std::io::Error::from(ErrorKind::PermissionDenied));
            }
            Ok(None)
        }

        fn kill(&mut self) -> std::io::Result<()> {
            self.killed.store(true, Ordering::Release);
            Ok(())
        }

        fn wait(&mut self) -> std::io::Result<ExitStatus> {
            Err(std::io::Error::from(ErrorKind::PermissionDenied))
        }
    }

    /// Drive `run_captured` with a child that fails one of the wait loop's two
    /// calls, and assert the error is reported, both readers are left
    /// abandoned, *and* the child was killed on the way out.
    ///
    /// `fail_poll` picks which call fails; see [`FailingChild`].
    fn assert_wait_failure_abandons_both_readers(
        fail_poll: bool,
        timeout: Duration,
        expected: &str,
    ) {
        let killed = Arc::new(AtomicBool::new(false));
        let child = FailingChild {
            fail_poll,
            killed: Arc::clone(&killed),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let out_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let err_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stdout = Capture::draining(Some(EndlessPipe {
            reads: Arc::clone(&out_reads),
            stop: Arc::clone(&stop),
        }));
        let stderr = Capture::draining(Some(EndlessPipe {
            reads: Arc::clone(&err_reads),
            stop: Arc::clone(&stop),
        }));

        let wait_for_reads = |counter: &Arc<std::sync::atomic::AtomicUsize>, target: usize| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while counter.load(Ordering::Acquire) < target && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                counter.load(Ordering::Acquire) >= target,
                "the reader stalled before {target} reads"
            );
        };

        // Both readers are accumulating before the call, so an empty buffer
        // afterwards is the abandon working rather than a reader that never ran.
        wait_for_reads(&out_reads, 4);
        wait_for_reads(&err_reads, 4);

        let error = run_captured(child, stdout.clone(), stderr.clone(), timeout, "process")
            .expect_err("a failed wait must be reported, not swallowed");
        assert!(
            format!("{error:#}").contains(expected),
            "expected {expected:?} in {error:#}"
        );

        // ~200KB more per stream: what a reader that was never marked abandoned
        // accumulates into a buffer nobody will ever read again.
        wait_for_reads(&out_reads, out_reads.load(Ordering::Acquire) + 50);
        wait_for_reads(&err_reads, err_reads.load(Ordering::Acquire) + 50);
        for (name, capture) in [("stdout", &stdout), ("stderr", &stderr)] {
            let leaked = capture.lock().len();
            assert_eq!(
                leaked, 0,
                "an error out of the wait loop returns before the `take()` pair, so it must \
                 abandon both readers itself or they grow a buffer nobody will read for as \
                 long as a grandchild holds the pipe; {name} accumulated {leaked} bytes"
            );
        }

        // Abandoning the readers is only half of it: the caller drops the child
        // on this path without reaping it, so a wait loop that returns without
        // killing leaves a process running that nobody will ever wait on again.
        assert!(
            killed.load(Ordering::Acquire),
            "every exit from the wait loop must kill the child first; this one returned the \
             error with the child still running"
        );

        stop.store(true, Ordering::Release);
    }

    #[test]
    fn a_failure_to_poll_the_child_still_abandons_both_readers() {
        assert_wait_failure_abandons_both_readers(
            true,
            Duration::from_secs(30),
            "failed to poll process",
        );
    }

    #[test]
    fn a_failure_to_reap_a_timed_out_child_still_abandons_both_readers() {
        // Zero timeout so the loop reaches the post-kill reap — the other `?` —
        // on its first pass.
        assert_wait_failure_abandons_both_readers(
            false,
            Duration::ZERO,
            "failed to reap timed-out process",
        );
    }

    /// A successful `ExitStatus` to feed `assemble_output`.
    ///
    /// `ExitStatus` has no portable public constructor; the per-platform
    /// `ExitStatusExt` is the supported way to make one without spawning a
    /// process purely to obtain it.
    fn success_status() -> ExitStatus {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt as _;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt as _;
        ExitStatus::from_raw(0)
    }

    #[test]
    fn the_poll_interval_backs_off_from_a_few_milliseconds_to_the_flat_one() {
        // The flat 50 ms this loop had would make a probe that exits at once take
        // 50 ms, where the helpers it replaces polled every 20 to 25 ms.
        let mut pause = FIRST_POLL;
        let mut seen = vec![pause];
        for _ in 0..5 {
            pause = next_poll_interval(pause);
            seen.push(pause);
        }
        let millis: Vec<u128> = seen.iter().map(Duration::as_millis).collect();
        assert_eq!(millis, [5, 10, 20, 40, 50, 50]);
        assert!(FIRST_POLL < Duration::from_millis(20));
    }

    /// A child that has not exited for its first `remaining` polls.
    struct ExitsAfterPolls {
        remaining: usize,
    }

    impl Waitable for ExitsAfterPolls {
        fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
            if self.remaining == 0 {
                return Ok(Some(success_status()));
            }
            self.remaining -= 1;
            Ok(None)
        }

        fn kill(&mut self) -> std::io::Result<()> {
            Ok(())
        }

        fn wait(&mut self) -> std::io::Result<ExitStatus> {
            Ok(success_status())
        }
    }

    fn millis(slept: &[Duration]) -> Vec<u128> {
        slept.iter().map(Duration::as_millis).collect()
    }

    #[test]
    fn the_wait_loop_sleeps_the_back_off_schedule_between_polls() {
        // The sleeps are recorded, not taken, so the schedule the loop really
        // follows is what is asserted. A loop that slept `POLL_INTERVAL` every
        // time would charge a probe that exits at once the full 50 ms.
        let mut slept = Vec::new();
        let mut child = ExitsAfterPolls { remaining: 6 };

        let (status, timed_out) =
            wait_bounded(&mut child, Duration::from_mins(1), "process", |pause| {
                slept.push(pause);
            })
            .expect("the child exits on the seventh poll");

        assert!(status.success() && !timed_out);
        assert_eq!(millis(&slept), [5, 10, 20, 40, 50, 50]);
    }

    /// Two readers over [`EndlessPipe`]s that stop when `stop` is raised.
    fn two_endless_captures(stop: &Arc<AtomicBool>) -> (Capture, Capture) {
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipe = || EndlessPipe {
            reads: Arc::clone(&reads),
            stop: Arc::clone(stop),
        };
        (
            Capture::draining(Some(pipe())),
            Capture::draining(Some(pipe())),
        )
    }

    #[test]
    fn the_drain_wait_sleeps_the_back_off_schedule_until_both_readers_finish() {
        let stop = Arc::new(AtomicBool::new(false));
        let (stdout, stderr) = two_endless_captures(&stop);
        let mut slept = Vec::new();

        let drained = wait_for_readers(&stdout, &stderr, Duration::from_secs(30), |pause| {
            slept.push(pause);
            if slept.len() == 3 {
                // End the pipes, and give the reader threads real time to see it.
                stop.store(true, Ordering::Release);
                thread::sleep(Duration::from_millis(100));
            }
        });

        assert!(drained, "both readers reached the end of their pipes");
        assert_eq!(millis(&slept[..3]), [5, 10, 20]);
    }

    #[test]
    fn a_drain_wait_that_runs_out_of_grace_says_the_readers_were_still_going() {
        let stop = Arc::new(AtomicBool::new(false));
        let (stdout, stderr) = two_endless_captures(&stop);

        let drained = wait_for_readers(&stdout, &stderr, Duration::ZERO, |_| {
            unreachable!("no grace, so no sleep")
        });

        stop.store(true, Ordering::Release);
        assert!(!drained);
    }

    /// Drive `settle` for a child that fails the wait loop the way
    /// [`FailingChild`] says, over two readers that are accumulating.
    fn settle_a_failing_child(fail_poll: bool, timeout: Duration) -> (Settled, bool) {
        let killed = Arc::new(AtomicBool::new(false));
        let child = FailingChild {
            fail_poll,
            killed: Arc::clone(&killed),
        };
        let stop = Arc::new(AtomicBool::new(false));
        // One counter per stream: a single shared one is satisfied by one reader
        // doing all the reads, which leaves the other stream's bytes to chance.
        let out_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let err_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipe = |reads: &Arc<std::sync::atomic::AtomicUsize>| EndlessPipe {
            reads: Arc::clone(reads),
            stop: Arc::clone(&stop),
        };
        let stdout = Capture::draining(Some(pipe(&out_reads)));
        let stderr = Capture::draining(Some(pipe(&err_reads)));
        let deadline = Instant::now() + Duration::from_secs(10);
        while (out_reads.load(Ordering::Acquire) < 4 || err_reads.load(Ordering::Acquire) < 4)
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            out_reads.load(Ordering::Acquire) >= 4 && err_reads.load(Ordering::Acquire) >= 4,
            "both readers must have started"
        );

        let settled = settle(child, stdout, stderr, timeout, "process");

        stop.store(true, Ordering::Release);
        (settled, killed.load(Ordering::Acquire))
    }

    #[test]
    fn a_failure_to_wait_is_a_capture_that_keeps_the_bytes_that_had_arrived() {
        // `run_raw` returned `124` with the bytes joined after a failed poll, so
        // the capture keeps them rather than dropping them with the error, and
        // the child is killed on the way out.
        for (fail_poll, timeout, expected) in [
            (true, Duration::from_secs(30), "failed to poll process"),
            (false, Duration::ZERO, "failed to reap timed-out process"),
        ] {
            let (settled, killed) = settle_a_failing_child(fail_poll, timeout);
            assert!(killed, "{expected}: the child must be killed");

            let capture = capture_of(vec!["fake".to_owned()], settled);

            assert_eq!(capture.outcome, CommandOutcome::WaitFailed, "{expected}");
            assert!(capture.status.is_none(), "{expected}");
            assert!(
                capture.stdout_raw.len() >= 4096 && capture.stdout_raw.iter().all(|&b| b == b'x'),
                "{expected}: stdout bytes: {}",
                capture.stdout_raw.len()
            );
            assert!(
                capture.stderr_raw.len() >= 4096 && capture.stderr_raw.iter().all(|&b| b == b'x'),
                "{expected}: stderr bytes: {}",
                capture.stderr_raw.len()
            );
            assert_eq!(capture.exit_status, 1, "{expected}");
            let detail = capture.detail.expect("the wait error is kept in words");
            assert!(detail.contains(expected), "{detail}");
        }
    }

    fn waited(
        timed_out: bool,
        drained: bool,
        stdout: (Vec<u8>, Option<std::io::Error>),
        stderr: (Vec<u8>, Option<std::io::Error>),
    ) -> Settled {
        Settled::Waited {
            status: success_status(),
            timed_out,
            drained,
            stdout,
            stderr,
        }
    }

    #[test]
    fn a_timed_out_settle_is_a_capture_with_the_partial_output_and_the_outcome() {
        let capture = capture_of(
            vec!["fake".to_owned()],
            waited(
                true,
                true,
                (b"partial".to_vec(), None),
                (b"warning".to_vec(), None),
            ),
        );

        assert_eq!(capture.outcome, CommandOutcome::TimedOut);
        assert_eq!(capture.stdout_raw, b"partial");
        assert_eq!(capture.stderr, "warning");
        assert_eq!(capture.exit_status, 1, "a timeout has no exit code");
        assert!(capture.detail.is_none());
    }

    #[test]
    fn a_read_failure_on_either_stream_keeps_the_bytes_and_the_outcome_the_child_had() {
        // `run_with_timeout` turns this into an error and drops the bytes
        // (`a_read_failure_without_a_timeout_is_reported_rather_than_returned_as_output`);
        // the capture keeps both, as the older helpers did with `let _ =`.
        let stdout_only = capture_of(
            vec!["fake".to_owned()],
            waited(
                false,
                true,
                (b"partial".to_vec(), read_failure()),
                (Vec::new(), None),
            ),
        );
        assert_eq!(stdout_only.outcome, CommandOutcome::Exited(0));
        assert_eq!(stdout_only.stdout, "partial");
        let detail = stdout_only.detail.expect("the read error is kept in words");
        assert!(detail.contains("failed to read stdout"), "{detail}");
        assert!(!detail.contains("stderr"), "{detail}");

        let stderr_only = capture_of(
            vec!["fake".to_owned()],
            waited(
                false,
                true,
                (b"out".to_vec(), None),
                (b"partial".to_vec(), read_failure()),
            ),
        );
        assert_eq!(stderr_only.outcome, CommandOutcome::Exited(0));
        assert_eq!(stderr_only.stderr, "partial");
        let detail = stderr_only.detail.expect("the read error is kept in words");
        assert!(detail.contains("failed to read stderr"), "{detail}");
        assert!(!detail.contains("stdout"), "{detail}");

        let both = capture_of(
            vec!["fake".to_owned()],
            waited(
                false,
                true,
                (Vec::new(), read_failure()),
                (Vec::new(), read_failure()),
            ),
        );
        let detail = both.detail.expect("both read errors are kept in words");
        assert!(
            detail.contains("failed to read stdout") && detail.contains("failed to read stderr"),
            "{detail}"
        );
    }

    #[test]
    fn a_capture_says_when_a_reader_was_still_going_at_the_end_of_the_grace() {
        let short = capture_of(
            vec!["fake".to_owned()],
            waited(false, false, (b"early".to_vec(), None), (Vec::new(), None)),
        );
        assert_eq!(short.outcome, CommandOutcome::Exited(0));
        assert_eq!(short.stdout, "early");
        assert_eq!(short.detail.as_deref(), Some(STILL_ARRIVING));

        let complete = capture_of(
            vec!["fake".to_owned()],
            waited(false, true, (b"early".to_vec(), None), (Vec::new(), None)),
        );
        assert!(complete.detail.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn output_written_after_the_grace_is_missing_and_the_capture_says_so() {
        // The descendant writes 0.5 s after the child exits, which is past the
        // 200 ms grace: the capture holds `early`, and `detail` says it is short.
        let mut command = Command::new("sh");
        command.args(["-c", "echo early; (sleep 0.5; echo late) &"]);

        let capture = run_bounded(command, Duration::from_secs(20), "process");

        assert_eq!(capture.outcome, CommandOutcome::Exited(0));
        assert_eq!(capture.stdout, "early\n");
        assert_eq!(capture.detail.as_deref(), Some(STILL_ARRIVING));
    }

    #[test]
    fn a_program_that_cannot_be_started_is_a_capture_not_an_error() {
        let missing = "/nonexistent-dir-for-the-runner-pair/tool";
        let mut command = Command::new(missing);
        command.arg("--flag");

        let capture = run_bounded(command, Duration::from_secs(5), "process");

        assert_eq!(capture.outcome, CommandOutcome::NotFound);
        assert_eq!(capture.argv, [missing, "--flag"]);
        assert!(capture.stdout_raw.is_empty() && capture.stderr_raw.is_empty());
        assert_eq!(capture.exit_status, 1);
        assert!(capture.status.is_none());
        assert!(capture.detail.is_some(), "the OS error is kept in words");
    }

    #[cfg(unix)]
    #[test]
    fn a_program_that_exists_but_cannot_be_executed_is_a_spawn_failure_not_not_found() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("not-executable");
        std::fs::write(&path, "#!/bin/sh\n").expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");

        // Not skipped for root: `execve` refuses a file with no execute bit for
        // every user.
        let capture = run_bounded(Command::new(&path), Duration::from_secs(5), "process");

        assert_eq!(
            capture.outcome,
            CommandOutcome::SpawnFailed(ErrorKind::PermissionDenied)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_run_is_a_capture_with_its_exit_code_or_its_signal_and_both_streams() {
        let capture = run_bounded(
            {
                let mut command = Command::new("sh");
                command.args(["-c", "echo out; echo err >&2; exit 3"]);
                command
            },
            Duration::from_secs(10),
            "process",
        );
        assert_eq!(capture.outcome, CommandOutcome::Exited(3));
        assert_eq!(capture.exit_status, 3);
        assert_eq!(capture.stdout, "out\n");
        assert_eq!(capture.stderr_raw, b"err\n");
        assert_eq!(capture.argv, ["sh", "-c", "echo out; echo err >&2; exit 3"]);
        assert_eq!(capture.status.and_then(|status| status.code()), Some(3));
        assert!(capture.detail.is_none());

        let capture = run_bounded(
            {
                let mut command = Command::new("sh");
                command.args(["-c", "echo partial; kill -9 $$"]);
                command
            },
            Duration::from_secs(10),
            "process",
        );
        assert_eq!(capture.outcome, CommandOutcome::Signaled(Some(9)));
        assert_eq!(capture.exit_status, 1);
        assert_eq!(capture.stdout, "partial\n");
    }

    #[cfg(windows)]
    #[test]
    fn a_run_is_a_capture_with_its_exit_code_and_both_streams() {
        let mut command = Command::new("cmd");
        command.args(["/C", "echo out&echo err 1>&2&exit 3"]);

        let capture = run_bounded(command, Duration::from_secs(10), "process");

        assert_eq!(capture.outcome, CommandOutcome::Exited(3));
        assert_eq!(capture.stdout.trim(), "out");
        assert_eq!(capture.stderr.trim(), "err");
    }

    #[cfg(unix)]
    #[test]
    fn a_timeout_is_a_capture_with_what_the_child_had_written() {
        // `exec` so killing the shell leaves nothing on the pipe: what is under
        // test is the partial output and the outcome, not the join.
        let mut command = Command::new("sh");
        command.args(["-c", "echo before; exec sleep 30"]);

        // Room for a loaded host to start the shell and run its `echo` before the
        // deadline, since the assertion is on what the child had written by then.
        let started = Instant::now();
        let capture = run_bounded(command, Duration::from_secs(3), "process");

        assert_eq!(capture.outcome, CommandOutcome::TimedOut);
        assert_eq!(capture.stdout, "before\n");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    fn read_failure() -> Option<std::io::Error> {
        Some(std::io::Error::from(ErrorKind::BrokenPipe))
    }

    #[test]
    fn a_read_failure_without_a_timeout_is_reported_rather_than_returned_as_output() {
        // The silent short read this function exists to prevent: handing back
        // `Ok` with a truncated buffer sends the caller off to parse half a
        // JSON document and blame the document.
        let error = assemble_output(
            success_status(),
            false,
            (b"partial".to_vec(), read_failure()),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a stream that failed to read must not come back as success");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to read the output of process"),
            "{rendered:?}"
        );
    }

    #[test]
    fn a_failure_on_either_stream_is_reported() {
        // stdout is checked first, so a stderr-only failure is a separate
        // branch: `stdout_failure.or(stderr_failure)` reduced to
        // `stdout_failure` would still pass the case above.
        let error = assemble_output(
            success_status(),
            false,
            (Vec::new(), None),
            (b"partial".to_vec(), read_failure()),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a stderr stream that failed to read must not come back as success");

        assert!(
            format!("{error:#}").contains("failed to read the output of process"),
            "{error:#}"
        );
    }

    #[test]
    fn a_timeout_outranks_a_read_failure() {
        // Killing the child is what broke the pipe, so a read error here is the
        // symptom. Reporting it would name the consequence and hide the cause,
        // and the caller would go looking for a kernel problem.
        let error = assemble_output(
            success_status(),
            true,
            (Vec::new(), read_failure()),
            (b"the real explanation".to_vec(), read_failure()),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("exceeded 2s timeout"),
            "the timeout is the cause and must be what is reported: {rendered:?}"
        );
        assert!(
            rendered.contains("the real explanation"),
            "the child's own account must survive: {rendered:?}"
        );
        assert!(
            !rendered.contains("failed to read the output of"),
            "the read error is downstream of the kill and must not displace it: {rendered:?}"
        );
    }

    #[test]
    fn a_timeout_quotes_stderr_in_preference_to_stdout() {
        let error = assemble_output(
            success_status(),
            true,
            (b"stdout-data".to_vec(), None),
            (b"stderr-explanation".to_vec(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("stderr-explanation"),
            "stderr is where a child explains itself: {rendered:?}"
        );
        assert!(
            !rendered.contains("stdout-data"),
            "quoting both would push the explanation behind a payload: {rendered:?}"
        );
    }

    #[test]
    fn a_timeout_falls_back_to_stdout_and_then_to_saying_there_was_none() {
        // Every other timeout fixture writes to stderr, so without these two
        // the fallback chain could be reordered or deleted unnoticed — and a
        // child whose whole output is stdout is exactly the amd-smi case.
        let error = assemble_output(
            success_status(),
            true,
            (b"stdout-only".to_vec(), None),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");
        assert!(
            format!("{error:#}").contains("stdout-only"),
            "a child that only wrote stdout must still be quoted: {error:#}"
        );

        let error = assemble_output(
            success_status(),
            true,
            (Vec::new(), None),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("no output"),
            "a silent child must say so rather than trailing a bare colon: {rendered:?}"
        );
        assert!(rendered.contains("exceeded 2s timeout"), "{rendered:?}");
    }

    #[test]
    fn a_clean_run_returns_both_streams() {
        let output = assemble_output(
            success_status(),
            false,
            (b"out".to_vec(), None),
            (b"err".to_vec(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect("a clean run must succeed");

        assert!(output.status.success());
        assert_eq!(output.stdout, b"out");
        assert_eq!(output.stderr, b"err");
    }

    #[cfg(unix)]
    #[test]
    fn a_child_that_floods_both_pipes_does_not_deadlock() {
        // The defect this function exists to prevent. Both pipes are bounded by
        // the OS (commonly ~64KiB), so a child writing past that blocks in
        // `write` and never exits — and a version of this function that drains
        // only after the child exits never reads a byte, so the child is killed
        // at the timeout and a successful command is reported as too slow.
        //
        // Flooding *both* streams is deliberate: a single-pipe child still
        // passes against an implementation that drains stdout fully and only
        // then starts on stderr, which is its own deadlock.
        //
        // `sh` rather than a real subject program (amd-smi, an SDK install),
        // because the property is about this function's pipe handling and
        // nothing else, and a shell makes it a unit test rather than one that
        // needs a multi-GPU host.
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "yes deadlock-canary | head -c 2000000; yes deadlock-canary | head -c 2000000 >&2",
        ]);

        // Short, because a regression fails by exhausting this timeout: ten
        // seconds is far past the ~0.1s a working drain takes, and far short of
        // stalling the suite.
        let output = run_with_timeout(command, Duration::from_secs(10), "process")
            .expect("a child that outruns the pipe buffer must not be killed as a timeout");

        assert!(output.status.success(), "{output:?}");
        // Without these the test would keep passing if the child ever stopped
        // producing more than one buffer, which is to say it would stop testing
        // the hazard.
        assert!(
            output.stdout.len() > 64 * 1024,
            "stdout must exceed one pipe buffer: {} bytes",
            output.stdout.len()
        );
        assert!(
            output.stderr.len() > 64 * 1024,
            "stderr must exceed one pipe buffer: {} bytes",
            output.stderr.len()
        );
    }

    /// How large a quoted-output assertion lets the whole error message get.
    ///
    /// Comfortably above `MAX_QUOTED_OUTPUT_CHARS` plus the label and marker,
    /// and two orders of magnitude below the ~200KB the fixtures below feed in,
    /// so the assertion answers "is it bounded at all" rather than pinning the
    /// exact constant — which is a tuning decision, not a contract.
    const QUOTE_ASSERTION_CEILING: usize = 4000;

    /// The flooding fixtures below write this many bytes, past one pipe buffer
    /// and ~100x the quote bound.
    const FLOOD_BYTES: usize = 200_000;

    const FLOOD_CANARY: &str = "flood-canary";

    /// The assertions shared by both platforms' timeout-quote cases.
    ///
    /// Checked together because they trade off against each other: a bound that
    /// kept nothing would satisfy the size check while making the message
    /// useless, and quoting everything keeps it useful while costing the logs
    /// the thing this bound exists to protect.
    fn assert_timeout_quote_is_bounded_and_useful(message: &str, timeout_secs: u64) {
        assert!(
            message.contains(&format!("exceeded {timeout_secs}s timeout")),
            "the timeout must still be named: {message:?}"
        );
        assert!(
            message.contains(FLOOD_CANARY),
            "the child's own output must still be quoted, not dropped: {message:?}"
        );
        assert!(
            message.contains("[truncated]"),
            "a shortened quote must say so, or a reader takes the head for the whole: {message:?}"
        );
        let length = message.chars().count();
        assert!(
            length <= QUOTE_ASSERTION_CEILING,
            "the quote must be bounded: the message is {length} characters, from a child that \
             printed {FLOOD_BYTES} bytes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_timed_out_child_has_its_output_quoted_within_a_bound() {
        for shell in grandchild_shells() {
            // Both halves are load-bearing. The flood is what makes the quote
            // large; the backgrounded `sleep` is what makes this a timeout,
            // which is the only path that quotes the child's output at all.
            //
            // Unbounded, this message travels: `rocmd` renders the error chain
            // into the bridge snapshot's `note`, and the GPU-metrics watcher
            // writes that note into two JSONL logs every 60s, neither ever
            // rotated or capped. A real amd-smi on a multi-GPU host prints
            // well past this fixture's size.
            let mut command = Command::new(shell);
            command.args([
                "-c",
                &format!("yes {FLOOD_CANARY} | head -c {FLOOD_BYTES} >&2; sleep 30 & wait"),
            ]);

            let error = run_with_timeout(command, Duration::from_secs(1), "process")
                .expect_err("a command that outlives its timeout must fail");

            assert_timeout_quote_is_bounded_and_useful(&format!("{error:#}"), 1);
        }
    }

    /// A scratch directory holding a Windows fixture's script, and for
    /// [`Self::flooding`] fixtures the payload beside it.
    ///
    /// Per-process unique so concurrent test binaries cannot collide on it, and
    /// removed on drop so a failing assertion does not leave either behind.
    #[cfg(windows)]
    struct WindowsFixture {
        dir: std::path::PathBuf,
    }

    #[cfg(windows)]
    impl WindowsFixture {
        /// Write `script` as a batch file in a scratch directory of its own.
        ///
        /// A batch file rather than `cmd /C "<command line>"`: cmd's rules for
        /// stripping the quotes Rust adds around an argument containing spaces
        /// and redirections are subtle enough that the fixture would become the
        /// thing under test. A bare path has none of that, and `%~dp0` inside
        /// the script resolves anything beside it without the test passing a
        /// path in.
        fn script(name: &str, script: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("rocm-cli-process-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("failed to create the fixture directory");
            std::fs::write(dir.join("fixture.bat"), script).expect("failed to write the script");

            Self { dir }
        }

        /// A [`Self::script`] fixture with a `FLOOD_BYTES` payload beside it for
        /// the script to print.
        fn flooding(name: &str, script: &str) -> Self {
            let fixture = Self::script(name, script);

            let line = format!("{FLOOD_CANARY}\r\n");
            let mut payload = String::with_capacity(FLOOD_BYTES + line.len());
            while payload.len() < FLOOD_BYTES {
                payload.push_str(&line);
            }
            std::fs::write(fixture.dir.join("payload.txt"), payload)
                .expect("failed to write the payload");

            fixture
        }

        fn command(&self) -> Command {
            let mut command = Command::new("cmd");
            command.args(["/C"]).arg(self.dir.join("fixture.bat"));
            command
        }
    }

    #[cfg(windows)]
    impl Drop for WindowsFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_child_that_floods_both_pipes_does_not_deadlock() {
        // The Windows half of the unix case above, and the same property: both
        // pipes are bounded by the OS here too — more tightly than on Linux —
        // so an implementation that drains only after the child exits, or
        // drains stdout to completion before starting on stderr, never lets
        // this child finish. `type` twice is the shell-free equivalent of the
        // unix `yes | head`; cmd has no `yes`.
        let fixture = WindowsFixture::flooding(
            "flood",
            "@echo off\r\ntype \"%~dp0payload.txt\"\r\ntype \"%~dp0payload.txt\" 1>&2\r\n",
        );

        let output = run_with_timeout(fixture.command(), Duration::from_secs(30), "process")
            .expect("a child that outruns the pipe buffer must not be killed as a timeout");

        assert!(output.status.success(), "{output:?}");
        assert!(
            output.stdout.len() > 64 * 1024,
            "stdout must exceed one pipe buffer: {} bytes",
            output.stdout.len()
        );
        assert!(
            output.stderr.len() > 64 * 1024,
            "stderr must exceed one pipe buffer: {} bytes",
            output.stderr.len()
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_timed_out_child_has_its_output_quoted_within_a_bound() {
        // Covers on Windows what three separate unix cases cover there, because
        // one fixture happens to exhibit all of it: `ping -n 30` runs for ~29
        // seconds, so the call times out; `ping` is a grandchild of the `cmd`
        // this function kills at the deadline, and killing a process on Windows
        // does not kill its children, so it keeps the inherited write ends open
        // and no EOF arrives — an implementation that joins its readers waits
        // the full 29 seconds; and the preceding flood makes the resulting
        // quote large enough for the bound to matter.
        let fixture = WindowsFixture::flooding(
            "timeout",
            "@echo off\r\ntype \"%~dp0payload.txt\" 1>&2\r\nping -n 30 127.0.0.1\r\n",
        );

        let started = Instant::now();
        let error = run_with_timeout(fixture.command(), Duration::from_secs(2), "process")
            .expect_err("a command that outlives its timeout must fail");
        let elapsed = started.elapsed();

        assert_timeout_quote_is_bounded_and_useful(&format!("{error:#}"), 2);
        assert!(
            elapsed < Duration::from_secs(10),
            "the call must return at its deadline rather than waiting for an abandoned \
             grandchild to exit, but it took {elapsed:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_child_that_reads_stdin_is_given_an_empty_one() {
        // `set /p` leaves the variable untouched when its read reaches EOF, so
        // pre-setting it to the marker is cmd's equivalent of the unix
        // `read || echo`. A stdin left open blocks in `set /p` instead.
        //
        // Both callers in `rocmd` run on Windows -- the amd-smi probe in
        // `apps/rocmd/src/host_snapshot.rs` and the `rocm` capture that
        // `apps/rocmd/src/common.rs` makes through `run_rocm_capture_with_env`
        // -- and `run_with_timeout` has no `cfg(windows)` branch, so this covers
        // the same promise with the same two-sidedness as the unix case.
        let fixture = WindowsFixture::script(
            "stdin",
            &format!(
                "@echo off\r\nset \"LINE={STDIN_EOF_MARKER}\"\r\nset /p LINE=\r\necho %LINE%\r\n"
            ),
        );

        assert_the_child_is_given_an_empty_stdin(fixture.command());
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
