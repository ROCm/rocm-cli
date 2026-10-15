// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Process-spawn and -terminate primitives.
//!
//! Hidden-console/detached spawn and exit-wait on Windows, process/process-tree
//! termination on both platforms, and the exclusive `FileLock` used to
//! serialize the daemon autostart path, the managed-launch GPU claim, and
//! `uv` installs across processes.

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

use crate::net::DEFAULT_LOCAL_HOST;
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT,
    CreateProcessW, DETACHED_PROCESS, GetExitCodeProcess, INFINITE, OpenProcess,
    PROCESS_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    STARTF_USESHOWWINDOW, STARTF_USESTDHANDLES, STARTUPINFOW, TerminateProcess,
    WaitForSingleObject,
};

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
/// serialize check-then-act sequences over shared on-disk state — the daemon
/// autostart decision and the managed-serve GPU select-then-claim — so two
/// concurrent invocations cannot both pass the same TOCTOU check.
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
        let path = path.into();
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
        file.lock()
            .with_context(|| format!("failed to acquire lock {}", path.display()))?;
        Ok(Self { file, path })
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
    /// not hang forever behind a `rocm serve` launch holding the same lock.
    /// A `rocm serve` launch itself uses the unbounded [`Self::acquire`]: its
    /// own worst-case hold time (the already-running probe inside the launch
    /// transaction) can exceed [`MANAGED_LAUNCH_LOCK_TIMEOUT`], so bounding it
    /// would trade a real, slow-but-working launch for a spurious failure.
    ///
    /// # Errors
    /// Fails when the lock file cannot be created/locked, or when another
    /// holder keeps it for the whole timeout.
    pub fn acquire_bounded(path: impl Into<PathBuf>, timeout: Duration) -> Result<Self> {
        let path = path.into();
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
/// `rocmd`'s recovery/restart path is bounded this way; a `rocm serve` launch holds the
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
