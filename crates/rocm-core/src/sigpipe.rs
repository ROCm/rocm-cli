// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Writing to a pipe whose reader has gone away, without being killed for it.
//!
//! Rust starts every process with SIGPIPE ignored, so a write to a closed pipe
//! returns `EPIPE`. `rocm` restores the default action at startup so it ends
//! quietly when its own output is piped into `head`, and `rocm daemon`
//! hosts the daemon in that same process. Under the default action a write to
//! a child that has already exited kills the writer before it can report why
//! the child exited, so code that writes to a child's stdin wraps the write in
//! [`with_sigpipe_ignored`].

/// Run `f` with SIGPIPE ignored, restoring the previous disposition afterwards.
///
/// A write that reaches a closed pipe inside `f` then returns an `EPIPE` error
/// instead of delivering a signal. Where the disposition was already ignored
/// (Rust's startup default) this changes nothing. The disposition is
/// process-wide, so keep `f` to the write itself. Windows has no SIGPIPE and
/// runs `f` as is.
#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI
pub fn with_sigpipe_ignored<T>(f: impl FnOnce() -> T) -> T {
    // SAFETY: reading/installing a signal disposition is always safe.
    let previous = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    let result = f();
    // SAFETY: restoring the disposition captured above is always safe.
    unsafe {
        libc::signal(libc::SIGPIPE, previous);
    }
    result
}

#[cfg(not(unix))]
pub fn with_sigpipe_ignored<T>(f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(all(test, unix))]
mod tests {
    use super::with_sigpipe_ignored;
    use std::io::Write as _;
    use std::os::unix::process::ExitStatusExt as _;
    use std::process::{Command, ExitStatus, Stdio};

    /// A write to a dead child's stdin under the default SIGPIPE action is
    /// fatal, and the same write inside [`with_sigpipe_ignored`] returns an
    /// error instead.
    ///
    /// The bug only reproduces under `SIG_DFL`; the test harness leaves SIGPIPE
    /// at Rust's default `SIG_IGN`, and flipping it process-wide would race
    /// sibling tests. So this re-execs itself in fresh processes that install
    /// `SIG_DFL` first, and checks BOTH directions:
    /// - unguarded write to a dead child's stdin is fatal (killed by SIGPIPE) —
    ///   this is the regression the plain `SIG_DFL` reset introduced;
    /// - the same write wrapped in [`with_sigpipe_ignored`] survives and returns
    ///   an error instead — this is the fix.
    ///
    /// Each child reaps its short-lived grandchild before writing, so the read
    /// end is closed deterministically and the outcome does not race.
    #[test]
    #[allow(unsafe_code)] // libc::signal, to restore the default action in the child
    fn engine_stdin_write_under_sig_dfl_is_guarded_not_fatal() {
        // Grandchild helper: install the default action, spawn a process that
        // exits immediately, reap it, then write to its now-closed stdin. The
        // "guarded" arm wraps the write; the "unguarded" arm does not.
        if let Some(mode) = std::env::var_os("ROCM_TEST_SIGPIPE_ARM") {
            // SAFETY: installing a signal disposition is always safe.
            unsafe {
                libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            }
            let mut grandchild = Command::new("true")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn a process that exits immediately");
            let mut stdin = grandchild.stdin.take().expect("grandchild stdin");
            grandchild.wait().expect("reap grandchild");
            // Heap-allocated so the buffer doesn't trip clippy::large-stack-arrays;
            // it must exceed the pipe buffer so the write actually reaches the
            // closed read end rather than being swallowed by kernel buffering.
            let payload = vec![b'x'; 64 * 1024];
            if mode == "guarded" {
                let result = with_sigpipe_ignored(|| stdin.write_all(&payload));
                assert!(
                    result.is_err(),
                    "guarded write to a dead pipe should return BrokenPipe, not succeed"
                );
            } else {
                // Unguarded: under SIG_DFL this write delivers SIGPIPE and never
                // returns. If the process somehow survives, exit 0 so the parent's
                // "should have been signalled" assertion fails loudly.
                let _ = stdin.write_all(&payload);
            }
            return;
        }

        let exe = std::env::current_exe().expect("current test executable");
        let run_arm = |mode: &str| -> ExitStatus {
            Command::new(&exe)
                .args([
                    "sigpipe::tests::engine_stdin_write_under_sig_dfl_is_guarded_not_fatal",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_SIGPIPE_ARM", mode)
                .status()
                .expect("re-exec the test arm in a child process")
        };

        let unguarded = run_arm("unguarded");
        assert_eq!(
            unguarded.signal(),
            Some(libc::SIGPIPE),
            "sanity check: an unguarded stdin write under SIG_DFL must be killed by \
             SIGPIPE (got {unguarded:?}); if not, the test no longer proves the guard matters"
        );

        let guarded = run_arm("guarded");
        assert!(
            guarded.success(),
            "guarded child-stdin write was fatal under SIG_DFL ({guarded:?}) — \
             SIGPIPE reached the process instead of surfacing as an error"
        );
    }
}
