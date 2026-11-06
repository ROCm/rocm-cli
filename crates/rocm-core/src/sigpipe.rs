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
///
/// Guarded regions may overlap, on one thread or on several (the stderr
/// forwarder of one engine request runs while another request writes to its
/// child). The disposition in force before the first region is remembered once
/// and put back when the last region ends, so an overlap can neither restore
/// "ignored" over a default action nor leave "ignored" behind.
#[cfg(unix)]
pub fn with_sigpipe_ignored<T>(f: impl FnOnce() -> T) -> T {
    let _guard = unix::IgnoreGuard::enter();
    f()
}

#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI
mod unix {
    use std::sync::{Mutex, PoisonError};

    /// How many guarded regions are open, and the disposition that was in force
    /// before the first of them.
    static OPEN: Mutex<(usize, libc::sighandler_t)> = Mutex::new((0, libc::SIG_DFL));

    /// Holds SIGPIPE ignored from [`IgnoreGuard::enter`] until it is dropped, so
    /// a panic inside the guarded region still restores the disposition.
    pub(super) struct IgnoreGuard;

    impl IgnoreGuard {
        pub(super) fn enter() -> Self {
            let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
            if open.0 == 0 {
                // SAFETY: reading/installing a signal disposition is always safe.
                open.1 = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
            }
            open.0 += 1;
            Self
        }
    }

    impl Drop for IgnoreGuard {
        fn drop(&mut self) {
            let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
            open.0 -= 1;
            if open.0 == 0 {
                // SAFETY: restoring the disposition captured by the first guard.
                unsafe {
                    libc::signal(libc::SIGPIPE, open.1);
                }
            }
        }
    }
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

    /// The disposition SIGPIPE has now (read by installing the same one back).
    #[allow(unsafe_code)] // libc::signal
    fn current_disposition() -> libc::sighandler_t {
        // SAFETY: reading/installing a signal disposition is always safe.
        let current = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
        // SAFETY: putting back what was just read.
        unsafe {
            libc::signal(libc::SIGPIPE, current);
        }
        current
    }

    /// Guarded regions that overlap, nested on one thread or interleaved across
    /// two, leave SIGPIPE ignored for as long as any of them is open and put the
    /// original default action back only when the last one closes. With a guard
    /// that saved and restored per call, the region that ended first would
    /// restore "ignored" over the second's `SIG_DFL` capture, or restore the
    /// default under the region still open. Runs in a re-executed child, since
    /// the disposition is process-wide and the harness leaves it ignored.
    #[test]
    #[allow(unsafe_code)] // libc::signal, to install the default action in the child
    fn overlapping_guards_restore_the_original_disposition_once() {
        if std::env::var_os("ROCM_TEST_SIGPIPE_OVERLAP_ARM").is_some() {
            use std::sync::Barrier;
            // SAFETY: installing a signal disposition is always safe.
            unsafe {
                libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            }

            // Nested on one thread.
            with_sigpipe_ignored(|| {
                with_sigpipe_ignored(|| {});
                assert_eq!(
                    current_disposition(),
                    libc::SIG_IGN,
                    "the inner region ended while the outer one was open"
                );
            });
            assert_eq!(current_disposition(), libc::SIG_DFL, "nested");

            // Two threads whose regions overlap, the first to enter leaving first.
            let entered = Barrier::new(2);
            let first_left = Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    with_sigpipe_ignored(|| {
                        entered.wait();
                    });
                    first_left.wait();
                });
                scope.spawn(|| {
                    with_sigpipe_ignored(|| {
                        entered.wait();
                        first_left.wait();
                        assert_eq!(
                            current_disposition(),
                            libc::SIG_IGN,
                            "the first region ended while this one was open"
                        );
                    });
                });
            });
            assert_eq!(current_disposition(), libc::SIG_DFL, "interleaved");
            return;
        }

        let status = Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "sigpipe::tests::overlapping_guards_restore_the_original_disposition_once",
                "--exact",
                "--nocapture",
            ])
            .env("ROCM_TEST_SIGPIPE_OVERLAP_ARM", "1")
            .status()
            .expect("re-exec the test arm in a child process");
        assert!(status.success(), "{status:?}");
    }

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
