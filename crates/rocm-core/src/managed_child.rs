// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether a managed-service child this process spawned is still running.
//!
//! Split out of `process.rs` rather than living beside [`process_is_running`]:
//! that module's checks answer for *any* pid, while this one is only meaningful
//! for a child the caller still holds, and the distinction is the whole point
//! of the type (see its doc comment for the zombie case it exists to handle).

// Only the Windows arm calls it; the Unix arm reaps through the retained
// `Child`. Gated to match, or the Unix `-D warnings` build rejects it unused.
#[cfg(windows)]
use crate::process::process_is_running;

/// Confirms whether a managed-service child is still running, using a
/// platform-appropriate check that can tell a truly running process apart from
/// one whose exit has not yet been reaped (ROCMAI-541).
///
/// On Unix, [`process_is_running`]'s `kill(pid, 0)` succeeds for a zombie — a
/// process that already exited but whose parent has not yet called `wait()` —
/// because the kernel keeps its entry around until reaped. The managed engine
/// is spawned as a *direct* child of the `rocm` process: [`detach_command_session`]
/// only calls `setsid()` in `pre_exec`, there is no second fork to orphan it to
/// init. So while `rocm` is still watching it (the launch readiness wait, or the
/// attached `--verbose` log-streaming loop), a child that dies becomes a zombie
/// that a signal probe alone would misreport as alive — exactly the original
/// ROCMAI-541 symptom, on the platform the first round of this fix claimed to
/// have covered. Retaining the spawned [`std::process::Child`] and calling
/// `try_wait()` instead reaps it (so the zombie entry is released) and reads
/// its *real* wait status rather than merely probing whether the pid still
/// exists.
///
/// On Windows there is no equivalent gap: [`process_is_running`] there opens a
/// fresh handle with `OpenProcess` and reads the actual exit code via
/// `GetExitCodeProcess`, which needs no parent-side reaping — so the bare pid
/// is enough, and this variant just defers to it.
pub enum ManagedChildLiveness {
    #[cfg(not(windows))]
    Unix(std::process::Child),
    #[cfg(windows)]
    Windows(u32),
}

impl ManagedChildLiveness {
    /// Build from the just-spawned child on Unix, where a real `Child` handle
    /// is available to reap later.
    #[cfg(not(windows))]
    pub const fn from_child(child: std::process::Child) -> Self {
        Self::Unix(child)
    }

    /// Build from a bare pid on Windows, where [`process_is_running`] already
    /// reads the real exit code without needing a retained handle.
    #[cfg(windows)]
    pub const fn from_pid(pid: u32) -> Self {
        Self::Windows(pid)
    }

    /// True while the child is confirmed still running. Reaps it on Unix if it
    /// has just exited — see the type's doc comment for why that reap (rather
    /// than a signal probe) is the point of this type. An error from
    /// `try_wait` (in practice only reachable if the pid had already been
    /// reaped by something else) is treated as "not running": a failure report
    /// must never be blocked on being unable to positively confirm the child
    /// is gone.
    pub fn is_running(&mut self) -> bool {
        match self {
            #[cfg(not(windows))]
            Self::Unix(child) => matches!(child.try_wait(), Ok(None)),
            #[cfg(windows)]
            Self::Windows(pid) => process_is_running(*pid),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ROCMAI-541: pins the core contract `ManagedChildLiveness` exists
    /// for — a child that has exited must read as not-running, and a second
    /// call after it has already been reaped must keep reading that way
    /// rather than panicking or erroring (this type's whole reason to exist
    /// is that a bare `process_is_running` pid probe gets this wrong for a
    /// Unix zombie; see the type's doc comment). The Windows arm
    /// (`ManagedChildLiveness::Windows`, backed by `process_is_running` over
    /// `OpenProcess`/`GetExitCodeProcess`) is not exercised here: it cannot be
    /// built on a non-Windows test target, and this repo's test environment
    /// has no Windows runner to exercise it on.
    #[cfg(not(windows))]
    #[test]
    fn managed_child_liveness_unix_reports_dead_after_exit_and_does_not_double_reap() {
        use std::process::Command;
        use std::time::{Duration, Instant};

        let child = Command::new("true")
            .spawn()
            .expect("spawn a trivial, immediately-exiting child");
        let mut liveness = ManagedChildLiveness::from_child(child);

        // Poll rather than assume a fixed sleep always outraces the scheduler:
        // `true` exits almost immediately, but the test only needs it to be
        // gone by the time it checks, not instantly.
        let deadline = Instant::now() + Duration::from_secs(5);
        while liveness.is_running() {
            assert!(
                Instant::now() < deadline,
                "child did not report dead within the timeout"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        // The second call is the point of this test: reaping must not leave
        // the type unable to answer again, or answering "running" once it no
        // longer can reap a real exit status.
        assert!(
            !liveness.is_running(),
            "a second call after reaping must still report dead, not panic or flip to `true`"
        );
    }

    /// The Windows counterpart, covering the arm that is the actual
    /// crash-detection primitive on that platform (ROCMAI-541 review).
    ///
    /// The Unix case above cannot reach it — `ManagedChildLiveness::Windows`
    /// cannot even be constructed off Windows — and the scripted-engine e2e
    /// scenarios run on the Linux no-GPU lane, where the Unix arm does the
    /// work. So without this, the half of the liveness check that decides
    /// crash-vs-starting on the reported platform had nothing exercising it,
    /// while its sibling (`spawn_detached_with_log`) did.
    ///
    /// No reaping is involved here: the Windows arm reads the real exit code
    /// through `OpenProcess`/`GetExitCodeProcess`, so the contract is simply
    /// "alive while running, dead once terminated, and still dead when asked
    /// again".
    #[cfg(windows)]
    #[test]
    fn managed_child_liveness_windows_tracks_a_real_child_through_termination() {
        use std::time::{Duration, Instant};

        // A child that outlives the assertions below without finishing on its
        // own, so "still running" is a real observation rather than a race.
        let comspec = std::env::var_os("COMSPEC")
            .unwrap_or_else(|| OsString::from(r"C:\Windows\System32\cmd.exe"));
        let mut child = Command::new(&comspec)
            .args(["/c", "ping -n 30 127.0.0.1 >nul"])
            .spawn()
            .expect("spawn a long-running child");
        let pid = child.id();
        let mut liveness = ManagedChildLiveness::from_pid(pid);

        assert!(
            liveness.is_running(),
            "a child that has not been terminated must read as running"
        );

        terminate_process(pid).expect("terminate the fixture child");

        let deadline = Instant::now() + Duration::from_secs(10);
        while liveness.is_running() {
            assert!(
                Instant::now() < deadline,
                "a terminated child must stop reading as running"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            !liveness.is_running(),
            "a second call must keep reporting dead"
        );

        let _ = child.wait();
    }
}
