// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether a managed-service child this process spawned is still running.
//!
//! Split out of `process.rs` rather than living beside [`process_is_running`]:
//! that module's checks answer for *any* pid, while this one is only meaningful
//! for a child the caller still holds, and the distinction is the whole point
//! of the type (see its doc comment for the zombie case it exists to handle).

use crate::process::ProcessProbe;
// Only the Windows arm calls it; the Unix arm reaps through the retained
// `Child`. Gated to match, or the Unix `-D warnings` build rejects it unused.
#[cfg(windows)]
use crate::process::probe_process;

/// The pid-probe liveness policy the Windows arm runs: a probe that *cannot*
/// answer means the child **may still be running**, never that it is gone.
///
/// Extracted from [`ManagedChildLiveness::is_running`] and compiled on every
/// platform — only the call site is `#[cfg(windows)]` — so the Linux lane can
/// pin the discrimination the whole type exists for. The
/// `ManagedChildLiveness::Windows` variant cannot even be constructed off
/// Windows, so without this seam the policy was exercised by nothing but a real
/// Windows CI run against a real child, which is never
/// [`ProcessProbe::Unqueryable`] in the first place.
///
/// Callers turn a `false` here into a crash verdict that stamps the service
/// record and the engine state file `failed`, so collapsing `Unqueryable` into
/// "not running" — exactly what [`crate::process_is_running`] does, and what its
/// own doc comment warns against — reports a healthy engine as crashed on a
/// transient `ERROR_ACCESS_DENIED` (ROCMAI-541's own symptom).
///
/// The probe is a parameter purely so the tests below can drive all three
/// answers on any platform; production always passes [`probe_process`].
#[cfg_attr(not(windows), allow(dead_code))]
fn pid_child_is_running(pid: u32, probe: impl FnOnce(u32) -> ProcessProbe) -> bool {
    probe(pid).may_be_running()
}

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
/// On Windows there is no equivalent gap to close: a fresh `OpenProcess` plus
/// `GetExitCodeProcess` reads the real exit code with no parent-side reaping,
/// so the bare pid is enough and no handle is retained. The read itself goes
/// through [`probe_process`], not [`process_is_running`] — see [`Self::is_running`]
/// for why collapsing `Unqueryable` into "not running" is the wrong answer for
/// a decision that stamps a record `failed`.
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

    /// Build from a bare pid on Windows, where the exit code can be read
    /// without a retained handle. [`Self::is_running`] does that read through
    /// [`probe_process`], so a process this one cannot open is not mistaken
    /// for one that has exited.
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
    ///
    /// The Windows arm asks [`probe_process`] rather than `process_is_running`,
    /// whose own doc comment says a decision that must not take a process for
    /// gone has to keep [`ProcessProbe::Unqueryable`] apart. This is such a
    /// decision: callers turn a `false` here into a crash verdict that stamps
    /// the record `failed`. `process_is_running` collapses `Unqueryable` into
    /// "not running", so a transient `OpenProcess` failure — an
    /// `ERROR_ACCESS_DENIED` on a live process this one may not open — would
    /// report a healthy engine as crashed. Unqueryable therefore reads as "may
    /// still be running", which errs toward waiting rather than toward a false
    /// crash report.
    ///
    /// [`ProcessProbe::Unqueryable`]: crate::ProcessProbe::Unqueryable
    pub fn is_running(&mut self) -> bool {
        match self {
            #[cfg(not(windows))]
            Self::Unix(child) => matches!(child.try_wait(), Ok(None)),
            #[cfg(windows)]
            Self::Windows(pid) => pid_child_is_running(*pid, probe_process),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ROCMAI-541 review: the pid-backed liveness policy must not read "I could
    /// not ask" as "it is gone".
    ///
    /// This is the one assertion that discriminates [`pid_child_is_running`]
    /// from the [`crate::process_is_running`] probe the launch path used to
    /// pass: the two agree on `Running` and on `Gone`, and differ **only** on
    /// `Unqueryable`, which `process_is_running` collapses into `false`. That
    /// `false` is not advisory — it reaches `settle_startup_death`, which stamps
    /// the manifest and the engine state file `failed` and aborts the launch, so
    /// collapsing it turns a transient `OpenProcess` denial on a healthy new
    /// engine into a reported crash.
    ///
    /// Runs on every platform deliberately. The real `Self::Windows` arm cannot
    /// be built off Windows and, even on Windows, the existing child-fixture
    /// test below can only produce `Running`/`Gone` from a real process — so the
    /// middle answer, the only one the two candidate probes disagree about, had
    /// nothing exercising it anywhere.
    #[test]
    fn an_unqueryable_pid_child_is_not_reported_gone() {
        for (probe, expected) in [
            (ProcessProbe::Running, true),
            (ProcessProbe::Unqueryable, true),
            (ProcessProbe::Gone, false),
        ] {
            assert_eq!(
                pid_child_is_running(4242, |pid| {
                    assert_eq!(pid, 4242, "the probe must be asked about the child's pid");
                    probe
                }),
                expected,
                "{probe:?} must read as running={expected}"
            );
        }
    }

    /// ROCMAI-541: pins the core contract `ManagedChildLiveness` exists
    /// for — a child that has exited must read as not-running, and a second
    /// call after it has already been reaped must keep reading that way
    /// rather than panicking or erroring (this type's whole reason to exist
    /// is that a bare `process_is_running` pid probe gets this wrong for a
    /// Unix zombie; see the type's doc comment). The Windows arm
    /// (`ManagedChildLiveness::Windows`, backed by `probe_process` over
    /// `OpenProcess`/`GetExitCodeProcess`) is not exercised *by this test*
    /// only because the variant cannot be built on a non-Windows target. It is
    /// not unexercised: `ci.yml`'s `windows-debug` job runs
    /// `cargo test --workspace --all-targets` on `windows-latest` for any `.rs`
    /// change, so the `#[cfg(windows)]` test below it is this repo's coverage
    /// of that arm.
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
        // In scope implicitly while this lived in `process.rs`; named here
        // because the move took it out of that module.
        use crate::process::terminate_process;
        use std::ffi::OsString;
        use std::process::Command;
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
