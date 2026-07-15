// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Verified, bounded process termination.
//!
//! Stopping a managed service by a *persisted* PID is unsafe on its own: PIDs
//! are recycled, so a stale PID may have been reassigned to an unrelated process
//! by the time a stop is requested. This module pairs each PID with the kernel's
//! start-time for that PID — an identity that survives recycling — and refuses to
//! signal a PID whose identity no longer matches. It also reports termination
//! truthfully: a stop is only "graceful" once the recorded process is observed to
//! have actually exited within a bounded grace period, escalating to `SIGKILL`
//! only when the caller opts into a forced stop.

use std::time::{Duration, Instant};

/// How often the bounded waits poll for the target's exit.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Breadth of a termination request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillScope {
    /// Signal only the recorded PID.
    Single,
    /// Signal the recorded PID plus every transitive child. Engines such as
    /// vLLM spawn workers that hold the GPU allocation, so the whole tree must
    /// be signalled to avoid leaking device memory.
    Tree,
}

/// A spawned process identified in a way that is robust to PID recycling.
///
/// `start_ticks` is the kernel start-time of the process (clock ticks since
/// boot). It is `None` on platforms without `/proc`, where identity cannot be
/// verified and callers fall back to best-effort behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_ticks: Option<u64>,
}

impl ProcessIdentity {
    /// Capture the identity of `pid` as it exists right now (e.g. just after
    /// spawning it, while it is guaranteed to be alive).
    #[must_use]
    pub fn capture(pid: u32) -> Self {
        Self {
            pid,
            start_ticks: process_start_ticks(pid),
        }
    }

    /// Reconstruct an identity from persisted values.
    #[must_use]
    pub const fn new(pid: u32, start_ticks: Option<u64>) -> Self {
        Self { pid, start_ticks }
    }
}

/// Whether a PID still refers to the recorded process instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityState {
    /// The PID is live and, where verifiable, its start-time matches. Also the
    /// best-effort verdict when no identity was recorded (legacy state files).
    Matches,
    /// The PID is live but is provably a *different* process: both start-times
    /// were readable and differ, so the recorded process has exited and its PID
    /// was recycled.
    Recycled,
    /// The PID is live and an identity was recorded, but the current start-time
    /// cannot be read, so it can be neither confirmed nor refuted. The process
    /// must not be signalled, and it must not be reported as stopped.
    Indeterminate,
    /// The PID is not live (or is a not-yet-reaped zombie).
    Gone,
}

/// Classify the current state of `id`'s PID relative to its recorded identity.
///
/// Deliberately conservative about killing the wrong process: a recorded
/// identity that cannot be confirmed (start-time unreadable) is
/// [`IdentityState::Indeterminate`], never a risky match. When no identity was
/// recorded (legacy state files), it degrades to best-effort
/// [`IdentityState::Matches`].
#[must_use]
pub fn identity_state(id: &ProcessIdentity) -> IdentityState {
    if !crate::process_is_running(id.pid) || process_has_exited(id.pid) {
        return IdentityState::Gone;
    }
    match (id.start_ticks, process_start_ticks(id.pid)) {
        (Some(expected), Some(actual)) => {
            if expected == actual {
                IdentityState::Matches
            } else {
                IdentityState::Recycled
            }
        }
        // Identity recorded but unconfirmable right now: neither signal nor
        // claim a stop.
        (Some(_), None) => IdentityState::Indeterminate,
        // No recorded identity (legacy state): best-effort proceed.
        (None, _) => IdentityState::Matches,
    }
}

/// Whether `state` means the recorded process is definitively no longer running.
///
/// [`IdentityState::Indeterminate`] is intentionally excluded: a process we can
/// neither confirm nor refute is treated as possibly-alive, so a bounded wait
/// keeps waiting rather than declaring a premature exit.
const fn is_exited(state: IdentityState) -> bool {
    matches!(state, IdentityState::Gone | IdentityState::Recycled)
}

/// The truthful result of a verified termination attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationOutcome {
    /// The recorded process was already gone; nothing was signalled.
    AlreadyGone,
    /// The PID now belongs to a different process (recycled), so the recorded
    /// process has already exited; nothing was signalled.
    IdentityMismatch,
    /// The recorded PID is live but its identity could not be confirmed, so it
    /// was deliberately left untouched and its state is unknown.
    Unverified,
    /// Every signalled process exited after `SIGTERM`, within the grace period.
    Graceful,
    /// Termination required escalating to `SIGKILL`.
    Forced,
    /// At least one signalled process was still alive after the bounded deadline.
    TimedOut,
}

impl TerminationOutcome {
    /// Is the recorded service confirmed to be no longer running as a result?
    ///
    /// `false` for [`TimedOut`](Self::TimedOut) (still alive) and
    /// [`Unverified`](Self::Unverified) (could not be confirmed either way).
    #[must_use]
    pub const fn stopped(self) -> bool {
        !matches!(self, Self::TimedOut | Self::Unverified)
    }

    /// Did the process stop without needing a forced kill (or was it already
    /// gone / recycled)? Only `true` when no `SIGKILL` was required.
    #[must_use]
    pub const fn graceful(self) -> bool {
        matches!(
            self,
            Self::Graceful | Self::AlreadyGone | Self::IdentityMismatch
        )
    }

    /// A stable, log-friendly label for the outcome.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AlreadyGone => "already_gone",
            Self::IdentityMismatch => "identity_mismatch",
            Self::Unverified => "unverified",
            Self::Graceful => "graceful",
            Self::Forced => "forced",
            Self::TimedOut => "timed_out",
        }
    }
}

#[derive(Clone, Copy)]
enum Signal {
    /// Request a graceful shutdown (`SIGTERM` on Unix).
    Term,
    /// Force termination (`SIGKILL` on Unix, `TerminateProcess` on Windows).
    Kill,
}

/// Terminate the process recorded in `id`, verifying identity first and
/// reporting the outcome truthfully.
///
/// The process is only signalled when its PID still matches `id`. Under
/// [`KillScope::Tree`] the descendant set is snapshotted (each child bound to
/// its own identity) so termination is confirmed for the *whole* tree — not just
/// the root — which matters when the root is a thin launcher and a child holds
/// the real resource (e.g. a GPU worker). A graceful `SIGTERM` is sent first and
/// every signalled process is polled for actual exit up to `grace`. When any do
/// not exit in time, a forced stop (`force == true`) escalates to `SIGKILL` and
/// waits again; a non-forced stop reports [`TerminationOutcome::TimedOut`]
/// rather than pretending success.
#[must_use]
pub fn terminate_verified(
    id: &ProcessIdentity,
    scope: KillScope,
    grace: Duration,
    force: bool,
) -> TerminationOutcome {
    match identity_state(id) {
        IdentityState::Gone => return TerminationOutcome::AlreadyGone,
        IdentityState::Recycled => return TerminationOutcome::IdentityMismatch,
        IdentityState::Indeterminate => return TerminationOutcome::Unverified,
        IdentityState::Matches => {}
    }

    let tree = matches!(scope, KillScope::Tree);

    // Snapshot the exact processes to account for while the root is still alive.
    // For a tree this binds each descendant PID to its own start-time, so a PID
    // recycled during the wait is never mistaken for a survivor.
    let members: Vec<ProcessIdentity> = if tree {
        crate::process_tree_pids(id.pid)
            .into_iter()
            .map(ProcessIdentity::capture)
            .collect()
    } else {
        vec![*id]
    };

    // Graceful request first: signal the whole scope, then wait for all to exit.
    send_signal(id.pid, Signal::Term, tree);
    if wait_for_all_exit(&members, grace) {
        return TerminationOutcome::Graceful;
    }

    if !force {
        return TerminationOutcome::TimedOut;
    }

    // Bounded escalation to a forced kill.
    //
    // While the root is still ours, SIGKILL the live tree from it (re-enumerated,
    // root verified) — the safest way to reach current children. Then catch any
    // reparented survivor, but only one we can still *positively* re-verify: a
    // member with no recorded start-time cannot be distinguished from a process
    // that recycled its PID during the wait, so its PID is never targeted
    // directly (it is left to time out rather than risk signalling a stranger).
    if matches!(identity_state(id), IdentityState::Matches) {
        send_signal(id.pid, Signal::Kill, tree);
    }
    for member in &members {
        if member.pid == id.pid || member.start_ticks.is_none() {
            continue;
        }
        if matches!(identity_state(member), IdentityState::Matches) {
            send_signal(member.pid, Signal::Kill, false);
        }
    }
    if wait_for_all_exit(&members, grace) {
        TerminationOutcome::Forced
    } else {
        TerminationOutcome::TimedOut
    }
}

/// Poll until every process in `members` has exited, or `grace` elapses.
///
/// A member is exited once its PID is gone or now belongs to a different process
/// ([`is_exited`]); a member that is merely [`IdentityState::Indeterminate`]
/// keeps the wait going rather than being counted as exited.
fn wait_for_all_exit(members: &[ProcessIdentity], grace: Duration) -> bool {
    let deadline = Instant::now() + grace;
    loop {
        if members
            .iter()
            .all(|member| is_exited(identity_state(member)))
        {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        std::thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(now)));
    }
}

#[cfg(not(windows))]
fn send_signal(pid: u32, signal: Signal, tree: bool) -> bool {
    let raw = match signal {
        Signal::Term => libc::SIGTERM,
        Signal::Kill => libc::SIGKILL,
    };
    crate::signal_process_scope(pid, raw, tree)
}

#[cfg(windows)]
fn send_signal(pid: u32, signal: Signal, _tree: bool) -> bool {
    // Windows has no graceful process signal. A `Term` request is therefore a
    // no-op — the bounded wait will time out and, only under a forced stop, the
    // `Kill` step force-terminates. This keeps the reported outcome truthful:
    // non-forced Windows stops surface `TimedOut` rather than a false graceful.
    match signal {
        Signal::Term => true,
        Signal::Kill => crate::terminate_process(pid).is_ok(),
    }
}

/// Read the kernel start-time (field 22 of `/proc/<pid>/stat`) for `pid`.
///
/// Returns `None` when the value cannot be read, including on non-Linux
/// platforms, where identity verification degrades to best-effort.
#[cfg(target_os = "linux")]
#[must_use]
pub fn process_start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    parse_start_ticks(&stat)
}

#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn process_start_ticks(_pid: u32) -> Option<u64> {
    None
}

/// Whether `pid` has already exited but not yet been reaped (a zombie).
///
/// A zombie still has a `/proc` entry and answers `kill(pid, 0)`, yet it is a
/// terminated process — treating it as still running would make a completed stop
/// look like a timeout. Detached engine processes are reparented to init and
/// reaped promptly, so this mainly guards the reaped-by-us and slow-init cases.
#[cfg(target_os = "linux")]
fn process_has_exited(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => parse_state_char(&stat) == Some('Z'),
        // No stat file: the process is gone; `process_is_running` handles the
        // authoritative check, so report "not a zombie" here.
        Err(_) => false,
    }
}

#[cfg(not(target_os = "linux"))]
fn process_has_exited(_pid: u32) -> bool {
    false
}

/// Parse the process state character (field 3) from `/proc/<pid>/stat`.
#[cfg(target_os = "linux")]
fn parse_state_char(stat: &str) -> Option<char> {
    let after_comm = stat.get(stat.rfind(')')? + 1..)?;
    after_comm.split_whitespace().next()?.chars().next()
}

/// Parse the `starttime` field (field 22) from the contents of
/// `/proc/<pid>/stat`.
///
/// The `comm` field (field 2) can contain spaces and parentheses, so parsing
/// begins after the final `)`. From there, field 3 (`state`) is index 0, making
/// `starttime` index 19.
#[cfg(any(target_os = "linux", test))]
fn parse_start_ticks(stat: &str) -> Option<u64> {
    let after_comm = stat.get(stat.rfind(')')? + 1..)?;
    after_comm.split_whitespace().nth(19)?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};

    /// Spawn a child that prints a line to stdout once it is ready, and block
    /// until that line arrives. This replaces sleep-based readiness guesses with
    /// a deterministic signal (e.g. that a shell has installed its SIGTERM trap),
    /// and returns any text on that first line (used to pass out a child PID).
    #[cfg(unix)]
    fn spawn_ready(script: &str) -> (Child, String) {
        use std::io::{BufRead, BufReader};
        let mut child = Command::new("sh")
            .args(["-c", script])
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn ready child");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("read readiness line");
        (child, line.trim().to_owned())
    }

    #[test]
    fn parse_start_ticks_reads_field_22() {
        // Fields 3..=22 after "(comm)"; field 22 (starttime) is the value 9988.
        let stat = "1234 (server) S 1 1234 1234 0 -1 4194304 100 0 0 0 5 6 0 0 20 0 1 0 9988 \
                    123456789 42 18446744073709551615 1 1 0 0 0 0 0 0 0";
        assert_eq!(parse_start_ticks(stat), Some(9988));
    }

    #[test]
    fn parse_start_ticks_handles_comm_with_spaces_and_parens() {
        // A comm containing spaces and a ')' must not fool the parser.
        let stat = "42 (weird ) name) S 1 42 42 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 7777 0 0";
        assert_eq!(parse_start_ticks(stat), Some(7777));
    }

    #[test]
    fn parse_start_ticks_rejects_malformed() {
        assert_eq!(parse_start_ticks("no parens here"), None);
        assert_eq!(parse_start_ticks("123 (short) S 1 2 3"), None);
    }

    #[test]
    fn outcome_truth_table() {
        // stopped(): everything except a still-running timeout.
        assert!(TerminationOutcome::Graceful.stopped());
        assert!(TerminationOutcome::Forced.stopped());
        assert!(TerminationOutcome::AlreadyGone.stopped());
        assert!(TerminationOutcome::IdentityMismatch.stopped());
        assert!(!TerminationOutcome::TimedOut.stopped());

        // Unverified: neither stopped nor graceful — we could not act.
        assert!(!TerminationOutcome::Unverified.stopped());
        assert!(!TerminationOutcome::Unverified.graceful());

        // graceful(): true only when no SIGKILL was needed.
        assert!(TerminationOutcome::Graceful.graceful());
        assert!(TerminationOutcome::AlreadyGone.graceful());
        assert!(TerminationOutcome::IdentityMismatch.graceful());
        assert!(!TerminationOutcome::Forced.graceful());
        assert!(!TerminationOutcome::TimedOut.graceful());
    }

    #[cfg(unix)]
    fn spawn(args: &[&str]) -> Child {
        Command::new(args[0])
            .args(&args[1..])
            .spawn()
            .expect("spawn test child")
    }

    #[cfg(unix)]
    fn reap(mut child: Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn identity_matches_our_own_process() {
        let pid = std::process::id();
        let id = ProcessIdentity::capture(pid);
        assert!(id.start_ticks.is_some(), "should read our own start-time");
        assert_eq!(identity_state(&id), IdentityState::Matches);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn identity_mismatch_when_start_ticks_differ() {
        // Same live PID (ours), but a different recorded start-time: a recycled
        // PID looks exactly like this, and must be treated as a different process.
        let pid = std::process::id();
        let real = process_start_ticks(pid).expect("own start-time");
        let stale = ProcessIdentity::new(pid, Some(real.wrapping_add(1)));
        assert_eq!(identity_state(&stale), IdentityState::Recycled);
    }

    #[cfg(unix)]
    #[test]
    fn identity_gone_after_exit() {
        let child = spawn(&["sh", "-c", "exit 0"]);
        let id = ProcessIdentity::capture(child.id());
        reap(child);
        assert_eq!(identity_state(&id), IdentityState::Gone);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn refuses_to_signal_on_identity_mismatch() {
        // Our own PID with a wrong start-time. A forced stop must NOT kill us:
        // if it signalled, this test process would die instead of asserting.
        let pid = std::process::id();
        let real = process_start_ticks(pid).expect("own start-time");
        let stale = ProcessIdentity::new(pid, Some(real.wrapping_add(1)));
        let outcome =
            terminate_verified(&stale, KillScope::Single, Duration::from_millis(50), true);
        assert_eq!(outcome, TerminationOutcome::IdentityMismatch);
        // Still alive to make the assertion at all — nothing was signalled.
        assert!(crate::process_is_running(pid));
    }

    #[cfg(unix)]
    #[test]
    fn already_gone_when_process_exited() {
        let child = spawn(&["sh", "-c", "exit 0"]);
        let id = ProcessIdentity::capture(child.id());
        reap(child);
        let outcome = terminate_verified(&id, KillScope::Single, Duration::from_millis(50), false);
        assert_eq!(outcome, TerminationOutcome::AlreadyGone);
    }

    #[cfg(unix)]
    #[test]
    fn graceful_stop_of_signal_respecting_child() {
        // A plain `sleep` exits on SIGTERM.
        let child = spawn(&["sleep", "30"]);
        let id = ProcessIdentity::capture(child.id());
        let outcome = terminate_verified(&id, KillScope::Single, Duration::from_secs(5), false);
        assert_eq!(outcome, TerminationOutcome::Graceful);
        assert_eq!(identity_state(&id), IdentityState::Gone);
        reap(child);
    }

    #[cfg(unix)]
    #[test]
    fn timed_out_then_forced_for_sigterm_ignoring_child() {
        // Trap and ignore SIGTERM, then print `ready` so we only signal after the
        // trap is installed — no sleep-based race with the default disposition.
        let (child, _) = spawn_ready("trap '' TERM; echo ready; while true; do sleep 1; done");
        let id = ProcessIdentity::capture(child.id());

        // Non-forced stop must truthfully report it did not stop.
        let soft = terminate_verified(&id, KillScope::Single, Duration::from_millis(300), false);
        assert_eq!(soft, TerminationOutcome::TimedOut);
        assert!(!soft.stopped());
        assert_eq!(identity_state(&id), IdentityState::Matches);

        // Forced stop escalates to SIGKILL and actually terminates it.
        let hard = terminate_verified(&id, KillScope::Single, Duration::from_secs(5), true);
        assert_eq!(hard, TerminationOutcome::Forced);
        assert!(hard.stopped());
        assert!(!hard.graceful());
        assert_eq!(identity_state(&id), IdentityState::Gone);
        reap(child);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn tree_stop_waits_for_descendants() {
        // A parent shell with a background child. The child (a grandchild of this
        // test) must be terminated too — a graceful Tree stop that only confirmed
        // the root would leave it alive.
        let (child, gpid_line) = spawn_ready("sleep 300 & echo $!; wait");
        let grandchild: u32 = gpid_line.parse().expect("grandchild pid");
        assert!(
            crate::process_is_running(grandchild),
            "grandchild should be running before stop"
        );
        let id = ProcessIdentity::capture(child.id());

        let outcome = terminate_verified(&id, KillScope::Tree, Duration::from_secs(5), false);
        assert_eq!(outcome, TerminationOutcome::Graceful);
        assert_eq!(identity_state(&id), IdentityState::Gone);
        assert!(
            !crate::process_is_running(grandchild),
            "Tree stop must terminate the descendant, not just the root"
        );
        reap(child);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn tree_forced_kill_reaches_sigterm_ignoring_descendant() {
        // Root shell backgrounds an inner shell that ignores SIGTERM, prints its
        // own PID, then becomes `sleep` (SIG_IGN survives exec). A forced Tree
        // stop must SIGKILL that reparented descendant after the root exits.
        let (child, gpid_line) =
            spawn_ready(r#"sh -c "trap '' TERM; echo \$\$; exec sleep 300" & wait"#);
        let grandchild: u32 = gpid_line.parse().expect("grandchild pid");
        assert!(crate::process_is_running(grandchild));
        let id = ProcessIdentity::capture(child.id());

        let outcome = terminate_verified(&id, KillScope::Tree, Duration::from_millis(300), true);
        assert_eq!(outcome, TerminationOutcome::Forced);
        assert!(
            !crate::process_is_running(grandchild),
            "forced Tree stop must SIGKILL the SIGTERM-ignoring descendant"
        );
        reap(child);
    }
}
