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
//!
//! **The recycling defence is Linux-only in practice.** [`process_start_ticks`]
//! reads the start-time from `/proc` and is a compile-time `None` everywhere
//! else, so on Windows and macOS every record looks like one carrying no
//! recorded identity, and [`identity_state`] degrades to best-effort
//! [`IdentityState::Matches`] — the paragraph above describes what this module
//! enforces *where the platform can answer*. The degradation is deliberate: a
//! host that cannot tell two processes apart must not therefore refuse to stop
//! anything, since that would make every service unstoppable rather than making
//! any of them safer. What holds on those hosts is the rest of the caller's
//! gate — the port reality-check, the endpoint identity probe, and aborting
//! with the recovery tooling intact on any unconfirmed stop.

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
///
/// Reads the PID's current start-time itself. A caller that has already read it
/// — because it also needs the raw observation — should pass that one reading to
/// [`identity_state_with_observed`] rather than calling this and reading again,
/// so a process that exits between the two reads cannot produce two verdicts
/// derived from disagreeing observations.
#[must_use]
pub fn identity_state(id: &ProcessIdentity) -> IdentityState {
    identity_state_from_probes(
        id,
        || crate::process_is_running(id.pid) && !process_has_exited(id.pid),
        || process_start_ticks(id.pid),
    )
}

/// [`identity_state`] with its two observations taken lazily, so the order they
/// are taken in is a property of this body rather than of a call site.
///
/// Liveness BEFORE the reading, and the early return is what enforces it. The
/// reading must not sit in argument position — Rust evaluates arguments before
/// entering the callee, which inverts the safe side of the race. A PID read
/// while it was still the recorded process, which then exits and is recycled
/// before the liveness check, would be compared as `expected == actual` and come
/// back [`IdentityState::Matches`]: the one verdict that authorises a kill,
/// handed out for a PID that is now somebody else. Reading only after liveness
/// means the reading always describes whatever holds the PID *now*, so a
/// recycled one disagrees and comes back [`IdentityState::Recycled`].
///
/// The two probes are parameters purely so that ordering is testable. Staging
/// the real race needs a process to exit and its PID to be reissued inside a
/// window of microseconds, which no test can do; two closures that record when
/// they were called pin it deterministically instead. They are `FnOnce`
/// generics, so this costs nothing at runtime — the real call above
/// monomorphizes back into the same two direct calls.
fn identity_state_from_probes(
    id: &ProcessIdentity,
    is_live: impl FnOnce() -> bool,
    read_start_ticks: impl FnOnce() -> Option<u64>,
) -> IdentityState {
    if !is_live() {
        return IdentityState::Gone;
    }
    // Compare only. Routing back through `identity_state_with_observed` would
    // re-run the liveness check just performed, which on Linux is two more
    // `/proc` reads per call — paid on every 25 ms tick of the bounded waits,
    // which is where this is called from in a loop.
    compare_start_ticks(id, read_start_ticks())
}

/// The identity comparison alone, for callers that have already established
/// liveness.
///
/// Split out so the ordering guarantee above does not have to pay for a second
/// liveness check. Deliberately not public: on its own it cannot return
/// [`IdentityState::Gone`], so a caller that had not checked liveness would get
/// [`IdentityState::Matches`] for a dead PID — the one verdict that authorises a
/// kill. The two callers that establish liveness first are in this file.
const fn compare_start_ticks(
    id: &ProcessIdentity,
    observed_start_ticks: Option<u64>,
) -> IdentityState {
    match (id.start_ticks, observed_start_ticks) {
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

/// [`identity_state`] against a start-time the caller has already observed.
///
/// `observed_start_ticks` is what [`process_start_ticks`] returned for `id.pid`:
/// `None` both where the platform has no `/proc` and where that one PID's
/// start-time could not be read.
///
/// Liveness is checked here too, so a PID that simply exits after the caller's
/// reading still yields [`IdentityState::Gone`]. What that check cannot catch is
/// exit *and recycle* between the reading and this call: the PID is live again,
/// and a reading taken while it was still the recorded process matches the
/// record, so the verdict is [`IdentityState::Matches`] for a process that is no
/// longer ours. Callers holding a reading across anything slow should re-read
/// rather than pass a stale one; [`identity_state`] avoids the window entirely
/// by reading only after its own liveness check.
#[must_use]
pub fn identity_state_with_observed(
    id: &ProcessIdentity,
    observed_start_ticks: Option<u64>,
) -> IdentityState {
    if !crate::process_is_running(id.pid) || process_has_exited(id.pid) {
        return IdentityState::Gone;
    }
    compare_start_ticks(id, observed_start_ticks)
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
    #[cfg(not(windows))]
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

    // Unix can request a graceful shutdown before escalating. Windows has no
    // equivalent signal: do not burn the grace period on a no-op. A non-forced
    // Windows stop reports TimedOut immediately; a forced stop proceeds directly
    // to the verified kill path below.
    #[cfg(not(windows))]
    {
        send_signal(id.pid, Signal::Term, tree);
        if wait_for_all_exit(&members, grace) {
            return TerminationOutcome::Graceful;
        }

        if !force {
            return TerminationOutcome::TimedOut;
        }
    }
    #[cfg(windows)]
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
    debug_assert!(matches!(signal, Signal::Kill));
    crate::terminate_process(pid).is_ok()
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
    #[cfg(unix)]
    use std::process::{Child, Command, Stdio};

    #[test]
    fn a_dead_process_is_never_read_for_a_start_time() {
        // The ordering this module's safety rests on, pinned rather than
        // inspected. If the reading is ever hoisted back into argument
        // position — which reads as a harmless inlining, and was exactly the
        // regression a review caught here — it happens before the liveness
        // check, and a PID that exits and is recycled in between comes back
        // `Matches`: the one verdict that authorises a kill, for a process that
        // is no longer the recorded one.
        //
        // The real race is microseconds wide and needs a PID-space wrap, so it
        // cannot be staged. This stages the observable consequence instead: a
        // reading taken for a process already known dead is a reading that
        // could not have informed the verdict.
        let id = ProcessIdentity {
            pid: 4321,
            start_ticks: Some(99),
        };
        let mut read_start_ticks = false;

        let state = identity_state_from_probes(
            &id,
            || false,
            || {
                read_start_ticks = true;
                Some(99)
            },
        );

        assert_eq!(
            state,
            IdentityState::Gone,
            "a process that fails the liveness check is Gone whatever its start time reads as"
        );
        assert!(
            !read_start_ticks,
            "the start time must not be read at all once liveness has failed — if it was, the \
             read is back in argument position and the recycle race is inverted"
        );
    }

    #[test]
    fn a_live_process_is_judged_on_a_reading_taken_after_its_liveness_check() {
        // The other half: when liveness passes, the reading is taken, and it is
        // the reading — not the record — that decides. A recycled PID reads
        // differently and must come back `Recycled`.
        //
        // The PID is arbitrary, and nothing here establishes whether it is
        // running — it does not need to. Both probes are stubbed, so this path
        // never consults the real process table at all, and that is the point:
        // reaching the comparison at all proves the verdict came from the
        // liveness this function was handed rather than from a second check of
        // its own. While the comparison went back through
        // `identity_state_with_observed`, this same call returned `Gone` from
        // that function's own liveness check, whatever the stub said.
        let id = ProcessIdentity {
            pid: 4321,
            start_ticks: Some(99),
        };

        assert_eq!(
            identity_state_from_probes(&id, || true, || Some(99)),
            IdentityState::Matches,
            "a live PID still reading as its recorded start time is the recorded process"
        );
        assert_eq!(
            identity_state_from_probes(&id, || true, || Some(1234)),
            IdentityState::Recycled,
            "a live PID reading as a different start time is somebody else"
        );
    }

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

    /// The observation the caller passes must be the one classified, not a fresh
    /// read of the same PID. A caller that needs the raw start-time *and* the
    /// verdict reads once and passes it down precisely so the two cannot
    /// disagree; an implementation that quietly re-read `/proc` would restore
    /// that hazard while every existing test still passed. Here the live child's
    /// real start-time matches its recorded identity, so a re-reading
    /// implementation returns `Matches` — only one that honours the argument
    /// returns `Recycled`.
    #[cfg(target_os = "linux")]
    #[test]
    fn identity_state_with_observed_classifies_the_reading_it_was_given() {
        let (child, _) = spawn_ready("echo ready; while true; do sleep 1; done");
        let id = ProcessIdentity::capture(child.id());
        let real_ticks = id.start_ticks.expect("linux records a start-time");

        // Collect every verdict first, then kill and reap, and only then assert.
        // The child holds the harness's stdout pipe open, so an assertion that
        // panics ahead of the kill does not merely fail this test — it leaks a
        // looping process and hangs the whole suite on the pipe. Ask the
        // questions, clean up unconditionally, then judge.
        let agreeing = identity_state_with_observed(&id, Some(real_ticks));
        let disagreeing = identity_state_with_observed(&id, Some(real_ticks.wrapping_add(1)));
        let unreadable = identity_state_with_observed(&id, None);

        let hard = terminate_verified(&id, KillScope::Single, Duration::from_secs(5), true);
        reap(child);

        assert!(hard.stopped(), "the child must not outlive the test");
        assert_eq!(
            agreeing,
            IdentityState::Matches,
            "the reading that agrees with the record is a match"
        );
        assert_eq!(
            disagreeing,
            IdentityState::Recycled,
            "a disagreeing reading must be classified, not discarded for a re-read"
        );
        assert_eq!(
            unreadable,
            IdentityState::Indeterminate,
            "an unreadable start-time against a recorded one is unconfirmable"
        );
    }

    /// A terminated child that has not been reaped yet is a zombie: it still has
    /// a PID, so `kill(pid, 0)` succeeds and `process_is_running` reports `true`.
    /// Termination checks must therefore go through `identity_state`, which
    /// treats `Z` as gone. Asserting on `process_is_running` instead makes a test
    /// depend on how quickly the host reaps, which varies between environments.
    #[cfg(target_os = "linux")]
    #[test]
    fn zombie_is_gone_to_identity_state_but_still_running_to_a_bare_pid_check() {
        let (mut child, _) = spawn_ready("echo ready");
        let id = ProcessIdentity::capture(child.id());

        // Wait for exit without reaping, so the process is parked as a zombie.
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !process_has_exited(id.pid) {
            std::thread::sleep(POLL_INTERVAL);
        }

        assert!(
            process_has_exited(id.pid),
            "child should be an unreaped zombie by now"
        );
        assert!(
            crate::process_is_running(id.pid),
            "a bare PID check cannot tell a zombie from a live process"
        );
        assert_eq!(
            identity_state(&id),
            IdentityState::Gone,
            "a zombie has exited and must not count as running"
        );
        assert!(is_exited(identity_state(&id)));

        let _ = child.wait();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn tree_stop_waits_for_descendants() {
        // A parent shell with a background child. The child (a grandchild of this
        // test) must be terminated too — a graceful Tree stop that only confirmed
        // the root would leave it alive.
        let (child, gpid_line) = spawn_ready("sleep 300 & echo $!; wait");
        let grandchild = ProcessIdentity::capture(gpid_line.parse().expect("grandchild pid"));
        assert_eq!(
            identity_state(&grandchild),
            IdentityState::Matches,
            "grandchild should be running before stop"
        );
        let id = ProcessIdentity::capture(child.id());

        let outcome = terminate_verified(&id, KillScope::Tree, Duration::from_secs(5), false);
        assert_eq!(outcome, TerminationOutcome::Graceful);
        assert_eq!(identity_state(&id), IdentityState::Gone);
        assert!(
            is_exited(identity_state(&grandchild)),
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
        let grandchild = ProcessIdentity::capture(gpid_line.parse().expect("grandchild pid"));
        assert_eq!(identity_state(&grandchild), IdentityState::Matches);
        let id = ProcessIdentity::capture(child.id());

        let outcome = terminate_verified(&id, KillScope::Tree, Duration::from_millis(300), true);
        assert_eq!(outcome, TerminationOutcome::Forced);
        assert!(
            is_exited(identity_state(&grandchild)),
            "forced Tree stop must SIGKILL the SIGTERM-ignoring descendant"
        );
        reap(child);
    }
}
