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
        crate::process::process_tree_pids(id.pid)
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

/// How long a managed stop waits on a recorded process before it escalates.
///
/// Each recorded process, tree included, gets this long to exit after `SIGTERM`
/// before the stop escalates to `SIGKILL`, and then this long again before it
/// reports a timeout. The one value behind `rocm services stop` and `rocmd`'s
/// `stop_server`: both stop the same records, so neither may wait for a
/// different time.
pub const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// What a managed stop did about one PID a service record names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordedPidOutcome {
    /// The PID is the stopping process's own. It is reported but never
    /// signalled, and it counts as stopped: `rocmd` and an attached `rocm
    /// serve` record themselves as the supervisor of the service they launched,
    /// and the launcher is not the service.
    OwnProcess,
    /// The PID was handed to the verified termination, with this verdict.
    Terminated(TerminationOutcome),
}

impl RecordedPidOutcome {
    /// The label a stop report carries: `self` for the stopping process's own
    /// PID, otherwise [`TerminationOutcome::as_str`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OwnProcess => "self",
            Self::Terminated(outcome) => outcome.as_str(),
        }
    }
}

/// One recorded PID's fate in a managed stop.
///
/// Kept per PID because the outcomes that leave a stop unconfirmed are reported
/// differently: a `timed_out` PID was signalled and it, or a process under it,
/// outlived the deadline, while an `unverified` one was not signalled at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedPidStop {
    pub pid: u32,
    /// Which recorded PID this is: `supervisor`, `engine`, or
    /// `supervisor_and_engine` when the record holds the same PID for both.
    pub role: &'static str,
    pub outcome: RecordedPidOutcome,
}

impl RecordedPidStop {
    /// Whether a signal went out to this PID. A `timed_out` PID counts: the
    /// signal was sent and only the exit was never observed, which
    /// [`stopped`](Self::stopped) reports separately.
    #[must_use]
    pub const fn signaled(self) -> bool {
        matches!(
            self.outcome,
            RecordedPidOutcome::Terminated(
                TerminationOutcome::Graceful
                    | TerminationOutcome::Forced
                    | TerminationOutcome::TimedOut
            )
        )
    }

    /// Whether the stop escalated to a forced kill for this PID. That is not
    /// "the forced kill worked": a forced stop only reaches `timed_out` by way
    /// of the escalation, and [`stopped`](Self::stopped) says whether the
    /// process actually went away.
    #[must_use]
    pub const fn forced(self) -> bool {
        matches!(
            self.outcome,
            RecordedPidOutcome::Terminated(
                TerminationOutcome::Forced | TerminationOutcome::TimedOut
            )
        )
    }

    /// Whether the recorded process is confirmed no longer running.
    #[must_use]
    pub const fn stopped(self) -> bool {
        match self.outcome {
            RecordedPidOutcome::OwnProcess => true,
            RecordedPidOutcome::Terminated(outcome) => outcome.stopped(),
        }
    }

    /// The entry a stop report lists for this PID under `pid_outcomes`.
    #[must_use]
    pub fn report_entry(self) -> serde_json::Value {
        serde_json::json!({
            "pid": self.pid,
            "role": self.role,
            "outcome": self.outcome.as_str(),
        })
    }
}

/// Terminate the processes a service record names, verifying each PID's
/// identity first, and report what happened to each.
///
/// PIDs are recycled, so a persisted PID is not by itself evidence that the
/// recorded process still holds it. Each PID is paired with **its own**
/// start-time token (`supervisor_start_ticks` with `supervisor_pid`,
/// `engine_start_ticks` with `engine_pid`) and handed to `terminate`, which
/// production supplies as [`terminate_recorded_pid`]. When the two PIDs
/// coincide, the entry is reported once, under the role `supervisor_and_engine`,
/// and keeps a known token in preference to a missing one: a token is never read
/// across from the PID it does not belong to. The supervisor is visited first.
///
/// A zero PID names nothing and is skipped. The stopping process's own PID is
/// reported as [`RecordedPidOutcome::OwnProcess`] and `terminate` is not called
/// for it.
///
/// `terminate` is injectable so a caller's test can supply the verdicts it
/// cannot construct on demand (a process that survives `SIGKILL`, one whose
/// identity cannot be read).
pub fn terminate_recorded_service_pids(
    record: &crate::ManagedServiceRecord,
    terminate: impl Fn(&ProcessIdentity) -> TerminationOutcome,
) -> Vec<RecordedPidStop> {
    let mut entries: Vec<(u32, &'static str, Option<u64>)> = Vec::new();
    for (pid, role, ticks) in [
        (
            Some(record.supervisor_pid),
            "supervisor",
            record.supervisor_start_ticks,
        ),
        (record.engine_pid, "engine", record.engine_start_ticks),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _, _)| *seen == pid) {
            // Only the engine entry can land here, aliasing the supervisor: the
            // one PID is both, and is reported as both.
            existing.1 = "supervisor_and_engine";
            if existing.2.is_none() {
                existing.2 = ticks;
            }
        } else {
            entries.push((pid, role, ticks));
        }
    }

    entries
        .into_iter()
        .map(|(pid, role, ticks)| {
            let outcome = if pid == std::process::id() {
                RecordedPidOutcome::OwnProcess
            } else {
                RecordedPidOutcome::Terminated(terminate(&ProcessIdentity::new(pid, ticks)))
            };
            RecordedPidStop { pid, role, outcome }
        })
        .collect()
}

/// Terminate one recorded process, and its descendants, after verifying its
/// identity. The production `terminate` of [`terminate_recorded_service_pids`].
///
/// The stop is forced: a managed stop is definitive, so it must not report
/// success while an engine worker still holds the GPU. Only a process whose
/// recorded identity is confirmed is ever reached by it.
///
/// On Windows the descendants are taken separately, first. [`terminate_verified`]
/// cannot walk a process tree there (`process_tree_pids` returns just the root),
/// and the engine runs one level below the recorded PID, because the daemon
/// launches it through an `__engine-serve-http` process: a root-only kill would
/// leave the engine running and still holding the device.
///
/// Identity verification does not reach that far on Windows.
/// [`process_start_ticks`] has no `/proc` to read there and returns `None`, so
/// no recorded identity can be refuted and the [`IdentityState::Matches`] gate
/// only ever turns away a PID that is not running at all. A recorded PID that
/// has been recycled by an unrelated process still passes it, and `taskkill /T
/// /F` then takes that process and its whole subtree. Closing that needs a
/// Windows process creation time in [`process_start_ticks`].
pub fn terminate_recorded_pid(identity: &ProcessIdentity) -> TerminationOutcome {
    terminate_recorded_pid_with(identity, platform_tree_kill(), identity_state, |identity| {
        terminate_verified(identity, KillScope::Tree, MANAGED_STOP_GRACE, true)
    })
}

/// [`terminate_recorded_pid`] with its three platform and system touchpoints
/// supplied by the caller, so a test drives the tree-kill step on every host:
/// `tree_kill` takes one PID's descendants down (`None` where the platform
/// needs no separate step), `state_of` reads the PID's identity, and `terminate`
/// is the verified termination of the root.
fn terminate_recorded_pid_with(
    identity: &ProcessIdentity,
    tree_kill: Option<&dyn Fn(u32)>,
    state_of: impl Fn(&ProcessIdentity) -> IdentityState,
    terminate: impl Fn(&ProcessIdentity) -> TerminationOutcome,
) -> TerminationOutcome {
    if let Some(tree_kill) = tree_kill
        && matches!(state_of(identity), IdentityState::Matches)
    {
        // Taken while the root is still alive: `taskkill /T` resolves children
        // through the live parent, so it cannot reach them once the root is gone.
        tree_kill(identity.pid);
        // `terminate` still owns the bounded wait and the verdict, but a root
        // the tree kill already took now reads back as `AlreadyGone` — which
        // would deny a stop this command did perform.
        return match terminate(identity) {
            TerminationOutcome::AlreadyGone => TerminationOutcome::Forced,
            outcome => outcome,
        };
    }
    terminate(identity)
}

/// The separate descendant kill this platform needs before the verified
/// termination of a root, if any: `taskkill /T /F` on Windows, none elsewhere,
/// where [`terminate_verified`] walks the tree itself.
// The `cfg` arms differ in what they return, so the non-Windows arm cannot be a
// `const fn` without giving the two platforms different signatures.
#[allow(clippy::missing_const_for_fn)]
fn platform_tree_kill() -> Option<&'static dyn Fn(u32)> {
    #[cfg(windows)]
    {
        Some(&taskkill_tree)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
fn taskkill_tree(pid: u32) {
    use std::process::{Command, Stdio};
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
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
    crate::process::signal_process_scope(pid, raw, tree)
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
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
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
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
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

    // ---- The managed-stop routine `rocm` and `rocmd` share. ----

    use std::cell::RefCell;

    /// A PID no host has in use.
    const SUPERVISOR_PID: u32 = 999_999_998;
    const ENGINE_PID: u32 = 999_999_997;

    fn record_naming(
        supervisor: u32,
        supervisor_ticks: Option<u64>,
        engine: Option<u32>,
        engine_ticks: Option<u64>,
    ) -> crate::ManagedServiceRecord {
        let paths = crate::AppPaths {
            config_dir: std::path::PathBuf::from("config"),
            data_dir: std::path::PathBuf::from("data"),
            cache_dir: std::path::PathBuf::from("cache"),
        };
        let mut record = crate::ManagedServiceRecord::new(
            &paths,
            "svc-managed-stop",
            "vllm",
            "m",
            "m",
            "127.0.0.1",
            9,
            "managed",
            supervisor,
            None,
            None,
            None,
        );
        record.supervisor_start_ticks = supervisor_ticks;
        record.engine_pid = engine;
        record.engine_start_ticks = engine_ticks;
        record
    }

    /// The identities `terminate_recorded_service_pids` handed its termination
    /// function, in call order, and the stops it reported.
    fn stop_with(
        record: &crate::ManagedServiceRecord,
        outcome: TerminationOutcome,
    ) -> (Vec<ProcessIdentity>, Vec<RecordedPidStop>) {
        let seen = RefCell::new(Vec::new());
        let stops = terminate_recorded_service_pids(record, |identity| {
            seen.borrow_mut().push(*identity);
            outcome
        });
        (seen.into_inner(), stops)
    }

    #[test]
    fn managed_stop_grace_is_ten_seconds() {
        assert_eq!(MANAGED_STOP_GRACE, Duration::from_secs(10));
    }

    #[test]
    fn managed_stop_terminates_supervisor_then_engine_each_with_its_own_start_time() {
        let record = record_naming(SUPERVISOR_PID, Some(11), Some(ENGINE_PID), Some(22));

        let (seen, stops) = stop_with(&record, TerminationOutcome::Graceful);

        assert_eq!(
            seen,
            vec![
                ProcessIdentity::new(SUPERVISOR_PID, Some(11)),
                ProcessIdentity::new(ENGINE_PID, Some(22)),
            ],
            "the supervisor first, and no start-time read across from the other PID"
        );
        assert_eq!(
            stops
                .iter()
                .map(|stop| (stop.pid, stop.role))
                .collect::<Vec<_>>(),
            vec![(SUPERVISOR_PID, "supervisor"), (ENGINE_PID, "engine")]
        );
    }

    #[test]
    fn managed_stop_terminates_a_pid_recorded_twice_once_and_keeps_a_known_start_time() {
        // Either entry may be the one that carries the token.
        for (supervisor_ticks, engine_ticks, expected) in [
            (None, Some(33), Some(33)),
            (Some(44), None, Some(44)),
            (Some(44), Some(33), Some(44)),
            (None, None, None),
        ] {
            let record = record_naming(
                SUPERVISOR_PID,
                supervisor_ticks,
                Some(SUPERVISOR_PID),
                engine_ticks,
            );

            let (seen, stops) = stop_with(&record, TerminationOutcome::Graceful);

            assert_eq!(
                seen,
                vec![ProcessIdentity::new(SUPERVISOR_PID, expected)],
                "{supervisor_ticks:?} / {engine_ticks:?}"
            );
            assert_eq!(stops.len(), 1);
            assert_eq!(stops[0].role, "supervisor_and_engine");
        }
    }

    #[test]
    fn managed_stop_skips_a_zero_pid_and_an_absent_engine() {
        let record = record_naming(0, None, Some(0), None);

        let (seen, stops) = stop_with(&record, TerminationOutcome::Graceful);

        assert!(seen.is_empty());
        assert!(stops.is_empty());
    }

    #[test]
    fn managed_stop_reports_its_own_pid_as_self_and_never_terminates_it() {
        let own = std::process::id();
        let record = record_naming(own, None, Some(ENGINE_PID), None);

        let (seen, stops) = stop_with(&record, TerminationOutcome::Graceful);

        assert_eq!(
            seen,
            vec![ProcessIdentity::new(ENGINE_PID, None)],
            "the stopping process is never signalled"
        );
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0].pid, own);
        assert_eq!(stops[0].role, "supervisor");
        assert_eq!(stops[0].outcome, RecordedPidOutcome::OwnProcess);
        assert_eq!(stops[0].outcome.as_str(), "self");
        assert!(!stops[0].signaled() && !stops[0].forced());
        assert!(stops[0].stopped(), "the launcher is not the service");
        assert_eq!(
            stops[0].report_entry(),
            serde_json::json!({"pid": own, "role": "supervisor", "outcome": "self"})
        );
    }

    #[test]
    fn managed_stop_reports_a_pid_that_is_both_roles_and_its_own_as_self() {
        let own = std::process::id();
        let record = record_naming(own, None, Some(own), None);

        let (seen, stops) = stop_with(&record, TerminationOutcome::Graceful);

        assert!(seen.is_empty());
        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0].role, "supervisor_and_engine");
        assert_eq!(stops[0].outcome, RecordedPidOutcome::OwnProcess);
    }

    #[test]
    fn managed_stop_classifies_each_outcome_as_signaled_forced_and_stopped() {
        use TerminationOutcome::{
            AlreadyGone, Forced, Graceful, IdentityMismatch, TimedOut, Unverified,
        };
        let record = record_naming(SUPERVISOR_PID, None, None, None);

        // (outcome, signaled, forced, stopped)
        let expected = [
            (AlreadyGone, false, false, true),
            (IdentityMismatch, false, false, true),
            (Unverified, false, false, false),
            (Graceful, true, false, true),
            (Forced, true, true, true),
            (TimedOut, true, true, false),
        ];
        for (outcome, signaled, forced, stopped) in expected {
            let (_, stops) = stop_with(&record, outcome);
            assert_eq!(stops.len(), 1);
            let stop = stops[0];
            assert_eq!(stop.outcome, RecordedPidOutcome::Terminated(outcome));
            assert_eq!(stop.outcome.as_str(), outcome.as_str());
            assert_eq!(
                (stop.signaled(), stop.forced(), stop.stopped()),
                (signaled, forced, stopped),
                "{outcome:?}"
            );
        }
    }

    /// What one `terminate_recorded_pid_with` call did, in order.
    #[derive(Debug, PartialEq, Eq)]
    enum Step {
        TreeKill(u32),
        Terminate(u32),
    }

    /// Run `terminate_recorded_pid_with` for `pid`, with a tree-kill step
    /// present or not, the identity in `state`, and the root's termination
    /// answering `outcome`. Returns the verdict and the steps taken.
    fn tree_kill_run(
        has_tree_kill: bool,
        state: IdentityState,
        outcome: TerminationOutcome,
    ) -> (TerminationOutcome, Vec<Step>) {
        let steps = RefCell::new(Vec::new());
        let tree_kill = |pid: u32| steps.borrow_mut().push(Step::TreeKill(pid));
        let verdict = terminate_recorded_pid_with(
            &ProcessIdentity::new(SUPERVISOR_PID, None),
            has_tree_kill.then_some(&tree_kill as &dyn Fn(u32)),
            |_| state,
            |identity| {
                steps.borrow_mut().push(Step::Terminate(identity.pid));
                outcome
            },
        );
        (verdict, steps.into_inner())
    }

    #[test]
    fn the_tree_is_killed_before_the_root_is_terminated() {
        let (_, steps) = tree_kill_run(true, IdentityState::Matches, TerminationOutcome::Forced);

        assert_eq!(
            steps,
            vec![
                Step::TreeKill(SUPERVISOR_PID),
                Step::Terminate(SUPERVISOR_PID)
            ],
            "`taskkill /T` resolves children through the live parent, so the \
             tree kill has to come first"
        );
    }

    #[test]
    fn a_root_the_tree_kill_already_took_is_reported_forced_not_already_gone() {
        let (verdict, _) = tree_kill_run(
            true,
            IdentityState::Matches,
            TerminationOutcome::AlreadyGone,
        );

        assert_eq!(
            verdict,
            TerminationOutcome::Forced,
            "the stop did perform the kill; reading the root back as gone must not deny it"
        );
    }

    #[test]
    fn the_verdict_after_a_tree_kill_is_the_terminations_own_for_every_other_outcome() {
        use TerminationOutcome::{Forced, Graceful, IdentityMismatch, TimedOut, Unverified};
        for outcome in [Graceful, Forced, TimedOut, Unverified, IdentityMismatch] {
            let (verdict, _) = tree_kill_run(true, IdentityState::Matches, outcome);

            assert_eq!(verdict, outcome);
        }
    }

    #[test]
    fn no_tree_kill_is_run_for_a_pid_that_is_gone_recycled_or_unverifiable() {
        for state in [
            IdentityState::Gone,
            IdentityState::Recycled,
            IdentityState::Indeterminate,
        ] {
            let (verdict, steps) = tree_kill_run(true, state, TerminationOutcome::AlreadyGone);

            assert_eq!(
                steps,
                vec![Step::Terminate(SUPERVISOR_PID)],
                "{state:?}: a stranger's process tree must not be taken"
            );
            assert_eq!(
                verdict,
                TerminationOutcome::AlreadyGone,
                "{state:?}: and the verdict is the termination's, unmapped"
            );
        }
    }

    #[test]
    fn without_a_tree_kill_step_the_root_is_terminated_and_its_verdict_unmapped() {
        let (verdict, steps) = tree_kill_run(
            false,
            IdentityState::Matches,
            TerminationOutcome::AlreadyGone,
        );

        assert_eq!(steps, vec![Step::Terminate(SUPERVISOR_PID)]);
        assert_eq!(verdict, TerminationOutcome::AlreadyGone);
    }

    #[test]
    fn only_windows_has_a_separate_tree_kill_step() {
        // Ties the injected tests above to what production actually runs: the
        // step they exercise is the one `terminate_recorded_pid` has on Windows
        // and does without elsewhere, where `terminate_verified` walks the tree.
        assert_eq!(platform_tree_kill().is_some(), cfg!(windows));
    }

    #[test]
    fn terminate_recorded_pid_reports_an_absent_pid_as_already_gone() {
        let outcome = terminate_recorded_pid(&ProcessIdentity::new(SUPERVISOR_PID, None));

        assert_eq!(outcome, TerminationOutcome::AlreadyGone);
    }

    #[cfg(unix)]
    #[test]
    fn terminate_recorded_pid_stops_a_running_verified_process() {
        let child = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let id = ProcessIdentity::capture(child.id());

        let outcome = terminate_recorded_pid(&id);

        assert!(outcome.stopped(), "{outcome:?}");
        assert!(matches!(
            outcome,
            TerminationOutcome::Graceful | TerminationOutcome::Forced
        ));
        reap(child);
        assert!(!crate::process_is_running(id.pid));
    }
}
