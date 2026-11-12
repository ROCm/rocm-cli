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

use crate::process::ProcessProbe;

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
/// boot) on Linux, and its creation time as a `FILETIME` on Windows. It is
/// `None` on other platforms, and for a record written before Windows recorded
/// one, where identity cannot be verified and callers fall back to best-effort
/// behaviour.
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
    /// cannot be read, so it can be neither confirmed nor refuted; or the PID
    /// cannot be queried at all (on Windows, a live process this one may not
    /// open), so it cannot be told from a process that is gone. The process must
    /// not be signalled, and it must not be reported as stopped.
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
    identity_state_with(
        id,
        crate::process::probe_process,
        process_has_exited,
        process_start_ticks,
    )
}

/// [`identity_state`] with its three system reads supplied by the caller: whether
/// the PID can be queried and is running, whether it is a zombie, and its
/// current start-time token. The liveness gate in front of every stop, so a test
/// drives each answer a read can give on every host.
///
/// A PID that cannot be queried is [`IdentityState::Indeterminate`], not
/// [`IdentityState::Gone`]: a failure to look is not an exit, and reading it as
/// one would let a stop report a live process as stopped. That is one of the two
/// routes to [`IdentityState::Indeterminate`] and so to an `unverified` stop; the
/// other is a recorded token that cannot be read back from a running process
/// ([`classify_live_identity`]).
///
/// A token that was recorded and can still be read, and that differs, outranks
/// the failed query: it proves a different process holds the PID, so the
/// recorded one is gone. On Windows this narrows to the case where the process
/// opens but the exit-code query fails: the token and the probe both open the
/// PID with `PROCESS_QUERY_LIMITED_INFORMATION`, so a process that cannot be
/// opened has no readable token either, and stays [`IdentityState::Indeterminate`].
fn identity_state_with(
    id: &ProcessIdentity,
    probe: impl Fn(u32) -> ProcessProbe,
    has_exited: impl Fn(u32) -> bool,
    start_ticks: impl Fn(u32) -> Option<u64>,
) -> IdentityState {
    match probe(id.pid) {
        ProcessProbe::Gone => IdentityState::Gone,
        ProcessProbe::Unqueryable => match (id.start_ticks, start_ticks(id.pid)) {
            (Some(recorded), Some(live)) if recorded != live => IdentityState::Recycled,
            _ => IdentityState::Indeterminate,
        },
        ProcessProbe::Running if has_exited(id.pid) => IdentityState::Gone,
        ProcessProbe::Running => classify_live_identity(id.start_ticks, start_ticks(id.pid)),
    }
}

/// The identity verdict for a PID that is running: what its recorded token
/// (`recorded`) says about the token read from the live process now (`live`).
///
/// The one decision every stop path turns on, apart from the system reads, so
/// a test covers each case on every host. A process is refused only when a
/// token was recorded and the live one is provably different. A record with no
/// token (an older record, or one written where no token could be read) cannot
/// be refuted and is a match whatever the live process reads, so its stop and
/// its tree kill go ahead as they always did. A recorded token that cannot be
/// confirmed now is neither signalled nor reported stopped.
const fn classify_live_identity(recorded: Option<u64>, live: Option<u64>) -> IdentityState {
    match (recorded, live) {
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

/// The verdict that ends a termination before anything is signalled, when the
/// recorded identity says not to: `None` for a process that matches and so goes
/// on to be stopped.
const fn refusal_before_signalling(state: IdentityState) -> Option<TerminationOutcome> {
    match state {
        IdentityState::Gone => Some(TerminationOutcome::AlreadyGone),
        IdentityState::Recycled => Some(TerminationOutcome::IdentityMismatch),
        IdentityState::Indeterminate => Some(TerminationOutcome::Unverified),
        IdentityState::Matches => None,
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
    ///
    /// Counted as stopped, on every platform, because a token that is recorded
    /// and readable and differs is positive evidence of a different process. The
    /// engine's own token comes from the same [`process_start_ticks`] that reads
    /// it back here, taken for that PID's process when it was spawned or reported
    /// (a creation time does not change over a process's life), so a mismatch for
    /// the real engine is not an expected outcome. It stays visible as
    /// `identity_mismatch` in the report rather than as a signal sent to a
    /// process nobody could prove was the server's.
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
    /// The recorded process itself is gone (this stop's own wait saw it exit, or
    /// it was already gone), but the separate kill of the processes under it did
    /// not complete (on Windows, a `taskkill` that failed, stalled, or found no
    /// such process while the root was still running), so one of them may still
    /// be running and holding the device. Deliberately not
    /// [`TimedOut`](Self::TimedOut): the root did not outlive any deadline.
    DescendantsUnconfirmed,
    /// The recorded process was gone by the time the separate kill of the
    /// processes under it ran (on Windows, `taskkill` found no such process and
    /// the root's re-read agrees): it exited by itself after the identity check.
    /// `taskkill /T` reaches children only through a live parent, and nothing
    /// recorded what was under the root, so this stop cannot say whether a
    /// process it left running is still there. Not a stop on its own; the
    /// aggregate over the record's processes confirms it when another recorded
    /// process is confirmed gone ([`terminate_recorded_service_pids`]).
    ExitedBeforeTreeKill,
}

impl TerminationOutcome {
    /// Is the recorded service confirmed to be no longer running as a result?
    ///
    /// `false` for [`TimedOut`](Self::TimedOut) (still alive),
    /// [`Unverified`](Self::Unverified) (could not be confirmed either way) and
    /// [`DescendantsUnconfirmed`](Self::DescendantsUnconfirmed) (the root is gone
    /// but a process under it may not be) and
    /// [`ExitedBeforeTreeKill`](Self::ExitedBeforeTreeKill) (the root is gone and
    /// nothing says what was under it).
    #[must_use]
    pub const fn stopped(self) -> bool {
        !matches!(
            self,
            Self::TimedOut
                | Self::Unverified
                | Self::DescendantsUnconfirmed
                | Self::ExitedBeforeTreeKill
        )
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
            Self::DescendantsUnconfirmed => "descendants_unconfirmed",
            Self::ExitedBeforeTreeKill => "exited_before_tree_kill",
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
    if let Some(verdict) = refusal_before_signalling(identity_state(id)) {
        return verdict;
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
                    | TerminationOutcome::DescendantsUnconfirmed
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
                TerminationOutcome::Forced
                    | TerminationOutcome::TimedOut
                    | TerminationOutcome::DescendantsUnconfirmed
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
///
/// One verdict is settled across the record rather than per PID. A root that
/// exited by itself before its processes could be taken down
/// ([`TerminationOutcome::ExitedBeforeTreeKill`]) cannot say what it left
/// running. It becomes [`TerminationOutcome::AlreadyGone`] only when another
/// process the record names is confirmed gone (a stop, or itself such a root):
/// the record's other process is the one thing that was under the root and is
/// independently verified. A root that is the only process the record names, or
/// whose sibling is still running or unconfirmed, stays unconfirmed. Processes
/// the record does not name that were under a root can be neither found nor
/// verified, which is the same limit as for a root that was already gone when a
/// stop began.
///
/// This relies on the engine being visited as its own role, which it is: the
/// engine PID is an entry of its own, with its own token and its own verdict, so
/// a worker the root left running is reached by the engine's own termination
/// when the record names it. Both callers fold every row (`rocm services stop`
/// in `stop_internal_managed_service_confirmed_with`, `rocmd` in
/// `stop_managed_service_with`): the stop is confirmed, and the recorded PIDs and
/// the endpoint key cleared, only when every recorded role is individually
/// stopped. A root that exited alone therefore never confirms a stop on its own
/// row, and the record is kept when the engine role is not confirmed.
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

    let mut stops: Vec<RecordedPidStop> = entries
        .into_iter()
        .map(|(pid, role, ticks)| {
            let outcome = if pid == std::process::id() {
                RecordedPidOutcome::OwnProcess
            } else {
                RecordedPidOutcome::Terminated(terminate(&ProcessIdentity::new(pid, ticks)))
            };
            RecordedPidStop { pid, role, outcome }
        })
        .collect();
    settle_roots_that_exited_alone(&mut stops);
    stops
}

/// Whether `stop` vouches for what was under another recorded root: a recorded
/// process this stop took down ([`Graceful`](TerminationOutcome::Graceful),
/// [`Forced`](TerminationOutcome::Forced)), found already gone
/// ([`AlreadyGone`](TerminationOutcome::AlreadyGone)), or a root that exited by
/// itself ([`TerminationOutcome::ExitedBeforeTreeKill`]). Not
/// [`IdentityMismatch`](TerminationOutcome::IdentityMismatch): it says a stranger
/// holds that PID now, which is "gone" for that process but is no evidence about
/// what was under a different root. The stopping process's own PID is not a
/// service process and never counts.
const fn recorded_process_is_gone(stop: RecordedPidStop) -> bool {
    matches!(
        stop.outcome,
        RecordedPidOutcome::Terminated(
            TerminationOutcome::Graceful
                | TerminationOutcome::Forced
                | TerminationOutcome::AlreadyGone
                | TerminationOutcome::ExitedBeforeTreeKill
        )
    )
}

/// Turn each [`TerminationOutcome::ExitedBeforeTreeKill`] into
/// [`TerminationOutcome::AlreadyGone`] when some other recorded process is
/// confirmed gone, and leave it unconfirmed otherwise. Judged on the verdicts as
/// they came in, so two roots that both exited by themselves corroborate each
/// other.
fn settle_roots_that_exited_alone(stops: &mut [RecordedPidStop]) {
    let corroborated: Vec<bool> = (0..stops.len())
        .map(|index| {
            stops
                .iter()
                .enumerate()
                .any(|(other, stop)| other != index && recorded_process_is_gone(*stop))
        })
        .collect();
    for (stop, corroborated) in stops.iter_mut().zip(corroborated) {
        if corroborated
            && matches!(
                stop.outcome,
                RecordedPidOutcome::Terminated(TerminationOutcome::ExitedBeforeTreeKill)
            )
        {
            stop.outcome = RecordedPidOutcome::Terminated(TerminationOutcome::AlreadyGone);
        }
    }
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
/// The tree kill runs only for a PID that passes the same identity gate as the
/// rest of the stop. A record written with a start-time token carries the
/// process's creation time on Windows ([`process_start_ticks`]), so a PID that
/// an unrelated process has since taken reads back [`IdentityState::Recycled`]
/// and neither it nor its subtree is touched. A record written before Windows
/// recorded a token has none to refute, and for it the gate only turns away a
/// PID that is not running at all: that old record can still name a recycled
/// PID, and `taskkill /T /F` then takes that process and its subtree. A stop
/// that confirms clears the record's PIDs, so that exposure shrinks to the
/// records written before the token existed and not yet stopped.
///
/// On Windows the PID is held from before the identity check until the stop
/// ends: a process's PID is not handed to another process while any handle to it
/// is open, so the creation time that was compared is the creation time of the
/// process `taskkill` and the root termination then reach by PID. Without the
/// hold, the process could exit and an unrelated one take its PID between the
/// check and the kill.
///
/// The descendant kill is bounded: a `taskkill` that has not finished within
/// [`MANAGED_STOP_GRACE`] is killed and reaped and counts as a tree kill that did
/// not complete, so a stalled helper cannot hang the stop.
///
/// A tree kill that did not complete leaves the stop unconfirmed, whatever the
/// root's own termination says: [`terminate_verified`] can only watch the root
/// on Windows, so a descendant that `taskkill` did not take down is one this stop
/// cannot see exit, and reporting the stop confirmed would clear the record and
/// the endpoint key while a worker may still hold the device.
pub fn terminate_recorded_pid(identity: &ProcessIdentity) -> TerminationOutcome {
    terminate_recorded_pid_within(identity, MANAGED_STOP_GRACE)
}

/// [`terminate_recorded_pid`] with the root's grace chosen by the caller.
///
/// For a process that has nothing to flush, such as a launch child cleaned up
/// because its record could not be published. Everything else is the same
/// routine: the PID hold, the identity gate, and on Windows the descendant kill,
/// whose own deadline stays [`MANAGED_STOP_GRACE`].
pub fn terminate_recorded_pid_within(
    identity: &ProcessIdentity,
    grace: std::time::Duration,
) -> TerminationOutcome {
    terminate_recorded_pid_with(
        identity,
        platform_tree_kill(),
        hold_pid,
        identity_state,
        |identity| terminate_verified(identity, KillScope::Tree, grace, true),
    )
}

/// Whether the separate descendant kill took the tree down.
// Only Windows has the separate step, so nothing builds a value elsewhere; the
// tests build both, and the decision on them is shared by every platform.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKill {
    /// The kill ran and reported that every process it reached is terminated.
    Completed,
    /// The kill ran and reported that it found no such process (`taskkill` exits
    /// 128): the root was gone by the time it looked. Not by itself a failure,
    /// and not a success either; the root's own verdict decides which.
    NotFound,
    /// The kill could not run, or reported that it did not take every process
    /// down. Which descendants survive is not known.
    Failed,
}

/// The verdict for a stop whose descendant kill ended as `tree` and whose root
/// termination answered `outcome`.
///
/// When the tree kill did not complete ([`TreeKill::Failed`], or
/// [`TreeKill::NotFound`] while the root was still there), a root that this
/// stop saw exit ([`Graceful`](TerminationOutcome::Graceful),
/// [`Forced`](TerminationOutcome::Forced)) or that was already gone is reported
/// [`TerminationOutcome::DescendantsUnconfirmed`]: the root is gone, which is
/// what the wait observed, but nothing watched the processes under it, so the
/// stop cannot be called confirmed. It is not `timed_out`, which would claim
/// that something outlived a deadline. The verdicts that already deny the stop
/// ([`TimedOut`](TerminationOutcome::TimedOut), the root itself outlived the
/// deadline; [`Unverified`](TerminationOutcome::Unverified)) stay as they are.
///
/// A kill that found no such process ([`TreeKill::NotFound`]) is judged by the
/// root's own re-read, which `outcome` carries. When the root is gone
/// ([`TerminationOutcome::AlreadyGone`]) it exited on its own between the
/// identity check and the kill: that is [`TerminationOutcome::ExitedBeforeTreeKill`],
/// not a stop. `taskkill /T` reaches children only through a live parent, so a
/// child the root left running is out of its reach, and the stop kept no list of
/// what was under the root to check afterwards. The aggregate over the record's
/// processes decides whether another recorded process corroborates the exit
/// ([`terminate_recorded_service_pids`]). When the root is still there, "no such
/// process" contradicts what the termination found, so the kill is treated as
/// failed.
const fn after_tree_kill(tree: TreeKill, outcome: TerminationOutcome) -> TerminationOutcome {
    match (tree, outcome) {
        (TreeKill::Completed, outcome) => outcome,
        (TreeKill::NotFound, TerminationOutcome::AlreadyGone) => {
            TerminationOutcome::ExitedBeforeTreeKill
        }
        (
            TreeKill::NotFound | TreeKill::Failed,
            TerminationOutcome::Graceful
            | TerminationOutcome::Forced
            | TerminationOutcome::AlreadyGone,
        ) => TerminationOutcome::DescendantsUnconfirmed,
        (TreeKill::NotFound | TreeKill::Failed, outcome) => outcome,
    }
}

/// [`terminate_recorded_pid`] with its platform and system touchpoints supplied
/// by the caller, so a test drives the tree-kill step on every host:
/// `tree_kill` takes one PID's descendants down (`None` where the platform needs
/// no separate step), `hold` keeps the PID reserved, answering `None` when it
/// could not, and is dropped when the call returns, `state_of` reads the PID's
/// identity, and `terminate` is the verified termination of the root.
fn terminate_recorded_pid_with<Held>(
    identity: &ProcessIdentity,
    tree_kill: Option<&dyn Fn(u32) -> TreeKill>,
    hold: impl FnOnce(&ProcessIdentity) -> Option<Held>,
    state_of: impl Fn(&ProcessIdentity) -> IdentityState,
    terminate: impl Fn(&ProcessIdentity) -> TerminationOutcome,
) -> TerminationOutcome {
    // Before anything reads the identity, and until the end of the call.
    let held = hold(identity);
    if let Some(tree_kill) = tree_kill {
        // The identity read that decides about the tree kill also decides about
        // the stop. A refusal here (gone, recycled, unverifiable) is the verdict:
        // the root is not handed to `terminate`, which re-reads the identity and
        // could find a process that was unverifiable a moment ago verifiable now,
        // and then stop the root alone with the tree kill skipped.
        if let Some(refusal) = refusal_before_signalling(state_of(identity)) {
            return refusal;
        }
        // An identity that matches is only worth acting on while the PID cannot
        // change hands before `taskkill` reaches it. Without a hold nothing
        // reserves it between this read and the kill, which takes a whole tree by
        // PID, so the stop is not confirmed and nothing is signalled. A process
        // that could not be opened and also cannot be read was refused above.
        if held.is_none() {
            return TerminationOutcome::Unverified;
        }
        // Taken while the root is still alive: `taskkill /T` resolves children
        // through the live parent, so it cannot reach them once the root is gone.
        let tree = tree_kill(identity.pid);
        // `terminate` still owns the bounded wait and the verdict. A root the
        // tree kill already took now reads back as `AlreadyGone`, which would
        // deny a stop this command did perform, so it counts as forced; but when
        // the kill found no such process the root went by itself, and
        // `AlreadyGone` is the truth.
        let outcome = match (tree, terminate(identity)) {
            (TreeKill::NotFound, outcome) => outcome,
            (_, TerminationOutcome::AlreadyGone) => TerminationOutcome::Forced,
            (_, outcome) => outcome,
        };
        return after_tree_kill(tree, outcome);
    }
    terminate(identity)
}

/// What keeps a PID reserved while a stop works on it: on Windows an open
/// handle to the process, which stops the OS giving the PID to another process;
/// nothing elsewhere, where the stop signals by PID and `terminate_verified`
/// re-verifies it.
#[cfg(windows)]
struct PidHold {
    _handle: std::os::windows::io::OwnedHandle,
}

#[cfg(not(windows))]
struct PidHold;

/// Hold `identity`'s PID, or `None` when the process cannot be opened. A stop
/// whose identity matches but whose PID is not held is not carried out, see
/// [`terminate_recorded_pid_with`].
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn hold_pid(identity: &ProcessIdentity) -> Option<PidHold> {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    // SAFETY: `OpenProcess` takes only plain values and returns a handle or null;
    // there is no pointer to keep valid.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, identity.pid) };
    if handle.is_null() {
        return None;
    }
    Some(PidHold {
        // SAFETY: `handle` is non-null, `OpenProcess` just returned it and nothing
        // else owns or closes it, so the `OwnedHandle` may take ownership and
        // close it exactly once on drop.
        _handle: unsafe { OwnedHandle::from_raw_handle(handle) },
    })
}

// The `cfg` sibling calls non-const code, so this arm cannot be `const` without
// giving the two platforms different signatures.
#[cfg(not(windows))]
#[allow(clippy::missing_const_for_fn, clippy::unnecessary_wraps)]
fn hold_pid(_identity: &ProcessIdentity) -> Option<PidHold> {
    Some(PidHold)
}

/// The separate descendant kill this platform needs before the verified
/// termination of a root, if any: `taskkill /T /F` on Windows, none elsewhere,
/// where [`terminate_verified`] walks the tree itself.
// The `cfg` arms differ in what they return, so the non-Windows arm cannot be a
// `const fn` without giving the two platforms different signatures.
#[allow(clippy::missing_const_for_fn)]
fn platform_tree_kill() -> Option<&'static dyn Fn(u32) -> TreeKill> {
    #[cfg(windows)]
    {
        Some(&taskkill_tree)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// `taskkill`'s exit code for "the process was not found".
#[cfg(any(windows, test))]
const TASKKILL_PROCESS_NOT_FOUND: i32 = 128;

/// The verdict on one `taskkill` run, from whether it started and the exit code
/// it ended with. A clean exit (0) is a completed tree kill and 128 is
/// [`TreeKill::NotFound`], left to the root's re-read to judge. A failed launch,
/// a run that did not finish within its deadline
/// ([`std::io::ErrorKind::TimedOut`], see [`run_to_exit_code`]), any other
/// nonzero code (a process it could not terminate) and a run that ended without
/// a code are all [`TreeKill::Failed`]. Split out of the process spawn so a test
/// runs it on every host.
#[cfg(any(windows, test))]
fn classify_taskkill(run: std::io::Result<Option<i32>>) -> TreeKill {
    match run {
        Ok(Some(0)) => TreeKill::Completed,
        Ok(Some(TASKKILL_PROCESS_NOT_FOUND)) => TreeKill::NotFound,
        Ok(_) | Err(_) => TreeKill::Failed,
    }
}

/// How long the descendant kill may run. It happens before
/// [`terminate_verified`]'s bounded waits, so without its own deadline a stalled
/// helper would hang the stop; the same bound as every other wait of a managed
/// stop.
#[cfg(windows)]
const TREE_KILL_DEADLINE: Duration = MANAGED_STOP_GRACE;

/// Start `command` and wait for it to exit, for at most `deadline`. Returns its
/// exit code (`None` when it ended without one). A child still running at the
/// deadline is killed and reaped, so none is left behind, and the call fails
/// with [`std::io::ErrorKind::TimedOut`].
#[cfg(any(windows, test))]
fn run_to_exit_code(
    command: &mut std::process::Command,
    deadline: Duration,
) -> std::io::Result<Option<i32>> {
    let mut child = command.spawn()?;
    wait_for_exit_code(&mut child, deadline)
}

/// The wait of [`run_to_exit_code`], on a child already started. Kills and reaps
/// `child` when the deadline passes, or when it cannot be polled, rather than
/// leaving it running.
#[cfg(any(windows, test))]
fn wait_for_exit_code(
    child: &mut std::process::Child,
    deadline: Duration,
) -> std::io::Result<Option<i32>> {
    let end = Instant::now() + deadline;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status.code()),
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        }
        let now = Instant::now();
        if now >= end {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "the process did not exit before its deadline",
            ));
        }
        std::thread::sleep(POLL_INTERVAL.min(end.saturating_duration_since(now)));
    }
}

/// The arguments that take `pid` and everything under it down: by PID, with its
/// tree (`/T`), forced (`/F`). Split out so a test pins them on every host.
#[cfg(any(windows, test))]
fn taskkill_args(pid: u32) -> [String; 4] {
    [
        "/PID".to_owned(),
        pid.to_string(),
        "/T".to_owned(),
        "/F".to_owned(),
    ]
}

#[cfg(windows)]
fn taskkill_tree(pid: u32) -> TreeKill {
    use std::process::{Command, Stdio};
    classify_taskkill(run_to_exit_code(
        Command::new(crate::windows_system_tool("taskkill.exe"))
            .args(taskkill_args(pid))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
        TREE_KILL_DEADLINE,
    ))
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
/// Returns `None` when the value cannot be read, including on platforms with
/// neither `/proc` nor a creation time to read (neither Linux nor Windows),
/// where identity verification degrades to best-effort.
#[cfg(target_os = "linux")]
#[must_use]
pub fn process_start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    parse_start_ticks(&stat)
}

/// Read the creation time of `pid` (`GetProcessTimes`) as the token that tells
/// the process from a later one that reuses its PID.
///
/// The value is the creation time as a Windows `FILETIME` (100 ns intervals
/// since 1601), not clock ticks since boot: only equality with a value read
/// earlier from the same PID means anything. `None` when the process cannot be
/// opened for querying (it is gone, or belongs to a more privileged account).
#[cfg(windows)]
#[must_use]
#[allow(unsafe_code)] // Win32 FFI
pub fn process_start_ticks(pid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    if pid == 0 {
        return None;
    }
    // SAFETY: plain Win32 calls. The handle is checked before use and closed on
    // every path, and the four out-parameters are valid for the one call.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut creation = FILETIME::default();
        let mut exit = FILETIME::default();
        let mut kernel = FILETIME::default();
        let mut user = FILETIME::default();
        let read = GetProcessTimes(
            handle,
            &raw mut creation,
            &raw mut exit,
            &raw mut kernel,
            &raw mut user,
        );
        CloseHandle(handle);
        (read != 0).then(|| filetime_ticks(creation.dwLowDateTime, creation.dwHighDateTime))
    }
}

/// The 64-bit value of a Windows `FILETIME`, which a Win32 call returns as two
/// 32-bit halves. Kept apart from the FFI so a test can run it on every host.
#[cfg(any(windows, test))]
fn filetime_ticks(low: u32, high: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

#[cfg(not(any(target_os = "linux", windows)))]
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
        // The root is gone but a process under it may not be: not a stop, not
        // graceful, and not the label of a timeout.
        assert!(!TerminationOutcome::DescendantsUnconfirmed.stopped());
        assert!(!TerminationOutcome::DescendantsUnconfirmed.graceful());
        assert_eq!(
            TerminationOutcome::DescendantsUnconfirmed.as_str(),
            "descendants_unconfirmed"
        );
        assert_ne!(
            TerminationOutcome::DescendantsUnconfirmed.as_str(),
            TerminationOutcome::TimedOut.as_str()
        );
        // A root that exited by itself before its tree kill: not a stop alone.
        assert!(!TerminationOutcome::ExitedBeforeTreeKill.stopped());
        assert!(!TerminationOutcome::ExitedBeforeTreeKill.graceful());
        assert_eq!(
            TerminationOutcome::ExitedBeforeTreeKill.as_str(),
            "exited_before_tree_kill"
        );

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

    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn identity_matches_our_own_process() {
        let pid = std::process::id();
        let id = ProcessIdentity::capture(pid);
        assert!(id.start_ticks.is_some(), "should read our own start-time");
        assert_eq!(identity_state(&id), IdentityState::Matches);
    }

    #[cfg(any(target_os = "linux", windows))]
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

    #[cfg(any(target_os = "linux", windows))]
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
            AlreadyGone, DescendantsUnconfirmed, ExitedBeforeTreeKill, Forced, Graceful,
            IdentityMismatch, TimedOut, Unverified,
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
            // Signalled and forced by the tree kill, but not confirmed gone.
            (DescendantsUnconfirmed, true, true, false),
            // Nothing signalled: the root was gone before the tree kill ran.
            (ExitedBeforeTreeKill, false, false, false),
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
        Hold(u32),
        /// The hold could not be taken: nothing reserves the PID.
        HoldFailed(u32),
        TreeKill(u32),
        Terminate(u32),
        Release,
    }

    /// A stand-in for the Windows PID hold: records that it was taken and, when
    /// it is dropped, that it was released.
    struct RecordedHold<'a>(&'a RefCell<Vec<Step>>);

    impl Drop for RecordedHold<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push(Step::Release);
        }
    }

    /// Run `terminate_recorded_pid_with` for `pid`, with a tree-kill step
    /// present or not and ending as `tree`, the identity in `state`, and the
    /// root's termination answering `outcome`. Returns the verdict and the steps
    /// taken, the hold's release included.
    fn tree_kill_run_with(
        has_tree_kill: bool,
        tree: TreeKill,
        state: IdentityState,
        outcome: TerminationOutcome,
    ) -> (TerminationOutcome, Vec<Step>) {
        let steps = RefCell::new(Vec::new());
        let tree_kill = |pid: u32| {
            steps.borrow_mut().push(Step::TreeKill(pid));
            tree
        };
        let verdict = terminate_recorded_pid_with(
            &ProcessIdentity::new(SUPERVISOR_PID, None),
            has_tree_kill.then_some(&tree_kill as &dyn Fn(u32) -> TreeKill),
            |identity| {
                steps.borrow_mut().push(Step::Hold(identity.pid));
                Some(RecordedHold(&steps))
            },
            |_| state,
            |identity| {
                steps.borrow_mut().push(Step::Terminate(identity.pid));
                outcome
            },
        );
        (verdict, steps.into_inner())
    }

    /// [`tree_kill_run_with`] for a tree kill that completes.
    fn tree_kill_run(
        has_tree_kill: bool,
        state: IdentityState,
        outcome: TerminationOutcome,
    ) -> (TerminationOutcome, Vec<Step>) {
        tree_kill_run_with(has_tree_kill, TreeKill::Completed, state, outcome)
    }

    #[test]
    fn the_tree_is_killed_before_the_root_is_terminated() {
        let (_, steps) = tree_kill_run(true, IdentityState::Matches, TerminationOutcome::Forced);

        assert_eq!(
            steps,
            vec![
                Step::Hold(SUPERVISOR_PID),
                Step::TreeKill(SUPERVISOR_PID),
                Step::Terminate(SUPERVISOR_PID),
                Step::Release,
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
                vec![Step::Hold(SUPERVISOR_PID), Step::Release],
                "{state:?}: a stranger's process tree must not be taken, and the root \
                 is not handed to the termination, which would re-read the identity"
            );
            let expected = match state {
                IdentityState::Gone => TerminationOutcome::AlreadyGone,
                IdentityState::Recycled => TerminationOutcome::IdentityMismatch,
                _ => TerminationOutcome::Unverified,
            };
            assert_eq!(
                verdict, expected,
                "{state:?}: the initial refusal is the verdict"
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

        assert_eq!(
            steps,
            vec![
                Step::Hold(SUPERVISOR_PID),
                Step::Terminate(SUPERVISOR_PID),
                Step::Release
            ]
        );
        assert_eq!(verdict, TerminationOutcome::AlreadyGone);
    }

    /// Every way a recorded token and the live process's token can stand, with
    /// what the stop does about each: whether the Windows tree kill runs, and the
    /// verdict the stop reports (the label and whether it counts as stopped).
    ///
    /// The live process is running in every row; the tokens go through the same
    /// decision production uses ([`classify_live_identity`] and
    /// [`refusal_before_signalling`]), with the system reads and the signalling
    /// supplied, so each row runs on every host. A process is refused only when
    /// a token was recorded and the live one is provably different.
    #[test]
    fn what_a_stop_does_for_each_recorded_and_live_token() {
        const LIVE: u64 = 0x01DC_0000_0000_0001;
        struct Row {
            name: &'static str,
            recorded: Option<u64>,
            live: Option<u64>,
            tree_kill_runs: bool,
            label: &'static str,
            stopped: bool,
        }
        let row = |name, recorded, live, tree_kill_runs, label, stopped| Row {
            name,
            recorded,
            live,
            tree_kill_runs,
            label,
            stopped,
        };
        let rows = [
            // The shape `service-stop-07` records, and any record older than the
            // token: nothing to refute, so the whole tree is taken as before.
            row("no token recorded", None, Some(LIVE), true, "forced", true),
            row(
                "no token recorded, none readable",
                None,
                None,
                true,
                "forced",
                true,
            ),
            row(
                "token matches",
                Some(LIVE),
                Some(LIVE),
                true,
                "forced",
                true,
            ),
            // A recycled PID: the process is a stranger, left alone.
            row(
                "token differs",
                Some(LIVE + 1),
                Some(LIVE),
                false,
                "identity_mismatch",
                true,
            ),
            // A recorded token that cannot be read back is neither signalled nor
            // reported stopped.
            row(
                "token unreadable",
                Some(LIVE),
                None,
                false,
                "unverified",
                false,
            ),
        ];
        for Row {
            name: row,
            recorded,
            live,
            tree_kill_runs,
            label,
            stopped,
        } in rows
        {
            let steps = RefCell::new(Vec::new());
            let tree_kill = |pid: u32| {
                steps.borrow_mut().push(Step::TreeKill(pid));
                TreeKill::Completed
            };
            let verdict = terminate_recorded_pid_with(
                &ProcessIdentity::new(SUPERVISOR_PID, recorded),
                Some(&tree_kill as &dyn Fn(u32) -> TreeKill),
                |id| {
                    steps.borrow_mut().push(Step::Hold(id.pid));
                    Some(RecordedHold(&steps))
                },
                |id| classify_live_identity(id.start_ticks, live),
                |id| {
                    steps.borrow_mut().push(Step::Terminate(id.pid));
                    // What the verified termination answers: its refusal for a
                    // process the identity rules out, otherwise the tree kill
                    // has already taken the root and it reads back as gone.
                    refusal_before_signalling(classify_live_identity(id.start_ticks, live))
                        .unwrap_or(TerminationOutcome::AlreadyGone)
                },
            );
            let expected_steps = if tree_kill_runs {
                vec![
                    Step::Hold(SUPERVISOR_PID),
                    Step::TreeKill(SUPERVISOR_PID),
                    Step::Terminate(SUPERVISOR_PID),
                    Step::Release,
                ]
            } else {
                // Refused on the initial read: not handed to the termination.
                vec![Step::Hold(SUPERVISOR_PID), Step::Release]
            };
            assert_eq!(steps.into_inner(), expected_steps, "{row}: steps");
            assert_eq!(verdict.as_str(), label, "{row}: verdict");
            assert_eq!(verdict.stopped(), stopped, "{row}: counts as stopped");
        }
    }

    #[test]
    fn a_recorded_token_refutes_a_live_one_only_when_both_are_known_and_differ() {
        use IdentityState::{Indeterminate, Matches, Recycled};
        assert_eq!(classify_live_identity(None, None), Matches);
        assert_eq!(classify_live_identity(None, Some(7)), Matches);
        assert_eq!(classify_live_identity(Some(7), Some(7)), Matches);
        assert_eq!(classify_live_identity(Some(7), Some(8)), Recycled);
        assert_eq!(classify_live_identity(Some(7), None), Indeterminate);
        assert_eq!(refusal_before_signalling(Matches), None);
    }

    #[test]
    fn only_windows_has_a_separate_tree_kill_step() {
        // Ties the injected tests above to what production actually runs: the
        // step they exercise is the one `terminate_recorded_pid` has on Windows
        // and does without elsewhere, where `terminate_verified` walks the tree.
        assert_eq!(platform_tree_kill().is_some(), cfg!(windows));
    }

    /// Run `terminate_recorded_pid_with` for a PID whose hold could not be taken
    /// (the process could not be opened), the identity reading as `state`.
    fn unheld_run(
        has_tree_kill: bool,
        state: IdentityState,
        outcome: TerminationOutcome,
    ) -> (TerminationOutcome, Vec<Step>) {
        let steps = RefCell::new(Vec::new());
        let tree_kill = |pid: u32| {
            steps.borrow_mut().push(Step::TreeKill(pid));
            TreeKill::Completed
        };
        let verdict = terminate_recorded_pid_with(
            &ProcessIdentity::new(SUPERVISOR_PID, None),
            has_tree_kill.then_some(&tree_kill as &dyn Fn(u32) -> TreeKill),
            |identity| {
                steps.borrow_mut().push(Step::HoldFailed(identity.pid));
                None::<RecordedHold<'_>>
            },
            |_| state,
            |identity| {
                steps.borrow_mut().push(Step::Terminate(identity.pid));
                outcome
            },
        );
        (verdict, steps.into_inner())
    }

    #[test]
    fn a_tree_kill_is_not_run_for_a_pid_that_could_not_be_held_even_when_its_identity_matches() {
        // Nothing reserves the PID between the identity read and `taskkill`, which
        // takes a whole tree by PID, so a recycled PID would lose its tree. The
        // stop is not confirmed, and neither the tree kill nor the termination runs.
        let (verdict, steps) = unheld_run(true, IdentityState::Matches, TerminationOutcome::Forced);

        assert_eq!(verdict, TerminationOutcome::Unverified);
        assert!(!verdict.stopped());
        assert_eq!(steps, vec![Step::HoldFailed(SUPERVISOR_PID)]);
    }

    #[test]
    fn an_unheld_pid_that_is_gone_recycled_or_unverifiable_keeps_its_refusal() {
        // The hold only adds a refusal for an identity that matches: a process
        // already known to be gone, a stranger, or unreadable keeps the verdict
        // the identity read gives it.
        for (state, expected) in [
            (IdentityState::Gone, TerminationOutcome::AlreadyGone),
            (
                IdentityState::Recycled,
                TerminationOutcome::IdentityMismatch,
            ),
            (IdentityState::Indeterminate, TerminationOutcome::Unverified),
        ] {
            let (verdict, steps) = unheld_run(true, state, TerminationOutcome::Forced);

            assert_eq!(verdict, expected, "{state:?}");
            assert_eq!(steps, vec![Step::HoldFailed(SUPERVISOR_PID)], "{state:?}");
        }
    }

    #[test]
    fn without_a_tree_kill_step_an_unheld_pid_is_still_terminated() {
        // Elsewhere nothing is held and the termination re-verifies the identity
        // itself, so the missing hold must not turn a stop into a refusal.
        let (verdict, steps) =
            unheld_run(false, IdentityState::Matches, TerminationOutcome::Graceful);

        assert_eq!(verdict, TerminationOutcome::Graceful);
        assert_eq!(
            steps,
            vec![
                Step::HoldFailed(SUPERVISOR_PID),
                Step::Terminate(SUPERVISOR_PID)
            ]
        );
    }

    #[test]
    fn the_pid_is_held_before_its_identity_is_read_and_released_after_the_stop() {
        // The hold is what makes the creation time that was compared the
        // creation time of the process the kill then reaches by PID: taken
        // first, released last. `Hold` comes before anything else and `Release`
        // after the root's termination, with the tree kill between.
        let (_, steps) = tree_kill_run(true, IdentityState::Matches, TerminationOutcome::Forced);

        assert_eq!(steps.first(), Some(&Step::Hold(SUPERVISOR_PID)));
        assert_eq!(steps.last(), Some(&Step::Release));
        assert_eq!(
            steps
                .iter()
                .filter(|step| matches!(step, Step::Release))
                .count(),
            1
        );
    }

    #[test]
    fn a_tree_kill_that_did_not_complete_leaves_the_stop_unconfirmed() {
        use TerminationOutcome::{
            AlreadyGone, DescendantsUnconfirmed, Forced, Graceful, TimedOut, Unverified,
        };
        // Every root verdict that would confirm the stop becomes the one that
        // says the root is gone but a process under it may still be running, not
        // `timed_out`, which would say something outlived a deadline; the
        // verdicts that already deny the stop are kept.
        for (root, expected) in [
            (Forced, DescendantsUnconfirmed),
            (Graceful, DescendantsUnconfirmed),
            (AlreadyGone, DescendantsUnconfirmed),
            (TimedOut, TimedOut),
            (Unverified, Unverified),
        ] {
            let (verdict, steps) =
                tree_kill_run_with(true, TreeKill::Failed, IdentityState::Matches, root);

            assert_eq!(verdict, expected, "root {root:?}");
            assert!(
                !verdict.stopped(),
                "root {root:?}: the stop is not confirmed"
            );
            assert!(
                steps.contains(&Step::Terminate(SUPERVISOR_PID)),
                "the root is still terminated, best effort: {steps:?}"
            );
        }
    }

    /// `taskkill` answering "no such process" is judged by the root's own
    /// re-read. The root that went away by itself between the identity check and
    /// the kill is not reported as a kill this stop performed, and not as a stop
    /// either on its own: `taskkill /T` reaches children only through a live
    /// parent, so a child the root left running was out of its reach, and nothing
    /// recorded what was under it. The record-level settling decides.
    #[test]
    fn a_tree_kill_that_found_no_process_with_the_root_gone_is_not_a_stop_by_itself() {
        let (verdict, steps) = tree_kill_run_with(
            true,
            TreeKill::NotFound,
            IdentityState::Matches,
            TerminationOutcome::AlreadyGone,
        );

        assert_eq!(
            verdict,
            TerminationOutcome::ExitedBeforeTreeKill,
            "the root exited on its own and nothing says what it left running"
        );
        assert!(!verdict.stopped());
        assert!(
            steps.contains(&Step::Terminate(SUPERVISOR_PID)),
            "the root is re-read by the termination: {steps:?}"
        );
    }

    /// The identity read that gates the tree kill is the one that decides the
    /// stop. If it is unverifiable and a second read, inside the termination,
    /// would find the process verifiable, the root must not be stopped alone with
    /// the tree kill skipped: the initial refusal is the verdict and the
    /// termination is never called.
    #[test]
    fn an_unverifiable_first_read_is_not_overtaken_by_a_verifiable_second_read() {
        use std::cell::Cell;
        let reads = Cell::new(0_u32);
        let terminated = Cell::new(false);
        let tree_killed = Cell::new(false);
        let tree_kill = |_pid: u32| {
            tree_killed.set(true);
            TreeKill::Completed
        };

        let verdict = terminate_recorded_pid_with(
            &ProcessIdentity::new(SUPERVISOR_PID, Some(7)),
            Some(&tree_kill as &dyn Fn(u32) -> TreeKill),
            |_| Some(()),
            |_| {
                // Indeterminate the first time, matching from then on.
                reads.set(reads.get() + 1);
                if reads.get() == 1 {
                    IdentityState::Indeterminate
                } else {
                    IdentityState::Matches
                }
            },
            |_| {
                terminated.set(true);
                TerminationOutcome::Forced
            },
        );

        assert_eq!(verdict, TerminationOutcome::Unverified);
        assert!(!verdict.stopped());
        assert!(!terminated.get(), "the root is not stopped alone");
        assert!(!tree_killed.get());
        assert_eq!(reads.get(), 1, "the initial read is the only one");
    }

    /// A sibling whose PID a stranger now holds ([`IdentityMismatch`]) says
    /// nothing about what was under this root, so it does not corroborate it.
    #[test]
    fn an_identity_mismatch_sibling_does_not_corroborate_a_root_that_exited_alone() {
        use TerminationOutcome::{ExitedBeforeTreeKill, IdentityMismatch};
        let record = record_naming(SUPERVISOR_PID, None, Some(ENGINE_PID), None);

        let verdicts = settled(
            &record,
            &[
                (SUPERVISOR_PID, ExitedBeforeTreeKill),
                (ENGINE_PID, IdentityMismatch),
            ],
        );

        assert_eq!(verdicts, vec![ExitedBeforeTreeKill, IdentityMismatch]);
        assert!(!verdicts[0].stopped());
    }

    /// What the record's processes say together about a root that exited by
    /// itself. `exits` maps each recorded PID to the verdict `terminate` gives.
    fn settled(
        record: &crate::ManagedServiceRecord,
        exits: &[(u32, TerminationOutcome)],
    ) -> Vec<TerminationOutcome> {
        terminate_recorded_service_pids(record, |identity| {
            exits
                .iter()
                .find(|(pid, _)| *pid == identity.pid)
                .map(|(_, outcome)| *outcome)
                .expect("a verdict for every recorded pid")
        })
        .into_iter()
        .map(|stop| match stop.outcome {
            RecordedPidOutcome::Terminated(outcome) => outcome,
            RecordedPidOutcome::OwnProcess => panic!("not the stopping process"),
        })
        .collect()
    }

    /// A parent that exits by itself while its child keeps running is not a
    /// stop: with only the parent recorded, nothing independently verifies the
    /// child, so the stop stays unconfirmed (and the record, the stop request
    /// and the key are kept by the callers). This is the case a root's exit
    /// cannot settle.
    #[test]
    fn a_parent_that_exits_alone_with_nothing_else_recorded_is_not_a_confirmed_stop() {
        use TerminationOutcome::ExitedBeforeTreeKill;
        let record = record_naming(SUPERVISOR_PID, None, None, None);

        let verdicts = settled(&record, &[(SUPERVISOR_PID, ExitedBeforeTreeKill)]);

        assert_eq!(verdicts, vec![ExitedBeforeTreeKill]);
        assert!(!verdicts[0].stopped());
    }

    /// The same parent with its engine recorded: the engine is the process under
    /// it that is independently verified, by its own entry. Confirmed gone, the
    /// stop is confirmed, which keeps the concern that a root exiting on its own
    /// must not leave a stop unconfirmed for nothing.
    #[test]
    fn a_parent_that_exits_alone_is_confirmed_when_the_recorded_engine_is_confirmed_gone() {
        use TerminationOutcome::{AlreadyGone, ExitedBeforeTreeKill, Forced, Graceful};
        let record = record_naming(SUPERVISOR_PID, None, Some(ENGINE_PID), None);

        for engine in [Forced, Graceful, AlreadyGone] {
            let verdicts = settled(
                &record,
                &[(SUPERVISOR_PID, ExitedBeforeTreeKill), (ENGINE_PID, engine)],
            );

            assert_eq!(verdicts, vec![AlreadyGone, engine], "engine {engine:?}");
            assert!(verdicts.iter().all(|verdict| verdict.stopped()));
        }
        // And the other way round: the engine is the root that exited alone.
        let verdicts = settled(
            &record,
            &[
                (SUPERVISOR_PID, Graceful),
                (ENGINE_PID, ExitedBeforeTreeKill),
            ],
        );
        assert_eq!(verdicts, vec![Graceful, AlreadyGone]);
    }

    /// A parent that exits while the recorded engine is still running, or not
    /// confirmed, stays unconfirmed: the engine is the child that survived.
    #[test]
    fn a_parent_that_exits_alone_stays_unconfirmed_while_the_recorded_engine_is_not_gone() {
        use TerminationOutcome::{
            DescendantsUnconfirmed, ExitedBeforeTreeKill, TimedOut, Unverified,
        };
        let record = record_naming(SUPERVISOR_PID, None, Some(ENGINE_PID), None);

        for engine in [TimedOut, Unverified, DescendantsUnconfirmed] {
            let verdicts = settled(
                &record,
                &[(SUPERVISOR_PID, ExitedBeforeTreeKill), (ENGINE_PID, engine)],
            );

            assert_eq!(
                verdicts,
                vec![ExitedBeforeTreeKill, engine],
                "engine {engine:?}"
            );
            assert!(
                !verdicts[0].stopped(),
                "the parent is not settled by {engine:?}"
            );
        }
    }

    /// Two roots that both exited by themselves corroborate each other: every
    /// process the record names is gone.
    #[test]
    fn two_roots_that_both_exited_alone_are_a_confirmed_stop() {
        use TerminationOutcome::{AlreadyGone, ExitedBeforeTreeKill};
        let record = record_naming(SUPERVISOR_PID, None, Some(ENGINE_PID), None);

        let verdicts = settled(
            &record,
            &[
                (SUPERVISOR_PID, ExitedBeforeTreeKill),
                (ENGINE_PID, ExitedBeforeTreeKill),
            ],
        );

        assert_eq!(verdicts, vec![AlreadyGone, AlreadyGone]);
    }

    /// The stopping process's own PID is not a service process: it never
    /// corroborates a root that exited alone.
    #[test]
    fn the_stopping_process_does_not_corroborate_a_root_that_exited_alone() {
        use TerminationOutcome::ExitedBeforeTreeKill;
        let record = record_naming(SUPERVISOR_PID, None, Some(std::process::id()), None);

        let stops = terminate_recorded_service_pids(&record, |_| ExitedBeforeTreeKill);

        assert_eq!(
            stops[0].outcome,
            RecordedPidOutcome::Terminated(ExitedBeforeTreeKill)
        );
        assert_eq!(stops[1].outcome, RecordedPidOutcome::OwnProcess);
        assert!(!stops[0].stopped());
    }

    /// "No such process" while the root is still there is a contradiction, so it
    /// is a failed kill and the stop stays unconfirmed, as for any other nonzero
    /// exit.
    #[test]
    fn a_tree_kill_that_found_no_process_while_the_root_runs_is_a_failed_kill() {
        use TerminationOutcome::{DescendantsUnconfirmed, Forced, Graceful, TimedOut, Unverified};
        // The termination saw the root exit (Graceful, Forced) although the kill
        // said there was no such process: the exit was observed, so the report
        // must not say it outlived a deadline, but nothing watched what is under
        // it.
        for (root, expected) in [
            (Forced, DescendantsUnconfirmed),
            (Graceful, DescendantsUnconfirmed),
            (TimedOut, TimedOut),
            (Unverified, Unverified),
        ] {
            let (verdict, _) =
                tree_kill_run_with(true, TreeKill::NotFound, IdentityState::Matches, root);

            assert_eq!(verdict, expected, "root {root:?}");
            assert!(!verdict.stopped(), "root {root:?}");
        }
    }

    /// Any other nonzero exit with the root gone still leaves the stop
    /// unconfirmed: unlike "not found", it says a process could not be
    /// terminated, so a descendant may survive the root.
    #[test]
    fn a_failed_tree_kill_with_the_root_gone_is_still_unconfirmed() {
        let (verdict, _) = tree_kill_run_with(
            true,
            TreeKill::Failed,
            IdentityState::Matches,
            TerminationOutcome::AlreadyGone,
        );

        assert_eq!(verdict, TerminationOutcome::DescendantsUnconfirmed);
        assert!(!verdict.stopped());
    }

    /// A recorded token that can be read and differs proves another process
    /// holds the PID, even when the rest of the query failed; without a readable
    /// token the PID stays unverifiable.
    #[test]
    fn a_readable_token_mismatch_outranks_a_failed_query() {
        use crate::process::ProcessProbe::Unqueryable;
        let state = |recorded, live| {
            identity_state_with(
                &ProcessIdentity::new(SUPERVISOR_PID, recorded),
                |_| Unqueryable,
                |_| false,
                |_| live,
            )
        };

        assert_eq!(state(Some(7), Some(8)), IdentityState::Recycled);
        assert_eq!(state(Some(7), Some(7)), IdentityState::Indeterminate);
        assert_eq!(state(Some(7), None), IdentityState::Indeterminate);
        assert_eq!(state(None, Some(8)), IdentityState::Indeterminate);
        assert_eq!(state(None, None), IdentityState::Indeterminate);
    }

    #[test]
    fn a_completed_tree_kill_leaves_the_roots_verdict_alone() {
        use TerminationOutcome::{Forced, Graceful, TimedOut, Unverified};
        for outcome in [Forced, Graceful, TimedOut, Unverified] {
            let (verdict, _) =
                tree_kill_run_with(true, TreeKill::Completed, IdentityState::Matches, outcome);

            assert_eq!(verdict, outcome);
        }
    }

    /// A command that exits at once with `code`, and one that runs far longer
    /// than any test waits, on whichever platform the tests run.
    fn exits_with(code: i32) -> std::process::Command {
        #[cfg(unix)]
        let mut command = {
            let mut command = std::process::Command::new("sh");
            command.args(["-c", &format!("exit {code}")]);
            command
        };
        #[cfg(windows)]
        let mut command = {
            let mut command = std::process::Command::new("cmd");
            command.args(["/C", &format!("exit {code}")]);
            command
        };
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        command
    }

    fn runs_for_thirty_seconds() -> std::process::Command {
        #[cfg(unix)]
        let mut command = {
            let mut command = std::process::Command::new("sleep");
            command.arg("30");
            command
        };
        #[cfg(windows)]
        let mut command = {
            let mut command = std::process::Command::new("ping");
            command.args(["-n", "30", "127.0.0.1"]);
            command
        };
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        command
    }

    #[test]
    fn a_helper_that_exits_in_time_gives_its_exit_code() {
        let deadline = Duration::from_secs(20);

        assert_eq!(
            run_to_exit_code(&mut exits_with(0), deadline).unwrap(),
            Some(0)
        );
        assert_eq!(
            run_to_exit_code(&mut exits_with(3), deadline).unwrap(),
            Some(3)
        );
    }

    #[test]
    fn a_helper_that_cannot_be_started_is_an_error() {
        let mut command = std::process::Command::new("rocm-core-test-no-such-program");

        assert!(run_to_exit_code(&mut command, Duration::from_secs(1)).is_err());
    }

    /// A helper that stalls must not hold the stop: it is killed and reaped at
    /// its deadline, the call ends close to it, and the verdict is a failed tree
    /// kill, which leaves the stop unconfirmed.
    #[test]
    fn a_helper_that_stalls_is_killed_and_reaped_at_its_deadline() {
        let mut command = runs_for_thirty_seconds();
        let mut child = command.spawn().expect("spawn the stalled helper");
        let pid = child.id();
        let started = Instant::now();

        let run = wait_for_exit_code(&mut child, Duration::from_millis(200));
        let elapsed = started.elapsed();

        let error = run.expect_err("a stalled helper does not finish");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(
            elapsed >= Duration::from_millis(200) && elapsed < Duration::from_secs(10),
            "returned after {elapsed:?}, far from the deadline"
        );
        // Reaped: its status is already collected, and the process is gone.
        assert!(
            matches!(child.try_wait(), Ok(Some(_))),
            "the killed helper was waited for"
        );
        assert!(
            !crate::process_is_running(pid),
            "the helper was killed, not left running"
        );
        assert_eq!(classify_taskkill(Err(error)), TreeKill::Failed);
    }

    #[test]
    fn taskkill_is_asked_for_the_pid_its_tree_and_force() {
        assert_eq!(taskkill_args(4242), ["/PID", "4242", "/T", "/F"]);
    }

    #[test]
    fn only_a_clean_taskkill_exit_is_a_completed_tree_kill_and_128_is_not_found() {
        use std::io::{Error, ErrorKind};
        assert_eq!(classify_taskkill(Ok(Some(0))), TreeKill::Completed);
        // A process it could not terminate.
        assert_eq!(classify_taskkill(Ok(Some(1))), TreeKill::Failed);
        assert_eq!(classify_taskkill(Ok(Some(5))), TreeKill::Failed);
        // No such process: not a failure by itself, the root's re-read decides.
        assert_eq!(classify_taskkill(Ok(Some(128))), TreeKill::NotFound);
        // Ended without an exit code.
        assert_eq!(classify_taskkill(Ok(None)), TreeKill::Failed);
        // Could not be started, or did not finish before its deadline.
        assert_eq!(
            classify_taskkill(Err(Error::from(ErrorKind::NotFound))),
            TreeKill::Failed
        );
        assert_eq!(
            classify_taskkill(Err(Error::from(ErrorKind::TimedOut))),
            TreeKill::Failed
        );
    }

    /// A PID that cannot be queried is neither gone nor a match, through the
    /// same liveness gate every stop starts with.
    #[test]
    fn a_pid_that_cannot_be_queried_is_unverified_not_already_gone() {
        use crate::process::ProcessProbe::{Gone, Running, Unqueryable};
        let state = |probe, zombie, recorded, live| {
            identity_state_with(
                &ProcessIdentity::new(SUPERVISOR_PID, recorded),
                |_| probe,
                |_| zombie,
                |_| live,
            )
        };
        // Whatever token was recorded, and whatever can be read of the process,
        // short of a readable token that differs (`a_readable_token_mismatch_...`).
        for (recorded, live) in [
            (None, None),
            (None, Some(7)),
            (None, Some(8)),
            (Some(7), None),
            (Some(7), Some(7)),
        ] {
            assert_eq!(
                state(Unqueryable, false, recorded, live),
                IdentityState::Indeterminate,
                "recorded {recorded:?}, live {live:?}"
            );
        }
        assert_eq!(state(Gone, false, Some(7), Some(7)), IdentityState::Gone);
        assert_eq!(state(Running, true, Some(7), Some(7)), IdentityState::Gone);
        assert_eq!(
            state(Running, false, Some(7), Some(7)),
            IdentityState::Matches
        );
        assert_eq!(
            state(Running, false, Some(7), Some(8)),
            IdentityState::Recycled
        );
        assert_eq!(
            state(Running, false, Some(7), None),
            IdentityState::Indeterminate
        );
    }

    /// The whole stop for a live process the caller may not query: not
    /// signalled, no tree kill, and reported as unverified rather than as
    /// stopped, so the record and the endpoint key are kept.
    #[test]
    fn a_stop_of_a_live_process_it_cannot_query_is_not_confirmed() {
        use crate::process::ProcessProbe::Unqueryable;
        let steps = RefCell::new(Vec::new());
        let tree_kill = |pid: u32| {
            steps.borrow_mut().push(Step::TreeKill(pid));
            TreeKill::Completed
        };
        let state_of =
            |id: &ProcessIdentity| identity_state_with(id, |_| Unqueryable, |_| false, |_| None);
        let verdict = terminate_recorded_pid_with(
            &ProcessIdentity::new(SUPERVISOR_PID, Some(7)),
            Some(&tree_kill as &dyn Fn(u32) -> TreeKill),
            |_| Some(()),
            state_of,
            // The verified termination's own gate, with the same read.
            |id| refusal_before_signalling(state_of(id)).unwrap_or(TerminationOutcome::Forced),
        );

        assert_eq!(verdict, TerminationOutcome::Unverified);
        assert!(!verdict.stopped());
        assert!(
            steps.into_inner().is_empty(),
            "a process that cannot be told from a stranger is not tree-killed"
        );
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

    #[test]
    fn a_filetime_is_its_high_word_above_its_low_word() {
        // The Windows creation time arrives as two 32-bit halves; swapping them
        // or dropping the shift would make every process read as a different one.
        assert_eq!(
            filetime_ticks(0x89AB_CDEF, 0x0123_4567),
            0x0123_4567_89AB_CDEF
        );
        assert_eq!(filetime_ticks(1, 0), 1);
        assert_eq!(filetime_ticks(0, 1), 1 << 32);
        assert_eq!(filetime_ticks(u32::MAX, u32::MAX), u64::MAX);
    }

    /// The managed stop's own entry point refuses a PID whose recorded identity no
    /// longer matches, and reaches neither it nor its tree. Our own PID with a
    /// wrong start-time is what a recycled PID looks like; if the stop went
    /// ahead, `taskkill /T /F` on Windows (a plain forced kill elsewhere) would
    /// end this test process instead of asserting. On Windows this runs the real
    /// tree-kill gate: the verdict has to come from the identity check, before
    /// any `taskkill`.
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn terminate_recorded_pid_refuses_a_pid_whose_recorded_identity_no_longer_matches() {
        let pid = std::process::id();
        let real = process_start_ticks(pid).expect("own start-time");
        let stale = ProcessIdentity::new(pid, Some(real.wrapping_add(1)));

        let outcome = terminate_recorded_pid(&stale);

        assert_eq!(outcome, TerminationOutcome::IdentityMismatch);
        assert!(crate::process_is_running(pid), "nothing was signalled");
    }

    /// The same refusal against a bystander that is a different process holding
    /// the recorded PID (the recycled case), then the stop of that process when
    /// its identity does match. Windows only: that is where a tree kill by PID
    /// follows the identity gate, and where a creation-time token is read.
    #[cfg(windows)]
    #[test]
    fn terminate_recorded_pid_leaves_a_recycled_pid_alone_and_stops_the_verified_one() {
        use std::process::{Command, Stdio};

        let mut child = Command::new("ping")
            .args(["-n", "120", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn test child");
        let pid = child.id();
        let real = process_start_ticks(pid).expect("a live process has a creation time");
        assert_eq!(
            process_start_ticks(pid),
            Some(real),
            "the creation time of one process does not change"
        );
        assert_ne!(
            Some(real),
            process_start_ticks(std::process::id()),
            "two processes do not share one"
        );

        let outcome =
            terminate_recorded_pid(&ProcessIdentity::new(pid, Some(real.wrapping_add(1))));
        assert_eq!(outcome, TerminationOutcome::IdentityMismatch);
        assert!(
            child.try_wait().expect("check the bystander").is_none(),
            "a PID whose recorded creation time differs is a stranger and must keep running"
        );

        let outcome = terminate_recorded_pid(&ProcessIdentity::capture(pid));
        assert!(outcome.stopped(), "{outcome:?}");
        let _ = child.wait();
        assert!(!crate::process_is_running(pid));
    }

    /// A record with no token (written before Windows recorded one, and the shape
    /// `service-stop-07` plants) cannot be refuted and still gets the whole stop:
    /// the process ends. Windows only, where a real process is stopped by PID.
    #[cfg(windows)]
    #[test]
    fn terminate_recorded_pid_stops_a_process_recorded_without_a_token() {
        use std::process::{Command, Stdio};

        let mut child = Command::new("ping")
            .args(["-n", "120", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn test child");
        let pid = child.id();

        let outcome = terminate_recorded_pid(&ProcessIdentity::new(pid, None));

        assert!(outcome.stopped(), "{outcome:?}");
        assert_eq!(outcome.as_str(), "forced");
        assert!(
            child.wait().is_ok() && !crate::process_is_running(pid),
            "the process recorded without a token must be gone"
        );
    }
}
