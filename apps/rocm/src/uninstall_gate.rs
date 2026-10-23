// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The stop-before-remove gate behind `rocm uninstall` (EAI-8014).
//!
//! Owns the types describing what a stop pass managed to do and the functions
//! that stop the background helper and every live managed service, classify
//! what is still answering afterwards, and decide whether removal may proceed.
//! The command handler in `uninstall.rs` calls in; the shared
//! `UninstallOptions`/`UninstallPlan` types stay in the crate root.

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, AutomationRuntimeState, ManagedServiceRecord};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use crate::chat::loopback_tcp_port_is_reachable;
use crate::{
    MANAGED_STOP_GRACE, endpoint_keys, host_port_resolves, load_managed_services,
    managed_service_is_live, managed_service_record_is_in_use,
    stop_internal_managed_service_confirmed,
};

/// A managed service uninstall could not confirm stopped, and why.
///
/// The reason is carried rather than discarded so the abort message says what
/// went wrong — an operator facing "could not stop svc-x" with no cause has
/// nothing to act on.
#[derive(Debug, Clone)]
pub(crate) struct FailedManagedServiceStop {
    pub(crate) service_id: String,
    pub(crate) reason: String,
    pub(crate) remedy: StopFailureRemedy,
}

/// What will actually clear a failed stop.
///
/// Not cosmetic. This gate refuses to remove anything while a stop is
/// unconfirmed, so the advice it prints is the operator's only way out, and
/// advice that cannot work turns the refusal into a dead end: `rocm services
/// stop` re-reads the same unparseable JSON and fails the same way, so a record
/// that will not parse would abort every retry identically. Each failure class
/// carries the remedy that can actually clear it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopFailureRemedy {
    /// The record parses and its processes are still there, so `rocm services
    /// stop` can act on it.
    StopTheService,
    /// The recorded processes are gone but the endpoint still serves — an engine
    /// grandchild outlived its supervisor. `rocm services stop` has nothing left
    /// to kill, so the process holding the port has to be found and stopped.
    StopWhatHoldsThePort,
    /// A record already stopped before this run, whose old port now answers with
    /// an authentication refusal. That is not evidence of a survivor of ours (a
    /// keyed server of someone else's looks identical), and the record is what
    /// keeps tripping the check, so removing it is the way out if it isn't ours.
    RemoveTheStoppedRecord,
    /// The record does not parse, so no `rocm` command can act on it — the file
    /// itself has to be repaired or removed.
    RepairTheRecord,
    /// The background helper is still alive, so it can restart what was just
    /// stopped. It has to go before anything is removed.
    StopTheDaemon,
    /// A live record carries no start time for a pid that is still alive, so the
    /// pid cannot be proven to be the server and is left alone.
    ConfirmTheServicePid,
    /// A live pid is recorded as the helper but cannot be proven to be `rocmd`
    /// (a pre-upgrade state file, or a reboot since). Killing it would risk an
    /// unrelated process, so the operator has to decide, and the state file is the
    /// thing to delete if it is not the helper.
    ConfirmTheDaemonPid,
    /// The helper's runtime-state file does not parse, so no pid was ever
    /// recovered from it — "kill that pid" is advice nobody can act on. The file
    /// itself has to be repaired or deleted, so the remedy names it.
    RepairTheDaemonState,
}

impl StopFailureRemedy {
    /// The recovery advice for this class, naming `ids` where the advice is
    /// useless without them.
    ///
    /// Exhaustive on purpose, like [`StoppedRecordVerdict::blocks`]: a new
    /// variant must get advice, because the gate's whole contract is that its
    /// advice is the operator's only way out.
    pub(crate) fn advice(self, ids: &[String]) -> String {
        match self {
            Self::StopTheService => "Stop them with `rocm services stop <id> --yes`, then re-run \
                                     uninstall. A server started by another user, or one wedged \
                                     in the kernel, needs elevated privileges or a manual kill \
                                     first."
                .to_owned(),
            Self::StopWhatHoldsThePort => format!(
                "Every process recorded for {} is gone, yet the endpoint still answers; the \
                 engine outlived its supervisor, so `rocm services stop` has nothing left to \
                 kill. Find what holds that port (`ss -ltnp` on Linux, `Get-NetTCPConnection \
                 -LocalPort <port>` on Windows), stop it, then re-run uninstall.",
                ids.join(", ")
            ),
            Self::ConfirmTheServicePid => format!(
                "No start time was recorded for a live pid of each service listed, so it cannot \
                 be proven to still be the server (the pid may have been reused after a reboot) \
                 and it was left alone: {}. Check what those pids are. If one is your server, \
                 stop it yourself (`kill <pid>` on Linux, `Stop-Process -Id <pid>` on Windows). \
                 If it is something else, delete the service's record file under the data \
                 directory's `services` folder by hand. Then re-run uninstall.",
                ids.join(", ")
            ),
            Self::StopTheDaemon => "The background helper restarts managed services on its own, \
                                    so it has to be stopped before uninstall can safely remove \
                                    anything: kill that pid (`kill <pid>` on Linux, `Stop-Process -Id <pid> -Force` \
                                    on Windows), then re-run uninstall."
                .to_owned(),
            Self::RemoveTheStoppedRecord => format!(
                "An authenticated server answers on the old port of a record that was already \
                 stopped. If that server is not one of yours, delete the stale record with \
                 `rocm services remove <id> --yes` and re-run uninstall; if it is, stop it first: \
                 {}.",
                ids.join(", ")
            ),
            Self::ConfirmTheDaemonPid => format!(
                "The recorded pid could not be proven to be the background helper, so it was left \
                 alone. Check what that pid is ({}): if it is `rocmd`, stop it; if it is something \
                 else (the state file may predate a reboot), delete the runtime state file named \
                 there, then re-run uninstall.",
                ids.join(", ")
            ),
            Self::RepairTheDaemonState => format!(
                "The background helper's runtime state does not parse, so no pid could be read \
                 from it and `rocm` cannot tell whether the helper is running. Check for a live \
                 `rocmd` process and stop it, then repair or delete the file and re-run \
                 uninstall: {}.",
                ids.join(", ")
            ),
            Self::RepairTheRecord => format!(
                "No `rocm` command can act on an unparseable record, so these have to be handled \
                 on disk: check whether the server each one describes is still running (`rocm \
                 services list` skips them), stop it, then repair or delete the file and re-run \
                 uninstall: {}.",
                ids.join(", ")
            ),
        }
    }

    /// Where this class sits in the printed advice.
    ///
    /// Also exhaustive, and for a second reason beyond ordering: without it, a
    /// new variant could compile an `advice()` arm and still never be printed,
    /// because nothing would have added it to the list of classes to walk.
    /// Ranking every variant means the set that gets advice is derived from the
    /// failures themselves rather than from a list somebody has to remember to
    /// extend. Most actionable first; the two "repair a file by hand" classes
    /// last.
    pub(crate) const fn advice_rank(self) -> u8 {
        match self {
            Self::StopTheService => 0,
            Self::StopWhatHoldsThePort => 1,
            Self::RemoveTheStoppedRecord => 2,
            Self::ConfirmTheServicePid => 3,
            Self::StopTheDaemon => 4,
            Self::ConfirmTheDaemonPid => 5,
            Self::RepairTheDaemonState => 6,
            Self::RepairTheRecord => 7,
        }
    }
}

/// The processes a stop of `record` would signal, each paired with the start
/// time recorded for it, de-duplicated on pid and preferring a known token.
///
/// Shared by the stop itself and by this gate's pre-check, so the two cannot
/// disagree about which pid still needs a start time.
pub(crate) fn recorded_pid_identities(record: &ManagedServiceRecord) -> Vec<(u32, Option<u64>)> {
    let mut entries: Vec<(u32, Option<u64>)> = Vec::new();
    for (pid, ticks) in [
        (Some(record.supervisor_pid), record.supervisor_start_ticks),
        (record.engine_pid, record.engine_start_ticks),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0 && *pid != std::process::id()) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _)| *seen == pid) {
            if existing.1.is_none() {
                existing.1 = ticks;
            }
        } else {
            entries.push((pid, ticks));
        }
    }
    entries
}

/// The live pids of `record` that carry no start time, and so cannot be told
/// apart from a reused pid. The stop refuses a record that has any, rather than
/// force-killing a tree it cannot prove is ours; the plan reads the same answer
/// so it never promises a stop this refuses.
pub(crate) fn unprovable_live_pids(record: &ManagedServiceRecord) -> Vec<u32> {
    recorded_pid_identities(record)
        .into_iter()
        .filter(|(pid, ticks)| {
            record_predates_start_ticks(*ticks) && rocm_core::process_is_running(*pid)
        })
        .map(|(pid, _)| pid)
        .collect()
}

/// How long the gate waits for a still-listening endpoint to say what it serves.
///
/// Only reached when something already answered a TCP connect, and only for
/// records that were already stopped, so it costs nothing on the normal path.
pub(crate) const ENDPOINT_IDENTITY_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// What [`stop_managed_services_before_uninstall`] managed to do, so the caller
/// can report the services it stopped and refuse to proceed while any is still
/// alive.
#[derive(Debug, Default)]
pub(crate) struct ManagedServiceStopReport {
    /// Services confirmed stopped (every recorded process observed gone).
    pub(crate) stopped: Vec<String>,
    /// Services that could not be confirmed stopped — a still-serving endpoint,
    /// an unverifiable process, or a record too corrupt to locate one. Uninstall
    /// must abort rather than remove the tooling that stops them.
    pub(crate) failed: Vec<FailedManagedServiceStop>,
    /// Every place this pass decided to proceed without proving the port was
    /// free, for the caller to put on stderr.
    ///
    /// These are values rather than `eprintln!`s so a test can assert on them.
    /// While they were printed in place, the disclosure that keeps each
    /// fail-open a tradeoff rather than a silent removal was pinned by nothing:
    /// emptying a warning body left every test green while changing what a
    /// destructive command tells its operator. Returning them makes the
    /// disclosure part of this function's answer, and `stderr` the caller's
    /// business.
    pub(crate) warnings: Vec<String>,
    /// One relaunch command per entry of `stopped`, built from that service's own
    /// record by the shared `rocm_engine_protocol::relaunch_command`, so the
    /// abort message names the flags each one needs instead of a generic line.
    pub(crate) relaunch: Vec<String>,
    /// The background helper (`rocmd`) was stopped by this pass. Kept apart from
    /// `stopped` because it is not a service: it has no endpoint key, and the
    /// consequence of its being down is that nothing supervises or recovers
    /// services until it is restarted.
    pub(crate) helper_stopped: bool,
}

/// The kill breadth the stop applies to the background helper.
///
/// A constant so a test can see the scope the production stop hands the
/// termination, rather than only that some termination was requested.
pub(crate) const HELPER_KILL_SCOPE: rocm_core::KillScope = rocm_core::KillScope::Tree;

/// How much of a process tree the stop signals, in words, for the plan and the
/// docs: only Linux enumerates descendants (`process_tree_pids` returns just the
/// root on Windows), so the claim cannot be the same on both.
pub(crate) const fn tree_scope_words() -> &'static str {
    if rocm_core::runtime_is_windows() {
        "only the recorded process (descendants are not enumerated on Windows)"
    } else {
        "its whole process tree"
    }
}

/// What the gate decides about the background helper, read without touching it.
///
/// The one decision behind the plan's note, the pre-flight and the stop itself,
/// so the plan cannot promise a stop that the stop then refuses.
pub(crate) enum HelperDecision {
    /// No state file, `running` false, a free pid, or a pid that belongs to
    /// someone else.
    NothingToStop,
    /// The state file exists but does not read or parse.
    UnreadableState(String),
    /// A live pid is recorded as the helper but cannot be proven to be it.
    Unverifiable { pid: u32 },
    /// Verified as the helper: stop it.
    Stop {
        pid: u32,
        identity: rocm_core::ProcessIdentity,
    },
}

/// Decide [`HelperDecision`] from the state file and one reading of the pid.
pub(crate) fn helper_decision(paths: &AppPaths) -> HelperDecision {
    // Three outcomes, not two. `load` returns `Ok(None)` only when there is no
    // state file at all; a permission error or a half-written file returns
    // `Err`, and discarding that would skip the daemon stop silently and let
    // uninstall delete the tooling while a live `rocmd` respawns what the
    // service pass just stopped.
    let state = match AutomationRuntimeState::load(paths) {
        Ok(Some(state)) => state,
        Ok(None) => return HelperDecision::NothingToStop,
        Err(error) => return HelperDecision::UnreadableState(format!("{error:#}")),
    };
    // `running` is the same inactive contract `background_helper_already_running`
    // spawns on. Only the clean-shutdown path writes `running: false`, so a false
    // flag means the recorded pid belongs to a daemon that already exited, and
    // anything alive under it now inherited the number.
    if !state.running || state.daemon_pid == 0 || state.daemon_pid == std::process::id() {
        return HelperDecision::NothingToStop;
    }
    if !rocm_core::process_is_running(state.daemon_pid) {
        return HelperDecision::NothingToStop;
    }
    // Kill nothing this cannot identify. `terminate_verified` with `force` is a
    // SIGKILL across the whole process tree, and `runtime-state.json` outlives a
    // crash, OOM-kill or reboot with `running` still true, so the recorded pid
    // can belong to an unrelated process by the time uninstall runs. An absent
    // start-time is unverifiable on its own (see `record_predates_start_ticks`),
    // and one reading of the daemon's start-time serves the whole decision, so
    // a pid that flickers cannot clear one check and fail another.
    let identity = rocm_core::ProcessIdentity::new(state.daemon_pid, state.daemon_start_ticks);
    let observed_start_ticks = rocm_core::process_start_ticks(state.daemon_pid);
    match daemon_identity_outcome(
        rocm_core::identity_state_with_observed(&identity, observed_start_ticks),
        record_predates_start_ticks(state.daemon_start_ticks),
    ) {
        DaemonIdentityOutcome::NothingOfOurs => HelperDecision::NothingToStop,
        DaemonIdentityOutcome::Unverifiable => HelperDecision::Unverifiable {
            pid: state.daemon_pid,
        },
        DaemonIdentityOutcome::Stop => HelperDecision::Stop {
            pid: state.daemon_pid,
            identity,
        },
    }
}

/// The refusal a [`HelperDecision`] amounts to, if any.
fn helper_refusal(paths: &AppPaths, decision: &HelperDecision) -> Option<FailedManagedServiceStop> {
    match decision {
        // Name the file, not a pid: nothing parsed, so no pid was ever
        // recovered and "kill that pid" would be advice nobody can follow.
        HelperDecision::UnreadableState(error) => Some(FailedManagedServiceStop {
            service_id: format!("rocmd ({})", paths.automation_state_path().display()),
            reason: format!(
                "the background helper's runtime state could not be read, so it cannot be \
                 confirmed stopped: {error}"
            ),
            remedy: StopFailureRemedy::RepairTheDaemonState,
        }),
        HelperDecision::Unverifiable { pid } => Some(daemon_identity_unverified(
            *pid,
            &paths.automation_state_path(),
        )),
        HelperDecision::NothingToStop | HelperDecision::Stop { .. } => None,
    }
}

/// What the plan should tell the operator about the background helper, derived
/// from the same decision the stop pass acts on, so the line cannot promise a
/// stop that pass then refuses.
///
/// `None` when there is nothing to say: nothing to stop, or a refusal the plan's
/// consolidated refusal line already names. `run_refuses` is whether the run
/// aborts before stopping anything: a helper that is verified but sits in such a
/// run is left running, and saying "will be stopped" would be false.
/// `refusal_line_present` is whether the pre-flight ran and its refusal line is
/// in the plan; it is absent when the service records could not be read at all,
/// and then a helper that cannot be verified has to be named here.
pub(crate) fn background_helper_plan_note(
    paths: &AppPaths,
    removes_recovery_tooling: bool,
    run_refuses: bool,
    refusal_line_present: bool,
) -> Option<String> {
    let state_file = paths.automation_state_path();
    let decision = helper_decision(paths);
    match (&decision, removes_recovery_tooling) {
        (HelperDecision::NothingToStop, _) => None,
        // Named by the refusal line, which comes from the same decision.
        (HelperDecision::UnreadableState(_) | HelperDecision::Unverifiable { .. }, true)
            if refusal_line_present =>
        {
            None
        }
        (HelperDecision::UnreadableState(_) | HelperDecision::Unverifiable { .. }, true) => {
            helper_refusal(paths, &decision).map(|failure| {
                format!(
                    "{} ({}); uninstall will refuse to remove anything until this is resolved",
                    failure.service_id, failure.reason
                )
            })
        }
        (HelperDecision::UnreadableState(error), false) => Some(format!(
            "could not read the background helper's runtime state {}: {error}. This run keeps \
             the tooling, so it does not check it",
            state_file.display()
        )),
        (HelperDecision::Unverifiable { pid } | HelperDecision::Stop { pid, .. }, false) => {
            Some(format!(
                "the background helper (rocmd, pid {pid}) is running; this removal keeps the \
                 tooling, so it is left running"
            ))
        }
        (HelperDecision::Stop { pid, .. }, true) if run_refuses => Some(format!(
            "the background helper (rocmd, pid {pid}) is running; uninstall will refuse (see \
             above) before stopping anything, so it is left running"
        )),
        (HelperDecision::Stop { pid, .. }, true) => Some(format!(
            "the background helper (rocmd, pid {pid}) is running; it will be stopped before \
             removal, and the kill covers {}",
            tree_scope_words()
        )),
    }
}

/// The failure recorded when the background helper is live but cannot be proven
/// to be `rocmd`, so it was deliberately left alone.
pub(crate) fn daemon_identity_unverified(
    daemon_pid: u32,
    state_file: &std::path::Path,
) -> FailedManagedServiceStop {
    FailedManagedServiceStop {
        service_id: format!(
            "rocmd (pid {daemon_pid}; state file {})",
            state_file.display()
        ),
        reason:
            "the background helper's identity could not be verified (no start time was recorded \
                 for it, or its start time cannot be read now), so it was left running rather \
                 than risk signalling an unrelated process that inherited its pid"
                .to_owned(),
        remedy: StopFailureRemedy::ConfirmTheDaemonPid,
    }
}

/// What the gate does with the daemon's recorded pid once its identity has been
/// read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DaemonIdentityOutcome {
    /// The pid is free or belongs to someone else: nothing of ours to stop.
    NothingOfOurs,
    /// Could not be proven ours, so it is neither killed nor ignored: abort.
    Unverifiable,
    /// Verified as the daemon: stop it.
    Stop,
}

/// Map an identity reading to the gate's action. Pure so every input is pinned
/// by a test: folding `Indeterminate` into `NothingOfOurs` would remove the
/// tooling while a live `rocmd` keeps respawning services.
pub(crate) const fn daemon_identity_outcome(
    state: rocm_core::IdentityState,
    record_predates_start_ticks: bool,
) -> DaemonIdentityOutcome {
    match state {
        rocm_core::IdentityState::Gone | rocm_core::IdentityState::Recycled => {
            DaemonIdentityOutcome::NothingOfOurs
        }
        rocm_core::IdentityState::Indeterminate => DaemonIdentityOutcome::Unverifiable,
        rocm_core::IdentityState::Matches if record_predates_start_ticks => {
            DaemonIdentityOutcome::Unverifiable
        }
        rocm_core::IdentityState::Matches => DaemonIdentityOutcome::Stop,
    }
}

/// Stop the background helper before uninstall stops the services it supervises.
///
/// `rocmd` recovers managed services: a `ready`/`running` record whose endpoint
/// is unreachable is treated as recoverable and respawned
/// (`endpoint_status_unreachable`). That is precisely the state the stop pass
/// creates — the engine is killed, and the record still says `ready` until
/// `stop_internal_managed_service` writes it back. A daemon polling in that
/// window brings the server straight back, after which uninstall would delete
/// the binaries and every service record while a brand-new engine holds the GPU.
/// Stopping the supervisor first closes the window instead of racing it.
///
/// A daemon that cannot be confirmed stopped is a blocking failure for the same
/// reason a service is: it can resurrect a server after the tooling is gone.
///
/// It is never killed on the recorded pid alone. `runtime-state.json` outlives a
/// crash, OOM-kill or reboot with `running` still true, so that pid can name an
/// unrelated process — and this call site signals a whole tree with `force`.
/// Only a pid whose recorded start-time still matches is signalled; a recycled
/// one means the daemon is already gone (nothing to stop), and one that can be
/// neither confirmed nor refuted is left alone and reported as a failure, which
/// aborts the uninstall with the tooling intact. Being told to kill a pid is
/// recoverable; having an unrelated process tree killed is not.
///
/// A record whose `running` is false is not signalled at all. Only rocmd's clean
/// shutdown writes that flag, on its way out, so the pid in such a record names
/// a process that has already exited and anything live under that number
/// inherited it. This is the same inactive contract `background_helper_already_running`
/// spawns on.
///
/// Linux (`/proc`) and Windows (`GetProcessTimes` creation time) can record and
/// compare a start-time, so a recycled pid is told apart there.
///
/// One residual gap, and the list above is otherwise the complete set of
/// inputs: a *missing* `runtime-state.json` is taken at face value as "no
/// daemon". Deleting that file by hand while `rocmd` is live therefore skips the
/// daemon stop silently. This is deliberate — a missing file is the ordinary
/// never-started case, and there is no pid to verify or signal without it — but
/// it does mean the gate is only as good as the state file. An unreadable one is
/// the case that aborts; an absent one is the case that proceeds.
///
/// The verified termination is a parameter, so a test can make the stop come
/// back unconfirmed and see the [`HELPER_KILL_SCOPE`] the production stop asks
/// for; production passes `terminate_verified` with `force`.
pub(crate) fn stop_background_helper_with(
    paths: &AppPaths,
    report: &mut ManagedServiceStopReport,
    terminate: impl FnOnce(
        &rocm_core::ProcessIdentity,
        rocm_core::KillScope,
    ) -> rocm_core::TerminationOutcome,
) {
    let decision = helper_decision(paths);
    if let Some(failure) = helper_refusal(paths, &decision) {
        report.failed.push(failure);
        return;
    }
    let HelperDecision::Stop { pid, identity } = decision else {
        return;
    };
    println!("stopping the background helper (rocmd)");
    let outcome = terminate(&identity, HELPER_KILL_SCOPE);
    if outcome.stopped() {
        report.helper_stopped = true;
    } else if matches!(outcome, rocm_core::TerminationOutcome::Unverified) {
        report.failed.push(daemon_identity_unverified(
            pid,
            &paths.automation_state_path(),
        ));
    } else {
        report.failed.push(FailedManagedServiceStop {
            service_id: format!("rocmd (pid {pid})"),
            reason: "the background helper was signalled but could not be confirmed stopped (the \
                     servers it supervised may already be down), and it restarts managed \
                     services whose endpoint stops answering"
                .to_owned(),
            remedy: StopFailureRemedy::StopTheDaemon,
        });
    }
}

/// Whether the stop pass has to deal with `record`: it reads as running, or its
/// status says something other than `stopped` while a recorded process is still
/// alive.
///
/// The status string alone is not enough: `refresh_from_engine_state` adopts
/// `failed` from the engine's own state file, and the liveness refresh returns
/// early for a non-live status without consulting the pids, so a server whose
/// engine reported failure while its process is still up reads as removable.
/// `services remove` guards with [`managed_service_record_is_in_use`] for the
/// same reason. A record that reads `stopped` is left to the port probe: that
/// status is written only once the recorded processes are confirmed gone, so a
/// live pid under it is a reused number, and refusing on it would block every
/// uninstall that finds an old record.
pub(crate) fn record_needs_stopping(record: &ManagedServiceRecord) -> bool {
    managed_service_is_live(record)
        || (record.status != "stopped" && managed_service_record_is_in_use(record))
}

/// The records a stop pass partitions after the live ones are set aside: those
/// already stopped, and those stopped but sharing an endpoint with a live record.
struct RecordPartition<'a> {
    already_stopped: Vec<&'a ManagedServiceRecord>,
    deferred_partners: Vec<(&'a ManagedServiceRecord, Vec<String>)>,
}

fn partition_records(records: &[ManagedServiceRecord]) -> RecordPartition<'_> {
    // A record already stopped before this run needs no stop to be judged, and a
    // survivor from the previous run reads exactly like that: probe those first.
    //
    // Except a stopped record that shares its host:port with a live one: serving
    // the same model again after a stop leaves exactly that pair, and the live
    // record's own server answers there. That answer says nothing about the
    // stopped record until the live one has been stopped, so it is probed after.
    let live_endpoints: Vec<(String, Vec<String>, u16)> = records
        .iter()
        .filter(|record| record_needs_stopping(record))
        .map(|record| {
            (
                record.service_id.clone(),
                probe_hosts(&record.host),
                record.port,
            )
        })
        .collect();
    // Overlap, not equality: `0.0.0.0`, `localhost` and `127.0.0.1` can all name
    // the one listener, so any shared probe address on the same port counts.
    let live_partners = |record: &ManagedServiceRecord| -> Vec<String> {
        let mine = probe_hosts(&record.host);
        live_endpoints
            .iter()
            .filter(|(_, hosts, port)| {
                *port == record.port
                    && mine
                        .iter()
                        .any(|a| hosts.iter().any(|b| hosts_name_one_listener(a, b)))
            })
            .map(|(id, _, _)| id.clone())
            .collect()
    };
    RecordPartition {
        already_stopped: records
            .iter()
            .filter(|record| !record_needs_stopping(record) && live_partners(record).is_empty())
            .collect(),
        deferred_partners: records
            .iter()
            .filter(|record| !record_needs_stopping(record))
            .map(|record| (record, live_partners(record)))
            .filter(|(_, partners)| !partners.is_empty())
            .collect(),
    }
}

/// Everything the stop pass would refuse before it signals or removes anything,
/// decided without touching a process, and the records it read to decide.
///
/// This is the one place those refusals live. The stop pass runs it first and
/// returns on any failure, so a run that is going to abort has stopped nothing
/// on the way (each confirmed stop drops that service's endpoint key, which
/// cannot be re-minted); the plan runs the same function, so the line the
/// operator confirms against describes the run that will happen. Order matters
/// and is the order of cost: an unparseable manifest, then a live pid that
/// cannot be proven ours, then a stopped record whose port still answers as its
/// own model, then a background helper that cannot be identified. Each group
/// returns before the next, as the stop pass always did.
///
/// Fail-closed on discovery too: if the services directory exists but cannot be
/// enumerated, the error propagates so uninstall aborts rather than deleting the
/// tooling while blind to what it manages.
fn preflight_with_records(
    paths: &AppPaths,
) -> Result<(ManagedServiceStopReport, Vec<ManagedServiceRecord>)> {
    let mut report = ManagedServiceStopReport::default();
    // One reading of the helper serves the whole pre-flight, and its refusal is
    // appended to whatever else refuses, so the plan can name it on a run that
    // already refuses for another reason instead of the operator learning of it
    // only on the retry.
    let helper_failure = helper_refusal(paths, &helper_decision(paths));
    let with_helper = |mut report: ManagedServiceStopReport| {
        report.failed.extend(helper_failure.clone());
        report
    };
    // A manifest that exists but does not parse can describe a live,
    // GPU-holding server; `load_managed_services` skips it silently.
    for manifest in unreadable_service_manifests(paths)? {
        report.failed.push(FailedManagedServiceStop {
            // The full path, not the file name: the only remedy is to act on the
            // file, so the message has to say which file.
            service_id: manifest.display().to_string(),
            reason:
                "service record could not be parsed, so its server cannot be located or stopped"
                    .to_owned(),
            remedy: StopFailureRemedy::RepairTheRecord,
        });
    }
    if !report.failed.is_empty() {
        return Ok((with_helper(report), Vec::new()));
    }
    let records = load_managed_services(paths)?;
    // A live record with no recorded start time cannot prove its pid is still the
    // server (the pid may have been reused after a reboot), and the stop below
    // force-kills a process tree. Refuse before anything is stopped, as the
    // daemon path does for a tick-less state file.
    for record in records
        .iter()
        .filter(|record| record_needs_stopping(record))
    {
        let unprovable = unprovable_live_pids(record);
        if !unprovable.is_empty() {
            report.failed.push(FailedManagedServiceStop {
                service_id: record.service_id.clone(),
                reason: format!(
                    "no start time was recorded for live pid(s) {unprovable:?}, so they cannot be \
                     told apart from a reused pid"
                ),
                remedy: StopFailureRemedy::ConfirmTheServicePid,
            });
        }
    }
    if !report.failed.is_empty() {
        return Ok((with_helper(report), records));
    }
    let partition = partition_records(&records);
    probe_records_for_survivors(paths, &partition.already_stopped, &[], &mut report);
    Ok((with_helper(report), records))
}

/// [`preflight_with_records`] for the plan: the refusals a run would hit and the
/// fail-open disclosures it would make, with nothing stopped.
pub(crate) fn uninstall_preflight(paths: &AppPaths) -> Result<ManagedServiceStopReport> {
    preflight_with_records(paths).map(|(report, _)| report)
}

/// The plan's one line for a run that will refuse, built from the failures
/// [`uninstall_preflight`] found. It says what the run does not do as well as
/// what blocks it: the pre-flight returns before anything is stopped.
pub(crate) fn preflight_refusal_line(report: &ManagedServiceStopReport) -> Option<String> {
    if report.failed.is_empty() {
        return None;
    }
    let detail = report
        .failed
        .iter()
        .map(|failure| format!("{} ({})", failure.service_id, failure.reason))
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!(
        "uninstall will refuse to remove anything, and will stop nothing, until this is \
         resolved: {detail}"
    ))
}

/// Stop every managed service that is in use before uninstall removes the
/// binaries and service records needed to stop them.
///
/// The defect this closes (EAI-8014): uninstall reported success while a
/// publicly-bound, GPU-holding managed server kept serving, and deleted the
/// `rocm`/`rocmd` binaries and service manifests — so the supported
/// `rocm services stop` path was gone and only a manual PID kill remained.
///
/// A service is only counted stopped when the stop confirms every recorded
/// process gone (the boolean [`stop_internal_managed_service_confirmed`]
/// returns — the report's `status` string cannot say it, because a record that
/// already read `stopped` keeps that status through a stop that confirmed
/// nothing); anything else lands in `failed` so the caller aborts and keeps the
/// tooling.
pub(crate) fn stop_managed_services_before_uninstall(
    paths: &AppPaths,
) -> Result<ManagedServiceStopReport> {
    stop_managed_services_confirmed_with(paths, stop_internal_managed_service_confirmed)
}

/// [`stop_managed_services_before_uninstall`] with a stop that reports only a
/// `status`, which counts as confirmed when it reads `stopped`. Unix-test-only: the
/// production stop decides on the confirmed flag, and the tests that care drive
/// [`stop_managed_services_confirmed_with`] directly.
#[cfg(all(test, unix))]
pub(crate) fn stop_managed_services_with(
    paths: &AppPaths,
    stop_service: impl Fn(&AppPaths, &str) -> Result<serde_json::Value>,
) -> Result<ManagedServiceStopReport> {
    stop_managed_services_confirmed_with(paths, |paths, service_id| {
        let result = stop_service(paths, service_id)?;
        let confirmed = result.get("status").and_then(serde_json::Value::as_str) == Some("stopped");
        Ok((result, confirmed))
    })
}

/// [`stop_managed_services_before_uninstall`] with the per-service stop
/// injectable, so a test can make it fail or fail to confirm — which no real
/// process reliably does.
pub(crate) fn stop_managed_services_confirmed_with(
    paths: &AppPaths,
    stop_service: impl Fn(&AppPaths, &str) -> Result<(serde_json::Value, bool)>,
) -> Result<ManagedServiceStopReport> {
    stop_managed_services_with_helper_stop(paths, stop_service, |identity, scope| {
        rocm_core::terminate_verified(identity, scope, MANAGED_STOP_GRACE, true)
    })
}

/// [`stop_managed_services_confirmed_with`] with the helper's termination
/// injectable too, so a test can make the helper stop come back unconfirmed
/// and see that no service is stopped after it.
pub(crate) fn stop_managed_services_with_helper_stop(
    paths: &AppPaths,
    stop_service: impl Fn(&AppPaths, &str) -> Result<(serde_json::Value, bool)>,
    terminate_helper: impl FnOnce(
        &rocm_core::ProcessIdentity,
        rocm_core::KillScope,
    ) -> rocm_core::TerminationOutcome,
) -> Result<ManagedServiceStopReport> {
    // The refusals are decided before anything is stopped, by the function the
    // plan also calls. Past this point a stop can still fail, the helper can
    // still not die, and a survivor can still answer after the stops, so the
    // abort message names what the run already did.
    let (mut report, records) = preflight_with_records(paths)?;
    if !report.failed.is_empty() {
        return Ok(report);
    }
    let partition = partition_records(&records);
    stop_background_helper_with(paths, &mut report, terminate_helper);
    if !report.failed.is_empty() {
        return Ok(report);
    }
    let mut attempted: Vec<&ManagedServiceRecord> = Vec::new();
    for record in &records {
        if !record_needs_stopping(record) {
            continue;
        }
        attempted.push(record);
        // Each stop waits out a bounded grace per recorded process (and the
        // engine's own stop before that), so name the service first: without
        // this, an uninstall with a live server reads as a hang.
        println!("stopping managed service {}", record.service_id);
        match stop_service(paths, &record.service_id) {
            Ok((_, true)) => {
                report.stopped.push(record.service_id.clone());
                report.relaunch.push(format!(
                    "{}: `{}`",
                    record.service_id,
                    rocm_engine_protocol::relaunch_command(&record.host, record.requires_api_key)
                ));
            }
            Ok((result, false)) => {
                let status = result
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                report.failed.push(FailedManagedServiceStop {
                    service_id: record.service_id.clone(),
                    reason: if status == "stopped" {
                        "its record reads \"stopped\", but the stop could not confirm every \
                         recorded process gone"
                            .to_owned()
                    } else {
                        format!("still \"{status}\" after the stop attempt")
                    },
                    remedy: StopFailureRemedy::StopTheService,
                });
            }
            Err(error) => report.failed.push(FailedManagedServiceStop {
                service_id: record.service_id.clone(),
                reason: format!("{error:#}"),
                remedy: StopFailureRemedy::StopTheService,
            }),
        }
    }
    // Ground the PID bookkeeping in what is actually being served. Confirming a
    // stop from recorded PIDs alone is thin: on Windows the kill scope is the
    // recorded process only, so an engine grandchild can outlive it and keep the
    // port and the GPU while the record reads "stopped".
    //
    // Every record is probed, not just the ones this pass stopped. A stop
    // persists `status = "stopped"` before this loop runs, so the record that
    // failed the gate reads as not-live on the very next run — probing only this
    // pass's own work would let the retry the abort message asks for sail
    // through and remove the tooling while the survivor keeps serving. That is
    // the defect this gate exists to prevent, reached by following its own
    // instructions.
    //
    // What differs is the evidence required, because a stopped record keeps its
    // manifest and its old port until `rocm services remove` or `prune` deletes
    // it, so an unrelated process that later binds that port must not brick
    // uninstall with no way out:
    //
    //   * stopped by this pass — any listener fails the gate. We killed the
    //     recorded processes seconds ago; something answering there is the
    //     survivor.
    //   * already stopped before this run — a listener alone proves nothing, so
    //     it has to identify itself as this record's own model before it counts.
    //     An unrelated service on a recycled port does not, and is not blocked.
    let stopped_this_pass: Vec<&ManagedServiceRecord> = attempted.clone();
    probe_records_for_survivors(paths, &stopped_this_pass, &attempted, &mut report);
    // A deferred record whose live partner failed to stop is already explained by
    // the partner's own failure: the listener there IS the partner, so judging the
    // stopped record too would name one survivor twice, under the wrong remedy.
    let deferred: Vec<&ManagedServiceRecord> = partition
        .deferred_partners
        .iter()
        .filter(|(_, partners)| {
            !partners
                .iter()
                .any(|id| report.failed.iter().any(|f| &f.service_id == id))
        })
        .map(|(record, _)| *record)
        .collect();
    probe_records_for_survivors(paths, &deferred, &attempted, &mut report);
    Ok(report)
}

/// Probe the endpoints of `to_probe` for a survivor, recording any that should
/// block the removal. `attempted` is the set this pass stopped; for those any
/// listener blocks, for the rest it has to identify itself as the record's own.
fn probe_records_for_survivors(
    paths: &AppPaths,
    to_probe: &[&ManagedServiceRecord],
    attempted: &[&ManagedServiceRecord],
    report: &mut ManagedServiceStopReport,
) {
    for &record in to_probe {
        if report
            .failed
            .iter()
            .any(|failure| failure.service_id == record.service_id)
        {
            continue;
        }
        // A wildcard record yields both loopback families; whichever answers is
        // the one the identity probe below has to talk to, so keep it rather
        // than re-deriving a single host and guessing the family wrong.
        let candidates = probe_hosts(&record.host);
        let reachable_hosts: Vec<String> = candidates
            .iter()
            .filter(|candidate| loopback_tcp_port_is_reachable(candidate, record.port))
            .cloned()
            .collect();
        if reachable_hosts.is_empty() {
            // Not reaching the port is the normal, silent case: nothing is
            // listening, which is what a stopped service looks like. But
            // `loopback_tcp_port_is_reachable` returns the same `false` when the
            // address does not resolve at all, and those are opposite
            // situations. A refused connect is evidence the port is free; a name
            // that no longer resolves is evidence of nothing, and skipping on it
            // means proceeding to delete the tooling without ever having asked
            // whether a server is up. That is a fail-open, and it is disclosed.
            if !candidates
                .iter()
                .any(|candidate| host_port_resolves(candidate, record.port))
            {
                report.warnings.push(format!(
                    "{} is recorded for service {}, but that name does not resolve here, so \
                     whether anything is still serving on port {} could not be checked at all, \
                     and uninstall is proceeding without that check. A record written on \
                     another machine, or under a hostname since removed, looks like this. If \
                     that service may still be running, it has to be found and stopped by hand: \
                     this uninstall does not wait for you to, and removes the tooling that \
                     would have stopped it.",
                    record.host, record.service_id, record.port
                ));
            }
            continue;
        }
        let stopped_by_this_pass = attempted
            .iter()
            .any(|candidate| candidate.service_id == record.service_id);
        if !stopped_by_this_pass {
            let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
            // A wildcard record is probed on every loopback family that accepts
            // a connection, and ANY of them blocking blocks the removal: a
            // stranger answering on `127.0.0.1` must not hide the record's own
            // model still serving on `::1`. Each probe asks the address the
            // reachability probe just succeeded against, because
            // `record.endpoint_url` is built from the recorded host, so a
            // `0.0.0.0` or `::` bind would be connected to literally — which
            // does not resolve, fails the probe, and takes a fail-open branch,
            // removing the tooling while a wildcard-bound engine is serving.
            let mut blocking: Option<StoppedRecordVerdict> = None;
            let mut proceeding: Vec<StoppedRecordVerdict> = Vec::new();
            for reachable_host in &reachable_hosts {
                let mut probe_record = record.clone();
                probe_record.endpoint_url =
                    rocm_core::format_http_base_url(reachable_host, record.port);
                let probe = rocm_core::managed_service_endpoint_identity(
                    &probe_record,
                    endpoint_api_key.as_deref(),
                    ENDPOINT_IDENTITY_PROBE_TIMEOUT,
                );
                // Short-circuits: the extra round trip only happens when the
                // probe produced no usable listing, which is the only case its
                // answer can change.
                let auth_refused = probe.is_err()
                    && endpoint_refused_authorization(
                        &probe_record.endpoint_url,
                        endpoint_api_key.as_deref(),
                    );
                // Every arm below is decided in `stopped_record_verdict` and
                // pinned there by
                // `every_identity_answer_maps_to_exactly_one_gate_outcome`, with
                // `an_endpoint_listing_nothing_does_not_block_uninstall` and
                // `a_listener_naming_another_model_does_not_block_uninstall`
                // driving the two proceed-on-a-live-socket arms through this
                // call site. The tests live in `main.rs`'s `mod tests`, far from
                // here; change an arm and expect them, not this loop, to be what
                // goes red.
                let verdict = stopped_record_verdict(probe.ok(), auth_refused);
                if verdict.blocks() {
                    blocking.get_or_insert(verdict);
                } else if !proceeding.contains(&verdict) {
                    proceeding.push(verdict);
                }
            }
            let Some(verdict) = blocking else {
                // The two fail-open outcomes disclose themselves; only
                // `ProceedUnrelated` is silent, because a listener that named
                // its models and did not name ours is the one case that is
                // actually evidence of a stranger. They go into the report
                // rather than straight to stderr so the disclosure is a value a
                // test can assert on — see the field's own comment.
                for verdict in proceeding {
                    match verdict {
                        StoppedRecordVerdict::ProceedListingNothing => {
                            report.warnings.push(format!(
                                "{}:{} still accepts connections and answered the identity probe \
                                 with an empty model list, so it cannot be told from an unrelated \
                                 service; {} is already recorded stopped; proceeding. An engine \
                                 still loading or unloading looks like this. If that is a server \
                                 of yours, it has to be found and stopped by hand: this uninstall \
                                 does not wait for you to.",
                                record.host, record.port, record.service_id
                            ));
                        }
                        StoppedRecordVerdict::ProceedUnidentified => {
                            report.warnings.push(format!(
                                "{}:{} still accepts connections but did not answer the identity \
                                 probe with a usable model list, and service {} is already \
                                 recorded stopped; proceeding. That can be a wedged engine, an \
                                 unrelated server on the port, or a stale endpoint key. If it is \
                                 a server of yours, it has to be found and stopped by hand: this \
                                 uninstall does not wait for you to.",
                                record.host, record.port, record.service_id
                            ));
                        }
                        StoppedRecordVerdict::ProceedUnrelated => {}
                        StoppedRecordVerdict::BlockServingOurModel
                        | StoppedRecordVerdict::BlockAuthRefused => {
                            unreachable!("only non-blocking verdicts are collected")
                        }
                    }
                }
                continue;
            };
            if verdict == StoppedRecordVerdict::BlockAuthRefused {
                report.failed.push(FailedManagedServiceStop {
                    service_id: record.service_id.clone(),
                    reason: format!(
                        "an authenticated server answers on {}:{}, this record's old port",
                        record.host, record.port
                    ),
                    remedy: StopFailureRemedy::RemoveTheStoppedRecord,
                });
                continue;
            }
        }
        // A confirmed stop already dropped this service's endpoint key, and it
        // cannot be re-minted: say so on the failure rather than let the abort
        // message read as if the survivor were untouched.
        let key_dropped = report.stopped.contains(&record.service_id);
        report
            .stopped
            .retain(|stopped| stopped != &record.service_id);
        report
            .relaunch
            .retain(|line| !line.starts_with(&format!("{}: ", record.service_id)));
        report.failed.push(FailedManagedServiceStop {
            service_id: record.service_id.clone(),
            // Two different situations reach here and they need different
            // sentences. Only one of them involved a stop: the other is a
            // record that was already marked stopped before this run, which
            // this pass never attempted to stop, and telling its operator the
            // endpoint survived "the stop" describes something that did not
            // happen — on the one output they have to reason from.
            reason: if stopped_by_this_pass {
                format!(
                    "{}:{} still accepts connections after the stop{}",
                    record.host,
                    record.port,
                    if key_dropped {
                        "; the stop already dropped this service's endpoint key"
                    } else {
                        ""
                    }
                )
            } else {
                format!(
                    "{}:{} is recorded stopped, but something there is still serving this \
                     record's own model",
                    record.host, record.port
                )
            },
            // Not `StopTheService`: the recorded processes are gone, so
            // `rocm services stop` has nothing left to kill and every retry
            // would abort identically.
            remedy: StopFailureRemedy::StopWhatHoldsThePort,
        });
    }
}

/// Whether two probe addresses can reach the same listener on one port.
///
/// `localhost` names whichever loopback family the resolver puts first — `::1`
/// on many Windows hosts — so it can reach a listener on either, and is never
/// told apart from one.
pub(crate) fn hosts_name_one_listener(a: &str, b: &str) -> bool {
    let is_loopback = |h: &str| matches!(h, "127.0.0.1" | "::1");
    a == b || (a == "localhost" && is_loopback(b)) || (b == "localhost" && is_loopback(a))
}

/// The addresses to probe for a service recorded on `host`, in order.
///
/// A service bound to a wildcard address is reachable on loopback; connecting to
/// the wildcard itself is not portable. Which loopback, though, depends on what
/// the listener actually bound: `::` with the usual `IPV6_V6ONLY=1` (the default
/// on Windows) answers on `::1` and *refuses* `127.0.0.1`, while an `0.0.0.0`
/// bind is the mirror image. Probing one family only would read a live,
/// port-holding engine as "nothing is serving" and wave the removal through —
/// the exact defect this gate exists to close — so a wildcard yields both and
/// the caller blocks if either answers.
///
/// The normalized form is what gets probed, not just what gets classified.
/// `loopback_tcp_port_is_reachable` resolves with `(host, port)`, which rejects
/// a bracketed literal like `[::1]` and anything with stray whitespace — and a
/// resolution failure reads as "nothing is serving", which is the wrong
/// direction for a check whose whole job is to catch a surviving engine
/// grandchild. Records do carry bracketed spellings (`loopback_host_key`
/// normalizes them too) because `--host` is free-form.
pub(crate) fn probe_hosts(host: &str) -> Vec<String> {
    // Trimmed and case-folded so the spellings a record can carry — `0.0.0.0`,
    // `::`, `[::]`, `0:0:0:0:0:0:0:0`, `*`, or empty — all resolve to loopback
    // rather than being probed literally (a literal wildcard connect is not
    // portable, and would silently read as "nothing is serving").
    let normalized = host.trim().trim_matches(['[', ']']).to_ascii_lowercase();
    match normalized.as_str() {
        "0.0.0.0" | "::" | "0:0:0:0:0:0:0:0" | "*" | "" => {
            vec!["127.0.0.1".to_owned(), "::1".to_owned()]
        }
        _ => vec![normalized],
    }
}

/// What the gate does about a listener answering on an already-stopped record's
/// recorded port.
///
/// Lifted out of the stop pass so each outcome can be asserted directly. Two of
/// these arms once went untested because reaching them meant standing up a
/// server that answers in a particular way and then reading stderr: mutating
/// either "proceed" arm into a block, or deleting a warning, left every test
/// green while changing what a destructive command does.
///
/// Both halves of that are closed now, so the next reader should not infer a
/// gap from the paragraph above. `every_identity_answer_maps_to_exactly_one_gate_outcome`
/// pins the mapping here; `an_endpoint_listing_nothing_does_not_block_uninstall`
/// and `a_listener_naming_another_model_does_not_block_uninstall` drive the two
/// proceed-on-a-live-socket arms through the real call site; and the warnings
/// are values on [`ManagedServiceStopReport`] rather than `eprintln!`s, so the
/// disclosure is asserted rather than merely emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoppedRecordVerdict {
    /// Serving this record's own model: the engine outlived the supervisor whose
    /// death marked the record stopped.
    BlockServingOurModel,
    /// Refused our credentials — a live server stating it guards this path.
    BlockAuthRefused,
    /// Named its models and ours was not among them. The only "not ours" that is
    /// evidence of anything, so the only one that proceeds silently.
    ProceedUnrelated,
    /// Answered, but listed nothing. Not evidence of a stranger: an engine still
    /// loading or mid-unload looks exactly like this while holding the port.
    ProceedListingNothing,
    /// Listening but unidentifiable — not an OpenAI endpoint, an unparseable
    /// reply, or a server erroring on a rotated key.
    ProceedUnidentified,
}

impl StoppedRecordVerdict {
    /// Whether this outcome stops the uninstall.
    ///
    /// Exhaustive on purpose, rather than `matches!` over the blocking pair.
    /// The wildcard that `matches!` implies would default a newly added variant
    /// to "proceed" — the direction that removes the tooling — and it would
    /// compile, leaving the mistake to be caught by a test that can only
    /// enumerate the variants that existed when it was written. Spelled out,
    /// the compiler stops the next variant until somebody decides which side of
    /// a destructive command it belongs on.
    pub(crate) const fn blocks(self) -> bool {
        match self {
            Self::BlockServingOurModel | Self::BlockAuthRefused => true,
            Self::ProceedUnrelated | Self::ProceedListingNothing | Self::ProceedUnidentified => {
                false
            }
        }
    }
}

/// Decide [`StoppedRecordVerdict`] from what the identity probe answered.
///
/// `identity` is `None` when the probe produced no usable model list at all;
/// `auth_refused` then says whether that was a 401/403 from a live server, which
/// is stronger evidence the port is held than a listing would be.
pub(crate) const fn stopped_record_verdict(
    identity: Option<rocm_core::EndpointIdentity>,
    auth_refused: bool,
) -> StoppedRecordVerdict {
    match identity {
        Some(rocm_core::EndpointIdentity::ServesExpectedModel) => {
            StoppedRecordVerdict::BlockServingOurModel
        }
        Some(rocm_core::EndpointIdentity::ServesOtherModels) => {
            StoppedRecordVerdict::ProceedUnrelated
        }
        Some(rocm_core::EndpointIdentity::ListsNoModels) => {
            StoppedRecordVerdict::ProceedListingNothing
        }
        None if auth_refused => StoppedRecordVerdict::BlockAuthRefused,
        None => StoppedRecordVerdict::ProceedUnidentified,
    }
}

/// Whether a record carrying no start-time cannot be proven ours. Both supported
/// platforms (Linux, Windows) can record one, so an absent value means the
/// record predates the field or its writer never stamped it; either way the pid
/// is unproven.
pub(crate) const fn record_predates_start_ticks(recorded_start_ticks: Option<u64>) -> bool {
    recorded_start_ticks.is_none()
}

/// Whether `endpoint_url` answered the identity probe with an auth refusal.
///
/// A 401/403 is not a failure to reach the endpoint — it is a live HTTP server
/// stating that it guards this path, which is stronger evidence that the port is
/// still held than a model listing would be. It is also the shape the retry run
/// takes for a public service: stopping the recorded processes clears the stored
/// key, so the next `rocm uninstall` probes an authenticated survivor with no
/// credentials and gets exactly this.
///
/// Anything else — unreachable, a timeout, a non-HTTP listener, a 200 whose body
/// did not parse — is not an answer this can act on, and stays with the caller's
/// fail-open.
pub(crate) fn endpoint_refused_authorization(
    endpoint_url: &str,
    endpoint_api_key: Option<&str>,
) -> bool {
    matches!(
        rocm_core::http_get_with_auth(
            endpoint_url,
            "/v1/models",
            endpoint_api_key,
            ENDPOINT_IDENTITY_PROBE_TIMEOUT,
        ),
        Ok(parts) if parts.status == 401 || parts.status == 403
    )
}

/// Service manifests that exist but cannot be parsed back into a record.
///
/// [`load_managed_services`] skips these silently, which is right for listing —
/// one bad file should not break `rocm services list` — but wrong for uninstall:
/// a corrupt manifest can describe a live, GPU-holding server, and removing the
/// tooling while blind to it is the exact defect this gate closes. Returns the
/// file names so the abort message can point at what to inspect.
///
/// Only `*.json` directly under the services directory is a manifest; engine
/// state files live under their own engine directory
/// ([`AppPaths::service_engine_state_path`]), so they are not misread as corrupt
/// records here.
pub(crate) fn unreadable_service_manifests(paths: &AppPaths) -> Result<Vec<PathBuf>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut unreadable = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if serde_json::from_slice::<ManagedServiceRecord>(&bytes).is_err() {
            unreadable.push(path);
        }
    }
    unreadable.sort();
    Ok(unreadable)
}

/// Decide whether uninstall may proceed to remove files, given the outcome of
/// stopping managed services.
///
/// Returns an optional line to print before removal proceeds, or an error that
/// aborts uninstall with nothing removed when a service could not be stopped —
/// so the binaries and service records needed to recover stay in place. Kept
/// separate from the removal it guards so the abort branch (the core safety
/// guarantee) is unit-testable without an unkillable process.
pub(crate) fn uninstall_removal_gate(report: &ManagedServiceStopReport) -> Result<Option<String>> {
    if !report.failed.is_empty() {
        let detail = report
            .failed
            .iter()
            .map(|failure| format!("{} ({})", failure.service_id, failure.reason))
            .collect::<Vec<_>>()
            .join("; ");
        // Say what the aborted run already did. The stop pass is not atomic: the
        // pre-flight refuses before anything is stopped, but a stop that fails,
        // a helper that will not die, or a survivor found after the stops all
        // abort a run that has already stopped some services. Each stopped one
        // is down for good, and stopping it dropped its endpoint key, which
        // cannot be re-minted — a publicly bound service, or one launched with
        // `--require-api-key`, must be served again to come back. "No files
        // were removed" alone would read as "nothing happened".
        let services_note = if report.stopped.is_empty() {
            String::new()
        } else {
            format!(
                " These services were stopped during this run and stay stopped: {}. Stopping \
                 them dropped their endpoint keys, so a service that was publicly bound, or \
                 launched with `--require-api-key`, has to be served again; the command for \
                 each, with its own model and port filled in: {} {}.",
                report.stopped.join(", "),
                report.relaunch.join("; "),
                rocm_engine_protocol::RELAUNCH_KEY_CHOICE
            )
        };
        let helper_note = if report.helper_stopped {
            " The background helper (rocmd) was stopped and stays stopped: nothing supervises or \
             recovers managed services until it is started again."
        } else if report
            .failed
            .iter()
            .any(|failure| failure.remedy == StopFailureRemedy::StopTheDaemon)
        {
            " The background helper (rocmd) was signalled but not confirmed gone, so the servers \
             it supervised may already be down."
        } else {
            ""
        };
        let already_stopped = if services_note.is_empty() && helper_note.is_empty() {
            String::new()
        } else {
            format!(" No files were removed, but:{services_note}{helper_note}")
        };
        // Advice per failure class. A record that will not parse cannot be
        // stopped by `rocm services stop` — that command loads the same file and
        // fails identically — so pointing at it would make every retry abort the
        // same way, which is exactly the dead end this gate must not create.
        //
        // The classes walked here are derived from the failures themselves and
        // ordered by `advice_rank`, so no variant can be dropped by forgetting
        // to extend a list; both that and `advice` are exhaustive matches.
        let mut classes = report
            .failed
            .iter()
            .map(|failure| failure.remedy)
            .collect::<Vec<_>>();
        classes.sort_unstable_by_key(|remedy| remedy.advice_rank());
        classes.dedup();
        let remedies = classes
            .into_iter()
            .map(|remedy| {
                let ids = report
                    .failed
                    .iter()
                    .filter(|failure| failure.remedy == remedy)
                    .map(|failure| failure.service_id.clone())
                    .collect::<Vec<_>>();
                remedy.advice(&ids)
            })
            .collect::<Vec<_>>();
        let may_be_serving = report.failed.iter().any(|failure| {
            matches!(
                failure.remedy,
                StopFailureRemedy::StopTheService | StopFailureRemedy::StopWhatHoldsThePort
            )
        });
        bail!(
            "uninstall aborted: could not stop managed service(s): {detail}.{} {}{}",
            if may_be_serving {
                " Their endpoints may still be serving and holding the GPU."
            } else {
                ""
            },
            remedies.join(" "),
            if already_stopped.is_empty() {
                " No files were removed."
            } else {
                &already_stopped
            }
        );
    }
    if report.stopped.is_empty() && !report.helper_stopped {
        return Ok(None);
    }
    let mut line = String::new();
    if !report.stopped.is_empty() {
        line = format!(
            "stopped {} managed service(s) before removal",
            report.stopped.len()
        );
    }
    if report.helper_stopped {
        if !line.is_empty() {
            line.push_str("; ");
        }
        line.push_str("stopped the background helper (rocmd)");
    }
    Ok(Some(line))
}
