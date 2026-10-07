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
use std::path::PathBuf;
use std::time::Duration;

use crate::*;

/// A managed service uninstall could not confirm stopped, and why.
///
/// The reason is carried rather than discarded so the abort message says what
/// went wrong — an operator facing "could not stop svc-x" with no cause has
/// nothing to act on.
#[derive(Debug)]
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
    /// The record does not parse, so no `rocm` command can act on it — the file
    /// itself has to be repaired or removed.
    RepairTheRecord,
    /// The background helper is still alive, so it can restart what was just
    /// stopped. It has to go before anything is removed.
    StopTheDaemon,
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
                "Every process recorded for {} is gone, yet the endpoint still answers — the \
                 engine outlived its supervisor, so `rocm services stop` has nothing left to \
                 kill. Find what holds that port (`ss -ltnp` on Linux, `Get-NetTCPConnection \
                 -LocalPort <port>` on Windows), stop it, then re-run uninstall.",
                ids.join(", ")
            ),
            Self::StopTheDaemon => "The background helper restarts managed services on its own, \
                                    so it has to be stopped before uninstall can safely remove \
                                    anything: kill that pid (`kill <pid>` on Linux, `Stop-Process -Id <pid> -Force` \
                                    on Windows), then re-run uninstall."
                .to_owned(),
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
            Self::StopTheDaemon => 2,
            Self::RepairTheDaemonState => 3,
            Self::RepairTheRecord => 4,
        }
    }
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
}

/// The failure recorded when the background helper is live but cannot be proven
/// to be `rocmd`, so it was deliberately left alone.
pub(crate) fn daemon_identity_unverified(daemon_pid: u32) -> FailedManagedServiceStop {
    FailedManagedServiceStop {
        service_id: format!("rocmd (pid {daemon_pid})"),
        reason: "the background helper's identity could not be verified, so it was left running \
                 rather than risk signalling an unrelated process that inherited its pid"
            .to_owned(),
        remedy: StopFailureRemedy::StopTheDaemon,
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
/// spawns on, and off Linux it is the only one that applies.
///
/// Linux (`/proc`) and Windows (`GetProcessTimes` creation time) can record and
/// compare a start-time, so a recycled pid is told apart there. On a platform
/// with neither (macOS) no start-time exists, so this degrades to the same
/// best-effort match the managed-service kills already use there rather than
/// making uninstall unusable whenever the daemon is up: `running` and a live-pid
/// check are the whole of the protection. Whether this platform can
/// read a start-time at all is asked of a process known to be alive — this one —
/// so a failed reading of the daemon's pid is never mistaken for a platform that
/// cannot read them.
///
/// Two residual gaps, and the list above is otherwise the complete set of
/// inputs. First, a *missing* `runtime-state.json` is taken at face value as "no
/// daemon". Deleting that file by hand while `rocmd` is live therefore skips the
/// daemon stop silently. This is deliberate — a missing file is the ordinary
/// never-started case, and there is no pid to verify or signal without it — but
/// it does mean the gate is only as good as the state file. An unreadable one is
/// the case that aborts; an absent one is the case that proceeds. Second, on
/// macOS a pid recycled while `running` was still true (a crash, not a clean
/// exit) cannot be told from the daemon itself.
pub(crate) fn stop_background_helper_before_uninstall(
    paths: &AppPaths,
    report: &mut ManagedServiceStopReport,
) {
    // Three outcomes, not two. `load` returns `Ok(None)` only when there is no
    // state file at all; a permission error or a half-written file returns
    // `Err`, and discarding that would skip the daemon stop silently and let
    // uninstall delete the tooling while a live `rocmd` respawns what the
    // service pass just stopped — the exact defect this gate exists to close.
    // The service side already treats an unparseable record as a hard failure
    // (see `unreadable_service_manifests`); this is the same call.
    let state = match AutomationRuntimeState::load(paths) {
        Ok(Some(state)) => state,
        Ok(None) => return,
        Err(error) => {
            // Name the file, not a pid: nothing parsed, so no pid was ever
            // recovered and "kill that pid" would be advice nobody can follow.
            // Repairing or deleting this file is the only action that clears it,
            // and uninstall has no `--force`, so the abort has to say so.
            report.failed.push(FailedManagedServiceStop {
                service_id: format!("rocmd ({})", paths.automation_state_path().display()),
                reason: format!(
                    "the background helper's runtime state could not be read, so it cannot be \
                     confirmed stopped: {error:#}"
                ),
                remedy: StopFailureRemedy::RepairTheDaemonState,
            });
            return;
        }
    };
    // `running` is the same inactive contract `background_helper_already_running`
    // spawns on, and honoring it here is what keeps this path off an unrelated
    // process. Only the clean-shutdown path writes `running: false` — a daemon
    // started without `--automations-enabled` returns before its first state
    // write — so a false flag means the recorded pid belongs to a daemon that
    // already exited, and anything alive under it now inherited the number.
    // Where no start-time can be read (macOS) this flag is the only guard against
    // a force-kill landing on a stranger.
    if !state.running || state.daemon_pid == 0 || state.daemon_pid == std::process::id() {
        return;
    }
    if !rocm_core::process_is_running(state.daemon_pid) {
        return;
    }
    // Kill nothing this cannot identify. `terminate_verified` with `force` is a
    // SIGKILL across the whole process tree, and `runtime-state.json` outlives a
    // crash, OOM-kill or reboot with `running` still true — so the recorded PID
    // can belong to an unrelated process by the time uninstall runs.
    //
    // `identity_state` maps an unrecorded start-time to `Matches` (best-effort,
    // for legacy state files), which is exactly wrong here: a state file written
    // by a pre-upgrade `rocmd` has no `daemon_start_ticks`, and treating that as
    // a match would force-kill a whole tree at a PID this cannot prove is ours —
    // the ordinary upgrade path. So when this platform *can* read a start-time
    // (`/proc`) yet none was recorded, the record simply predates the field:
    // treat it as unverifiable, leave the process alone, and abort. Being told
    // to kill a PID is recoverable; killing an unrelated process tree is not.
    //
    // Only where no start-time can ever be read (macOS) does
    // this fall back to the best-effort match the managed-service kills already
    // use there — otherwise uninstall could never stop a live daemon on those
    // platforms. That residual gap is documented on `daemon_start_ticks`.
    //
    // Which platform this is gets answered by a process that is definitely
    // alive — this one — rather than by whether the daemon's own reading
    // happened to succeed. Asking the daemon's PID cannot tell "no `/proc` on
    // this OS" from "that one read just failed", and those must not be
    // conflated: on Linux a legacy record whose PID was momentarily unreadable
    // would otherwise answer `None` to both, land in the best-effort `Matches`
    // arm meant for Windows, and force-kill a tree on a record it never
    // verified. Reading our own PID has no such window — if the platform can
    // report a start-time at all, it reports ours.
    //
    // One reading of the *daemon's* start-time then serves the identity question
    // below. Reading it twice let two answers come from different observations,
    // so a PID that flickered could clear one check and fail the other.
    let identity = rocm_core::ProcessIdentity::new(state.daemon_pid, state.daemon_start_ticks);
    let observed_start_ticks = rocm_core::process_start_ticks(state.daemon_pid);
    let unverifiable_pre_upgrade_record = record_predates_start_ticks(state.daemon_start_ticks);
    match daemon_identity_outcome(
        rocm_core::identity_state_with_observed(&identity, observed_start_ticks),
        unverifiable_pre_upgrade_record,
    ) {
        DaemonIdentityOutcome::NothingOfOurs => return,
        DaemonIdentityOutcome::Unverifiable => {
            report
                .failed
                .push(daemon_identity_unverified(state.daemon_pid));
            return;
        }
        DaemonIdentityOutcome::Stop => {}
    }
    println!("stopping the background helper (rocmd)");
    let outcome = rocm_core::terminate_verified(
        &identity,
        rocm_core::KillScope::Tree,
        MANAGED_STOP_GRACE,
        true,
    );
    if outcome.stopped() {
        report.stopped.push("rocmd (background helper)".to_owned());
    } else {
        report.failed.push(FailedManagedServiceStop {
            service_id: format!("rocmd (pid {})", state.daemon_pid),
            reason: "the background helper could not be confirmed stopped, and it restarts \
                     managed services whose endpoint stops answering"
                .to_owned(),
            remedy: StopFailureRemedy::StopTheDaemon,
        });
    }
}

/// Stop every live managed service before uninstall removes the binaries and
/// service records needed to stop them.
///
/// The defect this closes (EAI-8014): uninstall reported success while a
/// publicly-bound, GPU-holding managed server kept serving, and deleted the
/// `rocm`/`rocmd` binaries and service manifests — so the supported
/// `rocm services stop` path was gone and only a manual PID kill remained.
///
/// A service is only counted stopped when [`stop_internal_managed_service`]
/// confirms every recorded process is gone (its `status` reaches `stopped`);
/// anything else lands in `failed` so the caller aborts and keeps the tooling.
///
/// Fail-closed on discovery too: if the services directory exists but cannot be
/// enumerated, the error propagates so uninstall aborts rather than deleting the
/// tooling while blind to what it manages. (`load_managed_services` returns an
/// empty list — not an error — when no services directory exists, so a clean
/// install still uninstalls.) A manifest that exists but does not parse is
/// likewise a failure, not a silent skip — see
/// [`unreadable_service_manifests`].
pub(crate) fn stop_managed_services_before_uninstall(
    paths: &AppPaths,
) -> Result<ManagedServiceStopReport> {
    stop_managed_services_with(paths, stop_internal_managed_service)
}

/// [`stop_managed_services_before_uninstall`] with the per-service stop
/// injectable, so a test can make it fail — which no real process reliably does.
pub(crate) fn stop_managed_services_with(
    paths: &AppPaths,
    stop_service: impl Fn(&AppPaths, &str) -> Result<serde_json::Value>,
) -> Result<ManagedServiceStopReport> {
    let mut report = ManagedServiceStopReport::default();
    // Everything that can doom the run is decided BEFORE anything is stopped.
    //
    // Stopping is not free and not undoable: each confirmed stop drops that
    // service's endpoint key, and this command's own abort text says a publicly
    // bound service has to be served again with an explicit flag to come back.
    // Once the gate is going to abort — the tooling stays, nothing is removed —
    // every stop performed on the way there is pure cost to the operator, paid
    // for a removal that will not happen. So a helper stop that recorded a
    // failure returns here, and an unparseable manifest (which can perfectly
    // well describe a live, GPU-holding server) is collected up front rather
    // than after every other service is already down.
    //
    // The manifest scan goes first because it is the only one of the two that
    // costs nothing: it reads the services directory and signals nothing. The
    // helper stop is itself destructive — it force-kills a process tree — so
    // running it ahead of a read that can doom the run would terminate a live,
    // perfectly verifiable `rocmd` for an uninstall that then removes nothing.
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
        return Ok(report);
    }
    stop_background_helper_before_uninstall(paths, &mut report);
    if !report.failed.is_empty() {
        return Ok(report);
    }
    let records = load_managed_services(paths)?;
    let mut attempted: Vec<&ManagedServiceRecord> = Vec::new();
    for record in &records {
        if !managed_service_is_live(record) {
            continue;
        }
        attempted.push(record);
        // Each stop waits out a bounded grace per recorded process (and the
        // engine's own stop before that), so name the service first: without
        // this, an uninstall with a live server reads as a hang.
        println!("stopping managed service {}", record.service_id);
        match stop_service(paths, &record.service_id) {
            Ok(result) => {
                let status = result
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                if status == "stopped" {
                    report.stopped.push(record.service_id.clone());
                } else {
                    report.failed.push(FailedManagedServiceStop {
                        service_id: record.service_id.clone(),
                        reason: format!("still \"{status}\" after the stop attempt"),
                        remedy: StopFailureRemedy::StopTheService,
                    });
                }
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
    // manifest and its old port forever (nothing prunes them, and there is no
    // `services remove`), so an unrelated process that later binds that port
    // must not brick uninstall with no way out:
    //
    //   * stopped by this pass — any listener fails the gate. We killed the
    //     recorded processes seconds ago; something answering there is the
    //     survivor.
    //   * already stopped before this run — a listener alone proves nothing, so
    //     it has to identify itself as this record's own model before it counts.
    //     An unrelated service on a recycled port does not, and is not blocked.
    for record in &records {
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
        let Some(reachable_host) = candidates
            .iter()
            .find(|candidate| loopback_tcp_port_is_reachable(candidate, record.port))
            .cloned()
        else {
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
                     whether anything is still serving on port {} could not be checked at all — \
                     proceeding. A record written on another machine, or under a hostname since \
                     removed, looks like this. If that service may still be running, stop it \
                     before re-running uninstall.",
                    record.host, record.service_id, record.port
                ));
            }
            continue;
        };
        let stopped_by_this_pass = attempted
            .iter()
            .any(|candidate| candidate.service_id == record.service_id);
        if !stopped_by_this_pass {
            let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
            // Ask the same address the reachability probe just succeeded against.
            // `record.endpoint_url` is built from the recorded host, so a `0.0.0.0`
            // or `::` bind would be connected to literally — which does not
            // resolve, fails the probe, and takes the fail-open branch below,
            // removing the tooling while a wildcard-bound engine is still
            // serving. That is the defect this gate exists to close, so the
            // identity probe gets the normalized host too.
            let mut probe_record = record.clone();
            probe_record.endpoint_url =
                rocm_core::format_http_base_url(&reachable_host, record.port);
            let probe = rocm_core::managed_service_endpoint_identity(
                &probe_record,
                endpoint_api_key.as_deref(),
                ENDPOINT_IDENTITY_PROBE_TIMEOUT,
            );
            // Short-circuits: the extra round trip only happens when the probe
            // produced no usable listing, which is the only case its answer can
            // change.
            let auth_refused = probe.is_err()
                && endpoint_refused_authorization(
                    &probe_record.endpoint_url,
                    endpoint_api_key.as_deref(),
                );
            // Every arm below is decided in `stopped_record_verdict` and pinned
            // there by `every_identity_answer_maps_to_exactly_one_gate_outcome`,
            // with `an_endpoint_listing_nothing_does_not_block_uninstall` and
            // `a_listener_naming_another_model_does_not_block_uninstall` driving
            // the two proceed-on-a-live-socket arms through this call site. The
            // tests live at the bottom of this file, far from here; change an
            // arm and expect them, not this match, to be what goes red.
            let verdict = stopped_record_verdict(probe.ok(), auth_refused);
            if !verdict.blocks() {
                // The two fail-open outcomes disclose themselves; only
                // `ProceedUnrelated` is silent, because a listener that named
                // its models and did not name ours is the one case that is
                // actually evidence of a stranger. They go into the report
                // rather than straight to stderr so the disclosure is a value a
                // test can assert on — see the field's own comment.
                match verdict {
                    StoppedRecordVerdict::ProceedListingNothing => report.warnings.push(format!(
                        "{}:{} still accepts connections and answered the identity probe with an \
                         empty model list, so it cannot be told from an unrelated service; {} is \
                         already recorded stopped — proceeding. An engine still loading or \
                         unloading looks like this. If that is a server of yours, stop whatever \
                         holds that port first.",
                        record.host, record.port, record.service_id
                    )),
                    StoppedRecordVerdict::ProceedUnidentified => report.warnings.push(format!(
                        "{}:{} still accepts connections but did not answer the identity probe \
                         with a usable model list, and service {} is already recorded stopped — \
                         proceeding. That can be a wedged engine, an unrelated server on the \
                         port, or a stale endpoint key. If it is a server of yours, stop whatever \
                         holds that port first.",
                        record.host, record.port, record.service_id
                    )),
                    StoppedRecordVerdict::ProceedUnrelated => {}
                    StoppedRecordVerdict::BlockServingOurModel
                    | StoppedRecordVerdict::BlockAuthRefused => unreachable!("guarded by blocks()"),
                }
                continue;
            }
            if verdict == StoppedRecordVerdict::BlockAuthRefused {
                report.failed.push(FailedManagedServiceStop {
                    service_id: record.service_id.clone(),
                    reason: format!(
                        "{}:{} refused the identity probe's credentials, so an authenticated \
                         server is still serving there",
                        record.host, record.port
                    ),
                    remedy: StopFailureRemedy::StopWhatHoldsThePort,
                });
                continue;
            }
        }
        report
            .stopped
            .retain(|stopped| stopped != &record.service_id);
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
                    "{}:{} still accepts connections after the stop",
                    record.host, record.port
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
    Ok(report)
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

/// Whether a record carrying no start-time predates the field, as opposed to
/// coming from a platform that cannot report one.
///
/// Deliberately takes no PID, and that is the whole point of it being a function
/// rather than two lines at the call site. The question is about the *platform*,
/// and asking it of the process under inspection cannot tell "this OS has no
/// `/proc`" from "that one read just failed" — a conflation that sends a legacy
/// record down the best-effort `Matches` arm and force-kills a tree it never
/// verified. Answering from our own PID has no such window: the process asking
/// is, by construction, running. Keeping the target PID out of the signature
/// makes that conflation unrepresentable here rather than merely avoided.
pub(crate) fn record_predates_start_ticks(recorded_start_ticks: Option<u64>) -> bool {
    recorded_start_ticks.is_none() && rocm_core::process_start_ticks(std::process::id()).is_some()
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
        // Say what the aborted run already did. The stop pass is not atomic: a
        // service stopped before the failing one is down for good, and stopping
        // it dropped its endpoint key, which cannot be re-minted — a public
        // service must be served again with `--allow-public-bind` to come back.
        // "No files were removed" alone would read as "nothing happened".
        let already_stopped = if report.stopped.is_empty() {
            String::new()
        } else {
            format!(
                " No files were removed, but these services were stopped before the failure and \
                 stay stopped: {}. Stopping them dropped their endpoint keys, so a publicly bound \
                 one has to be served again with `rocm serve --allow-public-bind` to return.",
                report.stopped.join(", ")
            )
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
    if report.stopped.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "stopped {} managed service(s) before removal",
        report.stopped.len()
    )))
}
