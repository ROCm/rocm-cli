// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reconciling running local servers against the ROCm runtime being activated.
//!
//! `rocm runtimes activate` and `rocm runtimes rollback` change which runtime
//! the *next* launch uses. A server that is already running keeps serving on
//! the runtime it started with, and this module is what lets the CLI say so
//! from state rather than from a fixed note: it reads the live service records,
//! classifies each against the runtime being activated, renders that as the
//! service half of the activation report, and — behind `--restart-services
//! --yes` — moves the stale ones onto the new runtime.
//!
//! Full domain extraction per `docs/architecture.md`: this file owns
//! [`ServiceRuntimeState`], [`RuntimeServiceEntry`], [`FailedServiceRestart`],
//! [`RuntimeServiceReconciliation`] and [`ServiceRuntimePin`]. `RuntimesCommand`
//! and `fn runtimes()` stay in `main.rs`, which is the documented exception for
//! a subsystem with its own clap subcommand.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord};

use crate::{
    engine_manages_own_runtime, load_managed_service, load_managed_services,
    managed_service_is_live, restart_internal_managed_service, runtime_manifest_for_selector,
    therock,
};

/// Where one live local server stands relative to the runtime being activated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServiceRuntimeState {
    /// The service already runs on this runtime, or is pinned to something
    /// this activation does not move (a self-managed engine runtime).
    Matches,
    /// The service still runs on a different, recorded ROCm runtime.
    Stale { recorded: String },
    /// The service records no runtime, so which runtime it actually loaded
    /// cannot be derived from the record.
    Unknown,
}

/// One live local server named in the activation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeServiceEntry {
    pub(crate) service_id: String,
    pub(crate) engine: String,
    /// The runtime key recorded on the service. `None` for `Unknown` entries.
    pub(crate) recorded_runtime: Option<String>,
}

impl RuntimeServiceEntry {
    fn recorded_runtime_display(&self) -> &str {
        self.recorded_runtime.as_deref().unwrap_or("<unset>")
    }
}

/// A local server whose restart onto the newly active runtime did not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FailedServiceRestart {
    pub(crate) service_id: String,
    pub(crate) error: String,
}

/// The live local servers seen at activation time, grouped by what the
/// activation means for them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RuntimeServiceReconciliation {
    /// Live services still running on a different recorded runtime.
    pub(crate) stale: Vec<RuntimeServiceEntry>,
    /// Live services whose runtime cannot be derived from their record.
    pub(crate) unknown: Vec<RuntimeServiceEntry>,
    /// Stale services successfully moved onto the new runtime.
    pub(crate) restarted: Vec<String>,
    /// Stale services that could not be moved. Their records are put back on
    /// the runtime they recorded before the attempt. Whether the service itself
    /// is still up depends on how far the attempt got, so each one that is
    /// still serving stays listed in `stale` too.
    pub(crate) failed: Vec<FailedServiceRestart>,
    /// Whether `restart_stale_runtime_services` ran. Read only to decide
    /// whether offering `--restart-services` is advice the user can still act
    /// on: a service left in `stale` by a failed attempt is not one that flag
    /// would move, because it was just tried.
    pub(crate) restart_attempted: bool,
}

/// Classify one record against the runtime key being activated.
///
/// One class of live service is deliberately never stale:
///
/// - engines that manage their own runtime (`engine_manages_own_runtime`)
///   record an engine-private key such as `lemonade-embeddable-<version>`,
///   which is never a ROCm runtime key, so comparing it to one would report
///   every lemonade server as permanently stale.
///
/// `env_id` is deliberately NOT treated as a pin. Both vLLM and Lemonade write
/// it unconditionally on every launch — `engines/vllm/src/lib.rs` and
/// `engines/lemonade/src/lib.rs` both record `request.env_id` falling back to
/// the runtime's own `env_id`, which is never empty — and
/// `refresh_from_engine_state` adopts it without ever clearing it. A record
/// therefore acquires an `env_id` whether or not the user passed `--env-id`,
/// so treating it as a pin would silently exclude every service from
/// reconciliation and defeat the feature entirely.
///
/// A record that has an `env_id` but no `runtime_id` falls through to
/// `Unknown`, which is reported under `services_with_unrecorded_runtime` and
/// never restarted — `restart_stale_runtime_services` iterates `services.stale`
/// only. In practice this arm is rarely reached: both engines write a non-empty
/// `runtime_id` into the engine state file on every launch, which
/// `refresh_from_engine_state` adopts into `record.runtime_id`, so the record
/// almost always arrives here with a recorded runtime.
pub(crate) fn classify_service_runtime_state(
    record: &ManagedServiceRecord,
    manifests: &[therock::InstalledRuntimeManifest],
    runtime_key: &str,
) -> ServiceRuntimeState {
    if engine_manages_own_runtime(&record.engine) {
        return ServiceRuntimeState::Matches;
    }
    match record.runtime_id.as_deref() {
        Some(recorded) if recorded == runtime_key => ServiceRuntimeState::Matches,
        // The recorded value is not guaranteed to be an exact runtime key.
        // Every launch path records one, but `refresh_from_engine_state`
        // afterwards adopts whatever the engine reports, and an engine may
        // report the manifest's `runtime_id` instead — the family form
        // (`therock-release:gfx120X-all`) that every installed version of that
        // family shares. Resolve it the way every other selector in this file
        // is resolved before calling the service stale, or a server already on
        // the runtime being activated is named as left behind and
        // `--restart-services` stops and respawns it for nothing.
        //
        // The resolution is only as exact as the recorded value: a family form
        // shared by two installed versions resolves to neither, so such a
        // record falls through to `Stale` and is reported (and, with
        // `--restart-services`, moved) even when the server is already on the
        // runtime being activated. That is the conservative direction — the
        // record genuinely does not say which version — and it shrinks as
        // records are rewritten, since `refresh_from_engine_state` now prefers
        // the engine's exact `requested_runtime_id` over the family form.
        Some(recorded)
            if runtime_manifest_for_selector(manifests, recorded)
                .is_some_and(|manifest| manifest.runtime_key == runtime_key) =>
        {
            ServiceRuntimeState::Matches
        }
        Some(recorded) => ServiceRuntimeState::Stale {
            recorded: recorded.to_owned(),
        },
        None => ServiceRuntimeState::Unknown,
    }
}

/// Read the live local servers and work out which of them the activation of
/// `runtime_key` leaves behind.
///
/// Called BEFORE the activation writes, so a services folder that cannot be
/// read fails the command while the previous runtime is still fully in place.
///
/// `load_managed_services` refreshes each record against the engine state and
/// the real processes, and rewrites the manifest when that refresh corrects a
/// status — so reading service state here can touch disk. That is safe on this
/// path: unlike `build_service_prune_plan`, which snapshots manifest mtimes
/// first because its age gate reads them, nothing in activation looks at
/// manifest timestamps. The refresh is also what makes the report trustworthy,
/// since a dead process must not be reported as a service still holding the
/// previous runtime.
pub(crate) fn reconcile_services_for_runtime(
    paths: &AppPaths,
    manifests: &[therock::InstalledRuntimeManifest],
    runtime_key: &str,
) -> Result<RuntimeServiceReconciliation> {
    let mut reconciliation = RuntimeServiceReconciliation::default();
    for record in load_managed_services(paths)? {
        if !managed_service_is_live(&record) {
            continue;
        }
        match classify_service_runtime_state(&record, manifests, runtime_key) {
            ServiceRuntimeState::Matches => {}
            ServiceRuntimeState::Stale { recorded } => {
                reconciliation.stale.push(RuntimeServiceEntry {
                    service_id: record.service_id,
                    engine: record.engine,
                    recorded_runtime: Some(recorded),
                });
            }
            ServiceRuntimeState::Unknown => reconciliation.unknown.push(RuntimeServiceEntry {
                service_id: record.service_id,
                engine: record.engine,
                recorded_runtime: None,
            }),
        }
    }
    Ok(reconciliation)
}

/// Render the service half of an activation/rollback report.
///
/// Always prints a count, including `services_on_previous_runtime: 0`, so a
/// clean switch is distinguishable from a report that simply forgot to look —
/// which is exactly what the old fixed note could not tell a user.
pub(crate) fn render_runtime_service_reconciliation(
    services: &RuntimeServiceReconciliation,
    active_runtime_key: &str,
) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "  services_on_previous_runtime: {}",
        services.stale.len()
    );
    for entry in &services.stale {
        let _ = writeln!(
            output,
            "    - {} engine={} recorded_runtime={}",
            entry.service_id,
            entry.engine,
            entry.recorded_runtime_display()
        );
    }
    if !services.unknown.is_empty() {
        let _ = writeln!(
            output,
            "  services_with_unrecorded_runtime: {}",
            services.unknown.len()
        );
        for entry in &services.unknown {
            let _ = writeln!(
                output,
                "    - {} engine={} recorded_runtime={}",
                entry.service_id,
                entry.engine,
                entry.recorded_runtime_display()
            );
        }
    }
    if !services.restarted.is_empty() {
        let _ = writeln!(output, "  services_restarted: {}", services.restarted.len());
        for service_id in &services.restarted {
            let _ = writeln!(output, "    - {service_id}");
        }
    }
    if !services.failed.is_empty() {
        let _ = writeln!(
            output,
            "  services_restart_failed: {}",
            services.failed.len()
        );
        for failure in &services.failed {
            let _ = writeln!(output, "    - {}: {}", failure.service_id, failure.error);
        }
    }
    // Only offered when no restart was attempted. `restart_stale_runtime_services`
    // leaves an entry in `stale` only when the attempt failed AND the service is
    // still up, and telling that user to add the flag they just passed is advice
    // that cannot help — `bail_on_failed_service_restarts` names the real next
    // step for them instead.
    //
    // The command is spelled out rather than left as "add --restart-services":
    // three of the four reports that print this come from a command that has no
    // such flag (`rocm install sdk`, `rocm update --apply --activate`) or one
    // where re-running it with the flag would switch the runtime a second time
    // (`rocm runtimes rollback`). Activating the runtime that is already active
    // is the one invocation that moves the servers without moving the runtime —
    // and `activate_runtime` now leaves `previous_runtime_key` alone on that
    // path, so following this note no longer destroys the rollback target the
    // next line offers.
    if !services.stale.is_empty() && !services.restart_attempted {
        let _ = writeln!(
            output,
            "  note: those keep serving on their recorded runtime until they are restarted; \
             run `rocm runtimes activate {active_runtime_key} --restart-services --yes` to \
             move them"
        );
    }
    output
}

/// `--restart-services` stops and respawns running local servers, so it takes
/// the same explicit approval every other service mutation takes. This path
/// never prompts — `--yes` is the only approval it accepts.
pub(crate) fn ensure_service_restart_approved(
    restart_services: bool,
    yes: bool,
    command: &str,
) -> Result<()> {
    if restart_services && !yes {
        bail!(
            "restarting running local servers requires --yes.\n\nTry: rocm runtimes {command} --restart-services --yes"
        );
    }
    Ok(())
}

/// Move every stale live service onto the newly active runtime.
///
/// Best-effort per service: one failure does not abandon the rest, and the
/// caller reports every failure and exits non-zero.
pub(crate) fn restart_stale_runtime_services(
    paths: &AppPaths,
    runtime_key: &str,
    services: &mut RuntimeServiceReconciliation,
) {
    // `services_on_previous_runtime` counts `stale`, so an entry stays there
    // only while the claim it makes is still true. A service that was moved, or
    // that the attempt took down, is no longer a live server serving from the
    // previous runtime and must not be counted as one — but neither is the
    // converse safe to assume: a restart refused BEFORE the stop leaves the
    // server up, and dropping it would under-report a live server on the old
    // runtime just as badly. So each entry is drained, and only put back when
    // the service is read back as still running.
    for mut entry in std::mem::take(&mut services.stale) {
        match restart_service_onto_runtime(paths, &entry.service_id, runtime_key) {
            Ok(()) => services.restarted.push(entry.service_id),
            Err(error) => {
                services.failed.push(FailedServiceRestart {
                    service_id: entry.service_id.clone(),
                    error: format!("{error:#}"),
                });
                if let Some(record) = service_still_serving(paths, &entry.service_id) {
                    // Re-read rather than reused: `restart_service_onto_runtime`
                    // restores the record's pin on failure and that restore is
                    // itself fallible, so the runtime this names is taken from
                    // the record as it now stands.
                    entry.recorded_runtime = record.runtime_id;
                    services.stale.push(entry);
                }
            }
        }
    }
    services.restart_attempted = true;
}

/// The service's record if it is still running, having failed to restart.
///
/// A failed restart does NOT imply a stopped server:
/// `restart_internal_managed_service` refuses a public bind with no endpoint key
/// before it stops anything, and `restart_service_onto_runtime` can fail earlier
/// still while rewriting the record. `load_managed_service` refreshes the record
/// against the real processes, so this is read from the world rather than
/// inferred from how far the code got. A record that cannot be read back
/// returns `None` here, and the report is a bi-state rather than a tri-state:
/// [`bail_on_failed_service_restarts`] derives "stopped" as the complement of
/// "still serving", so such a service is named as stopped by the attempt. That
/// is acceptable because reaching it takes a double fault — the record
/// vanishing or corrupting between the reconciliation that read it and this
/// retry — and the restart failure itself is named in the error either way.
pub(crate) fn service_still_serving(
    paths: &AppPaths,
    service_id: &str,
) -> Option<ManagedServiceRecord> {
    load_managed_service(paths, service_id)
        .ok()
        .filter(managed_service_is_live)
}

/// Rewrite the service record onto `runtime_key`, THEN restart it.
///
/// The order is load-bearing. `restart_internal_managed_service` rebuilds the
/// child's argv from the record on disk and passes `record.runtime_id`
/// verbatim, so restarting first would bring the service back up on the runtime
/// it was already using and report success — the very bug this path exists to
/// fix. The restart itself goes through the unchanged managed-serve path, so
/// device policy (including gpu_required) is validated exactly as it is for a
/// fresh `rocm serve`.
///
/// The record's `env_id` is cleared in the same write, and that is equally
/// load-bearing: `builtin_engine_serve_http_args` leaves `--runtime-id` out of
/// the child's argv whenever `env_id` is set, and `env_root_for_service` then
/// hands the child no engine environment root either. A record that kept its
/// `env_id` would come back on whatever runtime the engine resolves for itself
/// — the most recently *installed* one, not the one just activated — while the
/// record claimed the new key. The engine writes its own `env_id` back into its
/// state on the next launch, so nothing is lost by dropping it here.
///
/// On failure the record is put back on the runtime it last actually ran on.
/// What a failure leaves behind is not decided here: the attempt may be refused
/// before anything is stopped (leaving the service up on the runtime it
/// recorded) or fail after the stop (leaving it down). That is why the caller
/// reads the outcome back with [`service_still_serving`] instead of inferring
/// it from how far the attempt got.
pub(crate) fn restart_service_onto_runtime(
    paths: &AppPaths,
    service_id: &str,
    runtime_key: &str,
) -> Result<()> {
    let previous_pin = pin_service_record_to_runtime(paths, service_id, runtime_key)?;
    match restart_internal_managed_service(paths, service_id) {
        Ok(_) => Ok(()),
        // A restore that fails is reported with the restart failure rather than
        // swallowed: silently keeping the new key would leave the record
        // describing a runtime the service never loaded, which is the state
        // this whole path exists to prevent.
        Err(error) => Err(
            match restore_service_record_pin(paths, service_id, previous_pin) {
                Ok(()) => error,
                Err(restore_error) => error.context(format!(
                    "the service record could not be put back on the runtime it ran on either \
                     ({restore_error:#}), so it now names `{runtime_key}`"
                )),
            },
        ),
    }
}

/// What a service record said about its runtime before a restart rewrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServiceRuntimePin {
    pub(crate) runtime_id: Option<String>,
    pub(crate) env_id: Option<String>,
}

/// Point a service record at `runtime_key`, returning the pin it replaced.
///
/// Clearing `env_id` is half the job, not tidying: see
/// [`restart_service_onto_runtime`] for why a record that keeps it is restarted
/// onto a runtime nobody chose.
pub(crate) fn pin_service_record_to_runtime(
    paths: &AppPaths,
    service_id: &str,
    runtime_key: &str,
) -> Result<ServiceRuntimePin> {
    let mut record = load_managed_service(paths, service_id)?;
    let previous = ServiceRuntimePin {
        runtime_id: record.runtime_id.clone(),
        env_id: record.env_id.clone(),
    };
    record.runtime_id = Some(runtime_key.to_owned());
    record.env_id = None;
    record.write()?;
    Ok(previous)
}

/// Put a service record's runtime pin back to what it was before a restart
/// attempt rewrote it.
pub(crate) fn restore_service_record_pin(
    paths: &AppPaths,
    service_id: &str,
    pin: ServiceRuntimePin,
) -> Result<()> {
    let mut record = load_managed_service(paths, service_id)?;
    record.runtime_id = pin.runtime_id;
    record.env_id = pin.env_id;
    record.write()
}

/// Severity and trailing detail for the audit line an activation writes.
///
/// A restart failure does not undo the switch, so the event still says
/// "activated" — but it must not say it at `info` with nothing else, because
/// the command exits non-zero straight afterwards and the audit log is where an
/// incident is reconstructed from.
pub(crate) fn service_restart_audit_outcome(
    services: &RuntimeServiceReconciliation,
) -> (&'static str, String) {
    if services.failed.is_empty() {
        let restarted = services.restarted.len();
        let detail = if restarted == 0 {
            String::new()
        } else {
            format!(" services_restarted={restarted}")
        };
        return ("info", detail);
    }
    (
        "warn",
        format!(
            " services_restarted={} services_restart_failed={}",
            services.restarted.len(),
            services.failed.len()
        ),
    )
}

/// Fail the command when any requested restart did not happen, naming every
/// service that was left behind. The report has already been printed, so this
/// only has to make the exit code match what the report says.
///
/// What a failure left behind is split out per service rather than asserted for
/// all of them: the restart stops a server before respawning it, so most
/// failures leave it down — but one refused before the stop leaves it serving,
/// and telling that user their server is down sends them away from a live
/// server on the runtime they just switched off.
pub(crate) fn bail_on_failed_service_restarts(
    services: &RuntimeServiceReconciliation,
) -> Result<()> {
    if services.failed.is_empty() {
        return Ok(());
    }
    let still_serving = services
        .stale
        .iter()
        .map(|entry| entry.service_id.as_str())
        .collect::<Vec<_>>();
    let stopped = services
        .failed
        .iter()
        .map(|failure| failure.service_id.as_str())
        .filter(|service_id| !still_serving.contains(service_id))
        .collect::<Vec<_>>();
    // Every failure lands in exactly one of the two lists below, so the ids are
    // named there rather than up front as well.
    // "Each record was put back" is not said outright: `restart_service_onto_runtime`
    // reports a restore that failed too, and that record now names a runtime its
    // server never loaded. The per-service lines below carry which is which, so
    // the summary points at them rather than overriding them with a promise the
    // code already knows can be false.
    let mut message = format!(
        "the runtime was switched, but {} local server(s) could not be restarted onto it. Each \
         record was put back on the runtime it last ran on unless its line below says otherwise.",
        services.failed.len()
    );
    if !stopped.is_empty() {
        let _ = write!(
            message,
            " These were stopped by the attempt and are no longer serving: {}.",
            stopped.join(", ")
        );
    }
    if !still_serving.is_empty() {
        let _ = write!(
            message,
            " These are still serving from the runtime they recorded, and stay counted under \
             services_on_previous_runtime: {}.",
            still_serving.join(", ")
        );
    }
    let _ = write!(
        message,
        " Check `rocm services logs <id>`, then `rocm services restart <id> --yes` once the cause \
         is fixed."
    );
    bail!(message);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(service_id: &str) -> FailedServiceRestart {
        FailedServiceRestart {
            service_id: service_id.to_owned(),
            error: "restart refused".to_owned(),
        }
    }

    #[test]
    fn an_activation_that_restarted_nothing_audits_as_plain_info() {
        let services = RuntimeServiceReconciliation::default();
        assert_eq!(
            service_restart_audit_outcome(&services),
            ("info", String::new())
        );
    }

    #[test]
    fn restarts_that_all_succeeded_audit_as_info_naming_the_count() {
        let services = RuntimeServiceReconciliation {
            restarted: vec!["svc-a".to_owned(), "svc-b".to_owned()],
            restart_attempted: true,
            ..RuntimeServiceReconciliation::default()
        };
        assert_eq!(
            service_restart_audit_outcome(&services),
            ("info", " services_restarted=2".to_owned())
        );
    }

    // The severity is the point: the command exits non-zero right after writing
    // this event, so an `info` line would leave the audit log claiming a clean
    // activation for a run that failed.
    #[test]
    fn a_failed_restart_audits_as_warn_naming_both_counts() {
        let services = RuntimeServiceReconciliation {
            restarted: vec!["svc-a".to_owned()],
            failed: vec![failed("svc-b")],
            restart_attempted: true,
            ..RuntimeServiceReconciliation::default()
        };
        assert_eq!(
            service_restart_audit_outcome(&services),
            (
                "warn",
                " services_restarted=1 services_restart_failed=1".to_owned()
            )
        );
    }

    // Every restart failing is the same severity as some failing, and still has
    // to report the zero rather than fall back to the `info` shape.
    #[test]
    fn restarts_that_all_failed_still_report_the_zero() {
        let services = RuntimeServiceReconciliation {
            failed: vec![failed("svc-a"), failed("svc-b")],
            restart_attempted: true,
            ..RuntimeServiceReconciliation::default()
        };
        assert_eq!(
            service_restart_audit_outcome(&services),
            (
                "warn",
                " services_restarted=0 services_restart_failed=2".to_owned()
            )
        );
    }
}
