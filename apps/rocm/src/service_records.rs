// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Removing and pruning managed-service records: the files one record owns
//! (`ServiceRecordArtifacts`, `service_record_artifacts`, which rebuilds every
//! path from the validated id and engine rather than trusting the record),
//! removing one record (`remove_managed_service_record`) and the `rocm services
//! prune` plan, render and apply steps (`build_service_prune_plan`,
//! `render_service_prune_plan`, `apply_service_prune_plan`,
//! `prune_managed_service_records`) together with the age gate behind them
//! (`DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS`, `service_prune_min_age_hours`).
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. This module reads record state through
//! `managed_service_status.rs` and `services_text.rs` (`load_managed_service`,
//! `validate_service_id`) and builds its advice text with `service_hints.rs`;
//! none of those modules depends on this one. `services_cmd.rs`'s `services()`
//! calls `remove_managed_service_record`, `service_prune_min_age_hours` and
//! `prune_managed_service_records`, and `main.rs`'s clap
//! `ServicesCommand::Prune` definition uses
//! `DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS`, which `main.rs` re-imports; all four
//! are `pub(crate)`. Every other item has no caller outside this module and
//! stays private.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching the shared `test_paths`, `plant_service_record` and
//! `plant_service_record_without_its_key` fixtures through `crate::tests::`.
//! `services_prune_waits_for_the_managed_launch_lock_before_sweeping` is the
//! test that pins the launch-lock behavior of `prune_managed_service_records`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord, process_is_running};

use crate::managed_service_status::{
    load_managed_services, managed_service_is_live, recorded_service_pids,
};
use crate::services_text::{load_managed_service, validate_service_id};
use crate::{cli_progress, cli_report, endpoint_keys, service_hints};

/// Default age gate for `rocm services prune`.
///
/// A record that has only just stopped is the one a user is most likely to still
/// want: `rocm services list --all` advertises `rocm services restart <id> --yes`
/// for exactly that record, and pruning it destroys both that affordance and the
/// log explaining why it died. Defaulting to a day means a bulk cleanup reclaims
/// the accumulated history without swallowing the failure the user is currently
/// looking at. `--older-than-hours 0` opts out.
pub(crate) const DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS: u64 = 24;

/// The on-disk files one managed-service record owns.
///
/// Note `engine_state` lives *outside* `services_dir`, under
/// `<data>/engines/<engine>/state/`, so deleting the two files in `services_dir`
/// leaves it behind — that is how orphaned engine state accumulates today.
///
/// Every path is rebuilt here from [`AppPaths`] plus the *validated* id and
/// engine, deliberately **not** read from the record's own `manifest_path` /
/// `log_path` / `engine_state_path` fields. Those are deserialized from a JSON
/// file under `~/.rocm` that anything can write, and this is the one code path
/// that deletes what they name.
#[derive(Debug, Clone)]
struct ServiceRecordArtifacts {
    manifest: PathBuf,
    log: PathBuf,
    engine_state: PathBuf,
    endpoint_key: PathBuf,
}

impl ServiceRecordArtifacts {
    /// Every artifact, in the order they are reported and deleted.
    fn paths(&self) -> [&Path; 4] {
        [
            &self.manifest,
            &self.log,
            &self.engine_state,
            &self.endpoint_key,
        ]
    }
}

/// Validate a manifest-supplied engine name as a single filesystem path
/// component.
///
/// [`AppPaths::service_engine_state_path`] joins the engine name into a path
/// this module deletes, so an engine of `../../..` in a hand-edited manifest
/// would otherwise aim the removal outside `<data>/engines`. Reuses the same
/// rules as [`validate_service_id`] — `ServiceId` is the repo's single source of
/// truth for "safe as one path component", and nothing about those rules is
/// specific to service ids.
fn validate_engine_component(engine: &str) -> Result<()> {
    rocm_core::ServiceId::new(engine).with_context(|| {
        format!("managed service engine `{engine}` is not a safe path component")
    })?;
    Ok(())
}

fn service_record_artifacts(
    paths: &AppPaths,
    service_id: &str,
    engine: &str,
) -> Result<ServiceRecordArtifacts> {
    validate_service_id(service_id)?;
    validate_engine_component(engine)?;
    Ok(ServiceRecordArtifacts {
        manifest: paths.service_manifest_path(service_id),
        log: paths.service_log_path(service_id),
        engine_state: paths.service_engine_state_path(engine, service_id),
        endpoint_key: endpoint_keys::endpoint_key_file_path(paths, service_id),
    })
}

/// Delete every artifact that exists, collecting failures instead of stopping at
/// the first one.
///
/// A missing file is not an error: a record whose log was already deleted by
/// hand (the workaround this command replaces) must still be removable. An
/// unremovable one is returned as a message rather than propagated, because
/// `prune` walks many records and aborting mid-loop would throw away the
/// rendered plan and the audit event covering everything already deleted in the
/// same run — losing the record of a destructive action exactly when something
/// went wrong.
fn try_remove_service_record_artifacts(
    artifacts: &ServiceRecordArtifacts,
) -> (Vec<PathBuf>, Vec<String>) {
    let mut removed = Vec::new();
    let mut failures = Vec::new();
    for path in artifacts.paths() {
        match fs::remove_file(path) {
            Ok(()) => removed.push(path.to_path_buf()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => failures.push(format!("{}: {error}", path.display())),
        }
    }
    (removed, failures)
}

/// Delete every artifact that exists, returning the paths actually removed.
///
/// The strict form, for `remove`, which owns exactly one record and has no
/// partial progress to report: any file it could not delete is the command
/// failing. Every artifact is still attempted first, so a single stubborn file
/// does not strand the other three.
fn remove_service_record_artifacts(artifacts: &ServiceRecordArtifacts) -> Result<Vec<PathBuf>> {
    let (removed, failures) = try_remove_service_record_artifacts(artifacts);
    if !failures.is_empty() {
        bail!("failed to remove {}", failures.join("; "));
    }
    Ok(removed)
}

/// Whether a record is too alive to delete: the CLI reads it as live, *or* a
/// process it recorded is still running.
///
/// The status string alone is not enough for a destructive command.
/// `refresh_from_engine_state` adopts `failed` straight from the engine's own
/// state file, and `refresh_managed_service_runtime_liveness` then returns early
/// for any non-live status without ever consulting the recorded pids — so a
/// server whose engine reported failure while its process is still up reads as
/// removable, and deleting it takes the log and the 0600 endpoint key out from
/// under a process that may still be serving.
pub(crate) fn managed_service_record_is_in_use(record: &ManagedServiceRecord) -> bool {
    managed_service_is_live(record)
        || recorded_service_pids(record)
            .iter()
            .any(|pid| process_is_running(*pid))
}

/// Remove one non-running local server record and everything it owns.
///
/// Refuses a live record rather than stopping it: a removal that silently killed
/// a serving process would be a very different action from the one the user
/// asked for. Liveness is read from the record `load_managed_service` returns,
/// which has already been refreshed against the engine state file and the real
/// processes — a manifest still saying `ready` for a dead PID has demoted to
/// `stopped` by then, and must stay removable.
pub(crate) fn remove_managed_service_record(
    paths: &AppPaths,
    service_id: &str,
    yes: bool,
) -> Result<String> {
    validate_service_id(service_id)?;
    let record = load_managed_service(paths, service_id)?;
    // Checked before `--yes` on purpose: for a running server "stop it first" is
    // the actionable error, and repeating the command with --yes must not be the
    // advice a user takes away from it.
    if managed_service_record_is_in_use(&record) {
        bail!(
            "local server `{service_id}` is {} and cannot be removed while it is running.\n\nTry: {}",
            record.status,
            service_hints::service_stop_hint(service_id)
        );
    }
    if !yes {
        bail!(
            "Removing local server record `{service_id}` requires --yes.\n\nTry: {}",
            service_hints::service_action_hint("remove", service_id)
        );
    }

    let artifacts = service_record_artifacts(paths, &record.service_id, &record.engine)?;
    let mut output = String::new();
    // Printed *before* the delete, because after it there is nothing left to
    // point at: this is the last moment the log path is useful.
    let _ = writeln!(
        output,
        "Removing local server record `{service_id}` (status: {}).",
        record.status
    );
    let _ = writeln!(output, "  log: {}", artifacts.log.display());
    let _ = writeln!(
        output,
        "After this, `rocm services logs {service_id}` and `{}` no longer work.",
        service_hints::service_action_hint("restart", service_id)
    );
    let _ = writeln!(output);

    let removed = remove_service_record_artifacts(&artifacts)?;
    let mut report = cli_report::ActionReport::new("Local server record removed")
        .detail("service", &record.service_id)
        .detail("engine", &record.engine)
        .detail("files removed", removed.len());
    for path in &removed {
        report = report.detail("removed", path.display());
    }
    let _ = write!(output, "{}", report.render());
    Ok(output)
}

/// One record `rocm services prune` would delete.
#[derive(Debug, Clone)]
struct ServicePruneEntry {
    service_id: String,
    engine: String,
    status: String,
    artifacts: ServiceRecordArtifacts,
}

#[derive(Debug, Default)]
struct ServicePrunePlan {
    remove: Vec<ServicePruneEntry>,
    /// Files whose record is already gone — chiefly engine state under
    /// `<data>/engines/<engine>/state/`, which no removal path has ever swept.
    orphans: Vec<PathBuf>,
    /// Human-readable reasons, one per record or file left in place.
    skipped: Vec<String>,
    /// How many records were skipped purely because they are still running.
    skipped_live: usize,
    /// How many were removable in every respect but too recent. Counted apart
    /// from `skipped` so the summary can name the flag that includes them: a
    /// silent "nothing to do" on a host full of fresh failures reads as the
    /// command being broken.
    skipped_recent: usize,
}

/// Modification time of `path`, or `None` when it cannot be read.
fn path_modified(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

/// Age of a modification time relative to `now`. `None` when the time is missing
/// or in the future, which is treated as "too new to touch".
fn age_from_modified(modified: Option<SystemTime>, now: SystemTime) -> Option<Duration> {
    now.duration_since(modified?).ok()
}

/// Whether a modification time is old enough to prune, given a threshold.
///
/// Fail-closed: a time that cannot be determined is kept, except when the
/// threshold is zero (the explicit "everything that is not running" opt-out).
fn prunable_by_modified(modified: Option<SystemTime>, min_age: Duration, now: SystemTime) -> bool {
    if min_age.is_zero() {
        return true;
    }
    age_from_modified(modified, now).is_some_and(|age| age >= min_age)
}

/// Whether `path` is old enough to prune, given a threshold in hours.
///
/// Only sound for files nothing in this run rewrites. Service manifests are
/// rewritten by [`load_managed_services`], so their times are snapshotted up
/// front by [`service_manifest_modified_times`] and gated with
/// [`prunable_by_modified`] instead.
fn prunable_by_age(path: &Path, min_age: Duration, now: SystemTime) -> bool {
    prunable_by_modified(path_modified(path), min_age, now)
}

/// Modification times of every `*.json` in the services directory, taken before
/// anything in this run can rewrite them.
///
/// [`load_managed_services`] refreshes each record against the engine state and
/// the real processes, and persists the result whenever that changes the status
/// — which is exactly what happens the first time anything observes that a
/// `ready`/`running`/`starting` server has died. That rewrite lands *after* the
/// `now` the age gate compares against, so a manifest read afterwards looks
/// newer than the run itself and the fail-closed branch keeps it forever. A host
/// whose servers died weeks ago but were never listed since would see
/// `rocm services prune --yes` remove nothing while reporting those records as
/// too recent to touch.
fn service_manifest_modified_times(paths: &AppPaths) -> HashMap<PathBuf, SystemTime> {
    let mut times = HashMap::new();
    let Ok(entries) = fs::read_dir(paths.services_dir()) else {
        return times;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        if let Some(modified) = path_modified(&path) {
            times.insert(path, modified);
        }
    }
    times
}

/// The age threshold one `prune` invocation runs with, in hours.
///
/// `--any-age` is the discoverable spelling of `--older-than-hours 0` — clap
/// rejects the two together — so this only has to collapse the flag. It is a
/// function rather than a line inside the dispatch `match` so a test can drive
/// the real parsed arguments through the same mapping the command uses.
pub(crate) const fn service_prune_min_age_hours(older_than_hours: u64, any_age: bool) -> u64 {
    if any_age { 0 } else { older_than_hours }
}

fn describe_hours(hours: u64) -> String {
    if hours == 1 {
        "1 hour".to_owned()
    } else {
        format!("{hours} hours")
    }
}

/// Files in `services_dir` and the engine state dirs that no longer belong to
/// any record.
///
/// A `<id>.log` or `<id>.endpoint-key` counts as orphaned only when `<id>.json`
/// is *absent from disk* — not merely absent from `records`. An unparseable
/// manifest is skipped by `load_managed_services`, and treating its siblings as
/// orphans would quietly delete the log of the one record a user most needs to
/// investigate. The corrupt manifest itself is never removed for the same
/// reason.
///
/// "No `<id>.json` beside it" is also exactly what a *launch in progress* looks
/// like: `serve` writes the 0600 `<id>.endpoint-key` before
/// `spawn_managed_engine_child` (in `service_launch.rs`) writes the first
/// `<id>.json`. Nothing about
/// the file distinguishes the two cases — under `--any-age` there is no age gate
/// left to ask — so this function does not try. It is only ever reached with the
/// managed-launch lock held, which excludes that window outright; see
/// [`prune_managed_service_records`].
///
/// `services_dir` also holds `launch.lock`, which is shared by every managed
/// launch rather than owned by one service (see
/// [`AppPaths::managed_launch_lock_path`]). Only the three per-service
/// extensions are considered, so the lock is never a candidate — including the
/// one this sweep is itself running under.
fn collect_service_orphans(paths: &AppPaths, min_age: Duration, now: SystemTime) -> Vec<PathBuf> {
    let services_dir = paths.services_dir();
    let mut orphans = Vec::new();

    let push_if_orphaned = |path: &Path, orphans: &mut Vec<PathBuf>| {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            return;
        };
        if services_dir.join(format!("{stem}.json")).exists() {
            return;
        }
        if prunable_by_age(path, min_age, now) {
            orphans.push(path.to_path_buf());
        }
    };

    if let Ok(entries) = fs::read_dir(&services_dir) {
        for path in entries.flatten().map(|entry| entry.path()) {
            if matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("log" | "endpoint-key")
            ) {
                push_if_orphaned(&path, &mut orphans);
            }
        }
    }

    // `<data>/engines/<engine>/state/<id>.json`. `engines/plugins` has no
    // `state` subdirectory, so it never yields candidates.
    if let Ok(engines) = fs::read_dir(paths.data_dir.join("engines")) {
        for engine_dir in engines.flatten().map(|entry| entry.path()) {
            let Some(engine) = engine_dir.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Ok(states) = fs::read_dir(paths.engine_state_dir(engine)) else {
                continue;
            };
            for path in states.flatten().map(|entry| entry.path()) {
                if path.extension().and_then(|value| value.to_str()) == Some("json") {
                    push_if_orphaned(&path, &mut orphans);
                }
            }
        }
    }

    orphans.sort();
    orphans
}

fn build_service_prune_plan(
    paths: &AppPaths,
    min_age: Duration,
    hours: u64,
    now: SystemTime,
) -> Result<ServicePrunePlan> {
    let mut plan = ServicePrunePlan::default();
    // Taken first, because `load_managed_services` below rewrites the manifest of
    // every record whose status its refresh corrects — see
    // `service_manifest_modified_times` for why reading the time afterwards makes
    // the age gate keep exactly the long-dead records prune exists to remove.
    let manifest_times = service_manifest_modified_times(paths);
    // `load_managed_services` refreshes each record against the engine state and
    // the real processes before returning it, so liveness below is read from the
    // refreshed view, never from the manifest as it was on disk.
    for record in load_managed_services(paths)? {
        if managed_service_record_is_in_use(&record) {
            plan.skipped_live += 1;
            plan.skipped.push(format!(
                "{} is {} - stop it first with `{}`",
                record.service_id,
                record.status,
                service_hints::service_stop_hint(&record.service_id)
            ));
            continue;
        }
        let artifacts = match service_record_artifacts(paths, &record.service_id, &record.engine) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                plan.skipped.push(format!("{}: {error}", record.service_id));
                continue;
            }
        };
        if !prunable_by_modified(
            manifest_times.get(&artifacts.manifest).copied(),
            min_age,
            now,
        ) {
            plan.skipped_recent += 1;
            plan.skipped.push(format!(
                "{} changed less than {} ago",
                record.service_id,
                describe_hours(hours)
            ));
            continue;
        }
        plan.remove.push(ServicePruneEntry {
            service_id: record.service_id.clone(),
            engine: record.engine.clone(),
            status: record.status.clone(),
            artifacts,
        });
    }
    plan.orphans = collect_service_orphans(paths, min_age, now);
    plan.remove
        .sort_by(|left, right| left.service_id.cmp(&right.service_id));
    plan.skipped.sort();
    Ok(plan)
}

fn render_service_prune_plan(plan: &ServicePrunePlan, hours: u64, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Local server record review");
    let _ = writeln!(output);
    if hours == 0 {
        let _ = writeln!(
            output,
            "Including every record that is not running, however recent."
        );
    } else {
        let _ = writeln!(
            output,
            "Only records and files untouched for at least {}.",
            describe_hours(hours)
        );
    }
    let _ = writeln!(output);

    if plan.remove.is_empty() && plan.orphans.is_empty() {
        let _ = writeln!(output, "Nothing would be removed.");
    }
    if plan.skipped_recent > 0 {
        let _ = writeln!(
            output,
            "{} record(s) changed less than {} ago and are kept; add --any-age to include them.",
            plan.skipped_recent,
            describe_hours(hours)
        );
    }
    if !plan.remove.is_empty() {
        let _ = writeln!(
            output,
            "{} local server record(s) would be removed:",
            plan.remove.len()
        );
        for entry in &plan.remove {
            let _ = writeln!(
                output,
                "  - {} (status: {}, engine: {})",
                entry.service_id, entry.status, entry.engine
            );
            for path in entry.artifacts.paths() {
                if path.exists() {
                    let _ = writeln!(output, "      {}", path.display());
                }
            }
        }
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "Their logs go with them, and `rocm services restart <service-id> --yes` stops working for them."
        );
    }
    if !plan.orphans.is_empty() {
        if !plan.remove.is_empty() {
            let _ = writeln!(output);
        }
        let _ = writeln!(
            output,
            "{} leftover file(s) with no local server record would be removed:",
            plan.orphans.len()
        );
        for path in &plan.orphans {
            let _ = writeln!(output, "  - {}", path.display());
        }
    }
    if !plan.skipped.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Left alone:");
        for skipped in &plan.skipped {
            let _ = writeln!(output, "  - {skipped}");
        }
    }
    if dry_run {
        let _ = writeln!(output);
        // "Nothing was removed", not "nothing was changed": building the plan
        // loads every record, and loading corrects a status that no longer
        // matches the real processes and persists that correction. No file is
        // deleted, which is the promise a dry run of a removal command makes.
        let _ = writeln!(
            output,
            "Nothing was removed. Re-run without --dry-run to remove."
        );
    }
    output
}

#[derive(Debug, Default)]
pub(crate) struct ServicePruneOutcome {
    pub(crate) text: String,
    pub(crate) removed_records: usize,
    pub(crate) removed_files: usize,
    /// Records left in place purely because they are still running.
    skipped_live: usize,
    skipped_recent: usize,
    /// Paths the run could not delete, one message each. Collected rather than
    /// propagated so the plan still prints and the audit event is still
    /// recorded; the caller turns a non-empty list into a failing exit.
    pub(crate) failures: Vec<String>,
}

/// Carry out everything `plan` names, recording what happened in `outcome`.
///
/// Split out of [`prune_managed_service_records`] rather than inlined there so a
/// test can hand it a plan entry whose record came back to life after the plan
/// was built. That window is the only reason the liveness re-check below exists,
/// and no test driving `prune` end to end can open it: one call builds the plan
/// and applies it, so a record is either live for both halves or dead for both.
fn apply_service_prune_plan(
    paths: &AppPaths,
    plan: &ServicePrunePlan,
    outcome: &mut ServicePruneOutcome,
) {
    for entry in &plan.remove {
        // Liveness was snapshotted while the plan was built. A
        // `rocm services restart <id> --yes` landing in that window would have
        // its log and 0600 endpoint key deleted out from under a live process,
        // so re-read the record immediately before touching its files —
        // `remove` narrows the same window by loading the record it deletes.
        // Neither closes it: the check and the delete are not atomic either way.
        if load_managed_service(paths, &entry.service_id)
            .is_ok_and(|record| managed_service_record_is_in_use(&record))
        {
            outcome.skipped_live += 1;
            let _ = writeln!(
                outcome.text,
                "  {} started again while this ran and was left alone.",
                entry.service_id
            );
            continue;
        }
        let (removed, failures) = try_remove_service_record_artifacts(&entry.artifacts);
        outcome.removed_files += removed.len();
        if failures.is_empty() {
            outcome.removed_records += 1;
        }
        outcome.failures.extend(failures);
    }
    for path in &plan.orphans {
        match fs::remove_file(path) {
            Ok(()) => outcome.removed_files += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => outcome
                .failures
                .push(format!("{}: {error}", path.display())),
        }
    }
}

/// Bulk-remove the local server records that are no longer running.
///
/// Unlike [`remove_managed_service_record`] a live record is *skipped*, not an
/// error: a bulk cleanup that aborted because one server happened to be serving
/// would be unusable on the hosts that need it most. The count is reported so
/// the skip is never silent.
///
/// # Why this takes the managed-launch lock
///
/// [`collect_service_orphans`] calls a file with no `<id>.json` beside it a
/// leftover, and that is also what a launch looks like mid-flight: `serve`
/// writes the 0600 `<id>.endpoint-key` at the top of the managed path, and the
/// first `<id>.json` only lands at `record.write()` inside
/// `spawn_managed_engine_child` (in `service_launch.rs`). Between those two writes a live server's
/// secret is indistinguishable from a leftover, and `--any-age` removes the age
/// gate that used to hide the window, so a concurrent
/// `rocm services prune --any-age --yes` deleted it.
///
/// `serve` already holds `managed_launch_lock_path` across the whole of that
/// window — [`select_gpu_indices_under_launch_lock`] acquires it, hands the
/// guard back to its caller, and `start_managed_service`/`run_attached_service`
/// only drop it once the record is persisted (see that function's own comment,
/// which this one must not drift from). The gap was simply that prune never
/// acquired it: the helper above was the single acquirer in the tree. Taking it
/// here closes the window deterministically, for any number of records, which no
/// age floor can — the window is not short and is not bounded, because
/// `spawn_managed_engine_child`'s idempotency guard runs `load_managed_services`
/// first, and that refreshes every `ready`/`running` record through a 750ms
/// listing plus an up-to-8s inference probe ([`rocm_core::INFERENCE_PROBE_TIMEOUT`]),
/// sequentially, over a record count the user controls.
///
/// **Scope: the whole operation, not just the scan.** Holding it across
/// [`build_service_prune_plan`] alone is very nearly enough — a launch already
/// under way blocks prune until its manifest exists, at which point its key is
/// no longer an orphan, and a launch starting after the scan has not written a
/// key yet so it cannot be in the plan. "Cannot be in the plan" leans on
/// `generate_service_id` minting a millisecond-unique id, though, which is a
/// property of an unrelated function and a backwards clock step would break.
/// Covering [`apply_service_prune_plan`] too makes the exclusion unconditional,
/// and costs nothing worth measuring next to the scan it already serializes:
/// the apply phase is `unlink` calls plus one manifest read per record.
///
/// That wider scope is *not* pinned by a test, and neither the unit test nor
/// `service-cleanup-07` fails if the guard is released after plan building. It
/// cannot be pinned at that level: the only behaviour the extra span changes is
/// the id-collision case above, which needs `generate_service_id` to mint an id
/// a previous run already used — i.e. a backwards clock step — and any test that
/// instead tried to catch a launch slipping in between the two phases would be
/// racing a microsecond-wide window, so it would pass on a timing-lucky run
/// rather than flake. Treat the scope as a deliberately conservative choice
/// argued from the code, not as a property under regression cover: narrowing it
/// will not turn anything red.
///
/// **What it costs.** Prune's own plan building calls `load_managed_services`,
/// so prune can now delay a launch for as long as its own scan runs, and a
/// launch already under way delays prune for as long as *it* runs. Neither
/// direction is bounded: the per-record worst case is the same 8.75s, over a
/// record count the user controls, and `FileLock::acquire` has no timeout. The
/// usual wait is milliseconds, because those are ceilings on probes that hang
/// rather than costs every record pays — but "usually imperceptible" is the
/// honest claim, not "a few seconds". Because the wait has no upper bound, it is
/// announced on screen ([`cli_progress::AnimatedSpinner`]) instead of leaving
/// the command silent. `--dry-run` takes the lock as well: a preview that
/// disagreed with what `--yes` would do is worth less than the launch it blocks.
///
/// **What `--dry-run` writes.** `FileLock::acquire` creates the lock file and
/// any missing parents, and `services` only calls `AppPaths::discover`, never
/// `ensure` — so a preview on a host that has never served now creates
/// `<data>/services/` and an empty `launch.lock`, and fails outright where the
/// data directory is not writable. Both are accepted. The lock file is a 0-byte
/// rendezvous point inside the CLI's own data directory, in the directory
/// `rocm serve` creates there anyway. And a data directory the CLI cannot write
/// is one no managed server could have written a record into, so the preview
/// that now errors had nothing to report; where records *do* exist, deleting
/// them needs write permission on the very directory the lock needs, so a
/// preview that still succeeded would be predicting a `--yes` run that cannot
/// run — the same kind of disagreement the paragraph above declines to ship.
///
/// **No nested acquire.** `FileLock::acquire` blocks with no `try_` variant, so
/// a second acquire on this same lock path while this process already holds
/// it would hang forever. Nothing reachable from either phase re-acquires
/// `managed_launch_lock_path`: the only other production call site on this
/// same path is `select_gpu_indices_under_launch_lock`, which is only ever
/// reached from `serve`, and `serve` never runs prune in-process or as a
/// subprocess (test code acquires this path directly too, but that is test
/// setup, not something reachable from this function's own call graph).
///
/// Separately, prune does cross a *different* lock — recording its audit
/// event takes that store's own `jsonl_lock_path` — but that call
/// (`record_cli_audit_event` in the `services` dispatch) happens only after
/// *this* function has already returned and `_launch_lock` has dropped, so
/// the two locks are never held at once and their acquisition order cannot
/// matter.
pub(crate) fn prune_managed_service_records(
    paths: &AppPaths,
    hours: u64,
    dry_run: bool,
    yes: bool,
) -> Result<ServicePruneOutcome> {
    if !dry_run && !yes {
        bail!(
            "Removing local server records requires --yes.\n\nTry: rocm services prune --dry-run\nThen: rocm services prune --yes"
        );
    }
    // Held until this function returns, so no managed launch can be between its
    // key write and its record write while the sweep looks at the directory.
    //
    // The acquire blocks with no timeout, and the thing it blocks behind is a
    // launch whose own duration is unbounded — so it is announced rather than
    // left as a silent hang, in the command a user reaches for precisely when a
    // launch has gone wrong. `AnimatedSpinner` rather than the caller-driven
    // `Spinner` because there is no loop here to tick one: the wait is a single
    // blocking syscall, so only a background ticker can keep it moving. It is
    // TTY-gated and erased on drop, so an uncontended prune — the normal case,
    // where this costs microseconds — leaves nothing behind, and piped or
    // redirected output never sees it at all.
    let _launch_lock = {
        let _waiting =
            cli_progress::AnimatedSpinner::start("Waiting for a launch already under way…");
        rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?
    };
    let min_age = Duration::from_secs(hours.saturating_mul(3600));
    let plan = build_service_prune_plan(paths, min_age, hours, SystemTime::now())?;
    let mut outcome = ServicePruneOutcome {
        text: render_service_prune_plan(&plan, hours, dry_run),
        skipped_live: plan.skipped_live,
        skipped_recent: plan.skipped_recent,
        ..ServicePruneOutcome::default()
    };
    if dry_run {
        return Ok(outcome);
    }

    apply_service_prune_plan(paths, &plan, &mut outcome);

    let _ = writeln!(outcome.text);
    let _ = write!(
        outcome.text,
        "{}",
        cli_report::ActionReport::new("Local server records removed")
            .detail("records removed", outcome.removed_records)
            .detail("files removed", outcome.removed_files)
            .detail("still running, left alone", outcome.skipped_live)
            .detail("too recent, kept", outcome.skipped_recent)
            .render()
    );
    // A bulk cleanup that silently keeps things is indistinguishable from one
    // that found nothing, and the records most worth reading are exactly the
    // ones this keeps.
    if outcome.skipped_recent > 0 {
        let _ = writeln!(
            outcome.text,
            "  Those are recent enough to still be worth reading: `rocm services logs <id>`.\n  \
             Run `rocm services prune --any-age --yes` to remove them too."
        );
    }
    if !outcome.failures.is_empty() {
        let _ = writeln!(outcome.text);
        let _ = writeln!(
            outcome.text,
            "{} file(s) could not be removed:",
            outcome.failures.len()
        );
        for failure in &outcome.failures {
            let _ = writeln!(outcome.text, "  - {failure}");
        }
        let _ = writeln!(
            outcome.text,
            "Re-running is safe: everything already removed stays removed."
        );
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::tests::{plant_service_record, plant_service_record_without_its_key, test_paths};
    use crate::{Cli, Command, ServicesCommand};

    #[test]
    fn services_remove_deletes_every_artifact_and_leaves_the_shared_lock() -> Result<()> {
        let (root, paths) = test_paths("services-remove-artifacts");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-remove-me", "failed", 999_999_999)?;
        // `launch.lock` lives in `services_dir` but belongs to every managed
        // launch, not to one service. Removing a record must not touch it.
        let lock = paths.managed_launch_lock_path();
        fs::write(&lock, "")?;

        let rendered = remove_managed_service_record(&paths, "svc-remove-me", true);
        let rendered = match rendered {
            Ok(text) => text,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let engine_state_exists = record.engine_state_path.exists();
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, "svc-remove-me").exists();
        let lock_exists = lock.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(!manifest_exists, "the details file must be deleted");
        assert!(!log_exists, "the log must be deleted");
        assert!(
            !engine_state_exists,
            "the engine state file must be deleted — it lives outside the services folder, \
             so deleting only the two files beside the manifest is what orphans it"
        );
        assert!(!key_exists, "the endpoint key file must be deleted");
        assert!(lock_exists, "the shared launch lock must be left alone");
        assert!(rendered.contains("Local server record removed"));
        assert!(rendered.contains("  files removed: 4"));
        // The log path is the last thing worth knowing before it disappears.
        assert!(
            rendered.contains(&format!("  log: {}", record.log_path.display())),
            "the log path must be printed before the delete:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn services_remove_refuses_a_running_record_and_names_stop() -> Result<()> {
        let (root, paths) = test_paths("services-remove-live");
        paths.ensure()?;
        // The current process is a guaranteed-live PID, so the liveness refresh
        // keeps this record in a running state.
        plant_service_record(&paths, "svc-live", "ready", std::process::id())?;

        let error = remove_managed_service_record(&paths, "svc-live", true)
            .expect_err("removing a running local server must fail");
        let manifest_exists = paths.service_manifest_path("svc-live").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(
            message.contains("cannot be removed while it is running"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("rocm services stop svc-live --yes"),
            "the error must name the stop command: {message}"
        );
        assert!(manifest_exists, "a refused removal must delete nothing");
        Ok(())
    }

    /// A manifest still saying `ready` for a process that is gone is the common
    /// case this command exists for. `load_managed_services` demotes it to
    /// `stopped` while reading, so the live guard has to run on the refreshed
    /// record — checking the status as it sits on disk would refuse forever.
    #[test]
    fn services_remove_accepts_a_stale_ready_record_that_refreshed_to_stopped() -> Result<()> {
        let (root, paths) = test_paths("services-remove-stale-ready");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-stale", "ready", 999_999_999)?;
        assert_eq!(
            serde_json::from_slice::<ManagedServiceRecord>(&fs::read(&record.manifest_path)?)?
                .status,
            "ready",
            "premise: the manifest on disk still claims `ready`"
        );

        let result = remove_managed_service_record(&paths, "svc-stale", true);
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        result?;
        assert!(!manifest_exists, "the stale record must be removable");
        Ok(())
    }

    #[test]
    fn services_remove_requires_yes() -> Result<()> {
        let (root, paths) = test_paths("services-remove-yes");
        paths.ensure()?;
        plant_service_record(&paths, "svc-needs-yes", "failed", 999_999_999)?;

        let error = remove_managed_service_record(&paths, "svc-needs-yes", false)
            .expect_err("removal without --yes must fail");
        let manifest_exists = paths.service_manifest_path("svc-needs-yes").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(message.contains("requires --yes"), "unexpected: {message}");
        assert!(
            message.contains("rocm services remove svc-needs-yes --yes"),
            "unexpected: {message}"
        );
        assert!(manifest_exists, "a refused removal must delete nothing");
        Ok(())
    }

    #[test]
    fn service_record_artifacts_reject_an_engine_that_escapes_the_state_dir() -> Result<()> {
        let (root, paths) = test_paths("services-remove-traversal");
        paths.ensure()?;

        let by_engine = service_record_artifacts(&paths, "svc-ok", "../../../etc");
        let by_id = service_record_artifacts(&paths, "../../etc/passwd", "vllm");
        let _ = fs::remove_dir_all(&root);

        assert!(
            by_engine.is_err(),
            "an engine name with `..` must not reach a path this code deletes"
        );
        assert!(by_id.is_err(), "a traversing service id must be rejected");
        Ok(())
    }

    #[test]
    fn services_prune_skips_running_records_and_reports_the_count() -> Result<()> {
        let (root, paths) = test_paths("services-prune-live");
        paths.ensure()?;
        plant_service_record(&paths, "svc-dead", "failed", 999_999_999)?;
        plant_service_record(&paths, "svc-running", "ready", std::process::id())?;

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let dead_exists = paths.service_manifest_path("svc-dead").exists();
        let running_exists = paths.service_manifest_path("svc-running").exists();
        let _ = fs::remove_dir_all(&root);

        assert!(!dead_exists, "the stopped record must be pruned");
        assert!(running_exists, "a running record must be left alone");
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(outcome.skipped_live, 1);
        assert!(
            outcome.text.contains("  still running, left alone: 1"),
            "prune must report how many records it skipped:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("rocm services stop svc-running --yes"),
            "the skip reason must name the stop command:\n{}",
            outcome.text
        );
        // ROCMAI-543: the skip line is built by `build_service_prune_plan` as
        // "<id> is <status> - stop it first with `<cmd>`" and printed verbatim.
        // Asserting the whole rendered plan rather than that one sentence also
        // covers the counts, the headings, and any line added to the plan later.
        assert!(
            outcome.text.is_ascii(),
            "`rocm services prune` output must be ASCII-only; non-ASCII punctuation renders \
             as mojibake on a Windows console using the OEM code page (ROCMAI-543):\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The engine state file lives outside `services_dir`, so hand-deleting the
    /// manifest and log — the workaround this command replaces — leaves it
    /// behind forever. Prune has to sweep it.
    #[test]
    fn services_prune_sweeps_engine_state_left_behind_by_a_deleted_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-orphans");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-orphaned", "failed", 999_999_999)?;
        // Exactly what the manual workaround leaves: the two files in the
        // services folder are gone, the engine state and endpoint key are not.
        fs::remove_file(&record.manifest_path)?;
        let orphan_state = record.engine_state_path;
        let orphan_log = record.log_path;
        let orphan_key = endpoint_keys::endpoint_key_file_path(&paths, "svc-orphaned");

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let state_exists = orphan_state.exists();
        let log_exists = orphan_log.exists();
        let key_exists = orphan_key.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !state_exists,
            "the orphaned engine state file must be swept:\n{}",
            outcome.text
        );
        assert!(!log_exists, "the orphaned log must be swept");
        assert!(!key_exists, "the orphaned endpoint key must be swept");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 3);
        Ok(())
    }

    /// The window the launch lock exists to close here: `serve` writes
    /// the 0600 endpoint key *before* the first manifest, so between those two
    /// writes a live server's key has no `<id>.json` beside it — the exact shape
    /// the sweep calls a leftover, and `--any-age` leaves no age gate to hide it
    /// behind. `serve` holds the managed-launch lock across the whole of that
    /// interval, so the fix is for `prune` to acquire it too.
    ///
    /// Shape of the test, since `FileLock::acquire` blocks with no `try_`
    /// variant and calling `prune` on the thread that holds the lock would
    /// simply deadlock: the main thread stands in for the launch and holds the
    /// lock with only the key written, a second thread runs the real
    /// `--any-age --yes` argv, and the main thread then does what the launch
    /// does next — writes the record — *before* releasing. So the sweep sees the
    /// directory only in its published state.
    ///
    /// Two independent assertions each fail on their own if `prune` stops
    /// acquiring the lock: the negative wait (prune returns while the lock is
    /// held, so it never waited for it) and the endpoint key (prune scans before
    /// the record lands and deletes it). The negative wait is reported first
    /// only because it names the cause; it is not load-bearing on its own, so
    /// deleting it as "timing-sensitive" still leaves a pure filesystem fact
    /// behind it. For that second assertion to mean anything, the record has to
    /// be published *without* rewriting the key — hence
    /// [`plant_service_record_without_its_key`] below rather than the full
    /// planter. The negative wait in turn cannot pass vacuously: the worker
    /// physically cannot report completion without first holding the lock the
    /// main thread has.
    #[test]
    fn services_prune_waits_for_the_managed_launch_lock_before_sweeping() -> Result<()> {
        use std::sync::mpsc;

        /// Long enough that a `prune` over an all-but-empty data dir would have
        /// finished many times over, short enough to keep the suite quick. Only
        /// ever compared against "did not finish", so a slow machine makes this
        /// more conclusive, never flakier.
        const WAIT_PROOF: Duration = Duration::from_millis(500);
        const ID: &str = "svc-launching";

        let (root, paths) = test_paths("services-prune-launch-lock");
        paths.ensure()?;
        // Write 1 of 2, byte for byte as `serve` does it at the top of the
        // managed path. No manifest yet: `spawn_managed_engine_child` has not
        // reached `record.write()`.
        endpoint_keys::store_endpoint_api_key(&paths, ID, "live-secret")?;
        assert!(
            !paths.service_manifest_path(ID).exists(),
            "premise: the manifest must not be written yet"
        );
        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (waited, published, finished) = std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).expect("signal about to prune");
                let outcome =
                    parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
                        .and_then(|(hours, dry_run, yes)| {
                            prune_managed_service_records(&paths, hours, dry_run, yes)
                        });
                let _ = done_tx.send(
                    outcome
                        .map(|outcome| (outcome.removed_files, outcome.skipped_live, outcome.text))
                        .map_err(|error| error.to_string()),
                );
            });
            // The worker is running before the negative check below, so that
            // check is about the lock and not about scheduling latency.
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("prune thread started");
            let early = done_rx.recv_timeout(WAIT_PROOF);
            let waited = early.is_err();
            // Write 2 of 2, still under the lock: the launch publishes its
            // record, and the key stops looking like a leftover. `serve` does
            // this at `record.write()`, well before it drops the guard.
            // Only the record: the key was written above, by the launch, and
            // must not be re-planted here. `plant_service_record` would rewrite
            // it, and a rewritten key is present at the end of the test whether
            // or not the sweep deleted it — which is exactly how the two key
            // assertions below would stop being tripwires.
            let published =
                plant_service_record_without_its_key(&paths, ID, "starting", std::process::id())
                    .map(|_| ());
            // Nothing above this line may panic: the worker is blocked on this
            // guard, and `scope` would join it forever.
            drop(launch_lock);
            let finished = match early {
                Ok(finished) => finished,
                Err(_) => done_rx
                    .recv_timeout(Duration::from_mins(1))
                    .expect("prune completes once the launch lock is released"),
            };
            (waited, published, finished)
        });
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, ID).exists();
        let key_value = endpoint_keys::endpoint_api_key(&paths, ID);
        let _ = fs::remove_dir_all(&root);

        published.context("failed to publish the launch's record under the lock")?;
        assert!(
            waited,
            "prune finished while the managed-launch lock was held, so it never \
             acquired it — a launch between its key write and its record write \
             is still exposed:\n{finished:?}"
        );
        let (removed_files, skipped_live, text) =
            finished.map_err(|error| anyhow::anyhow!("prune failed: {error}"))?;
        assert!(
            key_exists,
            "the starting server's endpoint key must survive a concurrent \
             --any-age prune:\n{text}"
        );
        assert_eq!(
            key_value.as_deref(),
            Some("live-secret"),
            "the key the starting server needs must survive intact"
        );
        assert_eq!(
            removed_files, 0,
            "the published record makes its key a companion, not a leftover:\n{text}"
        );
        assert_eq!(
            skipped_live, 1,
            "the launch is live by the time prune reads the directory:\n{text}"
        );
        Ok(())
    }

    #[test]
    fn services_prune_dry_run_reports_the_plan_and_removes_nothing() -> Result<()> {
        let (root, paths) = test_paths("services-prune-dry-run");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-dry", "failed", 999_999_999)?;

        let outcome = prune_managed_service_records(&paths, 0, true, false);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let engine_state_exists = record.engine_state_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(manifest_exists, "--dry-run must not delete the record");
        assert!(engine_state_exists, "--dry-run must not delete anything");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        assert!(outcome.text.contains("- svc-dry (status: failed"));
        assert!(
            outcome
                .text
                .contains("Nothing was removed. Re-run without --dry-run to remove.")
        );
        Ok(())
    }

    /// The `--all` view advertises `rocm services restart <id> --yes` for every
    /// record it lists. Pruning a server that died a minute ago would destroy
    /// that affordance and the log explaining the failure, so the default age
    /// gate keeps it.
    #[test]
    fn services_prune_default_age_keeps_a_just_stopped_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a record written seconds ago must survive the default prune:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert!(
            outcome
                .text
                .contains("svc-fresh changed less than 24 hours ago"),
            "prune must say why it was kept:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The age gate reads the manifest's modification time, and
    /// `load_managed_services` rewrites that manifest the first time it observes
    /// that a `ready` server has died — a crash, a kill, a reboot. Read after
    /// that rewrite the record looks newer than the prune run itself, so the
    /// fail-closed branch keeps it: on a host whose servers died weeks ago and
    /// have not been listed since, `rocm services prune --yes` removed nothing
    /// at all and called every one of them too recent to touch.
    #[test]
    fn services_prune_default_age_removes_a_record_the_refresh_rewrote() -> Result<()> {
        let (root, paths) = test_paths("services-prune-refresh-rewrite");
        paths.ensure()?;
        // `ready` with a dead pid is the state that triggers the rewrite: the
        // liveness refresh demotes the status and persists the demotion.
        let record = plant_service_record(&paths, "svc-long-dead", "ready", 999_999_999)?;
        let month_ago = SystemTime::now() - Duration::from_hours(24 * 30);
        fs::File::options()
            .write(true)
            .open(&record.manifest_path)?
            .set_modified(month_ago)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !manifest_exists,
            "a record last written a month ago must be pruned even though the \
             liveness refresh rewrote its manifest during this run:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(
            outcome.skipped_recent, 0,
            "the refresh's own rewrite must not make a month-old record recent:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// A cleanup that keeps things silently is indistinguishable from one that
    /// found nothing, and what it keeps is exactly what a user debugging a fresh
    /// failure still wants. The count and the way to override it both have to be
    /// on screen.
    #[test]
    fn services_prune_says_how_many_it_kept_for_being_recent() -> Result<()> {
        let (root, paths) = test_paths("services-prune-recent-report");
        paths.ensure()?;
        plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let _ = fs::remove_dir_all(&root);

        assert_eq!(outcome.skipped_recent, 1);
        assert!(
            outcome.text.contains("  too recent, kept: 1"),
            "the summary must count what it kept:\n{}",
            outcome.text
        );
        assert!(
            outcome.text.contains("rocm services prune --any-age --yes"),
            "the summary must name the flag that includes them:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// Parse a real `rocm services prune` command line and return exactly what
    /// the dispatch at `ServicesCommand::Prune` would hand
    /// [`prune_managed_service_records`]. Nothing here re-implements the flag
    /// mapping: it runs [`service_prune_min_age_hours`], the same function the
    /// command uses, so a test driving this covers the wiring and not a copy of
    /// it.
    fn parse_services_prune_args(argv: &[&str]) -> Result<(u64, bool, bool)> {
        let cli = Cli::try_parse_from(argv)?;
        let Some(Command::Services {
            command:
                Some(ServicesCommand::Prune {
                    older_than_hours,
                    any_age,
                    dry_run,
                    yes,
                }),
        }) = cli.command
        else {
            bail!("{argv:?} did not parse as `services prune`");
        };
        Ok((
            service_prune_min_age_hours(older_than_hours, any_age),
            dry_run,
            yes,
        ))
    }

    /// `--any-age` is the reachable form of `--older-than-hours 0`: the summary
    /// points at it, so it has to actually take the record the default kept.
    ///
    /// Driven from the argument vector rather than by passing 0 by hand —
    /// otherwise this is just another call with `hours = 0` and the flag's only
    /// wiring, the collapse in [`service_prune_min_age_hours`], is never
    /// executed by any test.
    #[test]
    fn services_prune_any_age_removes_a_just_stopped_record() -> Result<()> {
        let (root, paths) = test_paths("services-prune-any-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let outcome =
            parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
                .and_then(|(hours, dry_run, yes)| {
                    assert_eq!(hours, 0, "--any-age must collapse to the zero-age rule");
                    assert!(!dry_run);
                    assert!(yes);
                    prune_managed_service_records(&paths, hours, dry_run, yes)
                });
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            !manifest_exists,
            "--any-age must remove the record the default keeps:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 1);
        assert_eq!(
            outcome.skipped_recent, 0,
            "nothing is 'too recent' once the age rule is off:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// Without the flag the very same parsed command line must keep the record,
    /// which is what makes the assertion above about `--any-age` and not about
    /// prune deleting things in general.
    #[test]
    fn services_prune_without_any_age_keeps_the_default_threshold() -> Result<()> {
        let (root, paths) = test_paths("services-prune-no-any-age");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-fresh", "failed", 999_999_999)?;

        let parsed = parse_services_prune_args(&["rocm", "services", "prune", "--yes"]);
        let outcome = parsed.and_then(|(hours, dry_run, yes)| {
            assert_eq!(
                hours, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS,
                "no --any-age means the default age rule still applies"
            );
            prune_managed_service_records(&paths, hours, dry_run, yes)
        });
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "the default must keep it:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.skipped_recent, 1);
        Ok(())
    }

    /// `collect_service_orphans`' doc comment leans on this: an *unparseable*
    /// manifest is skipped by `load_managed_services`, so widening the orphan
    /// rule from "no `<id>.json` on disk" to "no record in the list" would
    /// delete the log of the one record a user most needs to read, and the
    /// corrupt manifest with it. Nothing asserted that until now — the other
    /// tests only ever plant a fully absent manifest.
    #[test]
    fn services_prune_keeps_a_corrupt_manifest_and_its_siblings() -> Result<()> {
        let (root, paths) = test_paths("services-prune-corrupt");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-corrupt", "failed", 999_999_999)?;
        // Valid JSON, not a valid record: `serde_json::from_slice` fails, so
        // `load_managed_services` skips it without reporting an error.
        fs::write(&record.manifest_path, b"{\"service_id\": 12345}")?;
        let key_path = endpoint_keys::endpoint_key_file_path(&paths, "svc-corrupt");

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let state_exists = record.engine_state_path.exists();
        let key_exists = key_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a manifest that cannot be parsed must never be deleted:\n{}",
            outcome.text
        );
        assert!(
            log_exists,
            "the log of an unreadable record is exactly what a user needs:\n{}",
            outcome.text
        );
        assert!(state_exists, "the engine state must not look orphaned");
        assert!(key_exists, "the endpoint key must not look orphaned");
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        Ok(())
    }

    /// The status string is not proof of death. `refresh_from_engine_state`
    /// adopts `failed` straight from the engine's own state file and the
    /// liveness refresh then returns early for a non-live status, so a server
    /// whose engine reported failure while its process is still up would be
    /// removable — taking the log and the 0600 endpoint key of a live process.
    #[test]
    fn services_removal_refuses_a_failed_record_whose_process_is_still_alive() -> Result<()> {
        let (root, paths) = test_paths("services-remove-live-pid");
        paths.ensure()?;
        // This test process: a pid that is unambiguously running.
        let record = plant_service_record(&paths, "svc-zombie", "failed", std::process::id())?;

        let remove_error = remove_managed_service_record(&paths, "svc-zombie", true);
        let prune = prune_managed_service_records(&paths, 0, false, true);
        let prune = match prune {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        let message = remove_error
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("cannot be removed while it is running"),
            "a record with a live pid must be refused whatever its status: {message}"
        );
        assert!(manifest_exists, "nothing may be deleted:\n{}", prune.text);
        assert_eq!(prune.removed_records, 0);
        assert_eq!(
            prune.skipped_live, 1,
            "prune must count it as still running:\n{}",
            prune.text
        );
        Ok(())
    }

    /// A file `prune` cannot delete must be *reported*, not swallowed and not
    /// propagated: the plan, the per-record progress and the audit event all
    /// have to survive it, because a destructive command that loses its own
    /// account of what it deleted is worse than one that fails. The record whose
    /// file survived also must not be counted as removed.
    ///
    /// A non-empty directory standing where the engine state file belongs is the
    /// portable way to make `fs::remove_file` fail on both supported hosts; the
    /// premise is asserted rather than assumed.
    #[test]
    fn services_prune_reports_a_file_it_could_not_remove() -> Result<()> {
        let (root, paths) = test_paths("services-prune-undeletable");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-stuck", "failed", 999_999_999)?;
        let stuck = record.engine_state_path.clone();
        fs::remove_file(&stuck)?;
        fs::create_dir(&stuck)?;
        fs::write(stuck.join("held.json"), b"{}")?;
        assert!(
            fs::remove_file(&stuck).is_err(),
            "premise: {} must be undeletable by `remove_file`",
            stuck.display()
        );

        let outcome = prune_managed_service_records(&paths, 0, false, true);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let manifest_exists = record.manifest_path.exists();
        let stuck_exists = stuck.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(stuck_exists, "premise: the stuck path must survive");
        assert!(
            !manifest_exists,
            "one unremovable file must not strand the other three:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_files, 3);
        assert_eq!(
            outcome.removed_records, 0,
            "a record that still owns a file on disk is not removed:\n{}",
            outcome.text
        );
        assert_eq!(
            outcome.failures.len(),
            1,
            "the failure must be collected so the caller can fail the command:\n{}",
            outcome.text
        );
        assert!(
            outcome.failures[0].contains(&stuck.display().to_string()),
            "the failure must name the path: {:?}",
            outcome.failures
        );
        assert!(
            outcome
                .text
                .contains("1 local server record(s) would be removed"),
            "the plan must still be rendered:\n{}",
            outcome.text
        );
        assert!(
            outcome.text.contains("1 file(s) could not be removed:"),
            "the run must say what it could not delete:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("Re-running is safe: everything already removed stays removed."),
            "the user needs to know a re-run is not destructive twice over:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// The window the pre-delete liveness re-check exists to narrow: the plan is
    /// built from a snapshot, and a `rocm services restart <id> --yes` landing
    /// between that snapshot and the delete would otherwise have its log and its
    /// 0600 endpoint key deleted out from under a serving process.
    ///
    /// Driven through [`apply_service_prune_plan`] with a hand-built plan
    /// because the window cannot be opened from outside: `prune` builds and
    /// applies the plan in one call, so a record is either live for both halves
    /// or dead for both. The two `skipped_live` assertions elsewhere in this
    /// module both plant an already-live record, which is satisfied by the
    /// *plan-building* skip and never reaches this branch.
    #[test]
    fn services_prune_leaves_a_record_that_restarted_after_the_plan_was_built() -> Result<()> {
        let (root, paths) = test_paths("services-prune-relaunched");
        paths.ensure()?;
        // Live at apply time. The current process is a guaranteed-live pid.
        let record = plant_service_record(&paths, "svc-relaunched", "ready", std::process::id())?;
        // The entry a plan built moments earlier, while the record was still
        // stopped, would carry into the delete loop.
        let artifacts = match service_record_artifacts(&paths, "svc-relaunched", "vllm") {
            Ok(artifacts) => artifacts,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let plan = ServicePrunePlan {
            remove: vec![ServicePruneEntry {
                service_id: "svc-relaunched".to_owned(),
                engine: "vllm".to_owned(),
                status: "stopped".to_owned(),
                artifacts,
            }],
            ..ServicePrunePlan::default()
        };

        let mut outcome = ServicePruneOutcome::default();
        apply_service_prune_plan(&paths, &plan, &mut outcome);
        let manifest_exists = record.manifest_path.exists();
        let log_exists = record.log_path.exists();
        let key_exists = endpoint_keys::endpoint_key_file_path(&paths, "svc-relaunched").exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            manifest_exists,
            "a record that came back to life must not be deleted:\n{}",
            outcome.text
        );
        assert!(
            log_exists,
            "the log of a live process must survive:\n{}",
            outcome.text
        );
        assert!(
            key_exists,
            "the 0600 endpoint key of a live process must survive:\n{}",
            outcome.text
        );
        assert_eq!(outcome.removed_records, 0);
        assert_eq!(outcome.removed_files, 0);
        assert_eq!(
            outcome.skipped_live, 1,
            "the re-check's skip must be counted like any other:\n{}",
            outcome.text
        );
        assert!(
            outcome
                .text
                .contains("  svc-relaunched started again while this ran and was left alone."),
            "the skip must be on screen, not silent:\n{}",
            outcome.text
        );
        Ok(())
    }

    /// `prunable_by_modified` promises to fail *closed*: a modification time
    /// that yields no age — a file stamped in the future by clock skew or a
    /// stray `touch` — is kept, and only the explicit zero-age opt-out overrides
    /// that. Neither half was asserted anywhere: every other test plants a
    /// readable past time, for which `is_some_and` and `is_none_or` agree and
    /// the `min_age.is_zero()` early return is unreachable.
    #[test]
    fn services_prune_keeps_a_future_dated_record_until_any_age() -> Result<()> {
        let (root, paths) = test_paths("services-prune-future-mtime");
        paths.ensure()?;
        let record = plant_service_record(&paths, "svc-future", "failed", 999_999_999)?;
        // `now.duration_since(future)` is an error, so `age_from_modified` has
        // no age to compare against the threshold.
        let backdate = |to: SystemTime| -> Result<()> {
            fs::File::options()
                .write(true)
                .open(&record.manifest_path)?
                .set_modified(to)?;
            Ok(())
        };
        let next_year = SystemTime::now() + Duration::from_hours(24 * 365);
        backdate(next_year)?;
        assert!(
            age_from_modified(path_modified(&record.manifest_path), SystemTime::now()).is_none(),
            "premise: a future-stamped manifest must have no age"
        );

        let kept =
            prune_managed_service_records(&paths, DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS, false, true);
        let kept = match kept {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let kept_manifest = record.manifest_path.exists();

        // Re-stamped so the second run really does meet the no-age case rather
        // than a time the first run's refresh may have rewritten to `now`.
        // Ignored rather than `?`-ed: if the first run wrongly deleted the
        // manifest there is nothing left to stamp, and the `kept_manifest`
        // assertion below has to be what reports that, not an "os error 2".
        let _ = backdate(next_year);
        let taken = parse_services_prune_args(&["rocm", "services", "prune", "--any-age", "--yes"])
            .and_then(|(hours, dry_run, yes)| {
                assert_eq!(hours, 0, "--any-age must collapse to the zero-age rule");
                prune_managed_service_records(&paths, hours, dry_run, yes)
            });
        let taken = match taken {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = fs::remove_dir_all(&root);
                return Err(error);
            }
        };
        let taken_manifest = record.manifest_path.exists();
        let _ = fs::remove_dir_all(&root);

        assert!(
            kept_manifest,
            "a time that cannot be aged must fail closed under the default rule:\n{}",
            kept.text
        );
        assert_eq!(kept.removed_records, 0);
        assert_eq!(
            kept.skipped_recent, 1,
            "the keep must be reported, not silent:\n{}",
            kept.text
        );
        assert!(
            !taken_manifest,
            "--any-age is the opt-out that takes it anyway:\n{}",
            taken.text
        );
        assert_eq!(taken.removed_records, 1);
        assert_eq!(taken.skipped_recent, 0);
        Ok(())
    }

    #[test]
    fn services_prune_requires_yes_unless_dry_run() -> Result<()> {
        let (root, paths) = test_paths("services-prune-yes");
        paths.ensure()?;
        plant_service_record(&paths, "svc-prune-yes", "failed", 999_999_999)?;

        let error = prune_managed_service_records(&paths, 0, false, false)
            .expect_err("prune without --yes must fail");
        let manifest_exists = paths.service_manifest_path("svc-prune-yes").exists();
        let _ = fs::remove_dir_all(&root);

        let message = error.to_string();
        assert!(message.contains("requires --yes"), "unexpected: {message}");
        assert!(
            message.contains("rocm services prune --dry-run"),
            "the error must name the preview command: {message}"
        );
        assert!(manifest_exists, "a refused prune must delete nothing");
        Ok(())
    }
}
