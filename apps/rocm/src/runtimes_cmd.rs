// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm runtimes` command: its dispatch, activate/rollback, and the runtime
//! list, which `runtimes_listing` computes once into a `RuntimesListing` that
//! prints either as the text table or, with `--json`, as that value serialized
//! (`docs/json-output.md`). Import and adopt (and recovering the runtime
//! `rocm setup` created) live in `runtime_adopt.rs`, and uninstall in
//! `runtime_uninstall.rs`. Registry selection and manifest validation live in
//! `runtime_registry.rs`; the names other modules import from here
//! (`current_runtime_manifest`, `select_runtime_manifest`,
//! `validate_runtime_manifest_for_activation`, `runtime_usability_status`,
//! and the registry path helpers) are re-exported from this module.
//!
//! `runtimes(command)` is the entry point `main.rs`'s `dispatch()` calls for
//! `rocm runtimes`, imported there via
//! `use crate::runtimes_cmd::{runtimes, ...};`.
//! `RuntimesCommand` lives in `cli.rs` (part of the `Command`/`Cli` clap
//! definition) and is reached via `use crate::RuntimesCommand`, a re-import in
//! `main.rs`, the same way `engines_cmd.rs` reaches `EnginesCommand`.
//! `RuntimeActivationResult` stays
//! `pub(crate)`: it is read by `append_update_activate_summary` in
//! `update_cmd.rs`'s `apply_runtime_update` flow. The uninstall plan and apply
//! code lives in `runtime_uninstall.rs`, and the active-runtime marker model and
//! `write_active_runtime_marker` in `runtime_registry.rs`; activation and
//! rollback here write the marker through it, as uninstall does when it drops a
//! rollback target. A failed activation's restore and three removals touch the
//! marker without it. The restore puts the earlier bytes back with
//! `rocm_core::atomic_write::write_file_atomically`, or removes the marker when
//! there was none before. Uninstalling the runtime in use removes it, and
//! `rocm config set-default-runtime` and `rocm config clear-default-runtime`
//! remove it and ignore a failed removal.
//! `uninstall_runtime`, `should_remove_runtime_install_root`,
//! `InstallRootDecision`, `ActiveRuntimeMarker` and `active_runtime_marker_path`
//! are re-exported from here for `storage.rs` and `main.rs`.
//! `recover_setup_runtime_registration`, `newest_installed_runtime_manifest`
//! and `setup_runtime_paths` (from `runtime_adopt.rs`) and
//! `normalize_path_for_compare` (from `runtime_registry.rs`) are re-exported
//! from here for the modules that import them from this one. The
//! SDK-install-finalization/torch-alignment/dependency bundle
//! (`finish_sdk_install` and friends) lives in the sibling
//! `runtime_install_finalize.rs`, which calls into this module (for
//! `activate_runtime`, `newest_installed_runtime_manifest` and
//! `recover_setup_runtime_registration`); this module reaches back into it
//! only from its own test module.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ConfigLock, RocmCliConfig, require_yes_or_interactive};
use serde::Serialize;

use crate::runtime_adopt::{
    AdoptRuntimeOptions, adopt_runtime_from_python_options, import_runtime_manifest,
};
pub(crate) use crate::runtime_adopt::{
    newest_installed_runtime_manifest, recover_setup_runtime_registration, setup_runtime_paths,
};
pub(crate) use crate::runtime_registry::{
    ActiveRuntimeMarker, active_runtime_marker_path, current_runtime_manifest,
    default_runtime_id_matches, normalize_path_for_compare, runtime_keys_text,
    runtime_manifest_path, runtime_registry_dir, runtime_usability_status, select_runtime_manifest,
    validate_runtime_manifest_for_activation,
};
use crate::runtime_registry::{
    RuntimeProvenance, unusable_status_text, write_active_runtime_marker,
};
pub(crate) use crate::runtime_uninstall::{
    InstallRootDecision, lemonade_record_report_line, should_remove_runtime_install_root,
    uninstall_runtime,
};
use crate::runtime_uninstall::{
    RuntimeUninstallConfirmation, apply_runtime_uninstall,
    confirm_and_revalidate_runtime_uninstall, plan_runtime_uninstall, print_runtime_uninstall_plan,
};
use crate::therock;
use crate::uninstall_plan::confirm_uninstall;
use crate::{
    ROLLBACK_RECOVERY_HINT, RuntimesCommand, cli_report, record_cli_audit_event, runtime_services,
};

/// Refuse `--restart-services` without `--yes`, naming the command the user
/// ran (`activate <runtime>` or `rollback`) in the retry hint, so following it
/// repeats the same activation or rollback with the approval added.
pub(crate) fn ensure_runtime_service_restart_approved(command: &RuntimesCommand) -> Result<()> {
    match command {
        RuntimesCommand::Activate {
            runtime,
            restart_services,
            yes,
        } => runtime_services::ensure_service_restart_approved(
            *restart_services,
            *yes,
            &format!("activate {runtime}"),
        ),
        RuntimesCommand::Rollback {
            restart_services,
            yes,
        } => runtime_services::ensure_service_restart_approved(*restart_services, *yes, "rollback"),
        RuntimesCommand::List { .. }
        | RuntimesCommand::Uninstall { .. }
        | RuntimesCommand::Import { .. }
        | RuntimesCommand::Adopt { .. } => Ok(()),
    }
}

pub(crate) fn runtimes(command: Option<RuntimesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    let command = command.unwrap_or(RuntimesCommand::List { json: false });
    // Checked before anything is written: a refused restart must leave the
    // previously active runtime in place, not switch the runtime and then
    // decline the half of the job the user actually asked for.
    ensure_runtime_service_restart_approved(&command)?;

    match command {
        RuntimesCommand::List { json } => {
            cli_report::print_json_or_text(json, &runtimes_listing(&paths, &config)?)?;
        }
        RuntimesCommand::Activate {
            runtime,
            restart_services,
            yes: _,
        } => {
            let mut result = activate_runtime(&paths, &mut config, &runtime)?;
            if restart_services {
                runtime_services::restart_stale_runtime_services(
                    &paths,
                    &result.runtime_key,
                    &mut result.services,
                );
            }
            println!("runtime activated");
            println!("  runtime_id: {}", result.runtime_id);
            println!("  runtime_key: {}", result.runtime_key);
            println!(
                "  changed_from_runtime_key: {}",
                changed_from_runtime_key_text(&result)
            );
            print!(
                "{}",
                runtime_services::render_runtime_service_reconciliation(
                    &result.services,
                    &result.runtime_key
                )
            );
            if result.previous_runtime_key.is_some() {
                println!("{ROLLBACK_RECOVERY_HINT}");
            }
            println!("  marker: {}", active_runtime_marker_path(&paths).display());
            println!("  config: {}", paths.config_path().display());
            // Severity and detail come from the restart outcome, not from the
            // switch alone. The switch did succeed, so "activated" is true — but
            // the command is about to exit non-zero, and an audit line reading
            // `info: activated` is where someone reconstructing the incident
            // would stop looking.
            let (severity, detail) =
                runtime_services::service_restart_audit_outcome(&result.services);
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_activate",
                severity,
                format!(
                    "activated runtime_key={} runtime_id={}{detail}",
                    result.runtime_key, result.runtime_id
                ),
                None,
            );
            runtime_services::bail_on_failed_service_restarts(&result.services)?;
        }
        RuntimesCommand::Rollback {
            restart_services,
            yes: _,
        } => {
            let mut result = rollback_runtime(&paths, &mut config)?;
            if restart_services {
                runtime_services::restart_stale_runtime_services(
                    &paths,
                    &result.runtime_key,
                    &mut result.services,
                );
            }
            println!("runtime rolled back");
            println!("  runtime_id: {}", result.runtime_id);
            println!("  runtime_key: {}", result.runtime_key);
            println!(
                "  changed_from_runtime_key: {}",
                result.previous_runtime_key.as_deref().unwrap_or("<unset>")
            );
            print!(
                "{}",
                runtime_services::render_runtime_service_reconciliation(
                    &result.services,
                    &result.runtime_key
                )
            );
            println!("  marker: {}", active_runtime_marker_path(&paths).display());
            println!("  config: {}", paths.config_path().display());
            let (severity, detail) =
                runtime_services::service_restart_audit_outcome(&result.services);
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_rollback",
                severity,
                format!(
                    "rolled back to runtime_key={} runtime_id={}{detail}",
                    result.runtime_key, result.runtime_id
                ),
                None,
            );
            runtime_services::bail_on_failed_service_restarts(&result.services)?;
        }
        RuntimesCommand::Uninstall {
            runtime,
            yes,
            dry_run,
        } => {
            let plan = plan_runtime_uninstall(&paths, &config, &runtime)?;
            print_runtime_uninstall_plan(&plan);

            if dry_run {
                println!("dry run: no changes made");
                return Ok(());
            }

            let plan = if require_yes_or_interactive(yes, "runtimes uninstall")? {
                plan
            } else {
                match confirm_and_revalidate_runtime_uninstall(&paths, plan, confirm_uninstall)? {
                    RuntimeUninstallConfirmation::Cancelled => {
                        println!("runtime uninstall cancelled");
                        return Ok(());
                    }
                    RuntimeUninstallConfirmation::Confirmed {
                        plan: revalidated,
                        config: reloaded,
                    } => {
                        config = *reloaded;
                        *revalidated
                    }
                }
            };
            let result = apply_runtime_uninstall(&paths, &mut config, plan)?;

            let mut report = cli_report::ActionReport::new("runtime removed")
                .detail("runtime_id", &result.runtime_id)
                .detail("runtime_key", &result.runtime_key)
                .detail("registry_removed", result.registry_path.display());
            match result.removed_install_root.as_ref() {
                Some(path) => {
                    report = report.detail("folder_removed", path.display());
                }
                None if result.read_only => {
                    report = report.detail("folder_removed", "no").detail(
                        "note",
                        "existing external runtime folder was left untouched",
                    );
                }
                None if result.manifest_mismatch => {
                    report = report.detail("folder_removed", "no").detail(
                        "note",
                        "local runtime manifest did not match the registry; the folder was \
                         left in place to avoid deleting the wrong install",
                    );
                }
                None => {
                    report = report.detail("folder_removed", "no");
                }
            }
            if let Some(line) = lemonade_record_report_line(&result.lemonade_record) {
                report = report.detail("lemonade", line);
            }
            if result.default_runtime_cleared {
                report = report
                    .detail("default_runtime", "cleared")
                    .detail("next step", "rocm runtimes activate <runtime_key>");
            }
            report = report.detail("config", paths.config_path().display());
            print!("{}", report.render());
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_uninstall",
                "info",
                format!(
                    "removed runtime_key={} runtime_id={} removed_install_root={} \
                     was_active={} default_runtime_cleared={}",
                    result.runtime_key,
                    result.runtime_id,
                    result
                        .removed_install_root
                        .as_ref()
                        .map_or_else(|| "none".to_owned(), |path| path.display().to_string()),
                    result.was_active,
                    result.default_runtime_cleared,
                ),
                None,
            );
        }
        RuntimesCommand::Import { manifest, replace } => {
            let imported = import_runtime_manifest(&paths, &manifest, replace)?;
            println!("runtime imported");
            println!("  runtime_id: {}", imported.runtime_id);
            println!("  runtime_key: {}", imported.runtime_key);
            println!("  mode: read-only");
            println!("  source: {}", manifest.display());
            println!(
                "  registry: {}",
                runtime_manifest_path(&paths, &imported.runtime_key).display()
            );
            println!(
                "  next step: rocm runtimes activate {}",
                imported.runtime_key
            );
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_import",
                "info",
                format!(
                    "imported read-only runtime_key={} runtime_id={} source={}",
                    imported.runtime_key,
                    imported.runtime_id,
                    manifest.display()
                ),
                None,
            );
        }
        RuntimesCommand::Adopt {
            python,
            root,
            runtime_id,
            runtime_key,
            channel,
            replace,
        } => {
            let adopted = adopt_runtime_from_python_options(
                &paths,
                AdoptRuntimeOptions {
                    python_input: python,
                    install_root: root,
                    runtime_id,
                    runtime_key,
                    channel,
                    replace,
                },
            )?;
            println!("runtime adopted");
            println!("  runtime_id: {}", adopted.runtime_id);
            println!("  runtime_key: {}", adopted.runtime_key);
            println!("  mode: read-only");
            println!(
                "  python_executable: {}",
                adopted.python_executable.as_deref().unwrap_or("<unset>")
            );
            println!("  root: {}", adopted.install_root.display());
            println!(
                "  registry: {}",
                runtime_manifest_path(&paths, &adopted.runtime_key).display()
            );
            println!(
                "  next step: rocm runtimes activate {}",
                adopted.runtime_key
            );
            record_cli_audit_event(
                &paths,
                "runtime",
                "runtime_adopt",
                "info",
                format!(
                    "adopted read-only runtime_key={} runtime_id={} root={}",
                    adopted.runtime_key,
                    adopted.runtime_id,
                    adopted.install_root.display()
                ),
                None,
            );
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeActivationResult {
    pub(crate) runtime_id: String,
    pub(crate) runtime_key: String,
    pub(crate) previous_runtime_key: Option<String>,
    /// The runtime asked for was already the active one, so nothing was
    /// switched and `previous_runtime_key` is the rollback target the earlier
    /// activation recorded, not a runtime this call left.
    pub(crate) reactivated: bool,
    /// What the live local servers were running on when this activation
    /// happened. Read from disk, never assumed: a fixed "running services keep
    /// their recorded runtime" note would be printed whether or not a single
    /// service existed.
    pub(crate) services: runtime_services::RuntimeServiceReconciliation,
}

/// What `changed_from_runtime_key:` says in the activation report.
///
/// A re-activation left nothing, so naming the rollback target it retained
/// would claim the user just came from it.
fn changed_from_runtime_key_text(result: &RuntimeActivationResult) -> &str {
    if result.reactivated {
        "<unchanged>"
    } else {
        result.previous_runtime_key.as_deref().unwrap_or("<unset>")
    }
}

/// The persisted state an activation replaces, captured before the first
/// write so a half-applied activation can be undone.
///
/// Activation writes two independent files — the config and the active-runtime
/// marker. Without this, a marker write that fails leaves the config naming the
/// new runtime while the marker still names the old one, and the marker feeds
/// both runtime resolution and the storage retention holds.
struct ActivationSnapshot {
    default_runtime_id: Option<String>,
    active_runtime_key: Option<String>,
    previous_runtime_key: Option<String>,
    marker: MarkerSnapshot,
}

/// What stood at the active-runtime marker path before an activation wrote it.
///
/// "No marker" and "a marker this process could not read" are deliberately
/// distinct. Collapsing them — which `fs::read(..).ok()` does — makes the
/// restore treat an unreadable marker as one it is entitled to delete, so a
/// permissions or IO fault on a marker whose bytes were never captured would
/// be destroyed by the very path that exists to put state back.
enum MarkerSnapshot {
    /// The marker's bytes, to be written back verbatim.
    Contents(Vec<u8>),
    /// There was no marker, so restoring means removing the one that was
    /// written over it.
    Absent,
    /// A marker was there but could not be read. Its bytes are not held, so
    /// the only honest restore is to leave whatever is there alone.
    Unreadable,
}

impl ActivationSnapshot {
    fn capture(paths: &AppPaths, config: &RocmCliConfig) -> Self {
        Self {
            default_runtime_id: config.default_runtime_id.clone(),
            active_runtime_key: config.active_runtime_key.clone(),
            previous_runtime_key: config.previous_runtime_key.clone(),
            marker: match fs::read(active_runtime_marker_path(paths)) {
                Ok(bytes) => MarkerSnapshot::Contents(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    MarkerSnapshot::Absent
                }
                Err(_) => MarkerSnapshot::Unreadable,
            },
        }
    }

    /// Put the caller's in-memory config back, without touching disk.
    ///
    /// Split out because the first write an activation makes is the config
    /// save through the lock: when that is what failed, nothing reached disk and
    /// saving again would only produce a second failure to explain, but the
    /// caller still holds a config describing a runtime that was never
    /// activated.
    fn restore_in_memory(&self, config: &mut RocmCliConfig) {
        config
            .default_runtime_id
            .clone_from(&self.default_runtime_id);
        config
            .active_runtime_key
            .clone_from(&self.active_runtime_key);
        config
            .previous_runtime_key
            .clone_from(&self.previous_runtime_key);
    }

    fn restore(
        self,
        paths: &AppPaths,
        lock: &ConfigLock,
        config: &mut RocmCliConfig,
    ) -> Result<()> {
        self.restore_in_memory(config);
        lock.save(config)?;
        let path = active_runtime_marker_path(paths);
        match self.marker {
            // Written through the same atomic helper the forward path uses, not
            // a bare `fs::write`. This runs only when something has already
            // failed, so it is the write least able to afford leaving a
            // truncated marker behind — which runtime resolution and the
            // storage retention holds would both read as "nothing is active".
            // The helper creates the `runtimes` folder when it is missing.
            MarkerSnapshot::Contents(bytes) => {
                rocm_core::atomic_write::write_file_atomically(&path, &bytes)
                    .with_context(|| format!("failed to restore {}", path.display()))?;
            }
            // Only a regular file is removed: whatever else may sit at that
            // path was there before this activation, so leaving it is what
            // "restore the previous state" means.
            MarkerSnapshot::Absent if path.is_file() => fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?,
            MarkerSnapshot::Absent | MarkerSnapshot::Unreadable => {}
        }
        Ok(())
    }
}

/// Persist the config an activation just mutated, putting the in-memory copy
/// back if that write fails.
///
/// The config save is the FIRST of an activation's two writes, so a failure
/// here leaves disk untouched — but not the in-memory settings, which already
/// name the runtime that was not activated and which [`activate_runtime`]
/// hands back to its caller. Callers keep using them (to report what is
/// active, for one), so unrestored settings would misreport an activation
/// that was refused.
fn save_activated_config(
    lock: &ConfigLock,
    config: &mut RocmCliConfig,
    snapshot: &ActivationSnapshot,
) -> Result<()> {
    lock.save(config).map_err(|error| {
        snapshot.restore_in_memory(config);
        error.context(
            "failed to record the active ROCm runtime; the previously active runtime is still \
             the active one",
        )
    })
}

/// Undo a half-applied activation and describe what the user is left with.
///
/// Mirrors the shape of `revalidate_runtime_uninstall_plan`: refuse the
/// operation with a message that says exactly which state is on disk and which
/// command puts it right.
fn restore_after_failed_activation(
    paths: &AppPaths,
    lock: &ConfigLock,
    config: &mut RocmCliConfig,
    snapshot: ActivationSnapshot,
    error: anyhow::Error,
) -> anyhow::Error {
    match snapshot.restore(paths, lock, config) {
        Ok(()) => discard_failed_marker_replacement(error).context(
            "failed to record the active ROCm runtime; the previously active runtime is still \
             the active one",
        ),
        Err(restore_error) => discard_failed_marker_replacement(error).context(format!(
            "failed to record the active ROCm runtime, and restoring the previous state failed \
             too ({restore_error:#}); the ROCm CLI config and the active runtime marker may now \
             disagree. Run `rocm runtimes list` to see both, then re-run `rocm runtimes activate \
             <runtime_key>` for the runtime you want to write them again"
        )),
    }
}

/// Drop the marker a failed activation left behind, and the advice to use it.
///
/// On Windows a marker publish can fail after taking the old marker away, and
/// the shared writer then keeps the new marker in its temp file and advises
/// renaming it into place (`rocm_core::atomic_write::KeptReplacement`). The
/// activation is being rolled back, so that file holds the marker of an
/// activation that failed, and renaming it would install that marker over
/// whatever the rollback restored. So the file is removed, and the error is
/// rebuilt without the advice. If the file cannot be removed, the error names
/// it and says to delete it instead.
fn discard_failed_marker_replacement(error: anyhow::Error) -> anyhow::Error {
    let Some(kept) = error.downcast_ref::<rocm_core::atomic_write::KeptReplacement>() else {
        return error;
    };
    let failure = format!(
        "failed to write active runtime marker {}: {}",
        kept.destination().display(),
        kept.cause()
    );
    match fs::remove_file(kept.kept()) {
        Ok(()) => anyhow::anyhow!(failure),
        Err(remove_error) if remove_error.kind() == std::io::ErrorKind::NotFound => {
            anyhow::anyhow!(failure)
        }
        Err(remove_error) => anyhow::anyhow!(
            "{failure}; {} holds the marker of the activation that failed and could not be \
             removed ({remove_error}): delete it, do not rename it into place",
            kept.kept().display()
        ),
    }
}

/// Marker shown beside the active runtime in `rocm runtimes list`. Every
/// place that renders this glyph MUST use this constant so the rendered
/// character and the legend text stay in sync.
const ACTIVE_RUNTIME_MARKER: &str = "*";
/// Marker shown beside the rollback-target runtime. See
/// [`ACTIVE_RUNTIME_MARKER`].
const ROLLBACK_RUNTIME_MARKER: &str = "-";

/// What `rocm runtimes list` reports, computed once and printed either as text
/// or, with `--json`, as this struct serialized.
///
/// Every per-runtime fact the text shows is a field here, so the text marker
/// and the JSON `active`/`rollback_target` flags come from the same
/// computation. The text adds only presentation: the marker legend, the
/// next-step hint for an empty registry, the build date it derives from
/// `version`, the `unusable (…)` wording around `unusable_reason`, and the word
/// `unrecorded` for a provenance fact the JSON writes as `null`. The JSON
/// form is additive-only: fields may be added, but none is renamed, removed or
/// given a new meaning (see `docs/json-output.md`).
#[derive(Debug, Serialize)]
pub(crate) struct RuntimesListing {
    /// `default_runtime_id` from the config, as configured; `null` when unset.
    /// Not necessarily the active runtime's `runtime_id`: an `active_runtime_key`
    /// that names a runtime takes precedence, and the id may differ in letter
    /// case or match nothing. The text form labels it `active_runtime_id:`, as
    /// `rocm examine` does.
    default_runtime_id: Option<String>,
    active_runtime_key: Option<String>,
    previous_runtime_key: Option<String>,
    /// Paths are written with `to_string_lossy`, so a path that is not valid
    /// UTF-8 is listed with U+FFFD in place of its bad bytes rather than
    /// failing the whole document.
    #[serde(serialize_with = "serialize_path_lossy")]
    registry: PathBuf,
    #[serde(serialize_with = "serialize_path_lossy")]
    marker: PathBuf,
    /// What is wrong with the configured active runtime setting, or `null`.
    /// See [`active_runtime_problem`] for when it is set beside an `active`
    /// runtime.
    active_problem: Option<ActiveRuntimeProblem>,
    /// Newest install first, ties broken by `runtime_key`, as the registry
    /// loader sorts them (`sort_manifests_newest_install_first`).
    runtimes: Vec<RuntimeListingEntry>,
}

/// What is wrong with the configured active runtime setting. The text form
/// prints it as the `active_status:` line.
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ActiveRuntimeProblem {
    /// `active_runtime_key` is set but names no registered runtime, ASCII
    /// letter case ignored. Reported even when `default_runtime_id` then
    /// resolves a runtime, which the listing marks `active`.
    MissingManifestForKey { runtime_key: String },
    /// No `active_runtime_key`; `default_runtime_id` matches no manifest's id,
    /// ASCII letter case ignored.
    MissingManifestForId { runtime_id: String },
    /// No `active_runtime_key`; `default_runtime_id` matches more than one
    /// manifest's id (letter case ignored), so none is active until one key is
    /// activated.
    AmbiguousRuntimeId {
        runtime_id: String,
        runtime_keys: Vec<String>,
    },
}

/// One registered runtime in [`RuntimesListing`].
#[derive(Debug, Serialize)]
struct RuntimeListingEntry {
    runtime_key: String,
    runtime_id: String,
    /// The version the manifest records; the text form adds the build date.
    version: String,
    format: String,
    family: String,
    mode: RuntimeMode,
    /// Serialized as the two JSON fields `status` and `unusable_reason`.
    #[serde(flatten)]
    usability: RuntimeUsability,
    toolchain: RuntimeToolchain,
    /// The runtime `current_runtime_manifest` resolves for this config: the
    /// one `rocm examine`, `rocm runtimes activate` and `rollback` treat as
    /// the active runtime.
    active: bool,
    /// The runtime `select_runtime_manifest` resolves `previous_runtime_key`
    /// to, which is the runtime `rocm runtimes rollback` selects (and then
    /// refuses if it is unusable).
    rollback_target: bool,
    #[serde(serialize_with = "serialize_path_lossy")]
    install_root: PathBuf,
    /// Where the runtime came from, as its manifest recorded it
    /// (`InstalledRuntimeManifest::provenance`). The text form prints it as the
    /// `provenance:` line under `install_root:`; the JSON form as an object
    /// with the same names, `null` where the text prints `unrecorded`.
    provenance: RuntimeListingProvenance,
}

/// An owned copy of a runtime's [`RuntimeProvenance`], so the listing can
/// outlive the manifests it was computed from. It holds the recorded values
/// (trimmed, `None` when unrecorded). The text prints them through
/// [`RuntimeProvenance::fields`], the same as `rocm examine`, and the JSON
/// writes them through [`RuntimeProvenance::recorded_fields`], so the two forms
/// take their names and order from one list.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeListingProvenance {
    channel: Option<String>,
    source: Option<String>,
    source_layout_generation: Option<String>,
    device_target: Option<String>,
}

impl RuntimeListingProvenance {
    fn recorded(provenance: &RuntimeProvenance<'_>) -> Self {
        Self {
            channel: provenance.channel.map(str::to_owned),
            source: provenance.source.map(str::to_owned),
            source_layout_generation: provenance.source_layout_generation.map(str::to_owned),
            device_target: provenance.device_target.map(str::to_owned),
        }
    }

    fn as_provenance(&self) -> RuntimeProvenance<'_> {
        RuntimeProvenance {
            channel: self.channel.as_deref(),
            source: self.source.as_deref(),
            source_layout_generation: self.source_layout_generation.as_deref(),
            device_target: self.device_target.as_deref(),
        }
    }
}

/// One JSON object holding each fact under the name the text prints it with,
/// as a string, or `null` when it was not recorded. Not the text's
/// `unrecorded` word: a script must not read that placeholder as a value.
impl Serialize for RuntimeListingProvenance {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap as _;
        let fields = self.as_provenance().recorded_fields();
        let mut object = serializer.serialize_map(Some(fields.len()))?;
        for (name, value) in fields {
            object.serialize_entry(name, &value)?;
        }
        object.end()
    }
}

/// A path as a JSON string, through `to_string_lossy`. serde's own `Path`
/// impl fails on a path that is not valid UTF-8, which would leave
/// `rocm runtimes list --json` printing nothing for a listing the text form
/// prints (with the same replacement, through `Path::display`).
fn serialize_path_lossy<S: serde::Serializer>(
    path: &std::path::Path,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&path.to_string_lossy())
}

/// Whether a runtime carries the compiler toolchain. The text and the JSON
/// print the same word, the one `toolchain_state_text` gives `rocm examine`.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum RuntimeToolchain {
    Included,
    Excluded,
}

impl RuntimeToolchain {
    const fn from_includes_devel(includes_devel: bool) -> Self {
        if includes_devel {
            Self::Included
        } else {
            Self::Excluded
        }
    }

    const fn as_str(self) -> &'static str {
        toolchain_state_text(matches!(self, Self::Included))
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum RuntimeMode {
    /// ROCm CLI installed it and may remove its folder.
    Managed,
    /// Imported or adopted; ROCm CLI never changes its folder.
    ReadOnly,
}

impl RuntimeMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Managed => "managed",
            Self::ReadOnly => "read-only",
        }
    }
}

/// Whether a runtime passes the manifest check that `rocm runtimes activate`
/// applies (`validate_runtime_manifest_for_activation`). Passing it does not
/// promise the activation succeeds: that can still refuse on an ambiguous
/// selector, unreadable server records or a failed config or marker write. An
/// unusable runtime always carries the reason, so the listing cannot report one
/// without saying why.
#[derive(Debug)]
enum RuntimeUsability {
    Ready,
    /// The human-readable message the manifest check failed with (not a
    /// stable code).
    Unusable {
        reason: String,
    },
}

impl RuntimeUsability {
    /// The JSON `status` value.
    const fn status(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unusable { .. } => "unusable",
        }
    }

    /// The JSON `unusable_reason` value: `None` exactly when ready.
    fn unusable_reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::Unusable { reason } => Some(reason),
        }
    }
}

/// Two fields, `status` (`ready` or `unusable`) and `unusable_reason` (`null`
/// when ready), flattened into each runtime's JSON object.
impl Serialize for RuntimeUsability {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut fields = serializer.serialize_struct("RuntimeUsability", 2)?;
        fields.serialize_field("status", self.status())?;
        fields.serialize_field("unusable_reason", &self.unusable_reason())?;
        fields.end()
    }
}

/// Recover the runtime `rocm setup` created, load the registry, and compute
/// everything `rocm runtimes list` reports, in either form.
pub(crate) fn runtimes_listing(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<RuntimesListing> {
    recover_setup_runtime_registration(paths, config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    Ok(listing_from_manifests(
        config,
        manifests,
        runtime_registry_dir(paths),
        active_runtime_marker_path(paths),
    ))
}

/// The listing of `manifests`, in the order given, for `config`. Split from
/// the registry read so every configuration can be tested in memory, including
/// keys that differ only in letter case, which one registry folder cannot hold
/// on a case-insensitive filesystem.
///
/// `active` and `rollback_target` are not decided by rules of the listing's
/// own: they are the runtimes the commands themselves resolve, by the same
/// functions, so the listing marks what those commands act on.
fn listing_from_manifests(
    config: &RocmCliConfig,
    manifests: Vec<therock::InstalledRuntimeManifest>,
    registry: PathBuf,
    marker: PathBuf,
) -> RuntimesListing {
    // Compared by position, not by key: keys can differ only in letter case,
    // and the resolver's choice between such twins is part of what is listed.
    let position = |resolved: &therock::InstalledRuntimeManifest| {
        manifests
            .iter()
            .position(|manifest| std::ptr::eq(manifest, resolved))
    };
    let active_index = current_runtime_manifest(config, &manifests).and_then(position);
    // `rocm runtimes rollback` selects `previous_runtime_key` this way; a value
    // it refuses (no match, or ambiguous) names no rollback target.
    let rollback_index = config
        .previous_runtime_key
        .as_deref()
        .and_then(|selector| select_runtime_manifest(&manifests, selector).ok())
        .and_then(position);
    let active_problem = active_runtime_problem(config, &manifests);

    let runtimes = manifests
        .into_iter()
        .enumerate()
        .map(|(index, manifest)| {
            // The manifest check `rocm runtimes activate` applies, so "ready"
            // here means the runtime passes that check, not that an activation
            // is sure to succeed (see `RuntimeUsability`).
            let usability = match validate_runtime_manifest_for_activation(&manifest) {
                Ok(()) => RuntimeUsability::Ready,
                Err(error) => RuntimeUsability::Unusable {
                    reason: error.to_string(),
                },
            };
            let mode = if manifest.read_only {
                RuntimeMode::ReadOnly
            } else {
                RuntimeMode::Managed
            };
            // The compiler is opt-in and `rocm update` reinstalls whatever this
            // says, so an install missing it should not be silent about that.
            // `rocm examine` reports the same thing for the active runtime.
            let toolchain = RuntimeToolchain::from_includes_devel(manifest.includes_devel());
            let provenance = RuntimeListingProvenance::recorded(&manifest.provenance());
            RuntimeListingEntry {
                runtime_key: manifest.runtime_key,
                runtime_id: manifest.runtime_id,
                version: manifest.version,
                format: manifest.format,
                family: manifest.family,
                mode,
                usability,
                toolchain,
                active: active_index == Some(index),
                rollback_target: rollback_index == Some(index),
                install_root: manifest.install_root,
                provenance,
            }
        })
        .collect();

    RuntimesListing {
        default_runtime_id: config.default_runtime_id.clone(),
        active_runtime_key: config.active_runtime_key.clone(),
        previous_runtime_key: config.previous_runtime_key.clone(),
        registry,
        marker,
        active_problem,
        runtimes,
    }
}

/// What is wrong with the configured active runtime setting, by the matching
/// `current_runtime_manifest` applies.
///
/// It reports the configuration, so it can disagree with `active` in exactly
/// one case: an `active_runtime_key` that names no runtime is reported as
/// `MissingManifestForKey` even when the resolver then falls back to
/// `default_runtime_id` and finds one, which is marked active. The commands act
/// on that runtime, so the listing marks it; the key is still dead, so the
/// listing says so. With no `active_runtime_key`, the problem comes from the
/// same `default_runtime_id_matches` the resolver falls back to, so it is set
/// exactly when nothing is active.
fn active_runtime_problem(
    config: &RocmCliConfig,
    manifests: &[therock::InstalledRuntimeManifest],
) -> Option<ActiveRuntimeProblem> {
    if let Some(runtime_key) = config.active_runtime_key.as_deref() {
        // The resolver's key step on its own: the same call with the default
        // id it would fall back to taken away.
        let mut key_only = config.clone();
        key_only.default_runtime_id = None;
        return current_runtime_manifest(&key_only, manifests)
            .is_none()
            .then(|| ActiveRuntimeProblem::MissingManifestForKey {
                runtime_key: runtime_key.to_owned(),
            });
    }
    let runtime_id = config.default_runtime_id.as_deref()?;
    match default_runtime_id_matches(config, manifests).as_slice() {
        [] => Some(ActiveRuntimeProblem::MissingManifestForId {
            runtime_id: runtime_id.to_owned(),
        }),
        [_] => None,
        matches => Some(ActiveRuntimeProblem::AmbiguousRuntimeId {
            runtime_id: runtime_id.to_owned(),
            runtime_keys: matches
                .iter()
                .map(|manifest| manifest.runtime_key.clone())
                .collect(),
        }),
    }
}

impl cli_report::JsonOrText for RuntimesListing {
    fn render_text(&self) -> String {
        let mut output = String::new();
        let unset = |value: &Option<String>| value.clone().unwrap_or_else(|| "<unset>".to_owned());
        let _ = writeln!(output, "registered ROCm runtimes");
        let _ = writeln!(
            output,
            "  active_runtime_id: {}",
            unset(&self.default_runtime_id)
        );
        let _ = writeln!(
            output,
            "  active_runtime_key: {}",
            unset(&self.active_runtime_key)
        );
        let _ = writeln!(
            output,
            "  previous_runtime_key: {}",
            unset(&self.previous_runtime_key)
        );
        let _ = writeln!(output, "  registry: {}", self.registry.display());
        let _ = writeln!(output, "  marker: {}", self.marker.display());
        match &self.active_problem {
            None => {}
            Some(ActiveRuntimeProblem::MissingManifestForKey { runtime_key }) => {
                let _ = writeln!(
                    output,
                    "  active_status: missing manifest for active_runtime_key={runtime_key}"
                );
            }
            Some(ActiveRuntimeProblem::MissingManifestForId { runtime_id }) => {
                let _ = writeln!(
                    output,
                    "  active_status: missing manifest for active_runtime_id={runtime_id}"
                );
            }
            Some(ActiveRuntimeProblem::AmbiguousRuntimeId {
                runtime_id,
                runtime_keys,
            }) => {
                let _ = writeln!(
                    output,
                    "  active_status: ambiguous runtime_id={runtime_id}; activate one runtime_key: {}",
                    runtime_keys.join(", ")
                );
            }
        }
        if self.runtimes.is_empty() {
            let _ = writeln!(output, "  installed: none");
            let _ = writeln!(
                output,
                "  next step: rocm install sdk --channel release --format wheel"
            );
            return output;
        }

        let _ = writeln!(output, "  installed:");
        let _ = writeln!(
            output,
            "    legend: {ACTIVE_RUNTIME_MARKER} = active, {ROLLBACK_RUNTIME_MARKER} = rollback target"
        );
        let _ = writeln!(output);
        for runtime in &self.runtimes {
            // Active wins when a runtime is both. `rocm runtimes activate` and
            // `rollback` never record a runtime as its own rollback target, so
            // only a change made outside the CLI reaches this: a hand-edited
            // config, or the active runtime's registry entry deleted while the
            // default id falls back to the rollback target.
            let marker = if runtime.active {
                Some(ACTIVE_RUNTIME_MARKER)
            } else if runtime.rollback_target {
                Some(ROLLBACK_RUNTIME_MARKER)
            } else {
                None
            };
            let status = match &runtime.usability {
                RuntimeUsability::Ready => runtime.usability.status().to_owned(),
                RuntimeUsability::Unusable { reason } => unusable_status_text(reason),
            };
            cli_report::write_marked_row(
                &mut output,
                "  ",
                marker,
                format_args!(
                    "{} runtime_id={} version={} format={} family={} mode={} status={} toolchain={}",
                    runtime.runtime_key,
                    runtime.runtime_id,
                    therock::runtime_version_display(&runtime.version),
                    runtime.format,
                    runtime.family,
                    runtime.mode.as_str(),
                    status,
                    runtime.toolchain.as_str()
                ),
            );
            let _ = writeln!(
                output,
                "      install_root: {}",
                runtime.install_root.display()
            );
            let _ = writeln!(
                output,
                "      provenance: {}",
                provenance_text(&runtime.provenance.as_provenance())
            );
        }
        output
    }
}

/// The text form of `rocm runtimes list`, for callers that want the string
/// (the chat assistant's read-only tool) rather than printing it.
pub(crate) fn render_runtimes_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    use cli_report::JsonOrText as _;
    Ok(runtimes_listing(paths, config)?.render_text())
}

/// Make the runtime `selector` names the active one: record it in
/// `config.json` and in the active-runtime marker.
///
/// Both writes, and the read of the settings they start from, happen under the
/// config lock (`RocmCliConfig::lock`), so a concurrent activation, rollback or
/// settings change can neither be undone by this one nor leave `config.json`
/// and the marker naming different runtimes. The caller's `config` is only an
/// output: the activation starts from the settings on disk, and once it has
/// loaded them under the lock, `config` is replaced with the settings it
/// leaves there, whether it succeeds or fails. A failure before that (an
/// unknown runtime, unreadable server records, the lock not free, an
/// unreadable `config.json`) leaves `config` as it was. If the marker write
/// fails and restoring the settings fails too, `config` holds the earlier
/// settings while the file may still name the new runtime; the error says so.
///
/// The live servers are checked first, before the lock is taken: that check
/// asks each running server over HTTP and can take seconds per server, and
/// every other settings change would wait for it. It needs only the runtime
/// registry and the target runtime, never the settings. The servers it finds
/// on another runtime are the ones the report lists and `--restart-services`
/// moves; a server started after the check is in neither. Holding the lock
/// across the check would not close that gap, since starting a server does
/// not take this lock.
pub(crate) fn activate_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    selector: &str,
) -> Result<RuntimeActivationResult> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, selector)?;
    validate_runtime_manifest_for_activation(manifest)?;
    let runtime_key = manifest.runtime_key.clone();
    let services = check_live_services(paths, &manifests, &runtime_key, "the switch to")?;
    with_activation_lock(paths, config, |lock, config| {
        activate_runtime_locked(paths, lock, config, &runtime_key, services)
    })
}

/// The live servers an activation or rollback to `runtime_key` leaves on
/// another runtime. Read before anything is written, so an unreadable services
/// folder refuses the switch instead of half-applying it.
fn check_live_services(
    paths: &AppPaths,
    manifests: &[therock::InstalledRuntimeManifest],
    runtime_key: &str,
    switch: &str,
) -> Result<runtime_services::RuntimeServiceReconciliation> {
    runtime_services::reconcile_services_for_runtime(paths, manifests, runtime_key).with_context(
        || {
            format!(
                "the local server records could not be read, so {switch} `{runtime_key}` was not \
                 made and the previously active runtime is unchanged"
            )
        },
    )
}

/// Run an activation or rollback against the settings on disk, under the
/// config lock, then hand the settings it left on disk back in `config`.
fn with_activation_lock(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    activation: impl FnOnce(&ConfigLock, &mut RocmCliConfig) -> Result<RuntimeActivationResult>,
) -> Result<RuntimeActivationResult> {
    let lock = RocmCliConfig::lock(paths)?;
    let mut current = lock.load()?;
    let result = activation(&lock, &mut current);
    *config = current;
    result
}

/// The part of [`activate_runtime`] under the config lock. The registry is read
/// again here, so a runtime removed since the live servers were checked is
/// refused rather than activated.
fn activate_runtime_locked(
    paths: &AppPaths,
    lock: &ConfigLock,
    config: &mut RocmCliConfig,
    runtime_key: &str,
    services: runtime_services::RuntimeServiceReconciliation,
) -> Result<RuntimeActivationResult> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, runtime_key)?;
    validate_runtime_manifest_for_activation(manifest)?;
    let current = current_runtime_manifest(config, &manifests);
    // Re-selecting the runtime already in use is a no-op to the user: they have
    // not left anything, so there is no new rollback target — and no reason to
    // forget the one the activation that put them here recorded. Deriving the
    // target from `current` unconditionally would clear it, silently removing
    // the recovery path that activation had just offered.
    let reactivating_active_runtime = current
        .as_ref()
        .is_some_and(|current| current.runtime_key == manifest.runtime_key);
    let (previous_runtime_key, previous_runtime_id) = if reactivating_active_runtime {
        let runtime_key = config.previous_runtime_key.clone();
        let runtime_id = runtime_key
            .as_deref()
            .and_then(|runtime_key| {
                manifests
                    .iter()
                    .find(|candidate| candidate.runtime_key == runtime_key)
            })
            .map(|previous| previous.runtime_id.clone());
        (runtime_key, runtime_id)
    } else {
        // `current` differs from the requested runtime here by construction, so
        // this can never record a runtime as its own rollback target.
        (
            current
                .as_ref()
                .map(|manifest| manifest.runtime_key.clone()),
            current.as_ref().map(|manifest| manifest.runtime_id.clone()),
        )
    };

    let snapshot = ActivationSnapshot::capture(paths, config);

    config.default_runtime_id = Some(manifest.runtime_id.clone());
    config.active_runtime_key = Some(manifest.runtime_key.clone());
    config.previous_runtime_key = previous_runtime_key.clone();
    save_activated_config(lock, config, &snapshot)?;
    if let Err(error) = write_active_runtime_marker(
        paths,
        ActiveRuntimeMarker {
            runtime_id: manifest.runtime_id.clone(),
            runtime_key: manifest.runtime_key.clone(),
            manifest_path: runtime_manifest_path(paths, &manifest.runtime_key),
            install_root: manifest.install_root.clone(),
            previous_runtime_id,
            previous_runtime_key: previous_runtime_key.clone(),
            activated_at_unix_ms: rocm_core::unix_time_millis(),
        },
    ) {
        return Err(restore_after_failed_activation(
            paths, lock, config, snapshot, error,
        ));
    }

    Ok(RuntimeActivationResult {
        runtime_id: manifest.runtime_id.clone(),
        runtime_key: manifest.runtime_key.clone(),
        previous_runtime_key,
        reactivated: reactivating_active_runtime,
        services,
    })
}

const NO_ROLLBACK_TARGET: &str =
    "no previous runtime is recorded; activate another runtime before rollback";

/// Switch back to the rollback target, under the config lock and with the
/// caller's `config` as an output, as [`activate_runtime`] does. The live
/// servers are checked first, outside the lock, against the target read from
/// disk then; under the lock the target is read again and must not have
/// changed in between.
fn rollback_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
) -> Result<RuntimeActivationResult> {
    let target = RocmCliConfig::load(paths)?
        .previous_runtime_key
        .context(NO_ROLLBACK_TARGET)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let previous = select_runtime_manifest(&manifests, &target)?;
    validate_runtime_manifest_for_activation(previous)?;
    let runtime_key = previous.runtime_key.clone();
    let services = check_live_services(paths, &manifests, &runtime_key, "the rollback to")?;
    with_activation_lock(paths, config, |lock, config| {
        rollback_runtime_locked(paths, lock, config, &target, services)
    })
}

/// The part of [`rollback_runtime`] under the config lock. `checked_target` is
/// the rollback target the live servers were checked for.
fn rollback_runtime_locked(
    paths: &AppPaths,
    lock: &ConfigLock,
    config: &mut RocmCliConfig,
    checked_target: &str,
    services: runtime_services::RuntimeServiceReconciliation,
) -> Result<RuntimeActivationResult> {
    let previous_key = config
        .previous_runtime_key
        .clone()
        .context(NO_ROLLBACK_TARGET)?;
    if previous_key != checked_target {
        bail!(
            "another rocm command changed the rollback target from `{checked_target}` to \
             `{previous_key}` while this one was checking the local servers, so the rollback was \
             not made and the active runtime is unchanged; run `rocm runtimes rollback` again \
             to roll back to `{previous_key}`"
        );
    }
    let manifests = therock::load_runtime_manifests(paths)?;
    let previous = select_runtime_manifest(&manifests, &previous_key)?;
    validate_runtime_manifest_for_activation(previous)?;
    let current = current_runtime_manifest(config, &manifests);
    let new_previous_key = current
        .as_ref()
        .map(|manifest| manifest.runtime_key.clone())
        .filter(|runtime_key| runtime_key != &previous.runtime_key);
    let new_previous_id = current
        .as_ref()
        .map(|manifest| manifest.runtime_id.clone())
        .filter(|_| new_previous_key.is_some());

    let snapshot = ActivationSnapshot::capture(paths, config);

    config.default_runtime_id = Some(previous.runtime_id.clone());
    config.active_runtime_key = Some(previous.runtime_key.clone());
    config.previous_runtime_key = new_previous_key.clone();
    save_activated_config(lock, config, &snapshot)?;
    if let Err(error) = write_active_runtime_marker(
        paths,
        ActiveRuntimeMarker {
            runtime_id: previous.runtime_id.clone(),
            runtime_key: previous.runtime_key.clone(),
            manifest_path: runtime_manifest_path(paths, &previous.runtime_key),
            install_root: previous.install_root.clone(),
            previous_runtime_id: new_previous_id,
            previous_runtime_key: new_previous_key.clone(),
            activated_at_unix_ms: rocm_core::unix_time_millis(),
        },
    ) {
        return Err(restore_after_failed_activation(
            paths, lock, config, snapshot, error,
        ));
    }

    Ok(RuntimeActivationResult {
        runtime_id: previous.runtime_id.clone(),
        runtime_key: previous.runtime_key.clone(),
        previous_runtime_key: new_previous_key,
        reactivated: false,
        services,
    })
}

/// A runtime's provenance as the `name=value` pairs `rocm runtimes list` prints
/// on its `provenance:` line. `rocm examine` prints the same pairs for the
/// active runtime as `active_runtime_<name>:` lines.
fn provenance_text(provenance: &RuntimeProvenance<'_>) -> String {
    provenance
        .fields()
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// How the CLI names the toolchain state of a runtime.
///
/// `rocm runtimes list` reports it per runtime and `rocm examine` reports it
/// for the active one; sharing the vocabulary here keeps the two from drifting
/// into different words for the same fact, which is the kind of difference a
/// user reads as a difference in meaning.
pub(crate) const fn toolchain_state_text(includes_devel: bool) -> &'static str {
    if includes_devel {
        "included"
    } else {
        "excluded"
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::append_examine_runtime_state;
    use crate::engine_runtime_select::resolve_runtime_selector_to_exact_key;
    use crate::runtime_adopt::{AdoptRuntimeRequest, adopt_runtime_from_probe};
    use crate::runtime_install_finalize::{
        finalize_successful_sdk_install, render_sdk_install_success,
    };
    use crate::runtime_registry::{ManifestOverwrite, write_runtime_manifest};
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::{
        NEW_RUNTIME_KEY, OLD_RUNTIME_KEY, plant_service_record_on_runtime, test_paths,
    };

    #[test]
    fn runtime_selector_recovers_setup_runtime_registry_from_local_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-selector-recover-setup");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-local-manifest",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let install_root = manifest.install_root.clone();
        let mut config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };
        config.setup.completed = true;
        config.setup.therock_venv = Some(install_root.clone());
        config.save(&paths)?;

        let rebased_paths = paths.with_managed_root(install_root, false);
        let rebased_registry = runtime_registry_dir(&rebased_paths);
        let _ = fs::remove_dir_all(&rebased_registry);

        assert!(!runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());
        assert_eq!(
            resolve_runtime_selector_to_exact_key(
                &rebased_paths,
                &manifest.runtime_key,
                "test active runtime"
            )?,
            manifest.runtime_key
        );
        assert!(runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());

        let rendered = render_runtimes_text(&rebased_paths, &config)?;
        assert!(rendered.contains("release-pip-gfx120x-all-local-manifest"));
        assert!(rendered.contains("status=ready"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn sdk_install_finalization_activates_runtime_and_setup_root() -> Result<()> {
        let (root, paths) = test_paths("sdk-install-finalization");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-finalized",
            "therock-release:gfx120X-all",
            "7.13.0",
            42,
        )?;

        let finalized = finalize_successful_sdk_install(&paths)?
            .context("sdk install finalization should select the installed runtime")?;
        let rebased_paths = paths.with_managed_root(manifest.install_root.clone(), false);
        let config = RocmCliConfig::load(&rebased_paths)?;

        assert_eq!(finalized.runtime_key, manifest.runtime_key);
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some(manifest.runtime_id.as_str())
        );
        assert!(config.setup.completed);
        assert_eq!(
            config.setup.therock_venv.as_deref(),
            Some(manifest.install_root.as_path())
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(manifest.runtime_key.as_str())
        );
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some(manifest.runtime_id.as_str())
        );
        assert!(runtime_manifest_path(&rebased_paths, &manifest.runtime_key).is_file());
        assert!(active_runtime_marker_path(&rebased_paths).is_file());

        let success = render_sdk_install_success(&finalized);
        assert!(success.contains("ROCm SDK installed successfully."));
        assert!(success.contains("next step: run `rocm help`"));
        assert!(success.contains(&manifest.install_root.display().to_string()));
        assert!(!success.contains("config:"));
        assert!(!success.contains("marker:"));
        assert!(
            !success.contains("rocm runtimes rollback"),
            "the first install has no previous runtime, so rollback would hard-error; \
             must not hint at a command that immediately fails:\n{success}"
        );

        let mut examine = String::new();
        append_examine_runtime_state(&mut examine, &rebased_paths, &config)?;
        assert!(examine.contains("active_runtime_status: ready"));
        assert!(examine.contains("setup_runtime_root:"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// A second `install sdk` switches the active runtime the same way
    /// `runtimes activate` does, and `activate_runtime` records the previous
    /// runtime either way — so this path must surface the same rollback hint
    /// `runtimes activate` and `update --apply --activate` already do, not
    /// silently drop the recovery advice because it went through the install
    /// finalization path instead.
    #[test]
    fn sdk_install_finalization_hints_rollback_after_a_second_install() -> Result<()> {
        let (root, paths) = test_paths("sdk-install-finalization-second-install");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-first",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        let first = finalize_successful_sdk_install(&paths)?
            .context("first sdk install finalization should select the installed runtime")?;
        assert_eq!(first.previous_runtime_key, None);

        let second_manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-second",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let second = finalize_successful_sdk_install(&paths)?
            .context("second sdk install finalization should select the newest runtime")?;
        assert_eq!(second.runtime_key, second_manifest.runtime_key);
        assert_eq!(
            second.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-first")
        );

        let success = render_sdk_install_success(&second);
        assert!(
            success.contains("next step: if this causes problems, run `rocm runtimes rollback`"),
            "a previous runtime was recorded, so the install summary should hint at rollback \
             as a recovery path, got:\n{success}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_runtimes_text_reports_missing_configured_active_runtime() -> Result<()> {
        let (root, paths) = test_paths("runtime-active-missing");
        let config = RocmCliConfig {
            active_runtime_key: Some("missing-runtime-key".to_owned()),
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_runtimes_text(&paths, &config)?;

        assert!(rendered.contains("active_runtime_key: missing-runtime-key"));
        assert!(rendered.contains(
            "active_status: missing manifest for active_runtime_key=missing-runtime-key"
        ));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_runtimes_text_includes_marker_legend() -> Result<()> {
        let (root, paths) = test_paths("runtime-marker-legend");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some("release-pip-gfx120x-all-7-13-0".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_runtimes_text(&paths, &config)?;
        let legend = format!(
            "legend: {ACTIVE_RUNTIME_MARKER} = active, {ROLLBACK_RUNTIME_MARKER} = rollback target\n\n"
        );
        let entry = format!("{ACTIVE_RUNTIME_MARKER} release-pip-gfx120x-all-7-13-0");
        let legend_pos = rendered.find(&legend).expect("legend line present");
        let entry_pos = rendered.find(&entry).expect("active entry present");
        assert!(
            legend_pos < entry_pos,
            "legend must appear before the entries it explains:\n{rendered}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_runtimes_text_keeps_every_row_in_the_marker_layout() -> Result<()> {
        let (root, paths) = test_paths("runtime-row-layout");
        for (key, version, build) in [
            ("rt-active", "7.13.0", 30),
            ("rt-rollback", "7.12.0", 20),
            ("rt-idle", "7.11.0", 10),
        ] {
            write_test_pip_runtime(&paths, key, "therock-release:gfx120X-all", version, build)?;
        }
        let config = RocmCliConfig {
            active_runtime_key: Some("rt-active".to_owned()),
            previous_runtime_key: Some("rt-rollback".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_runtimes_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        for expected in [
            format!("  {ACTIVE_RUNTIME_MARKER} rt-active runtime_id="),
            format!("  {ROLLBACK_RUNTIME_MARKER} rt-rollback runtime_id="),
            "    rt-idle runtime_id=".to_owned(),
        ] {
            assert!(
                rendered.lines().any(|line| line.starts_with(&expected)),
                "expected a row starting {expected:?}:\n{rendered}"
            );
        }
        Ok(())
    }

    /// Whether a runtime carries the compiler toolchain is otherwise invisible,
    /// and `rocm update` reinstalls whatever the runtime recorded — so a user
    /// who installed without it has no way to see that, or to understand why a
    /// later build step fails. This is the per-runtime half; `rocm examine`
    /// reports the same fact for the active one.
    #[test]
    fn runtime_list_reports_whether_the_toolchain_is_installed() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-toolchain");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        // The fixture records `devel: true` with no composition.
        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(
            rendered.contains("toolchain=included"),
            "a toolchain install must say so:\n{rendered}"
        );

        // A runtime-only install reports the other way. Written through the
        // recorded specs, which is what `includes_devel` actually reads.
        let runtime_only = therock::InstalledRuntimeManifest {
            devel: false,
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "canonical".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some("gfx1201".to_owned()),
            }),
            ..manifest
        };
        fs::write(
            runtime_manifest_path(&paths, &runtime_only.runtime_key),
            serde_json::to_vec_pretty(&runtime_only)?,
        )?;
        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(
            rendered.contains("toolchain=excluded"),
            "a runtime-only install must say so:\n{rendered}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The row a runtime's `provenance:` line belongs to, found by its key: the
    /// first `provenance:` line after that runtime's marked row.
    pub(crate) fn provenance_line<'a>(rendered: &'a str, runtime_key: &str) -> &'a str {
        let row = format!(" {runtime_key} runtime_id=");
        rendered
            .lines()
            .skip_while(|line| !line.contains(&row))
            .skip(1)
            .find(|line| line.trim_start().starts_with("provenance:"))
            .unwrap_or_else(|| panic!("no provenance line for {runtime_key}:\n{rendered}"))
    }

    /// A nightly ROCm 10 wheel runtime installed for one exact GPU, with every
    /// provenance fact recorded the way `rocm install sdk` records it. The
    /// family is the family-wide `gfx120X-all`, so a target inferred from the
    /// family instead of read from the install shows up as a wrong value.
    fn nightly_exact_target_manifest(
        base: therock::InstalledRuntimeManifest,
    ) -> therock::InstalledRuntimeManifest {
        therock::InstalledRuntimeManifest {
            runtime_key: "nightly-wheel-gfx1201-10-1-0a20260822".to_owned(),
            runtime_id: "therock-nightly:gfx120X-all".to_owned(),
            channel: "nightly".to_owned(),
            family: "gfx120X-all".to_owned(),
            version: "10.1.0a20260822".to_owned(),
            selected_artifact_url: "https://example.invalid/whl-multi-arch".to_owned(),
            index_url: Some("https://example.invalid/whl-multi-arch".to_owned()),
            source_layout_generation: Some("next-v1".to_owned()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "next-v1".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==10.1.0a20260822".to_owned()],
                rocm_sdk_target: Some("gfx1201".to_owned()),
            }),
            ..base
        }
    }

    fn rewrite_manifest(
        paths: &AppPaths,
        manifest: &therock::InstalledRuntimeManifest,
    ) -> Result<()> {
        fs::write(
            runtime_manifest_path(paths, &manifest.runtime_key),
            serde_json::to_vec_pretty(manifest)?,
        )?;
        Ok(())
    }

    /// The install report names the channel, source, layout and exact GPU
    /// target; afterwards the only record of them is the manifest, so the
    /// listing is where a user finds out which stream and which GPU a runtime
    /// is for. Each value is asserted exactly, so a field printed from the
    /// wrong source (the family for the target, the channel for the layout)
    /// fails here.
    #[test]
    fn runtime_list_names_where_each_runtime_came_from() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-provenance");
        let base = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;

        // Written before the index, the layout and the composition were
        // recorded: only the artifact URL every manifest carries is left.
        let legacy = therock::InstalledRuntimeManifest {
            index_url: None,
            source_layout_generation: None,
            wheel_composition: None,
            ..base.clone()
        };
        rewrite_manifest(&paths, &legacy)?;

        let nightly = nightly_exact_target_manifest(base.clone());
        write_test_pip_runtime(
            &paths,
            &nightly.runtime_key,
            &nightly.runtime_id,
            &nightly.version,
            20,
        )?;
        rewrite_manifest(&paths, &nightly)?;

        // A tarball install records the archive it downloaded and its layout;
        // it was never installed for one exact GPU.
        let tarball = therock::InstalledRuntimeManifest {
            runtime_key: "release-tarball-gfx120x-all-7-13-0".to_owned(),
            format: "tarball".to_owned(),
            selected_artifact_url:
                "https://example.invalid/tarball/therock-dist-linux-gfx120X-all-7.13.0.tar.gz"
                    .to_owned(),
            index_url: None,
            source_layout_generation: Some("multi-arch-v2".to_owned()),
            wheel_composition: None,
            ..base
        };
        rewrite_manifest(&paths, &tarball)?;

        let rendered = render_runtimes_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        // `cargo xtask e2e` prewarm (`assess` in xtask/src/e2e_prewarm.rs)
        // reads the line directly after a runtime's row as its
        // `install_root:`, so the order row, `install_root:`, `provenance:` is
        // pinned here by index; `provenance_line` searches ahead and would not
        // notice a reorder.
        let lines: Vec<&str> = rendered.lines().collect();
        let row = format!(" {} runtime_id=", nightly.runtime_key);
        let row_index = lines
            .iter()
            .position(|line| line.contains(&row))
            .unwrap_or_else(|| panic!("no row for {}:\n{rendered}", nightly.runtime_key));
        assert_eq!(
            lines.get(row_index + 1).copied(),
            Some(format!("      install_root: {}", nightly.install_root.display()).as_str()),
            "the line after the row must be its install_root:\n{rendered}"
        );
        assert_eq!(
            lines.get(row_index + 2).copied(),
            Some(provenance_line(&rendered, &nightly.runtime_key)),
            "the line after install_root: must be its provenance:\n{rendered}"
        );

        assert_eq!(
            provenance_line(&rendered, &nightly.runtime_key),
            "      provenance: channel=nightly source=https://example.invalid/whl-multi-arch \
             source_layout_generation=next-v1 device_target=gfx1201",
            "{rendered}"
        );
        assert!(
            rendered.contains("version=10.1.0a20260822 (build 2026-08-22)"),
            "the nightly's version must show its build date:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &legacy.runtime_key),
            "      provenance: channel=release source=https://example.invalid/therock \
             source_layout_generation=unrecorded device_target=unrecorded",
            "{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &tarball.runtime_key),
            "      provenance: channel=release \
             source=https://example.invalid/tarball/therock-dist-linux-gfx120X-all-7.13.0.tar.gz \
             source_layout_generation=multi-arch-v2 device_target=unrecorded",
            "{rendered}"
        );
        Ok(())
    }

    /// The layout and target are read from the same recorded fields
    /// `rocm update` reads; it does not read the recorded source. Each chain
    /// moves to its next field only when a field is absent: package index,
    /// else artifact URL (never the adoption marker); layout field, else the
    /// wheel composition's layout; `rocm_sdk_target`, else the
    /// `device-<target>` package extra. The value chosen is printed trimmed,
    /// and a fact recorded nowhere, or recorded blank, reads `unrecorded`. One
    /// runtime per rule:
    ///
    /// - index and artifact URL both recorded and differing: the index wins;
    /// - layout field and composition layout differing, and `rocm_sdk_target`
    ///   and package extra differing: the field earlier in each chain wins;
    /// - layout only in the composition (manifests older than the field): it is
    ///   still the layout;
    /// - target only as a package extra (manifests older than
    ///   `rocm_sdk_target`): it is still the target;
    /// - an adopted runtime (marker, no index): source `unrecorded`;
    /// - a blank index, layout field and `rocm_sdk_target` in front of a
    ///   recorded artifact URL, composition layout and package extra: each
    ///   reads `unrecorded`, since a blank value is not skipped;
    /// - an empty `device-` extra and no `rocm_sdk_target`: `unrecorded`;
    /// - values padded with whitespace: printed trimmed;
    /// - every field blank: each fact `unrecorded`, never an empty `name=`.
    ///
    /// Blank or padded values are not what `rocm install sdk` writes; they can
    /// come from a hand-edited manifest or one brought in with
    /// `rocm runtimes import`.
    #[test]
    fn runtime_list_reads_each_provenance_fact_from_where_it_was_recorded() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-provenance-precedence");
        let base = write_test_pip_runtime(
            &paths,
            "release-pip-index-and-artifact",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;

        let index_and_artifact = therock::InstalledRuntimeManifest {
            index_url: Some("https://example.invalid/simple".to_owned()),
            selected_artifact_url: "https://example.invalid/whl/rocm-7.13.0.tar.gz".to_owned(),
            ..base.clone()
        };
        rewrite_manifest(&paths, &index_and_artifact)?;

        let composition_layout = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-composition-layout".to_owned(),
            source_layout_generation: None,
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "next-v1".to_owned(),
                package_specs: vec!["rocm[libraries]==7.13.0".to_owned()],
                rocm_sdk_target: None,
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &composition_layout)?;

        let spec_target = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-spec-target".to_owned(),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "multi-arch-v2".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: None,
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &spec_target)?;

        let adopted = therock::InstalledRuntimeManifest {
            runtime_key: "adopted-release-pip".to_owned(),
            index_url: None,
            selected_artifact_url: crate::runtime_registry::ADOPTED_RUNTIME_ARTIFACT_URL.to_owned(),
            read_only: true,
            ..base.clone()
        };
        rewrite_manifest(&paths, &adopted)?;

        let blank_before_fallback = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-blank-before-fallback".to_owned(),
            index_url: Some(String::new()),
            selected_artifact_url: "https://example.invalid/therock".to_owned(),
            source_layout_generation: Some(String::new()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "next-v1".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some(String::new()),
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &blank_before_fallback)?;

        let first_in_chain_wins = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-first-in-chain-wins".to_owned(),
            source_layout_generation: Some("multi-arch-v2".to_owned()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "next-v1".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some("gfx1100".to_owned()),
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &first_in_chain_wins)?;

        let padded = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-padded".to_owned(),
            channel: " release ".to_owned(),
            index_url: Some(" https://example.invalid/simple ".to_owned()),
            source_layout_generation: Some(" multi-arch-v2 ".to_owned()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "multi-arch-v2".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some(" gfx1201 ".to_owned()),
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &padded)?;

        let empty_device_extra = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-empty-device-extra".to_owned(),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "multi-arch-v2".to_owned(),
                package_specs: vec!["rocm[libraries,device-]==7.13.0".to_owned()],
                rocm_sdk_target: None,
            }),
            ..base.clone()
        };
        rewrite_manifest(&paths, &empty_device_extra)?;

        let all_blank = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-all-blank".to_owned(),
            channel: String::new(),
            index_url: Some(" ".to_owned()),
            selected_artifact_url: String::new(),
            source_layout_generation: Some(String::new()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: String::new(),
                package_specs: vec!["rocm[libraries]==7.13.0".to_owned()],
                rocm_sdk_target: Some(String::new()),
            }),
            ..base
        };
        rewrite_manifest(&paths, &all_blank)?;

        let rendered = render_runtimes_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            provenance_line(&rendered, &index_and_artifact.runtime_key),
            "      provenance: channel=release source=https://example.invalid/simple \
             source_layout_generation=unrecorded device_target=unrecorded",
            "the index must win over the artifact URL:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &composition_layout.runtime_key),
            "      provenance: channel=release source=https://example.invalid/therock \
             source_layout_generation=next-v1 device_target=unrecorded",
            "a layout recorded only in the composition is still recorded:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &spec_target.runtime_key),
            "      provenance: channel=release source=https://example.invalid/therock \
             source_layout_generation=multi-arch-v2 device_target=gfx1201",
            "a target recorded only in the package specs is still recorded:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &adopted.runtime_key),
            "      provenance: channel=release source=unrecorded \
             source_layout_generation=unrecorded device_target=unrecorded",
            "the adoption marker is not a package source:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &first_in_chain_wins.runtime_key),
            "      provenance: channel=release source=https://example.invalid/therock \
             source_layout_generation=multi-arch-v2 device_target=gfx1100",
            "the first recorded value in each chain must win:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &blank_before_fallback.runtime_key),
            "      provenance: channel=release source=unrecorded \
             source_layout_generation=unrecorded device_target=unrecorded",
            "a blank first value reads unrecorded, not the field after it:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &empty_device_extra.runtime_key),
            "      provenance: channel=release source=https://example.invalid/therock \
             source_layout_generation=multi-arch-v2 device_target=unrecorded",
            "an empty device extra names no target:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &padded.runtime_key),
            "      provenance: channel=release source=https://example.invalid/simple \
             source_layout_generation=multi-arch-v2 device_target=gfx1201",
            "shown values must be trimmed:\n{rendered}"
        );
        assert_eq!(
            provenance_line(&rendered, &all_blank.runtime_key),
            "      provenance: channel=unrecorded source=unrecorded \
             source_layout_generation=unrecorded device_target=unrecorded",
            "a fact recorded only as blank must read unrecorded:\n{rendered}"
        );
        Ok(())
    }

    /// `rocm examine` reports the active runtime's provenance as one line per
    /// fact. They must be the facts and words the listing shows for that same
    /// runtime, so this compares the two outputs as well as pinning each.
    /// Two active runtimes: a nightly install that recorded every fact, and an
    /// adopted one whose manifest holds only the adoption marker where an
    /// artifact URL would be, which must not be reported as its source.
    #[test]
    fn examine_names_the_active_runtime_provenance_the_listing_shows() -> Result<()> {
        type Build = fn(therock::InstalledRuntimeManifest) -> therock::InstalledRuntimeManifest;
        let adopted: Build = |base| therock::InstalledRuntimeManifest {
            index_url: None,
            selected_artifact_url: crate::runtime_registry::ADOPTED_RUNTIME_ARTIFACT_URL.to_owned(),
            read_only: true,
            imported_from: Some(base.install_root.clone()),
            ..base
        };
        let cases: [(&str, &str, &str, Build, [&str; 4]); 2] = [
            (
                "nightly-wheel-gfx1201-10-1-0a20260822",
                "therock-nightly:gfx120X-all",
                "10.1.0a20260822",
                nightly_exact_target_manifest,
                [
                    "  active_runtime_channel: nightly",
                    "  active_runtime_source: https://example.invalid/whl-multi-arch",
                    "  active_runtime_source_layout_generation: next-v1",
                    "  active_runtime_device_target: gfx1201",
                ],
            ),
            (
                "adopted-release-pip-gfx120x-all-7-13-0",
                "therock-release:gfx120X-all",
                "7.13.0",
                adopted,
                [
                    "  active_runtime_channel: release",
                    "  active_runtime_source: unrecorded",
                    "  active_runtime_source_layout_generation: unrecorded",
                    "  active_runtime_device_target: unrecorded",
                ],
            ),
        ];
        for (runtime_key, runtime_id, version, build, expected) in cases {
            let (root, paths) = test_paths("examine-runtime-provenance");
            let base = write_test_pip_runtime(&paths, runtime_key, runtime_id, version, 10)?;
            let manifest = build(base);
            rewrite_manifest(&paths, &manifest)?;
            let config = RocmCliConfig {
                active_runtime_key: Some(manifest.runtime_key.clone()),
                ..RocmCliConfig::default()
            };

            let listed = render_runtimes_text(&paths, &config)?;
            let mut examined = String::new();
            append_examine_runtime_state(&mut examined, &paths, &config)?;
            let _ = fs::remove_dir_all(root);

            let listed_pairs = provenance_line(&listed, &manifest.runtime_key)
                .trim_start()
                .strip_prefix("provenance: ")
                .expect("provenance prefix")
                .split(' ')
                .map(|pair| pair.split_once('=').expect("name=value pair"))
                .map(|(name, value)| format!("  active_runtime_{name}: {value}"))
                .collect::<Vec<_>>();
            assert_eq!(listed_pairs, expected, "{runtime_key}:\n{listed}");
            for line in expected {
                assert!(
                    examined.lines().any(|examined_line| examined_line == line),
                    "{runtime_key}: examine must report {line:?} as the listing does:\n{examined}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn runtime_lists_display_build_date_from_version_string() -> Result<()> {
        let (root, paths) = test_paths("runtime-build-date-display");
        let runtime_key = "release-pip-gfx120x-all-7-14-0a20260601";
        let manifest = write_test_pip_runtime(
            &paths,
            runtime_key,
            "therock-release:gfx120X-all",
            "7.14.0a20260601",
            20,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            default_runtime_id: Some(manifest.runtime_id),
            ..RocmCliConfig::default()
        };

        let runtimes = render_runtimes_text(&paths, &config)?;
        assert!(runtimes.contains("version=7.14.0a20260601 (build 2026-06-01)"));

        let mut examine = String::new();
        append_examine_runtime_state(&mut examine, &paths, &config)?;
        assert!(examine.contains("active_runtime_version: 7.14.0a20260601 (build 2026-06-01)"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// What `rocm runtimes list --json` prints for `config`, as the JSON value a
    /// script reads.
    fn runtimes_list_json(paths: &AppPaths, config: &RocmCliConfig) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(runtimes_listing(paths, config)?)?)
    }

    /// The `runtime_key`s whose `flag` is true, in listing order.
    fn runtime_keys_flagged(listing: &serde_json::Value, flag: &str) -> Vec<String> {
        listing["runtimes"]
            .as_array()
            .expect("runtimes is an array")
            .iter()
            .filter(|runtime| {
                // Every runtime must carry `flag` as a boolean: read as
                // `runtime[flag] == true`, a missing or renamed field would
                // count as false and "nothing is flagged" would still pass.
                runtime
                    .get(flag)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or_else(|| panic!("a runtime has no boolean `{flag}`:\n{listing:#}"))
            })
            .map(|runtime| {
                runtime["runtime_key"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect()
    }

    fn listed_runtime<'a>(
        listing: &'a serde_json::Value,
        runtime_key: &str,
    ) -> &'a serde_json::Value {
        listing["runtimes"]
            .as_array()
            .expect("runtimes is an array")
            .iter()
            .find(|runtime| runtime["runtime_key"] == runtime_key)
            .unwrap_or_else(|| panic!("{runtime_key} is listed:\n{listing:#}"))
    }

    #[test]
    fn runtimes_list_json_with_nothing_registered_lists_nothing_and_no_active_runtime() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-list-json-empty");
        let listing = runtimes_list_json(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(listing["runtimes"], serde_json::json!([]));
        for unset in [
            "default_runtime_id",
            "active_runtime_key",
            "previous_runtime_key",
            "active_problem",
        ] {
            assert_null_field(&listing, unset);
        }
        for path in ["registry", "marker"] {
            assert!(
                listing.get(path).is_some_and(serde_json::Value::is_string),
                "`{path}` is always a path:\n{listing:#}"
            );
        }
        assert!(
            listing.get("next_step").is_none(),
            "the next-step hint is text-only:\n{listing:#}"
        );
        Ok(())
    }

    #[test]
    fn runtimes_list_json_flags_the_active_runtime_and_the_rollback_target() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-active-rollback");
        for (key, version, build) in [
            ("rt-first", "7.12.0", 10),
            ("rt-second", "7.13.0", 20),
            ("rt-idle", "7.11.0", 5),
        ] {
            write_test_pip_runtime(&paths, key, "therock-release:gfx120X-all", version, build)?;
        }
        let config = RocmCliConfig {
            active_runtime_key: Some("rt-second".to_owned()),
            previous_runtime_key: Some("rt-first".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(runtime_keys_flagged(&listing, "active"), ["rt-second"]);
        assert_eq!(
            runtime_keys_flagged(&listing, "rollback_target"),
            ["rt-first"]
        );
        assert_eq!(listing["active_runtime_key"], "rt-second");
        assert_eq!(listing["previous_runtime_key"], "rt-first");
        assert_null_field(&listing, "active_problem");
        let second = listed_runtime(&listing, "rt-second");
        assert_eq!(second["runtime_id"], "therock-release:gfx120X-all");
        assert_eq!(second["version"], "7.13.0");
        assert_eq!(second["format"], "wheel");
        assert_eq!(second["family"], "gfx120X-all");
        assert_eq!(second["mode"], "managed");
        assert_eq!(second["status"], "ready");
        assert_null_field(second, "unusable_reason");
        assert_eq!(second["toolchain"], "included");
        Ok(())
    }

    #[test]
    fn runtimes_list_json_marks_the_one_runtime_matching_the_default_id_active() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-default-id");
        write_test_pip_runtime(
            &paths,
            "rt-gfx120x",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        write_test_pip_runtime(
            &paths,
            "rt-gfx110x",
            "therock-release:gfx110X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(runtime_keys_flagged(&listing, "active"), ["rt-gfx120x"]);
        assert_eq!(listing["default_runtime_id"], "therock-release:gfx120X-all");
        assert_null_field(&listing, "active_runtime_key");
        assert_null_field(&listing, "active_problem");
        Ok(())
    }

    #[test]
    fn runtimes_list_json_reports_an_active_key_with_no_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-missing-key");
        write_test_pip_runtime(
            &paths,
            "rt-present",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some("rt-gone".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            listing["active_problem"],
            serde_json::json!({"kind": "missing_manifest_for_key", "runtime_key": "rt-gone"})
        );
        assert!(runtime_keys_flagged(&listing, "active").is_empty());
        Ok(())
    }

    #[test]
    fn runtimes_list_json_reports_a_default_id_with_no_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-missing-id");
        write_test_pip_runtime(
            &paths,
            "rt-present",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx110X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            listing["active_problem"],
            serde_json::json!({
                "kind": "missing_manifest_for_id",
                "runtime_id": "therock-release:gfx110X-all"
            })
        );
        assert!(runtime_keys_flagged(&listing, "active").is_empty());
        Ok(())
    }

    #[test]
    fn runtimes_list_json_reports_a_default_id_matching_two_runtimes_as_ambiguous() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-ambiguous-id");
        write_test_pip_runtime(
            &paths,
            "rt-newer",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        write_test_pip_runtime(
            &paths,
            "rt-older",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let text = render_runtimes_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            listing["active_problem"],
            serde_json::json!({
                "kind": "ambiguous_runtime_id",
                "runtime_id": "therock-release:gfx120X-all",
                "runtime_keys": ["rt-newer", "rt-older"]
            })
        );
        assert!(runtime_keys_flagged(&listing, "active").is_empty());
        // The text form prints the same problem from the same value.
        assert!(
            text.contains(
                "  active_status: ambiguous runtime_id=therock-release:gfx120X-all; \
                 activate one runtime_key: rt-newer, rt-older\n"
            ),
            "{text}"
        );
        Ok(())
    }

    /// `key` is present in `value` and `null`. Indexing a `serde_json::Value`
    /// returns `null` for a missing key too, so `value[key] == Value::Null`
    /// would stay green if the key were dropped from the document.
    fn assert_null_field(value: &serde_json::Value, key: &str) {
        assert_eq!(
            value.as_object().and_then(|object| object.get(key)),
            Some(&serde_json::Value::Null),
            "`{key}` must be present and null:\n{value:#}"
        );
    }

    /// The `runtime_key`s the text form marks with `glyph`, in listing order.
    fn runtime_keys_marked(text: &str, glyph: &str) -> Vec<String> {
        text.lines()
            .filter_map(|line| line.strip_prefix(&format!("  {glyph} ")))
            .filter_map(|row| row.split_whitespace().next())
            .map(str::to_owned)
            .collect()
    }

    /// With no `active_runtime_key`, the listing matches `default_runtime_id`
    /// ignoring ASCII letter case, and `active` and `active_problem` use that
    /// one rule (`default_runtime_id_matches`). A default id that differs from
    /// its one runtime's id only in case marks that runtime active and reports
    /// no problem; `rocm config set-default-runtime` stores the id as typed,
    /// so this state is one command away. Because both fields use the same
    /// rule, `active_problem` never reports as missing a runtime that `active`
    /// marks.
    #[test]
    fn runtimes_list_matches_a_default_id_differing_only_in_case_with_one_rule() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-default-id-case");
        write_test_pip_runtime(
            &paths,
            "rt-gfx120x",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("THEROCK-RELEASE:GFX120X-ALL".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let text = render_runtimes_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(runtime_keys_flagged(&listing, "active"), ["rt-gfx120x"]);
        assert_null_field(&listing, "active_problem");
        assert_eq!(
            runtime_keys_marked(&text, ACTIVE_RUNTIME_MARKER),
            ["rt-gfx120x"]
        );
        assert!(!text.contains("active_status:"), "{text}");
        Ok(())
    }

    /// Under the same rule, a default id matching two runtime ids that differ
    /// only in letter case matches both: neither is active, and the listing
    /// reports the id as ambiguous, naming both keys in listing order. The
    /// runtime whose id equals the default exactly counts as one match among
    /// the two, not as the winner.
    #[test]
    fn runtimes_list_reports_a_default_id_matching_ids_differing_only_in_case_as_ambiguous()
    -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-ambiguous-id-case");
        write_test_pip_runtime(
            &paths,
            "rt-upper",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        write_test_pip_runtime(
            &paths,
            "rt-lower",
            "therock-release:gfx120x-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let text = render_runtimes_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            listing["active_problem"],
            serde_json::json!({
                "kind": "ambiguous_runtime_id",
                "runtime_id": "therock-release:gfx120X-all",
                "runtime_keys": ["rt-upper", "rt-lower"]
            })
        );
        assert!(runtime_keys_flagged(&listing, "active").is_empty());
        assert!(
            text.contains(
                "  active_status: ambiguous runtime_id=therock-release:gfx120X-all; \
                 activate one runtime_key: rt-upper, rt-lower\n"
            ),
            "{text}"
        );
        Ok(())
    }

    // ── State matrix: what the listing marks, for every configuration ──
    //
    // Built in memory through `listing_from_manifests`, so keys that differ
    // only in letter case are listed as they would be on a case-sensitive
    // filesystem without writing them to one: on a case-insensitive one the
    // second `<key>.json` would overwrite the first.

    /// A registered runtime with just the fields the listing reads. Its
    /// install root does not exist, so it is listed `unusable`, which no row
    /// below depends on.
    fn listed_runtime_manifest(
        runtime_key: &str,
        runtime_id: &str,
    ) -> therock::InstalledRuntimeManifest {
        serde_json::from_value(serde_json::json!({
            "runtime_key": runtime_key,
            "runtime_id": runtime_id,
            "channel": "release",
            "format": "wheel",
            "family": "gfx120X-all",
            "family_source": "test",
            "version": "7.13.0",
            "install_root": format!("/nonexistent/rocm-list-state/{runtime_key}"),
            "selected_artifact_url": "https://example.invalid/therock",
            "installed_at_unix_ms": 10,
        }))
        .expect("a minimal runtime manifest deserializes")
    }

    const ID_A: &str = "therock-release:gfx120X-all";
    const ID_B: &str = "therock-release:gfx110X-all";

    /// Runtimes `rt-a` (id `ID_A`) and `rt-b` (id `ID_B`), in that order.
    fn two_distinct_runtimes() -> Vec<therock::InstalledRuntimeManifest> {
        vec![
            listed_runtime_manifest("rt-a", ID_A),
            listed_runtime_manifest("rt-b", ID_B),
        ]
    }

    /// Keys `rt-twin` and `RT-TWIN`, in that order, with distinct ids.
    fn case_twin_runtimes() -> Vec<therock::InstalledRuntimeManifest> {
        vec![
            listed_runtime_manifest("rt-twin", ID_A),
            listed_runtime_manifest("RT-TWIN", ID_B),
        ]
    }

    /// One row of the state matrix: a configuration, the runtimes, and what
    /// the listing must say about them in both forms.
    struct ListingState {
        config: RocmCliConfig,
        manifests: Vec<therock::InstalledRuntimeManifest>,
        /// The key marked `active` (JSON) and `*` (text).
        active: Option<&'static str>,
        /// The key flagged `rollback_target` (JSON); marked `-` in the text
        /// unless it is also active.
        rollback: Option<&'static str>,
        /// The JSON `active_problem`.
        problem: serde_json::Value,
        /// The text's `active_status:` line, without its indentation.
        status_line: Option<&'static str>,
    }

    fn config_with(
        default_runtime_id: Option<&str>,
        active_runtime_key: Option<&str>,
        previous_runtime_key: Option<&str>,
    ) -> RocmCliConfig {
        RocmCliConfig {
            default_runtime_id: default_runtime_id.map(str::to_owned),
            active_runtime_key: active_runtime_key.map(str::to_owned),
            previous_runtime_key: previous_runtime_key.map(str::to_owned),
            ..RocmCliConfig::default()
        }
    }

    fn assert_listing_state(state: &ListingState) -> Result<()> {
        use cli_report::JsonOrText as _;
        let listing = listing_from_manifests(
            &state.config,
            state.manifests.clone(),
            PathBuf::from("registry"),
            PathBuf::from("active.json"),
        );
        let json = serde_json::to_value(&listing)?;
        let text = listing.render_text();
        let context = format!("{json:#}\n{text}");

        // The listing marks what the commands resolve: `current_runtime_manifest`
        // for the active runtime, `select_runtime_manifest` for rollback.
        let resolved_active = current_runtime_manifest(&state.config, &state.manifests)
            .map(|manifest| manifest.runtime_key.as_str());
        assert_eq!(resolved_active, state.active, "resolver vs row:\n{context}");
        let resolved_rollback = state
            .config
            .previous_runtime_key
            .as_deref()
            .and_then(|selector| select_runtime_manifest(&state.manifests, selector).ok())
            .map(|manifest| manifest.runtime_key.as_str());
        assert_eq!(
            resolved_rollback, state.rollback,
            "resolver vs row:\n{context}"
        );

        let expected_active: Vec<String> = state.active.iter().map(|k| (*k).to_owned()).collect();
        let expected_rollback: Vec<String> =
            state.rollback.iter().map(|k| (*k).to_owned()).collect();
        assert_eq!(
            runtime_keys_flagged(&json, "active"),
            expected_active,
            "{context}"
        );
        assert_eq!(
            runtime_keys_flagged(&json, "rollback_target"),
            expected_rollback,
            "{context}"
        );
        assert_eq!(json["active_problem"], state.problem, "{context}");
        assert_eq!(
            runtime_keys_marked(&text, ACTIVE_RUNTIME_MARKER),
            expected_active,
            "{context}"
        );
        let rollback_only: Vec<String> = expected_rollback
            .into_iter()
            .filter(|key| Some(key.as_str()) != state.active)
            .collect();
        assert_eq!(
            runtime_keys_marked(&text, ROLLBACK_RUNTIME_MARKER),
            rollback_only,
            "{context}"
        );
        let status_lines: Vec<&str> = text
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with("active_status:"))
            .collect();
        assert_eq!(
            status_lines,
            state.status_line.into_iter().collect::<Vec<_>>(),
            "{context}"
        );
        Ok(())
    }

    fn state_no_config() -> ListingState {
        ListingState {
            config: RocmCliConfig::default(),
            manifests: two_distinct_runtimes(),
            active: None,
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_key_exact() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), None),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// The default id names the other runtime, so the key alone decides.
    fn state_key_in_other_case() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("RT-A"), None),
            manifests: two_distinct_runtimes(),
            active: Some("rt-a"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// The one documented case where `active_problem` is set beside an
    /// `active` runtime: the key is dead, the default id still resolves.
    fn state_stale_key_default_resolves() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-gone"), None),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: None,
            problem: serde_json::json!({"kind": "missing_manifest_for_key", "runtime_key": "rt-gone"}),
            status_line: Some("active_status: missing manifest for active_runtime_key=rt-gone"),
        }
    }

    fn state_stale_key_default_missing() -> ListingState {
        ListingState {
            config: config_with(Some("therock-release:gfx90a"), Some("rt-gone"), None),
            manifests: two_distinct_runtimes(),
            active: None,
            rollback: None,
            problem: serde_json::json!({"kind": "missing_manifest_for_key", "runtime_key": "rt-gone"}),
            status_line: Some("active_status: missing manifest for active_runtime_key=rt-gone"),
        }
    }

    fn state_stale_key_default_ambiguous() -> ListingState {
        ListingState {
            config: config_with(Some(ID_A), Some("rt-gone"), None),
            manifests: vec![
                listed_runtime_manifest("rt-a", ID_A),
                listed_runtime_manifest("rt-a2", ID_A),
            ],
            active: None,
            rollback: None,
            problem: serde_json::json!({"kind": "missing_manifest_for_key", "runtime_key": "rt-gone"}),
            status_line: Some("active_status: missing manifest for active_runtime_key=rt-gone"),
        }
    }

    fn state_default_matches_none() -> ListingState {
        ListingState {
            config: config_with(Some("therock-release:gfx90a"), None, None),
            manifests: two_distinct_runtimes(),
            active: None,
            rollback: None,
            problem: serde_json::json!({
                "kind": "missing_manifest_for_id",
                "runtime_id": "therock-release:gfx90a"
            }),
            status_line: Some(
                "active_status: missing manifest for active_runtime_id=therock-release:gfx90a",
            ),
        }
    }

    fn state_default_matches_one() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), None, None),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_default_matches_one_in_other_case() -> ListingState {
        ListingState {
            config: config_with(Some("THEROCK-RELEASE:GFX110X-ALL"), None, None),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_default_matches_two() -> ListingState {
        ListingState {
            config: config_with(Some(ID_A), None, None),
            manifests: vec![
                listed_runtime_manifest("rt-a", ID_A),
                listed_runtime_manifest("rt-a2", ID_A),
            ],
            active: None,
            rollback: None,
            problem: serde_json::json!({
                "kind": "ambiguous_runtime_id",
                "runtime_id": ID_A,
                "runtime_keys": ["rt-a", "rt-a2"]
            }),
            status_line: Some(
                "active_status: ambiguous runtime_id=therock-release:gfx120X-all; \
                 activate one runtime_key: rt-a, rt-a2",
            ),
        }
    }

    /// Ids differing only in case are two matches, even though one equals the
    /// default exactly.
    fn state_default_matches_two_differing_in_case() -> ListingState {
        ListingState {
            config: config_with(Some(ID_A), None, None),
            manifests: vec![
                listed_runtime_manifest("rt-a", ID_A),
                listed_runtime_manifest("rt-a-lower", "therock-release:gfx120x-all"),
            ],
            active: None,
            rollback: None,
            problem: serde_json::json!({
                "kind": "ambiguous_runtime_id",
                "runtime_id": ID_A,
                "runtime_keys": ["rt-a", "rt-a-lower"]
            }),
            status_line: Some(
                "active_status: ambiguous runtime_id=therock-release:gfx120X-all; \
                 activate one runtime_key: rt-a, rt-a-lower",
            ),
        }
    }

    /// Case twins: the resolver takes the first key matching in list order,
    /// even when a later one equals the active key exactly, and the listing
    /// marks that first one, because that is what the commands act on.
    fn state_case_twins_key_names_the_second() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("RT-TWIN"), None),
            manifests: case_twin_runtimes(),
            active: Some("rt-twin"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_case_twins_key_names_the_first() -> ListingState {
        ListingState {
            config: config_with(Some(ID_A), Some("rt-twin"), None),
            manifests: case_twin_runtimes(),
            active: Some("rt-twin"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_previous_key_exact() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), Some("rt-a")),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: Some("rt-a"),
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_previous_key_in_other_case() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), Some("RT-A")),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: Some("rt-a"),
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_previous_key_is_a_runtime_id() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), Some(ID_A)),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: Some("rt-a"),
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_previous_key_with_whitespace() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), Some(" rt-a\n")),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: Some("rt-a"),
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    fn state_previous_key_missing() -> ListingState {
        ListingState {
            config: config_with(Some(ID_B), Some("rt-b"), Some("rt-gone")),
            manifests: two_distinct_runtimes(),
            active: Some("rt-b"),
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// Matches both twins only by letter case: rollback refuses it as
    /// ambiguous, so nothing is the rollback target.
    fn state_previous_key_ambiguous_case_twins() -> ListingState {
        ListingState {
            config: config_with(None, None, Some("Rt-Twin")),
            manifests: case_twin_runtimes(),
            active: None,
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// A runtime id two runtimes share: rollback refuses it as ambiguous.
    fn state_previous_key_ambiguous_runtime_id() -> ListingState {
        ListingState {
            config: config_with(None, None, Some(ID_A)),
            manifests: vec![
                listed_runtime_manifest("rt-a", ID_A),
                listed_runtime_manifest("rt-a2", ID_A),
            ],
            active: None,
            rollback: None,
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// Rollback takes an exact key over its case twin.
    fn state_previous_key_names_one_case_twin_exactly() -> ListingState {
        ListingState {
            config: config_with(None, None, Some("RT-TWIN")),
            manifests: case_twin_runtimes(),
            active: None,
            rollback: Some("RT-TWIN"),
            problem: serde_json::Value::Null,
            status_line: None,
        }
    }

    /// The active runtime's registry entry was deleted and the default id it
    /// shared with the rollback target now resolves that one: it is both, and
    /// the text marks it `*`.
    fn state_stale_key_falls_back_to_the_rollback_target() -> ListingState {
        ListingState {
            config: config_with(Some(ID_A), Some("rt-gone"), Some("rt-a")),
            manifests: two_distinct_runtimes(),
            active: Some("rt-a"),
            rollback: Some("rt-a"),
            problem: serde_json::json!({"kind": "missing_manifest_for_key", "runtime_key": "rt-gone"}),
            status_line: Some("active_status: missing manifest for active_runtime_key=rt-gone"),
        }
    }

    #[test]
    fn listing_state_no_config() -> Result<()> {
        assert_listing_state(&state_no_config())
    }

    #[test]
    fn listing_state_key_exact() -> Result<()> {
        assert_listing_state(&state_key_exact())
    }

    #[test]
    fn listing_state_key_in_other_case() -> Result<()> {
        assert_listing_state(&state_key_in_other_case())
    }

    #[test]
    fn listing_state_stale_key_default_resolves() -> Result<()> {
        assert_listing_state(&state_stale_key_default_resolves())
    }

    #[test]
    fn listing_state_stale_key_default_missing() -> Result<()> {
        assert_listing_state(&state_stale_key_default_missing())
    }

    #[test]
    fn listing_state_stale_key_default_ambiguous() -> Result<()> {
        assert_listing_state(&state_stale_key_default_ambiguous())
    }

    #[test]
    fn listing_state_default_matches_none() -> Result<()> {
        assert_listing_state(&state_default_matches_none())
    }

    #[test]
    fn listing_state_default_matches_one() -> Result<()> {
        assert_listing_state(&state_default_matches_one())
    }

    #[test]
    fn listing_state_default_matches_one_in_other_case() -> Result<()> {
        assert_listing_state(&state_default_matches_one_in_other_case())
    }

    #[test]
    fn listing_state_default_matches_two() -> Result<()> {
        assert_listing_state(&state_default_matches_two())
    }

    #[test]
    fn listing_state_default_matches_two_differing_in_case() -> Result<()> {
        assert_listing_state(&state_default_matches_two_differing_in_case())
    }

    #[test]
    fn listing_state_case_twins_key_names_the_second() -> Result<()> {
        assert_listing_state(&state_case_twins_key_names_the_second())
    }

    #[test]
    fn listing_state_case_twins_key_names_the_first() -> Result<()> {
        assert_listing_state(&state_case_twins_key_names_the_first())
    }

    #[test]
    fn listing_state_previous_key_exact() -> Result<()> {
        assert_listing_state(&state_previous_key_exact())
    }

    #[test]
    fn listing_state_previous_key_in_other_case() -> Result<()> {
        assert_listing_state(&state_previous_key_in_other_case())
    }

    #[test]
    fn listing_state_previous_key_is_a_runtime_id() -> Result<()> {
        assert_listing_state(&state_previous_key_is_a_runtime_id())
    }

    #[test]
    fn listing_state_previous_key_with_whitespace() -> Result<()> {
        assert_listing_state(&state_previous_key_with_whitespace())
    }

    #[test]
    fn listing_state_previous_key_missing() -> Result<()> {
        assert_listing_state(&state_previous_key_missing())
    }

    #[test]
    fn listing_state_previous_key_ambiguous_case_twins() -> Result<()> {
        assert_listing_state(&state_previous_key_ambiguous_case_twins())
    }

    #[test]
    fn listing_state_previous_key_ambiguous_runtime_id() -> Result<()> {
        assert_listing_state(&state_previous_key_ambiguous_runtime_id())
    }

    #[test]
    fn listing_state_previous_key_names_one_case_twin_exactly() -> Result<()> {
        assert_listing_state(&state_previous_key_names_one_case_twin_exactly())
    }

    #[test]
    fn listing_state_stale_key_falls_back_to_the_rollback_target() -> Result<()> {
        assert_listing_state(&state_stale_key_falls_back_to_the_rollback_target())
    }

    /// A data dir whose path is not valid UTF-8 still lists as JSON: the
    /// paths are written lossily, as the text form writes them, instead of
    /// failing serialization and leaving stdout empty.
    #[cfg(unix)]
    #[test]
    fn runtimes_list_json_writes_a_non_utf8_path_lossily() -> Result<()> {
        use std::os::unix::ffi::OsStrExt as _;
        let data_dir = PathBuf::from(std::ffi::OsStr::from_bytes(b"/data-\xff"));
        let mut runtime = listed_runtime_manifest("rt-a", ID_A);
        runtime.install_root = data_dir.join("rt-a");
        let listing = listing_from_manifests(
            &RocmCliConfig::default(),
            vec![runtime],
            data_dir.join("registry"),
            data_dir.join("active.json"),
        );
        let json = serde_json::to_value(&listing)?;
        assert_eq!(json["registry"], "/data-\u{fffd}/registry", "{json:#}");
        assert_eq!(json["marker"], "/data-\u{fffd}/active.json", "{json:#}");
        assert_eq!(
            listed_runtime(&json, "rt-a")["install_root"],
            "/data-\u{fffd}/rt-a",
            "{json:#}"
        );
        Ok(())
    }

    /// The text prints `toolchain=` from `RuntimeToolchain::as_str`, which is
    /// `toolchain_state_text`, the word `rocm examine` prints; the JSON prints
    /// the serde `lowercase` rename. Both are documented as `included` and
    /// `excluded`, so a change to either word fails here instead of silently
    /// changing the JSON. As for the mode test below, a new variant must be
    /// added to `all` as well as to the match.
    #[test]
    fn runtime_toolchain_text_and_json_words_agree_for_every_variant() -> Result<()> {
        let all = [RuntimeToolchain::Included, RuntimeToolchain::Excluded];
        for toolchain in all {
            match toolchain {
                RuntimeToolchain::Included | RuntimeToolchain::Excluded => {}
            }
            assert_eq!(
                serde_json::to_value(toolchain)?,
                toolchain.as_str(),
                "{toolchain:?}"
            );
        }
        assert_eq!(
            RuntimeToolchain::from_includes_devel(true).as_str(),
            "included"
        );
        assert_eq!(
            RuntimeToolchain::from_includes_devel(false).as_str(),
            "excluded"
        );
        Ok(())
    }

    /// The text prints `mode=` from `RuntimeMode::as_str` and the JSON from the
    /// serde `kebab-case` rename; both are documented as the same words, so
    /// every variant must agree. The exhaustive `match` turns a new variant
    /// into a compile error here, which forces a look at this test, but it
    /// does not add the variant to `all`: whoever adds the match arm must add
    /// the variant to `all` too.
    #[test]
    fn runtime_mode_text_and_json_words_agree_for_every_variant() -> Result<()> {
        let all = [RuntimeMode::Managed, RuntimeMode::ReadOnly];
        for mode in all {
            match mode {
                RuntimeMode::Managed | RuntimeMode::ReadOnly => {}
            }
            assert_eq!(serde_json::to_value(mode)?, mode.as_str(), "{mode:?}");
        }
        Ok(())
    }

    /// The documented shape of `rocm runtimes list --json` (docs/json-output.md):
    /// exactly these top-level and per-runtime keys, the registry and marker
    /// paths, each runtime's own install root, and newest install first with
    /// ties broken by `runtime_key`.
    #[test]
    fn runtimes_list_json_has_exactly_the_documented_fields_newest_install_first() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-contract");
        let mut install_roots = std::collections::BTreeMap::new();
        // Five runtimes share one install time and are planted in reverse key
        // order, so the order the registry is read in cannot pass for the
        // `runtime_key` tie-break by accident.
        for (key, build) in [
            ("rt-old", 10),
            ("rt-tie-e", 20),
            ("rt-tie-d", 20),
            ("rt-new", 30),
            ("rt-tie-c", 20),
            ("rt-tie-b", 20),
            ("rt-tie-a", 20),
        ] {
            let manifest = write_test_pip_runtime(
                &paths,
                key,
                "therock-release:gfx120X-all",
                "7.13.0",
                build,
            )?;
            install_roots.insert(key, manifest.install_root);
        }
        let listing = runtimes_list_json(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(&root);

        // Compared as sets: the key order of a `serde_json::Map` depends on
        // whether its `preserve_order` feature is on, and is not documented.
        let key_set = |value: &serde_json::Value| -> std::collections::BTreeSet<String> {
            value
                .as_object()
                .expect("an object")
                .keys()
                .cloned()
                .collect()
        };
        assert_eq!(
            key_set(&listing),
            std::collections::BTreeSet::from(
                [
                    "default_runtime_id",
                    "active_runtime_key",
                    "previous_runtime_key",
                    "registry",
                    "marker",
                    "active_problem",
                    "runtimes",
                ]
                .map(String::from)
            )
        );
        assert_eq!(
            listing["registry"],
            serde_json::to_value(runtime_registry_dir(&paths))?
        );
        assert_eq!(
            listing["marker"],
            serde_json::to_value(active_runtime_marker_path(&paths))?
        );

        let runtimes = listing["runtimes"]
            .as_array()
            .expect("runtimes is an array");
        let order: Vec<&str> = runtimes
            .iter()
            .map(|runtime| runtime["runtime_key"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(
            order,
            [
                "rt-new", "rt-tie-a", "rt-tie-b", "rt-tie-c", "rt-tie-d", "rt-tie-e", "rt-old",
            ]
        );
        for runtime in runtimes {
            assert_eq!(
                key_set(runtime),
                std::collections::BTreeSet::from(
                    [
                        "runtime_key",
                        "runtime_id",
                        "version",
                        "format",
                        "family",
                        "mode",
                        "status",
                        "unusable_reason",
                        "toolchain",
                        "active",
                        "rollback_target",
                        "install_root",
                        "provenance",
                    ]
                    .map(String::from)
                )
            );
            assert_eq!(
                key_set(&runtime["provenance"]),
                std::collections::BTreeSet::from(
                    [
                        "channel",
                        "source",
                        "source_layout_generation",
                        "device_target",
                    ]
                    .map(String::from)
                )
            );
            let key = runtime["runtime_key"].as_str().unwrap_or_default();
            assert_eq!(
                runtime["install_root"],
                serde_json::to_value(&install_roots[key])?,
                "{key}"
            );
        }
        Ok(())
    }

    /// Plants the runtimes whose provenance the JSON tests read: one per way a
    /// fact can be recorded or not. Returns their keys, in this order: a
    /// nightly that recorded every fact, a runtime from before the layout and
    /// target were recorded, an adopted one (the adoption marker is not a
    /// source), a tarball install (no exact target), one whose values are
    /// padded with whitespace, and one whose every value is blank.
    fn plant_provenance_runtimes(paths: &AppPaths) -> Result<[String; 6]> {
        let base = write_test_pip_runtime(
            paths,
            "release-pip-legacy",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let legacy = therock::InstalledRuntimeManifest {
            index_url: None,
            source_layout_generation: None,
            wheel_composition: None,
            ..base.clone()
        };
        let nightly = nightly_exact_target_manifest(base);
        write_test_pip_runtime(
            paths,
            &nightly.runtime_key,
            &nightly.runtime_id,
            &nightly.version,
            20,
        )?;
        let adopted = therock::InstalledRuntimeManifest {
            runtime_key: "adopted-release-pip".to_owned(),
            index_url: None,
            selected_artifact_url: crate::runtime_registry::ADOPTED_RUNTIME_ARTIFACT_URL.to_owned(),
            read_only: true,
            ..legacy.clone()
        };
        let tarball = therock::InstalledRuntimeManifest {
            runtime_key: "release-tarball-gfx120x-all-7-13-0".to_owned(),
            format: "tarball".to_owned(),
            selected_artifact_url: "https://example.invalid/tarball/rocm-7.13.0.tar.gz".to_owned(),
            source_layout_generation: Some("multi-arch-v2".to_owned()),
            ..legacy.clone()
        };
        let padded = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-padded".to_owned(),
            channel: " release ".to_owned(),
            index_url: Some(" https://example.invalid/simple ".to_owned()),
            source_layout_generation: Some(" multi-arch-v2 ".to_owned()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "multi-arch-v2".to_owned(),
                package_specs: vec!["rocm[libraries]==7.13.0".to_owned()],
                rocm_sdk_target: Some(" gfx1201 ".to_owned()),
            }),
            ..legacy.clone()
        };
        let all_blank = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-all-blank".to_owned(),
            channel: String::new(),
            index_url: Some(" ".to_owned()),
            selected_artifact_url: String::new(),
            source_layout_generation: Some(String::new()),
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: String::new(),
                package_specs: vec!["rocm[libraries]==7.13.0".to_owned()],
                rocm_sdk_target: Some(String::new()),
            }),
            ..legacy.clone()
        };
        let planted = [nightly, legacy, adopted, tarball, padded, all_blank];
        for manifest in &planted {
            rewrite_manifest(paths, manifest)?;
        }
        Ok(planted.map(|manifest| manifest.runtime_key))
    }

    /// The JSON names each runtime's provenance facts as strings, trimmed, and
    /// writes `null` for one recorded nowhere or recorded blank — never the
    /// text's `unrecorded` placeholder, which a script would read as a value.
    /// Each object is compared whole, so a fact under the wrong name, a missing
    /// one or a placeholder in place of `null` fails here.
    #[test]
    fn runtimes_list_json_names_recorded_provenance_and_null_for_unrecorded() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-provenance");
        let [nightly, legacy, adopted, tarball, padded, all_blank] =
            plant_provenance_runtimes(&paths)?;
        let listing = runtimes_list_json(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        let provenance = |key: &str| listed_runtime(&listing, key)["provenance"].clone();
        for (key, expected) in [
            (
                nightly,
                serde_json::json!({
                    "channel": "nightly",
                    "source": "https://example.invalid/whl-multi-arch",
                    "source_layout_generation": "next-v1",
                    "device_target": "gfx1201",
                }),
            ),
            (
                legacy,
                serde_json::json!({
                    "channel": "release",
                    "source": "https://example.invalid/therock",
                    "source_layout_generation": null,
                    "device_target": null,
                }),
            ),
            (
                adopted,
                serde_json::json!({
                    "channel": "release",
                    "source": null,
                    "source_layout_generation": null,
                    "device_target": null,
                }),
            ),
            (
                tarball,
                serde_json::json!({
                    "channel": "release",
                    "source": "https://example.invalid/tarball/rocm-7.13.0.tar.gz",
                    "source_layout_generation": "multi-arch-v2",
                    "device_target": null,
                }),
            ),
            (
                padded,
                serde_json::json!({
                    "channel": "release",
                    "source": "https://example.invalid/simple",
                    "source_layout_generation": "multi-arch-v2",
                    "device_target": "gfx1201",
                }),
            ),
            (
                all_blank,
                serde_json::json!({
                    "channel": null,
                    "source": null,
                    "source_layout_generation": null,
                    "device_target": null,
                }),
            ),
        ] {
            assert_eq!(provenance(&key), expected, "{key}:\n{listing:#}");
        }
        Ok(())
    }

    /// The text's `provenance:` line and the JSON's `provenance` object name
    /// the same facts with the same values for every runtime, the text printing
    /// `unrecorded` exactly where the JSON has `null`. Both forms are rendered
    /// from one listing of the same registry and compared fact by fact, over
    /// every way a fact can be recorded or not.
    #[test]
    fn runtimes_list_text_and_json_name_the_same_provenance() -> Result<()> {
        use cli_report::JsonOrText as _;
        let (root, paths) = test_paths("runtime-list-provenance-agree");
        let keys = plant_provenance_runtimes(&paths)?;
        let listing = runtimes_listing(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);
        let text = listing.render_text();
        let json = serde_json::to_value(&listing)?;

        for key in keys {
            let text_pairs: Vec<(String, String)> = provenance_line(&text, &key)
                .trim_start()
                .strip_prefix("provenance: ")
                .expect("provenance prefix")
                .split(' ')
                .map(|pair| pair.split_once('=').expect("name=value pair"))
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
                .collect();
            let object = listed_runtime(&json, &key)["provenance"]
                .as_object()
                .unwrap_or_else(|| panic!("{key} has no provenance object:\n{json:#}"))
                .clone();
            assert_eq!(
                text_pairs.len(),
                object.len(),
                "{key}: the text and the JSON name different facts:\n{text}\n{json:#}"
            );
            for (name, shown) in text_pairs {
                let in_json = object
                    .get(&name)
                    .unwrap_or_else(|| panic!("{key}: the JSON has no {name}:\n{json:#}"));
                let expected = match in_json {
                    serde_json::Value::Null => crate::runtime_registry::UNRECORDED_PROVENANCE,
                    serde_json::Value::String(value) => value.as_str(),
                    other => panic!("{key}: {name} is neither a string nor null: {other}"),
                };
                assert_eq!(shown, expected, "{key}: {name}:\n{text}\n{json:#}");
            }
        }
        Ok(())
    }

    /// A runtime can be both the active one and the recorded rollback target
    /// when the config says so (`rocm runtimes activate` and `rollback` never
    /// write that, but the config file is read as it is). Both JSON flags are true;
    /// the text has one marker column, and active wins it.
    #[test]
    fn runtimes_list_marks_a_runtime_both_active_and_rollback_target_as_active() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-active-and-rollback");
        write_test_pip_runtime(
            &paths,
            "rt-both",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some("rt-both".to_owned()),
            previous_runtime_key: Some("rt-both".to_owned()),
            ..RocmCliConfig::default()
        };
        let listing = runtimes_list_json(&paths, &config)?;
        let text = render_runtimes_text(&paths, &config)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(runtime_keys_flagged(&listing, "active"), ["rt-both"]);
        assert_eq!(
            runtime_keys_flagged(&listing, "rollback_target"),
            ["rt-both"]
        );
        assert_eq!(
            runtime_keys_marked(&text, ACTIVE_RUNTIME_MARKER),
            ["rt-both"]
        );
        assert!(
            runtime_keys_marked(&text, ROLLBACK_RUNTIME_MARKER).is_empty(),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn runtimes_list_json_reports_a_runtime_whose_install_root_is_gone_as_unusable() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-unusable");
        let manifest = write_test_pip_runtime(
            &paths,
            "rt-gone",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        fs::remove_dir_all(&manifest.install_root)?;
        let listing = runtimes_list_json(&paths, &RocmCliConfig::default())?;
        let text = render_runtimes_text(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        let runtime = listed_runtime(&listing, "rt-gone");
        assert_eq!(runtime["status"], "unusable");
        let reason = runtime["unusable_reason"]
            .as_str()
            .expect("an unusable runtime says why");
        assert!(reason.starts_with("install root is missing: "), "{reason}");
        assert!(
            text.contains(&format!("status=unusable ({reason}) toolchain=")),
            "the text shows the same reason:\n{text}"
        );
        Ok(())
    }

    #[test]
    fn runtimes_list_json_reports_an_imported_runtime_as_read_only() -> Result<()> {
        let (root, paths) = test_paths("runtime-list-json-read-only");
        let manifest = write_test_pip_runtime(
            &paths,
            "rt-imported",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let imported = therock::InstalledRuntimeManifest {
            read_only: true,
            ..manifest
        };
        fs::write(
            runtime_manifest_path(&paths, &imported.runtime_key),
            serde_json::to_vec_pretty(&imported)?,
        )?;
        let listing = runtimes_list_json(&paths, &RocmCliConfig::default())?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(listed_runtime(&listing, "rt-imported")["mode"], "read-only");
        Ok(())
    }

    #[cfg(unix)]
    fn test_active_runtime_marker(paths: &AppPaths, runtime_key: &str) -> ActiveRuntimeMarker {
        ActiveRuntimeMarker {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: runtime_key.to_owned(),
            manifest_path: runtime_manifest_path(paths, runtime_key),
            install_root: paths.data_dir.join("runtimes").join(runtime_key),
            previous_runtime_id: None,
            previous_runtime_key: None,
            activated_at_unix_ms: rocm_core::unix_time_millis(),
        }
    }

    #[cfg(unix)]
    fn leftover_marker_temp_files(paths: &AppPaths) -> Result<Vec<String>> {
        let dir = active_runtime_marker_path(paths)
            .parent()
            .context("marker has a parent")?
            .to_path_buf();
        let mut leftovers = Vec::new();
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if name.starts_with("active.json.tmp-") {
                leftovers.push(name);
            }
        }
        Ok(leftovers)
    }

    /// Two activations writing the marker at once must both succeed, and a
    /// reader must find a complete marker throughout. #605 reported a writer
    /// that removed `active.json` before renaming the replacement in, and two
    /// writers in the same millisecond sharing one temp file, so one failed
    /// with "failed to move active runtime marker" and the other could publish
    /// the shorter payload followed by the longer one's tail. #424 fixed that;
    /// this change routes the marker through `rocm_core::atomic_write`, the
    /// writer `rocm` and `rocmd` now share, and this test pins that the marker
    /// keeps the property through it.
    ///
    /// The bug is a race, so against the old code this fails with high
    /// probability per round rather than deterministically; against the
    /// current writer nothing in it is timing-dependent. Unix only: the
    /// property under test is that `rename` replaces in one step. The Windows
    /// publish tries `rename` first too, but this test has not been run
    /// against it there, nor against the `ReplaceFileW` fallback.
    #[cfg(unix)]
    #[test]
    fn concurrent_active_runtime_marker_writes_keep_a_complete_marker_published() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier};

        const ROUNDS: usize = 200;
        let (root, paths) = test_paths("marker-concurrent");
        write_active_runtime_marker(&paths, test_active_runtime_marker(&paths, "initial"))?;
        let marker_path = active_runtime_marker_path(&paths);

        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let done = Arc::clone(&done);
            let marker_path = marker_path.clone();
            std::thread::spawn(move || {
                let mut problems = Vec::new();
                while !done.load(Ordering::Acquire) {
                    match fs::read(&marker_path) {
                        Ok(bytes) => {
                            if let Err(error) =
                                serde_json::from_slice::<ActiveRuntimeMarker>(&bytes)
                            {
                                problems.push(format!("unparsable marker: {error}"));
                            }
                        }
                        Err(error) => problems.push(format!("marker unreadable: {error}")),
                    }
                }
                problems
            })
        };

        let barrier = Arc::new(Barrier::new(2));
        let writers: Vec<_> = [
            "short",
            "a-runtime-key-long-enough-to-leave-a-tail-behind-a-shorter-payload",
        ]
        .into_iter()
        .map(|runtime_key| {
            let barrier = Arc::clone(&barrier);
            let paths = paths.clone();
            std::thread::spawn(move || {
                let mut errors = Vec::new();
                for _ in 0..ROUNDS {
                    barrier.wait();
                    if let Err(error) = write_active_runtime_marker(
                        &paths,
                        test_active_runtime_marker(&paths, runtime_key),
                    ) {
                        errors.push(format!("{error:#}"));
                    }
                }
                errors
            })
        })
        .collect();

        let mut writer_errors = Vec::new();
        for writer in writers {
            writer_errors.extend(writer.join().expect("writer thread panicked"));
        }
        done.store(true, Ordering::Release);
        let reader_problems = reader.join().expect("reader thread panicked");

        let published: ActiveRuntimeMarker = serde_json::from_slice(&fs::read(&marker_path)?)?;
        let leftovers = leftover_marker_temp_files(&paths)?;
        let _ = fs::remove_dir_all(root);

        assert!(
            writer_errors.is_empty(),
            "{} of {} marker writes failed; first: {:?}",
            writer_errors.len(),
            2 * ROUNDS,
            writer_errors.first()
        );
        assert!(
            reader_problems.is_empty(),
            "a reader found no complete marker {} times; first: {:?}",
            reader_problems.len(),
            reader_problems.first()
        );
        assert!(
            [
                "short",
                "a-runtime-key-long-enough-to-leave-a-tail-behind-a-shorter-payload"
            ]
            .contains(&published.runtime_key.as_str()),
            "unexpected marker: {published:?}"
        );
        assert_eq!(leftovers, Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn runtime_activation_records_exact_key_and_rollback() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-12-0",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;

        let mut config = RocmCliConfig::default();
        let first = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-12-0")?;
        assert_eq!(first.previous_runtime_key, None);
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let second = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-13-0")?;
        assert_eq!(
            second.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let marker: ActiveRuntimeMarker =
            serde_json::from_slice(&fs::read(active_runtime_marker_path(&paths))?)?;
        assert_eq!(marker.runtime_key, "release-pip-gfx120x-all-7-13-0");
        assert_eq!(
            marker.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("* release-pip-gfx120x-all-7-13-0"));
        assert!(rendered.contains("- release-pip-gfx120x-all-7-12-0"));
        assert!(rendered.contains("status=ready"));

        let rolled_back = rollback_runtime(&paths, &mut config)?;
        assert_eq!(rolled_back.runtime_key, "release-pip-gfx120x-all-7-12-0");
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0")
        );

        // Rolling back a second time toggles back to where the first rollback
        // came from — this is the guarantee the `rollback --help` text makes
        // ("it remembers only the runtime you just left"). This test fails the
        // moment that toggle stops holding.
        let rolled_back_again = rollback_runtime(&paths, &mut config)?;
        assert_eq!(
            rolled_back_again.runtime_key,
            "release-pip-gfx120x-all-7-13-0"
        );
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some("release-pip-gfx120x-all-7-12-0")
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Register both runtimes and make `OLD_RUNTIME_KEY` the active one, which
    /// is the state every "switch away from it" test starts from.
    fn activate_old_runtime(paths: &AppPaths) -> Result<RocmCliConfig> {
        write_test_pip_runtime(
            paths,
            OLD_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            paths,
            NEW_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        activate_runtime(paths, &mut config, OLD_RUNTIME_KEY)?;
        Ok(config)
    }

    /// Replace the active-runtime marker with a NON-EMPTY directory, so the
    /// next `write_active_runtime_marker` fails at a predictable point: the
    /// temp file is staged beside it, and publishing that file onto a
    /// non-empty directory fails on both supported platforms without needing
    /// privileges or a real full disk.
    fn break_active_runtime_marker(paths: &AppPaths) -> Result<PathBuf> {
        let marker = active_runtime_marker_path(paths);
        let _ = fs::remove_file(&marker);
        fs::create_dir_all(&marker)?;
        let sentinel = marker.join("occupied.txt");
        fs::write(&sentinel, "not a marker")?;
        Ok(sentinel)
    }

    /// Names of any `active.json.tmp-*` files left beside the marker.
    fn marker_temp_file_names(paths: &AppPaths) -> Result<Vec<String>> {
        let marker = active_runtime_marker_path(paths);
        let dir = marker
            .parent()
            .context("active runtime marker path has no parent directory")?;
        Ok(fs::read_dir(dir)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect())
    }

    #[test]
    fn a_failed_activation_leaves_an_unreadable_marker_alone() -> Result<()> {
        let (root, paths) = test_paths("activation-snapshot-unreadable-marker");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let marker_path = active_runtime_marker_path(&paths);
        // A marker the snapshot cannot read. A directory is the portable way to
        // make `fs::read` fail with something other than NotFound on both
        // supported platforms; a permission-denied file would do the same, and
        // is what this stands in for.
        fs::remove_file(&marker_path)?;
        fs::create_dir_all(&marker_path)?;

        let snapshot = ActivationSnapshot::capture(&paths, &config);
        assert!(
            matches!(snapshot.marker, MarkerSnapshot::Unreadable),
            "a marker that is there but unreadable is not an absent marker"
        );
        // Whatever the failed activation is being rolled back from, the restore
        // must not act on bytes it never captured.
        fs::remove_dir_all(&marker_path)?;
        fs::write(
            &marker_path,
            b"{\"runtime_key\":\"written-by-someone-else\"}",
        )?;

        snapshot.restore(&paths, &RocmCliConfig::lock(&paths)?, &mut config)?;

        let survived = fs::read(&marker_path)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            survived,
            b"{\"runtime_key\":\"written-by-someone-else\"}".to_vec(),
            "the snapshot holds no bytes for this marker, so deleting it would \
             destroy state on the very path that exists to put state back"
        );
        Ok(())
    }

    #[test]
    fn a_failed_config_save_leaves_the_caller_holding_the_old_runtime() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-config-save-failure");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // Two activations before the failure, not one: with a single activation
        // `previous_runtime_key` is already `None`, so the assertion below would
        // hold even if `restore_in_memory` never put the field back. Switching
        // once gives it a value that a missed restore would visibly change.
        activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        // The config save is the FIRST of an activation's two writes, so failing
        // it never reaches the marker-failure recovery path. The activation
        // loads the settings itself under the config lock, so the file has to
        // stay readable: the save is failed through the lock's test seam.
        rocm_core::config::fail_config_lock_saves_on_this_thread(true);
        let outcome = activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY);
        rocm_core::config::fail_config_lock_saves_on_this_thread(false);
        let _ = fs::remove_dir_all(&root);
        let error = outcome.expect_err("a failed config save must fail the activation");

        let error = format!("{error:#}");
        assert!(
            error.contains("the previously active runtime is still the active one"),
            "nothing reached disk, so the error must say which runtime is \
             active:\n{error}"
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(NEW_RUNTIME_KEY),
            "the caller goes on using the settings it is handed back, so \
             settings naming the runtime that was NOT activated would \
             misreport which one is active"
        );
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the rollback target belongs to the activation that succeeded, not \
             to the one that failed before it wrote anything"
        );
        Ok(())
    }

    /// The live-server check, which asks each running server over HTTP and can
    /// take seconds per server, runs before the config lock is taken, so other
    /// settings changes do not wait for it. Checked by holding the lock on this
    /// thread: a check made under the lock would fail on the nested lock
    /// instead of on the unreadable services folder.
    #[test]
    fn activation_and_rollback_check_live_servers_before_taking_the_config_lock() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-services-before-lock");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        fs::create_dir_all(paths.services_dir().join("unreadable.json"))?;

        let held = RocmCliConfig::lock(&paths)?;
        let activation = activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY).map(drop);
        let rollback = rollback_runtime(&paths, &mut config).map(drop);
        drop(held);
        let _ = fs::remove_dir_all(&root);

        for (what, outcome) in [("activation", activation), ("rollback", rollback)] {
            let message = format!(
                "{:#}",
                outcome.expect_err("the services folder is unreadable")
            );
            assert!(
                message.contains("the local server records could not be read")
                    && !message.contains("already holds the lock"),
                "{what} checked the live servers under the config lock: {message}"
            );
        }
        Ok(())
    }

    /// An activation and a rollback start from the settings on disk, not from
    /// the copy the caller holds: a setting another command saved after that
    /// copy was read survives, and the caller gets the settings as saved.
    #[test]
    fn activation_and_rollback_keep_a_setting_saved_after_the_callers_copy() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-keeps-newer-setting");
        paths.ensure()?;
        let mut stale = activate_old_runtime(&paths)?;
        RocmCliConfig::update(&paths, |config| {
            config.telemetry.mode = rocm_core::TELEMETRY_MODE_OFF.to_owned();
            Ok(())
        })?;
        activate_runtime(&paths, &mut stale, NEW_RUNTIME_KEY)?;
        let after_activation = RocmCliConfig::load(&paths)?;
        let handed_back = serde_json::to_value(&stale)?;

        let mut stale_again = stale.clone();
        RocmCliConfig::update(&paths, |config| {
            config.default_engine = Some("vllm".to_owned());
            Ok(())
        })?;
        rollback_runtime(&paths, &mut stale_again)?;
        let after_rollback = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            after_activation.telemetry.mode,
            rocm_core::TELEMETRY_MODE_OFF
        );
        assert_eq!(
            after_activation.active_runtime_key.as_deref(),
            Some(NEW_RUNTIME_KEY)
        );
        assert_eq!(handed_back, serde_json::to_value(&after_activation)?);
        assert_eq!(after_rollback.default_engine.as_deref(), Some("vllm"));
        assert_eq!(after_rollback.telemetry.mode, rocm_core::TELEMETRY_MODE_OFF);
        assert_eq!(
            after_rollback.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        Ok(())
    }

    /// The rollback checks the live servers for the target it read before
    /// taking the lock. If another command changed the target in between, it
    /// refuses and changes nothing, and running `rocm runtimes rollback` again,
    /// as the error says, rolls back to the new target.
    #[test]
    fn a_rollback_whose_target_changed_meanwhile_is_refused() -> Result<()> {
        let (root, paths) = test_paths("runtime-rollback-target-changed");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;

        let lock = RocmCliConfig::lock(&paths)?;
        let mut current = lock.load()?;
        let refused = rollback_runtime_locked(
            &paths,
            &lock,
            &mut current,
            "the-target-checked-earlier",
            runtime_services::RuntimeServiceReconciliation::default(),
        )
        .map(drop);
        drop(lock);
        let after_refusal = RocmCliConfig::load(&paths)?;
        let again = rollback_runtime(&paths, &mut config).map(drop);
        let after_again = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        let message = format!("{:#}", refused.expect_err("the target changed"));
        assert!(
            message.contains("changed the rollback target")
                && message.contains("run `rocm runtimes rollback` again"),
            "{message}"
        );
        assert_eq!(
            after_refusal.active_runtime_key.as_deref(),
            Some(NEW_RUNTIME_KEY)
        );
        assert_eq!(
            after_refusal.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        again?;
        assert_eq!(
            after_again.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        Ok(())
    }

    #[test]
    fn runtime_activation_reports_live_service_on_previous_runtime() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-live-service");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // The current process is a guaranteed-live pid, and "starting" keeps the
        // liveness refresh from probing an endpoint nothing is serving — the
        // record stays live without a server behind it.
        plant_service_record_on_runtime(
            &paths,
            "svc-on-old-runtime",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;

        let result = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;

        assert_eq!(
            result.services.stale,
            vec![runtime_services::RuntimeServiceEntry {
                service_id: "svc-on-old-runtime".to_owned(),
                engine: "vllm".to_owned(),
                recorded_runtime: Some(OLD_RUNTIME_KEY.to_owned()),
            }],
            "a live service still recording the previous runtime is exactly what \
             the activation report exists to name"
        );
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert!(
            rendered.contains("services_on_previous_runtime: 1"),
            "the report must count what it found:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "- svc-on-old-runtime engine=vllm recorded_runtime={OLD_RUNTIME_KEY}"
            )),
            "the report must name the service and the runtime it is still on:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "run `rocm runtimes activate {NEW_RUNTIME_KEY} --restart-services --yes` to move \
                 them"
            )),
            "having found something to move, the report must name the exact \
             invocation that moves it — the reports printed by `rocm install \
             sdk` and `rocm update` reach commands that have no such flag at \
             all:\n{rendered}"
        );
        assert!(
            !rendered.contains("running services keep their recorded runtime"),
            "the fixed note this replaced was printed whether or not any service \
             existed; it must not survive alongside the derived report:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn runtime_activation_ignores_stopped_services() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-stopped-service");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // A stopped service holds no runtime: it will pick up whatever is active
        // when it is next launched, so reporting it would send the user chasing
        // a restart that changes nothing.
        plant_service_record_on_runtime(
            &paths,
            "svc-stopped",
            "vllm",
            "stopped",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;

        let result = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert!(result.services.stale.is_empty(), "{:?}", result.services);
        assert!(result.services.unknown.is_empty(), "{:?}", result.services);
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "a clean switch must still print the count, so it reads differently \
             from a report that never looked:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn runtime_activation_ignores_self_managed_engine_services() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-self-managed");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // lemonade pins its own embeddable runtime, which is never a ROCm
        // runtime key — comparing it against one would report every lemonade
        // server as permanently stale, on every activation, forever.
        plant_service_record_on_runtime(
            &paths,
            "svc-lemonade",
            "lemonade",
            "starting",
            std::process::id(),
            Some("lemonade-embeddable-8.1.11"),
            None,
        )?;

        let result = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert!(result.services.stale.is_empty(), "{:?}", result.services);
        assert!(result.services.unknown.is_empty(), "{:?}", result.services);
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "a self-managed engine service is not moved by a ROCm runtime switch:\n{rendered}"
        );
        Ok(())
    }

    /// A vLLM record whose engine state file carries a `runtime_id` — the way a
    /// real engine always writes it — and also has an `env_id` is classified by
    /// that runtime like any other record, so it is reported when it differs
    /// from the one being activated. Treating `env_id` as a pin would classify
    /// it `Matches` and never report it, which makes this test fail.
    #[test]
    fn runtime_activation_reports_env_pinned_service_the_engine_recorded_a_runtime_for()
    -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-env-pinned-with-runtime-id");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // Plant a vLLM record with an env_id (as `--env-id custom-env` would
        // produce) but WITHOUT a runtime_id in the in-memory record, mirroring
        // what `plant_service_record_on_runtime` gives us when runtime_id is
        // None. Then overwrite the engine state file with a realistic payload
        // that includes a non-empty `runtime_id`, exactly as `write_running_state`
        // in the engine emits it — this is what `refresh_from_engine_state` reads
        // and adopts into `record.runtime_id`.
        let record = plant_service_record_on_runtime(
            &paths,
            "svc-env-pinned-with-runtime",
            "vllm",
            "starting",
            std::process::id(),
            None,
            Some("custom-env"),
        )?;
        // Overwrite the engine state file with a realistic payload that carries
        // a non-empty `runtime_id`. `refresh_from_engine_state` reads this and
        // sets `record.runtime_id`, which is then what `classify_service_runtime_state`
        // sees — so the record is judged by its runtime, and the `env_id` it
        // also carries does not hide it.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "starting",
                "runtime_id": OLD_RUNTIME_KEY,
            }))?,
        )?;

        let result = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert!(
            result
                .services
                .stale
                .iter()
                .any(|e| e.service_id == "svc-env-pinned-with-runtime"),
            "a vLLM service whose engine state file records a runtime must appear \
             in stale, not be silently excluded because it also has an env_id:\n{rendered}"
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 1"),
            "the report must count the env-pinned service that the engine \
             recorded a runtime for:\n{rendered}"
        );
        Ok(())
    }

    /// A live service whose engine state file is absent or unreadable has no
    /// recorded runtime, so it cannot be classified as stale. It must appear
    /// under `services_with_unrecorded_runtime` (Unknown), must NOT be counted
    /// in `services_on_previous_runtime`, and must NOT be touched by
    /// `restart_stale_runtime_services`, which iterates `services.stale` only.
    #[test]
    fn runtime_activation_reports_service_with_no_recorded_runtime_as_unknown() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-unknown-runtime");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        // Plant a record with no runtime_id; the engine state file written by
        // `plant_service_record_on_runtime` contains only `{"status": "starting"}`,
        // so `refresh_from_engine_state` finds no `runtime_id` to adopt.
        plant_service_record_on_runtime(
            &paths,
            "svc-no-runtime",
            "vllm",
            "starting",
            std::process::id(),
            None,
            None,
        )?;

        let result = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert!(
            result.services.stale.is_empty(),
            "a service with no recorded runtime must not appear in stale:\n{:?}",
            result.services
        );
        assert!(
            result
                .services
                .unknown
                .iter()
                .any(|e| e.service_id == "svc-no-runtime"),
            "a service with no recorded runtime must appear in unknown:\n{:?}",
            result.services
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "unknown services are not counted in services_on_previous_runtime:\n{rendered}"
        );
        assert!(
            rendered.contains("services_with_unrecorded_runtime: 1"),
            "the report must surface the service with no recorded runtime:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn runtime_activation_rolls_back_when_marker_write_fails() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-marker-failure");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let sentinel = break_active_runtime_marker(&paths)?;

        let error = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)
            .expect_err("a failed marker write must fail the activation");

        let persisted = RocmCliConfig::load(&paths)?;
        let sentinel_survived = sentinel.is_file();
        let stranded_temp_files = marker_temp_file_names(&paths)?;
        let _ = fs::remove_dir_all(&root);

        let error = format!("{error:#}");
        assert!(
            error.contains("the previously active runtime is still the active one"),
            "the error must say the activation did not take effect:\n{error}"
        );
        assert!(
            stranded_temp_files.is_empty(),
            "the activation is recoverable and the user retries it, so a marker \
             write that failed must not strand its temp file in the folder \
             `rocm runtimes` reads: {stranded_temp_files:?}"
        );
        assert_eq!(
            persisted.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the config on disk must still name the runtime the marker names; \
             a torn write here also changes storage GC hold decisions"
        );
        assert_eq!(
            persisted.default_runtime_id.as_deref(),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(persisted.previous_runtime_key, None);
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the caller's in-memory config must be restored too, or the next \
             report from these settings names the state the restore just undid"
        );
        assert!(
            sentinel_survived,
            "whatever occupied the marker path predates this activation, so \
             restoring must leave it exactly as it was"
        );
        Ok(())
    }

    #[test]
    fn runtime_rollback_reports_live_service_on_previous_runtime() -> Result<()> {
        let (root, paths) = test_paths("runtime-rollback-live-service");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        // Serving started on the newer runtime; rolling back leaves this server
        // behind exactly the way activating forward does.
        plant_service_record_on_runtime(
            &paths,
            "svc-on-new-runtime",
            "vllm",
            "starting",
            std::process::id(),
            Some(NEW_RUNTIME_KEY),
            None,
        )?;

        let result = rollback_runtime(&paths, &mut config)?;
        let rendered = runtime_services::render_runtime_service_reconciliation(
            &result.services,
            &result.runtime_key,
        );
        let _ = fs::remove_dir_all(&root);

        assert_eq!(result.runtime_key, OLD_RUNTIME_KEY);
        assert_eq!(
            result.services.stale,
            vec![runtime_services::RuntimeServiceEntry {
                service_id: "svc-on-new-runtime".to_owned(),
                engine: "vllm".to_owned(),
                recorded_runtime: Some(NEW_RUNTIME_KEY.to_owned()),
            }]
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 1"),
            "rollback counts what it found, not only the line beneath the \
             count — without this the count could be dropped and the test \
             would stay green:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "- svc-on-new-runtime engine=vllm recorded_runtime={NEW_RUNTIME_KEY}"
            )),
            "rollback reports live services the same way activate does:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn runtime_rollback_rolls_back_when_marker_write_fails() -> Result<()> {
        let (root, paths) = test_paths("runtime-rollback-marker-failure");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        let sentinel = break_active_runtime_marker(&paths)?;

        let error = rollback_runtime(&paths, &mut config)
            .expect_err("a failed marker write must fail the rollback");

        let persisted = RocmCliConfig::load(&paths)?;
        let sentinel_survived = sentinel.is_file();
        let _ = fs::remove_dir_all(&root);

        let error = format!("{error:#}");
        assert!(
            error.contains("the previously active runtime is still the active one"),
            "the error must say the rollback did not take effect:\n{error}"
        );
        assert_eq!(
            persisted.active_runtime_key.as_deref(),
            Some(NEW_RUNTIME_KEY),
            "a failed rollback must leave the runtime it failed to leave active"
        );
        assert_eq!(
            persisted.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the rollback target must survive a failed rollback, or the user \
             loses the only state that makes a retry possible"
        );
        assert_eq!(config.active_runtime_key.as_deref(), Some(NEW_RUNTIME_KEY));
        assert!(sentinel_survived);
        Ok(())
    }

    #[test]
    fn runtime_activation_resolves_a_recorded_family_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-family-id");
        paths.ensure()?;
        // Distinct runtime_ids, so each family id names exactly one installed
        // runtime and resolves without ambiguity.
        write_test_pip_runtime(
            &paths,
            OLD_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            NEW_RUNTIME_KEY,
            "therock-release:gfx1100",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY)?;
        // `refresh_from_engine_state` adopts what the engine reports, which for
        // a TheRock runtime can be the manifest's family runtime_id rather than
        // the versioned key. A verbatim string compare against the key would
        // call this server stale while it is running on the very runtime being
        // activated — and `--restart-services` would then stop and respawn it
        // for nothing.
        plant_service_record_on_runtime(
            &paths,
            "svc-family-id",
            "vllm",
            "starting",
            std::process::id(),
            Some("therock-release:gfx1100"),
            None,
        )?;

        let matched = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?.services;

        // The same record is genuinely stale once the OTHER runtime is
        // activated, so the resolution must not simply excuse every family id.
        let stale = rollback_runtime(&paths, &mut config)?.services;
        let _ = fs::remove_dir_all(&root);

        assert!(
            matched.stale.is_empty() && matched.unknown.is_empty(),
            "the recorded family id resolves to the runtime being activated: {matched:?}"
        );
        assert_eq!(
            stale.stale,
            vec![runtime_services::RuntimeServiceEntry {
                service_id: "svc-family-id".to_owned(),
                engine: "vllm".to_owned(),
                recorded_runtime: Some("therock-release:gfx1100".to_owned()),
            }],
            "activating a different runtime does leave this server behind"
        );
        Ok(())
    }

    #[test]
    fn a_failed_activation_restores_the_marker_it_replaced() -> Result<()> {
        let (root, paths) = test_paths("activation-snapshot-restore");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let marker_path = active_runtime_marker_path(&paths);
        let before = fs::read(&marker_path)?;

        let snapshot = ActivationSnapshot::capture(&paths, &config);
        // Stand in for a half-applied activation: both halves moved on.
        config.default_runtime_id = Some("therock-release:gfx1100".to_owned());
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        config.previous_runtime_key = Some(OLD_RUNTIME_KEY.to_owned());
        config.save(&paths)?;
        fs::write(&marker_path, b"{\"runtime_key\":\"half-applied\"}")?;

        snapshot.restore(&paths, &RocmCliConfig::lock(&paths)?, &mut config)?;

        let persisted = RocmCliConfig::load(&paths)?;
        let after = fs::read(&marker_path)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            after, before,
            "restoring has to put the marker back byte for byte; the marker \
             feeds runtime resolution and the storage retention holds, so a \
             config-only restore still leaves the two disagreeing"
        );
        assert_eq!(
            persisted.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        assert_eq!(persisted.previous_runtime_key, None);
        assert_eq!(config.active_runtime_key.as_deref(), Some(OLD_RUNTIME_KEY));
        Ok(())
    }

    /// The restore's own write has to be atomic, not merely correct. Its sibling
    /// above asserts the bytes come back; swapping `write_file_atomically` for a
    /// bare `fs::write` satisfies that and leaves the arm that runs *after
    /// something already failed* able to strand a truncated marker. The inode is
    /// the observable difference, so this test reads it on Unix only; the
    /// equivalent `rocm-core` config-save test also has a Windows variant that
    /// reads the file identity instead.
    #[cfg(unix)]
    #[test]
    fn restoring_a_marker_replaces_the_file_rather_than_rewriting_it_in_place() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        let (root, paths) = test_paths("activation-snapshot-restore-atomic");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let marker_path = active_runtime_marker_path(&paths);

        let snapshot = ActivationSnapshot::capture(&paths, &config);
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        config.save(&paths)?;
        fs::write(&marker_path, b"{\"runtime_key\":\"half-applied\"}")?;
        // Read AFTER the half-applied write, so the comparison is against the
        // file the restore actually replaces.
        let before_ino = fs::metadata(&marker_path)?.ino();

        snapshot.restore(&paths, &RocmCliConfig::lock(&paths)?, &mut config)?;

        let after_ino = fs::metadata(&marker_path)?.ino();
        let leftovers: Vec<String> = fs::read_dir(
            marker_path
                .parent()
                .context("marker path has no parent directory")?,
        )?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp-"))
        .collect();
        let _ = fs::remove_dir_all(&root);

        assert_ne!(
            before_ino, after_ino,
            "the restore must publish a new file over the marker rather than \
             rewrite it in place; an in-place write can be interrupted and this \
             runs when something has already gone wrong"
        );
        assert!(
            leftovers.is_empty(),
            "a successful restore must not strand a staging file: {leftovers:?}"
        );
        Ok(())
    }

    #[test]
    fn a_failed_activation_removes_a_marker_that_was_not_there_before() -> Result<()> {
        let (root, paths) = test_paths("activation-snapshot-restore-absent");
        paths.ensure()?;
        // Captured with no marker on disk at all, which is the state a first
        // activation starts from. Restoring then means removing the marker the
        // activation wrote, not putting bytes back — the one `MarkerSnapshot`
        // arm whose `fs::remove_file` could be replaced with a no-op and go
        // unnoticed.
        let mut config = RocmCliConfig::default();
        write_test_pip_runtime(
            &paths,
            OLD_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        let marker_path = active_runtime_marker_path(&paths);
        assert!(!marker_path.exists());

        let snapshot = ActivationSnapshot::capture(&paths, &config);
        assert!(matches!(snapshot.marker, MarkerSnapshot::Absent));
        activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY)?;
        assert!(marker_path.is_file(), "the activation wrote a marker");

        snapshot.restore(&paths, &RocmCliConfig::lock(&paths)?, &mut config)?;

        let marker_still_there = marker_path.exists();
        let persisted = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            !marker_still_there,
            "there was no marker to put back, so restoring means removing the \
             one written over nothing — leaving it names a runtime the config \
             no longer claims"
        );
        assert_eq!(persisted.active_runtime_key, None);
        assert_eq!(persisted.default_runtime_id, None);
        Ok(())
    }

    /// The Windows partial-replace failure during activation: the marker publish
    /// took the old marker away and kept the new one in its temp file. Once the
    /// restore has put the previous marker back, the kept file holds the failed
    /// activation's marker, so it is removed, and the error no longer tells the
    /// user to rename it into place. Driven through the shared writer's publish
    /// seam, so it runs on every platform.
    #[test]
    fn a_restored_activation_drops_the_marker_a_partial_replace_kept() -> Result<()> {
        let (root, paths) = test_paths("activation-partial-replace");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let snapshot = ActivationSnapshot::capture(&paths, &config);
        let previous_marker = fs::read(active_runtime_marker_path(&paths))?;
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        config.previous_runtime_key = Some(OLD_RUNTIME_KEY.to_owned());
        config.save(&paths)?;

        let marker_path = active_runtime_marker_path(&paths);
        let write_error = partial_replace_marker_error(&paths, |_| Ok(()));
        assert!(
            !marker_temp_file_names(&paths)?.is_empty(),
            "the shared writer keeps the new marker when the old one is gone"
        );

        let error = restore_after_failed_activation(
            &paths,
            &RocmCliConfig::lock(&paths)?,
            &mut config,
            snapshot,
            write_error,
        );
        let rendered = format!("{error:#}");
        let restored_marker = fs::read(&marker_path)?;
        let leftovers = marker_temp_file_names(&paths)?;
        let persisted = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(restored_marker, previous_marker);
        assert_eq!(
            persisted.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        assert!(
            leftovers.is_empty(),
            "the failed activation's marker was left beside the restored one: {leftovers:?}"
        );
        assert!(
            rendered.contains("the previously active runtime is still the active one"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("rename that file"),
            "renaming would install the failed activation's marker:\n{rendered}"
        );
        Ok(())
    }

    /// Produce the error the shared writer returns when a marker publish took
    /// the old marker away and kept the new one, through its publish seam.
    /// `prepare_kept` runs on the kept temp file before the error is returned.
    fn partial_replace_marker_error(
        paths: &AppPaths,
        prepare_kept: impl FnOnce(&std::path::Path) -> std::io::Result<()>,
    ) -> anyhow::Error {
        let marker_path = active_runtime_marker_path(paths);
        rocm_core::atomic_write::write_file_atomically_with_publish(
            &marker_path,
            b"{\"runtime_key\":\"new\"}",
            |attempt| std::ffi::OsString::from(format!("partial-{attempt}")),
            || {},
            |tmp, path| {
                fs::remove_file(path)?;
                prepare_kept(tmp)?;
                Err(rocm_core::atomic_write::destination_removed_error(
                    std::io::Error::from_raw_os_error(1176),
                    std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                ))
            },
        )
        .expect_err("the simulated partial replace fails the write")
        .context(format!(
            "failed to write active runtime marker {}",
            marker_path.display()
        ))
    }

    /// When the kept marker cannot be removed, the error names it and says to
    /// delete it, and nothing in it still says to rename it into place. The
    /// kept path is made a non-empty directory, which `remove_file` refuses on
    /// every platform.
    #[test]
    fn a_kept_marker_that_cannot_be_removed_is_named_with_delete_advice_only() -> Result<()> {
        let (root, paths) = test_paths("activation-partial-replace-stuck");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let snapshot = ActivationSnapshot::capture(&paths, &config);
        let previous_marker = fs::read(active_runtime_marker_path(&paths))?;
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        config.save(&paths)?;
        let write_error = partial_replace_marker_error(&paths, |tmp| {
            fs::remove_file(tmp)?;
            fs::create_dir(tmp)?;
            fs::write(tmp.join("occupied.txt"), "stuck")
        });
        let kept = marker_temp_file_names(&paths)?;

        let error = restore_after_failed_activation(
            &paths,
            &RocmCliConfig::lock(&paths)?,
            &mut config,
            snapshot,
            write_error,
        );
        let rendered = format!("{error:#}");
        let restored_marker = fs::read(active_runtime_marker_path(&paths))?;
        let still_there = marker_temp_file_names(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(still_there, kept, "the stuck file is left where it was");
        assert_eq!(restored_marker, previous_marker);
        assert!(
            rendered.contains(&kept[0]),
            "the stuck file is named:\n{rendered}"
        );
        assert!(
            rendered.contains("delete it, do not rename it"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("rename that file"),
            "renaming would install the failed activation's marker:\n{rendered}"
        );
        Ok(())
    }

    /// A kept marker that is already gone by the time the rollback looks for it
    /// is as good as removed: the error neither names it nor tells the user to
    /// delete or rename it.
    #[test]
    fn a_kept_marker_already_gone_is_treated_as_removed() -> Result<()> {
        let (root, paths) = test_paths("activation-partial-replace-gone");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let snapshot = ActivationSnapshot::capture(&paths, &config);
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        let write_error = partial_replace_marker_error(&paths, |_| Ok(()));
        let kept = marker_temp_file_names(&paths)?;
        let runtimes_dir = active_runtime_marker_path(&paths)
            .parent()
            .context("active runtime marker path has no parent directory")?
            .to_path_buf();
        for name in &kept {
            fs::remove_file(runtimes_dir.join(name))?;
        }

        let error = restore_after_failed_activation(
            &paths,
            &RocmCliConfig::lock(&paths)?,
            &mut config,
            snapshot,
            write_error,
        );
        let rendered = format!("{error:#}");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(kept.len(), 1, "{kept:?}");
        assert!(
            rendered.contains("the previously active runtime is still the active one"),
            "{rendered}"
        );
        assert!(!rendered.contains(&kept[0]), "{rendered}");
        assert!(!rendered.contains("rename that file"), "{rendered}");
        assert!(!rendered.contains("delete it"), "{rendered}");
        Ok(())
    }

    /// When the rollback itself fails as well, the kept marker is still the
    /// failed activation's, so it goes too, and the error keeps the rollback's
    /// own recovery advice without the advice to rename it. The config path is
    /// made a non-empty directory so the restore's config save fails.
    #[test]
    fn a_failed_rollback_still_drops_the_marker_a_partial_replace_kept() -> Result<()> {
        let (root, paths) = test_paths("activation-partial-replace-double-fault");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let snapshot = ActivationSnapshot::capture(&paths, &config);
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        let write_error = partial_replace_marker_error(&paths, |_| Ok(()));
        assert!(!marker_temp_file_names(&paths)?.is_empty());
        let config_path = paths.config_path();
        let _ = fs::remove_file(&config_path);
        fs::create_dir_all(&config_path)?;
        fs::write(config_path.join("occupied.txt"), "not a config")?;

        let error = restore_after_failed_activation(
            &paths,
            &RocmCliConfig::lock(&paths)?,
            &mut config,
            snapshot,
            write_error,
        );
        let rendered = format!("{error:#}");
        let leftovers = marker_temp_file_names(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            rendered.contains("restoring the previous state failed too"),
            "{rendered}"
        );
        assert!(rendered.contains("rocm runtimes list"), "{rendered}");
        assert!(!rendered.contains("rename that file"), "{rendered}");
        assert!(leftovers.is_empty(), "{leftovers:?}");
        Ok(())
    }

    // A restore that fails too must say so and name the way out, rather than
    // claim the previous runtime is still the active one. The marker's parent is
    // made a regular file, so writing the captured bytes back fails at
    // `create_dir_all` on every platform without needing privileges.
    // Mutation: returning the single-fault message from the `Err(restore_error)`
    // arm of `restore_after_failed_activation` makes the second assertion fail.
    #[test]
    fn a_restore_that_fails_too_names_both_failures_and_how_to_recover() -> Result<()> {
        let (root, paths) = test_paths("activation-snapshot-double-fault");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let snapshot = ActivationSnapshot::capture(&paths, &config);
        assert!(
            matches!(snapshot.marker, MarkerSnapshot::Contents(_)),
            "only a captured marker is written back, which is the restore that can fail"
        );
        // Stand in for a half-applied activation: the config moved on, in memory
        // and on disk.
        config.default_runtime_id = Some("therock-release:gfx1100".to_owned());
        config.active_runtime_key = Some(NEW_RUNTIME_KEY.to_owned());
        config.previous_runtime_key = Some(OLD_RUNTIME_KEY.to_owned());
        config.save(&paths)?;
        let marker_parent = active_runtime_marker_path(&paths)
            .parent()
            .context("active runtime marker path has no parent directory")?
            .to_path_buf();
        fs::remove_dir_all(&marker_parent)?;
        fs::write(&marker_parent, b"not a directory")?;

        let error = restore_after_failed_activation(
            &paths,
            &RocmCliConfig::lock(&paths)?,
            &mut config,
            snapshot,
            anyhow::anyhow!("marker write failed"),
        );
        let rendered = format!("{error:#}");
        let persisted = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            rendered.contains("marker write failed"),
            "the failure that started the rollback must survive:\n{rendered}"
        );
        assert!(
            rendered.contains("restoring the previous state failed too"),
            "a restore that failed must not be reported as one that worked:\n{rendered}"
        );
        assert!(
            rendered.contains("rocm runtimes list"),
            "the user needs the command that shows the disagreeing state:\n{rendered}"
        );
        assert!(
            !rendered.contains("the previously active runtime is still the active one"),
            "that sentence is only true when the restore worked:\n{rendered}"
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the in-memory config is put back before anything that can fail"
        );
        assert_eq!(config.previous_runtime_key, None);
        assert_eq!(
            persisted.active_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the config half of the restore still happened"
        );
        Ok(())
    }

    #[test]
    fn reactivating_the_active_runtime_keeps_the_rollback_target() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-reactivate");
        paths.ensure()?;
        let mut config = activate_old_runtime(&paths)?;
        let switched = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        assert_eq!(
            switched.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );

        // The report's own advice: re-activate the runtime that is already
        // active, with `--restart-services --yes`, to move the servers left
        // behind without moving the runtime. That note prints one line above
        // `ROLLBACK_RECOVERY_HINT`, so a no-op activation that discarded the
        // rollback target would break the very next thing the user is told.
        let reactivated = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;

        let marker: ActiveRuntimeMarker =
            serde_json::from_slice(&fs::read(active_runtime_marker_path(&paths))?)?;
        let persisted = RocmCliConfig::load(&paths)?;
        // Proves the retained target is usable, not merely recorded.
        let rolled_back = rollback_runtime(&paths, &mut config)?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(reactivated.runtime_key, NEW_RUNTIME_KEY);
        assert_eq!(
            reactivated.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "a switch that does not switch has no history to record and none \
             to discard"
        );
        assert_eq!(
            persisted.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        assert_eq!(
            marker.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the marker carries the rollback target too, and is what the \
             storage retention holds read"
        );
        assert!(
            marker.previous_runtime_id.is_some(),
            "the retained key resolves to a runtime that is still installed, \
             so its runtime_id belongs in the marker beside it"
        );
        assert_eq!(
            rolled_back.runtime_key, OLD_RUNTIME_KEY,
            "rollback after a no-op activation must still reach the runtime \
             the user last switched away from"
        );
        Ok(())
    }

    // The rollback target's runtime can be uninstalled while it is still the
    // recorded target. Re-activating the active runtime then keeps the recorded
    // key but cannot name an id for a runtime that is no longer registered. The
    // two runtimes have different ids so that an id taken from the active
    // runtime instead of the target is visible.
    // Mutation: resolving `previous_runtime_id` from `current` rather than from
    // `manifests` makes the marker carry the active runtime's id and fails the
    // `previous_runtime_id` assertion; reporting `reactivated: false` for a
    // re-activation fails the `reactivated.reactivated` assertion.
    #[test]
    fn reactivating_after_the_rollback_target_was_uninstalled_records_no_id_for_it() -> Result<()> {
        let (root, paths) = test_paths("runtime-activation-reactivate-uninstalled-target");
        paths.ensure()?;
        write_test_pip_runtime(
            &paths,
            OLD_RUNTIME_KEY,
            "therock-release:gfx120X-all-old",
            "7.12.0",
            10,
        )?;
        let new_manifest = write_test_pip_runtime(
            &paths,
            NEW_RUNTIME_KEY,
            "therock-release:gfx120X-all-new",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY)?;
        let switched = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;
        fs::remove_file(runtime_manifest_path(&paths, OLD_RUNTIME_KEY))?;

        let reactivated = activate_runtime(&paths, &mut config, NEW_RUNTIME_KEY)?;

        let marker: ActiveRuntimeMarker =
            serde_json::from_slice(&fs::read(active_runtime_marker_path(&paths))?)?;
        let persisted = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            !switched.reactivated,
            "moving between two runtimes is a switch"
        );
        assert!(
            reactivated.reactivated,
            "asking for the runtime that is already active switches nothing"
        );
        assert_eq!(marker.runtime_key, NEW_RUNTIME_KEY);
        assert_eq!(
            marker.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the recorded rollback target is kept as the key it was recorded under"
        );
        assert_eq!(
            marker.previous_runtime_id, None,
            "no registered runtime has that key any more, so there is no id to \
             record for it; the active runtime's id `{}` is not the target's",
            new_manifest.runtime_id
        );
        assert_eq!(
            persisted.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY)
        );
        Ok(())
    }

    #[test]
    fn the_activation_report_says_unchanged_only_when_nothing_was_switched() {
        let result = |reactivated: bool, previous: Option<&str>| RuntimeActivationResult {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: NEW_RUNTIME_KEY.to_owned(),
            previous_runtime_key: previous.map(ToOwned::to_owned),
            reactivated,
            services: runtime_services::RuntimeServiceReconciliation::default(),
        };
        let reactivated_with_target = result(true, Some(OLD_RUNTIME_KEY));
        let switched = result(false, Some(OLD_RUNTIME_KEY));
        let first_activation = result(false, None);

        // Mutation: dropping the `reactivated` branch of
        // `changed_from_runtime_key_text` fails this first assertion.
        assert_eq!(
            changed_from_runtime_key_text(&reactivated_with_target),
            "<unchanged>",
            "naming the retained rollback target would claim the user just came from it"
        );
        assert_eq!(changed_from_runtime_key_text(&switched), OLD_RUNTIME_KEY);
        assert_eq!(changed_from_runtime_key_text(&first_activation), "<unset>");
    }

    #[test]
    fn runtime_activation_rejects_ambiguous_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("runtime-ambiguous");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-12-0",
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();

        let error = activate_runtime(&paths, &mut config, "therock-release:gfx120X-all")
            .unwrap_err()
            .to_string();

        assert!(error.contains("matches multiple installed runtimes"));
        assert!(error.contains("release-pip-gfx120x-all-7-12-0"));
        assert!(error.contains("release-pip-gfx120x-all-7-13-0"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_activation_rejects_unusable_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-unusable");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        fs::remove_file(manifest.install_root.join("Scripts").join("python.exe")).ok();
        fs::remove_file(manifest.install_root.join("bin").join("python")).ok();
        let mut config = RocmCliConfig::default();

        let error = activate_runtime(&paths, &mut config, "release-pip-gfx120x-all-7-13-0")
            .unwrap_err()
            .to_string();

        assert!(error.contains("runtime Python executable is missing"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_import_records_read_only_manifest_without_mutating_runtime_root() -> Result<()> {
        let (root, paths) = test_paths("runtime-import");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;
        fs::remove_file(runtime_manifest_path(&paths, &manifest.runtime_key))?;
        fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;

        let imported = import_runtime_manifest(&paths, &exported_manifest, false)?;

        assert!(imported.read_only);
        let canonical_export = exported_manifest.canonicalize()?;
        assert_eq!(
            imported.imported_from.as_deref(),
            Some(canonical_export.as_path())
        );
        assert!(
            !manifest
                .install_root
                .join(".rocm-cli-runtime.json")
                .exists()
        );

        let imported_registry: therock::InstalledRuntimeManifest = serde_json::from_slice(
            &fs::read(runtime_manifest_path(&paths, &manifest.runtime_key))?,
        )?;
        assert!(imported_registry.read_only);

        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &manifest.runtime_key)?;
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("external-pip-gfx120x-all-7-13-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("mode=read-only"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// A Python environment outside ROCm CLI's folders that `rocm runtimes adopt`
    /// accepts: an interpreter, the SDK libraries the probe names, and the probe
    /// that reports them. Returns the environment folder, the request that adopts
    /// it, and the probe.
    fn adoptable_environment(
        root: &Path,
        replace: bool,
    ) -> Result<(PathBuf, AdoptRuntimeRequest, therock::RocmSdkPythonProbe)> {
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root
            .join("Lib")
            .join("site-packages")
            .join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let request = AdoptRuntimeRequest {
            python_executable,
            install_root: external_root.clone(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
            replace,
        };
        let probe = therock::RocmSdkPythonProbe {
            import_ok: true,
            rocm_sdk_version: Some("7.13.0".to_owned()),
            root_path: Some(sdk_root.clone()),
            bin_path: Some(sdk_bin.clone()),
            runtime_roots: vec![sdk_root],
            bin_paths: vec![sdk_bin.clone()],
            library_paths: vec![sdk_bin],
            resolved_libraries: vec![
                therock::RocmSdkLibraryProbe {
                    shortname: "amdhip64".to_owned(),
                    paths: vec![amdhip],
                },
                therock::RocmSdkLibraryProbe {
                    shortname: "hipblas".to_owned(),
                    paths: vec![hipblas],
                },
            ],
            resolved_target_family: Some("gfx120X-all".to_owned()),
            ..therock::RocmSdkPythonProbe::default()
        };
        Ok((external_root, request, probe))
    }

    #[test]
    fn runtime_adopt_records_read_only_manifest_from_probe() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt");
        let (external_root, request, probe) = adoptable_environment(&root, false)?;

        let adopted = adopt_runtime_from_probe(&paths, request, probe)?;

        assert!(adopted.read_only);
        assert_eq!(adopted.channel, "release");
        assert_eq!(adopted.family, "gfx120X-all");
        assert_eq!(adopted.version, "7.13.0");
        assert_eq!(
            adopted.imported_from.as_deref(),
            Some(external_root.canonicalize()?.as_path())
        );
        assert!(!external_root.join(".rocm-cli-runtime.json").exists());
        assert!(runtime_manifest_path(&paths, &adopted.runtime_key).is_file());
        // No CMake path in the probe means no compiler toolchain in that
        // environment. `rocm update` reinstalls from this field, so recording
        // it wrongly would add a toolchain the user never had.
        assert!(
            !adopted.devel,
            "a probe without a CMake path must not claim the toolchain"
        );

        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &adopted.runtime_key)?;
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some("adopted-release-pip-gfx120x-all-7-13-0")
        );

        let rendered = render_runtimes_text(&paths, &config)?;
        assert!(rendered.contains("* adopted-release-pip-gfx120x-all-7-13-0"));
        assert!(rendered.contains("mode=read-only"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Which of the two files a registered runtime can have exist: its registry
    /// entry, and the `.rocm-cli-runtime.json` ROCm CLI keeps in the install
    /// root as proof that it owns the folder.
    fn registered_files(paths: &AppPaths, install_root: &Path, runtime_key: &str) -> (bool, bool) {
        (
            runtime_manifest_path(paths, runtime_key).is_file(),
            install_root.join(".rocm-cli-runtime.json").is_file(),
        )
    }

    /// A runtime ROCm CLI installed itself, with both of its manifest files
    /// deleted so a test sees only what the write under test creates.
    fn installed_runtime_without_manifest_files(
        paths: &AppPaths,
    ) -> Result<therock::InstalledRuntimeManifest> {
        let manifest = write_test_pip_runtime(
            paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        fs::remove_file(runtime_manifest_path(paths, &manifest.runtime_key))?;
        fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;
        assert_eq!(
            registered_files(paths, &manifest.install_root, &manifest.runtime_key),
            (false, false)
        );
        Ok(manifest)
    }

    /// Registering a runtime whose registry entry exists is refused until the
    /// caller asks to replace it. `register(replace)` makes the attempt; the
    /// message and the entry on disk are asserted together, and the same call
    /// with `replace` set is what clears the refusal.
    fn assert_duplicate_registration_needs_replace(
        paths: &AppPaths,
        runtime_key: &str,
        register: impl Fn(bool) -> Result<()>,
    ) -> Result<()> {
        let entry = runtime_manifest_path(paths, runtime_key);
        fs::write(&entry, "stale entry")?;

        let error = register(false).unwrap_err().to_string();

        assert!(
            error.contains("already exists") && error.contains("--replace"),
            "{error}"
        );
        assert_eq!(
            fs::read_to_string(&entry)?,
            "stale entry",
            "a refused registration must leave the existing entry alone"
        );
        register(true)?;
        let replaced: therock::InstalledRuntimeManifest =
            serde_json::from_slice(&fs::read(&entry)?)?;
        assert_eq!(replaced.runtime_key, runtime_key);
        Ok(())
    }

    /// Installing writes both files, and a second install of the same key
    /// overwrites both without being asked: it is the same runtime again.
    #[test]
    fn runtime_install_writes_the_registry_entry_and_the_install_root_manifest() -> Result<()> {
        let (root, paths) = test_paths("runtime-files-install");
        let manifest = installed_runtime_without_manifest_files(&paths)?;

        write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace)?;

        assert_eq!(
            registered_files(&paths, &manifest.install_root, &manifest.runtime_key),
            (true, true)
        );
        let reinstalled = therock::InstalledRuntimeManifest {
            version: "7.14.0".to_owned(),
            ..manifest.clone()
        };
        write_runtime_manifest(&paths, &reinstalled, ManifestOverwrite::Replace)?;
        for path in [
            runtime_manifest_path(&paths, &manifest.runtime_key),
            manifest.install_root.join(".rocm-cli-runtime.json"),
        ] {
            let written: therock::InstalledRuntimeManifest =
                serde_json::from_slice(&fs::read(&path)?)?;
            assert_eq!(written.version, "7.14.0", "{}", path.display());
        }

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Importing registers a folder somebody else owns, so it writes the
    /// registry entry and nothing into that folder, and refuses to replace an
    /// existing entry unless asked.
    #[test]
    fn runtime_import_writes_the_registry_entry_only_and_refuses_a_duplicate() -> Result<()> {
        let (root, paths) = test_paths("runtime-files-import");
        let manifest = installed_runtime_without_manifest_files(&paths)?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;

        import_runtime_manifest(&paths, &exported_manifest, false)?;

        assert_eq!(
            registered_files(&paths, &manifest.install_root, &manifest.runtime_key),
            (true, false)
        );
        assert_duplicate_registration_needs_replace(&paths, &manifest.runtime_key, |replace| {
            import_runtime_manifest(&paths, &exported_manifest, replace).map(drop)
        })?;
        assert_eq!(
            registered_files(&paths, &manifest.install_root, &manifest.runtime_key),
            (true, false),
            "replacing an import must not start writing into the folder either"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Adopting has the same contract as importing: registry entry only, and a
    /// duplicate needs `--replace`.
    #[test]
    fn runtime_adopt_writes_the_registry_entry_only_and_refuses_a_duplicate() -> Result<()> {
        let (root, paths) = test_paths("runtime-files-adopt");
        let (external_root, request, probe) = adoptable_environment(&root, false)?;
        let runtime_key = request.runtime_key.clone();

        adopt_runtime_from_probe(&paths, request.clone(), probe.clone())?;

        assert_eq!(
            registered_files(&paths, &external_root, &runtime_key),
            (true, false)
        );
        assert_duplicate_registration_needs_replace(&paths, &runtime_key, |replace| {
            let request = AdoptRuntimeRequest {
                replace,
                ..request.clone()
            };
            adopt_runtime_from_probe(&paths, request, probe.clone()).map(drop)
        })?;
        assert_eq!(
            registered_files(&paths, &external_root, &runtime_key),
            (true, false),
            "replacing an adoption must not start writing into the environment either"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Recovering the runtime `rocm setup` created restores its registry entry
    /// from the manifest in its folder, leaves that manifest in place, and does
    /// not replace an entry that is already there.
    #[test]
    fn setup_recovery_restores_the_registry_entry_and_keeps_the_install_root_manifest() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-files-recovery");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(manifest.install_root.clone());
        let entry = runtime_manifest_path(&paths, &manifest.runtime_key);
        fs::remove_file(&entry)?;
        assert_eq!(
            registered_files(&paths, &manifest.install_root, &manifest.runtime_key),
            (false, true)
        );

        let recovered = recover_setup_runtime_registration(&paths, &config)?;

        assert_eq!(recovered.as_deref(), Some(manifest.runtime_key.as_str()));
        assert_eq!(
            registered_files(&paths, &manifest.install_root, &manifest.runtime_key),
            (true, true)
        );
        fs::write(&entry, "entry written since")?;
        recover_setup_runtime_registration(&paths, &config)?;
        assert_eq!(
            fs::read_to_string(&entry)?,
            "entry written since",
            "recovery only fills a missing entry"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Rewrite a manifest file as an older rocm-cli wrote it: without the keys
    /// that hold no value, which a newer binary reads back as their defaults and
    /// writes out as explicit `null`s. Returns the bytes now on disk.
    fn rewrite_as_legacy_shaped(path: &Path) -> Result<Vec<u8>> {
        let mut document: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let object = document
            .as_object_mut()
            .context("a manifest is a JSON object")?;
        let empty: Vec<String> = object
            .iter()
            .filter(|(_, value)| value.is_null())
            .map(|(key, _)| key.clone())
            .collect();
        assert!(
            !empty.is_empty(),
            "the fixture must hold a null-valued key for this to mean anything"
        );
        for key in empty {
            object.remove(&key);
        }
        let legacy = serde_json::to_vec_pretty(&document)?;
        fs::write(path, &legacy)?;
        Ok(legacy)
    }

    /// Recovering the setup runtime from a manifest an older binary wrote, in a
    /// folder the user cannot write to: the file decodes to the very manifest the
    /// recovery would write, so it is already current and is left alone. Judging
    /// it by bytes would read the missing keys as a change, write the registry
    /// entry, and then fail on the read-only file. The registry entry and the
    /// file's bytes are asserted together. (Where the file stays writable
    /// regardless, as for an administrator, the unchanged bytes still prove the
    /// file was not rewritten.)
    #[test]
    #[allow(clippy::permissions_set_readonly_false)]
    fn setup_recovery_leaves_a_read_only_legacy_shaped_install_root_manifest_alone() -> Result<()> {
        let (root, paths) = test_paths("runtime-files-recovery-legacy");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        config.setup.therock_venv = Some(manifest.install_root.clone());
        let entry = runtime_manifest_path(&paths, &manifest.runtime_key);
        let local = manifest.install_root.join(".rocm-cli-runtime.json");
        fs::remove_file(&entry)?;
        let legacy = rewrite_as_legacy_shaped(&local)?;
        let mut permissions = fs::metadata(&local)?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&local, permissions.clone())?;

        let recovered = recover_setup_runtime_registration(&paths, &config);

        permissions.set_readonly(false);
        fs::set_permissions(&local, permissions)?;
        let recovered = recovered
            .context("a usable manifest in a folder that cannot be written is recovered")?;
        assert_eq!(recovered.as_deref(), Some(manifest.runtime_key.as_str()));
        assert!(entry.is_file(), "the registry entry is restored");
        assert_eq!(
            fs::read(&local)?,
            legacy,
            "the install root manifest is already current and must not be rewritten"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The writer gives a runtime a manifest in its folder exactly when
    /// uninstall, which prune asks the same question of, may delete that folder.
    /// The two are written in different modules; this is what ties them.
    #[test]
    fn a_runtime_gets_an_install_root_manifest_exactly_when_uninstall_may_remove_its_folder()
    -> Result<()> {
        let (root, paths) = test_paths("runtime-files-ownership");
        for (label, read_only, imported) in [
            ("owned", false, false),
            ("read-only", true, false),
            ("imported", false, true),
        ] {
            let mut manifest = write_test_pip_runtime(
                &paths,
                &format!("release-pip-{label}"),
                "therock-release:gfx120X-all",
                "7.13.0",
                20,
            )?;
            manifest.read_only = read_only;
            manifest.imported_from = imported.then(|| PathBuf::from("elsewhere/manifest.json"));
            fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;

            write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace)?;

            let has_manifest = manifest
                .install_root
                .join(".rocm-cli-runtime.json")
                .is_file();
            let may_remove =
                should_remove_runtime_install_root(&manifest)? == InstallRootDecision::Remove;
            assert_eq!(has_manifest, may_remove, "{label}");
            assert_eq!(has_manifest, manifest.owns_install_root(), "{label}");
        }

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_removes_managed_root_and_clears_active_state() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-managed");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();
        activate_runtime(&paths, &mut config, &manifest.runtime_key)?;
        config.setup.completed = true;
        config.setup.therock_venv = Some(manifest.install_root.clone());
        config.save(&paths)?;

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(removed.runtime_key, manifest.runtime_key);
        assert!(removed.was_active);
        assert_eq!(
            removed.removed_install_root.as_deref(),
            Some(manifest.install_root.as_path())
        );
        assert!(!manifest.install_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());
        assert!(!active_runtime_marker_path(&paths).exists());
        assert_eq!(config.active_runtime_key, None);
        assert_eq!(config.default_runtime_id, None);
        assert_eq!(config.setup.therock_venv, None);
        assert!(!config.setup.completed);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
