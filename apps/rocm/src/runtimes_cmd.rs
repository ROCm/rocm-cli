// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm runtimes` command: its dispatch, activate/rollback, and the runtime
//! list rendering. Import and adopt (and recovering the runtime `rocm setup`
//! created) live in `runtime_adopt.rs`, and uninstall in
//! `runtime_uninstall.rs`. Registry selection and manifest validation live in
//! `runtime_registry.rs`; the names other modules import from here
//! (`current_runtime_manifest`, `select_runtime_manifest`,
//! `validate_runtime_manifest_for_activation`, `runtime_usability_status`,
//! and the registry path helpers) are re-exported from this module.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays byte-identical (`runtimes(command)`,
//! re-imported via `use crate::runtimes_cmd::{runtimes, ...};`).
//! `RuntimesCommand` remains in the crate root (part of the `Command`/`Cli`
//! clap definition) and is reached via `use crate::RuntimesCommand`, the same
//! way `EnginesCommand` stayed behind when `engines()` moved to
//! `engines_cmd.rs` in Phase 6a. `RuntimeActivationResult` stays
//! `pub(crate)`: it is read by `append_update_activate_summary` in
//! `update_cmd.rs`'s `apply_runtime_update` flow. The uninstall plan and apply
//! code lives in `runtime_uninstall.rs`, and the active-runtime marker model in
//! `runtime_registry.rs`; `uninstall_runtime`,
//! `should_remove_runtime_install_root`, `InstallRootDecision`,
//! `ActiveRuntimeMarker` and `active_runtime_marker_path` are re-exported from
//! here for `storage.rs` and `main.rs`.
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

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, interactive_terminal};

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
pub(crate) use crate::runtime_uninstall::{
    InstallRootDecision, should_remove_runtime_install_root, uninstall_runtime,
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
        RuntimesCommand::List
        | RuntimesCommand::Uninstall { .. }
        | RuntimesCommand::Import { .. }
        | RuntimesCommand::Adopt { .. } => Ok(()),
    }
}

pub(crate) fn runtimes(command: Option<RuntimesCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    let command = command.unwrap_or(RuntimesCommand::List);
    // Checked before anything is written: a refused restart must leave the
    // previously active runtime in place, not switch the runtime and then
    // decline the half of the job the user actually asked for.
    ensure_runtime_service_restart_approved(&command)?;

    match command {
        RuntimesCommand::List => {
            print!("{}", render_runtimes_text(&paths, &config)?);
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

            let plan = if yes {
                plan
            } else {
                if !interactive_terminal() {
                    bail!("runtimes uninstall requires --yes outside an interactive terminal");
                }
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
    /// Split out because the first write an activation makes is
    /// `config.save`: when that is what failed, nothing reached disk and
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

    fn restore(self, paths: &AppPaths, config: &mut RocmCliConfig) -> Result<()> {
        self.restore_in_memory(config);
        config.save(paths)?;
        let path = active_runtime_marker_path(paths);
        match self.marker {
            // Written through the same atomic helper the forward path uses, not
            // a bare `fs::write`. This runs only when something has already
            // failed, so it is the write least able to afford leaving a
            // truncated marker behind — which runtime resolution and the
            // storage retention holds would both read as "nothing is active".
            MarkerSnapshot::Contents(bytes) => {
                let parent = path
                    .parent()
                    .context("active runtime marker path has no parent directory")?;
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
                therock::write_file_atomically(&path, &bytes)
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

/// Persist the config an activation just mutated, putting the caller's
/// in-memory copy back if that write fails.
///
/// `config.save` is the FIRST of an activation's two writes, so a failure here
/// leaves disk untouched — but not the caller's struct, which already names the
/// runtime that was not activated. Callers keep using it: `rocm update --apply
/// --activate` saves the same config again straight after, so an unrestored
/// struct is a route to persisting an activation that was refused.
fn save_activated_config(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    snapshot: &ActivationSnapshot,
) -> Result<()> {
    config.save(paths).map_err(|error| {
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
    config: &mut RocmCliConfig,
    snapshot: ActivationSnapshot,
    error: anyhow::Error,
) -> anyhow::Error {
    match snapshot.restore(paths, config) {
        Ok(()) => error.context(
            "failed to record the active ROCm runtime; the previously active runtime is still \
             the active one",
        ),
        Err(restore_error) => error.context(format!(
            "failed to record the active ROCm runtime, and restoring the previous state failed \
             too ({restore_error:#}); the ROCm CLI config and the active runtime marker may now \
             disagree. Run `rocm runtimes list` to see both, then re-run `rocm runtimes activate \
             <runtime_key>` for the runtime you want to write them again"
        )),
    }
}

/// Marker shown beside the active runtime in `rocm runtimes list`. Every
/// place that renders this glyph MUST use this constant so the rendered
/// character and the legend text stay in sync.
const ACTIVE_RUNTIME_MARKER: &str = "*";
/// Marker shown beside the rollback-target runtime. See
/// [`ACTIVE_RUNTIME_MARKER`].
const ROLLBACK_RUNTIME_MARKER: &str = "-";

pub(crate) fn render_runtimes_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    recover_setup_runtime_registration(paths, config)?;
    let manifests = therock::load_runtime_manifests(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "registered ROCm runtimes");
    let _ = writeln!(
        output,
        "  active_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  registry: {}",
        runtime_registry_dir(paths).display()
    );
    let _ = writeln!(
        output,
        "  marker: {}",
        active_runtime_marker_path(paths).display()
    );
    if let Some(active_runtime_key) = config.active_runtime_key.as_deref()
        && !manifests
            .iter()
            .any(|manifest| manifest.runtime_key == active_runtime_key)
    {
        let _ = writeln!(
            output,
            "  active_status: missing manifest for active_runtime_key={active_runtime_key}"
        );
    }
    if config.active_runtime_key.is_none()
        && let Some(default_runtime_id) = config.default_runtime_id.as_deref()
    {
        let matches = manifests
            .iter()
            .filter(|manifest| manifest.runtime_id == default_runtime_id)
            .collect::<Vec<_>>();
        if matches.is_empty() {
            let _ = writeln!(
                output,
                "  active_status: missing manifest for active_runtime_id={default_runtime_id}"
            );
        } else if matches.len() > 1 {
            let _ = writeln!(
                output,
                "  active_status: ambiguous runtime_id={default_runtime_id}; activate one runtime_key: {}",
                runtime_keys_text(&matches)
            );
        }
    }
    if manifests.is_empty() {
        let _ = writeln!(output, "  installed: none");
        let _ = writeln!(
            output,
            "  next step: rocm install sdk --channel release --format wheel"
        );
        return Ok(output);
    }

    let default_runtime_matches = default_runtime_id_matches(config, &manifests);
    let single_default_runtime_key =
        if config.active_runtime_key.is_none() && default_runtime_matches.len() == 1 {
            Some(default_runtime_matches[0].runtime_key.clone())
        } else {
            None
        };
    drop(default_runtime_matches);

    let _ = writeln!(output, "  installed:");
    let _ = writeln!(
        output,
        "    legend: {ACTIVE_RUNTIME_MARKER} = active, {ROLLBACK_RUNTIME_MARKER} = rollback target"
    );
    let _ = writeln!(output);
    for manifest in manifests {
        let active = config
            .active_runtime_key
            .as_deref()
            .is_some_and(|runtime_key| runtime_key == manifest.runtime_key)
            || single_default_runtime_key.as_deref() == Some(manifest.runtime_key.as_str());
        let rollback = config
            .previous_runtime_key
            .as_deref()
            .is_some_and(|runtime_key| runtime_key == manifest.runtime_key);
        let marker = if active {
            Some(ACTIVE_RUNTIME_MARKER)
        } else if rollback {
            Some(ROLLBACK_RUNTIME_MARKER)
        } else {
            None
        };
        let status = runtime_usability_status(&manifest);
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        // The compiler is opt-in and `rocm update` reinstalls whatever this
        // says, so an install missing it should not be silent about that.
        // `rocm examine` reports the same thing for the active runtime.
        let toolchain = toolchain_state_text(manifest.includes_devel());
        cli_report::write_marked_row(
            &mut output,
            "  ",
            marker,
            format_args!(
                "{} runtime_id={} version={} format={} family={} mode={} status={} toolchain={}",
                manifest.runtime_key,
                manifest.runtime_id,
                therock::runtime_version_display(&manifest.version),
                manifest.format,
                manifest.family,
                mode,
                status,
                toolchain
            ),
        );
        let _ = writeln!(
            output,
            "      install_root: {}",
            manifest.install_root.display()
        );
    }

    Ok(output)
}

pub(crate) fn activate_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    selector: &str,
) -> Result<RuntimeActivationResult> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, selector)?;
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

    // Read the live services before anything is written, so an unreadable
    // services folder refuses the activation instead of half-applying it.
    let services =
        runtime_services::reconcile_services_for_runtime(paths, &manifests, &manifest.runtime_key)
            .with_context(|| {
                format!(
                    "the local server records could not be read, so the switch to `{}` was not \
                     made and the previously active runtime is unchanged",
                    manifest.runtime_key
                )
            })?;
    let snapshot = ActivationSnapshot::capture(paths, config);

    config.default_runtime_id = Some(manifest.runtime_id.clone());
    config.active_runtime_key = Some(manifest.runtime_key.clone());
    config.previous_runtime_key = previous_runtime_key.clone();
    save_activated_config(paths, config, &snapshot)?;
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
            paths, config, snapshot, error,
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

fn rollback_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
) -> Result<RuntimeActivationResult> {
    let previous_key = config
        .previous_runtime_key
        .clone()
        .context("no previous runtime is recorded; activate another runtime before rollback")?;
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

    let services =
        runtime_services::reconcile_services_for_runtime(paths, &manifests, &previous.runtime_key)
            .with_context(|| {
                format!(
                    "the local server records could not be read, so the rollback to `{}` was not \
                     made and the previously active runtime is unchanged",
                    previous.runtime_key
                )
            })?;
    let snapshot = ActivationSnapshot::capture(paths, config);

    config.default_runtime_id = Some(previous.runtime_id.clone());
    config.active_runtime_key = Some(previous.runtime_key.clone());
    config.previous_runtime_key = new_previous_key.clone();
    save_activated_config(paths, config, &snapshot)?;
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
            paths, config, snapshot, error,
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

fn write_active_runtime_marker(paths: &AppPaths, marker: ActiveRuntimeMarker) -> Result<()> {
    let path = active_runtime_marker_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("active runtime marker path has no parent directory")?,
    )?;
    // The shared helper rather than a local temp-file-and-rename: it is the same
    // write the failed-activation restore uses, and it is strictly stronger than
    // a bare `fs::rename`, which on Windows is `MoveFileExW` and fails with a
    // sharing violation when another process holds the marker open. The helper
    // uses `ReplaceFileW` there whenever the destination exists. The
    // destination is never removed first, on any platform — that would leave a
    // window with no marker at all, which runtime resolution and the storage
    // retention holds both read as "nothing is active" — and a failed publish
    // takes its temp file with it, so a retried activation cannot strand one in
    // the folder `rocm runtimes` reads.
    therock::write_file_atomically(
        &path,
        &serde_json::to_vec_pretty(&marker).context("failed to serialize active runtime marker")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::append_examine_runtime_state;
    use crate::engine_runtime_select::resolve_runtime_selector_to_exact_key;
    use crate::runtime_adopt::{AdoptRuntimeRequest, adopt_runtime_from_probe};
    use crate::runtime_install_finalize::{
        finalize_successful_sdk_install, render_sdk_install_success,
    };
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
    /// next `write_active_runtime_marker` fails at a predictable point: its
    /// `create_dir_all` of the parent succeeds, and the final `fs::rename` of
    /// the temp file onto a non-empty directory fails on both supported
    /// platforms without needing privileges or a real full disk.
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

        snapshot.restore(&paths, &mut config)?;

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
        // Same trick as `break_active_runtime_marker`, aimed at the config: the
        // save's final rename onto a non-empty directory fails, which is the
        // FIRST of an activation's two writes and so never reaches the
        // marker-failure recovery path.
        let config_path = paths.config_path();
        fs::remove_file(&config_path)?;
        fs::create_dir_all(&config_path)?;
        fs::write(config_path.join("occupied.txt"), "not a config")?;

        let error = activate_runtime(&paths, &mut config, OLD_RUNTIME_KEY)
            .expect_err("a failed config save must fail the activation");
        let _ = fs::remove_dir_all(&root);

        let error = format!("{error:#}");
        assert!(
            error.contains("the previously active runtime is still the active one"),
            "nothing reached disk, so the error must say which runtime is \
             active:\n{error}"
        );
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(NEW_RUNTIME_KEY),
            "the caller goes on using this struct — `rocm update --apply \
             --activate` saves it again right after — so a config left naming \
             the runtime that was NOT activated is a route to persisting it"
        );
        assert_eq!(
            config.previous_runtime_key.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the rollback target belongs to the activation that succeeded, not \
             to the one that failed before it wrote anything"
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
             `config.save` on this path writes the state the restore just undid"
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

        snapshot.restore(&paths, &mut config)?;

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

        snapshot.restore(&paths, &mut config)?;

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

        snapshot.restore(&paths, &mut config)?;

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

    #[test]
    fn runtime_adopt_records_read_only_manifest_from_probe() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt");
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

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root.clone(),
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
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
            },
        )?;

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
