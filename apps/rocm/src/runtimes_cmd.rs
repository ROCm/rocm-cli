// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm runtimes` command: activate/rollback/uninstall/import/adopt, and
//! runtime manifest validation/registry.
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays byte-identical (`runtimes(command)`,
//! re-imported via `use crate::runtimes_cmd::{runtimes, ...};`).
//! `RuntimesCommand` remains in the crate root (part of the `Command`/`Cli`
//! clap definition) and is reached via `use crate::RuntimesCommand`, the same
//! way `EnginesCommand` stayed behind when `engines()` moved to
//! `engines_cmd.rs` in Phase 6a. `RuntimeActivationResult` and
//! `ActiveRuntimeMarker` stay `pub(crate)`: the former is read by
//! `append_update_activate_summary` in `update_cmd.rs`'s
//! `apply_runtime_update` flow, the latter by `storage.rs`. `select_runtime_manifest`,
//! `validate_runtime_manifest_for_activation`, `normalize_path_for_compare`,
//! `recover_setup_runtime_registration`, `active_runtime_marker_path`,
//! `should_remove_runtime_install_root`, and `runtime_usability_status` are
//! also `pub(crate)` and read from `engines_cmd.rs`, `storage.rs`, and
//! `comfyui.rs`. The SDK-install-finalization/torch-alignment/dependency bundle
//! (`finish_sdk_install` and friends) lives in the sibling
//! `runtime_install_finalize.rs`, which calls into this module (for
//! `activate_runtime`, `newest_installed_runtime_manifest` and
//! `recover_setup_runtime_registration`); this module reaches back into it
//! only from its own test module.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, interactive_terminal, managed_pip_cache_dir,
    runtime_install_root_is_protected, runtime_python_executable_in_env,
};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ActiveRuntimeMarker {
    runtime_id: String,
    pub(crate) runtime_key: String,
    manifest_path: PathBuf,
    install_root: PathBuf,
    previous_runtime_id: Option<String>,
    pub(crate) previous_runtime_key: Option<String>,
    activated_at_unix_ms: u128,
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

#[derive(Debug, Clone)]
pub(crate) struct RuntimeUninstallResult {
    pub(crate) runtime_id: String,
    pub(crate) runtime_key: String,
    registry_path: PathBuf,
    pub(crate) removed_install_root: Option<PathBuf>,
    read_only: bool,
    manifest_mismatch: bool,
    was_active: bool,
    default_runtime_cleared: bool,
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
            ACTIVE_RUNTIME_MARKER
        } else if rollback {
            ROLLBACK_RUNTIME_MARKER
        } else {
            " "
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
        let _ = writeln!(
            output,
            "  {marker} {} runtime_id={} version={} format={} family={} mode={} status={} toolchain={}",
            manifest.runtime_key,
            manifest.runtime_id,
            therock::runtime_version_display(&manifest.version),
            manifest.format,
            manifest.family,
            mode,
            status,
            toolchain
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

/// A runtime's install folder is only ever removed when ROCm CLI is confident
/// it owns that folder; `ReadOnly` and `ManifestMismatch` are distinct reasons
/// for leaving it alone, surfaced separately so a real problem (a stale or
/// corrupt local manifest) doesn't look identical to an intentional no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstallRootDecision {
    Remove,
    ReadOnly,
    ManifestMismatch,
}

impl InstallRootDecision {
    pub(crate) const fn should_remove(self) -> bool {
        matches!(self, Self::Remove)
    }
}

#[derive(Debug, Clone)]
struct RuntimeUninstallPlan {
    manifest: therock::InstalledRuntimeManifest,
    registry_path: PathBuf,
    was_active: bool,
    /// Whether applying this plan will clear `config.default_runtime_id`.
    /// This is true when the config's default still points at this
    /// manifest's `runtime_id` and either this manifest was the active one,
    /// or it is the last remaining install sharing that `runtime_id` (the
    /// id is shared across side-by-side installs, so removing one sibling
    /// does not by itself orphan the default while others remain).
    clears_default_runtime: bool,
    install_root_decision: InstallRootDecision,
}

impl RuntimeUninstallPlan {
    fn will_remove_install_root(&self) -> bool {
        self.install_root_decision.should_remove() && self.manifest.install_root.exists()
    }
}

fn print_runtime_uninstall_plan(plan: &RuntimeUninstallPlan) {
    let install_folder = if plan.will_remove_install_root() {
        format!(
            "{} (would be removed)",
            plan.manifest.install_root.display()
        )
    } else {
        match plan.install_root_decision {
            InstallRootDecision::Remove => "not present, nothing to remove".to_owned(),
            InstallRootDecision::ReadOnly => {
                "left untouched (external/read-only runtime)".to_owned()
            }
            InstallRootDecision::ManifestMismatch => {
                "left untouched (local runtime manifest did not match the registry)".to_owned()
            }
        }
    };

    let mut report = cli_report::ActionReport::new("runtime uninstall plan")
        .detail("runtime_id", &plan.manifest.runtime_id)
        .detail("runtime_key", &plan.manifest.runtime_key)
        .detail("registry_entry", plan.registry_path.display())
        .detail("install_folder", install_folder);
    if plan.clears_default_runtime {
        report = report.detail("default_runtime", "would be cleared");
    }
    print!("{}", report.render());
}

fn plan_runtime_uninstall(
    paths: &AppPaths,
    config: &RocmCliConfig,
    selector: &str,
) -> Result<RuntimeUninstallPlan> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let manifest = select_runtime_manifest(&manifests, selector)?.clone();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    let was_active = current_runtime_manifest(config, &manifests)
        .is_some_and(|current| current.runtime_key == manifest.runtime_key);
    let clears_default_runtime = config
        .default_runtime_id
        .as_deref()
        .is_some_and(|runtime_id| runtime_id.eq_ignore_ascii_case(&manifest.runtime_id))
        && (was_active
            || !manifests.iter().any(|other| {
                other.runtime_key != manifest.runtime_key
                    && other.runtime_id.eq_ignore_ascii_case(&manifest.runtime_id)
            }));
    let install_root_decision = should_remove_runtime_install_root(&manifest)?;
    Ok(RuntimeUninstallPlan {
        manifest,
        registry_path,
        was_active,
        clears_default_runtime,
        install_root_decision,
    })
}

/// Re-derives the uninstall plan from disk and refuses to proceed if it no
/// longer matches what the user approved. The interactive confirmation this
/// guards can wait indefinitely; if another process activates a different
/// runtime or replaces the install folder while the prompt is open, applying
/// the stale plan could clear the wrong `active_runtime_key` or recursively
/// delete a folder that is no longer the one that was vetted as safe to
/// remove.
fn revalidate_runtime_uninstall_plan(
    paths: &AppPaths,
    config: &RocmCliConfig,
    plan: RuntimeUninstallPlan,
) -> Result<RuntimeUninstallPlan> {
    let fresh = plan_runtime_uninstall(paths, config, &plan.manifest.runtime_key)?;
    if fresh.manifest.runtime_id != plan.manifest.runtime_id
        || fresh.manifest.install_root != plan.manifest.install_root
        || fresh.was_active != plan.was_active
        || fresh.clears_default_runtime != plan.clears_default_runtime
        || fresh.install_root_decision != plan.install_root_decision
    {
        bail!(
            "runtime state for {} changed while waiting for confirmation; re-run `rocm runtimes uninstall {}` to review the current plan before approving it",
            plan.manifest.runtime_key,
            plan.manifest.runtime_key
        );
    }
    Ok(fresh)
}

enum RuntimeUninstallConfirmation {
    Cancelled,
    // `RuntimeUninstallPlan`/`RocmCliConfig` are large; box them so the two
    // variants stay a similar size (clippy::large_enum_variant).
    Confirmed {
        plan: Box<RuntimeUninstallPlan>,
        config: Box<RocmCliConfig>,
    },
}

/// Runs the confirm-then-revalidate sequence used by an interactive
/// `runtimes uninstall`: waits for the caller-supplied confirmation, then
/// reloads config from disk and re-derives the plan against it, so a state
/// change that happened while the (potentially indefinite) prompt was open
/// cannot be applied against stale data.
fn confirm_and_revalidate_runtime_uninstall(
    paths: &AppPaths,
    plan: RuntimeUninstallPlan,
    confirm: impl FnOnce() -> Result<bool>,
) -> Result<RuntimeUninstallConfirmation> {
    if !confirm()? {
        return Ok(RuntimeUninstallConfirmation::Cancelled);
    }
    let config = RocmCliConfig::load(paths)?;
    let plan = revalidate_runtime_uninstall_plan(paths, &config, plan)?;
    Ok(RuntimeUninstallConfirmation::Confirmed {
        plan: Box::new(plan),
        config: Box::new(config),
    })
}

pub(crate) fn uninstall_runtime(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    selector: &str,
) -> Result<RuntimeUninstallResult> {
    let plan = plan_runtime_uninstall(paths, config, selector)?;
    apply_runtime_uninstall(paths, config, plan)
}

fn apply_runtime_uninstall(
    paths: &AppPaths,
    config: &mut RocmCliConfig,
    plan: RuntimeUninstallPlan,
) -> Result<RuntimeUninstallResult> {
    let RuntimeUninstallPlan {
        manifest,
        registry_path,
        was_active,
        clears_default_runtime,
        install_root_decision,
    } = plan;

    let mut removed_install_root = None;
    if install_root_decision.should_remove() && manifest.install_root.exists() {
        fs::remove_dir_all(&manifest.install_root).with_context(|| {
            format!(
                "failed to remove runtime folder {} - the runtime registry entry has not \
                 been removed yet, so `rocm runtimes list` will still show this runtime as \
                 installed and pointing at this (now possibly partially deleted) folder \
                 until the removal succeeds",
                manifest.install_root.display()
            )
        })?;
        removed_install_root = Some(manifest.install_root.clone());
    }

    if registry_path.exists() {
        fs::remove_file(&registry_path).with_context(|| {
            format!(
                "failed to remove runtime registry entry {}",
                registry_path.display()
            )
        })?;
    }

    let mut config_changed = false;
    if config
        .active_runtime_key
        .as_deref()
        .is_some_and(|runtime_key| runtime_key.eq_ignore_ascii_case(&manifest.runtime_key))
    {
        config.active_runtime_key = None;
        config_changed = true;
    }
    if config
        .previous_runtime_key
        .as_deref()
        .is_some_and(|runtime_key| runtime_key.eq_ignore_ascii_case(&manifest.runtime_key))
    {
        config.previous_runtime_key = None;
        config_changed = true;
    }
    if clears_default_runtime {
        config.default_runtime_id = None;
        config_changed = true;
    }
    if config
        .setup
        .therock_venv
        .as_ref()
        .is_some_and(|path| paths_equivalent(path, &manifest.install_root))
    {
        config.setup.therock_venv = None;
        config.setup.completed = false;
        config.onboarding_dismissed = false;
        config_changed = true;
    }
    if config_changed {
        config.save(paths)?;
    }

    if active_runtime_marker_matches(paths, &manifest.runtime_key)? {
        let marker_path = active_runtime_marker_path(paths);
        if marker_path.exists() {
            fs::remove_file(&marker_path).with_context(|| {
                format!(
                    "failed to remove active runtime marker {}",
                    marker_path.display()
                )
            })?;
        }
    }

    Ok(RuntimeUninstallResult {
        runtime_id: manifest.runtime_id,
        runtime_key: manifest.runtime_key,
        registry_path,
        removed_install_root,
        read_only: manifest.read_only,
        manifest_mismatch: matches!(install_root_decision, InstallRootDecision::ManifestMismatch),
        was_active,
        default_runtime_cleared: clears_default_runtime,
    })
}

pub(crate) fn should_remove_runtime_install_root(
    manifest: &therock::InstalledRuntimeManifest,
) -> Result<InstallRootDecision> {
    if manifest.read_only || manifest.imported_from.is_some() {
        return Ok(InstallRootDecision::ReadOnly);
    }
    if !local_runtime_manifest_matches(manifest)? {
        return Ok(InstallRootDecision::ManifestMismatch);
    }
    ensure_runtime_install_root_is_safe_to_remove(&manifest.install_root)?;
    Ok(InstallRootDecision::Remove)
}

fn local_runtime_manifest_matches(manifest: &therock::InstalledRuntimeManifest) -> Result<bool> {
    let local_path = manifest.install_root.join(".rocm-cli-runtime.json");
    if !local_path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&local_path)
        .with_context(|| format!("failed to read {}", local_path.display()))?;
    let local: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", local_path.display()))?;
    Ok(local.runtime_key == manifest.runtime_key
        && local.runtime_id == manifest.runtime_id
        && paths_equivalent(&local.install_root, &manifest.install_root))
}

fn ensure_runtime_install_root_is_safe_to_remove(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() || path.parent().is_none() || path.file_name().is_none() {
        bail!(
            "refusing to remove unsafe runtime folder {}",
            path.display()
        );
    }
    // Belt and braces: a hand-edited or corrupted registry entry could point
    // `install_root` at a protected system location while still carrying a
    // matching in-tree `.rocm-cli-runtime.json`, slipping past
    // `local_runtime_manifest_matches`. `prune` already refuses these before
    // ever calling this function (see storage.rs); check it here too so the
    // single source of truth for "may ROCm CLI delete this folder?" refuses
    // it for every caller, including a direct `runtimes uninstall <key>`.
    if runtime_install_root_is_protected(path) {
        bail!(
            "refusing to remove runtime folder {} in a protected system location",
            path.display()
        );
    }
    Ok(())
}

fn active_runtime_marker_matches(paths: &AppPaths, runtime_key: &str) -> Result<bool> {
    let marker_path = active_runtime_marker_path(paths);
    if !marker_path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&marker_path)
        .with_context(|| format!("failed to read {}", marker_path.display()))?;
    let marker: ActiveRuntimeMarker = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", marker_path.display()))?;
    Ok(marker.runtime_key.eq_ignore_ascii_case(runtime_key))
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = normalize_path_for_compare(left);
    let right = normalize_path_for_compare(right);
    rocm_core::runtime_paths_equivalent(&left, &right)
}

pub(crate) fn normalize_path_for_compare(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

fn import_runtime_manifest(
    paths: &AppPaths,
    manifest_path: &Path,
    replace: bool,
) -> Result<therock::InstalledRuntimeManifest> {
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let mut manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_key",
            manifest_path.display()
        );
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_id",
            manifest_path.display()
        );
    }

    manifest.read_only = true;
    manifest.imported_from = Some(
        manifest_path
            .canonicalize()
            .unwrap_or_else(|_| manifest_path.to_path_buf()),
    );
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("imported runtime `{}` is not usable", manifest.runtime_key))?;

    write_runtime_registry_manifest(paths, &manifest, replace)?;

    Ok(manifest)
}

pub(crate) fn newest_installed_runtime_manifest(
    paths: &AppPaths,
) -> Result<Option<therock::InstalledRuntimeManifest>> {
    // `load_runtime_manifests` already returns newest install first with ties
    // broken on the runtime key, so re-sorting here would be a no-op.
    Ok(therock::load_runtime_manifests(paths)?.into_iter().next())
}

#[derive(Debug, Clone)]
struct AdoptRuntimeRequest {
    python_executable: PathBuf,
    install_root: PathBuf,
    runtime_id: String,
    runtime_key: String,
    replace: bool,
}

#[derive(Debug, Clone)]
struct AdoptRuntimeOptions {
    python_input: PathBuf,
    install_root: Option<PathBuf>,
    runtime_id: Option<String>,
    runtime_key: Option<String>,
    channel: Option<String>,
    replace: bool,
}

fn adopt_runtime_from_python_options(
    paths: &AppPaths,
    options: AdoptRuntimeOptions,
) -> Result<therock::InstalledRuntimeManifest> {
    let (python_executable, inferred_root) = resolve_adopt_python_input(&options.python_input)?;
    let probe = therock::probe_rocm_sdk_runtime(&python_executable)
        .with_context(|| format!("failed to probe {}", python_executable.display()))?;
    let request = infer_adopt_runtime_request(
        python_executable,
        options.install_root.or(inferred_root),
        options.runtime_id,
        options.runtime_key,
        options.channel,
        options.replace,
        &probe,
    )?;
    adopt_runtime_from_probe(paths, request, probe)
}

fn infer_adopt_runtime_request(
    python_executable: PathBuf,
    install_root: Option<PathBuf>,
    runtime_id: Option<String>,
    runtime_key: Option<String>,
    channel: Option<String>,
    replace: bool,
    probe: &therock::RocmSdkPythonProbe,
) -> Result<AdoptRuntimeRequest> {
    let install_root = install_root.with_context(|| {
        format!(
            "could not infer the Python environment folder from {}; pass --root",
            python_executable.display()
        )
    })?;
    let runtime_id = match runtime_id {
        Some(value) if !value.trim().is_empty() => {
            if let Some(channel) = channel.as_deref() {
                let (parsed_channel, _) = parse_therock_runtime_id(&value)?;
                let requested_channel = normalize_adopt_channel(channel)?;
                if parsed_channel != requested_channel {
                    bail!(
                        "--channel {requested_channel} does not match runtime id channel {parsed_channel}"
                    );
                }
            }
            value
        }
        Some(_) => bail!("runtime_id must not be empty"),
        None => {
            let channel = normalize_adopt_channel(channel.as_deref().unwrap_or("release"))?;
            let family = probe
                .resolved_target_family
                .as_deref()
                .or(probe.default_target_family.as_deref())
                .filter(|value| !value.trim().is_empty())
                .context(
                    "rocm_sdk probe did not report a GPU package; pass --runtime-id explicitly",
                )?;
            format!("therock-{channel}:{family}")
        }
    };
    let runtime_key = match runtime_key {
        Some(value) if !value.trim().is_empty() => value,
        Some(_) => bail!("runtime_key must not be empty"),
        None => {
            let (channel, family) = parse_therock_runtime_id(&runtime_id)?;
            let version = probe
                .rocm_sdk_version
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("rocm_sdk probe did not report a version; cannot name adopted runtime")?;
            format!(
                "adopted-{channel}-pip-{}-{}",
                runtime_key_component(&family),
                runtime_key_component(version)
            )
        }
    };
    Ok(AdoptRuntimeRequest {
        python_executable,
        install_root,
        runtime_id,
        runtime_key,
        replace,
    })
}

fn normalize_adopt_channel(channel: &str) -> Result<String> {
    match channel.trim().to_ascii_lowercase().as_str() {
        "release" => Ok("release".to_owned()),
        "nightly" => Ok("nightly".to_owned()),
        other => bail!("adopt channel must be release or nightly, got `{other}`"),
    }
}

fn runtime_key_component(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    let trimmed = output.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "runtime".to_owned()
    } else {
        trimmed
    }
}

fn resolve_adopt_python_input(input: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let absolute = if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(input)
    };
    if absolute.is_dir() {
        let env_root = absolute.canonicalize().with_context(|| {
            format!(
                "failed to resolve Python environment folder {}",
                absolute.display()
            )
        })?;
        let python = runtime_python_executable_in_env(&env_root);
        if !python.is_file() {
            bail!(
                "Python executable is missing in {}",
                python.parent().unwrap_or(env_root.as_path()).display()
            );
        }
        return Ok((python, Some(env_root)));
    }
    if absolute.is_file() {
        let inferred_root = infer_python_env_root(&absolute);
        return Ok((absolute, inferred_root));
    }
    bail!(
        "Python executable or folder is missing: {}",
        absolute.display()
    );
}

fn infer_python_env_root(python_executable: &Path) -> Option<PathBuf> {
    let bin_dir = python_executable.parent()?;
    let bin_name = bin_dir.file_name()?.to_string_lossy();
    if bin_name.eq_ignore_ascii_case("Scripts") || bin_name == "bin" {
        return bin_dir.parent().map(Path::to_path_buf);
    }
    bin_dir.parent().map(Path::to_path_buf)
}

fn adopt_runtime_from_probe(
    paths: &AppPaths,
    request: AdoptRuntimeRequest,
    probe: therock::RocmSdkPythonProbe,
) -> Result<therock::InstalledRuntimeManifest> {
    if request.runtime_key.trim().is_empty() {
        bail!("runtime_key must not be empty");
    }
    if request.runtime_id.trim().is_empty() {
        bail!("runtime_id must not be empty");
    }
    let python_executable = absolute_existing_file_path_preserving_symlink(
        &request.python_executable,
        "runtime Python executable",
    )?;
    let install_root = request.install_root.canonicalize().with_context(|| {
        format!(
            "runtime install root is missing: {}",
            request.install_root.display()
        )
    })?;
    if !install_root.is_dir() {
        bail!(
            "runtime install root is missing: {}",
            install_root.display()
        );
    }
    let (channel, family) = parse_therock_runtime_id(&request.runtime_id)?;
    therock::validate_rocm_sdk_runtime_probe(&probe)?;
    let version = probe
        .rocm_sdk_version
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .context("rocm_sdk probe did not report a version; cannot adopt runtime explicitly")?
        .to_owned();

    let manifest = therock::InstalledRuntimeManifest {
        runtime_key: request.runtime_key,
        runtime_id: request.runtime_id,
        channel,
        format: "wheel".to_owned(),
        family,
        family_source: "runtime_id".to_owned(),
        version,
        install_root: install_root.clone(),
        selected_artifact_url: "adopted-read-only".to_owned(),
        source_layout_generation: None,
        index_url: None,
        tarball_file_name: None,
        python_launcher: None,
        python_executable: Some(python_executable.display().to_string()),
        pip_cache_dir: None,
        // The probe only reports a CMake path when the `devel` packages are
        // present, so it tells us what this pre-existing environment has.
        devel: probe.cmake_path.is_some(),
        rocm_sdk: Some(probe),
        // Adoption does not install torch, so the build is derived from the SDK
        // version instead.
        sdk_torch: None,
        wheel_composition: None,
        read_only: true,
        imported_from: Some(install_root),
        installed_at_unix_ms: rocm_core::unix_time_millis(),
    };
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("adopted runtime `{}` is not usable", manifest.runtime_key))?;
    write_runtime_registry_manifest(paths, &manifest, request.replace)?;
    Ok(manifest)
}

fn absolute_existing_file_path_preserving_symlink(path: &Path, label: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(path)
    };
    if !absolute.is_file() {
        bail!("{label} is missing: {}", absolute.display());
    }
    Ok(absolute)
}

fn parse_therock_runtime_id(runtime_id: &str) -> Result<(String, String)> {
    let (prefix, family) = runtime_id.split_once(':').with_context(|| {
        format!("runtime_id `{runtime_id}` must include a TheRock family suffix after ':'")
    })?;
    let family = family.trim();
    if family.is_empty() {
        bail!("runtime_id `{runtime_id}` has an empty TheRock family suffix");
    }
    let channel = match prefix.trim() {
        "therock-release" => "release",
        "therock-nightly" => "nightly",
        other => bail!(
            "runtime_id `{runtime_id}` must start with therock-release: or therock-nightly:, got `{other}`"
        ),
    };
    Ok((channel.to_owned(), family.to_owned()))
}

fn write_runtime_registry_manifest(
    paths: &AppPaths,
    manifest: &therock::InstalledRuntimeManifest,
    replace: bool,
) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    if registry_path.exists() && !replace {
        bail!(
            "runtime registry entry already exists: {}; pass --replace to overwrite it",
            registry_path.display()
        );
    }
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&manifest)
            .context("failed to serialize runtime registry manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;
    Ok(())
}

pub(crate) fn recover_setup_runtime_registration(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    let Some(setup_root) = config
        .setup
        .therock_venv
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
    else {
        return Ok(None);
    };
    if !setup_root.is_dir() {
        return Ok(None);
    }

    let local_manifest_path = setup_root.join(".rocm-cli-runtime.json");
    if !local_manifest_path.is_file() {
        return Ok(None);
    }

    let bytes = fs::read(&local_manifest_path)
        .with_context(|| format!("failed to read {}", local_manifest_path.display()))?;
    let manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", local_manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "setup runtime manifest {} has an empty runtime_key",
            local_manifest_path.display()
        );
    }
    if !paths_equivalent(&manifest.install_root, setup_root) {
        bail!(
            "setup runtime manifest {} points at {}, but setup is configured for {}",
            local_manifest_path.display(),
            manifest.install_root.display(),
            setup_root.display()
        );
    }
    validate_runtime_manifest_for_activation(&manifest).with_context(|| {
        format!(
            "setup runtime `{}` from {} is not usable",
            manifest.runtime_key,
            local_manifest_path.display()
        )
    })?;

    if !runtime_manifest_path(paths, &manifest.runtime_key).is_file() {
        write_runtime_registry_manifest(paths, &manifest, false).with_context(|| {
            format!(
                "failed to restore setup runtime `{}` into {}",
                manifest.runtime_key,
                runtime_registry_dir(paths).display()
            )
        })?;
    }
    Ok(Some(manifest.runtime_key))
}

/// The folder setup was configured for and its pip cache — the pair both
/// `examine` forms report. Shared so the two cannot name different folders.
///
/// An empty configured path counts as unset, matching
/// `recover_setup_runtime_registration`'s own guard: `config.json` is
/// hand-editable, and `managed_pip_cache_dir("")` is a bogus relative path.
pub(crate) fn setup_runtime_paths(config: &RocmCliConfig) -> Option<(&Path, PathBuf)> {
    let root = config
        .setup
        .therock_venv
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())?;
    Some((root, managed_pip_cache_dir(root)))
}

pub(crate) fn current_runtime_manifest<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Option<&'a therock::InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref()
        && let Some(manifest) = manifests
            .iter()
            .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(active_key))
    {
        return Some(manifest);
    }

    let matches = default_runtime_id_matches(config, manifests);
    match matches.as_slice() {
        [manifest] => Some(*manifest),
        _ => None,
    }
}

pub(crate) fn default_runtime_id_matches<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Vec<&'a therock::InstalledRuntimeManifest> {
    let Some(default_runtime_id) = config.default_runtime_id.as_deref() else {
        return Vec::new();
    };
    manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id))
        .collect()
}

pub(crate) fn runtime_keys_text(manifests: &[&therock::InstalledRuntimeManifest]) -> String {
    manifests
        .iter()
        .map(|manifest| manifest.runtime_key.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolves a user-typed selector to the one installed runtime it names.
///
/// Matching a `runtime_key` regardless of letter case is a deliberate
/// convenience, but it only names one runtime while no two installed keys
/// differ just in case — and they can. Generated keys are slugified to lower
/// case, while `runtimes adopt --runtime-key` takes the key the user typed and
/// `runtimes import` takes it from the supplied manifest; neither slugifies,
/// so the key is stored verbatim. The registry keeps one file per key, so
/// `ABC.json` and `abc.json` are two entries on a case-sensitive filesystem
/// and nothing rejects the pair.
///
/// So an exact key match is taken first and wins outright: it is the one
/// reading of the selector that cannot be mistaken. The case-insensitive
/// fallback applies only when it lands on a single record; two case twins are
/// reported as the ambiguity they are, the same way a `runtime_id` naming
/// several installs already is.
///
/// Callers rely on that being a function of the key alone, because several of
/// them resolve the same key more than once and must land on the same record
/// every time: `storage remove-old-installs` decides about a manifest and then
/// names it to the deleter by key, and the uninstall revalidation re-derives
/// its plan from disk after the confirmation prompt. Resolving by "newest
/// case-insensitive match" made that decide about one install and act on
/// another.
pub(crate) fn select_runtime_manifest<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    let selector = selector.trim();
    if selector.is_empty() {
        bail!("runtime selector must not be empty");
    }

    if let Some(manifest) = manifests
        .iter()
        .find(|manifest| manifest.runtime_key == selector)
    {
        return Ok(manifest);
    }

    let key_matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match key_matches.as_slice() {
        [manifest] => return Ok(manifest),
        [] => {}
        _ => bail!(
            "runtime selector `{selector}` matches several installed runtime keys that differ only in letter case; name one of them exactly: {}",
            runtime_keys_text(&key_matches)
        ),
    }

    let id_matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match id_matches.as_slice() {
        [manifest] => Ok(manifest),
        [] => bail!("installed runtime not found: {selector}"),
        _ => bail!(
            "runtime selector `{selector}` matches multiple installed runtimes; activate one by runtime_key: {}",
            runtime_keys_text(&id_matches)
        ),
    }
}

pub(crate) fn runtime_usability_status(manifest: &therock::InstalledRuntimeManifest) -> String {
    match validate_runtime_manifest_for_activation(manifest) {
        Ok(()) => "ready".to_owned(),
        Err(error) => format!("unusable ({error})"),
    }
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

pub(crate) fn validate_runtime_manifest_for_activation(
    manifest: &therock::InstalledRuntimeManifest,
) -> Result<()> {
    if manifest.runtime_key.trim().is_empty() {
        bail!("manifest runtime_key is empty");
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!("manifest runtime_id is empty");
    }
    if !manifest.install_root.is_dir() {
        bail!(
            "install root is missing: {}",
            manifest.install_root.display()
        );
    }
    let local_manifest = manifest.install_root.join(".rocm-cli-runtime.json");
    if !manifest.read_only && !local_manifest.is_file() {
        bail!(
            "local runtime manifest is missing: {}",
            local_manifest.display()
        );
    }

    match manifest.format.as_str() {
        "wheel" => validate_wheel_runtime_manifest(manifest),
        "tarball" => validate_tarball_runtime_manifest(manifest),
        other => bail!("unsupported runtime format in manifest: {other}"),
    }
}

fn validate_wheel_runtime_manifest(manifest: &therock::InstalledRuntimeManifest) -> Result<()> {
    let python_executable = manifest
        .python_executable
        .as_deref()
        .context("pip runtime manifest is missing python_executable")?;
    if !Path::new(python_executable).is_file() {
        bail!("runtime Python executable is missing: {python_executable}");
    }
    let probe = manifest
        .rocm_sdk
        .as_ref()
        .context("pip runtime manifest is missing rocm_sdk probe data")?;
    therock::validate_rocm_sdk_runtime_probe(probe)?;
    Ok(())
}

fn validate_tarball_runtime_manifest(manifest: &therock::InstalledRuntimeManifest) -> Result<()> {
    if runtime_install_root_has_payload(&manifest.install_root)? {
        Ok(())
    } else {
        bail!(
            "tarball runtime install root has no payload files: {}",
            manifest.install_root.display()
        )
    }
}

fn runtime_install_root_has_payload(path: &Path) -> Result<bool> {
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

pub(crate) fn runtime_manifest_path(paths: &AppPaths, runtime_key: &str) -> PathBuf {
    runtime_registry_dir(paths).join(format!("{runtime_key}.json"))
}

pub(crate) fn active_runtime_marker_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("active.json")
}

/// Replace the active-runtime marker in one step.
///
/// Two `rocm` processes can activate a runtime against the same data
/// directory at once (two terminals, a script), so the write goes through the
/// shared atomic writer: each writer stages into its own exclusively created
/// temp file, and the marker is replaced rather than removed first, so a
/// reader always finds a complete marker. It does not order the two
/// activations: the last one to publish wins, as before.
fn write_active_runtime_marker(paths: &AppPaths, marker: ActiveRuntimeMarker) -> Result<()> {
    let path = active_runtime_marker_path(paths);
    // The shared helper rather than a local temp-file-and-rename: it is the same
    // write the failed-activation restore uses, and it is strictly stronger than
    // a bare `fs::rename`, which on Windows is `MoveFileExW` and fails with a
    // sharing violation when another process holds the marker open. The helper
    // uses `ReplaceFileW` there whenever the destination exists. The
    // destination is never removed first, on any platform — that would leave a
    // window with no marker at all, which runtime resolution and the storage
    // retention holds both read as "nothing is active" — and a failed publish
    // takes its temp file with it, so a retried activation cannot strand one in
    // the folder `rocm runtimes` reads. It also creates the `runtimes` folder
    // when it is missing.
    let bytes =
        serde_json::to_vec_pretty(&marker).context("failed to serialize active runtime marker")?;
    rocm_core::atomic_write::write_file_atomically(&path, &bytes)
        .with_context(|| format!("failed to write active runtime marker {}", path.display()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::append_examine_runtime_state;
    use crate::engine_runtime_select::resolve_runtime_selector_to_exact_key;
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

    /// Regression for #605: the marker used to be staged at
    /// `active.json.tmp-<millis>` with a truncating write, so a writer that
    /// drew the same millisecond as another activation wrote into *that*
    /// writer's temp file and then moved it into place as the marker.
    ///
    /// Pre-creating a file at every name that scheme could pick while this
    /// test runs stands in for the other writer deterministically: the write
    /// has to succeed without touching any of them.
    #[test]
    fn active_runtime_marker_write_never_reuses_another_writers_temp_file() -> Result<()> {
        const WINDOW_MS: u128 = 2_000;
        const FOREIGN: &[u8] = b"another writer's staged marker";
        let (root, paths) = test_paths("marker-temp-collision");
        write_active_runtime_marker(&paths, test_active_runtime_marker(&paths, "previous"))?;
        let marker_path = active_runtime_marker_path(&paths);

        let start = rocm_core::unix_time_millis();
        let foreign: Vec<PathBuf> = (start..start + WINDOW_MS)
            .map(|millis| marker_path.with_extension(format!("json.tmp-{millis}")))
            .collect();
        for path in &foreign {
            fs::write(path, FOREIGN)?;
        }

        write_active_runtime_marker(&paths, test_active_runtime_marker(&paths, "next"))?;
        let finished = rocm_core::unix_time_millis();
        assert!(
            finished < start + WINDOW_MS,
            "inconclusive: the write ran after the {WINDOW_MS} ms window of pre-created names"
        );

        let published: ActiveRuntimeMarker = serde_json::from_slice(&fs::read(&marker_path)?)?;
        assert_eq!(published.runtime_key, "next");
        for path in &foreign {
            assert_eq!(
                fs::read(path).ok().as_deref(),
                Some(FOREIGN),
                "{} belongs to another writer but was written through or moved",
                path.display()
            );
        }
        for path in &foreign {
            fs::remove_file(path)?;
        }
        assert_eq!(leftover_marker_temp_files(&paths)?, Vec::<String>::new());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Regression for #605: two activations writing the marker at once must
    /// both succeed, and a reader must find a complete marker throughout —
    /// the old writer removed `active.json` before renaming the replacement
    /// in, and two writers in the same millisecond shared one temp file, so
    /// one failed with "failed to move active runtime marker" and the other
    /// could publish the shorter payload followed by the longer one's tail.
    ///
    /// The bug is a race, so against the old code this fails with high
    /// probability per round rather than deterministically; against the fix
    /// nothing in it is timing-dependent. Unix only: the property under test
    /// is that `rename` replaces in one step. The Windows publish path goes
    /// through `ReplaceFileW` instead, whose behaviour with two replacements
    /// of one file in flight this test has not been run against.
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

    /// Matching a key regardless of case is a convenience for a user who typed
    /// the wrong one; it must not quietly pick between two installs whose keys
    /// differ only in case. `runtimes adopt --runtime-key` stores the key
    /// verbatim while generated keys are lower-cased, so such a pair is
    /// reachable, and a caller that resolves the same key twice — prune decides
    /// about a manifest and then names it to the deleter by key — has to get
    /// the same install both times.
    ///
    /// No registry is involved here: the manifests are handed straight to the
    /// resolver, so this holds on every platform even though only a
    /// case-sensitive filesystem can store such a pair.
    #[test]
    fn runtime_selector_prefers_the_exact_key_over_a_case_twin() -> Result<()> {
        let (root, paths) = test_paths("runtime-case-twin");
        let base = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        // Newest first, as `load_runtime_manifests` hands them over. The
        // adopted twin is stamped with the moment it was adopted, so it leads.
        let manifests = vec![
            therock::InstalledRuntimeManifest {
                runtime_key: "RELEASE-PIP-GFX120X-ALL-7-13-0".to_owned(),
                installed_at_unix_ms: 99,
                read_only: true,
                ..base.clone()
            },
            base.clone(),
        ];

        let exact = select_runtime_manifest(&manifests, &base.runtime_key)?;
        assert_eq!(
            exact.runtime_key, base.runtime_key,
            "the key as stored resolves to its own record, not to the newer twin"
        );
        let twin = select_runtime_manifest(&manifests, "RELEASE-PIP-GFX120X-ALL-7-13-0")?;
        assert_eq!(twin.runtime_key, "RELEASE-PIP-GFX120X-ALL-7-13-0");

        // Neither key matches exactly, so the selector genuinely names both.
        // Reported as the ambiguity it is, like a runtime_id naming several.
        let error = select_runtime_manifest(&manifests, "Release-Pip-Gfx120X-All-7-13-0")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("differ only in letter case"),
            "an ambiguous selector must be refused, not silently resolved: {error}"
        );
        assert!(error.contains("release-pip-gfx120x-all-7-13-0"));
        assert!(error.contains("RELEASE-PIP-GFX120X-ALL-7-13-0"));

        // The convenience itself is untouched: with one candidate, any casing
        // still resolves.
        let only = vec![base.clone()];
        assert_eq!(
            select_runtime_manifest(&only, "Release-Pip-Gfx120X-All-7-13-0")?.runtime_key,
            base.runtime_key
        );

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
    fn runtime_import_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-import-replace");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;

        let error = import_runtime_manifest(&paths, &exported_manifest, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("already exists"));
        import_runtime_manifest(&paths, &exported_manifest, true)?;

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

    /// The other half of the same derivation: an adopted environment that does
    /// expose a CMake path has the toolchain, and the manifest must say so or
    /// the next `rocm update` would quietly reinstall without it.
    #[test]
    fn runtime_adopt_records_devel_when_the_probe_finds_cmake() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-devel");
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
        let cmake_path = sdk_root.join("lib").join("cmake");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::create_dir_all(&cmake_path)?;
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
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                cmake_path: Some(cmake_path),
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

        assert!(
            adopted.devel,
            "a probe reporting a CMake path must record the toolchain"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_infers_ids_from_probe() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable.clone(),
            Some(external_root.clone()),
            None,
            None,
            None,
            false,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.13.0a20260423".to_owned()),
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.python_executable, python_executable);
        assert_eq!(request.install_root, external_root);
        assert_eq!(request.runtime_id, "therock-release:gfx120X-all");
        assert_eq!(
            request.runtime_key,
            "adopted-release-pip-gfx120x-all-7-13-0a20260423"
        );
        assert!(!request.replace);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_honors_nightly_channel() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer-nightly");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable,
            Some(external_root),
            None,
            None,
            Some("nightly".to_owned()),
            true,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.14.0a20260531".to_owned()),
                default_target_family: Some("gfx1151".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.runtime_id, "therock-nightly:gfx1151");
        assert_eq!(
            request.runtime_key,
            "adopted-nightly-pip-gfx1151-7-14-0a20260531"
        );
        assert!(request.replace);

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

    #[test]
    fn runtime_uninstall_unregisters_read_only_runtime_without_deleting_external_root() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-uninstall-read-only");
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
        import_runtime_manifest(&paths, &exported_manifest, false)?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(removed.runtime_key, manifest.runtime_key);
        assert!(removed.read_only);
        assert_eq!(removed.removed_install_root, None);
        assert!(manifest.install_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_removes_rocm_cli_managed_custom_prefix_root() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-prefix");
        let mut manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let prefix_root = root.join("user-chosen-prefix");
        fs::rename(&manifest.install_root, &prefix_root)?;
        manifest.install_root = prefix_root.clone();
        let python = prefix_root
            .join(if cfg!(windows) { "Scripts" } else { "bin" })
            .join(if cfg!(windows) {
                "python.exe"
            } else {
                "python"
            });
        manifest.python_executable = Some(python.display().to_string());
        fs::write(
            runtime_manifest_path(&paths, &manifest.runtime_key),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::write(
            prefix_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert_eq!(
            removed.removed_install_root.as_deref(),
            Some(prefix_root.as_path())
        );
        assert!(!prefix_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_leaves_folder_on_local_manifest_mismatch() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-manifest-mismatch");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        fs::remove_file(manifest.install_root.join(".rocm-cli-runtime.json"))?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &manifest.runtime_key)?;

        assert!(removed.manifest_mismatch);
        assert!(!removed.read_only);
        assert_eq!(removed.removed_install_root, None);
        assert!(manifest.install_root.exists());
        assert!(!runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn ensure_runtime_install_root_rejects_protected_system_path() {
        // `prune` (storage.rs) refuses a runtime whose folder sits in a
        // protected system location before it ever calls
        // `should_remove_runtime_install_root`. A hand-edited or corrupted
        // registry entry could point a direct `runtimes uninstall <key>`
        // manifest's `install_root` at the same kind of path while still
        // carrying a matching in-tree `.rocm-cli-runtime.json`, slipping past
        // `local_runtime_manifest_matches`. The single source of truth for
        // "may ROCm CLI delete this folder?" must refuse it too, regardless
        // of caller.
        let protected = if cfg!(windows) {
            PathBuf::from("C:/Windows/rocm-cli-test-runtime")
        } else {
            PathBuf::from("/etc/rocm-cli-test-runtime")
        };

        let err = ensure_runtime_install_root_is_safe_to_remove(&protected)
            .expect_err("protected system path must be refused");
        assert!(err.to_string().contains("protected system location"));
    }

    /// The gate above is the last thing between a registry entry and
    /// `fs::remove_dir_all`, and the path it is handed is text somebody else
    /// wrote. One folder has many spellings, so the refusal has to survive all
    /// of them — a trailing separator, a doubled one, a `.`, or a `..` that
    /// walks back out of the user's home directory. Each of these names `/etc`.
    #[test]
    #[cfg(unix)]
    fn ensure_runtime_install_root_rejects_every_spelling_of_a_protected_path() {
        let home = rocm_core::runtime_home_dir().expect("a home directory");
        let home = home.display().to_string();
        let spellings = [
            "/etc/".to_owned(),
            "//etc".to_owned(),
            "/./etc".to_owned(),
            "/etc/.".to_owned(),
            format!("{home}/../../etc"),
        ];

        let accepted: Vec<String> = spellings
            .into_iter()
            .filter(|text| ensure_runtime_install_root_is_safe_to_remove(Path::new(text)).is_ok())
            .collect();

        assert!(
            accepted.is_empty(),
            "accepted as safe to recursively delete: {accepted:?}"
        );
    }

    #[test]
    fn plan_runtime_uninstall_does_not_mutate() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-plan-dry-run");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let config = RocmCliConfig::default();

        let plan = plan_runtime_uninstall(&paths, &config, &manifest.runtime_key)?;

        assert!(plan.will_remove_install_root());
        assert!(manifest.install_root.exists());
        assert!(runtime_manifest_path(&paths, &manifest.runtime_key).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_revalidation_detects_install_root_change() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-revalidate-install-root");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let config = RocmCliConfig::default();

        let plan = plan_runtime_uninstall(&paths, &config, &manifest.runtime_key)?;
        assert!(plan.will_remove_install_root());

        // Simulate another process relocating this runtime's install root
        // while the uninstall confirmation prompt was waiting on the user.
        let relocated_root = paths
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join("relocated-install-root");
        fs::rename(&manifest.install_root, &relocated_root)?;
        let mut relocated_manifest = manifest.clone();
        relocated_manifest.install_root = relocated_root.clone();
        fs::write(
            relocated_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&relocated_manifest)?,
        )?;
        fs::write(
            runtime_manifest_path(&paths, &manifest.runtime_key),
            serde_json::to_vec_pretty(&relocated_manifest)?,
        )?;

        let result = revalidate_runtime_uninstall_plan(&paths, &config, plan);
        assert!(
            result.is_err(),
            "revalidation should refuse a plan whose install_root moved since it was shown"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_revalidation_detects_runtime_id_change() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-revalidate-runtime-id");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let config = RocmCliConfig::default();

        let plan = plan_runtime_uninstall(&paths, &config, &manifest.runtime_key)?;
        assert!(plan.will_remove_install_root());

        // Simulate another process re-registering this runtime_key under a
        // different runtime_id while the uninstall confirmation prompt was
        // waiting on the user. Both the registry entry and the local marker
        // are updated together so `install_root_decision` stays `Remove` and
        // only `runtime_id` differs from the plan the user approved.
        let mut relabeled_manifest = manifest.clone();
        relabeled_manifest.runtime_id = "therock-release:gfx120X-all-relabeled".to_owned();
        fs::write(
            manifest.install_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&relabeled_manifest)?,
        )?;
        fs::write(
            runtime_manifest_path(&paths, &manifest.runtime_key),
            serde_json::to_vec_pretty(&relabeled_manifest)?,
        )?;

        let result = revalidate_runtime_uninstall_plan(&paths, &config, plan);
        assert!(
            result.is_err(),
            "revalidation should refuse a plan whose runtime_id changed since it was shown"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_revalidation_detects_was_active_change() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-revalidate-was-active");
        let target = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let other = write_test_pip_runtime(
            &paths,
            "release-pip-gfx110x-all-7-12-0",
            "therock-release:gfx110X-all",
            "7.12.0",
            10,
        )?;
        let mut config = RocmCliConfig {
            active_runtime_key: Some(other.runtime_key),
            ..RocmCliConfig::default()
        };

        let plan = plan_runtime_uninstall(&paths, &config, &target.runtime_key)?;
        assert!(!plan.was_active);

        // Simulate another process activating the target runtime while the
        // uninstall confirmation prompt was waiting on the user.
        config.active_runtime_key = Some(target.runtime_key);
        config.save(&paths)?;

        let result = revalidate_runtime_uninstall_plan(&paths, &config, plan);
        assert!(
            result.is_err(),
            "revalidation should refuse a plan whose was_active changed since it was shown"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_revalidation_detects_clears_default_runtime_change() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-revalidate-clears-default");
        let shared_runtime_id = "therock-release:gfx120X-all";
        let target = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            shared_runtime_id,
            "7.13.0",
            20,
        )?;
        let other = write_test_pip_runtime(
            &paths,
            "release-pip-gfx110x-all-7-12-0",
            "therock-release:gfx110X-all",
            "7.12.0",
            10,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(shared_runtime_id.to_owned()),
            active_runtime_key: Some(other.runtime_key),
            ..RocmCliConfig::default()
        };

        let plan = plan_runtime_uninstall(&paths, &config, &target.runtime_key)?;
        assert!(!plan.was_active);
        assert!(
            plan.clears_default_runtime,
            "target is the only install with the stale default runtime_id"
        );

        // Simulate another process installing a sibling that shares the
        // target's runtime_id while the uninstall confirmation prompt was
        // waiting on the user; the default would then survive on that
        // sibling instead of being cleared.
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-1",
            shared_runtime_id,
            "7.13.1",
            30,
        )?;

        let result = revalidate_runtime_uninstall_plan(&paths, &config, plan);
        assert!(
            result.is_err(),
            "revalidation should refuse a plan whose clears_default_runtime changed since it was shown"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_revalidation_detects_install_root_decision_change() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-revalidate-install-root-decision");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let config = RocmCliConfig::default();

        let plan = plan_runtime_uninstall(&paths, &config, &manifest.runtime_key)?;
        assert_eq!(plan.install_root_decision, InstallRootDecision::Remove);

        // Simulate another process overwriting the in-tree marker with one
        // for a different install while the uninstall confirmation prompt
        // was waiting on the user; the registry entry (and thus runtime_id
        // and install_root) is left untouched, so only install_root_decision
        // should differ from the plan the user approved.
        let mut mismatched_marker = manifest.clone();
        mismatched_marker.runtime_key = "some-other-runtime-key".to_owned();
        fs::write(
            manifest.install_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&mismatched_marker)?,
        )?;

        let result = revalidate_runtime_uninstall_plan(&paths, &config, plan);
        assert!(
            result.is_err(),
            "revalidation should refuse a plan whose install_root_decision changed since it was shown"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn confirm_and_revalidate_runtime_uninstall_refuses_state_changed_during_confirmation()
    -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-confirm-and-revalidate-wiring");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let config = RocmCliConfig::default();
        config.save(&paths)?;

        let plan = plan_runtime_uninstall(&paths, &config, &manifest.runtime_key)?;
        assert!(plan.will_remove_install_root());

        // The injected "confirm" closure plays the role of the user
        // approving the prompt; it relocates the install root before
        // returning, simulating another process racing the confirmation
        // exactly as `runtime_uninstall_revalidation_detects_install_root_change`
        // does for the leaf function. This exercises the actual
        // confirm -> reload-config -> revalidate wiring, not just the
        // revalidation function in isolation: if the reload/revalidate
        // calls were ever dropped from `confirm_and_revalidate_runtime_uninstall`,
        // the call would silently succeed on the stale plan and this
        // test's `result.is_err()` assertion below would fail, catching
        // the regression.
        let relocated_root = paths
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join("relocated-install-root");
        let install_root = manifest.install_root.clone();
        let runtime_key = manifest.runtime_key.clone();
        let paths_for_confirm = paths.clone();
        let result = confirm_and_revalidate_runtime_uninstall(&paths, plan, move || {
            fs::rename(&install_root, &relocated_root)?;
            let mut relocated_manifest = manifest.clone();
            relocated_manifest.install_root = relocated_root.clone();
            fs::write(
                relocated_root.join(".rocm-cli-runtime.json"),
                serde_json::to_vec_pretty(&relocated_manifest)?,
            )?;
            fs::write(
                runtime_manifest_path(&paths_for_confirm, &runtime_key),
                serde_json::to_vec_pretty(&relocated_manifest)?,
            )?;
            Ok(true)
        });

        assert!(
            result.is_err(),
            "confirm_and_revalidate_runtime_uninstall must refuse to proceed when the runtime \
             state changed while the confirmation callback was running"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_uninstall_clears_default_runtime_id_for_last_sibling_even_when_not_active()
    -> Result<()> {
        // `runtime_id` is shared across side-by-side installs of the same
        // release, while `runtime_key` is unique per install and
        // `config.default_runtime_id` tracks by the shared `runtime_id`,
        // independently of `config.active_runtime_key`. A stale
        // `default_runtime_id` left over from before a *different* runtime
        // was activated must still be cleared once its last remaining
        // sibling is uninstalled — even though that sibling is not, and
        // never was, the active runtime (`was_active` is pinned to the
        // unrelated active runtime and never falls back to the
        // default-id-uniqueness check while that active runtime is still
        // installed).
        let (root, paths) = test_paths("runtime-uninstall-shared-default-id");
        let shared_runtime_id = "therock-release:gfx120X-all";
        let manifest_a = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0-a",
            shared_runtime_id,
            "7.13.0",
            20,
        )?;
        let manifest_b = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0-b",
            shared_runtime_id,
            "7.13.0",
            21,
        )?;
        let active_manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx1151-7-14-0",
            "therock-release:gfx1151",
            "7.14.0",
            22,
        )?;
        // `default_runtime_id` is stale, left over from before
        // `active_manifest` was activated; `active_runtime_key` now points
        // at a manifest with a completely different `runtime_id`.
        let mut config = RocmCliConfig {
            default_runtime_id: Some(shared_runtime_id.to_owned()),
            active_runtime_key: Some(active_manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;

        // Removing the first sibling leaves the other one behind, so the
        // stale default (which still resolves to a real, remaining install)
        // must not be cleared.
        let plan_b = plan_runtime_uninstall(&paths, &config, &manifest_b.runtime_key)?;
        assert!(!plan_b.was_active);
        assert!(!plan_b.clears_default_runtime);
        let removed_b = uninstall_runtime(&paths, &mut config, &manifest_b.runtime_key)?;
        assert!(!removed_b.was_active);
        assert!(!removed_b.default_runtime_cleared);
        assert_eq!(
            config.default_runtime_id.as_deref(),
            Some(shared_runtime_id)
        );

        // Removing the last remaining sibling must clear the stale default
        // even though this install was never the active one, and
        // `active_runtime_key` still points at the unrelated, still-installed
        // `active_manifest` throughout.
        let plan_a = plan_runtime_uninstall(&paths, &config, &manifest_a.runtime_key)?;
        assert!(!plan_a.was_active);
        assert!(plan_a.clears_default_runtime);
        let removed_a = uninstall_runtime(&paths, &mut config, &manifest_a.runtime_key)?;
        assert!(!removed_a.was_active);
        assert!(removed_a.default_runtime_cleared);
        assert_eq!(config.default_runtime_id, None);
        assert_eq!(
            config.active_runtime_key.as_deref(),
            Some(active_manifest.runtime_key.as_str())
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn runtime_adopt_preserves_venv_python_symlink_path() -> Result<()> {
        use std::os::unix::fs::symlink;

        let (root, paths) = test_paths("runtime-adopt-python-symlink");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join("bin");
        let system_python = root.join("python3.12");
        let python_executable = scripts_dir.join("python");
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join("libamdhip64.so");
        let hipblas = sdk_bin.join("libhipblas.so");
        fs::write(&system_python, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;
        symlink(&system_python, &python_executable)?;

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable: python_executable.clone(),
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-symlink-python".to_owned(),
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

        assert_eq!(
            adopted.python_executable.as_deref(),
            Some(python_executable.as_path().to_string_lossy().as_ref())
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_rejects_runtime_id_without_family() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-no-family");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::write(&python_executable, "python")?;

        let error = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root),
                bin_path: Some(sdk_bin),
                ..therock::RocmSdkPythonProbe::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("must include a TheRock family suffix"));
        assert!(!runtime_registry_dir(&paths).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-replace");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
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
            ..therock::RocmSdkPythonProbe::default()
        };
        let request = AdoptRuntimeRequest {
            python_executable: python_executable.clone(),
            install_root: external_root.clone(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
            replace: false,
        };

        adopt_runtime_from_probe(&paths, request.clone(), probe.clone())?;
        let error = adopt_runtime_from_probe(&paths, request, probe.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already exists"));

        adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: true,
            },
            probe,
        )?;

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
