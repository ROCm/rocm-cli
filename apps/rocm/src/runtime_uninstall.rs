// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm runtimes uninstall`: planning the removal of a registered runtime,
//! re-validating that plan after the confirmation prompt, and applying it
//! (unregistering the runtime, clearing the active and default state that named
//! it, and removing its install root when ROCm CLI owns it). When that root
//! holds the Lemonade install its record points at (a self-managed engine
//! installs inside the active runtime's folder), the plan says so and applying
//! it clears the record, so Lemonade does not keep naming deleted files.
//!
//! Relocated from `runtimes_cmd.rs`; the Lemonade record handling is new.
//! `runtimes_cmd.rs` calls into this module from `runtimes()` and re-exports
//! `uninstall_runtime`, `should_remove_runtime_install_root`,
//! `InstallRootDecision` and `lemonade_record_report_line` for `storage.rs`.
//! The active-runtime marker check this module makes
//! (`active_runtime_marker_matches`, over the `ActiveRuntimeMarker` the
//! activation writes) and the path comparison it needs live in
//! `runtime_registry.rs`, so this module does not depend on `runtimes_cmd.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig};
use rocm_engine_lemonade::RecordCleanup;

use crate::cli_report;
use crate::runtime_registry::{
    active_runtime_marker_matches, active_runtime_marker_path, current_runtime_manifest,
    paths_equivalent, runtime_manifest_path, select_runtime_manifest,
};
use crate::therock;

#[derive(Debug, Clone)]
pub(crate) struct RuntimeUninstallResult {
    pub(crate) runtime_id: String,
    pub(crate) runtime_key: String,
    pub(crate) registry_path: PathBuf,
    pub(crate) removed_install_root: Option<PathBuf>,
    /// What happened to Lemonade's install record once the folder was removed:
    /// cleared, left dangling because deleting it failed, or not checkable.
    pub(crate) lemonade_record: RecordCleanup,
    pub(crate) read_only: bool,
    pub(crate) manifest_mismatch: bool,
    pub(crate) was_active: bool,
    pub(crate) default_runtime_cleared: bool,
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
pub(crate) struct RuntimeUninstallPlan {
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
    /// Lemonade's runtime folder when its install record points inside this
    /// runtime's folder: a self-managed engine installs there while the runtime
    /// is active, so removing the folder removes that engine too.
    lemonade_install_inside: Option<PathBuf>,
    /// Whether this runtime's folder holds a Lemonade tree at all, installed or
    /// partly installed. Only then can an unreadable install record matter to
    /// this uninstall.
    holds_lemonade_tree: bool,
}

impl RuntimeUninstallPlan {
    fn will_remove_install_root(&self) -> bool {
        self.install_root_decision.should_remove() && self.manifest.install_root.exists()
    }
}

/// The line `runtimes uninstall` (and `storage remove-old-installs`, which
/// delegates to it) report about Lemonade's install record, or `None` when the
/// removed runtime had nothing to do with it.
///
/// Built from the outcome the record handling returned, so each claim is the one
/// that happened: "no longer recorded" only follows a record that was deleted,
/// and a record that could not be deleted or read says so and how to fix it.
pub(crate) fn lemonade_record_report_line(outcome: &RecordCleanup) -> Option<String> {
    match outcome {
        RecordCleanup::NotAffected => None,
        RecordCleanup::Cleared(lemonade) => Some(format!(
            "its install at {} was inside this runtime and was removed with it; \
             Lemonade is no longer recorded as installed",
            lemonade.display()
        )),
        RecordCleanup::Failed {
            runtime_dir,
            record,
            error,
        } => Some(format!(
            "its install at {} was inside this runtime and was removed with it, but \
             Lemonade's install record could not be deleted ({error}); delete {} \
             so Lemonade is not left pointing at files that are gone",
            runtime_dir.display(),
            record.display()
        )),
        RecordCleanup::Unreadable(error) => Some(format!(
            "Lemonade's install record could not be read ({error}), so it was not \
             checked against this runtime; `rocm engines install lemonade` rewrites it \
             and keeps a current Lemonade install"
        )),
    }
}

pub(crate) fn print_runtime_uninstall_plan(plan: &RuntimeUninstallPlan) {
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
    if plan.will_remove_install_root()
        && let Some(lemonade) = plan.lemonade_install_inside.as_ref()
    {
        report = report.detail(
            "lemonade",
            format!(
                "its install at {} is inside this runtime folder and would be removed with it",
                lemonade.display()
            ),
        );
    }
    print!("{}", report.render());
}

pub(crate) fn plan_runtime_uninstall(
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
    let holds_lemonade_tree =
        rocm_engine_lemonade::install_tree_dirs(&rocm_engine_lemonade::install_tree_root(
            paths,
            Some(&manifest.install_root.join("engines")),
        ))
        .iter()
        .any(|part| part.exists());
    let lemonade_install_inside = rocm_engine_lemonade::recorded_runtime_dir(paths)
        .filter(|dir| rocm_core::runtime_path_is_same_or_inside(dir, &manifest.install_root));
    Ok(RuntimeUninstallPlan {
        manifest,
        registry_path,
        was_active,
        clears_default_runtime,
        install_root_decision,
        lemonade_install_inside,
        holds_lemonade_tree,
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

pub(crate) enum RuntimeUninstallConfirmation {
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
pub(crate) fn confirm_and_revalidate_runtime_uninstall(
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

pub(crate) fn apply_runtime_uninstall(
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
        lemonade_install_inside: _,
        holds_lemonade_tree,
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

    // Decided from the folder actually removed, not from the plan: the record
    // is only stale once its files are gone.
    let lemonade_record = removed_install_root
        .as_deref()
        .map_or(RecordCleanup::NotAffected, |removed| {
            rocm_engine_lemonade::forget_install_record_inside(paths, removed)
        });
    // An unreadable record says nothing about this runtime unless the runtime
    // could have held Lemonade; otherwise the line would repeat on every runtime
    // uninstall for as long as the record stays unreadable.
    let lemonade_record = match lemonade_record {
        RecordCleanup::Unreadable(_) if !holds_lemonade_tree => RecordCleanup::NotAffected,
        other => other,
    };

    Ok(RuntimeUninstallResult {
        runtime_id: manifest.runtime_id,
        runtime_key: manifest.runtime_key,
        registry_path,
        removed_install_root,
        lemonade_record,
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
    ensure_runtime_install_root_is_safe_to_remove_for_home(
        path,
        rocm_core::runtime_home_dir().as_deref(),
    )
}

/// [`ensure_runtime_install_root_is_safe_to_remove`] judged against `home`.
fn ensure_runtime_install_root_is_safe_to_remove_for_home(
    path: &Path,
    home: Option<&Path>,
) -> Result<()> {
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
    // runtime-folder rule (`rocm uninstall`'s own roots have their own) refuses
    // it for every caller, including a direct `runtimes uninstall <key>`.
    if let Some(why) = rocm_core::runtime_protected_location_for_home(path, home) {
        bail!(
            "refusing to remove runtime folder {} because it is {}",
            path.display(),
            why.describe()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_adopt::import_runtime_manifest;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::test_paths;

    /// While a runtime is active, `rocm serve` installs Lemonade inside that
    /// runtime's folder and records it there. Uninstalling the runtime deletes
    /// the folder; the record must go with it, or Lemonade keeps pointing at
    /// files that no longer exist. A record pointing elsewhere is left alone.
    #[test]
    fn runtime_uninstall_clears_a_lemonade_record_pointing_into_the_removed_folder() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-lemonade");
        let held = write_test_pip_runtime(
            &paths,
            "release-pip-gfx1151-7-14-1",
            "therock-release:gfx1151",
            "7.14.1",
            20,
        )?;
        let other = write_test_pip_runtime(
            &paths,
            "release-pip-gfx1151-7-13-0",
            "therock-release:gfx1151",
            "7.13.0",
            10,
        )?;
        let tree = rocm_engine_lemonade::install_tree_root(
            &paths,
            Some(&held.install_root.join("engines")),
        );
        crate::test_support::plant_lemonade_tree(&tree, 16)?;
        crate::test_support::record_lemonade_install(&paths, &tree)?;
        let mut config = RocmCliConfig::default();

        // Removing a runtime that does not hold the tree keeps the record.
        let removed = uninstall_runtime(&paths, &mut config, &other.runtime_key)?;
        assert_eq!(removed.lemonade_record, RecordCleanup::NotAffected);
        assert_eq!(
            rocm_engine_lemonade::recorded_runtime_dir(&paths),
            Some(tree.join("runtime"))
        );

        let plan = plan_runtime_uninstall(&paths, &config, &held.runtime_key)?;
        assert_eq!(plan.lemonade_install_inside, Some(tree.join("runtime")));
        let removed = apply_runtime_uninstall(&paths, &mut config, plan)?;

        assert_eq!(
            removed.lemonade_record,
            RecordCleanup::Cleared(tree.join("runtime"))
        );
        assert!(!tree.exists());
        assert_eq!(rocm_engine_lemonade::recorded_runtime_dir(&paths), None);
        let line = lemonade_record_report_line(&removed.lemonade_record).unwrap();
        assert!(
            line.contains(&tree.join("runtime").display().to_string())
                && line.ends_with("Lemonade is no longer recorded as installed"),
            "{line}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// A record that cannot be read may well name the runtime being removed, so
    /// the uninstall cannot tell whether it is left dangling. When the runtime
    /// held a Lemonade tree it must say so rather than treat the record as
    /// absent, name the cheap command that rewrites it (a plain
    /// `rocm engines install lemonade`, shown to do so by
    /// `a_plain_install_replaces_a_corrupt_record_and_keeps_the_current_tree` in
    /// the engine), not the wiping `--reinstall`, and leave the file alone.
    #[test]
    fn runtime_uninstall_reports_a_lemonade_record_it_could_not_read() -> Result<()> {
        let (root, paths) = test_paths("runtime-uninstall-lemonade-unreadable");
        let held = write_test_pip_runtime(
            &paths,
            "release-pip-gfx1151-7-14-1",
            "therock-release:gfx1151",
            "7.14.1",
            20,
        )?;
        crate::test_support::plant_lemonade_tree(
            &rocm_engine_lemonade::install_tree_root(
                &paths,
                Some(&held.install_root.join("engines")),
            ),
            16,
        )?;
        let record = paths.engine_manifests_dir("lemonade").join("runtime.json");
        fs::create_dir_all(record.parent().expect("record has a parent"))?;
        fs::write(&record, b"{ not json")?;
        let mut config = RocmCliConfig::default();

        let removed = uninstall_runtime(&paths, &mut config, &held.runtime_key)?;

        assert!(
            matches!(removed.lemonade_record, RecordCleanup::Unreadable(_)),
            "{:?}",
            removed.lemonade_record
        );
        let line = lemonade_record_report_line(&removed.lemonade_record).unwrap();
        assert!(line.contains("could not be read"), "{line}");
        assert!(
            line.contains("`rocm engines install lemonade` rewrites it"),
            "{line}"
        );
        assert!(!line.contains("--reinstall"), "{line}");
        assert!(record.exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The same unreadable record must not be reported on a runtime that never
    /// held Lemonade: nothing about that uninstall depends on it, and the line
    /// would otherwise repeat on every runtime uninstall.
    #[test]
    fn an_unreadable_lemonade_record_is_not_reported_for_a_runtime_without_lemonade() -> Result<()>
    {
        let (root, paths) = test_paths("runtime-uninstall-lemonade-unrelated");
        let runtime = write_test_pip_runtime(
            &paths,
            "release-pip-gfx1151-7-14-1",
            "therock-release:gfx1151",
            "7.14.1",
            20,
        )?;
        let record = paths.engine_manifests_dir("lemonade").join("runtime.json");
        fs::create_dir_all(record.parent().expect("record has a parent"))?;
        fs::write(&record, b"{ not json")?;
        let mut config = RocmCliConfig::default();

        let plan = plan_runtime_uninstall(&paths, &config, &runtime.runtime_key)?;
        assert!(!plan.holds_lemonade_tree);
        let removed = apply_runtime_uninstall(&paths, &mut config, plan)?;

        assert_eq!(removed.lemonade_record, RecordCleanup::NotAffected);
        assert_eq!(lemonade_record_report_line(&removed.lemonade_record), None);
        assert!(
            removed.removed_install_root.is_some(),
            "the runtime was removed"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// Each outcome's line says only what happened. A failed deletion must
    /// name the record to delete and must not claim Lemonade is no longer
    /// recorded; no outcome for an unrelated runtime prints anything. (That
    /// deleting the named record really clears the condition is shown by
    /// `a_record_that_cannot_be_deleted_is_reported_not_swallowed` in the
    /// engine.)
    #[test]
    fn the_lemonade_record_line_matches_what_happened() {
        assert_eq!(
            lemonade_record_report_line(&RecordCleanup::NotAffected),
            None
        );
        let cleared =
            lemonade_record_report_line(&RecordCleanup::Cleared(PathBuf::from("/rt/lemonade")))
                .unwrap();
        assert!(
            cleared.contains("no longer recorded as installed"),
            "{cleared}"
        );
        let failed = lemonade_record_report_line(&RecordCleanup::Failed {
            runtime_dir: PathBuf::from("/rt/lemonade"),
            record: PathBuf::from("/data/engines/lemonade/manifests/runtime.json"),
            error: "sharing violation".to_owned(),
        })
        .unwrap();
        assert!(failed.contains("sharing violation"), "{failed}");
        assert!(
            failed.contains("delete /data/engines/lemonade/manifests/runtime.json"),
            "{failed}"
        );
        assert!(!failed.contains("no longer recorded"), "{failed}");
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
        // `local_runtime_manifest_matches`. The runtime-folder rule for
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

    /// `rocm runtimes uninstall` and `rocm storage remove-old-installs` share
    /// the protected-location predicate with `rocm uninstall`, so a runtime
    /// whose `install_root` is the home folder itself (`--prefix ~`) is refused
    /// too — and the refusal says it is the home folder, not a system location.
    #[test]
    fn ensure_runtime_install_root_refuses_the_home_folder_by_name() {
        // A fixed home, so the check cannot fall back to some other refusal
        // on a host whose home is `/`.
        let home = Path::new("/home/alice");
        let err = ensure_runtime_install_root_is_safe_to_remove_for_home(home, Some(home))
            .expect_err("the home folder must be refused");
        assert_eq!(
            err.to_string(),
            "refusing to remove runtime folder /home/alice because it is your home folder"
        );
        // The real gate consults the real home.
        let real_home = rocm_core::runtime_home_dir().expect("a home directory");
        if real_home.parent().is_some() {
            let err = ensure_runtime_install_root_is_safe_to_remove(&real_home)
                .expect_err("the real home folder must be refused");
            assert!(err.to_string().ends_with("is your home folder"), "{err}");
        }
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
}
