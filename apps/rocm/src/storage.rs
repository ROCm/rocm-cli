// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm storage` command handlers.
//!
//! Every SDK install that resolves a new version gets its own multi-gigabyte
//! runtime folder (`runtime_key()` folds the version into the key), and nothing
//! ever removed the old ones. This module adds the two missing halves: a
//! read-only report of what is on disk, and a bounded, heavily guarded way to
//! reclaim the space.
//!
//! The removal path deliberately owns no deletion logic of its own. Runtime
//! removal delegates to [`crate::uninstall_runtime`] per selected key so
//! config/marker cleanup stays in one place, and the guard that decides whether
//! a folder may be touched at all is the existing
//! [`crate::should_remove_runtime_install_root`]. What lives here is only the
//! *selection* policy — which is exactly the part that needs to be conservative,
//! because prune acts on a set the user never enumerated.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, interactive_terminal, runtime_install_root_is_protected};
use serde::Serialize;

use crate::{
    ActiveRuntimeMarker, StorageCommand, UninstallPlan, UninstallPlanEntry,
    active_runtime_marker_path, confirm_uninstall, format_bytes, remove_path,
    should_remove_runtime_install_root, therock,
};

/// Recent installs kept per channel/format/family by default: the one in use
/// plus one rollback target. This is the natural floor rather than a tuned
/// value — see the maintainer question in the pull request description.
pub(crate) const DEFAULT_KEEP: usize = 2;

// ---------------------------------------------------------------------------
// Best-effort directory sizing
// ---------------------------------------------------------------------------

/// Result of walking one directory tree.
///
/// `complete` is false when any entry below the root could not be read. A
/// partial or failed measurement is never an error: a user who is out of disk
/// needs the rest of the report more than they need strictness (same tolerant
/// posture as `prune_old_logs`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Measurement {
    pub bytes: u64,
    pub complete: bool,
}

/// Walk `path` and total the apparent size of every regular file below it.
///
/// Symlinks *below* the root are counted as their own (tiny) size and never
/// followed, so a symlinked-in model folder is not attributed to the runtime
/// that points at it. The root itself is resolved with `metadata` rather than
/// `symlink_metadata`: relocating a cache to a bigger disk by symlinking it is
/// normal, and measuring the link instead of its target reported a multi-gigabyte
/// cache as a few bytes, marked complete. This also keeps sizing consistent with
/// the `Path::exists` gates in [`PathUsage::measure`] and [`build_report`],
/// which follow the link too.
pub(crate) fn measure_path(path: &Path) -> Measurement {
    let Ok(metadata) = std::fs::metadata(path) else {
        return Measurement {
            bytes: 0,
            complete: false,
        };
    };
    if !metadata.is_dir() {
        return Measurement {
            bytes: metadata.len(),
            complete: true,
        };
    }

    let mut total = 0_u64;
    let mut complete = true;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            complete = false;
            continue;
        };
        for entry in entries {
            let Ok(entry) = entry else {
                complete = false;
                continue;
            };
            let Ok(metadata) = entry
                .metadata()
                .or_else(|_| entry.path().symlink_metadata())
            else {
                complete = false;
                continue;
            };
            if metadata.is_dir() && !metadata.is_symlink() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Measurement {
        bytes: total,
        complete,
    }
}

/// One measured location in the report.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PathUsage {
    pub label: String,
    pub path: PathBuf,
    pub exists: bool,
    /// `None` when the size could not be determined at all.
    pub size_bytes: Option<u64>,
    /// False when part of the tree was unreadable and the total is a floor.
    pub size_complete: bool,
    /// Plain-English note, used to flag caches shared with other tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl PathUsage {
    fn measure(label: impl Into<String>, path: PathBuf, note: Option<String>) -> Self {
        let exists = path.exists();
        let measurement = exists.then(|| measure_path(&path));
        Self {
            label: label.into(),
            exists,
            size_bytes: measurement.map(|value| value.bytes),
            size_complete: measurement.is_some_and(|value| value.complete),
            note,
            path,
        }
    }

    fn size_text(&self) -> String {
        match self.size_bytes {
            None if !self.exists => "not present".to_owned(),
            None => "unknown size (unreadable)".to_owned(),
            Some(bytes) if self.size_complete => format_bytes(bytes),
            Some(bytes) => format!("{} or more (part unreadable)", format_bytes(bytes)),
        }
    }
}

// ---------------------------------------------------------------------------
// Retention policy
// ---------------------------------------------------------------------------

/// Why a runtime is being kept. Every skip is reported, in the existing
/// "Left alone:" style, so the user can see what prune declined to touch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HoldReason {
    Active,
    Previous,
    Default,
    Marker,
    NotOwned,
    WithinKeepLimit,
}

impl HoldReason {
    pub(crate) const fn describe(&self) -> &'static str {
        match self {
            Self::Active => "in use right now",
            Self::Previous => "the rollback target for `rocm runtimes rollback`",
            Self::Default => "the configured default",
            Self::Marker => "named by the active install marker",
            Self::NotOwned => "added with adopt or import, so ROCm CLI does not own the folder",
            Self::WithinKeepLimit => "one of the most recent installs kept for this GPU family",
        }
    }
}

/// The inputs the retention decision depends on, split out from `AppPaths` so
/// the policy is unit-testable without touching the filesystem.
#[derive(Debug, Clone, Default)]
pub(crate) struct RetentionInputs {
    pub active_runtime_key: Option<String>,
    pub previous_runtime_key: Option<String>,
    pub default_runtime_id: Option<String>,
    pub marker_runtime_keys: Vec<String>,
}

impl RetentionInputs {
    fn from_config(config: &RocmCliConfig, marker: Option<&ActiveRuntimeMarker>) -> Self {
        let mut marker_runtime_keys = Vec::new();
        if let Some(marker) = marker {
            marker_runtime_keys.push(marker.runtime_key.clone());
            if let Some(previous) = marker.previous_runtime_key.clone() {
                marker_runtime_keys.push(previous);
            }
        }
        Self {
            active_runtime_key: config.active_runtime_key.clone(),
            previous_runtime_key: config.previous_runtime_key.clone(),
            default_runtime_id: config.default_runtime_id.clone(),
            marker_runtime_keys,
        }
    }
}

fn matches_ignore_case(value: Option<&str>, other: &str) -> bool {
    value.is_some_and(|value| value.eq_ignore_ascii_case(other))
}

/// The install the configured default actually resolves to, if any.
///
/// `default_runtime_id` carries no version — it is `therock-<channel>:<family>`
/// — so every install of one channel and family shares it, and installing or
/// updating sets it on every real machine. Holding every id match would
/// therefore hold the whole family and make `--keep` inert. The default only
/// holds an install when it names exactly one, which is the same
/// "matches exactly one" rule `current_runtime_manifest` applies when it
/// resolves the default everywhere else. When it is ambiguous the install that
/// is genuinely in use is still held by [`HoldReason::Active`].
pub(crate) fn resolved_default_runtime_key(
    manifests: &[therock::InstalledRuntimeManifest],
    inputs: &RetentionInputs,
) -> Option<String> {
    let default_runtime_id = inputs.default_runtime_id.as_deref()?;
    let mut matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id));
    let first = matches.next()?;
    matches.next().is_none().then(|| first.runtime_key.clone())
}

/// The hold that applies to a runtime regardless of how recent it is.
///
/// `default_key` is the output of [`resolved_default_runtime_key`] for the same
/// manifest set.
///
/// Ordered most-specific first so the reported reason is the most useful one.
pub(crate) fn unconditional_hold(
    manifest: &therock::InstalledRuntimeManifest,
    inputs: &RetentionInputs,
    default_key: Option<&str>,
) -> Option<HoldReason> {
    let key = manifest.runtime_key.as_str();
    if matches_ignore_case(inputs.active_runtime_key.as_deref(), key) {
        return Some(HoldReason::Active);
    }
    if matches_ignore_case(inputs.previous_runtime_key.as_deref(), key) {
        return Some(HoldReason::Previous);
    }
    if matches_ignore_case(default_key, key) {
        return Some(HoldReason::Default);
    }
    if inputs
        .marker_runtime_keys
        .iter()
        .any(|marker_key| marker_key.eq_ignore_ascii_case(key))
    {
        return Some(HoldReason::Marker);
    }
    if manifest.read_only || manifest.imported_from.is_some() {
        return Some(HoldReason::NotOwned);
    }
    None
}

/// Retention group: a multi-GPU machine legitimately keeps one install per
/// family, so recency is only ever compared inside a channel/format/family.
fn retention_group(manifest: &therock::InstalledRuntimeManifest) -> (String, String, String) {
    (
        manifest.channel.to_ascii_lowercase(),
        manifest.format.to_ascii_lowercase(),
        manifest.family.to_ascii_lowercase(),
    )
}

/// Pure retention policy: which runtime keys are eligible for removal, and why
/// each of the others is being kept.
///
/// Filesystem-level guards (`should_remove_runtime_install_root`, the protected
/// path check) are applied later, in [`build_prune_plan`]; this half is
/// deliberately side-effect free so it can be tested directly.
pub(crate) fn select_runtimes_to_remove(
    manifests: &[therock::InstalledRuntimeManifest],
    inputs: &RetentionInputs,
    keep: usize,
) -> (Vec<String>, Vec<(String, HoldReason)>) {
    let mut held: Vec<(String, HoldReason)> = Vec::new();
    let mut groups: BTreeMap<(String, String, String), Vec<&therock::InstalledRuntimeManifest>> =
        BTreeMap::new();
    let default_key = resolved_default_runtime_key(manifests, inputs);

    for manifest in manifests {
        if let Some(reason) = unconditional_hold(manifest, inputs, default_key.as_deref()) {
            held.push((manifest.runtime_key.clone(), reason));
            continue;
        }
        groups
            .entry(retention_group(manifest))
            .or_default()
            .push(manifest);
    }

    let mut removable = Vec::new();
    for candidates in groups.values_mut() {
        // Newest first; ties broken by key so the order is deterministic.
        candidates.sort_by(|left, right| {
            right
                .installed_at_unix_ms
                .cmp(&left.installed_at_unix_ms)
                .then_with(|| left.runtime_key.cmp(&right.runtime_key))
        });
        for (index, manifest) in candidates.iter().enumerate() {
            if index < keep {
                held.push((manifest.runtime_key.clone(), HoldReason::WithinKeepLimit));
            } else {
                removable.push(manifest.runtime_key.clone());
            }
        }
    }

    removable.sort();
    held.sort_by(|left, right| left.0.cmp(&right.0));
    (removable, held)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimeUsage {
    pub runtime_key: String,
    pub runtime_id: String,
    pub channel: String,
    pub format: String,
    pub family: String,
    pub version: String,
    pub install_root: PathBuf,
    pub size_bytes: Option<u64>,
    pub size_complete: bool,
    pub active: bool,
    pub previous: bool,
    pub default_runtime: bool,
    /// `None` when nothing holds this install back from a future cleanup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hold_reason: Option<HoldReason>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StorageReport {
    pub runtimes: Vec<RuntimeUsage>,
    pub runtimes_total_bytes: u64,
    pub rocm_cli: Vec<PathUsage>,
    /// Caches ROCm CLI uses but does not own. Reported, never deleted.
    pub shared_with_other_tools: Vec<PathUsage>,
}

fn read_active_runtime_marker(paths: &AppPaths) -> Option<ActiveRuntimeMarker> {
    let path = active_runtime_marker_path(paths);
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Cache of downloaded SDK archives — everything here can be fetched again.
fn download_cache_dir(paths: &AppPaths) -> PathBuf {
    paths.cache_dir.join("therock")
}

/// Cache of downloaded helper-tool archives (the `uv` installer today).
fn tool_download_cache_dir(paths: &AppPaths) -> PathBuf {
    paths.cache_dir.join("tools")
}

/// Best guess at the user's shared `uv` cache, which is used by every uv
/// project on the machine, not just ROCm CLI.
fn shared_uv_cache_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("UV_CACHE_DIR").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(value));
    }
    let home = rocm_core::runtime_home_dir()?;
    if rocm_core::runtime_is_windows() {
        Some(home.join("AppData").join("Local").join("uv").join("cache"))
    } else {
        Some(home.join(".cache").join("uv"))
    }
}

/// Best guess at the shared Hugging Face cache. Model weights are large, slow
/// to refetch, sometimes gated, and often the user's own — reported only.
fn shared_model_cache_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("HF_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(value).join("hub"));
    }
    let home = rocm_core::runtime_home_dir()?;
    Some(home.join(".cache").join("huggingface"))
}

pub(crate) fn build_report(paths: &AppPaths, config: &RocmCliConfig) -> Result<StorageReport> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let marker = read_active_runtime_marker(paths);
    let inputs = RetentionInputs::from_config(config, marker.as_ref());
    let default_key = resolved_default_runtime_key(&manifests, &inputs);

    let mut runtimes = Vec::new();
    let mut runtimes_total_bytes = 0_u64;
    for manifest in &manifests {
        let measurement = manifest
            .install_root
            .exists()
            .then(|| measure_path(&manifest.install_root));
        if let Some(measurement) = measurement {
            runtimes_total_bytes = runtimes_total_bytes.saturating_add(measurement.bytes);
        }
        runtimes.push(RuntimeUsage {
            active: matches_ignore_case(
                inputs.active_runtime_key.as_deref(),
                &manifest.runtime_key,
            ),
            previous: matches_ignore_case(
                inputs.previous_runtime_key.as_deref(),
                &manifest.runtime_key,
            ),
            default_runtime: matches_ignore_case(default_key.as_deref(), &manifest.runtime_key),
            hold_reason: unconditional_hold(manifest, &inputs, default_key.as_deref()),
            runtime_key: manifest.runtime_key.clone(),
            runtime_id: manifest.runtime_id.clone(),
            channel: manifest.channel.clone(),
            format: manifest.format.clone(),
            family: manifest.family.clone(),
            version: manifest.version.clone(),
            install_root: manifest.install_root.clone(),
            size_bytes: measurement.map(|value| value.bytes),
            size_complete: measurement.is_some_and(|value| value.complete),
        });
    }
    runtimes.sort_by(|left, right| left.runtime_key.cmp(&right.runtime_key));

    let rocm_cli = vec![
        PathUsage::measure(
            "downloaded ROCm archives",
            download_cache_dir(paths),
            Some("can be downloaded again; safe to remove".to_owned()),
        ),
        PathUsage::measure(
            "downloaded helper tools",
            tool_download_cache_dir(paths),
            Some("can be downloaded again; safe to remove".to_owned()),
        ),
        PathUsage::measure("ROCm CLI cache folder", paths.cache_dir.clone(), None),
        PathUsage::measure("ROCm CLI data folder", paths.data_dir.clone(), None),
    ];

    let mut shared_with_other_tools = vec![PathUsage::measure(
        "downloaded models",
        paths.data_dir.join("models"),
        Some(
            "model files are slow to download again and may be yours; ROCm CLI never removes them"
                .to_owned(),
        ),
    )];
    if let Some(path) = shared_uv_cache_dir() {
        shared_with_other_tools.push(PathUsage::measure(
            "uv package cache",
            path,
            Some("shared with every other uv project on this computer".to_owned()),
        ));
    }
    if let Some(path) = shared_model_cache_dir() {
        shared_with_other_tools.push(PathUsage::measure(
            "Hugging Face model cache",
            path,
            Some("shared with other tools; ROCm CLI never removes it".to_owned()),
        ));
    }

    Ok(StorageReport {
        runtimes,
        runtimes_total_bytes,
        rocm_cli,
        shared_with_other_tools,
    })
}

pub(crate) fn render_report(report: &StorageReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "disk space used by ROCm CLI");
    let _ = writeln!(output);
    let _ = writeln!(output, "ROCm installs:");
    if report.runtimes.is_empty() {
        let _ = writeln!(output, "  none");
    }
    for runtime in &report.runtimes {
        let size = match runtime.size_bytes {
            None => "unknown size (folder missing or unreadable)".to_owned(),
            Some(bytes) if runtime.size_complete => format_bytes(bytes),
            Some(bytes) => format!("{} or more (part unreadable)", format_bytes(bytes)),
        };
        let _ = writeln!(
            output,
            "  - {} version={} {size}",
            runtime.runtime_key, runtime.version
        );
        let _ = writeln!(output, "      folder: {}", runtime.install_root.display());
        let mut labels = Vec::new();
        if runtime.active {
            labels.push("in use");
        }
        if runtime.previous {
            labels.push("rollback target");
        }
        if runtime.default_runtime {
            labels.push("default");
        }
        if !labels.is_empty() {
            let _ = writeln!(output, "      status: {}", labels.join(", "));
        }
    }
    let _ = writeln!(
        output,
        "  total: {}",
        format_bytes(report.runtimes_total_bytes)
    );

    let _ = writeln!(output);
    let _ = writeln!(output, "ROCm CLI folders:");
    for usage in &report.rocm_cli {
        let _ = writeln!(
            output,
            "  - {}: {} ({})",
            usage.label,
            usage.size_text(),
            usage.path.display()
        );
    }

    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Shared with other tools (never removed by ROCm CLI):"
    );
    for usage in &report.shared_with_other_tools {
        let _ = writeln!(
            output,
            "  - {}: {} ({})",
            usage.label,
            usage.size_text(),
            usage.path.display()
        );
        if let Some(note) = usage.note.as_deref() {
            let _ = writeln!(output, "      note: {note}");
        }
    }

    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "To free space: rocm storage remove-old-installs --dry-run"
    );
    output
}

// ---------------------------------------------------------------------------
// Prune plan
// ---------------------------------------------------------------------------

/// One install selected for removal, with the size the user gets back.
#[derive(Debug, Clone)]
pub(crate) struct PruneEntry {
    pub runtime_key: String,
    pub version: String,
    pub install_root: PathBuf,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PrunePlan {
    pub remove: Vec<PruneEntry>,
    pub skipped: Vec<String>,
}

pub(crate) fn build_prune_plan(
    paths: &AppPaths,
    config: &RocmCliConfig,
    keep: usize,
) -> Result<PrunePlan> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let marker = read_active_runtime_marker(paths);
    let inputs = RetentionInputs::from_config(config, marker.as_ref());
    let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, keep);

    let mut plan = PrunePlan::default();
    for (runtime_key, reason) in held {
        plan.skipped
            .push(format!("{runtime_key}: {}", reason.describe()));
    }

    for runtime_key in removable {
        let Some(manifest) = manifests
            .iter()
            .find(|manifest| manifest.runtime_key == runtime_key)
        else {
            continue;
        };

        // Belt and braces on top of `should_remove_runtime_install_root`: prune
        // acts on a set the user never typed out, so a runtime whose folder sits
        // in a protected system location is refused outright.
        if runtime_install_root_is_protected(&manifest.install_root) {
            plan.skipped.push(format!(
                "{runtime_key}: folder {} is in a protected system location",
                manifest.install_root.display()
            ));
            continue;
        }

        // The single source of truth for "may ROCm CLI delete this folder?".
        // It refuses read-only and imported records and requires a matching
        // in-tree manifest, and it runs `ensure_runtime_install_root_is_safe_to_remove`.
        match should_remove_runtime_install_root(manifest) {
            Ok(true) => {}
            Ok(false) => {
                plan.skipped.push(format!(
                    "{runtime_key}: ROCm CLI did not create this folder, so it is left in place"
                ));
                continue;
            }
            Err(error) => {
                plan.skipped
                    .push(format!("{runtime_key}: cannot be removed safely ({error})"));
                continue;
            }
        }

        plan.remove.push(PruneEntry {
            runtime_key: manifest.runtime_key.clone(),
            version: manifest.version.clone(),
            size_bytes: measure_path(&manifest.install_root).bytes,
            install_root: manifest.install_root.clone(),
        });
    }

    plan.skipped.sort();
    Ok(plan)
}

pub(crate) fn render_prune_plan(plan: &PrunePlan, keep: usize, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Old ROCm install review");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Keeping the {keep} most recent install(s) for each channel, format, and GPU family."
    );
    let _ = writeln!(output);
    if plan.remove.is_empty() {
        let _ = writeln!(output, "Nothing would be removed.");
    } else {
        let reclaimed: u64 = plan
            .remove
            .iter()
            .fold(0, |total, entry| total.saturating_add(entry.size_bytes));
        let _ = writeln!(
            output,
            "{} install(s) would be removed, freeing about {}:",
            plan.remove.len(),
            format_bytes(reclaimed)
        );
        for entry in &plan.remove {
            let _ = writeln!(
                output,
                "  - {} version={} {} ({})",
                entry.runtime_key,
                entry.version,
                format_bytes(entry.size_bytes),
                entry.install_root.display()
            );
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
        let _ = writeln!(
            output,
            "Nothing was changed. Re-run without --dry-run to remove."
        );
    }
    output
}

// ---------------------------------------------------------------------------
// Downloaded-archive plan
// ---------------------------------------------------------------------------

/// Collect the cached archives that can simply be downloaded again.
///
/// Reuses [`UninstallPlan`] so the review output and the removal loop stay the
/// same shape as `rocm uninstall`.
///
/// Both walk roots are checked with `symlink_metadata` before anything is
/// collected. Relocating the cache to a bigger disk with a symlink is a normal
/// thing for exactly the user running a disk-space command, and following the
/// link would collect real files outside the cache and delete them — with a
/// dry-run that shows them as ordinary cache paths, so nothing looks wrong.
/// Entries *below* the root were already handled correctly and still are.
pub(crate) fn build_downloads_plan(paths: &AppPaths) -> UninstallPlan {
    let mut plan = UninstallPlan::default();
    for (kind, dir) in [
        ("ROCm archive", download_cache_dir(paths)),
        ("helper tool archive", tool_download_cache_dir(paths)),
    ] {
        match std::fs::symlink_metadata(&dir) {
            Ok(metadata) if metadata.is_symlink() => {
                plan.skipped.push(format!(
                    "{} is a link to somewhere else, so its contents are left alone",
                    dir.display()
                ));
                continue;
            }
            Ok(_) => {}
            Err(_) => {
                plan.skipped
                    .push(format!("nothing downloaded in {}", dir.display()));
                continue;
            }
        }
        let mut found = false;
        let mut stack = vec![dir.clone()];
        while let Some(current) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&current) else {
                plan.skipped
                    .push(format!("could not read {}", current.display()));
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(metadata) = entry.path().symlink_metadata() else {
                    plan.skipped
                        .push(format!("could not read {}", path.display()));
                    continue;
                };
                if metadata.is_dir() && !metadata.is_symlink() {
                    stack.push(path);
                } else {
                    found = true;
                    plan.actions.push(UninstallPlanEntry { kind, path });
                }
            }
        }
        if !found {
            plan.skipped
                .push(format!("nothing downloaded in {}", dir.display()));
        }
    }
    plan.actions
        .sort_by(|left, right| left.path.cmp(&right.path));
    plan.skipped.sort();
    plan
}

pub(crate) fn render_downloads_plan(plan: &UninstallPlan, dry_run: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Downloaded file review");
    let _ = writeln!(output);
    if plan.actions.is_empty() {
        let _ = writeln!(output, "Nothing would be removed.");
    } else {
        let total: u64 = plan.actions.iter().fold(0, |total, entry| {
            total.saturating_add(measure_path(&entry.path).bytes)
        });
        let _ = writeln!(
            output,
            "{} downloaded file(s) would be removed, freeing about {}:",
            plan.actions.len(),
            format_bytes(total)
        );
        for entry in &plan.actions {
            let _ = writeln!(output, "  - {}: {}", entry.kind, entry.path.display());
        }
        let _ = writeln!(output);
        let _ = writeln!(output, "ROCm CLI downloads these again when it needs them.");
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
        let _ = writeln!(
            output,
            "Nothing was changed. Re-run without --dry-run to remove."
        );
    }
    output
}

// ---------------------------------------------------------------------------
// Command entry point
// ---------------------------------------------------------------------------

/// Confirmation gate shared by both mutating verbs. Outside a terminal there is
/// nobody to answer the prompt, so `--yes` is required (same rule as
/// `rocm uninstall`).
fn approved(action: &str, yes: bool) -> Result<bool> {
    if yes {
        return Ok(true);
    }
    if !interactive_terminal() {
        bail!("{action} requires --yes outside an interactive terminal");
    }
    confirm_uninstall()
}

pub(crate) fn storage(command: Option<StorageCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    // Deliberately not `unwrap_or_default()`. `load` already returns the
    // default for an absent file; an *unparseable* file returns an error, and
    // defaulting it away would silently drop the active, previous and default
    // runtime — every force-keep at once — on the one command here that
    // deletes. Failing loudly is the only safe reading of a corrupt config.
    let mut config = RocmCliConfig::load(&paths)?;

    match command.unwrap_or(StorageCommand::Report { json: false }) {
        StorageCommand::Report { json } => {
            let report = build_report(&paths, &config)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", render_report(&report));
            }
        }
        StorageCommand::RemoveOldInstalls { keep, dry_run, yes } => {
            let plan = build_prune_plan(&paths, &config, keep)?;
            print!("{}", render_prune_plan(&plan, keep, dry_run));
            if plan.remove.is_empty() || dry_run {
                return Ok(());
            }
            if !approved("removing old ROCm installs", yes)? {
                println!("nothing was removed");
                return Ok(());
            }
            for entry in &plan.remove {
                // The plan was built before the confirmation prompt blocked on
                // stdin, so a runtime could have been activated in another
                // terminal meanwhile. Re-check the holds against config as it
                // is now, immediately before deleting. (The ownership guard is
                // already re-checked inside `uninstall_runtime`.)
                let manifests = therock::load_runtime_manifests(&paths)?;
                let marker = read_active_runtime_marker(&paths);
                let inputs = RetentionInputs::from_config(&config, marker.as_ref());
                let default_key = resolved_default_runtime_key(&manifests, &inputs);
                if let Some(manifest) = manifests
                    .iter()
                    .find(|manifest| manifest.runtime_key == entry.runtime_key)
                    && let Some(reason) =
                        unconditional_hold(manifest, &inputs, default_key.as_deref())
                {
                    println!("left {} alone: {}", entry.runtime_key, reason.describe());
                    continue;
                }

                // Per-key delegation keeps config/marker cleanup in one place.
                let result = crate::uninstall_runtime(&paths, &mut config, &entry.runtime_key)
                    .with_context(|| format!("failed to remove {}", entry.runtime_key))?;
                // `uninstall_runtime` re-evaluates ownership itself and can
                // decline to delete the folder while still clearing the
                // registry entry. Reporting the space as reclaimed then would
                // be a lie the user cannot check: the tree stays, and with its
                // registry record gone it is invisible to `rocm storage report`
                // and un-prunable. Report what actually happened, as
                // `rocm runtimes uninstall` already does.
                if let Some(path) = result.removed_install_root.as_ref() {
                    println!(
                        "removed {} ({}) from {}",
                        result.runtime_key,
                        format_bytes(entry.size_bytes),
                        path.display()
                    );
                } else {
                    println!(
                        "removed the registry entry for {}, but left its folder {} in place",
                        result.runtime_key,
                        entry.install_root.display()
                    );
                    println!(
                        "  nothing was freed; remove that folder by hand if you want the space"
                    );
                }
                crate::record_cli_audit_event(
                    &paths,
                    "runtime",
                    "storage_remove_old_installs",
                    "info",
                    format!(
                        "removed runtime_key={} runtime_id={} folder_removed={}",
                        result.runtime_key,
                        result.runtime_id,
                        result.removed_install_root.is_some()
                    ),
                    None,
                );
            }
            println!("old ROCm installs removed");
        }
        StorageCommand::RemoveDownloads { dry_run, yes } => {
            let plan = build_downloads_plan(&paths);
            print!("{}", render_downloads_plan(&plan, dry_run));
            if plan.actions.is_empty() || dry_run {
                return Ok(());
            }
            if !approved("removing downloaded files", yes)? {
                println!("nothing was removed");
                return Ok(());
            }
            for entry in &plan.actions {
                remove_path(&entry.path)
                    .with_context(|| format!("failed to remove {}", entry.path.display()))?;
            }
            println!("{} downloaded file(s) removed", plan.actions.len());
            crate::record_cli_audit_event(
                &paths,
                "runtime",
                "storage_remove_downloads",
                "info",
                format!("removed {} downloaded file(s)", plan.actions.len()),
                None,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(
        runtime_key: &str,
        family: &str,
        version: &str,
        installed_at_unix_ms: u128,
    ) -> therock::InstalledRuntimeManifest {
        therock::InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: format!("therock-release:{family}"),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: family.to_owned(),
            family_source: "test".to_owned(),
            raw_arch: None,
            version: version.to_owned(),
            install_root: PathBuf::from("rocm-storage-test").join(runtime_key),
            selected_artifact_url: "https://example.invalid/therock".to_owned(),
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: None,
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            read_only: false,
            imported_from: None,
            installed_at_unix_ms,
        }
    }

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("storage")
            .join(format!(
                "rocm-cli-storage-test-{name}-{}-{}",
                std::process::id(),
                rocm_core::unix_time_millis()
            ));
        let _ = std::fs::remove_dir_all(&root);
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    /// Write a manifest pair (registry + in-tree copy) plus a payload file, so
    /// the runtime looks exactly like one ROCm CLI installed itself.
    fn install_fixture(
        paths: &AppPaths,
        mut record: therock::InstalledRuntimeManifest,
        payload_bytes: usize,
    ) -> Result<therock::InstalledRuntimeManifest> {
        let install_root = paths
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join(&record.runtime_key);
        record.install_root = install_root.clone();
        std::fs::create_dir_all(install_root.join("lib"))?;
        std::fs::write(
            install_root.join("lib").join("payload.bin"),
            vec![0_u8; payload_bytes],
        )?;
        let registry_dir = paths.data_dir.join("runtimes").join("registry");
        std::fs::create_dir_all(&registry_dir)?;
        let encoded = serde_json::to_vec_pretty(&record)?;
        std::fs::write(
            registry_dir.join(format!("{}.json", record.runtime_key)),
            &encoded,
        )?;
        std::fs::write(install_root.join(".rocm-cli-runtime.json"), &encoded)?;
        Ok(record)
    }

    #[test]
    fn keeps_the_most_recent_installs_per_family_and_removes_the_rest() {
        let manifests = vec![
            manifest("release-wheel-gfx110x-1", "gfx110X-all", "7.10.0", 10),
            manifest("release-wheel-gfx110x-2", "gfx110X-all", "7.11.0", 20),
            manifest("release-wheel-gfx110x-3", "gfx110X-all", "7.12.0", 30),
            // A second GPU family keeps its own N rather than competing with
            // the first: a multi-GPU machine needs one install per family.
            manifest("release-wheel-gfx120x-1", "gfx120X-all", "7.12.0", 5),
        ];

        let (removable, held) =
            select_runtimes_to_remove(&manifests, &RetentionInputs::default(), DEFAULT_KEEP);

        assert_eq!(removable, vec!["release-wheel-gfx110x-1".to_owned()]);
        let kept: Vec<&str> = held.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            kept,
            vec![
                "release-wheel-gfx110x-2",
                "release-wheel-gfx110x-3",
                "release-wheel-gfx120x-1",
            ]
        );
        assert!(
            held.iter()
                .all(|(_, reason)| *reason == HoldReason::WithinKeepLimit)
        );
    }

    #[test]
    fn never_selects_the_active_previous_default_or_marked_install() {
        let manifests = vec![
            manifest("active", "gfx110X-all", "7.10.0", 1),
            manifest("previous", "gfx110X-all", "7.10.0", 2),
            manifest("marked", "gfx110X-all", "7.10.0", 3),
            manifest("by-default-id", "gfx120X-all", "7.10.0", 4),
            manifest("plain-old", "gfx110X-all", "7.10.0", 5),
            manifest("plain-new", "gfx110X-all", "7.11.0", 6),
        ];
        let inputs = RetentionInputs {
            // Deliberately a different case: keys compare case-insensitively
            // everywhere else in the runtime code.
            active_runtime_key: Some("ACTIVE".to_owned()),
            previous_runtime_key: Some("previous".to_owned()),
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            marker_runtime_keys: vec!["marked".to_owned()],
        };

        // keep = 0 so nothing survives on recency alone; only hard holds do.
        let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, 0);

        assert_eq!(
            removable,
            vec!["plain-new".to_owned(), "plain-old".to_owned()]
        );
        let reason = |key: &str| {
            held.iter()
                .find(|(held_key, _)| held_key == key)
                .map(|(_, reason)| reason.clone())
        };
        assert_eq!(reason("active"), Some(HoldReason::Active));
        assert_eq!(reason("previous"), Some(HoldReason::Previous));
        assert_eq!(reason("by-default-id"), Some(HoldReason::Default));
        assert_eq!(reason("marked"), Some(HoldReason::Marker));
    }

    /// Regression test for the state every real machine is actually in.
    ///
    /// `runtime_id` is `therock-<channel>:<family>` with no version in it, and
    /// `activate_runtime` sets `default_runtime_id` on every install and every
    /// update — so a machine that has accumulated installs has one id shared by
    /// the whole family. Holding every id match held all of them, which made
    /// `--keep` (including `--keep 0`) do nothing at all. The earlier suite
    /// missed this because its `default_runtime_id` named a *different* family
    /// from the manifests under test, so the Default hold never fired.
    #[test]
    fn keep_still_applies_when_the_default_id_covers_a_whole_family() {
        // Six accumulated installs of one family, as `rocm update --apply`
        // leaves them.
        let manifests = vec![
            manifest("release-wheel-gfx110x-7-10-0", "gfx110X-all", "7.10.0", 10),
            manifest("release-wheel-gfx110x-7-11-0", "gfx110X-all", "7.11.0", 20),
            manifest("release-wheel-gfx110x-7-12-0", "gfx110X-all", "7.12.0", 30),
            manifest("release-wheel-gfx110x-7-13-0", "gfx110X-all", "7.13.0", 40),
            manifest("release-wheel-gfx110x-7-14-0", "gfx110X-all", "7.14.0", 50),
            manifest("release-wheel-gfx110x-7-15-0", "gfx110X-all", "7.15.0", 60),
        ];
        // Written exactly as `activate_runtime` writes it: the active key and
        // the default id both point into the same family.
        let inputs = RetentionInputs {
            active_runtime_key: Some("release-wheel-gfx110x-7-15-0".to_owned()),
            previous_runtime_key: Some("release-wheel-gfx110x-7-14-0".to_owned()),
            default_runtime_id: Some("therock-release:gfx110X-all".to_owned()),
            marker_runtime_keys: Vec::new(),
        };

        // The default id matches all four, so it identifies none of them and
        // must not hold anything; Active and Previous still hold the two the
        // user actually depends on.
        assert_eq!(resolved_default_runtime_key(&manifests, &inputs), None);

        let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, DEFAULT_KEEP);
        assert_eq!(
            removable,
            vec![
                "release-wheel-gfx110x-7-10-0".to_owned(),
                "release-wheel-gfx110x-7-11-0".to_owned(),
            ]
        );
        let reason = |key: &str| {
            held.iter()
                .find(|(held_key, _)| held_key == key)
                .map(|(_, reason)| reason.clone())
        };
        assert_eq!(
            reason("release-wheel-gfx110x-7-15-0"),
            Some(HoldReason::Active)
        );
        assert_eq!(
            reason("release-wheel-gfx110x-7-14-0"),
            Some(HoldReason::Previous)
        );
        assert!(
            held.iter()
                .all(|(_, reason)| *reason != HoldReason::Default),
            "an ambiguous default must not hold anything: {held:?}"
        );

        // `--keep 0` must now be a real button rather than a no-op, while the
        // in-use and rollback installs stay held regardless.
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, 0);
        assert_eq!(
            removable,
            vec![
                "release-wheel-gfx110x-7-10-0".to_owned(),
                "release-wheel-gfx110x-7-11-0".to_owned(),
                "release-wheel-gfx110x-7-12-0".to_owned(),
                "release-wheel-gfx110x-7-13-0".to_owned(),
            ]
        );
    }

    /// The other half of the same rule: when the default id names exactly one
    /// install it is a real, specific reference and still holds it.
    #[test]
    fn an_unambiguous_default_still_holds_its_install() {
        let manifests = vec![
            manifest("only-gfx120x", "gfx120X-all", "7.12.0", 10),
            manifest("other-family", "gfx110X-all", "7.12.0", 20),
        ];
        let inputs = RetentionInputs {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RetentionInputs::default()
        };

        assert_eq!(
            resolved_default_runtime_key(&manifests, &inputs).as_deref(),
            Some("only-gfx120x")
        );
        let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, 0);
        assert_eq!(removable, vec!["other-family".to_owned()]);
        assert_eq!(
            held.iter()
                .find(|(key, _)| key == "only-gfx120x")
                .map(|(_, reason)| reason.clone()),
            Some(HoldReason::Default)
        );
    }

    /// Every force-keep compares case-insensitively, not just the active one.
    /// Replacing any of these `eq_ignore_ascii_case` calls with `==` must fail
    /// a test.
    #[test]
    fn all_force_keeps_compare_case_insensitively() {
        let manifests = vec![
            manifest("active", "gfx110X-all", "7.10.0", 1),
            manifest("previous", "gfx110X-all", "7.10.0", 2),
            manifest("marked", "gfx110X-all", "7.10.0", 3),
            manifest("by-default-id", "gfx120X-all", "7.10.0", 4),
        ];
        let inputs = RetentionInputs {
            active_runtime_key: Some("ACTIVE".to_owned()),
            previous_runtime_key: Some("PreVioUs".to_owned()),
            default_runtime_id: Some("THEROCK-RELEASE:GFX120X-ALL".to_owned()),
            marker_runtime_keys: vec!["MARKED".to_owned()],
        };

        let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, 0);

        assert!(
            removable.is_empty(),
            "case differences must not defeat a force-keep: {removable:?}"
        );
        let reason = |key: &str| {
            held.iter()
                .find(|(held_key, _)| held_key == key)
                .map(|(_, reason)| reason.clone())
        };
        assert_eq!(reason("active"), Some(HoldReason::Active));
        assert_eq!(reason("previous"), Some(HoldReason::Previous));
        assert_eq!(reason("by-default-id"), Some(HoldReason::Default));
        assert_eq!(reason("marked"), Some(HoldReason::Marker));
    }

    #[test]
    fn never_selects_adopted_imported_or_read_only_installs() {
        let mut adopted = manifest("adopted", "gfx110X-all", "7.10.0", 1);
        adopted.read_only = true;
        let mut imported = manifest("imported", "gfx110X-all", "7.10.0", 2);
        imported.imported_from = Some(PathBuf::from("elsewhere/manifest.json"));
        let manifests = vec![adopted, imported];

        let (removable, held) =
            select_runtimes_to_remove(&manifests, &RetentionInputs::default(), 0);

        assert!(removable.is_empty());
        assert!(
            held.iter()
                .all(|(_, reason)| *reason == HoldReason::NotOwned)
        );
    }

    #[test]
    fn prune_plan_skips_installs_without_a_matching_in_tree_manifest() -> Result<()> {
        let (root, paths) = test_paths("prune-plan");
        install_fixture(&paths, manifest("old", "gfx110X-all", "7.10.0", 10), 2048)?;
        install_fixture(&paths, manifest("newer", "gfx110X-all", "7.11.0", 20), 1024)?;
        // Strip the in-tree copy from a third install: ROCm CLI can no longer
        // prove it owns that folder, so removal must be refused.
        let unowned = install_fixture(&paths, manifest("unowned", "gfx110X-all", "7.9.0", 5), 512)?;
        std::fs::remove_file(unowned.install_root.join(".rocm-cli-runtime.json"))?;

        let plan = build_prune_plan(&paths, &RocmCliConfig::default(), 1)?;

        let removed: Vec<&str> = plan
            .remove
            .iter()
            .map(|entry| entry.runtime_key.as_str())
            .collect();
        assert_eq!(removed, vec!["old"]);
        assert!(plan.remove[0].size_bytes >= 2048);
        assert!(
            plan.skipped
                .iter()
                .any(|line| line.starts_with("unowned: ROCm CLI did not create this folder")),
            "unowned install must be reported as left alone: {:?}",
            plan.skipped
        );
        assert!(
            plan.skipped.iter().any(|line| line.starts_with("newer:")),
            "kept install must be reported: {:?}",
            plan.skipped
        );

        let rendered = render_prune_plan(&plan, 1, true);
        assert!(rendered.contains("1 install(s) would be removed"));
        assert!(rendered.contains("Left alone:"));
        assert!(rendered.contains("Nothing was changed."));

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn report_lists_install_sizes_and_labels_shared_caches() -> Result<()> {
        let (root, paths) = test_paths("report");
        install_fixture(
            &paths,
            manifest("in-use", "gfx110X-all", "7.12.0", 20),
            4096,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some("in-use".to_owned()),
            ..RocmCliConfig::default()
        };

        let report = build_report(&paths, &config)?;
        assert_eq!(report.runtimes.len(), 1);
        assert!(report.runtimes[0].active);
        assert_eq!(report.runtimes[0].hold_reason, Some(HoldReason::Active));
        assert!(report.runtimes_total_bytes >= 4096);

        let rendered = render_report(&report);
        assert!(rendered.contains("in-use version=7.12.0"));
        assert!(rendered.contains("status: in use"));
        assert!(rendered.contains("Shared with other tools (never removed by ROCm CLI):"));
        assert!(rendered.contains("downloaded models"));

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn report_survives_a_missing_install_folder() -> Result<()> {
        let (root, paths) = test_paths("report-missing");
        let gone = install_fixture(&paths, manifest("gone", "gfx110X-all", "7.12.0", 20), 16)?;
        std::fs::remove_dir_all(&gone.install_root)?;

        let rendered = render_report(&build_report(&paths, &RocmCliConfig::default())?);
        assert!(rendered.contains("unknown size (folder missing or unreadable)"));

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn measuring_a_missing_path_reports_an_incomplete_zero() {
        let measurement = measure_path(Path::new("definitely-not-here-rocm-storage"));
        assert_eq!(
            measurement,
            Measurement {
                bytes: 0,
                complete: false
            }
        );
    }

    #[test]
    fn downloads_plan_collects_cached_archives_and_never_model_files() -> Result<()> {
        let (root, paths) = test_paths("downloads");
        let archives = paths.cache_dir.join("therock");
        std::fs::create_dir_all(archives.join("metadata"))?;
        std::fs::write(archives.join("sdk.tar.gz"), b"tarball")?;
        std::fs::write(archives.join("metadata").join("index.body"), b"index")?;
        std::fs::create_dir_all(paths.data_dir.join("models"))?;
        std::fs::write(
            paths.data_dir.join("models").join("weights.bin"),
            b"weights",
        )?;

        let plan = build_downloads_plan(&paths);

        let names: Vec<String> = plan
            .actions
            .iter()
            .map(|entry| entry.path.display().to_string())
            .collect();
        assert_eq!(plan.actions.len(), 2, "{names:?}");
        assert!(names.iter().all(|name| !name.contains("models")));
        let rendered = render_downloads_plan(&plan, true);
        assert!(rendered.contains("2 downloaded file(s) would be removed"));

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    /// Relocating the cache to a bigger disk with a symlink is a normal thing
    /// for exactly the user running a disk-space command. Following the link
    /// collected real files outside the cache and deleted them, and the
    /// dry-run showed them as ordinary cache paths so nothing looked wrong.
    #[cfg(unix)]
    #[test]
    fn downloads_plan_refuses_to_reach_through_a_symlinked_cache_root() -> Result<()> {
        let (root, paths) = test_paths("downloads-symlink-root");
        let outside = root.join("somewhere-else");
        std::fs::create_dir_all(&outside)?;
        std::fs::write(outside.join("not-a-download.bin"), b"precious")?;
        std::fs::create_dir_all(&paths.cache_dir)?;
        std::os::unix::fs::symlink(&outside, paths.cache_dir.join("therock"))?;

        let plan = build_downloads_plan(&paths);

        assert!(
            plan.actions.is_empty(),
            "nothing behind the link may be collected: {:?}",
            plan.actions
        );
        assert!(
            plan.skipped
                .iter()
                .any(|line| line.contains("is a link to somewhere else")),
            "the user must be told why it was skipped: {:?}",
            plan.skipped
        );
        // The plan is what the removal loop iterates, so an empty plan is what
        // keeps the file on disk.
        assert!(outside.join("not-a-download.bin").is_file());

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    /// A symlinked cache used to measure as the length of the link itself —
    /// about 17 bytes, reported as a complete measurement — which is exactly
    /// backwards for the caches this feature exists to surface.
    #[cfg(unix)]
    #[test]
    fn measuring_a_symlinked_directory_reports_the_target_size() -> Result<()> {
        let (root, paths) = test_paths("measure-symlink-root");
        let target = root.join("real-cache");
        std::fs::create_dir_all(&target)?;
        std::fs::write(target.join("payload.bin"), vec![0_u8; 4096])?;
        std::fs::create_dir_all(&paths.cache_dir)?;
        let link = paths.cache_dir.join("linked");
        std::os::unix::fs::symlink(&target, &link)?;

        let measurement = measure_path(&link);

        assert!(
            measurement.bytes >= 4096,
            "a symlinked cache must measure its target, got {measurement:?}"
        );
        assert!(measurement.complete);

        // Symlinks *below* the root are still counted as themselves and never
        // followed, so a linked-in model folder is not attributed to the
        // runtime pointing at it.
        let runtime_root = root.join("runtime");
        std::fs::create_dir_all(&runtime_root)?;
        std::os::unix::fs::symlink(&target, runtime_root.join("models"))?;
        assert!(measure_path(&runtime_root).bytes < 4096);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    /// The whole removal path end to end: plan, ownership guard, deletion,
    /// and the config/registry cleanup `uninstall_runtime` owns.
    #[test]
    fn removing_old_installs_deletes_only_the_planned_folders() -> Result<()> {
        let (root, paths) = test_paths("remove-executes");
        let old = install_fixture(&paths, manifest("old", "gfx110X-all", "7.10.0", 10), 2048)?;
        let kept = install_fixture(&paths, manifest("kept", "gfx110X-all", "7.11.0", 20), 2048)?;
        let active = install_fixture(
            &paths,
            manifest("active", "gfx110X-all", "7.12.0", 30),
            2048,
        )?;
        let mut config = RocmCliConfig {
            active_runtime_key: Some("active".to_owned()),
            // The real-machine state: a family-wide default id alongside the
            // active key.
            default_runtime_id: Some("therock-release:gfx110X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let plan = build_prune_plan(&paths, &config, 1)?;
        let planned: Vec<&str> = plan
            .remove
            .iter()
            .map(|entry| entry.runtime_key.as_str())
            .collect();
        assert_eq!(planned, vec!["old"]);

        for entry in &plan.remove {
            crate::uninstall_runtime(&paths, &mut config, &entry.runtime_key)?;
        }

        assert!(!old.install_root.exists(), "planned folder must be gone");
        assert!(kept.install_root.is_dir(), "kept folder must survive");
        assert!(active.install_root.is_dir(), "active folder must survive");
        assert!(
            !paths
                .data_dir
                .join("runtimes")
                .join("registry")
                .join("old.json")
                .exists()
        );

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    /// `--yes` is required when there is nobody to answer the prompt. This is
    /// the gate standing between a scripted invocation and a multi-gigabyte
    /// deletion, so it is worth a test of its own.
    #[test]
    fn removal_requires_yes_when_there_is_no_terminal_to_prompt() {
        assert!(approved("removing old ROCm installs", true).is_ok_and(|approved| approved));
        if !interactive_terminal() {
            let error = approved("removing old ROCm installs", false)
                .expect_err("must refuse without --yes outside a terminal");
            assert!(error.to_string().contains("requires --yes"));
        }
    }
}
