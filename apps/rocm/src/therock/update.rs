// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether an installed runtime has something newer to offer, and the bounded
//! startup check that asks. `runtime_update_plan` resolves the latest source for
//! one runtime (`resolve_latest_for_manifest`, which re-resolves a wheel
//! runtime through `resolve_pip_runtime_with_timeout` and a tarball one through
//! `resolve_tarball_artifact_with_timeout`, on the layout the runtime was
//! installed from) and `runtime_freshness` classifies it (`RuntimeFreshness`:
//! up to date, update available, repair available, or ahead of the index) into
//! a `RuntimeUpdatePlan`. `render_update_report` and `render_update_json` show
//! the plans for `rocm update` (the JSON form is what the dashboard's update
//! check parses), and `maybe_refresh_startup_update_check` keeps one
//! `StartupUpdateCheckRecord` in the cache so the check runs at most once an
//! interval and is bounded by a short download timeout.
//!
//! Builds on `family`, `io`, `process`, `python`, `resolve`, `source` and
//! `version`, and on `runtime_registry` for the manifests; `runtime_key`,
//! `wheel_runtime_key` and `has_nontrivial_directory_contents`, which name and
//! inspect the install a plan targets, stay in `therock.rs` with the other keys
//! and roots helpers.

use super::family::family_override_or_recovered_arch;
use super::io::write_file_atomically;
use super::process::SuppressProgressOutput;
use super::python::{
    PythonLauncher, python_launcher_install_ready, python_requirement, resolve_python_launcher,
    wheel_compatibility_for_python,
};
use super::resolve::{
    AggregateDeviceTarget, resolve_pip_runtime_with_timeout, resolve_tarball_artifact_with_timeout,
    wheel_runtime_composition,
};
use super::source::{SourceLayout, TheRockChannel};
use super::version::{runtime_version_display, version_relation};
use super::{has_nontrivial_directory_contents, runtime_key, wheel_runtime_key};
use crate::runtime_registry::{
    InstalledRuntimeManifest, WheelRuntimeComposition, load_runtime_manifests,
    wheel_composition_device_target,
};
use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, unix_time_millis};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const STARTUP_UPDATE_CHECK_INTERVAL_MS: u128 = 12 * 60 * 60 * 1_000;
const STARTUP_UPDATE_CHECK_TIMEOUT_SECS: u64 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StartupUpdateCheckRecord {
    pub runtime_key: String,
    pub runtime_id: String,
    pub channel: String,
    pub format: String,
    pub family: String,
    pub installed_version: String,
    #[serde(default)]
    pub latest_version: Option<String>,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
    pub checked_at_unix_ms: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeUpdatePlan {
    pub latest_version: String,
    pub latest_source: String,
    pub format: String,
    pub status: String,
    /// The runtime key the applied install will produce. Update apply selects
    /// the resulting manifest by this key rather than by version, because a
    /// same-version repair produces a sibling that version alone cannot name.
    pub target_runtime_key: String,
    /// Exact device payload encoded in the planned wheel composition.
    pub device_target: Option<String>,
    /// The layout this runtime was installed from, carried into the apply so a
    /// runtime never silently migrates between streams when it updates.
    pub source_layout_generation: Option<String>,
    pub repair_required: bool,
    pub update_available: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum RuntimeFreshness {
    UpToDate,
    UpdateAvailable,
    RepairAvailable,
    AheadOfIndex,
}

impl RuntimeFreshness {
    const fn status(self) -> &'static str {
        match self {
            Self::UpToDate => "up_to_date",
            Self::UpdateAvailable => "update_available",
            Self::RepairAvailable => "repair_available",
            Self::AheadOfIndex => "ahead_of_index",
        }
    }

    const fn update_available(self) -> bool {
        matches!(self, Self::UpdateAvailable | Self::RepairAvailable)
    }
}

#[derive(Debug, Clone)]
struct ResolvedRuntimeUpdate {
    latest_version: String,
    latest_source: String,
    target_runtime_key: String,
    format: String,
    source_layout_generation: String,
    wheel_composition: Option<WheelRuntimeComposition>,
}

pub(crate) fn render_update_report(paths: &AppPaths) -> Result<String> {
    let manifests = load_runtime_manifests(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "update");
    let _ = writeln!(
        output,
        "  policy: bounded startup check, cached metadata, prompt before mutating state."
    );
    if let Some(record) = load_startup_update_check(paths)? {
        let _ = writeln!(
            output,
            "  startup_check: runtime={} status={} checked_at_unix_ms={}",
            record.runtime_key, record.status, record.checked_at_unix_ms
        );
    }

    if manifests.is_empty() {
        let _ = writeln!(output, "  managed runtimes: none");
        let _ = writeln!(
            output,
            "  next step: run `rocm install sdk --channel release --dry-run` to resolve a TheRock runtime"
        );
        return Ok(output);
    }

    for manifest in &manifests {
        let plan = match runtime_update_plan(paths, manifest, &manifests, None) {
            Ok(plan) => Some(plan),
            Err(error) => {
                let _ = writeln!(
                    output,
                    "  runtime {} format={} status=error message={}",
                    manifest.runtime_key, manifest.format, error
                );
                None
            }
        };

        let Some(plan) = plan else {
            continue;
        };
        // `target=` names the runtime key an apply from this line would produce.
        // For a superseded legacy manifest that is its already-installed
        // replacement, which is how a reader — `xtask e2e-prewarm` above all —
        // learns which sibling to activate without re-deriving the composition.
        let _ = writeln!(
            output,
            "  runtime {} format={} channel={} family={} installed={} latest={} status={} target={}",
            manifest.runtime_key,
            plan.format,
            manifest.channel,
            manifest.family,
            runtime_version_display(&manifest.version),
            runtime_version_display(&plan.latest_version),
            plan.status,
            plan.target_runtime_key
        );
        let _ = writeln!(
            output,
            "    install_root: {}",
            manifest.install_root.display()
        );
        let _ = writeln!(output, "    source: {}", plan.latest_source);
        if plan.update_available {
            let next_step = if plan.repair_required {
                format!(
                    "run `rocm update --apply --runtime {}` to install a composition-keyed replacement side-by-side",
                    manifest.runtime_key
                )
            } else {
                format!(
                    "run `rocm update --apply --runtime {}` to install the newer runtime side-by-side",
                    manifest.runtime_key
                )
            };
            let _ = writeln!(output, "    next step: {next_step}");
            let _ = writeln!(
                output,
                "    activate: add `--activate` to make the installed runtime the default after install"
            );
        }
    }

    Ok(output)
}

/// Structured counterpart to [`render_update_report`], for `rocm update --json`.
/// Consumed by the dash TUI's background update-check job (parsed off a
/// single compact JSON line captured from the job's stdout), so field names
/// are a stable-ish contract — extend, don't rename, without checking callers.
///
/// Three outcomes a consumer must handle: `runtimes: []` (nothing managed),
/// one row per manifest with `status: "error"` for any that failed to
/// resolve (this struct still returns `Ok`), or no JSON at all with a
/// non-zero exit — [`load_runtime_manifests`] failing is not caught per-row
/// and fails the whole call.
#[derive(Debug, Serialize)]
pub(crate) struct UpdateJson {
    pub runtimes: Vec<UpdateJsonRuntime>,
}

/// `format`, `install_root`, and `source` are deliberately omitted: this
/// contract only needs to answer "is an update available," and every consumer
/// so far (the dash TUI's Updates tile) only reads `status`/`latest_version`.
/// Add a field when a real consumer needs it, not preemptively.
#[derive(Debug, Serialize)]
pub(crate) struct UpdateJsonRuntime {
    pub runtime_key: String,
    pub channel: String,
    pub family: String,
    pub installed_version: String,
    pub latest_version: Option<String>,
    /// `"update_available"` | `"repair_available"` | `"up_to_date"` | `"ahead_of_index"` | `"error"`.
    pub status: String,
    pub message: Option<String>,
}

pub(crate) fn render_update_json(
    paths: &AppPaths,
    download_timeout_secs: Option<u64>,
) -> Result<UpdateJson> {
    // Resolving a wheel-format manifest's latest version can fall through to
    // Python resolution/bootstrap, which otherwise prints progress lines (and
    // an installer's raw stdout) ahead of the JSON below, breaking the
    // documented single-line contract on `UpdateJson`.
    let _quiet = SuppressProgressOutput::new();
    let manifests = load_runtime_manifests(paths)?;
    let mut runtimes = Vec::with_capacity(manifests.len());
    for manifest in &manifests {
        match runtime_update_plan(paths, manifest, &manifests, download_timeout_secs) {
            Ok(plan) => runtimes.push(UpdateJsonRuntime {
                runtime_key: manifest.runtime_key.clone(),
                channel: manifest.channel.clone(),
                family: manifest.family.clone(),
                installed_version: manifest.version.clone(),
                latest_version: Some(plan.latest_version),
                status: plan.status,
                message: None,
            }),
            Err(error) => runtimes.push(UpdateJsonRuntime {
                runtime_key: manifest.runtime_key.clone(),
                channel: manifest.channel.clone(),
                family: manifest.family.clone(),
                installed_version: manifest.version.clone(),
                latest_version: None,
                status: "error".to_owned(),
                message: Some(error.to_string()),
            }),
        }
    }
    Ok(UpdateJson { runtimes })
}

/// Whether the composition-keyed replacement for `source` is already installed.
///
/// A repair installs a sibling and leaves the legacy manifest in place until the
/// retention pass removes it. Without this, that retained manifest keeps
/// reporting `repair_available` forever, so every pre-warm reinstalls a runtime
/// that is already there and every startup check re-notifies.
fn replacement_runtime_is_installed(
    manifests: &[InstalledRuntimeManifest],
    source: &InstalledRuntimeManifest,
    target_runtime_key: &str,
    required_composition: Option<&WheelRuntimeComposition>,
) -> bool {
    // `RepairAvailable` currently implies a wheel composition, but keep this
    // helper total if a future freshness state reaches it without one.
    let Some(required_composition) = required_composition else {
        return false;
    };
    manifests.iter().any(|candidate| {
        !candidate.read_only
            && candidate.runtime_key == target_runtime_key
            && candidate.channel == source.channel
            && candidate.format == source.format
            && candidate.family == source.family
            && candidate.wheel_composition.as_ref() == Some(required_composition)
            && has_nontrivial_directory_contents(&candidate.install_root).unwrap_or(false)
    })
}

/// Freshness of one runtime against the index, ignoring its siblings.
///
/// An installed version newer than the index is [`RuntimeFreshness::AheadOfIndex`]
/// before any composition is considered: that build cannot be reproduced from the
/// index at all, so calling it repairable would promise an install that must
/// either fail or silently roll the runtime back.
pub(super) fn runtime_freshness(
    manifest: &InstalledRuntimeManifest,
    latest_version: &str,
    required_composition: Option<&WheelRuntimeComposition>,
    target_runtime_key: &str,
) -> RuntimeFreshness {
    let Some(relation) = version_relation(&manifest.version, latest_version) else {
        return RuntimeFreshness::AheadOfIndex;
    };
    match relation {
        Ordering::Less => RuntimeFreshness::UpdateAvailable,
        Ordering::Greater => RuntimeFreshness::AheadOfIndex,
        Ordering::Equal
            if !manifest.read_only
                && required_composition.is_some()
                && (manifest.wheel_composition.as_ref() != required_composition
                    || manifest.runtime_key != target_runtime_key) =>
        {
            RuntimeFreshness::RepairAvailable
        }
        Ordering::Equal => RuntimeFreshness::UpToDate,
    }
}

fn runtime_freshness_with_manifests(
    manifests: &[InstalledRuntimeManifest],
    manifest: &InstalledRuntimeManifest,
    latest_version: &str,
    required_composition: Option<&WheelRuntimeComposition>,
    target_runtime_key: &str,
) -> RuntimeFreshness {
    let freshness = runtime_freshness(
        manifest,
        latest_version,
        required_composition,
        target_runtime_key,
    );
    if freshness == RuntimeFreshness::RepairAvailable
        && replacement_runtime_is_installed(
            manifests,
            manifest,
            target_runtime_key,
            required_composition,
        )
    {
        RuntimeFreshness::UpToDate
    } else {
        freshness
    }
}

pub(crate) fn runtime_update_plan(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    manifests: &[InstalledRuntimeManifest],
    download_timeout_secs: Option<u64>,
) -> Result<RuntimeUpdatePlan> {
    let latest = resolve_latest_for_manifest(paths, manifest, download_timeout_secs)?;
    let freshness = runtime_freshness_with_manifests(
        manifests,
        manifest,
        &latest.latest_version,
        latest.wheel_composition.as_ref(),
        &latest.target_runtime_key,
    );
    let device_target = latest
        .wheel_composition
        .as_ref()
        .and_then(|composition| wheel_composition_device_target(Some(composition)))
        .map(str::to_owned);
    Ok(RuntimeUpdatePlan {
        latest_version: latest.latest_version,
        latest_source: latest.latest_source,
        format: latest.format,
        status: freshness.status().to_owned(),
        target_runtime_key: latest.target_runtime_key,
        device_target,
        source_layout_generation: Some(latest.source_layout_generation),
        repair_required: freshness == RuntimeFreshness::RepairAvailable,
        update_available: freshness.update_available(),
    })
}

/// The layout a manifest's artifact came from.
///
/// Manifests written before the field existed fall back to the generation their
/// wheel composition recorded, and manifests older than compositions fall back
/// to the canonical stream, which is where they came from. No URL is inspected:
/// a base can be overridden or rehosted, and the recorded generation is the only
/// statement about the stream that survives that.
fn manifest_source_layout(manifest: &InstalledRuntimeManifest) -> Result<SourceLayout> {
    SourceLayout::from_generation(manifest.source_layout_generation.as_deref().or_else(|| {
        manifest
            .wheel_composition
            .as_ref()
            .map(|composition| composition.source_layout_generation.as_str())
    }))
}

/// The family override to re-resolve a wheel manifest's update *plan* with.
/// See [`family_override_or_recovered_arch`] — the composition recorded at
/// install time is this call's source for the exact arch.
fn manifest_wheel_family_override(manifest: &InstalledRuntimeManifest) -> String {
    let recorded_arch = wheel_composition_device_target(manifest.wheel_composition.as_ref());
    family_override_or_recovered_arch(&manifest.family, recorded_arch)
}

fn resolve_latest_for_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    download_timeout_secs: Option<u64>,
) -> Result<ResolvedRuntimeUpdate> {
    let channel = TheRockChannel::parse(&manifest.channel)?;
    let layout = manifest_source_layout(manifest)?;
    match manifest.format.as_str() {
        "wheel" => {
            // A runtime updated across a layout change needs the new layout's
            // interpreter, so the recorded one is re-checked rather than trusted
            // on `is_file()` alone; otherwise a canonical cp312 venv would carry
            // into a next-layout update and resolve nothing.
            let requirement = python_requirement(layout);
            let manifest_python = manifest
                .python_executable
                .as_deref()
                .map(PathBuf::from)
                .filter(|path| path.is_file())
                .filter(|path| python_launcher_install_ready(path, requirement).is_ok())
                .map(|executable| PythonLauncher {
                    executable,
                    source: "manifest",
                });
            let python_executable = match manifest_python {
                Some(python) => python,
                None => resolve_python_launcher(paths, requirement)?,
            };
            let wheel_compatibility =
                wheel_compatibility_for_python(&python_executable.executable)?;
            let family_override = manifest_wheel_family_override(manifest);
            let resolution = resolve_pip_runtime_with_timeout(
                paths,
                channel,
                Some(family_override.as_str()),
                &wheel_compatibility,
                None,
                Some(layout),
                // An update re-resolves for the install it will reinstall, so it
                // must resolve against the extras that runtime already has.
                manifest.includes_devel(),
                download_timeout_secs,
            )?;
            // Prefer the device payload this runtime was actually built with over
            // a fresh host probe. Planning must predict the key an apply will
            // produce, and re-probing would disagree with the installed runtime on
            // any host whose GPU is absent, hidden, or simply a second card.
            let device_target =
                wheel_composition_device_target(manifest.wheel_composition.as_ref()).map_or_else(
                    || resolution.device_target.clone(),
                    |target| {
                        AggregateDeviceTarget::resolve(
                            Some(target),
                            &resolution.family,
                            &resolution.published_device_targets,
                        )
                    },
                );
            // No exact target means no reproducible composition, so freshness
            // falls back to the version comparison rather than demanding a repair
            // this host could not perform.
            let wheel_composition = match &device_target {
                AggregateDeviceTarget::Exact(_) => Some(wheel_runtime_composition(
                    &resolution,
                    &device_target,
                    manifest.includes_devel(),
                )),
                AggregateDeviceTarget::Undetermined(_) => None,
            };
            let target_runtime_key = wheel_composition.as_ref().map_or_else(
                || manifest.runtime_key.clone(),
                |composition| wheel_runtime_key(channel, &resolution.latest_version, composition),
            );
            Ok(ResolvedRuntimeUpdate {
                latest_version: resolution.latest_version,
                latest_source: resolution.index_url,
                target_runtime_key,
                format: "wheel".to_owned(),
                source_layout_generation: resolution.layout.generation().to_owned(),
                wheel_composition,
            })
        }
        "tarball" => {
            let artifact = resolve_tarball_artifact_with_timeout(
                paths,
                channel,
                Some(manifest.family.as_str()),
                None,
                Some(layout),
                download_timeout_secs,
            )?;
            let target_runtime_key = runtime_key(
                channel,
                "tarball",
                &artifact.family,
                Some(&artifact.version),
            );
            Ok(ResolvedRuntimeUpdate {
                latest_version: artifact.version,
                latest_source: artifact.url,
                target_runtime_key,
                format: "tarball".to_owned(),
                source_layout_generation: artifact.layout.generation().to_owned(),
                wheel_composition: None,
            })
        }
        other => bail!("unknown manifest format `{other}`"),
    }
}

pub(crate) fn maybe_refresh_startup_update_check(
    paths: &AppPaths,
    active_runtime_key: Option<&str>,
) -> Result<Option<StartupUpdateCheckRecord>> {
    maybe_refresh_startup_update_check_at(paths, active_runtime_key, unix_time_millis())
}

fn maybe_refresh_startup_update_check_at(
    paths: &AppPaths,
    active_runtime_key: Option<&str>,
    now_unix_ms: u128,
) -> Result<Option<StartupUpdateCheckRecord>> {
    if startup_update_check_disabled() {
        return Ok(None);
    }

    let manifests = load_runtime_manifests(paths)?;
    let Some(manifest) = select_startup_update_manifest(&manifests, active_runtime_key) else {
        return Ok(None);
    };

    if let Some(previous) = load_startup_update_check(paths)?
        && previous.runtime_key == manifest.runtime_key
        && !startup_update_check_due(previous.checked_at_unix_ms, now_unix_ms)
    {
        return Ok(Some(previous));
    }

    let record = build_startup_update_check_record(
        paths,
        manifest,
        &manifests,
        now_unix_ms,
        Some(STARTUP_UPDATE_CHECK_TIMEOUT_SECS),
    );
    save_startup_update_check(paths, &record)?;
    Ok(Some(record))
}

fn startup_update_check_disabled() -> bool {
    std::env::var_os("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK").is_some()
}

const fn startup_update_check_due(previous_unix_ms: u128, now_unix_ms: u128) -> bool {
    now_unix_ms.saturating_sub(previous_unix_ms) >= STARTUP_UPDATE_CHECK_INTERVAL_MS
}

pub(super) fn select_startup_update_manifest<'a>(
    manifests: &'a [InstalledRuntimeManifest],
    active_runtime_key: Option<&str>,
) -> Option<&'a InstalledRuntimeManifest> {
    active_runtime_key
        .and_then(|key| {
            manifests
                .iter()
                .find(|manifest| manifest.runtime_key == key)
        })
        .or_else(|| manifests.first())
}

fn build_startup_update_check_record(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    manifests: &[InstalledRuntimeManifest],
    now_unix_ms: u128,
    download_timeout_secs: Option<u64>,
) -> StartupUpdateCheckRecord {
    match resolve_latest_for_manifest(paths, manifest, download_timeout_secs) {
        Ok(latest) => {
            let freshness = runtime_freshness_with_manifests(
                manifests,
                manifest,
                &latest.latest_version,
                latest.wheel_composition.as_ref(),
                &latest.target_runtime_key,
            );
            StartupUpdateCheckRecord {
                runtime_key: manifest.runtime_key.clone(),
                runtime_id: manifest.runtime_id.clone(),
                channel: manifest.channel.clone(),
                format: latest.format,
                family: manifest.family.clone(),
                installed_version: manifest.version.clone(),
                latest_version: Some(latest.latest_version),
                status: freshness.status().to_owned(),
                message: None,
                checked_at_unix_ms: now_unix_ms,
            }
        }
        Err(error) => StartupUpdateCheckRecord {
            runtime_key: manifest.runtime_key.clone(),
            runtime_id: manifest.runtime_id.clone(),
            channel: manifest.channel.clone(),
            format: manifest.format.clone(),
            family: manifest.family.clone(),
            installed_version: manifest.version.clone(),
            latest_version: None,
            status: "error".to_owned(),
            message: Some(error.to_string()),
            checked_at_unix_ms: now_unix_ms,
        },
    }
}

fn startup_update_check_path(paths: &AppPaths) -> PathBuf {
    paths
        .cache_dir
        .join("therock")
        .join("startup-update-check.json")
}

fn load_startup_update_check(paths: &AppPaths) -> Result<Option<StartupUpdateCheckRecord>> {
    let path = startup_update_check_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let record = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(record))
}

fn save_startup_update_check(paths: &AppPaths, record: &StartupUpdateCheckRecord) -> Result<()> {
    let path = startup_update_check_path(paths);
    write_file_atomically(
        &path,
        &serde_json::to_vec_pretty(record)
            .context("failed to serialize startup update check record")?,
    )
}

#[cfg(test)]
mod tests {
    use super::super::process::{PROGRESS_LINE_SINK, progress_output_suppressed};
    use super::super::python::{OverridePythonResolverEnv, PythonResolverEnv};
    use super::super::source::THEROCK_SOURCE_LAYOUT_GENERATION;
    use super::super::test_support::{
        test_paths, test_runtime_manifest, write_test_runtime_manifest,
    };
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;

    /// Updating a next-layout manifest whose family is a group label (as
    /// `gfx125X-dcgpu` always is) must recover the exact arch this runtime was
    /// installed with from its composition, not resolve an undetermined
    /// device target on the very host that arch came from.
    #[test]
    fn manifest_wheel_family_override_recovers_the_installed_arch() {
        let mut manifest = test_runtime_manifest("next-v1:gfx125X-dcgpu", "next:gfx125X-dcgpu", 0);
        manifest.wheel_composition = Some(test_wheel_composition("gfx1250"));

        assert_eq!(manifest_wheel_family_override(&manifest), "gfx1250");
    }

    /// A composition recorded for a different family (e.g. moved to another
    /// host) must not be trusted — the plain family label is the safe fallback.
    #[test]
    fn manifest_wheel_family_override_ignores_a_disagreeing_composition() {
        let mut manifest = test_runtime_manifest("next-v1:gfx125X-dcgpu", "next:gfx125X-dcgpu", 0);
        manifest.wheel_composition = Some(test_wheel_composition("gfx90a"));

        assert_eq!(manifest_wheel_family_override(&manifest), "gfx125X-dcgpu");
    }

    /// No recorded composition at all (an older manifest, or an undetermined
    /// install) falls back to the plain family label, same as before this fix.
    #[test]
    fn manifest_wheel_family_override_falls_back_without_a_composition() {
        let manifest = test_runtime_manifest("v1:gfx120X-all", "canonical:gfx120X-all", 0);

        assert_eq!(manifest_wheel_family_override(&manifest), "gfx120X-all");
    }

    /// An installed runtime must keep resolving the stream it came from, and a
    /// manifest written before the field existed must still be readable.
    #[test]
    fn manifest_layout_survives_older_manifests() {
        let mut manifest = test_runtime_manifest("runtime-key", "therock-release:gfx120X-all", 0);
        manifest.source_layout_generation = Some("next-v1".to_owned());
        assert_eq!(
            manifest_source_layout(&manifest).unwrap(),
            SourceLayout::Next
        );

        manifest.source_layout_generation = None;
        manifest.wheel_composition = Some(WheelRuntimeComposition {
            source_layout_generation: "next-v1".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx1200]==10.0.0".to_owned()],
            rocm_sdk_target: Some("gfx1200".to_owned()),
        });
        assert_eq!(
            manifest_source_layout(&manifest).unwrap(),
            SourceLayout::Next
        );

        manifest.wheel_composition = None;
        assert_eq!(
            manifest_source_layout(&manifest).unwrap(),
            SourceLayout::Canonical
        );

        manifest.source_layout_generation = Some("some-future-generation".to_owned());
        let error = manifest_source_layout(&manifest).unwrap_err().to_string();
        assert!(error.contains("unsupported TheRock source layout generation"));
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn render_update_json_installs_the_suppression_guard_around_resolution() -> Result<()> {
        // Unlike the two guard tests in `therock::process`
        // (`suppress_progress_output_contract` and
        // `progress_line_is_suppressed_while_guard_is_held`, which exercise the
        // guard in isolation), this proves `render_update_json` itself
        // installs the guard around a code path that *actually reaches* a
        // live `progress_line` call: a wheel-format manifest whose only PATH
        // python is a non-executable stub fails
        // `python_launcher_install_ready`, so `resolve_python_launcher_in`
        // falls back to the managed-Python path and calls
        // `progress_line("Python from PATH is not usable for this ROCm
        // install ...; using ROCm CLI's managed Python.")` before bailing out
        // (managed bootstrap is disabled here, so the whole thing stays
        // network-free). If `render_update_json` stopped installing the
        // guard, that call would land in `PROGRESS_LINE_SINK` instead of
        // being swallowed, and the assertion below would fail.
        //
        // The PATH-python-unusable branch is reached via a thread-local
        // `OverridePythonResolverEnv`, not a real process `PATH` mutation:
        // see issue #650, where mutating process-wide `PATH` to steer this
        // test raced any sibling test thread that spawned a PATH-resolved
        // binary (e.g. bare-name `tar`) during the window.
        //
        // The lock below is no longer for `PATH`: it's still needed because
        // this test mutates the real `ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP`
        // process env var below, and every process-env-mutating test in
        // `therock` (`therock.rs` and its child modules) takes this lock, which
        // lives in `test_support`.
        let _env_guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("render-update-json-installs-guard");

        let manifest = test_runtime_manifest("active", "therock-release:gfx120X-all", 1);
        write_test_runtime_manifest(&paths, &manifest)?;

        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        fs::write(bin_dir.join("python3"), "not an interpreter")?;

        let old_bootstrap_disabled = std::env::var_os("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP");
        unsafe {
            std::env::set_var("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP", "1");
        }
        let resolver_env_guard = OverridePythonResolverEnv::new(PythonResolverEnv {
            python_override: None,
            search_dirs: vec![bin_dir],
        });

        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        assert!(!progress_output_suppressed());
        let result = render_update_json(&paths, Some(1));

        drop(resolver_env_guard);
        unsafe {
            match old_bootstrap_disabled {
                Some(value) => {
                    std::env::set_var("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP", value);
                }
                None => std::env::remove_var("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"),
            }
        }

        assert!(
            !progress_output_suppressed(),
            "the guard must be dropped once render_update_json returns"
        );
        let captured = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
        assert!(
            captured.is_empty(),
            "a progress_line call reachable during resolution must be suppressed by \
             render_update_json's guard, not delivered to the sink: {captured:?}"
        );

        let document = result?;
        assert_eq!(document.runtimes.len(), 1);
        assert_eq!(
            document.runtimes[0].status, "error",
            "PATH python cannot create a venv and managed bootstrap is disabled, \
             so resolution must fail"
        );
        let message = document.runtimes[0]
            .message
            .as_deref()
            .expect("a failed resolution should carry an error message");
        assert!(
            message.contains("managed Python bootstrap is disabled"),
            "expected the deterministic bootstrap-disabled failure, got: {message}"
        );

        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn progress_suppression_guard_releases_on_ok_and_err_exit_paths() {
        // This only proves the guard's own release semantics on both of
        // `render_update_json`'s exit paths (both fixtures below exit before
        // ever reaching resolution, so `progress_line` is never called here).
        // It does NOT prove `render_update_json` installs the guard around a
        // live `progress_line` call — see
        // `render_update_json_installs_the_suppression_guard_around_resolution`
        // for that.
        //
        // Ok path: no `runtimes` directory at all, so `load_runtime_manifests`
        // returns `Ok(vec![])` and `render_update_json` succeeds trivially.
        let (ok_root, ok_paths) = test_paths("render-update-json-ok");
        fs::create_dir_all(&ok_root).unwrap();
        assert!(!progress_output_suppressed());
        let result = render_update_json(&ok_paths, Some(1));
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert!(
            !progress_output_suppressed(),
            "the guard must be dropped after the Ok path"
        );
        fs::remove_dir_all(&ok_root).ok();

        // Err path: `broken.json` is a directory, not a file, so the loader's
        // `fs::read` on it fails and the error bubbles through
        // `render_update_json`'s `?` before ever constructing an `UpdateJson`.
        let (err_root, err_paths) = test_paths("render-update-json-err");
        let registry_dir = err_root.join("data").join("runtimes").join("registry");
        fs::create_dir_all(registry_dir.join("broken.json")).unwrap();
        let result = render_update_json(&err_paths, Some(1));
        assert!(
            result.is_err(),
            "a directory named *.json must fail to be read as manifest bytes"
        );
        assert!(
            !progress_output_suppressed(),
            "the guard must be dropped after the Err path too"
        );
        fs::remove_dir_all(&err_root).ok();
    }

    fn test_wheel_composition(device_target: &str) -> WheelRuntimeComposition {
        WheelRuntimeComposition {
            source_layout_generation: THEROCK_SOURCE_LAYOUT_GENERATION.to_owned(),
            package_specs: vec![
                format!("rocm[libraries,devel,device-{device_target}]==7.14.0"),
                "torch==2.11.0+rocm7.14.0".to_owned(),
                "torchvision==0.26.0+rocm7.14.0".to_owned(),
                "torchaudio==2.11.0+rocm7.14.0".to_owned(),
            ],
            rocm_sdk_target: Some(device_target.to_owned()),
        }
    }

    #[test]
    fn legacy_manifest_without_a_composition_is_repaired_once_then_settles() {
        // Exactly what a runtime installed before composition-aware freshness
        // deserializes to: no `wheel_composition`, and a version-only key.
        let mut old_cache: InstalledRuntimeManifest = serde_json::from_value(serde_json::json!({
            "runtime_key": "release-wheel-multi-arch-7-14-0",
            "runtime_id": "therock-release:gfx94X-dcgpu",
            "channel": "release",
            "format": "wheel",
            "family": "gfx94X-dcgpu",
            "family_source": "managed-runtime",
            "version": "7.14.0",
            "install_root": "/tmp/release-wheel-multi-arch-7-14-0",
            "selected_artifact_url": "https://repo.amd.com/rocm/whl-multi-arch",
            "index_url": "https://repo.amd.com/rocm/whl-multi-arch",
            "tarball_file_name": null,
            "python_launcher": "/usr/bin/python3",
            "python_executable": "/tmp/release-wheel-multi-arch-7-14-0/bin/python",
            "pip_cache_dir": null,
            "rocm_sdk": null,
            "read_only": false,
            "imported_from": null,
            "installed_at_unix_ms": 1
        }))
        .unwrap();
        assert_eq!(
            old_cache.wheel_composition, None,
            "a pre-composition manifest must still load"
        );

        let required = test_wheel_composition("gfx942");
        let target_runtime_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required);

        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key),
            RuntimeFreshness::RepairAvailable
        );

        old_cache.wheel_composition = Some(required.clone());
        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key),
            RuntimeFreshness::RepairAvailable,
            "the right packages under the legacy identity still need side-by-side migration"
        );

        old_cache.runtime_key = target_runtime_key.clone();
        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key),
            RuntimeFreshness::UpToDate
        );
    }

    #[test]
    fn a_newer_index_version_outranks_a_composition_repair() {
        let mut manifest = test_runtime_manifest(
            "release-wheel-multi-arch-7-13-0",
            "therock-release:gfx94X-dcgpu",
            1,
        );
        manifest.version = "7.13.0".to_owned();
        let required = test_wheel_composition("gfx942");

        assert_eq!(
            runtime_freshness(
                &manifest,
                "7.14.0",
                Some(&required),
                &wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required),
            ),
            RuntimeFreshness::UpdateAvailable
        );
    }

    #[test]
    fn an_ahead_of_index_runtime_is_never_offered_an_unreproducible_repair() {
        // Its version is not in the index, so no install could reproduce it. A
        // repair here would have to roll the runtime back to the older index
        // build, which is exactly what `ahead_of_index` exists to prevent.
        let mut pinned = test_runtime_manifest(
            "release-wheel-multi-arch-7-15-0",
            "therock-release:gfx94X-dcgpu",
            1,
        );
        pinned.version = "7.15.0".to_owned();
        let required = test_wheel_composition("gfx942");

        assert_eq!(
            runtime_freshness(
                &pinned,
                "7.14.0",
                Some(&required),
                &wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required),
            ),
            RuntimeFreshness::AheadOfIndex
        );
    }

    #[test]
    fn a_read_only_runtime_is_never_repaired() {
        // `runtimes import` / `adopt` point at a folder this CLI does not own,
        // so a side-by-side "replacement" would be an install the user never
        // asked for, against packages they did not choose.
        let mut adopted =
            test_runtime_manifest("imported-rocm-7-14-0", "therock-release:gfx94X-dcgpu", 1);
        adopted.version = "7.14.0".to_owned();
        adopted.read_only = true;
        let required = test_wheel_composition("gfx942");

        assert_eq!(
            runtime_freshness(
                &adopted,
                "7.14.0",
                Some(&required),
                &wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required),
            ),
            RuntimeFreshness::UpToDate
        );
    }

    #[test]
    fn an_installed_replacement_sibling_suppresses_repeat_repair() {
        let required = test_wheel_composition("gfx942");
        let target_runtime_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required);
        let mut source = test_runtime_manifest(
            "release-wheel-multi-arch-7-14-0",
            "therock-release:gfx94X-dcgpu",
            1,
        );
        source.version = "7.14.0".to_owned();
        let mut replacement =
            test_runtime_manifest(&target_runtime_key, "therock-release:gfx94X-dcgpu", 2);
        replacement.version = "7.14.0".to_owned();
        replacement.wheel_composition = Some(required.clone());
        let (root, _) = test_paths("installed-replacement-sibling");
        replacement.install_root = root.join("replacement");
        fs::create_dir_all(&replacement.install_root).unwrap();
        fs::write(replacement.install_root.join("installed.marker"), b"ok").unwrap();

        let alone = vec![source.clone()];
        assert_eq!(
            runtime_freshness_with_manifests(
                &alone,
                &source,
                "7.14.0",
                Some(&required),
                &target_runtime_key,
            ),
            RuntimeFreshness::RepairAvailable,
            "the legacy runtime alone still has to be replaced"
        );

        let migrated = vec![source.clone(), replacement.clone()];
        assert_eq!(
            runtime_freshness_with_manifests(
                &migrated,
                &source,
                "7.14.0",
                Some(&required),
                &target_runtime_key,
            ),
            RuntimeFreshness::UpToDate,
            "the retained legacy manifest must not keep re-triggering the same repair"
        );

        // A sibling that merely shares the key without the composition is not the
        // replacement: accepting it would strand the tree one repair short.
        let mut impostor = replacement.clone();
        impostor.wheel_composition = Some(test_wheel_composition("gfx950"));
        assert_eq!(
            runtime_freshness_with_manifests(
                &[source.clone(), impostor],
                &source,
                "7.14.0",
                Some(&required),
                &target_runtime_key,
            ),
            RuntimeFreshness::RepairAvailable
        );

        // A correct manifest is not an installed replacement after its runtime
        // directory disappears; accepting it would suppress every repair.
        fs::remove_dir_all(&replacement.install_root).unwrap();
        assert_eq!(
            runtime_freshness_with_manifests(
                &[source.clone(), replacement],
                &source,
                "7.14.0",
                Some(&required),
                &target_runtime_key,
            ),
            RuntimeFreshness::RepairAvailable
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn update_json_reports_no_managed_runtimes_as_empty_list() -> Result<()> {
        let (root, paths) = test_paths("update-json-empty");

        let document = render_update_json(&paths, None)?;

        assert!(document.runtimes.is_empty());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn update_json_reports_per_manifest_error_without_failing_whole_report() -> Result<()> {
        let (root, paths) = test_paths("update-json-error");
        // An unsupported channel fails `TheRockChannel::parse` synchronously,
        // so this error path is deterministic and never depends on real
        // network reachability (unlike the manifest's placeholder index URL,
        // which `resolve_pip_runtime_with_timeout` doesn't even consult).
        //
        // Deliberate gap: this only covers the all-error case. Asserting a
        // successful row survives alongside a failing one would need a second
        // manifest that resolves for real, and no fixture in this module stands
        // up a resolvable wheel index — only the raw-body local HTTP servers
        // that the `therock::fetch` tests start exist (for example in
        // `native_http_download_and_get_round_trip_without_powershell`), for
        // their lower-level use. Not worth building just for this one assertion.
        let mut manifest = test_runtime_manifest("active", "therock-release:gfx120X-all", 1);
        manifest.channel = "unsupported-channel".to_owned();
        write_test_runtime_manifest(&paths, &manifest)?;

        let document = render_update_json(&paths, None)?;

        assert_eq!(document.runtimes.len(), 1);
        let row = &document.runtimes[0];
        assert_eq!(row.runtime_key, "active");
        assert_eq!(row.status, "error");
        assert!(row.latest_version.is_none());
        assert!(row.message.is_some());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn update_report_policy_mentions_bounded_startup_check() -> Result<()> {
        let (root, paths) = test_paths("update-report-policy");

        let rendered = render_update_report(&paths)?;

        assert!(rendered.contains("policy: bounded startup check, cached metadata"));
        assert!(rendered.contains("prompt before mutating state"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn startup_update_check_skips_first_run_without_creating_cache() -> Result<()> {
        let (root, paths) = test_paths("startup-no-runtime");

        let record = maybe_refresh_startup_update_check_at(&paths, None, 1_000)?;

        assert!(record.is_none());
        assert!(!paths.cache_dir.exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn startup_update_check_due_uses_bounded_interval() {
        assert!(!startup_update_check_due(
            1_000,
            1_000 + STARTUP_UPDATE_CHECK_INTERVAL_MS - 1
        ));
        assert!(startup_update_check_due(
            1_000,
            1_000 + STARTUP_UPDATE_CHECK_INTERVAL_MS
        ));
    }

    #[test]
    fn startup_update_check_prefers_active_runtime_key() {
        let newest = test_runtime_manifest("newer", "therock-release:gfx120X-all", 2);
        let active = test_runtime_manifest("active", "therock-release:gfx110X-all", 1);
        let manifests = vec![newest, active];

        let selected = select_startup_update_manifest(&manifests, Some("active"))
            .expect("active runtime should be selected");

        assert_eq!(selected.runtime_key, "active");
        assert_eq!(
            select_startup_update_manifest(&manifests, None)
                .expect("newest runtime should be selected")
                .runtime_key,
            "newer"
        );
    }

    #[test]
    fn startup_update_check_uses_recent_record_without_network() -> Result<()> {
        let (root, paths) = test_paths("startup-recent-record");
        let manifest = test_runtime_manifest("active", "therock-release:gfx120X-all", 1);
        write_test_runtime_manifest(&paths, &manifest)?;
        save_startup_update_check(
            &paths,
            &StartupUpdateCheckRecord {
                runtime_key: "active".to_owned(),
                runtime_id: manifest.runtime_id.clone(),
                channel: manifest.channel.clone(),
                format: manifest.format.clone(),
                family: manifest.family.clone(),
                installed_version: manifest.version.clone(),
                latest_version: Some(manifest.version),
                status: "up_to_date".to_owned(),
                message: None,
                checked_at_unix_ms: 2_000,
            },
        )?;

        let record = maybe_refresh_startup_update_check_at(&paths, Some("active"), 2_001)?
            .expect("recent check should be returned");

        assert_eq!(record.status, "up_to_date");
        assert!(!paths.cache_dir.join("therock").join("metadata").exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
