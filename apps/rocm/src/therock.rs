// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// Property-based tests for version ordering and update-source selection.
/// A child module so it can reach this module's private items, and the
/// `pub(super)` comparators in `version`, directly.
#[cfg(test)]
#[path = "therock_version_proptests.rs"]
mod version_proptests;

mod family;
mod fetch;
mod io;
mod probe;
mod process;
mod python;
mod resolve;
mod source;
#[cfg(test)]
mod test_support;
mod version;

use self::family::{family_override_or_recovered_arch, resolve_family};
pub(crate) use self::fetch::truthy_env;
use self::fetch::{download_file, preflight_tarball_space};
pub(crate) use self::io::write_file_atomically;
use self::probe::probe_rocm_sdk_runtime_for_target;
pub(crate) use self::probe::{
    RuntimeDeviceProbe, TorchAlignmentProbe, install_pinned_package, probe_rocm_sdk_runtime,
    probe_runtime_devices, probe_torch_alignment, requirement_pinned_version, split_local_version,
};
#[cfg(test)]
use self::process::{PROGRESS_LINE_SINK, progress_output_suppressed};
use self::process::{
    SuppressProgressOutput, extract_tarball_and_discard_archive, progress_line,
    run_uv_progress_command,
};
#[cfg(test)]
use self::python::{OverridePythonResolverEnv, PythonResolverEnv};
use self::python::{
    PythonLauncher, ensure_uv_venv, managed_python_version, python_launcher_install_ready,
    python_requirement, resolve_python_launcher, wheel_compatibility_for_python,
};
use self::resolve::{
    AggregateDeviceTarget, canonical_auto_select_needs_next_python, resolve_pip_runtime,
    resolve_pip_runtime_with_timeout, resolve_tarball_artifact,
    resolve_tarball_artifact_with_timeout, unverified_torchaudio_note, wheel_runtime_composition,
};
pub(crate) use self::source::RuntimeVersionSelector;
use self::source::{
    SourceLayout, THEROCK_NEXT_MIN_MAJOR, TheRockChannel, next_layout_requested,
    requested_source_layout, select_source_layout,
};
#[cfg(test)]
use self::source::{THEROCK_SOURCE_LAYOUT_GENERATION, resolve_source};
pub(crate) use self::version::runtime_version_display;
use self::version::{
    compare_version_strings, parse_host_version, parse_version_for_ordering,
    runtime_version_build_date, version_relation,
};
#[cfg(test)]
pub(crate) use crate::runtime_registry::RocmSdkLibraryProbe;
#[cfg(test)]
use crate::runtime_registry::runtime_registry_dir;
pub(crate) use crate::runtime_registry::{
    InstalledRuntimeManifest, RocmSdkPythonProbe, WheelRuntimeComposition, load_runtime_manifests,
    validate_rocm_sdk_runtime_probe,
};
use crate::runtime_registry::{
    load_runtime_manifests_reporting_unparsed, runtime_manifest_path, save_runtime_manifest,
    wheel_composition_device_target, wheel_composition_includes_devel,
};
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, detect_host_gpu_diagnostics, detect_legacy_rocm_summary,
    ensure_uv_binary, interactive_terminal, runtime_is_windows, runtime_python_executable_in_env,
    unix_time_millis, uv_pip_install_base, uv_venv_args,
};
#[cfg(test)]
use rocm_core::{disk_space, managed_uv_cache_dir, normalize_therock_family};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const STARTUP_UPDATE_CHECK_INTERVAL_MS: u128 = 12 * 60 * 60 * 1_000;
const STARTUP_UPDATE_CHECK_TIMEOUT_SECS: u64 = 2;
fn render_canonical_provenance(
    output: &mut String,
    channel: TheRockChannel,
    source_url: &str,
    layout_generation: &str,
    version: &str,
) {
    let build_date = runtime_version_build_date(version)
        .unwrap_or_else(|| "not encoded in stable version".to_owned());
    let _ = writeln!(output, "  channel: {}", channel.as_str());
    let _ = writeln!(output, "  canonical_source: {source_url}");
    let _ = writeln!(output, "  selected_rocm_version: {version}");
    let _ = writeln!(output, "  build_date: {build_date}");
    let _ = writeln!(output, "  source_layout_generation: {layout_generation}");
}

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
enum RuntimeFreshness {
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

/// Outcome of an `install sdk` request.
///
/// `mutated` is `false` for dry-run plans and for installs the user declined at
/// the confirmation prompt, so the caller can skip the post-install activation
/// and success reporting that only make sense after a real install.
#[derive(Debug)]
pub(crate) struct SdkInstallResult {
    pub output: String,
    pub mutated: bool,
}

impl SdkInstallResult {
    const fn plan(output: String) -> Self {
        Self {
            output,
            mutated: false,
        }
    }

    const fn installed(output: String) -> Self {
        Self {
            output,
            mutated: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InstallSourceOverride<'a> {
    family: Option<&'a str>,
    device_target: Option<&'a str>,
    layout: Option<SourceLayout>,
}

/// What to install, as resolved from the `rocm install sdk` arguments.
///
/// A struct rather than a parameter list: with `include_devel` added this is
/// past the point where positional `bool`s at a call site say anything about
/// what they mean.
#[derive(Debug)]
pub(crate) struct SdkInstallRequest<'a> {
    pub channel: &'a str,
    pub format: &'a str,
    pub prefix: Option<PathBuf>,
    pub version_selector: Option<RuntimeVersionSelector>,
    pub family_override: Option<&'a str>,
    pub dry_run: bool,
    /// Install the compiler and headers alongside the runtime libraries.
    pub include_devel: bool,
    /// The *source* of any up-front approval rather than a bare bool, because
    /// the progress line names it: `--yes` and
    /// `--approve-replacing-active-default` both clear this gate, but only the
    /// former also approves a `sudo` system-package install, so collapsing them
    /// would print "Approved by --yes" on every install ROCm CLI's own
    /// terminal-less surfaces make.
    pub consent: SdkInstallConsent,
}

impl Default for SdkInstallRequest<'_> {
    fn default() -> Self {
        Self {
            channel: "",
            format: "",
            prefix: None,
            version_selector: None,
            family_override: None,
            dry_run: false,
            include_devel: false,
            // Withholding consent is the safe default: a caller that leaves it
            // unset has granted nothing, so the gate prompts or refuses rather
            // than silently displacing the active default runtime.
            consent: SdkInstallConsent::Ask,
        }
    }
}

/// Install a TheRock SDK runtime.
pub(crate) fn install_sdk(
    paths: &AppPaths,
    request: SdkInstallRequest<'_>,
) -> Result<SdkInstallResult> {
    let SdkInstallRequest {
        channel,
        format,
        prefix,
        version_selector,
        family_override,
        dry_run,
        include_devel,
        consent,
    } = request;
    let channel = TheRockChannel::parse(channel)?;
    ensure_install_format_supported(format)?;
    match format {
        "wheel" => install_wheel_runtime(
            paths,
            channel,
            prefix,
            InstallSourceOverride {
                family: family_override,
                ..InstallSourceOverride::default()
            },
            version_selector.as_ref(),
            dry_run,
            include_devel,
            consent,
        ),
        "tarball" => {
            // A tarball catalog lists whole archives, not a resolvable package
            // graph, so there is nothing to pin a version against — except the
            // ROCm 10 catalog, which is reachable *only* by naming a version,
            // because nothing else distinguishes it from the canonical one.
            if let Some(selector) = version_selector.as_ref()
                && !next_layout_requested(selector)
            {
                bail!(
                    "specific TheRock version selection for tarball installs is only supported for a stable ROCm {THEROCK_NEXT_MIN_MAJOR}+ pin (e.g. `--version 10.0.0`); for any other version, use `--format wheel`"
                )
            }
            install_tarball_runtime(
                paths,
                channel,
                prefix,
                family_override,
                version_selector.as_ref(),
                None,
                dry_run,
                consent,
            )
        }
        other => bail!("unsupported install format: {other}"),
    }
}

/// Apply an update using the exact family, device payload, and source layout
/// resolved by its plan.
///
/// Consent is preapproved rather than asked for: the update targets the runtime
/// the user selected (or the active default), so installing over it is the
/// operation requested, and `rocm update` has no terminal contract — reaching a
/// prompt here would only fail the command. `rocm update` does accept a `--yes`
/// flag for consistency with other mutating commands, but it is inert and never
/// reaches this function, so it grants nothing here.
///
/// `activate_after_install` mirrors `rocm update --apply --activate` so the
/// approval line can state what will actually happen. Without it,
/// `apply_runtime_update` installs beside the active default and leaves the
/// default untouched, so the line must not claim an activation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn install_sdk_for_update(
    paths: &AppPaths,
    channel: &str,
    format: &str,
    family: &str,
    device_target: Option<&str>,
    source_layout_generation: Option<&str>,
    include_devel: bool,
    dry_run: bool,
    activate_after_install: bool,
) -> Result<SdkInstallResult> {
    let channel = TheRockChannel::parse(channel)?;
    ensure_install_format_supported(format)?;
    let consent = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::UpdateApply {
        activates: activate_after_install,
    });
    let layout = Some(SourceLayout::from_generation(source_layout_generation)?);
    match format {
        "wheel" => install_wheel_runtime(
            paths,
            channel,
            None,
            InstallSourceOverride {
                family: Some(family),
                device_target,
                layout,
            },
            None,
            dry_run,
            include_devel,
            consent,
        ),
        "tarball" => install_tarball_runtime(
            paths,
            channel,
            None,
            Some(family),
            None,
            layout,
            dry_run,
            consent,
        ),
        other => bail!("unsupported install format: {other}"),
    }
}

fn ensure_install_format_supported(format: &str) -> Result<()> {
    ensure_install_format_supported_for_platform(format, runtime_is_windows())
}

fn ensure_install_format_supported_for_platform(format: &str, windows: bool) -> Result<()> {
    if windows && format == "tarball" {
        bail!(
            "TheRock tarball installs are not supported on Windows; use `rocm install sdk --format wheel` for a managed wheel virtual environment"
        );
    }
    Ok(())
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
fn runtime_freshness(
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

fn select_startup_update_manifest<'a>(
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

#[allow(clippy::too_many_arguments)]
fn install_wheel_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    source_override: InstallSourceOverride<'_>,
    version_selector: Option<&RuntimeVersionSelector>,
    dry_run: bool,
    include_devel: bool,
    consent: SdkInstallConsent,
) -> Result<SdkInstallResult> {
    let InstallSourceOverride {
        family: family_override,
        device_target: device_target_override,
        layout: layout_override,
    } = source_override;
    // `device_target_override` (an update apply's exact recorded arch) is only
    // otherwise consulted after `resolve_pip_runtime` returns — too late for the
    // channel/arch check below, which needs an exact arch before it can run at
    // all. Recover it into the family override up front, same as `resolve_pip_runtime`
    // does internally.
    let recovered_family_override = family_override
        .map(|family| family_override_or_recovered_arch(family, device_target_override));
    // Fail fast on a request `resolve_pip_runtime` could only ever reject (wrong
    // channel, or a grouped family with no exact arch) before paying for Python
    // provisioning below, which can mean a real download for a layout this host
    // or channel can never actually use.
    select_source_layout(
        channel,
        &resolve_family(paths, recovered_family_override.as_deref())?,
        version_selector,
    )?;
    // The interpreter has to be picked before `resolve_pip_runtime`, which
    // selects wheels using that interpreter's own tags. `requested_source_layout`
    // only answers from the *request*: an explicit stable ROCm 10+ pin already
    // routes to `Next` and gets `cp314` that way, but a plain auto-select or a
    // nightly build-date/prerelease pin always reports `Canonical`, even once
    // the canonical/nightly feed's own "latest" has rotated past that major on
    // its own, with no `Next` routing involved. `canonical_auto_select_needs_next_python`
    // peeks the channel's own `rocm` listing to catch that case.
    let python_requirement = match requested_source_layout(layout_override, version_selector) {
        SourceLayout::Next => python_requirement(SourceLayout::Next),
        SourceLayout::Canonical => {
            if canonical_auto_select_needs_next_python(paths, channel, version_selector)? {
                python_requirement(SourceLayout::Next)
            } else {
                python_requirement(SourceLayout::Canonical)
            }
        }
    };
    progress_line(format!(
        "Checking Python for the ROCm install; if needed, ROCm CLI will prepare Python {}.",
        managed_python_version(python_requirement)
    ));
    let python_launcher = resolve_python_launcher(paths, python_requirement)?;
    progress_line(match python_launcher.source {
        "path" => format!(
            "Using Python from PATH: {}.",
            python_launcher.executable.display()
        ),
        "env" => format!(
            "Using Python from ROCM_CLI_PYTHON: {}.",
            python_launcher.executable.display()
        ),
        "managed" => format!(
            "Using ROCm CLI's portable Python: {}.",
            python_launcher.executable.display()
        ),
        _ => format!(
            "Using Python from {}.",
            python_launcher.executable.display()
        ),
    });
    let wheel_compatibility = wheel_compatibility_for_python(&python_launcher.executable)?;
    progress_line(format!(
        "Checking TheRock {} packages for this AMD GPU...",
        channel.as_str()
    ));
    let resolution = resolve_pip_runtime(
        paths,
        channel,
        recovered_family_override.as_deref(),
        &wheel_compatibility,
        version_selector,
        layout_override,
        include_devel,
    )?;
    let device_target = device_target_override.map_or_else(
        || resolution.device_target.clone(),
        |target| {
            AggregateDeviceTarget::resolve(
                Some(target),
                &resolution.family,
                &resolution.published_device_targets,
            )
        },
    );
    // The exact device payload, not the version alone, decides what this runtime
    // can run, so it is what identifies the runtime. A preview on a host with no
    // usable target still composes a key here — from the `<undetermined>` extras
    // — which no real install can ever produce, and the refusal below stops it
    // from reaching a manifest.
    let wheel_composition = wheel_runtime_composition(&resolution, &device_target, include_devel);
    progress_line(format!(
        "Found canonical TheRock aggregate version {} with a matching PyTorch stack for target family {}.",
        resolution.latest_version, resolution.family
    ));
    let runtime_key = wheel_runtime_key(channel, &resolution.latest_version, &wheel_composition);
    let install_root = resolved_install_root(paths, "wheel", &runtime_key, prefix);
    let manifest_path = runtime_manifest_path(paths, &runtime_key);

    let mut output = String::new();
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(
        output,
        "  summary: rocm-cli will install the ROCm SDK and matching PyTorch packages for this Python and operating system"
    );
    render_canonical_provenance(
        &mut output,
        channel,
        &resolution.index_url,
        resolution.layout.generation(),
        &resolution.latest_version,
    );
    let _ = writeln!(output, "  format: wheel");
    if let Some(selector) = version_selector {
        let _ = writeln!(output, "  requested: {}", selector.describe());
    }
    let _ = writeln!(output, "  target_family: {}", resolution.family);
    let _ = writeln!(
        output,
        "  target_family_source: {}",
        resolution.family_source
    );
    let _ = writeln!(output, "  device_target: {}", device_target.as_str());
    if let Some(reason) = device_target.reason() {
        let _ = writeln!(output, "  device_target_reason: {reason}");
    }
    let _ = writeln!(output, "  index_url: {}", resolution.index_url);
    let _ = writeln!(
        output,
        "  latest_compatible_version: {}",
        runtime_version_display(&resolution.latest_version)
    );
    // Probed once and reused by the progress line below: each call is a full
    // filesystem scan for a legacy ROCm, and the summary block and the visible
    // note report the same answer about the same resolved version.
    let host_version_newer = host_rocm_version_newer_than(&resolution.latest_version);
    if let Some(host_version) = host_version_newer.as_deref() {
        let _ = writeln!(
            output,
            "  version_note: {}",
            wheel_host_version_note(
                host_version,
                &runtime_version_display(&resolution.latest_version)
            )
        );
    }
    let _ = writeln!(
        output,
        "  compatibility_key: {}",
        runtime_version_display(&resolution.package_versions.compatibility_key)
    );
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    let _ = writeln!(
        output,
        "  python_launcher: {}",
        python_launcher.executable.display()
    );
    let _ = writeln!(output, "  python_source: {}", python_launcher.source);
    let _ = writeln!(
        output,
        "  python_wheel_tag: {}",
        wheel_compatibility.python_tag
    );
    let _ = writeln!(
        output,
        "  platform_wheel_tags: {}",
        wheel_compatibility.platform_tags.join(",")
    );
    // What resolution asked for, printed beside what the install will ask for.
    // The two are produced by different code paths from the same `include_devel`
    // and must name the same extras; showing only the second would hide a
    // resolve that constrained the version choice by a toolchain the user
    // declined. Absent on the canonical layout, which resolves by scraping each
    // package's index instead of compiling a requirement set.
    if let Some(specs) = resolution.version_resolution_specs.as_ref() {
        let _ = writeln!(output, "  version_resolution_specs: {}", specs.join(" "));
    }
    // Printed from the same value that chose the pin, so the line cannot claim
    // a verified build the requirements above do not name.
    if let Some(pin) = resolution.torchaudio_pin.as_ref() {
        let _ = writeln!(output, "  torchaudio_selection: {}", pin.describe());
    }
    let _ = writeln!(
        output,
        "  package_specs: {}",
        wheel_composition.package_specs.join(" ")
    );
    let _ = writeln!(
        output,
        "  package_policy: resolve the pinned target-complete rocm, torch, torchvision, and torchaudio plan from published package metadata, then install it in one uv transaction"
    );
    let no_wheel_warning = repo_version_without_wheels(
        resolution.newest_repo_version.as_deref(),
        &resolution.latest_version,
    )
    .map(|newest| {
        no_wheel_warning_message(
            &newest,
            &runtime_version_display(&resolution.latest_version),
        )
    });
    if let Some(warning) = no_wheel_warning.as_deref() {
        let _ = writeln!(output, "  warning: {warning}");
    }
    if dry_run {
        let env_python = venv_python_path(&install_root);
        let mut install_args = uv_pip_install_base(&env_python);
        install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
        if matches!(channel, TheRockChannel::Nightly) {
            install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
        }
        install_args.extend(wheel_composition.package_specs.iter().cloned());
        let venv_args = uv_venv_args(&python_launcher.executable, &install_root);
        let venv_args_display = venv_args
            .iter()
            .map(|arg| quote_display_arg(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let install_args_display = install_args
            .iter()
            .map(|arg| quote_display_arg(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(
            output,
            "  command: uv {venv_args_display} && uv {install_args_display}"
        );
        let _ = writeln!(
            output,
            "  activation: use the managed venv Python; TheRock libraries are resolved from that venv by rocm_sdk.initialize_process"
        );
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(SdkInstallResult::plan(output));
    }

    // Requirement: surface the "newest version has no wheels" case as a warning
    // on the real install path too, not only in the dry-run plan.
    if let Some(warning) = no_wheel_warning.as_deref() {
        progress_line(format!("Warning: {warning}"));
    }
    if let Some(note) = resolution
        .torchaudio_pin
        .as_ref()
        .and_then(unverified_torchaudio_note)
    {
        progress_line(note);
    }

    // Explain, in the visible install log, why an older TheRock ROCm is chosen
    // when this host reports a newer legacy ROCm (the ticket's 7.14 case). The
    // same note is recorded in the summary block above; surfacing it here keeps
    // it from being buried at the end of a long key/value dump.
    if let Some(host_version) = host_version_newer.as_deref() {
        progress_line(format!(
            "Note: this host reports ROCm {host_version}, but ROCm {resolved} is the newest TheRock ROCm with a matching PyTorch stack for this repository; installing {resolved} (pass `--version <VERSION>` to override).",
            resolved = runtime_version_display(&resolution.latest_version)
        ));
    }

    // Past the preview, the plan has to be installable. A runtime composed
    // without its exact device payload loads and then faults on the first
    // kernel, so an undetermined target is refused here rather than papered
    // over with every published payload.
    //
    // Checked *before* the consent gate: an install that cannot work should say
    // so, not first demand a consent flag for it. With the order reversed, a
    // non-interactive host with an unresolvable target reports only "re-run with
    // --approve-replacing-active-default", and supplying it just surfaces this
    // error instead.
    if let Some(reason) = device_target.reason() {
        bail!(
            "cannot compose a canonical TheRock {} runtime: {reason}.\n\
             The aggregate `rocm` distribution ships no GPU backend unless an exact `device-<target>` extra requests one, so this install would produce a runtime that cannot run a kernel.\n\
             Re-run `rocm install sdk` on the target host, or preview the plan with `--dry-run`.\n\n{}",
            channel.as_str(),
            detect_host_gpu_diagnostics()
        );
    }

    // Installs with no active default runtime proceed with just an informational
    // line. Only an install that would displace the current active default asks
    // for confirmation, and it asks regardless of family or channel because
    // activation is global. With no terminal to answer the prompt it refuses
    // instead, naming `--approve-replacing-active-default` as the
    // non-interactive approval — see `refuse_non_interactive_message` for why
    // that flag and not `--yes`.
    let existing = active_default_runtime_relation(
        paths,
        channel,
        &resolution.family,
        &resolution.latest_version,
    )?;
    match sdk_install_approval(existing.is_some(), consent, interactive_terminal()) {
        SdkInstallApproval::ProceedFresh => {
            progress_line(fresh_install_line(
                &runtime_version_display(&resolution.latest_version),
                &resolution.family,
            ));
        }
        SdkInstallApproval::ProceedApproved(source) => {
            progress_line(preapproved_install_line(
                source,
                existing.as_deref().unwrap_or_default(),
                &runtime_version_display(&resolution.latest_version),
            ));
        }
        SdkInstallApproval::PromptOverwrite => {
            if !confirm_overwrite_existing_sdk(
                channel,
                &resolution.family,
                &resolution.latest_version,
                existing.as_deref().unwrap_or_default(),
            )? {
                let _ = writeln!(
                    output,
                    "  status: cancelled by user; the existing ROCm SDK was left unchanged"
                );
                return Ok(SdkInstallResult::plan(output));
            }
        }
        SdkInstallApproval::RefuseNonInteractive => {
            bail!(refuse_non_interactive_message(
                existing.as_deref().unwrap_or_default()
            ));
        }
    }

    let uv = ensure_uv_binary(paths)?;
    fs::create_dir_all(
        install_root
            .parent()
            .context("runtime install root has no parent directory")?,
    )?;
    progress_line(format!(
        "Creating Python environment at {}.",
        install_root.display()
    ));
    ensure_uv_venv(paths, &uv, &python_launcher.executable, &install_root)?;
    let env_python = venv_python_path(&install_root);

    progress_line(format!(
        "Installing {} from {}",
        wheel_composition.package_specs.join(" "),
        resolution.index_url
    ));
    let mut install_args = uv_pip_install_base(&env_python);
    install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
    if matches!(channel, TheRockChannel::Nightly) {
        install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    install_args.extend(wheel_composition.package_specs.iter().cloned());
    run_uv_progress_command(
        paths,
        &uv,
        install_args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        // Read back out of the specs `uv` is being handed on this very call
        // rather than off `include_devel` a second time, so the line a user
        // watches cannot name a toolchain the install is not requesting.
        if wheel_composition_includes_devel(Some(&wheel_composition)).unwrap_or(include_devel) {
            "install TheRock SDK with the compiler toolchain, torch stack, and resolved dependencies"
        } else {
            "install TheRock SDK, torch stack, and resolved dependencies"
        },
    )?;

    progress_line("Checking the installed ROCm SDK...");
    let rocm_sdk_probe =
        probe_rocm_sdk_runtime_for_target(&env_python, Some(device_target.as_str()))
            .context("TheRock packages did not expose a usable rocm_sdk runtime")?;
    validate_rocm_sdk_runtime_probe(&rocm_sdk_probe)?;
    let installed_version = rocm_sdk_probe
        .rocm_sdk_version
        .clone()
        .unwrap_or_else(|| resolution.latest_version.clone());
    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), resolution.family),
        channel: channel.as_str().to_owned(),
        format: "wheel".to_owned(),
        family: resolution.family.clone(),
        family_source: resolution.family_source.clone(),
        version: installed_version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: resolution.index_url.clone(),
        source_layout_generation: Some(resolution.layout.generation().to_owned()),
        index_url: Some(resolution.index_url.clone()),
        tarball_file_name: None,
        python_launcher: Some(python_launcher.executable.display().to_string()),
        python_executable: Some(env_python.display().to_string()),
        pip_cache_dir: None,
        rocm_sdk: Some(rocm_sdk_probe.clone()),
        sdk_torch: Some(resolution.package_versions.torch.clone()),
        wheel_composition: Some(wheel_composition),
        read_only: false,
        imported_from: None,
        devel: include_devel,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(
        output,
        "  installed_version: {}",
        runtime_version_display(&installed_version)
    );
    let _ = writeln!(output, "  python_executable: {}", env_python.display());
    if let Some(site_packages) = rocm_sdk_probe.site_packages.as_ref() {
        let _ = writeln!(output, "  site_packages: {}", site_packages.display());
    }
    if let Some(root_path) = rocm_sdk_probe.root_path.as_ref() {
        let _ = writeln!(output, "  rocm_sdk_root: {}", root_path.display());
    }
    if let Some(bin_path) = rocm_sdk_probe.bin_path.as_ref() {
        let _ = writeln!(output, "  rocm_sdk_bin: {}", bin_path.display());
    }
    if let Some(version) = rocm_sdk_probe.rocm_sdk_version.as_deref() {
        let _ = writeln!(
            output,
            "  rocm_sdk_version: {}",
            runtime_version_display(version)
        );
    }
    if let Some(target_family) = rocm_sdk_probe.resolved_target_family.as_deref() {
        let _ = writeln!(output, "  rocm_sdk_target_family: {target_family}");
    }
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(SdkInstallResult::installed(output))
}

/// Quotes `value` for the dry-run preview line only -- it is never fed to a
/// real subprocess. Deliberately kept separate from `rocm_core::quoting`
/// (ROCMAI-727); see that module's doc comment for why.
fn quote_display_arg(value: &str) -> String {
    if value.is_empty()
        || value
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '[' | ']' | '(' | ')' | '&' | ';' | '|'))
    {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}

/// Describe the managed runtime this install would displace as the active
/// default: the one the runtime config's `active_runtime_key` (or an
/// unambiguous `default_runtime_id`) currently resolves to. Returns `None` only
/// when neither config pointer points at an active default — a genuinely fresh
/// install, where the new runtime takes a slot nothing occupies and there is
/// nothing to consent to.
///
/// Deliberately NOT scoped to the target family and channel. Activation is
/// global: `finalize_successful_sdk_install` activates whatever was just
/// installed regardless of family or channel, so `rocm install sdk --family
/// gfx120X-all` displaces an active `gfx110X-all` runtime just as surely as a
/// same-family upgrade does. A family/channel-scoped gate would wave exactly
/// that case through unconfirmed while every message promised otherwise.
///
/// Errors reading the manifest directory or the config are propagated rather
/// than treated as "no active default": silently falling back to a
/// fresh-install verdict on a read error would skip the confirmation gate
/// precisely when we are least sure what is currently active.
///
/// The same reasoning covers the ways an active default can fail to resolve
/// without any I/O error at all, and all of them used to reach the
/// fresh-install verdict. `current_runtime_manifest` resolves through two
/// pointers — `active_runtime_key` first, then `default_runtime_id` — and each
/// has its own failure shapes:
///
/// * a registry manifest that reads fine but does not deserialize —
///   `load_runtime_manifests` drops those silently, so a manifest written by an
///   older binary (`family_source`, `selected_artifact_url` and
///   `installed_at_unix_ms` carry no `#[serde(default)]`) vanishes from the
///   list and `current_runtime_manifest` misses;
/// * `active_runtime_key` naming a runtime whose manifest is not in the
///   registry at all;
/// * `default_runtime_id` matching no installed manifest — `runtime_id` is not
///   version-scoped, and `rocm config set-default-runtime` stores whatever it
///   is handed without validating it against the registry;
/// * `default_runtime_id` matching more than one installed manifest, which is
///   the ordinary state once two versions of the same family are installed,
///   because they share the one `therock-<channel>:<family>` id.
///   `current_runtime_manifest` resolves only an exactly-one match, so both the
///   zero-match and the multi-match shapes arrive here.
///
/// Every one of these is something `rocm runtimes list` already calls out, as
/// `active_status: missing manifest for active_runtime_key=...`,
/// `active_status: missing manifest for active_runtime_id=...` or
/// `active_status: ambiguous runtime_id=...`, so proceeding as a fresh install
/// would have one CLI assert both that a runtime is active and that none is.
/// These fail closed into the consent gate rather than into a hard error, so
/// `--approve-replacing-active-default` (or `--yes`) still gets an operator
/// through a stale or ambiguous config.
fn active_default_runtime_relation(
    paths: &AppPaths,
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
) -> Result<Option<String>> {
    let (manifests, unparsed) = load_runtime_manifests_reporting_unparsed(paths)?;
    let config = RocmCliConfig::load(paths)?;
    let Some(active) = crate::runtime_registry::current_runtime_manifest(&config, &manifests)
    else {
        return Ok(unresolved_active_default_relation_text(
            config.active_runtime_key.as_deref(),
            config.default_runtime_id.as_deref(),
            crate::runtime_registry::default_runtime_id_matches(&config, &manifests).len(),
            &unparsed,
        ));
    };
    Ok(Some(active_default_relation_text(
        active,
        channel,
        family,
        resolved_version,
    )))
}

/// Wording for the fail-closed half of [`active_default_runtime_relation`]:
/// nothing resolved, but the on-disk state says something should have.
///
/// `None` here is the only genuinely fresh verdict, and it needs *neither*
/// config pointer to claim an active default: with no `active_runtime_key` and
/// no `default_runtime_id`, neither pointer asserts that a runtime is active,
/// so an unparsable manifest is a registry wart rather than a displacement
/// risk and demanding a consent flag would be a false positive on a genuinely
/// fresh install. Once a pointer does claim something, anything unresolved
/// names what could not be resolved, because the relation string is what the
/// prompt, the preapproved progress line and the non-interactive refusal all
/// print, and "unknown" is the honest answer the operator needs to see.
///
/// `default_runtime_id_match_count` is the number of installed manifests whose
/// `runtime_id` equals `default_runtime_id`. The caller only reaches this
/// function when resolution failed, so that count is 0 (dangling) or greater
/// than 1 (ambiguous) — an exactly-one match is what
/// `current_runtime_manifest` resolves successfully.
///
/// `unparsed` is reported two different ways on purpose. An entry is only the
/// *cause* of an unresolved `active_runtime_key` when it is that key's own
/// manifest; every other unreadable entry is a separate registry wart that
/// happens to be visible at the same time, so it is appended as a suffix rather
/// than named as the reason. Blaming an unrelated file — the ordinary
/// older-binary manifest is exactly that — would point the operator at the
/// wrong path while a deleted runtime went unmentioned.
///
/// Both config pointers are named when both are set. `current_runtime_manifest`
/// tries `active_runtime_key` first and falls through to `default_runtime_id`,
/// so arriving here means *both* failed, and the message is what the prompt,
/// the preapproved progress line and the non-interactive refusal print: it has
/// to name everything that could not be resolved, not just the first pointer.
fn unresolved_active_default_relation_text(
    active_runtime_key: Option<&str>,
    default_runtime_id: Option<&str>,
    default_runtime_id_match_count: usize,
    unparsed: &[PathBuf],
) -> Option<String> {
    let unparsed_text = || {
        unparsed
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let unparsed_suffix = || {
        if unparsed.is_empty() {
            String::new()
        } else {
            format!("; unreadable runtime manifests: {}", unparsed_text())
        }
    };
    let non_empty = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    // The registry stores each manifest at `<runtime_key>.json`
    // (`runtime_manifest_path`), so the file stem is the key. Compared
    // case-insensitively because that is how `current_runtime_manifest` matches
    // `active_runtime_key` against `runtime_key`.
    let key_manifest_is_unparsable = |key: &str| {
        unparsed.iter().any(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.eq_ignore_ascii_case(key))
        })
    };
    // Reached only when `active_runtime_key` is also set and also unresolved, so
    // this is always a continuation of the key clause, never a sentence by
    // itself. The count is 0 or >1 for the reason given on the parameter.
    //
    // Pinned rather than left in prose because the else branch below states
    // "no installed runtime manifest matches it" as fact: at exactly 1 that
    // sentence is false, and the operator would be told nothing matched the
    // recorded default while a manifest did — sending them to re-register a
    // runtime that is already there.
    debug_assert_ne!(
        default_runtime_id_match_count, 1,
        "an exactly-one match is what `current_runtime_manifest` resolves, so this function is unreachable with it"
    );
    let also_unresolved_id = |id: &str| {
        if default_runtime_id_match_count > 1 {
            format!(
                "; the recorded default runtime_id `{id}` does not settle it either, because {default_runtime_id_match_count} installed runtime manifests match it"
            )
        } else {
            format!(
                "; the recorded default runtime_id `{id}` does not settle it either, because no installed runtime manifest matches it"
            )
        }
    };

    match (non_empty(active_runtime_key), non_empty(default_runtime_id)) {
        (Some(key), id) => {
            let cause = if key_manifest_is_unparsable(&key) {
                format!("recorded as `{key}`, but its manifest could not be read")
            } else {
                format!("recorded as `{key}`, but no installed runtime manifest matches it")
            };
            Some(format!(
                "{cause}, so what is currently active cannot be determined{}{}",
                id.as_deref().map(also_unresolved_id).unwrap_or_default(),
                unparsed_suffix()
            ))
        }
        (None, Some(id)) if default_runtime_id_match_count > 1 => Some(format!(
            "recorded as runtime_id `{id}`, which {default_runtime_id_match_count} installed runtime manifests match, so which one is currently active cannot be determined{}",
            unparsed_suffix()
        )),
        (None, Some(id)) => Some(format!(
            "recorded as runtime_id `{id}`, but no installed runtime manifest matches it, so what is currently active cannot be determined{}",
            unparsed_suffix()
        )),
        (None, None) => None,
    }
}

/// Pure wording for [`active_default_runtime_relation`].
///
/// Two shapes, because the two cases are not the same event. When the active
/// default is the same family and channel the install targets, the version
/// comparison is meaningful and the user wants to read "upgrade"/"downgrade"/
/// "reinstall". When it is a different family or channel, comparing versions
/// would invent a relation between two unrelated runtimes, so the text names
/// what is actually being displaced instead.
fn active_default_relation_text(
    active: &InstalledRuntimeManifest,
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
) -> String {
    if active.family == family && active.channel == channel.as_str() {
        // "reinstall" is a claim about versions, not strings: `7.0.0-rc1` and
        // `7.0.0rc1` are one version and must not read as an upgrade. When
        // either side cannot be identified (`version_relation` is `None`) there
        // is no better word available here, so the total order still decides
        // rather than inventing a fourth one; an unreadable version sorts below
        // every readable one, so it reads as the older side.
        let relation = match version_relation(resolved_version, &active.version)
            .unwrap_or_else(|| compare_version_strings(resolved_version, &active.version))
        {
            Ordering::Greater => "upgrade",
            Ordering::Less => "downgrade",
            Ordering::Equal => "reinstall",
        };
        format!(
            "{relation} from installed {installed} ({key})",
            installed = runtime_version_display(&active.version),
            key = active.runtime_key
        )
    } else {
        format!(
            "replaces active default {installed} for family {active_family} on the {active_channel} channel ({key})",
            installed = runtime_version_display(&active.version),
            active_family = active.family,
            active_channel = active.channel,
            key = active.runtime_key
        )
    }
}

/// The host's legacy/system ROCm version when it is strictly newer than the
/// version rocm-cli is about to install, otherwise `None`. Used to explain why a
/// seemingly older wheel version is selected over the host's ROCm.
fn host_rocm_version_newer_than(resolved_version: &str) -> Option<String> {
    host_version_newer_than(detect_legacy_rocm_summary().version, resolved_version)
}

/// Pure core of [`host_rocm_version_newer_than`]: given the host's detected ROCm
/// version (if any) and the version about to be installed, return the host
/// version only when it is strictly newer. Split out from the filesystem probe so
/// the newer-than decision is unit-testable without a real legacy ROCm on disk.
fn host_version_newer_than(host_version: Option<String>, resolved_version: &str) -> Option<String> {
    let host_version = host_version?;
    // Only claim the host is newer when BOTH versions parse and the host is
    // strictly greater. Only a string that fails to parse is "can't tell", and
    // "can't tell" is never "newer". Falling back to a lexicographic compare
    // here wrongly ranks e.g. `7.2.4-98` above `7.13.0` (because '2' > '1' at
    // the third char), inventing a host-newer note that misleads the user.
    //
    // Each side is read with the parser for where it came from.
    // `parse_host_version` additionally drops the packaging build number a
    // legacy ROCm reports (`7.2.4-98` -> 7.2.4); a two-component report
    // (`7.4` -> 7.4.0) and a `-rc1` pre-release are read the same way by both.
    // `resolved_version` is not a host string: it is the version resolved from
    // the index or the tarball catalogue, so it is read exactly as the
    // comparator that picked it reads it.
    let host_parsed = parse_host_version(&host_version)?;
    let resolved_parsed = parse_version_for_ordering(resolved_version)?;
    (host_parsed > resolved_parsed).then_some(host_version)
}

/// The wheel-path `version_note` body explaining why a host-newer legacy ROCm is
/// passed over for the newest TheRock version that still has a matching PyTorch
/// stack. Pure so the exact user-facing wording is unit-testable.
fn wheel_host_version_note(host_version: &str, resolved_display: &str) -> String {
    format!(
        "this host reports ROCm {host_version}, but {resolved_display} is the newest TheRock ROCm with a matching PyTorch stack, so it is selected; pass `--version <VERSION>` to override"
    )
}

/// The tarball-path `version_note` body explaining why a host-newer legacy ROCm is
/// passed over for the newest TheRock tarball for this GPU family.
fn tarball_host_version_note(host_version: &str, resolved_display: &str) -> String {
    format!(
        "this host reports ROCm {host_version}, but {resolved_display} is the newest TheRock ROCm tarball for this GPU family, so it is selected"
    )
}

/// The `warning` body surfaced when the repository's newest version has no
/// installable PyTorch wheels for this Python/platform, so an older one is used.
fn no_wheel_warning_message(newest: &str, resolved_display: &str) -> String {
    format!(
        "ROCm {newest} is the newest version in this repository but has no installable PyTorch wheels for this Python and platform; installing ROCm {resolved_display} instead"
    )
}

/// The repo's newest version when it is strictly newer than the version we are
/// about to install — i.e. the newest release exists in the index but has no
/// installable PyTorch wheels for this Python/platform, so an older one is used.
/// Returns `None` when the newest version is the one being installed (or is
/// unknown), so the caller only warns when there is a real gap.
fn repo_version_without_wheels(
    newest_repo: Option<&str>,
    resolved_version: &str,
) -> Option<String> {
    let newest = newest_repo?;
    // "Is there a genuinely newer version?", not "which string sorts higher?".
    // Two spellings of one version would otherwise produce a warning naming the
    // same version on both sides of "installing X instead", and a pair this
    // tool cannot identify would produce one it cannot justify.
    matches!(
        version_relation(newest, resolved_version),
        Some(Ordering::Greater)
    )
    .then(|| newest.to_owned())
}

/// Where an already-granted approval for displacing the active default came
/// from. Carried rather than collapsed to a bare bool so the line the CLI
/// prints can name the real source: `rocm update --apply` takes its approval
/// from the runtime the user selected, not from a flag, so a message crediting
/// `--yes` would name an approval that was never given. (`rocm update` does
/// accept a `--yes` flag, but it is inert by its own documentation — applying
/// never prompts — so it grants nothing to credit.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdkInstallApprovalSource {
    /// The user passed `--yes` to `rocm install sdk`, which also approves
    /// installing required system packages with `sudo`.
    AssumeYes,
    /// `--approve-replacing-active-default` was passed: the narrow consent, and
    /// the one ROCm CLI's own terminal-less surfaces (chat, MCP, `rocmd`, the
    /// dashboard, onboarding) inject. Kept distinct from `AssumeYes` so the
    /// install log names the flag that was actually given — those surfaces never
    /// pass `--yes`, and a line crediting it would tell a reader that consent to
    /// run `sudo` had been granted when it was not.
    ApproveReplacingActiveDefault,
    /// `rocm update --apply`, which owns its own consent: the user named the
    /// runtime to update, so replacing it is the operation asked for. Non-
    /// interactive by construction, so it must never reach a prompt.
    ///
    /// `activates` records whether `--activate` was given. Only then does the
    /// new install become the active default; without it `apply_runtime_update`
    /// leaves the current default alone and prints a `runtimes activate` hint.
    UpdateApply { activates: bool },
}

/// How consent for displacing the active default runtime is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdkInstallConsent {
    /// Ask the user: prompt when a terminal is attached, refuse when not.
    Ask,
    /// Already granted before the install started, by the named source.
    Preapproved(SdkInstallApprovalSource),
}

/// Whether a real SDK install needs the user's approval before it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SdkInstallApproval {
    /// No active default runtime — nothing is displaced, so install without asking.
    ProceedFresh,
    /// An active default runtime exists and consent was already granted —
    /// displace it without asking, crediting the source.
    ProceedApproved(SdkInstallApprovalSource),
    /// An active default runtime exists and there is a terminal — prompt before
    /// displacing it.
    PromptOverwrite,
    /// An active default runtime exists but there is no terminal and no
    /// preapproval — refuse.
    RefuseNonInteractive,
}

/// Decide whether an SDK install proceeds, prompts, or is refused. Installs with
/// no active default runtime to displace (`existing == false`) always proceed;
/// displacing the active default needs consent, and outside an interactive
/// terminal that consent must have been granted up front.
const fn sdk_install_approval(
    existing: bool,
    consent: SdkInstallConsent,
    interactive: bool,
) -> SdkInstallApproval {
    match (existing, consent) {
        (false, _) => SdkInstallApproval::ProceedFresh,
        (true, SdkInstallConsent::Preapproved(source)) => {
            SdkInstallApproval::ProceedApproved(source)
        }
        (true, SdkInstallConsent::Ask) if interactive => SdkInstallApproval::PromptOverwrite,
        (true, SdkInstallConsent::Ask) => SdkInstallApproval::RefuseNonInteractive,
    }
}

/// The progress line printed when an install displaces the active default
/// without asking. Pure so the exact wording is pinned by unit tests.
///
/// Each arm states only what that caller will actually do. `update --apply`
/// without `--activate` installs beside the active default and leaves it alone,
/// so claiming the new install "becomes the active default runtime" there would
/// be false — and crediting `--yes` would claim an approval the update path
/// never received, since `rocm update`'s `--yes` is inert and its approval
/// comes from the runtime selection instead.
///
/// `pub(crate)` so the `Update` subcommand's own test can pin that invariant at
/// the site of the flag it must not credit.
pub(crate) fn preapproved_install_line(
    source: SdkInstallApprovalSource,
    relation: &str,
    resolved_display: &str,
) -> String {
    match source {
        SdkInstallApprovalSource::AssumeYes => format!(
            "Approved by --yes: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::ApproveReplacingActiveDefault => format!(
            "Approved by --approve-replacing-active-default: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::UpdateApply { activates: true } => format!(
            "Requested by `rocm update --apply --activate`: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::UpdateApply { activates: false } => format!(
            "Requested by `rocm update --apply`: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display} alongside it. The active default runtime is unchanged; re-run with --activate, or use `rocm runtimes activate`, to switch to it."
        ),
    }
}

/// The progress line printed when nothing is displaced. States only that no
/// active default exists, because whether this install *becomes* the active
/// default depends on the caller: `rocm install sdk` activates what it
/// installed, `rocm update --apply` without `--activate` does not.
fn fresh_install_line(resolved_display: &str, family: &str) -> String {
    format!(
        "No active ROCm SDK runtime is configured; installing ROCm SDK {resolved_display} for family {family}."
    )
}

/// The error raised when the active default would be displaced but there is no
/// terminal to confirm it and no preapproval.
///
/// This is the only message a script or CI job sees when it hits this gate, so
/// it names the narrow flag first: that is the whole consent the caller needs
/// here, and recommending `--yes` instead would hand an unattended caller the
/// second consent it carries — approval to install system packages with `sudo`,
/// whose password prompt a job with no terminal cannot answer. `--yes` is still
/// named, because a user at a terminal who wants both should not have to
/// discover it elsewhere.
fn refuse_non_interactive_message(relation: &str) -> String {
    format!(
        "an existing ROCm SDK is the active default runtime ({relation}); continuing would make the newly installed ROCm the active default runtime instead. Re-run with --approve-replacing-active-default to approve this non-interactively, for example `rocm install sdk --approve-replacing-active-default`. Use --yes instead only if you also want to approve installing required system packages with sudo, which needs a terminal to answer a password prompt"
    )
}

/// Interactive confirmation gate for displacing the active default managed
/// runtime. Prints what would be replaced, then reads a yes/no answer from
/// stdin. Only reached when an active default runtime exists, consent was not
/// preapproved, and a terminal is attached (see `sdk_install_approval`).
///
/// "Displacing" rather than "overwriting" is deliberate: in the default managed
/// install root `runtime_key` embeds the resolved version, so an upgrade or
/// downgrade lands in its own install root with its own manifest and the previous
/// install stays on disk — what changes is which runtime is the active default.
/// Only a same-version reinstall reuses the same root. The prompt says so instead
/// of claiming a deletion that does not happen.
///
/// `--prefix` is the exception and the prompt does not claim otherwise:
/// `resolved_install_root` uses the given folder verbatim for every version, so
/// a second install into one prefix does replace the first in place (and
/// `ensure_uv_venv` will `remove_dir_all` it outright if the existing venv's
/// python no longer answers `--version`). The gate is unchanged either way —
/// what is being consented to is the change of active default, not a deletion.
fn confirm_overwrite_existing_sdk(
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
    relation: &str,
) -> Result<bool> {
    println!("sdk install: an existing ROCm SDK is the active default runtime");
    println!("  active default runtime: {relation}");
    println!(
        "  replacing with: ROCm {} for family {family} ({} channel)",
        runtime_version_display(resolved_version),
        channel.as_str()
    );
    println!(
        "  effect: this install becomes the active default runtime, overriding the current default"
    );
    // The host-newer ROCm explanation is already emitted as a visible progress
    // line before this prompt on the real install path, so it is not repeated
    // here.
    prompt_yes_no("Replace the existing ROCm SDK as the active default? [y/N]: ")
}

fn prompt_yes_no(prompt: &str) -> Result<bool> {
    rocm_core::confirm::print_prompt(&mut std::io::stdout(), prompt)
        .context("failed to write confirmation prompt")?;
    rocm_core::confirm::read_answer(&mut std::io::stdin().lock())
        .context("failed to read confirmation response")
}

#[allow(clippy::too_many_arguments)]
fn install_tarball_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    family_override: Option<&str>,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
    dry_run: bool,
    consent: SdkInstallConsent,
) -> Result<SdkInstallResult> {
    let artifact = resolve_tarball_artifact(
        paths,
        channel,
        family_override,
        version_selector,
        layout_override,
    )?;
    let runtime_key = runtime_key(
        channel,
        "tarball",
        &artifact.family,
        Some(&artifact.version),
    );
    let install_root = resolved_install_root(paths, "tarball", &runtime_key, prefix);
    let manifest_path = runtime_manifest_path(paths, &runtime_key);
    let cache_path = paths.cache_dir.join("therock").join(&artifact.file_name);

    let mut output = String::new();
    let _ = writeln!(output, "sdk install");
    render_canonical_provenance(
        &mut output,
        channel,
        &artifact.catalog_url,
        artifact.layout.generation(),
        &artifact.version,
    );
    let _ = writeln!(output, "  format: tarball");
    let _ = writeln!(output, "  family: {}", artifact.family);
    let _ = writeln!(output, "  family_source: {}", artifact.family_source);
    let _ = writeln!(output, "  tarball: {}", artifact.file_name);
    let _ = writeln!(output, "  tarball_url: {}", artifact.url);
    let _ = writeln!(
        output,
        "  latest_version: {}",
        runtime_version_display(&artifact.version)
    );
    // Probed once and reused by the progress line below; see the wheel path.
    let host_version_newer = host_rocm_version_newer_than(&artifact.version);
    if let Some(host_version) = host_version_newer.as_deref() {
        let _ = writeln!(
            output,
            "  version_note: {}",
            tarball_host_version_note(host_version, &runtime_version_display(&artifact.version))
        );
    }
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  cache_path: {}", cache_path.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(SdkInstallResult::plan(output));
    }

    // Mirror the wheel path: surface the host-newer ROCm explanation as a visible
    // line so "why this version and not the host's newer ROCm" is in the install
    // log rather than only in the trailing summary block.
    if let Some(host_version) = host_version_newer.as_deref() {
        progress_line(format!(
            "Note: this host reports ROCm {host_version}, but ROCm {resolved} is the newest TheRock ROCm tarball for this GPU family; installing {resolved}.",
            resolved = runtime_version_display(&artifact.version)
        ));
    }

    // Same gate as the wheel path: only an install that would displace the
    // current active default runtime asks for confirmation, and it asks
    // regardless of family or channel because activation is global.
    let existing =
        active_default_runtime_relation(paths, channel, &artifact.family, &artifact.version)?;
    match sdk_install_approval(existing.is_some(), consent, interactive_terminal()) {
        SdkInstallApproval::ProceedFresh => {
            progress_line(fresh_install_line(
                &runtime_version_display(&artifact.version),
                &artifact.family,
            ));
        }
        SdkInstallApproval::ProceedApproved(source) => {
            progress_line(preapproved_install_line(
                source,
                existing.as_deref().unwrap_or_default(),
                &runtime_version_display(&artifact.version),
            ));
        }
        SdkInstallApproval::PromptOverwrite => {
            if !confirm_overwrite_existing_sdk(
                channel,
                &artifact.family,
                &artifact.version,
                existing.as_deref().unwrap_or_default(),
            )? {
                let _ = writeln!(
                    output,
                    "  status: cancelled by user; the existing ROCm SDK was left unchanged"
                );
                return Ok(SdkInstallResult::plan(output));
            }
        }
        SdkInstallApproval::RefuseNonInteractive => {
            bail!(refuse_non_interactive_message(
                existing.as_deref().unwrap_or_default()
            ));
        }
    }

    fs::create_dir_all(paths.cache_dir.join("therock"))?;
    fs::create_dir_all(&install_root)?;
    if has_nontrivial_directory_contents(&install_root)? {
        bail!(
            "tarball install target {} is not empty; choose a clean prefix or remove the old extraction first",
            install_root.display()
        );
    }

    if let Some(warning) = preflight_tarball_space(&artifact.url, &cache_path, &install_root)? {
        let _ = writeln!(output, "  {warning}");
    }

    let download_label = format!("Downloading {}…", artifact.file_name);
    let spinner = crate::cli_progress::AnimatedSpinner::start(download_label.clone());
    let download_result = download_file(&artifact.url, &cache_path, &mut |bytes, total| {
        spinner.set_progress(&download_label, bytes, total);
    });
    drop(spinner);
    download_result?;

    let extract_spinner =
        crate::cli_progress::AnimatedSpinner::start(format!("Extracting {}…", artifact.file_name));
    let extract_result = extract_tarball_and_discard_archive(&cache_path, &install_root);
    drop(extract_spinner);
    if let Some(cleanup_warning) = extract_result? {
        progress_line(cleanup_warning);
    }

    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), artifact.family),
        channel: channel.as_str().to_owned(),
        format: "tarball".to_owned(),
        family: artifact.family.clone(),
        family_source: artifact.family_source.clone(),
        version: artifact.version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: artifact.url.clone(),
        source_layout_generation: Some(artifact.layout.generation().to_owned()),
        index_url: None,
        tarball_file_name: Some(artifact.file_name.clone()),
        python_launcher: None,
        python_executable: None,
        pip_cache_dir: None,
        rocm_sdk: None,
        sdk_torch: None,
        wheel_composition: None,
        read_only: false,
        imported_from: None,
        // Tarball artifacts ship the whole SDK; the extra is a wheel concept.
        devel: true,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(output, "  extracted: {}", install_root.display());
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(SdkInstallResult::installed(output))
}

/// Identify a wheel runtime by channel, version, AND the exact composition it
/// was installed from.
///
/// Two installs of the same version that request different device payloads are
/// different runtimes: one can run this host's kernels and the other cannot. A
/// version-only key gave them the same name, so a corrected composition
/// overwrote the old tree in place — the one thing side-by-side installs exist
/// to avoid — and left no way to tell the two apart afterwards.
///
/// The fingerprint is a truncated SHA-256 over the generation and the specs,
/// length-delimited so no regrouping of the same characters collides. Truncation
/// is safe here: this names sibling directories, it does not authenticate them.
fn wheel_runtime_key(
    channel: TheRockChannel,
    version: &str,
    composition: &WheelRuntimeComposition,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(composition.source_layout_generation.as_bytes());
    for package_spec in &composition.package_specs {
        hasher.update([0]);
        hasher.update(package_spec.as_bytes());
    }
    hasher.update([0]);
    hasher.update(b"rocm-sdk-target");
    if let Some(target) = &composition.rocm_sdk_target {
        hasher.update([0]);
        hasher.update(target.as_bytes());
    }
    let fingerprint = format!("{:x}", hasher.finalize());
    slugify(&format!(
        "{}-wheel-multi-arch-{version}-{}",
        channel.as_str(),
        &fingerprint[..16]
    ))
}

const fn platform_tarball_token() -> &'static str {
    if runtime_is_windows() {
        "windows"
    } else {
        "linux"
    }
}

fn runtime_key(
    channel: TheRockChannel,
    format: &str,
    family: &str,
    version: Option<&str>,
) -> String {
    match version {
        Some(version) if !version.trim().is_empty() => {
            slugify(&format!("{}-{format}-{family}-{version}", channel.as_str()))
        }
        _ => slugify(&format!("{}-{format}-{family}", channel.as_str())),
    }
}

fn managed_runtime_root(paths: &AppPaths, format: &str, runtime_key: &str) -> PathBuf {
    paths
        .data_dir
        .join("runtimes")
        .join(format)
        .join(runtime_key)
}

/// Where to build a runtime, resolved to its real location on disk first.
///
/// `install sdk` writes this path into three places that all outlive the command:
/// the registry manifest, the sidecar beside the runtime, and — via `uv` — the
/// `#!` line of every console script in the venv. So the path handed to the
/// installer has to name where the files land, not the route taken to get there.
/// Reaching `data/runtimes` through a symlink is enough to make those differ, and
/// once the link goes the runtime reports itself installed at a folder that is not
/// there while the files sit untouched next door.
///
/// Applies to `--prefix` too, which has the identical failure mode.
///
/// Only the root is resolved. `python_executable` is derived from it and must
/// keep the venv's own `bin/python`, which is itself a symlink to the base
/// interpreter — resolving that would record the system Python and break venv
/// semantics. The adopt path already draws the line in the same place: see
/// `main::adopt_runtime_from_probe`, which canonicalizes the install root next to
/// `absolute_existing_file_path_preserving_symlink` for the interpreter.
fn resolved_install_root(
    paths: &AppPaths,
    format: &str,
    runtime_key: &str,
    prefix: Option<PathBuf>,
) -> PathBuf {
    let requested = prefix.unwrap_or_else(|| managed_runtime_root(paths, format, runtime_key));
    rocm_core::resolve_path_through_symlinks(&requested)
}

fn has_nontrivial_directory_contents(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let entries =
        fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn venv_python_path(install_root: &Path) -> PathBuf {
    runtime_python_executable_in_env(install_root)
}

fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::test_support::workspace_test_artifact_dir;
    use super::test_support::{test_paths, test_runtime_manifest, write_test_runtime_manifest};
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
        // `therock` (this file and its child modules) takes this lock, which
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

    #[test]
    fn download_space_requirement_includes_the_safety_margin() {
        let archive = 2 * 1024 * 1024 * 1024;
        assert_eq!(
            disk_space::with_margin(archive),
            archive + archive / disk_space::SPACE_MARGIN_DIVISOR
        );
        assert!(disk_space::with_margin(archive) > archive);
    }

    #[test]
    fn extraction_estimate_exceeds_the_compressed_archive() {
        let archive = 3 * 1024 * 1024 * 1024;
        let estimate = disk_space::estimated_extracted_size(archive);
        assert!(
            estimate > archive,
            "extraction must reserve headroom beyond the archive: {estimate} vs {archive}"
        );
        assert_eq!(estimate, archive * disk_space::EXTRACTED_SIZE_MULTIPLIER);
    }

    #[test]
    fn write_file_atomically_reports_a_full_disk_clearly() {
        // Exercise the mapping the write path uses, without filling a disk.
        let error = disk_space::map_write_error(
            std::io::Error::from(std::io::ErrorKind::StorageFull),
            Path::new("/cache/rocm.tar.gz.tmp"),
        );
        let text = format!("{error:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn gfx90c_resolves_to_no_family_instead_of_the_datacenter_one() {
        // gfx90c is the Vega iGPU in Renoir/Cezanne APUs. It used to normalize to
        // `gfx90X-dcgpu`, so a plain `rocm install sdk` targeted data-center
        // packages and the error listed that family as a `--family` choice.
        assert_eq!(normalize_therock_family("gfx90c"), None);
    }

    #[test]
    fn canonical_channels_use_only_their_aggregate_streams() {
        let release = resolve_source(TheRockChannel::Release, SourceLayout::Canonical);
        assert_eq!(
            release.wheel_index,
            "https://repo.amd.com/rocm/whl-multi-arch"
        );
        assert_eq!(
            release.tarball_catalog,
            "https://repo.amd.com/rocm/tarball/"
        );
        assert_eq!(release.layout.generation(), "multi-arch-v2");

        let nightly = resolve_source(TheRockChannel::Nightly, SourceLayout::Canonical);
        assert_eq!(
            nightly.wheel_index,
            "https://rocm.nightlies.amd.com/whl-multi-arch"
        );
        assert_eq!(
            nightly.tarball_catalog,
            "https://rocm.nightlies.amd.com/tarball/"
        );
        assert_eq!(nightly.layout.generation(), "multi-arch-v2");
    }

    #[test]
    fn canonical_provenance_reports_required_dry_run_fields() {
        let source = resolve_source(TheRockChannel::Nightly, SourceLayout::Canonical);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Nightly,
            &source.wheel_index,
            source.layout.generation(),
            "10.1.0a20260822",
        );

        assert!(output.contains("channel: nightly"));
        assert!(output.contains("canonical_source: https://rocm.nightlies.amd.com/whl-multi-arch"));
        assert!(output.contains("selected_rocm_version: 10.1.0a20260822"));
        assert!(output.contains("build_date: 2026-08-22"));
        assert!(output.contains("source_layout_generation: multi-arch-v2"));
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

    /// The registry load itself breaks install-time ties on the runtime key,
    /// not only the sort helper it calls: the startup update check and
    /// `runtimes_cmd::newest_installed_runtime_manifest` both take the first
    /// entry `load_runtime_manifests` returns, and neither re-sorts.
    ///
    /// Both pairings of file name and runtime key are written, so whichever
    /// file `read_dir` returns first holds `bbb` in one of them: a load that
    /// kept the read order for a tie picks the wrong runtime there.
    #[test]
    fn registry_load_breaks_install_time_ties_on_the_runtime_key() -> Result<()> {
        for (first_file_key, second_file_key) in [("aaa", "bbb"), ("bbb", "aaa")] {
            let (root, paths) = test_paths("registry-load-tie");
            let registry = runtime_registry_dir(&paths);
            fs::create_dir_all(&registry)?;
            for (file, key) in [("first", first_file_key), ("second", second_file_key)] {
                fs::write(
                    registry.join(format!("{file}.json")),
                    serde_json::to_vec_pretty(&test_runtime_manifest(
                        key,
                        "therock-release:gfx120X-all",
                        2_000,
                    ))?,
                )?;
            }
            let pairing = format!("files first={first_file_key}, second={second_file_key}");

            let keys = load_runtime_manifests(&paths)?
                .into_iter()
                .map(|manifest| manifest.runtime_key)
                .collect::<Vec<_>>();
            assert_eq!(keys, ["aaa", "bbb"], "{pairing}");
            assert_eq!(
                crate::runtimes_cmd::newest_installed_runtime_manifest(&paths)?
                    .map(|manifest| manifest.runtime_key)
                    .as_deref(),
                Some("aaa"),
                "{pairing}"
            );

            let _ = fs::remove_dir_all(&root);
        }
        Ok(())
    }

    /// A data dir whose `runtimes` folder is a symlink to somewhere else, plus the
    /// real folder it points at — both canonical, so an assertion cannot pass or
    /// fail on whether the machine's temp dir is itself behind a link.
    ///
    /// Deliberately not `test_paths`: that builds paths by plain `join` and never
    /// canonicalizes, which is exactly the property under test here.
    #[cfg(unix)]
    fn linked_runtimes_paths(name: &str) -> (PathBuf, AppPaths, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-linked-runtimes-{name}-{}",
            crate::test_support::unique_suffix()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        let root = root.canonicalize().unwrap();
        let real = root.join("data").join("real-runtimes");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join("data").join("runtimes")).unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths, real)
    }

    #[cfg(unix)]
    #[test]
    fn linked_runtimes_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| {
            linked_runtimes_paths("same-label").0
        });
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_a_symlinked_runtimes_folder() {
        // The e2e harness points a scenario's `data/runtimes` at a shared tree this
        // way. Recording the link's path made the runtime name a folder that
        // vanished with the scenario, while the files stayed where they were
        // written (rocm-cli#315).
        let (root, paths, real) = linked_runtimes_paths("resolves");
        let runtime_key = "release-wheel-gfx120x-all-7-14-0";

        let resolved = resolved_install_root(&paths, "wheel", runtime_key, None);

        assert_eq!(resolved, real.join("wheel").join(runtime_key));
        assert!(
            !resolved.starts_with(paths.data_dir.join("runtimes")),
            "the recorded root must not be expressed through the link: {}",
            resolved.display()
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_a_symlinked_prefix() {
        // `--prefix` has the identical failure mode, so it gets the identical fix.
        let (root, paths, real) = linked_runtimes_paths("prefix");
        let prefix = paths.data_dir.join("runtimes").join("chosen-env");

        let resolved = resolved_install_root(&paths, "wheel", "unused-key", Some(prefix));

        assert_eq!(resolved, real.join("chosen-env"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_the_tarball_format_too() {
        let (root, paths, real) = linked_runtimes_paths("tarball");
        let runtime_key = "release-tarball-gfx120x-all-7-14-0";

        let resolved = resolved_install_root(&paths, "tarball", runtime_key, None);

        assert_eq!(resolved, real.join("tarball").join(runtime_key));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn install_root_is_unchanged_when_nothing_is_linked() {
        // Guards against gratuitous rewriting: on an ordinary tree the recorded
        // root must still be the plain layout path, so existing installs and the
        // paths compared against them do not move.
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-plain-runtimes-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        // Resolved rather than canonicalized: on Windows `canonicalize` returns a
        // verbatim `\\?\C:\…` path, so the expectation would carry a prefix the
        // resolver deliberately strips and the test would fail on that rather than
        // on whether the layout path moved.
        let root = rocm_core::resolve_path_through_symlinks(&root);
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let runtime_key = "release-wheel-gfx120x-all-7-14-0";

        assert_eq!(
            resolved_install_root(&paths, "wheel", runtime_key, None),
            managed_runtime_root(&paths, "wheel", runtime_key)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn managed_uv_cache_sits_under_the_data_dir_for_generated_runtime_folders() {
        let (_root, paths) = test_paths("managed-uv-cache");
        let runtime_key = "release-wheel-multi-arch-7-14-0";
        let install_root = managed_runtime_root(&paths, "wheel", runtime_key);
        assert!(install_root.starts_with(&paths.data_dir));
        // Without --prefix the generated runtime folder is itself under the data dir, so
        // the uv cache is reachable from the environment it populates without crossing a
        // mount point.
        assert!(managed_uv_cache_dir(&paths.data_dir).starts_with(&paths.data_dir));
    }

    #[test]
    fn uv_cache_does_not_follow_a_prefix_install_root() {
        // Documents a known gap rather than an intended behavior: `--prefix` relocates
        // install_root only, while the uv cache stays keyed off the data dir. When reaching
        // one from the other crosses a mount point uv falls back to copying — it is the
        // mount, not the filesystem, so a bind mount is enough. Tracked separately; see
        // the `--prefix` non-goal on the PR that introduced the colocation.
        let (_root, paths) = test_paths("prefix-uv-cache");
        let prefix_root = PathBuf::from("/mnt/elsewhere/envs/my-env");
        let cache = managed_uv_cache_dir(&paths.data_dir);

        assert!(
            !cache.starts_with(&prefix_root),
            "cache {} unexpectedly followed the --prefix root",
            cache.display()
        );
        assert!(cache.starts_with(&paths.data_dir));
    }

    #[test]
    fn display_command_quotes_package_extras() {
        assert_eq!(quote_display_arg("package[extra]"), "\"package[extra]\"");
        assert_eq!(
            quote_display_arg("C:\\Program Files\\Python\\python.exe"),
            "\"C:\\Program Files\\Python\\python.exe\""
        );
    }

    #[test]
    fn aggregate_wheel_runtime_key_separates_device_payloads_at_one_version() {
        let base = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx942]==7.14.0".to_owned()],
            rocm_sdk_target: Some("gfx942".to_owned()),
        };
        let other_payload = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx950]==7.14.0".to_owned()],
            rocm_sdk_target: Some("gfx950".to_owned()),
        };
        let other_generation = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v3".to_owned(),
            package_specs: base.package_specs.clone(),
            rocm_sdk_target: base.rocm_sdk_target.clone(),
        };
        let mut legacy_probe_default = base.clone();
        legacy_probe_default.rocm_sdk_target = None;

        let base_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &base);

        // Still names its channel and version: the retention policy and every
        // human reading `runtimes list` group on that prefix.
        assert!(
            base_key.starts_with("release-wheel-multi-arch-7-14-0-"),
            "{base_key}"
        );
        assert_eq!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &base),
            "the same composition must always name the same runtime"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &other_payload),
            "a different device payload is a different runtime, not an overwrite"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &legacy_probe_default,),
            "a runtime probed without the exact SDK target must be repaired side by side"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &other_generation),
            "a source-layout generation change must not reuse the old tree"
        );
    }

    #[test]
    fn stable_provenance_uses_neutral_build_date_wording() {
        let source = resolve_source(TheRockChannel::Release, SourceLayout::Canonical);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Release,
            &source.wheel_index,
            source.layout.generation(),
            "7.14.0",
        );
        assert!(output.contains("build_date: not encoded in stable version"));
        assert!(!output.contains("not published"));
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

    #[test]
    fn windows_v1_rejects_tarball_runtime_format() {
        let error = ensure_install_format_supported_for_platform("tarball", true)
            .unwrap_err()
            .to_string();

        assert!(error.contains("tarball installs are not supported on Windows"));
        assert!(error.contains("rocm install sdk --format wheel"));
        assert!(error.contains("managed wheel virtual environment"));
    }

    #[test]
    fn linux_allows_tarball_runtime_format() {
        ensure_install_format_supported_for_platform("tarball", false).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn install_sdk_rejects_tarball_on_windows_before_resolution() {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-cli-therock-test-{}",
            crate::test_support::unique_suffix()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };

        let error = install_sdk(
            &paths,
            SdkInstallRequest {
                channel: "release",
                format: "tarball",
                dry_run: true,
                consent: SdkInstallConsent::Preapproved(SdkInstallApprovalSource::AssumeYes),
                ..SdkInstallRequest::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("tarball installs are not supported on Windows"));
        assert!(error.contains("rocm install sdk --format wheel"));
    }

    /// Register `manifest` and make it the active default runtime, the way a
    /// completed `install sdk` does.
    fn write_active_test_runtime(
        paths: &AppPaths,
        manifest: &InstalledRuntimeManifest,
    ) -> Result<()> {
        write_test_runtime_manifest(paths, manifest)?;
        let mut config = RocmCliConfig::load(paths)?;
        config.default_runtime_id = Some(manifest.runtime_id.clone());
        config.active_runtime_key = Some(manifest.runtime_key.clone());
        config.save(paths)?;
        Ok(())
    }

    #[test]
    fn active_default_relation_classifies_upgrade_downgrade_reinstall() -> Result<()> {
        let (root, paths) = test_paths("active-default-relation");
        let mut manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        manifest.version = "7.13.0".to_owned();
        write_active_test_runtime(&paths, &manifest)?;

        let upgrade = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(upgrade.starts_with("upgrade from"), "got: {upgrade}");
        assert!(upgrade.contains("7.13.0"));

        let downgrade = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.12.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(downgrade.starts_with("downgrade from"), "got: {downgrade}");

        let reinstall = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.13.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(reinstall.starts_with("reinstall from"), "got: {reinstall}");

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_gates_a_different_family_or_channel() -> Result<()> {
        // The regression this gate exists for. `finalize_successful_sdk_install`
        // activates whatever was installed last regardless of family or channel,
        // so installing gfx120X-all while a gfx110X-all runtime is the active
        // default displaces it. A family/channel-scoped gate reported "no
        // existing SDK" here and let the displacement through unconfirmed.
        let (root, paths) = test_paths("active-default-relation-cross-family");
        let mut manifest = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        manifest.version = "7.13.0".to_owned();
        write_active_test_runtime(&paths, &manifest)?;

        let other_family = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("installing another family must still report the active default it displaces");
        assert!(
            other_family.contains("replaces active default")
                && other_family.contains("gfx110X-all"),
            "got: {other_family}"
        );

        let other_channel = active_default_runtime_relation(
            &paths,
            TheRockChannel::Nightly,
            "gfx110X-all",
            "7.14.0",
        )?
        .expect("installing another channel must still report the active default it displaces");
        assert!(
            other_channel.contains("replaces active default")
                && other_channel.contains("release channel"),
            "got: {other_channel}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_none_without_an_active_default() -> Result<()> {
        // Nothing installed at all, and — the second case — a registered runtime
        // that no config points at. Neither displaces anything, so neither may
        // prompt: an install with no active default runtime is the fresh path.
        let (root, paths) = test_paths("active-default-relation-empty");
        assert!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0"
            )?
            .is_none()
        );

        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &manifest)?;
        assert!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0"
            )?
            .is_none()
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_propagates_manifest_read_errors() -> Result<()> {
        // A manifest entry that cannot be read (here, a directory sitting where a
        // `*.json` manifest file is expected) must surface as an error, not be
        // silently treated as "no active default" — that would skip the
        // confirmation gate exactly when we're least sure what is active.
        let (root, paths) = test_paths("active-default-relation-error");
        let registry_dir = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(registry_dir.join("broken.json"))?;

        let error = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )
        .expect_err("a manifest read failure should be propagated, not swallowed");
        assert!(!error.to_string().is_empty());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_an_unparsable_active_manifest() -> Result<()> {
        // The other half of the same policy, and the half that used to fail open:
        // the manifest file *reads* fine, so nothing errors, but it does not
        // deserialize. `load_runtime_manifests` dropped it silently,
        // `current_runtime_manifest` then missed, and the gate was skipped with a
        // fresh-install verdict while `rocm runtimes list` still reported the
        // runtime as active. This is the older-manifest/newer-binary shape:
        // `family_source` carries no `#[serde(default)]`.
        let (root, paths) = test_paths("active-default-relation-unparsable");
        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_active_test_runtime(&paths, &manifest)?;

        // The helper carries the preconditions that make this the parse path and
        // not the I/O path: the file still reads, and it no longer deserializes.
        let _ = make_test_runtime_manifest_unparsable(&paths, &manifest.runtime_key)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("an unparsable active manifest must not yield a fresh-install verdict");
        assert!(
            relation.contains(&manifest.runtime_key),
            "the relation must name the runtime the config still calls active: {relation}"
        );
        assert!(
            relation.contains("could not be read"),
            "the relation must say why the active default is unknown: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_a_dangling_active_runtime_key() -> Result<()> {
        // Same policy, third shape: nothing is unreadable or unparsable, the
        // config simply names an active runtime the registry has no manifest for.
        // `rocm runtimes list` reports this as
        // `active_status: missing manifest for active_runtime_key=...`, so a
        // fresh-install verdict here would have one CLI assert both that a
        // runtime is active and that none is.
        let (root, paths) = test_paths("active-default-relation-dangling");
        let mut config = RocmCliConfig::load(&paths)?;
        config.active_runtime_key = Some("release-wheel-gfx120X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling active_runtime_key must not yield a fresh-install verdict");
        assert!(
            relation.contains("release-wheel-gfx120X-all"),
            "the relation must name the unresolved key: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_blames_an_unparsable_manifest_only_when_it_is_the_active_one()
    -> Result<()> {
        // A dangling `active_runtime_key` and an unparsable manifest belonging to
        // some *other* runtime are two independent faults that show up together
        // routinely: an older binary's manifest fails `from_slice` against this
        // one while the runtime the config calls active was removed outright.
        // Naming the stranger's file as "its manifest" would send the operator to
        // repair a path that has nothing to do with the problem and never mention
        // the runtime that actually went missing.
        let (root, paths) = test_paths("active-default-relation-unrelated-unparsable");

        let stranger = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &stranger)?;
        let stranger_path = make_test_runtime_manifest_unparsable(&paths, &stranger.runtime_key)?;

        let mut config = RocmCliConfig::load(&paths)?;
        // Nothing on disk is stored under this key, so the only unreadable entry
        // in the registry is the stranger's.
        config.active_runtime_key = Some("release-wheel-gfx120X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling active_runtime_key must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all`, but no installed runtime manifest matches it"
            ),
            "the missing runtime is the cause, not the stranger's manifest: {relation}"
        );
        assert!(
            !relation.contains("its manifest could not be read"),
            "an unrelated unparsable manifest must not be blamed as the active one's: {relation}"
        );
        assert!(
            relation.contains(&format!(
                "; unreadable runtime manifests: {}",
                stranger_path.display()
            )),
            "the unrelated unparsable manifest is still reported, as a suffix: {relation}"
        );

        // The other direction, in the same registry: once the active key's *own*
        // manifest is unparsable, "could not be read" is the right cause even
        // though the stranger's file is unreadable too.
        let active = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            20,
        );
        write_test_runtime_manifest(&paths, &active)?;
        let active_path = make_test_runtime_manifest_unparsable(&paths, &active.runtime_key)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("an unparsable active manifest must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all`, but its manifest could not be read"
            ),
            "the active key's own unparsable manifest is the cause here: {relation}"
        );
        assert!(
            relation.contains(&active_path.display().to_string())
                && relation.contains(&stranger_path.display().to_string()),
            "both unreadable entries are still listed: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_names_both_unresolved_config_pointers() -> Result<()> {
        // `current_runtime_manifest` tries `active_runtime_key` and falls through
        // to `default_runtime_id`, so arriving at the fail-closed path with both
        // set means both failed. `rocm runtimes activate` writes the pair
        // together, so removing that runtime while another version of the family
        // remains strands them together too. Reporting only the key would have
        // the operator repair half the config and hit the gate again.
        let (root, paths) = test_paths("active-default-relation-both-pointers");
        let runtime_id = "therock-release:gfx120X-all";
        let mut older = test_runtime_manifest("release-wheel-gfx120X-all-7130", runtime_id, 10);
        older.version = "7.13.0".to_owned();
        let mut newer = test_runtime_manifest("release-wheel-gfx120X-all-7140", runtime_id, 20);
        newer.version = "7.14.0".to_owned();
        write_test_runtime_manifest(&paths, &older)?;
        write_test_runtime_manifest(&paths, &newer)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.active_runtime_key = Some("release-wheel-gfx120X-all-7120".to_owned());
        config.default_runtime_id = Some(runtime_id.to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("two unresolved pointers must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all-7120`, but no installed runtime manifest matches it"
            ),
            "the relation must name the unresolved key: {relation}"
        );
        assert!(
            relation.contains(&format!(
                "; the recorded default runtime_id `{runtime_id}` does not settle it either, because 2 installed runtime manifests match it"
            )),
            "the relation must also name the ambiguous fallback id: {relation}"
        );

        // The zero-match half of the same pairing: the fallback is dangling
        // rather than ambiguous, and must still be named.
        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some("therock-release:gfx110X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("two unresolved pointers must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "; the recorded default runtime_id `therock-release:gfx110X-all` does not settle it either, because no installed runtime manifest matches it"
            ),
            "the relation must also name the dangling fallback id: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_an_ambiguous_default_runtime_id() -> Result<()> {
        // Same policy, but the *other* resolution path. `current_runtime_manifest`
        // falls back to `default_runtime_id` when `active_runtime_key` is unset,
        // and resolves only an exactly-one match. `runtime_id` is
        // `therock-<channel>:<family>` with no version in it, so two installed
        // versions of one family share it and the fallback returns `None` — while
        // `rocm runtimes list` reports `active_status: ambiguous runtime_id=...`.
        // `rocm config set-default-runtime` reaches this state directly: it stores
        // the id unvalidated and clears `active_runtime_key`.
        let (root, paths) = test_paths("active-default-relation-ambiguous-id");
        let runtime_id = "therock-release:gfx120X-all";
        let mut older = test_runtime_manifest("release-wheel-gfx120X-all-7130", runtime_id, 10);
        older.version = "7.13.0".to_owned();
        let mut newer = test_runtime_manifest("release-wheel-gfx120X-all-7140", runtime_id, 20);
        newer.version = "7.14.0".to_owned();
        write_test_runtime_manifest(&paths, &older)?;
        write_test_runtime_manifest(&paths, &newer)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some(runtime_id.to_owned());
        // Precondition: this is the `default_runtime_id` shape, not the
        // `active_runtime_key` shape the sibling tests already cover.
        config.active_runtime_key = None;
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("an ambiguous default_runtime_id must not yield a fresh-install verdict");
        assert!(
            relation.contains(runtime_id),
            "the relation must name the id that could not be resolved: {relation}"
        );
        assert!(
            relation.contains("2 installed runtime manifests match"),
            "the relation must say the id is ambiguous and how badly: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_a_dangling_default_runtime_id() -> Result<()> {
        // The zero-match half of the same fallback path: `_ => None` in
        // `current_runtime_manifest` swallows "no match" exactly as it swallows
        // "many matches". `rocm config set-default-runtime` does not check the id
        // against the registry, so a typo — or uninstalling the last runtime of a
        // family — leaves the config asserting an active default that is not
        // there, which `rocm runtimes list` reports as
        // `active_status: missing manifest for active_runtime_id=...`.
        let (root, paths) = test_paths("active-default-relation-dangling-id");
        let other = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &other)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some("therock-release:gfx120X-all".to_owned());
        config.active_runtime_key = None;
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling default_runtime_id must not yield a fresh-install verdict");
        assert!(
            relation.contains("therock-release:gfx120X-all"),
            "the relation must name the id that could not be resolved: {relation}"
        );
        assert!(
            relation.contains("no installed runtime manifest matches it"),
            "the relation must say the id resolved to nothing: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_is_fresh_when_no_config_pointer_claims_an_active_default()
    -> Result<()> {
        // The limit of the fail-closed policy. An unparsable registry manifest is
        // only evidence of a *displacement* risk if something claims an active
        // default; with neither `active_runtime_key` nor `default_runtime_id` set,
        // nothing does, and demanding a consent flag would block a genuinely fresh
        // install over an unrelated registry wart.
        let (root, paths) = test_paths("active-default-relation-fresh-unparsable");
        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &manifest)?;
        // The helper carries the preconditions that make this the parse path and
        // not the I/O path: the file still reads, and it no longer deserializes.
        let _ = make_test_runtime_manifest_unparsable(&paths, &manifest.runtime_key)?;

        let config = RocmCliConfig::load(&paths)?;
        assert!(config.active_runtime_key.is_none());
        assert!(config.default_runtime_id.is_none());

        assert_eq!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0",
            )?,
            None,
            "no config pointer claims an active default, so this is a fresh install"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn sdk_install_approval_only_prompts_when_an_active_default_is_displaced() {
        let assume_yes = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::AssumeYes);
        let update_apply = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::UpdateApply {
            activates: false,
        });

        // No active default runtime -> never prompt, regardless of terminal/consent.
        assert_eq!(
            sdk_install_approval(false, SdkInstallConsent::Ask, false),
            SdkInstallApproval::ProceedFresh
        );
        assert_eq!(
            sdk_install_approval(false, SdkInstallConsent::Ask, true),
            SdkInstallApproval::ProceedFresh
        );
        assert_eq!(
            sdk_install_approval(false, assume_yes, false),
            SdkInstallApproval::ProceedFresh
        );

        // Active default present: preapproved consent proceeds and is credited to
        // its real source; a terminal prompts; neither refuses.
        assert_eq!(
            sdk_install_approval(true, assume_yes, false),
            SdkInstallApproval::ProceedApproved(SdkInstallApprovalSource::AssumeYes)
        );
        assert_eq!(
            sdk_install_approval(
                true,
                SdkInstallConsent::Preapproved(
                    SdkInstallApprovalSource::ApproveReplacingActiveDefault
                ),
                false
            ),
            SdkInstallApproval::ProceedApproved(
                SdkInstallApprovalSource::ApproveReplacingActiveDefault
            )
        );
        assert_eq!(
            sdk_install_approval(true, update_apply, false),
            SdkInstallApproval::ProceedApproved(SdkInstallApprovalSource::UpdateApply {
                activates: false
            })
        );
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, true),
            SdkInstallApproval::PromptOverwrite
        );
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );
    }

    #[test]
    fn preapproved_install_line_credits_the_real_consent_source() {
        // `rocm update --apply` draws its approval from the runtime the user
        // selected, not from a flag — its `--yes` is inert — so a line
        // crediting `--yes` names an approval that was never given. And without
        // `--activate`, `apply_runtime_update` leaves the active default alone,
        // so claiming the install "becomes the active default runtime" is false.
        let by_yes = preapproved_install_line(
            SdkInstallApprovalSource::AssumeYes,
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(by_yes.starts_with("Approved by --yes:"), "got: {by_yes}");
        assert!(by_yes.contains("becomes the active default runtime"));

        // The narrow flag is not `--yes`: ROCm CLI's own terminal-less surfaces
        // pass only this one, and a line crediting `--yes` would tell whoever
        // reads the chat or dashboard transcript that consent to run `sudo` was
        // given when it never was.
        let by_narrow = preapproved_install_line(
            SdkInstallApprovalSource::ApproveReplacingActiveDefault,
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            by_narrow.starts_with("Approved by --approve-replacing-active-default:"),
            "got: {by_narrow}"
        );
        assert!(
            !by_narrow.contains("--yes"),
            "the narrow flag must not be credited to --yes: {by_narrow}"
        );
        assert!(by_narrow.contains("becomes the active default runtime"));

        let update_activates = preapproved_install_line(
            SdkInstallApprovalSource::UpdateApply { activates: true },
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            !update_activates.contains("--yes"),
            "the update path must not credit an approval `--yes` did not grant: {update_activates}"
        );
        assert!(update_activates.contains("becomes the active default runtime"));

        let update_only = preapproved_install_line(
            SdkInstallApprovalSource::UpdateApply { activates: false },
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            !update_only.contains("--yes"),
            "the update path must not credit an approval `--yes` did not grant: {update_only}"
        );
        assert!(
            !update_only.contains("becomes the active default runtime"),
            "`update --apply` without --activate does not change the active default: {update_only}"
        );
        assert!(update_only.contains("The active default runtime is unchanged"));
    }

    #[test]
    fn refusal_recommends_the_narrow_flag_before_yes() {
        // This is the only message a script or CI job reads when it hits the
        // gate, and it is the audience the narrow flag exists for. Recommending
        // `--yes` first would hand an unattended caller the second consent that
        // flag carries — approval to install system packages with `sudo` — and
        // park it on a password prompt it has no terminal to answer.
        let message = refuse_non_interactive_message(
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
        );

        let narrow = message
            .find("--approve-replacing-active-default")
            .unwrap_or_else(|| panic!("the refusal must name the narrow flag: {message}"));
        // `--yes` still has to appear: a user at a terminal who wants both
        // consents should not have to go looking for it.
        let yes = message
            .find("--yes")
            .unwrap_or_else(|| panic!("the refusal must still explain --yes: {message}"));
        assert!(
            narrow < yes,
            "the narrow flag must be recommended before --yes: {message}"
        );
        assert!(
            message.contains("system packages"),
            "the refusal must say what --yes additionally approves: {message}"
        );
        assert!(
            message.contains("is the active default runtime"),
            "the refusal must name what would be replaced: {message}"
        );
        assert!(
            !message.to_lowercase().contains("overwrit"),
            "an upgrade leaves the previous install on disk; it replaces the \
             active default rather than overwriting it: {message}"
        );
    }

    #[test]
    fn fresh_install_line_claims_no_absent_sdk() {
        // The fresh path is reached whenever no runtime is the active default,
        // which does not mean no SDK is installed anywhere. Saying "no existing
        // ROCm SDK found" there would be false on a host holding a registered
        // but unactivated runtime.
        let line = fresh_install_line("7.14.0", "gfx120X-all");
        assert!(
            !line.to_lowercase().contains("no existing rocm sdk"),
            "got: {line}"
        );
        assert!(line.contains("No active ROCm SDK runtime is configured"));
        assert!(line.contains("gfx120X-all"));
    }

    #[test]
    fn repo_version_without_wheels_warns_only_when_newest_is_newer() {
        // Newest repo version has no wheels (newer than the installable one) -> warn.
        assert_eq!(
            repo_version_without_wheels(Some("7.14.0"), "7.13.0").as_deref(),
            Some("7.14.0")
        );
        // Newest repo version is the one being installed -> no warning.
        assert!(repo_version_without_wheels(Some("7.13.0"), "7.13.0").is_none());
        // A specific version was requested (no "newest" known) -> no warning.
        assert!(repo_version_without_wheels(None, "7.13.0").is_none());
        // Defensive: an older "newest" (should not happen) never warns.
        assert!(repo_version_without_wheels(Some("7.12.0"), "7.13.0").is_none());
    }

    #[test]
    fn host_version_newer_than_reports_only_a_strictly_newer_host() {
        // Host ROCm is newer than the version being installed -> surface it.
        assert_eq!(
            host_version_newer_than(Some("7.14.0".to_owned()), "7.13.0").as_deref(),
            Some("7.14.0")
        );
        // Host ROCm matches the installed version -> nothing to explain.
        assert!(host_version_newer_than(Some("7.13.0".to_owned()), "7.13.0").is_none());
        // Host ROCm is older than the installed version -> nothing to explain.
        assert!(host_version_newer_than(Some("7.12.0".to_owned()), "7.13.0").is_none());
        // No legacy ROCm detected on the host -> nothing to explain.
        assert!(host_version_newer_than(None, "7.13.0").is_none());

        // A build-suffixed host version must not be lexicographically ranked
        // above the resolved version: `7.2.4-98` is numerically OLDER than
        // `7.13.0`, so no host-newer note. (This is the reported regression:
        // char-compare put `7.2…` above `7.13…`.)
        assert!(host_version_newer_than(Some("7.2.4-98".to_owned()), "7.13.0").is_none());
        // Two-component host reports are parsed as `.0`; still older here.
        assert!(host_version_newer_than(Some("7.4".to_owned()), "7.13.0").is_none());
        assert!(host_version_newer_than(Some("7.9".to_owned()), "7.13.0").is_none());
        // A build suffix on an equal version is not "newer".
        assert!(host_version_newer_than(Some("7.13.0-56".to_owned()), "7.13.0").is_none());
        // A genuinely newer build-suffixed host is surfaced, keeping the
        // original reported string (suffix included) for the user-facing note.
        assert_eq!(
            host_version_newer_than(Some("7.20.1-33".to_owned()), "7.13.0").as_deref(),
            Some("7.20.1-33")
        );
        // A host string we cannot parse is "can't tell", never "newer".
        assert!(host_version_newer_than(Some("unknown".to_owned()), "7.13.0").is_none());
    }

    #[test]
    fn host_version_notes_and_warning_render_the_expected_text() {
        // Wheel path: the note names both versions and offers the --version override.
        let wheel = wheel_host_version_note("7.14.0", "7.13.0");
        assert_eq!(
            wheel,
            "this host reports ROCm 7.14.0, but 7.13.0 is the newest TheRock ROCm with a matching PyTorch stack, so it is selected; pass `--version <VERSION>` to override"
        );

        // Tarball path: same explanation, phrased for the GPU-family tarball.
        let tarball = tarball_host_version_note("7.14.0", "7.13.0");
        assert_eq!(
            tarball,
            "this host reports ROCm 7.14.0, but 7.13.0 is the newest TheRock ROCm tarball for this GPU family, so it is selected"
        );

        // The no-wheels warning names the repo-newest and the fallback it installs.
        let warning = no_wheel_warning_message("7.14.0", "7.13.0");
        assert_eq!(
            warning,
            "ROCm 7.14.0 is the newest version in this repository but has no installable PyTorch wheels for this Python and platform; installing ROCm 7.13.0 instead"
        );
    }

    /// Rewrite an already-written registry manifest so it still *reads* but no
    /// longer deserializes, and return its path. Drops `family_source`, which
    /// carries no `#[serde(default)]` — the real older-binary/newer-binary shape,
    /// not an invented corruption.
    fn make_test_runtime_manifest_unparsable(
        paths: &AppPaths,
        runtime_key: &str,
    ) -> Result<PathBuf> {
        let path = runtime_manifest_path(paths, runtime_key);
        let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        value
            .as_object_mut()
            .expect("manifest is a JSON object")
            .remove("family_source")
            .expect("manifest carries family_source");
        fs::write(&path, serde_json::to_vec_pretty(&value)?)?;
        assert!(
            fs::read(&path).is_ok(),
            "the fixture must still read, or this is the I/O path, not the parse path"
        );
        assert!(
            serde_json::from_slice::<InstalledRuntimeManifest>(&fs::read(&path)?).is_err(),
            "the test fixture must be unparsable, or this asserts nothing"
        );
        Ok(path)
    }
}
