// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ManagedToolConfig, RocmCliConfig, detect_host_gfx_target,
    detect_host_gpu_diagnostics, detect_host_therock_family, detect_managed_therock_family,
    disk_space, ensure_uv_binary, known_therock_families, managed_tools_dir,
    normalize_runtime_path_for_host, normalize_runtime_path_for_storage,
    normalize_runtime_path_text_for_host, normalize_runtime_path_text_for_storage,
    normalize_therock_family, runtime_is_windows, runtime_os_name, runtime_path_for_windows_child,
    runtime_path_list_split, runtime_python_executable_in_env, unix_time_millis, uv_command_env,
    uv_pip_install_base, uv_venv_args, verify_rsa_pkcs1_sha256_signature,
};
#[cfg(test)]
use rocm_core::{
    generate_rsa_signing_keypair, managed_uv_cache_dir, sign_rsa_pkcs1_sha256_signature,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

const THEROCK_NIGHTLY_PIP_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/whl-multi-arch";
const THEROCK_RELEASE_PIP_INDEX_BASE: &str = "https://repo.amd.com/rocm/whl-multi-arch";
const THEROCK_RELEASE_TARBALL_BASE: &str = "https://repo.amd.com/rocm/tarball/";
const THEROCK_NIGHTLY_TARBALL_BASE: &str = "https://rocm.nightlies.amd.com/tarball/";
const THEROCK_SOURCE_LAYOUT_GENERATION: &str = "multi-arch-v2";
const DEFAULT_MANAGED_PYTHON_VERSION: &str = "3.12";
const STARTUP_UPDATE_CHECK_INTERVAL_MS: u128 = 12 * 60 * 60 * 1_000;
const STARTUP_UPDATE_CHECK_TIMEOUT_SECS: u64 = 2;
/// Timeout for the best-effort HEAD probe that sizes a download before starting it.
const THEROCK_HEAD_PROBE_TIMEOUT_SECS: u64 = 10;
/// Whole-transfer budget for an artifact download. A single-digit-gigabyte SDK
/// tarball on a slow link needs well past the ten minutes the metadata fetches
/// use; a retry that resumes cannot help if the attempt itself is cut short.
const THEROCK_DOWNLOAD_TIMEOUT: Duration = Duration::from_hours(1);
/// Largest `Content-Length` accepted as a real SDK tarball size.
///
/// SDK tarballs are single-digit gigabytes; anything past this is a
/// misconfigured proxy or a hostile header rather than a real artifact, and
/// must not be allowed to refuse an install on its own authority.
const THEROCK_MAX_PLAUSIBLE_TARBALL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TheRockChannel {
    Release,
    Nightly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CanonicalSource {
    wheel_index: &'static str,
    tarball_catalog: &'static str,
    layout_generation: &'static str,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum RuntimeVersionSelector {
    Version(String),
    BuildDate(String),
}

impl RuntimeVersionSelector {
    pub(crate) fn version(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            bail!("TheRock version cannot be empty");
        }
        if trimmed
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
        {
            bail!("TheRock version must be a single version string");
        }
        Ok(Self::Version(trimmed.to_owned()))
    }

    pub(crate) fn build_date(value: impl AsRef<str>) -> Result<Self> {
        Ok(Self::BuildDate(normalize_requested_build_date(
            value.as_ref(),
        )?))
    }

    fn describe(&self) -> String {
        match self {
            Self::Version(version) => format!("version {version}"),
            Self::BuildDate(date) => format!("build date {date}"),
        }
    }

    fn matches_version(&self, version: &str) -> bool {
        match self {
            Self::Version(requested) => version == requested,
            Self::BuildDate(date) => runtime_version_build_date(version).as_deref() == Some(date),
        }
    }
}

impl TheRockChannel {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "release" => Ok(Self::Release),
            "nightly" => Ok(Self::Nightly),
            other => bail!("unsupported TheRock channel: {other}"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }

    const fn canonical_source(self) -> CanonicalSource {
        canonical_source(self)
    }
}

const fn canonical_source(channel: TheRockChannel) -> CanonicalSource {
    match channel {
        TheRockChannel::Release => CanonicalSource {
            wheel_index: THEROCK_RELEASE_PIP_INDEX_BASE,
            tarball_catalog: THEROCK_RELEASE_TARBALL_BASE,
            layout_generation: THEROCK_SOURCE_LAYOUT_GENERATION,
        },
        TheRockChannel::Nightly => CanonicalSource {
            wheel_index: THEROCK_NIGHTLY_PIP_INDEX_BASE,
            tarball_catalog: THEROCK_NIGHTLY_TARBALL_BASE,
            layout_generation: THEROCK_SOURCE_LAYOUT_GENERATION,
        },
    }
}

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

fn validate_aggregate_index_layout(html: &str) -> Result<()> {
    let required_packages = ["rocm/", "torch/", "torchvision/", "torchaudio/"];
    if required_packages
        .iter()
        .all(|package| html.contains(package))
    {
        return Ok(());
    }
    bail!(
        "unknown canonical TheRock aggregate index layout: expected package links for rocm, torch, torchvision, and torchaudio"
    )
}

#[derive(Debug, Clone)]
struct FamilyResolution {
    family: String,
    source: String,
}

#[derive(Debug, Clone)]
struct PipRuntimeResolution {
    family: String,
    family_source: String,
    index_url: String,
    latest_version: String,
    package_versions: TheRockPipPackageVersions,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct TheRockPipPackageVersions {
    rocm: String,
    torch: String,
    torchvision: String,
    torchaudio: String,
    compatibility_key: String,
}

#[derive(Debug, Clone)]
struct WheelCompatibility {
    python_tag: String,
    platform_tags: Vec<String>,
}

#[derive(Debug, Clone)]
struct TarballArtifact {
    family: String,
    family_source: String,
    file_name: String,
    version: String,
    url: String,
}

#[derive(Debug, Clone)]
struct CachedHttpText {
    text: String,
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpMetadata {
    url: String,
    #[serde(default)]
    etag: Option<String>,
    #[serde(default)]
    last_modified: Option<String>,
    #[serde(default)]
    signature: Option<CachedHttpSignatureMetadata>,
    fetched_at_unix_ms: u128,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpSignatureMetadata {
    url: String,
    verified_at_unix_ms: u128,
    public_key_source: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct CachedHttpCacheEntry {
    metadata: CachedHttpMetadata,
    body: String,
    #[serde(default)]
    signature_bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
struct MetadataSignaturePolicy {
    required: bool,
    public_key_path: Option<PathBuf>,
    public_key_pem: Option<String>,
}

#[derive(Debug, Clone)]
struct PythonLauncher {
    executable: PathBuf,
    source: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedPythonManifest {
    executable: PathBuf,
    version: String,
    installed_at_unix_ms: u128,
}

#[derive(Debug)]
struct HttpResponseBody {
    status: u16,
    headers: String,
    body: Vec<u8>,
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
    pub target_runtime_key: String,
    pub repair_required: bool,
    pub update_available: bool,
}

/// Exact canonical wheel install intent last applied successfully to a runtime.
///
/// Version alone cannot identify a reusable environment: adding a required ROCm
/// extra at the same release version must make an older cache repairable.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct WheelRuntimeComposition {
    pub source_layout_generation: String,
    pub package_specs: Vec<String>,
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
    wheel_composition: Option<WheelRuntimeComposition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InstalledRuntimeManifest {
    pub runtime_key: String,
    pub runtime_id: String,
    pub channel: String,
    pub format: String,
    pub family: String,
    pub family_source: String,
    pub version: String,
    pub install_root: PathBuf,
    pub selected_artifact_url: String,
    #[serde(default)]
    pub index_url: Option<String>,
    #[serde(default)]
    pub tarball_file_name: Option<String>,
    #[serde(default)]
    pub python_launcher: Option<String>,
    #[serde(default)]
    pub python_executable: Option<String>,
    #[serde(default)]
    pub pip_cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub rocm_sdk: Option<RocmSdkPythonProbe>,
    /// The torch this SDK install wrote, e.g. `2.11.0+rocm7.13.0`.
    ///
    /// Recorded because an engine install later overwrites torch in the same
    /// environment. Reading the environment afterwards tells you what is there
    /// now, not which build belongs to these libraries; only the manifest still
    /// knows that, and it has to survive repeat installs to be worth anything.
    #[serde(default)]
    pub sdk_torch: Option<String>,
    /// Missing on manifests written before composition-aware freshness. Such a
    /// managed wheel runtime is repaired once and rewritten with this field.
    #[serde(default)]
    pub wheel_composition: Option<WheelRuntimeComposition>,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub imported_from: Option<PathBuf>,
    pub installed_at_unix_ms: u128,
}

impl InstalledRuntimeManifest {
    fn normalize_host_paths(mut self) -> Self {
        self.install_root = normalize_manifest_path(self.install_root);
        self.python_launcher = self
            .python_launcher
            .map(|value| normalize_runtime_path_text_for_host(&value));
        self.python_executable = self
            .python_executable
            .map(|value| normalize_runtime_path_text_for_host(&value));
        self.pip_cache_dir = self.pip_cache_dir.map(normalize_manifest_path);
        self.imported_from = self.imported_from.map(normalize_manifest_path);
        if let Some(probe) = self.rocm_sdk.as_mut() {
            probe.normalize_host_paths();
        }
        self
    }

    pub(crate) fn normalize_storage_paths(mut self) -> Self {
        self.install_root = normalize_storage_manifest_path(&self.install_root);
        self.python_launcher = self
            .python_launcher
            .map(|value| normalize_runtime_path_text_for_storage(&value));
        self.python_executable = self
            .python_executable
            .map(|value| normalize_runtime_path_text_for_storage(&value));
        self.pip_cache_dir = self
            .pip_cache_dir
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.imported_from = self
            .imported_from
            .as_deref()
            .map(normalize_storage_manifest_path);
        if let Some(probe) = self.rocm_sdk.as_mut() {
            probe.normalize_storage_paths();
        }
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RocmSdkPythonProbe {
    #[serde(default)]
    pub import_ok: bool,
    #[serde(default)]
    pub rocm_sdk_version: Option<String>,
    #[serde(default)]
    pub site_packages: Option<PathBuf>,
    #[serde(default)]
    pub root_path: Option<PathBuf>,
    #[serde(default)]
    pub bin_path: Option<PathBuf>,
    #[serde(default)]
    pub cmake_path: Option<PathBuf>,
    #[serde(default)]
    pub runtime_roots: Vec<PathBuf>,
    #[serde(default)]
    pub bin_paths: Vec<PathBuf>,
    #[serde(default)]
    pub library_paths: Vec<PathBuf>,
    #[serde(default)]
    pub default_target_family: Option<String>,
    #[serde(default)]
    pub available_target_families: Vec<String>,
    #[serde(default)]
    pub resolved_target_family: Option<String>,
    #[serde(default)]
    pub packages: Vec<RocmSdkPackageProbe>,
    #[serde(default)]
    pub library_shortnames: Vec<String>,
    #[serde(default)]
    pub resolved_libraries: Vec<RocmSdkLibraryProbe>,
    #[serde(default)]
    pub error: Option<String>,
}

impl RocmSdkPythonProbe {
    fn normalize_host_paths(&mut self) {
        self.site_packages = self.site_packages.take().map(normalize_manifest_path);
        self.root_path = self.root_path.take().map(normalize_manifest_path);
        self.bin_path = self.bin_path.take().map(normalize_manifest_path);
        self.cmake_path = self.cmake_path.take().map(normalize_manifest_path);
        self.runtime_roots = std::mem::take(&mut self.runtime_roots)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        self.bin_paths = std::mem::take(&mut self.bin_paths)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        self.library_paths = std::mem::take(&mut self.library_paths)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        for library in &mut self.resolved_libraries {
            library.paths = std::mem::take(&mut library.paths)
                .into_iter()
                .map(normalize_manifest_path)
                .collect();
        }
    }

    fn normalize_storage_paths(&mut self) {
        self.site_packages = self
            .site_packages
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.root_path = self
            .root_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.bin_path = self
            .bin_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.cmake_path = self
            .cmake_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.runtime_roots = self
            .runtime_roots
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        self.bin_paths = self
            .bin_paths
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        self.library_paths = self
            .library_paths
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        for library in &mut self.resolved_libraries {
            library.paths = library
                .paths
                .iter()
                .map(|path| normalize_storage_manifest_path(path))
                .collect();
        }
    }
}

fn normalize_manifest_path(path: PathBuf) -> PathBuf {
    normalize_runtime_path_for_host(&path)
}

fn normalize_storage_manifest_path(path: &Path) -> PathBuf {
    normalize_runtime_path_for_storage(path)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RocmSdkPackageProbe {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RocmSdkLibraryProbe {
    pub shortname: String,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct TarballIndexFile {
    name: String,
    mtime: f64,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct ParsedVersion {
    major: u32,
    minor: u32,
    patch: u32,
    stage: VersionStage,
    stage_number: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum VersionStage {
    Alpha,
    Rc,
    Stable,
}

pub(crate) fn install_sdk(
    paths: &AppPaths,
    channel: &str,
    format: &str,
    prefix: Option<PathBuf>,
    version_selector: Option<RuntimeVersionSelector>,
    family_override: Option<&str>,
    dry_run: bool,
) -> Result<String> {
    let channel = TheRockChannel::parse(channel)?;
    ensure_install_format_supported(format)?;
    match format {
        "wheel" => install_wheel_runtime(
            paths,
            channel,
            prefix,
            family_override,
            version_selector.as_ref(),
            dry_run,
        ),
        "tarball" => {
            if version_selector.is_some() {
                bail!("specific TheRock version selection is only supported for wheel installs")
            }
            install_tarball_runtime(paths, channel, prefix, family_override, dry_run)
        }
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
        let plan = match runtime_update_plan(paths, manifest, &manifests) {
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
        let _ = writeln!(
            output,
            "  runtime {} format={} channel={} family={} installed={} latest={} status={}",
            manifest.runtime_key,
            plan.format,
            manifest.channel,
            manifest.family,
            runtime_version_display(&manifest.version),
            runtime_version_display(&plan.latest_version),
            plan.status
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
    })
}

fn runtime_freshness(
    manifest: &InstalledRuntimeManifest,
    latest_version: &str,
    required_composition: Option<&WheelRuntimeComposition>,
    target_runtime_key: &str,
) -> RuntimeFreshness {
    match compare_version_strings(&manifest.version, latest_version) {
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
) -> Result<RuntimeUpdatePlan> {
    let latest = resolve_latest_for_manifest(paths, manifest, None)?;
    let freshness = runtime_freshness_with_manifests(
        manifests,
        manifest,
        &latest.latest_version,
        latest.wheel_composition.as_ref(),
        &latest.target_runtime_key,
    );
    Ok(RuntimeUpdatePlan {
        latest_version: latest.latest_version,
        latest_source: latest.latest_source,
        format: latest.format,
        status: freshness.status().to_owned(),
        target_runtime_key: latest.target_runtime_key,
        repair_required: freshness == RuntimeFreshness::RepairAvailable,
        update_available: freshness.update_available(),
    })
}

fn resolve_latest_for_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    download_timeout_secs: Option<u64>,
) -> Result<ResolvedRuntimeUpdate> {
    let channel = TheRockChannel::parse(&manifest.channel)?;
    match manifest.format.as_str() {
        "wheel" => {
            let manifest_python = manifest
                .python_executable
                .as_deref()
                .map(PathBuf::from)
                .filter(|path| path.is_file())
                .map(|executable| PythonLauncher {
                    executable,
                    source: "manifest",
                });
            let python_executable = match manifest_python {
                Some(python) => python,
                None => resolve_python_launcher(paths)?,
            };
            let wheel_compatibility =
                wheel_compatibility_for_python(&python_executable.executable)?;
            let resolution = resolve_pip_runtime_with_timeout(
                paths,
                channel,
                Some(manifest.family.as_str()),
                &wheel_compatibility,
                None,
                download_timeout_secs,
            )?;
            let device_target =
                wheel_composition_device_target(manifest.wheel_composition.as_ref())
                    .and_then(|target| {
                        canonical_aggregate_device_target(target, &resolution.family)
                    })
                    .map_or_else(
                        || {
                            required_aggregate_device_target(
                                detect_host_gfx_target().as_deref(),
                                &resolution.family,
                            )
                        },
                        Ok,
                    )?;
            let wheel_composition = wheel_runtime_composition(&resolution, &device_target);
            let target_runtime_key =
                wheel_runtime_key(channel, &resolution.latest_version, &wheel_composition);
            Ok(ResolvedRuntimeUpdate {
                latest_version: resolution.latest_version,
                latest_source: resolution.index_url,
                target_runtime_key,
                format: "wheel".to_owned(),
                wheel_composition: Some(wheel_composition),
            })
        }
        "tarball" => {
            let artifact = resolve_tarball_artifact_with_timeout(
                paths,
                channel,
                Some(manifest.family.as_str()),
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
    let parent = path
        .parent()
        .context("startup update check path has no parent directory")?;
    fs::create_dir_all(parent)?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(record)
            .context("failed to serialize startup update check record")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn install_wheel_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    family_override: Option<&str>,
    version_selector: Option<&RuntimeVersionSelector>,
    dry_run: bool,
) -> Result<String> {
    progress_line(format!(
        "Checking Python for the ROCm install; if needed, ROCm CLI will prepare Python {}.",
        managed_python_version()
    ));
    let python_launcher = resolve_python_launcher(paths)?;
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
        family_override,
        &wheel_compatibility,
        version_selector,
    )?;
    // An explicit-family preview may run on a host without that GPU. Keep the
    // preview non-mutating and honest about the unresolved device package; an
    // actual install still requires an exact host target before any files change.
    let detected_target = detect_host_gfx_target();
    let device_target =
        match required_aggregate_device_target(detected_target.as_deref(), &resolution.family) {
            Ok(target) => Some(target),
            Err(_) if dry_run => None,
            Err(error) => return Err(error),
        };
    let wheel_composition = device_target
        .as_deref()
        .map(|target| wheel_runtime_composition(&resolution, target));
    progress_line(format!(
        "Found canonical TheRock aggregate version {} with a matching PyTorch stack for target family {}.",
        resolution.latest_version, resolution.family
    ));
    let runtime_key = wheel_composition.as_ref().map_or_else(
        || {
            runtime_key(
                channel,
                "wheel-preview",
                &resolution.family,
                Some(&resolution.latest_version),
            )
        },
        |composition| wheel_runtime_key(channel, &resolution.latest_version, composition),
    );
    let install_root = resolved_install_root(paths, "wheel", &runtime_key, prefix);
    let manifest_path = runtime_manifest_path(paths, &runtime_key);

    let mut output = String::new();
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(
        output,
        "  summary: rocm-cli will install the ROCm SDK and matching PyTorch packages for this Python and operating system"
    );
    let source = channel.canonical_source();
    render_canonical_provenance(
        &mut output,
        channel,
        source.wheel_index,
        source.layout_generation,
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
    let _ = writeln!(
        output,
        "  device_target: {}",
        device_target
            .as_deref()
            .unwrap_or("<undetermined during dry-run>")
    );
    let _ = writeln!(output, "  index_url: {}", resolution.index_url);
    let _ = writeln!(
        output,
        "  latest_compatible_version: {}",
        runtime_version_display(&resolution.latest_version)
    );
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
    let _ = writeln!(
        output,
        "  package_specs: {}",
        wheel_composition.as_ref().map_or_else(
            || "<requires exact GPU target>".to_owned(),
            |composition| composition.package_specs.join(" "),
        )
    );
    let _ = writeln!(
        output,
        "  package_policy: find the newest TheRock ROCm SDK version that has a matching PyTorch stack in the same index, then install pinned target-complete rocm, torch, torchvision, and torchaudio versions in one uv transaction"
    );
    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        if let Some(wheel_composition) = wheel_composition.as_ref() {
            let env_python = venv_python_path(&install_root);
            let mut install_args = uv_pip_install_base(&env_python);
            install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
            if matches!(channel, TheRockChannel::Nightly) {
                install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
            }
            install_args.extend(wheel_composition.package_specs.iter().cloned());
            let venv_args_display = uv_venv_args(&python_launcher.executable, &install_root)
                .iter()
                .map(|arg| quote_display_arg(arg))
                .collect::<Vec<_>>()
                .join(" ");
            let install_args_display = install_args
                .iter()
                .map(|arg| quote_display_arg(arg))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                output,
                "  command: uv {venv_args_display} && uv {install_args_display}"
            );
        } else {
            let _ = writeln!(
                output,
                "  command: <requires an exact GPU target before installation>"
            );
        }
        let _ = writeln!(
            output,
            "  activation: use the managed venv Python; TheRock libraries are resolved from that venv by rocm_sdk.initialize_process"
        );
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(output);
    }
    let device_target =
        device_target.context("exact GPU target was not resolved for installation")?;
    let wheel_composition =
        wheel_composition.context("wheel package composition was not resolved for installation")?;

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
        "install TheRock devel SDK, torch stack, and resolved dependencies",
    )?;

    progress_line("Checking the installed ROCm SDK...");
    let rocm_sdk_probe = probe_rocm_sdk_runtime_for_target(&env_python, Some(&device_target))
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
    Ok(output)
}

fn therock_pip_package_specs(
    package_versions: &TheRockPipPackageVersions,
    rocm_extras: &str,
) -> Vec<String> {
    vec![
        format!("rocm[{rocm_extras}]=={}", package_versions.rocm),
        format!("torch=={}", package_versions.torch),
        format!("torchvision=={}", package_versions.torchvision),
        format!("torchaudio=={}", package_versions.torchaudio),
    ]
}

fn wheel_runtime_composition(
    resolution: &PipRuntimeResolution,
    device_target: &str,
) -> WheelRuntimeComposition {
    WheelRuntimeComposition {
        source_layout_generation: THEROCK_SOURCE_LAYOUT_GENERATION.to_owned(),
        package_specs: therock_pip_package_specs(
            &resolution.package_versions,
            &format!("libraries,devel,device-{device_target}"),
        ),
    }
}

fn wheel_composition_device_target(composition: Option<&WheelRuntimeComposition>) -> Option<&str> {
    composition?.package_specs.iter().find_map(|spec| {
        let extras = spec.strip_prefix("rocm[")?.split_once(']')?.0;
        extras
            .split(',')
            .map(str::trim)
            .find_map(|extra| extra.strip_prefix("device-"))
    })
}

fn required_aggregate_device_target(detected: Option<&str>, family: &str) -> Result<String> {
    let detected = detected.with_context(|| {
        format!(
            "unable to detect an exact GPU target for canonical aggregate family `{family}`; refusing to install every device payload"
        )
    })?;
    canonical_aggregate_device_target(detected, family).with_context(|| {
        format!(
            "detected GPU target `{detected}` does not match resolved family `{family}`; refusing to install every device payload"
        )
    })
}

fn canonical_aggregate_device_target(detected: &str, family: &str) -> Option<String> {
    if normalize_therock_family(detected).as_deref() != Some(family) {
        return None;
    }
    let target = detected
        .split(':')
        .next()
        .unwrap_or(detected)
        .to_ascii_lowercase();
    if target.starts_with("gfx94") {
        Some("gfx942".to_owned())
    } else {
        Some(target)
    }
}

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

fn install_tarball_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    family_override: Option<&str>,
    dry_run: bool,
) -> Result<String> {
    let artifact = resolve_tarball_artifact(paths, channel, family_override)?;
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
    let source = channel.canonical_source();
    render_canonical_provenance(
        &mut output,
        channel,
        source.tarball_catalog,
        source.layout_generation,
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
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  cache_path: {}", cache_path.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(output);
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

    download_file(&artifact.url, &cache_path)?;
    extract_tarball_and_discard_archive(&cache_path, &install_root)?;

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
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(output, "  extracted: {}", install_root.display());
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(output)
}

fn resolve_pip_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
) -> Result<PipRuntimeResolution> {
    resolve_pip_runtime_with_timeout(
        paths,
        channel,
        family_override,
        wheel_compatibility,
        version_selector,
        None,
    )
}

fn resolve_pip_runtime_with_timeout(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
    download_timeout_secs: Option<u64>,
) -> Result<PipRuntimeResolution> {
    let family_resolution = resolve_family(paths, family_override)?;
    let source = channel.canonical_source();
    let root_url = format!("{}/", source.wheel_index.trim_end_matches('/'));
    let root_html = download_text_cached(
        paths,
        &format!("canonical-wheel-root-{}", channel.as_str()),
        &root_url,
        download_timeout_secs,
    )?
    .text;
    validate_aggregate_index_layout(&root_html).with_context(|| {
        format!(
            "failed to resolve TheRock {} wheel runtime from canonical source {}",
            channel.as_str(),
            source.wheel_index
        )
    })?;
    resolve_pip_runtime_from_index(
        paths,
        channel,
        &family_resolution,
        source.wheel_index,
        wheel_compatibility,
        version_selector,
        download_timeout_secs,
    )
    .with_context(|| {
        format!(
            "failed to resolve TheRock {} wheel runtime from canonical source {}\n\n{}",
            channel.as_str(),
            source.wheel_index,
            canonical_wheel_resolution_hint(channel)
        )
    })
}

fn resolve_pip_runtime_from_index(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_resolution: &FamilyResolution,
    index_url: &str,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
    download_timeout_secs: Option<u64>,
) -> Result<PipRuntimeResolution> {
    let rocm_versions =
        load_simple_index_versions(paths, index_url, "rocm", None, download_timeout_secs)?;
    if matches!(channel, TheRockChannel::Release)
        && version_selector.is_none()
        && !rocm_versions
            .iter()
            .any(|version| is_stable_runtime_version(version))
    {
        bail!(
            "release channel only installs stable TheRock wheel versions, but no stable `rocm` package versions were found in {index_url}; try `rocm install sdk --channel release --format tarball` for stable release artifacts, or use `--channel nightly --format wheel` for preview builds"
        );
    }
    let torch_versions = load_simple_index_versions(
        paths,
        index_url,
        "torch",
        Some(wheel_compatibility),
        download_timeout_secs,
    )?;
    let torchvision_versions = load_simple_index_versions(
        paths,
        index_url,
        "torchvision",
        Some(wheel_compatibility),
        download_timeout_secs,
    )?;
    let torchaudio_versions = load_simple_index_versions(
        paths,
        index_url,
        "torchaudio",
        Some(wheel_compatibility),
        download_timeout_secs,
    )?;
    let package_versions = select_matching_pip_package_versions(
        channel,
        &rocm_versions,
        &torch_versions,
        &torchvision_versions,
        &torchaudio_versions,
        version_selector,
    )
    .with_context(|| {
        let requested = version_selector.map_or_else(|| "latest compatible version".to_owned(), RuntimeVersionSelector::describe);
        format!(
            "no mutually compatible target-complete TheRock ROCm SDK, torch, torchvision, and torchaudio versions were found for {requested} in {index_url}"
        )
    })?;
    let latest_version = package_versions.rocm.clone();
    Ok(PipRuntimeResolution {
        family: family_resolution.family.clone(),
        family_source: family_resolution.source.clone(),
        index_url: index_url.to_owned(),
        latest_version,
        package_versions,
    })
}

fn resolve_tarball_artifact(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
) -> Result<TarballArtifact> {
    resolve_tarball_artifact_with_timeout(paths, channel, family_override, None)
}

fn resolve_tarball_artifact_with_timeout(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    download_timeout_secs: Option<u64>,
) -> Result<TarballArtifact> {
    let family_resolution = resolve_family(paths, family_override)?;
    let source = channel.canonical_source();
    let html = download_text_cached(
        paths,
        &format!("tarball-index-{}", channel.as_str()),
        source.tarball_catalog,
        download_timeout_secs,
    )?
    .text;
    let files = parse_tarball_index_html(&html).with_context(|| {
        format!(
            "unknown canonical TheRock tarball catalog layout at {}",
            source.tarball_catalog
        )
    })?;
    let (file, version) = select_tarball_candidate(&files, channel, &family_resolution.family)
        .with_context(|| {
            format!(
                "canonical TheRock {} tarball stream is incomplete for the resolved GPU family\n\n{}",
                channel.as_str(),
                family_resolution_hint(
                    &family_resolution.source,
                    &family_resolution.family,
                    channel,
                    "tarball",
                )
            )
        })?;
    Ok(TarballArtifact {
        family: family_resolution.family,
        family_source: family_resolution.source,
        url: format!(
            "{}/{}",
            source.tarball_catalog.trim_end_matches('/'),
            file.name
        ),
        file_name: file.name,
        version,
    })
}

fn select_tarball_candidate(
    files: &[TarballIndexFile],
    channel: TheRockChannel,
    family: &str,
) -> Option<(TarballIndexFile, String)> {
    let prefix = format!("therock-dist-{}-{family}-", platform_tarball_token());
    let mut candidates = files
        .iter()
        .filter_map(|file| {
            let version = file
                .name
                .strip_prefix(&prefix)?
                .strip_suffix(".tar.gz")?
                .to_owned();
            if matches!(channel, TheRockChannel::Release) && !is_stable_runtime_version(&version) {
                return None;
            }
            Some((file.clone(), version))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.0
            .mtime
            .partial_cmp(&right.0.mtime)
            .unwrap_or(Ordering::Equal)
            .then_with(|| compare_version_strings(&left.1, &right.1))
    });
    candidates.pop()
}

fn resolve_family(paths: &AppPaths, family_override: Option<&str>) -> Result<FamilyResolution> {
    if let Some(value) = family_override
        && let Some(family) = normalize_therock_family(value)
    {
        return Ok(FamilyResolution {
            family,
            source: "manifest".to_owned(),
        });
    }

    if let Some(value) = std::env::var("ROCM_CLI_THEROCK_FAMILY").ok()
        && let Some(family) = normalize_therock_family(&value)
    {
        return Ok(FamilyResolution {
            family,
            source: "env".to_owned(),
        });
    }

    if let Some(family) = detect_managed_therock_family(paths) {
        return Ok(FamilyResolution {
            family,
            source: "managed-runtime".to_owned(),
        });
    }

    if let Some(family) = detect_host_therock_family() {
        return Ok(FamilyResolution {
            family,
            source: "host".to_owned(),
        });
    }

    bail!(
        "unable to resolve a supported TheRock GPU family for this host.\n\
         Re-run with an explicit package family: `rocm install sdk --family <FAMILY>`.\n\
         Recognized families: {}.\n\n{}",
        known_therock_families().join(", "),
        detect_host_gpu_diagnostics()
    )
}

fn select_matching_pip_package_versions(
    channel: TheRockChannel,
    rocm_versions: &[String],
    torch_versions: &[String],
    torchvision_versions: &[String],
    torchaudio_versions: &[String],
    version_selector: Option<&RuntimeVersionSelector>,
) -> Option<TheRockPipPackageVersions> {
    let mut rocm_candidates = if version_selector.is_some() {
        rocm_versions.to_vec()
    } else {
        channel_rocm_candidates(rocm_versions, channel)
    };
    if let Some(selector) = version_selector {
        rocm_candidates.retain(|version| selector.matches_version(version));
    }
    rocm_candidates.sort_by(|left, right| compare_version_strings(left, right));

    for rocm_version in rocm_candidates.into_iter().rev() {
        let mut torch_candidates = package_versions_matching_rocm(torch_versions, &rocm_version);
        torch_candidates.sort_by(|left, right| compare_version_strings(left, right));

        for torch_version in torch_candidates.into_iter().rev() {
            let Some(torch_base) = parse_package_base_version(&torch_version) else {
                continue;
            };
            let torchaudio_version =
                select_latest_stack_package(torchaudio_versions, &rocm_version, |base| {
                    pytorch_audio_matches_torch(&torch_base, base)
                });
            let torchvision_version =
                select_latest_stack_package(torchvision_versions, &rocm_version, |base| {
                    pytorch_vision_matches_torch(&torch_base, base)
                });
            if let (Some(torchaudio), Some(torchvision)) = (torchaudio_version, torchvision_version)
            {
                return Some(TheRockPipPackageVersions {
                    compatibility_key: rocm_version.clone(),
                    rocm: rocm_version,
                    torch: torch_version,
                    torchvision,
                    torchaudio,
                });
            }
        }
    }

    None
}

fn channel_rocm_candidates(versions: &[String], channel: TheRockChannel) -> Vec<String> {
    let mut all = versions.to_vec();
    all.sort_by(|left, right| compare_version_strings(left, right));
    if matches!(channel, TheRockChannel::Release) {
        return all
            .iter()
            .filter(|version| is_stable_runtime_version(version))
            .cloned()
            .collect::<Vec<_>>();
    }
    all
}

fn is_stable_runtime_version(version: &str) -> bool {
    parse_version(version).is_some_and(|parsed| parsed.stage == VersionStage::Stable)
}

fn select_latest_stack_package(
    versions: &[String],
    rocm_version: &str,
    matches_stack: impl Fn(&ParsedVersion) -> bool,
) -> Option<String> {
    let mut candidates = package_versions_matching_rocm(versions, rocm_version)
        .into_iter()
        .filter(|version| {
            parse_package_base_version(version).is_some_and(|base| matches_stack(&base))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| compare_version_strings(left, right));
    candidates.pop()
}

fn package_versions_matching_rocm(versions: &[String], rocm_version: &str) -> Vec<String> {
    versions
        .iter()
        .filter(|version| package_rocm_suffix(version).as_deref() == Some(rocm_version))
        .cloned()
        .collect()
}

fn pytorch_audio_matches_torch(torch_base: &ParsedVersion, audio_base: &ParsedVersion) -> bool {
    audio_base.major == torch_base.major
        && audio_base.minor == torch_base.minor
        && audio_base.stage == torch_base.stage
}

fn pytorch_vision_matches_torch(torch_base: &ParsedVersion, vision_base: &ParsedVersion) -> bool {
    let Some(expected_minor) = torch_base.minor.checked_add(15) else {
        return false;
    };
    vision_base.major == 0
        && vision_base.minor == expected_minor
        && vision_base.stage == torch_base.stage
}

fn parse_package_base_version(version: &str) -> Option<ParsedVersion> {
    parse_version(version.split('+').next().unwrap_or(version))
}

fn package_rocm_suffix(version: &str) -> Option<String> {
    let decoded = decode_simple_index_version(version);
    let lower = decoded.to_ascii_lowercase();
    let marker = "+rocm";
    let start = lower.rfind(marker)? + marker.len();
    decoded.get(start..).map(str::to_owned)
}

fn decode_simple_index_version(version: &str) -> String {
    version.replace("%2B", "+").replace("%2b", "+")
}

pub(crate) fn runtime_version_display(version: &str) -> String {
    if let Some(date) = runtime_version_build_date(version) {
        format!("{version} (build {date})")
    } else {
        version.to_owned()
    }
}

pub(crate) fn runtime_version_build_date(version: &str) -> Option<String> {
    let bytes = version.as_bytes();
    if bytes.len() < 8 {
        return None;
    }
    for window in bytes.windows(8) {
        if !window.iter().all(u8::is_ascii_digit) {
            continue;
        }
        let digits = std::str::from_utf8(window).ok()?;
        let year = digits[0..4].parse::<u32>().ok()?;
        let month = digits[4..6].parse::<u32>().ok()?;
        let day = digits[6..8].parse::<u32>().ok()?;
        if !(2000..=2099).contains(&year) || month == 0 || month > 12 {
            continue;
        }
        let max_day = days_in_month(year, month);
        if day == 0 || day > max_day {
            continue;
        }
        return Some(format!("{year:04}-{month:02}-{day:02}"));
    }
    None
}

fn normalize_requested_build_date(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("TheRock build date cannot be empty");
    }
    let digits = trimmed
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>();
    if digits.len() != 8 {
        bail!("TheRock build date `{trimmed}` must use YYYY-MM-DD, YYYYMMDD, or MMDDYYYY");
    }

    let parsed = if digits.starts_with("20") {
        parse_yyyy_mm_dd(&digits)
    } else if digits[4..].starts_with("20") {
        parse_mm_dd_yyyy(&digits)
    } else {
        None
    };
    let Some((year, month, day)) = parsed else {
        bail!("TheRock build date `{trimmed}` must use YYYY-MM-DD, YYYYMMDD, or MMDDYYYY");
    };
    validate_date_components(year, month, day)
        .with_context(|| format!("invalid TheRock build date `{trimmed}`"))?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

fn parse_yyyy_mm_dd(digits: &str) -> Option<(u32, u32, u32)> {
    Some((
        digits[0..4].parse().ok()?,
        digits[4..6].parse().ok()?,
        digits[6..8].parse().ok()?,
    ))
}

fn parse_mm_dd_yyyy(digits: &str) -> Option<(u32, u32, u32)> {
    Some((
        digits[4..8].parse().ok()?,
        digits[0..2].parse().ok()?,
        digits[2..4].parse().ok()?,
    ))
}

fn validate_date_components(year: u32, month: u32, day: u32) -> Result<()> {
    if !(2000..=2099).contains(&year) {
        bail!("year must be between 2000 and 2099");
    }
    if month == 0 || month > 12 {
        bail!("month must be between 1 and 12");
    }
    let max_day = days_in_month(year, month);
    if day == 0 || day > max_day {
        bail!("day must be between 1 and {max_day}");
    }
    Ok(())
}

const fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

const fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn load_simple_index_versions(
    paths: &AppPaths,
    index_url: &str,
    package_name: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
    download_timeout_secs: Option<u64>,
) -> Result<Vec<String>> {
    let url = format!("{}/{package_name}/", index_url.trim_end_matches('/'));
    let html = download_text_cached(
        paths,
        &format!("simple-index-{}-{}", slugify(index_url), package_name),
        &url,
        download_timeout_secs,
    )?
    .text;
    Ok(parse_simple_index_versions(
        &html,
        package_name,
        wheel_compatibility,
    ))
}

fn parse_simple_index_versions(
    html: &str,
    package_name: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
) -> Vec<String> {
    let marker = format!("{package_name}-");
    let mut versions = Vec::new();
    for line in html.lines() {
        let mut rest = line;
        while let Some(start) = rest.find(&marker) {
            let version_start = start + marker.len();
            let Some(candidate) = rest.get(version_start..) else {
                break;
            };
            if let Some((version, consumed)) =
                parse_simple_index_version_candidate(candidate, wheel_compatibility)
            {
                if let Some(version) = version {
                    versions.push(version);
                }
                rest = candidate.get(consumed..).unwrap_or_default();
            } else {
                break;
            }
        }
    }
    versions.sort_by(|left, right| compare_version_strings(left, right));
    versions.dedup();
    versions
}

fn parse_simple_index_version_candidate(
    candidate: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
) -> Option<(Option<String>, usize)> {
    let tar_pos = candidate.find(".tar.gz");
    let wheel_pos = candidate.find(".whl");
    match (tar_pos, wheel_pos) {
        (Some(tar_pos), Some(wheel_pos)) if tar_pos < wheel_pos => {
            let version = decode_simple_index_version(candidate.get(..tar_pos)?);
            Some((Some(version), tar_pos + ".tar.gz".len()))
        }
        (Some(tar_pos), None) => {
            let version = decode_simple_index_version(candidate.get(..tar_pos)?);
            Some((Some(version), tar_pos + ".tar.gz".len()))
        }
        (_, Some(wheel_pos)) => {
            let wheel_stem = candidate.get(..wheel_pos)?;
            if let Some(wheel_compatibility) = wheel_compatibility
                && !wheel_stem_matches_compatibility(wheel_stem, wheel_compatibility)
            {
                return Some((None, wheel_pos + ".whl".len()));
            }
            let version = wheel_stem.split('-').next()?;
            Some((
                Some(decode_simple_index_version(version)),
                wheel_pos + ".whl".len(),
            ))
        }
        (None, None) => None,
    }
}

fn wheel_compatibility_for_python(python_executable: &Path) -> Result<WheelCompatibility> {
    let python_tag = capture_python_stdout(
        python_executable,
        "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
        "inspect Python wheel tag",
    )
    .with_context(|| {
        format!(
            "failed to inspect Python wheel tag via {}",
            python_executable.display()
        )
    })?;
    let python_tag = python_tag.trim().to_owned();
    if python_tag.is_empty() {
        bail!("Python did not report a wheel tag");
    }
    Ok(WheelCompatibility {
        python_tag,
        platform_tags: current_platform_wheel_tags()?,
    })
}

fn current_platform_wheel_tags() -> Result<Vec<String>> {
    let platform_tag = match (runtime_os_name(), std::env::consts::ARCH) {
        ("windows", "x86_64") => "win_amd64",
        ("linux", "x86_64") => "linux_x86_64",
        ("linux", "aarch64") => "linux_aarch64",
        (os, arch) => bail!("TheRock wheel filtering is not implemented for {os}/{arch}"),
    };
    Ok(vec![platform_tag.to_owned(), "any".to_owned()])
}

fn wheel_stem_matches_compatibility(wheel_stem: &str, compatibility: &WheelCompatibility) -> bool {
    let mut parts = wheel_stem.rsplitn(4, '-');
    let Some(platform_tag) = parts.next() else {
        return false;
    };
    let Some(abi_tag) = parts.next() else {
        return false;
    };
    let Some(python_tag) = parts.next() else {
        return false;
    };
    if parts.next().is_none() {
        return false;
    }

    wheel_python_tag_matches(python_tag, &compatibility.python_tag)
        && wheel_abi_tag_matches(abi_tag, &compatibility.python_tag)
        && wheel_platform_tag_matches(platform_tag, &compatibility.platform_tags)
}

fn wheel_python_tag_matches(wheel_tag: &str, python_tag: &str) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| tag == python_tag || tag == "py3")
}

fn wheel_abi_tag_matches(wheel_tag: &str, python_tag: &str) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| tag == python_tag || tag == "abi3" || tag == "none")
}

fn wheel_platform_tag_matches(wheel_tag: &str, platform_tags: &[String]) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| platform_tags.iter().any(|platform| platform == tag))
}

#[cfg(test)]
fn select_latest_version(versions: &[String], channel: TheRockChannel) -> Option<String> {
    let mut stable = Vec::new();
    let mut all = versions.to_vec();
    all.sort_by(|left, right| compare_version_strings(left, right));
    for version in versions {
        if is_stable_runtime_version(version) {
            stable.push(version.clone());
        }
    }
    stable.sort_by(|left, right| compare_version_strings(left, right));
    match channel {
        TheRockChannel::Release => stable.pop(),
        TheRockChannel::Nightly => all.pop(),
    }
}

/// Pinned production metadata signing public key (trust root). Empty until the
/// repository owner publishes production keys (see docs/release-trust.md,
/// "Remaining Owner Step"). While empty, metadata verification stays opt-in
/// (enabled only via the `ROCM_CLI_METADATA_PUBLIC_KEY_*` env vars). Once
/// populated, metadata signatures are verified by default with this key as the
/// trust root.
const PINNED_METADATA_PUBLIC_KEY_PEM: &str = "";

/// The pinned metadata trust root, or `None` while the sentinel is still empty.
fn pinned_metadata_public_key() -> Option<String> {
    let pem = PINNED_METADATA_PUBLIC_KEY_PEM.trim();
    (!pem.is_empty()).then(|| pem.to_owned())
}

impl MetadataSignaturePolicy {
    fn from_env() -> Self {
        Self::resolve(
            truthy_env("ROCM_CLI_REQUIRE_METADATA_SIGNATURE"),
            env_path("ROCM_CLI_METADATA_PUBLIC_KEY_PATH"),
            env_nonempty("ROCM_CLI_METADATA_PUBLIC_KEY_PEM"),
            pinned_metadata_public_key(),
        )
    }

    /// Combine the env-provided inputs with the pinned trust root. An explicit
    /// env key (path or PEM) wins as an escape hatch; otherwise the pinned key is
    /// used, and its presence makes verification required by default.
    fn resolve(
        env_required: bool,
        env_path: Option<PathBuf>,
        env_pem: Option<String>,
        pinned_pem: Option<String>,
    ) -> Self {
        let pinned = if env_path.is_none() && env_pem.is_none() {
            pinned_pem
        } else {
            None
        };
        Self {
            required: env_required || pinned.is_some(),
            public_key_path: env_path,
            public_key_pem: env_pem.or(pinned),
        }
    }

    const fn active(&self) -> bool {
        self.required || self.public_key_path.is_some() || self.public_key_pem.is_some()
    }

    fn validate_configuration(&self) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        if let Some(public_key_path) = &self.public_key_path {
            if !public_key_path.is_file() {
                bail!(
                    "metadata public key not found: {}",
                    public_key_path.display()
                );
            }
            return Ok(());
        }
        if self.public_key_pem.is_some() {
            return Ok(());
        }
        bail!(
            "metadata signature verification requires ROCM_CLI_METADATA_PUBLIC_KEY_PATH or ROCM_CLI_METADATA_PUBLIC_KEY_PEM"
        )
    }
}

fn truthy_env(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env_nonempty(name).map(PathBuf::from)
}

fn with_metadata_public_key<T>(
    policy: &MetadataSignaturePolicy,
    temp_key_path: &Path,
    verify: impl FnOnce(&Path, &'static str) -> Result<T>,
) -> Result<Option<T>> {
    if !policy.active() {
        return Ok(None);
    }
    if let Some(public_key_path) = &policy.public_key_path {
        policy.validate_configuration()?;
        return verify(public_key_path, "path").map(Some);
    }
    if let Some(public_key_pem) = &policy.public_key_pem {
        let staged_key = stage_file_for_atomic_publish(temp_key_path, public_key_pem.as_bytes())?;
        let result = verify(&staged_key, "env-pem");
        let _ = fs::remove_file(staged_key);
        return result.map(Some);
    }
    bail!(
        "metadata signature verification requires ROCM_CLI_METADATA_PUBLIC_KEY_PATH or ROCM_CLI_METADATA_PUBLIC_KEY_PEM"
    )
}

fn metadata_signature_url(url: &str) -> String {
    format!("{url}.sig")
}

fn metadata_signature_path(body_path: &Path) -> PathBuf {
    body_path.with_extension("sig")
}

fn fetch_and_verify_metadata_signature(
    policy: &MetadataSignaturePolicy,
    url: &str,
    payload: &[u8],
    temp_key_path: &Path,
    max_time_secs: Option<u64>,
) -> Result<Option<(CachedHttpSignatureMetadata, Vec<u8>)>> {
    if !policy.active() {
        return Ok(None);
    }
    let signature_url = metadata_signature_url(url);
    let response = http_get(&signature_url, &[], max_time_secs)?;
    if response.status != 200 {
        bail!(
            "HTTP {} while fetching metadata signature {signature_url}",
            response.status
        );
    }
    let signature = response.body;
    let public_key_source =
        with_metadata_public_key(policy, temp_key_path, |public_key, source| {
            verify_metadata_signature_bytes(payload, &signature, public_key)?;
            Ok(source.to_owned())
        })?
        .context("metadata signature policy was active but no public key was resolved")?;
    Ok(Some((
        CachedHttpSignatureMetadata {
            url: signature_url,
            verified_at_unix_ms: unix_time_millis(),
            public_key_source,
        },
        signature,
    )))
}

#[cfg(test)]
fn verify_cached_metadata_signature(
    policy: &MetadataSignaturePolicy,
    payload_path: &Path,
    signature_path: &Path,
    temp_key_path: &Path,
) -> Result<()> {
    if !policy.active() {
        return Ok(());
    }
    if !signature_path.is_file() {
        bail!(
            "metadata signature verification requested but cached signature is missing: {}",
            signature_path.display()
        );
    }
    with_metadata_public_key(policy, temp_key_path, |public_key, _source| {
        verify_metadata_signature(payload_path, signature_path, public_key)
    })?;
    Ok(())
}

#[cfg(test)]
fn verify_metadata_signature(
    payload_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<()> {
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read metadata public key: {}",
            public_key_path.display()
        )
    })?;
    let signature = fs::read(signature_path).with_context(|| {
        format!(
            "failed to read metadata signature: {}",
            signature_path.display()
        )
    })?;
    let payload = fs::read(payload_path).with_context(|| {
        format!(
            "failed to read metadata payload: {}",
            payload_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, &payload, &signature, "metadata")
}

fn verify_metadata_signature_bytes(
    payload: &[u8],
    signature: &[u8],
    public_key_path: &Path,
) -> Result<()> {
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read metadata public key: {}",
            public_key_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, payload, signature, "metadata")
}

#[cfg(test)]
fn metadata_cache_can_revalidate(
    metadata: &CachedHttpMetadata,
    policy: &MetadataSignaturePolicy,
    signature_path: &Path,
) -> bool {
    !policy.active() || (metadata.signature.is_some() && signature_path.is_file())
}

fn load_cached_http_entry(
    metadata_path: &Path,
    legacy_body_path: &Path,
    legacy_signature_path: &Path,
) -> Option<CachedHttpCacheEntry> {
    let bytes = fs::read(metadata_path).ok()?;
    if let Ok(entry) = serde_json::from_slice(&bytes) {
        return Some(entry);
    }

    let metadata: CachedHttpMetadata = serde_json::from_slice(&bytes).ok()?;
    let body = fs::read_to_string(legacy_body_path).ok()?;
    let signature_bytes = metadata
        .signature
        .as_ref()
        .map(|_| fs::read(legacy_signature_path))
        .transpose()
        .ok()?;
    Some(CachedHttpCacheEntry {
        metadata,
        body,
        signature_bytes,
    })
}

const fn cached_http_entry_can_revalidate(
    entry: &CachedHttpCacheEntry,
    policy: &MetadataSignaturePolicy,
) -> bool {
    !policy.active() || (entry.metadata.signature.is_some() && entry.signature_bytes.is_some())
}

fn verify_cached_http_entry_signature(
    policy: &MetadataSignaturePolicy,
    entry: &CachedHttpCacheEntry,
    temp_key_path: &Path,
) -> Result<()> {
    if !policy.active() {
        return Ok(());
    }
    let signature = entry.signature_bytes.as_deref().context(
        "metadata signature verification requested but cached signature bytes are missing",
    )?;
    with_metadata_public_key(policy, temp_key_path, |public_key, _source| {
        verify_metadata_signature_bytes(entry.body.as_bytes(), signature, public_key)
    })?;
    Ok(())
}

fn write_cached_http_entry(path: &Path, entry: &CachedHttpCacheEntry) -> Result<()> {
    write_file_atomically(
        path,
        &serde_json::to_vec_pretty(entry).context("failed to serialize metadata cache entry")?,
    )
}

#[cfg(test)]
fn write_cached_http_entry_with<S, P, F>(
    path: &Path,
    entry: &CachedHttpCacheEntry,
    suffix_for_attempt: S,
    before_publish: P,
    publish: F,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    write_file_atomically_with_publish(
        path,
        &serde_json::to_vec_pretty(entry).context("failed to serialize metadata cache entry")?,
        suffix_for_attempt,
        before_publish,
        publish,
    )
}

fn download_text_cached(
    paths: &AppPaths,
    cache_key: &str,
    url: &str,
    max_time_secs: Option<u64>,
) -> Result<CachedHttpText> {
    let (body_path, metadata_path) = metadata_cache_paths(paths, cache_key);
    let signature_path = metadata_signature_path(&body_path);
    let signature_policy = MetadataSignaturePolicy::from_env();
    signature_policy.validate_configuration()?;
    let previous_entry = load_cached_http_entry(&metadata_path, &body_path, &signature_path)
        .filter(|entry| entry.metadata.url == url);
    let cache_dir = body_path
        .parent()
        .context("metadata cache path has no parent directory")?;
    fs::create_dir_all(cache_dir)?;

    let tmp_public_key = body_path.with_extension("public-key.pem");
    let mut headers = Vec::new();
    if let Some(etag) = previous_entry
        .as_ref()
        .filter(|entry| cached_http_entry_can_revalidate(entry, &signature_policy))
        .and_then(|entry| entry.metadata.etag.as_deref())
    {
        headers.push(("If-None-Match", etag));
    }

    let response = http_get(url, &headers, max_time_secs)?;
    if response.status == 304 {
        let entry = previous_entry.context(
            "metadata cache returned 304 but no complete cached generation is available",
        )?;
        verify_cached_http_entry_signature(&signature_policy, &entry, &tmp_public_key)?;
        return Ok(CachedHttpText { text: entry.body });
    }
    if response.status != 200 {
        bail!("HTTP {} while fetching {url}", response.status);
    }

    let body =
        String::from_utf8(response.body).context("metadata response body was not valid UTF-8")?;
    let fetched_signature = fetch_and_verify_metadata_signature(
        &signature_policy,
        url,
        body.as_bytes(),
        &tmp_public_key,
        max_time_secs,
    )?;
    let signature_metadata = fetched_signature
        .as_ref()
        .map(|(metadata, _)| metadata.clone());
    let metadata = CachedHttpMetadata {
        url: url.to_owned(),
        etag: http_header_value(&response.headers, "etag"),
        last_modified: http_header_value(&response.headers, "last-modified"),
        signature: signature_metadata,
        fetched_at_unix_ms: unix_time_millis(),
    };
    let entry = CachedHttpCacheEntry {
        metadata,
        body,
        signature_bytes: fetched_signature.map(|(_, signature)| signature),
    };
    write_cached_http_entry(&metadata_path, &entry)?;
    let _ = fs::remove_file(&body_path);
    let _ = fs::remove_file(&signature_path);
    Ok(CachedHttpText { text: entry.body })
}

fn metadata_cache_paths(paths: &AppPaths, cache_key: &str) -> (PathBuf, PathBuf) {
    let base = paths
        .cache_dir
        .join("therock")
        .join("metadata")
        .join(slugify(cache_key));
    (base.with_extension("body"), base.with_extension("json"))
}

fn http_header_value(headers: &str, name: &str) -> Option<String> {
    let prefix = format!("{}:", name.to_ascii_lowercase());
    let mut value = None;
    for line in headers.lines() {
        let trimmed = line.trim();
        if trimmed.to_ascii_lowercase().starts_with(&prefix) {
            value = trimmed
                .split_once(':')
                .map(|(_, rest)| rest.trim().to_owned())
                .filter(|rest| !rest.is_empty());
        }
    }
    value
}

/// Fetch an artifact to `destination`.
///
/// Streams rather than buffers: SDK tarballs are single-digit gigabytes, and
/// holding one in memory to write it out again costs that much RAM on top of
/// the same amount of disk. The primitive also handles the free-space
/// preflight, retry with resume, and the length cross-check that catches a
/// transfer the server ended early.
fn download_file(url: &str, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    rocm_core::download_file_streaming(&rocm_core::DownloadRequest::new(
        url,
        destination,
        THEROCK_DOWNLOAD_TIMEOUT,
    ))
    .with_context(|| format!("failed to fetch {url}"))?;
    Ok(())
}

/// Content length of `url` from a HEAD request, when the server reports one.
///
/// Best effort: any failure yields `None`, so a server that rejects HEAD or
/// omits `Content-Length` simply skips the preflight instead of blocking the
/// install.
fn head_content_length(url: &str) -> Option<u64> {
    let timeout = Duration::from_secs(THEROCK_HEAD_PROBE_TIMEOUT_SECS);
    let agent = ureq::AgentBuilder::new()
        // `timeout_connect` takes precedence over `timeout` and defaults to 30s,
        // so without it a host that blackholes rather than refuses would stall
        // the probe well past the intended ceiling.
        .timeout_connect(timeout)
        .timeout(timeout)
        .build();
    let response = agent.head(url).set("User-Agent", "rocm-cli").call().ok()?;
    if response.status() != 200 {
        return None;
    }
    let length: u64 = response.header("Content-Length")?.trim().parse().ok()?;
    // The header is unauthenticated and is never cross-checked against the body
    // the subsequent GET delivers, so an inflated value from a proxy or CDN
    // would refuse an install that would in fact succeed. Treat an implausible
    // size as no answer at all: the preflight is skipped and `download_file`
    // still checks the real, buffered body length before writing.
    (length <= THEROCK_MAX_PLAUSIBLE_TARBALL_BYTES).then_some(length)
}

/// Refuse (or warn) before a multi-GB SDK tarball download and extraction.
///
/// The download requirement comes from `Content-Length` and is exact, so a
/// shortfall is a hard error — it saves the user a long download that cannot
/// possibly succeed. The extraction requirement is only an estimate (see
/// [`disk_space::EXTRACTED_SIZE_MULTIPLIER`]), so a shortfall there is a
/// warning: a false refusal that blocks a valid install would be worse than a
/// late failure.
///
/// Any extraction warning is returned rather than printed, so the caller can
/// place it in the same accumulated output block as the rest of the install
/// report instead of having it appear ahead of that block.
fn preflight_tarball_space(
    url: &str,
    cache_path: &Path,
    install_root: &Path,
) -> Result<Option<String>> {
    let Some(download_bytes) = head_content_length(url) else {
        return Ok(None);
    };
    disk_space::ensure_space_for(
        "download the SDK tarball",
        cache_path,
        disk_space::with_margin(download_bytes),
    )?;

    // When the cache and the install root share a filesystem, the archive and
    // the extracted tree must both fit at the same time.
    let mut extract_estimate = disk_space::estimated_extracted_size(download_bytes);
    if disk_space::on_same_filesystem(cache_path, install_root) == Some(true) {
        extract_estimate = extract_estimate.saturating_add(download_bytes);
    }
    Ok(disk_space::warn_if_low_space(
        "extract the SDK tarball",
        install_root,
        disk_space::with_margin(extract_estimate),
    ))
}

fn http_get(
    url: &str,
    headers: &[(&str, &str)],
    max_time_secs: Option<u64>,
) -> Result<HttpResponseBody> {
    let timeout = max_time_secs
        .filter(|value| *value > 0)
        .map_or_else(|| Duration::from_mins(10), Duration::from_secs);
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let mut request = agent.get(url).set("User-Agent", "rocm-cli");
    for (name, value) in headers {
        request = request.set(name, value);
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => response,
        Err(error) => bail!("HTTP request failed for {url}: {error}"),
    };
    let status = response.status();
    let headers = response
        .headers_names()
        .into_iter()
        .filter_map(|name| {
            response
                .header(&name)
                .map(|value| format!("{name}: {value}"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut reader = response.into_reader();
    let mut body = Vec::new();
    reader
        .read_to_end(&mut body)
        .with_context(|| format!("failed to read HTTP response body for {url}"))?;
    Ok(HttpResponseBody {
        status,
        headers,
        body,
    })
}

fn linux_temp_dir(prefix: &str) -> Result<PathBuf> {
    let root = std::env::temp_dir();
    let base = format!("{prefix}-{}-{}", std::process::id(), unix_time_millis());
    for attempt in 0..100 {
        let dir = root.join(format!("{base}-{attempt}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", dir.display()));
            }
        }
    }
    bail!(
        "failed to create a unique temporary directory under {}",
        root.display()
    )
}

fn windows_temp_dir(prefix: &str) -> Result<PathBuf> {
    let root = windows_runtime_temp_root().unwrap_or_else(std::env::temp_dir);
    let base = format!("{prefix}-{}-{}", std::process::id(), unix_time_millis());
    for attempt in 0..100 {
        let dir = root.join(format!("{base}-{attempt}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", dir.display()));
            }
        }
    }
    bail!(
        "failed to create a unique temporary directory under {}",
        root.display()
    )
}

fn windows_runtime_temp_root() -> Option<PathBuf> {
    for name in ["TEMP", "TMP", "LOCALAPPDATA"] {
        if let Some(value) = std::env::var_os(name).filter(|value| !value.is_empty()) {
            let path = PathBuf::from(value);
            return Some(if name == "LOCALAPPDATA" {
                path.join("Temp")
            } else {
                path
            });
        }
    }
    None
}

fn windows_child_path(path: &Path) -> String {
    runtime_path_for_windows_child(path)
}

/// A unique temp path next to `path`, preserving the full file name so a
/// multi-extension artifact keeps its extensions (`sdk.tar.gz` becomes
/// `sdk.tar.gz.tmp-<id>`, where `with_extension` would drop `.gz`).
const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

fn temp_sibling_path(path: &Path, suffix: &OsStr) -> Result<PathBuf> {
    let parent = path.parent().context("file path has no parent directory")?;
    let mut file_name = path
        .file_name()
        .context("file path has no file name")?
        .to_os_string();
    file_name.push(".tmp-");
    file_name.push(suffix);
    Ok(parent.join(file_name))
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    write_file_atomically_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        || {},
    )
}

fn write_file_atomically_with<S, P>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    before_publish: P,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
{
    write_file_atomically_with_publish(
        path,
        bytes,
        suffix_for_attempt,
        before_publish,
        publish_temp_file,
    )
}

fn write_file_atomically_with_publish<S, P, F>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    before_publish: P,
    publish: F,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    let tmp = stage_file_for_atomic_publish_with(path, bytes, suffix_for_attempt)?;
    before_publish();
    publish_staged_file_with(&tmp, path, publish)
}

fn stage_file_for_atomic_publish(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    stage_file_for_atomic_publish_with(path, bytes, |attempt| {
        OsString::from(format!("{temp_id}-{attempt}"))
    })
}

fn stage_file_for_atomic_publish_with<S>(
    path: &Path,
    bytes: &[u8],
    mut suffix_for_attempt: S,
) -> Result<PathBuf>
where
    S: FnMut(u32) -> OsString,
{
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent)?;

    let mut reserved = None;
    for attempt in 0..ATOMIC_WRITE_TEMP_ATTEMPTS {
        let suffix = suffix_for_attempt(attempt);
        let tmp = temp_sibling_path(path, &suffix)?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => {
                reserved = Some((tmp, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", tmp.display()));
            }
        }
    }
    let Some((tmp, mut file)) = reserved else {
        bail!(
            "failed to reserve a temporary file next to {} after {} attempts",
            path.display(),
            ATOMIC_WRITE_TEMP_ATTEMPTS
        );
    };

    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, &tmp));
    }
    drop(file);
    Ok(tmp)
}

#[cfg(test)]
fn publish_staged_file(tmp: &Path, path: &Path) -> Result<()> {
    publish_staged_file_with(tmp, path, publish_temp_file)
}

fn publish_staged_file_with<F>(tmp: &Path, path: &Path, publish: F) -> Result<()>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    publish(tmp, path)
        .inspect_err(|_| {
            let _ = fs::remove_file(tmp);
        })
        .with_context(|| format!("failed to publish {}", path.display()))
}

#[cfg(not(windows))]
fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    fs::rename(tmp, path)
}

#[cfg(windows)]
fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    if path.try_exists()? {
        return replace_file_windows(path, tmp);
    }

    match fs::rename(tmp, path) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            if path.try_exists()? {
                replace_file_windows(path, tmp)
            } else {
                Err(rename_error)
            }
        }
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn replace_file_windows(path: &Path, replacement: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;

    let path_wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement_wide: Vec<u16> = replacement
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();

    // SAFETY: both path buffers are valid, NUL-terminated UTF-16 strings and
    // remain alive for the duration of the synchronous Windows API call. The
    // optional backup, exclude, and reserved pointers are intentionally null.
    let replaced = unsafe {
        ReplaceFileW(
            path_wide.as_ptr(),
            replacement_wide.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn extract_tarball(archive_path: &Path, target_dir: &Path) -> Result<()> {
    run_command(
        Path::new("tar"),
        &[
            "-xf",
            archive_path.to_string_lossy().as_ref(),
            "-C",
            target_dir.to_string_lossy().as_ref(),
        ],
        "extract TheRock tarball artifact",
    )
    .map_err(|error| {
        // The extraction preflight only warns, because the extracted size is an
        // estimate. When that warning turns out to be right, the failure arrives
        // as `tar` stderr rather than an `io::Error`, so it never reaches
        // `map_write_error` — without this the user gets the raw
        // "tar: ...: No space left on device" this feature exists to replace.
        disk_space::subprocess_full_disk_error(&format!("{error:#}"), target_dir).unwrap_or(error)
    })
}

/// Unpack the SDK archive and then delete it.
///
/// Only the extracted tree is used from here on, so keeping the archive would
/// double the disk cost of every installed version. This mirrors the cleanup
/// `ensure_uv_binary` already performs after unpacking its own download.
///
/// Removing the archive is best-effort: the install has already succeeded by
/// this point, so a cleanup failure is reported rather than raised.
fn extract_tarball_and_discard_archive(archive_path: &Path, target_dir: &Path) -> Result<()> {
    extract_tarball(archive_path, target_dir)?;
    if let Err(error) = fs::remove_file(archive_path) {
        progress_line(format!(
            "Could not remove the downloaded archive {}: {error}",
            archive_path.display()
        ));
    }
    Ok(())
}

fn ensure_uv_venv(
    paths: &AppPaths,
    uv: &Path,
    python_launcher: &Path,
    install_root: &Path,
) -> Result<()> {
    let env_python = venv_python_path(install_root);
    if env_python.is_file() {
        if run_command(
            &env_python,
            &["--version"],
            "verify existing managed TheRock runtime Python",
        )
        .is_ok()
        {
            return Ok(());
        }
        progress_line("Existing Python environment is incomplete; recreating it.");
        fs::remove_dir_all(install_root).with_context(|| {
            format!(
                "failed to remove incomplete Python environment at {}",
                install_root.display()
            )
        })?;
    }
    let args = uv_venv_args(python_launcher, install_root);
    run_command_with_env(
        uv,
        args.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        &uv_command_env(paths),
        "create managed TheRock runtime virtual environment",
    )?;
    if !env_python.is_file() {
        bail!(
            "managed Python environment did not create expected executable: {}",
            env_python.display()
        );
    }
    Ok(())
}

fn python_venv_args(install_root: &Path) -> Vec<String> {
    vec![
        "-m".to_owned(),
        "venv".to_owned(),
        install_root.to_string_lossy().to_string(),
    ]
}

pub(crate) fn probe_rocm_sdk_runtime(python_executable: &Path) -> Result<RocmSdkPythonProbe> {
    probe_rocm_sdk_runtime_for_target(python_executable, None)
}

fn probe_rocm_sdk_runtime_for_target(
    python_executable: &Path,
    device_target: Option<&str>,
) -> Result<RocmSdkPythonProbe> {
    let script = device_target.map_or_else(
        || ROCM_SDK_PROBE_SCRIPT.to_owned(),
        |target| {
            format!(
                "import os\nos.environ['ROCM_SDK_TARGET_FAMILY'] = {target:?}\n{ROCM_SDK_PROBE_SCRIPT}"
            )
        },
    );
    let text = capture_python_stdout(python_executable, &script, "launch rocm_sdk probe")
        .with_context(|| {
            format!(
                "failed to launch rocm_sdk probe via {}",
                python_executable.display()
            )
        })?;
    parse_rocm_sdk_probe(&text)
}

fn parse_rocm_sdk_probe(output: &str) -> Result<RocmSdkPythonProbe> {
    serde_json::from_str(output.trim()).context("failed to parse rocm_sdk probe output")
}

/// What the runtime's torch reports about the GPUs it can actually open, and
/// whether those GPUs can actually run a kernel.
///
/// [`validate_rocm_sdk_runtime_probe`] establishes that the SDK's libraries are
/// present and resolvable. That is not the same question as whether the torch
/// sharing the venv can enumerate a device: a torch built against a different
/// HIP version loads happily against those libraries and then reports no
/// devices at all.
///
/// Enumeration succeeding is in turn not the same question as the device being
/// usable. A torch built against a different HIP version can enumerate the
/// GPUs and then fault on the first kernel it launches, so the two failures are
/// reported separately and must not be conflated.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct RuntimeDeviceProbe {
    pub import_ok: bool,
    pub torch_version: Option<String>,
    pub hip_version: Option<String>,
    /// `None` when torch never imported, so "unknown" stays distinct from "zero".
    pub device_count: Option<u32>,
    /// An import or enumeration failure. Never a kernel failure.
    pub error: Option<String>,
    /// A GPU kernel failure observed *after* devices enumerated successfully.
    /// `None` when no kernel was attempted (no devices, or enumeration failed).
    #[serde(default)]
    pub kernel_error: Option<String>,
}

/// Ask the runtime's own interpreter whether torch can execute a GPU kernel.
///
/// `library_paths` must be the runtime's recorded ROCm library directories (see
/// [`RocmSdkPythonProbe::library_paths`]). They are prepended to
/// `LD_LIBRARY_PATH` for the child, which is how a served process resolves them.
///
/// This cannot be replaced with an in-process `rocm_sdk.initialize_process()`
/// call in the probe script. Measured on MI300X against a runtime that serves
/// correctly: with the library directories on `LD_LIBRARY_PATH` torch reports 8
/// devices, and with `initialize_process()` alone it reports 0. A probe built on
/// the latter would fail healthy runtimes and send people to reinstall them,
/// which is the operation that breaks them.
pub(crate) fn probe_runtime_devices(
    python_executable: &Path,
    library_paths: &[PathBuf],
) -> Result<RuntimeDeviceProbe> {
    let mut env = Vec::new();
    if !library_paths.is_empty() {
        let mut entries = library_paths.to_vec();
        if let Some(existing) = std::env::var_os(LIBRARY_PATH_ENV) {
            entries.extend(split_runtime_path(&existing));
        }
        let joined = std::env::join_paths(entries)
            .context("failed to compose the runtime library path for the device probe")?;
        env.push((
            LIBRARY_PATH_ENV.to_owned(),
            joined.to_string_lossy().into_owned(),
        ));
    }
    let text = capture_python_stdout_with_env(
        python_executable,
        RUNTIME_DEVICE_PROBE_SCRIPT,
        &env,
        "launch runtime device probe",
    )
    .with_context(|| {
        format!(
            "failed to launch runtime device probe via {}",
            python_executable.display()
        )
    })?;
    parse_runtime_device_probe(&text)
}

fn parse_runtime_device_probe(output: &str) -> Result<RuntimeDeviceProbe> {
    serde_json::from_str(output.trim()).context("failed to parse runtime device probe output")
}

/// The loader search-path variable used to expose the runtime's ROCm libraries.
#[cfg(windows)]
const LIBRARY_PATH_ENV: &str = "PATH";
#[cfg(not(windows))]
const LIBRARY_PATH_ENV: &str = "LD_LIBRARY_PATH";

/// Reports what torch sees, never raising: an unusable runtime must be
/// described, not turned into a probe crash.
const RUNTIME_DEVICE_PROBE_SCRIPT: &str = r#"
import json

out = {
    "import_ok": False,
    "torch_version": None,
    "hip_version": None,
    "device_count": None,
    "error": None,
    "kernel_error": None,
}

try:
    import torch

    out["import_ok"] = True
    out["torch_version"] = getattr(torch, "__version__", None)
    out["hip_version"] = getattr(getattr(torch, "version", None), "hip", None)
    device_count = int(torch.cuda.device_count())
    out["device_count"] = device_count
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)

# Enumeration is not execution. A runtime whose torch and HIP disagree can
# report devices and then fault on the first kernel, so the kernel is launched
# under its own guard and its failure is recorded in its own field. Only run it
# once enumeration actually produced a device: with no devices there is nothing
# to execute on, and a failed import has already been described.
if out["error"] is None and (out["device_count"] or 0) > 0:
    try:
        probe = torch.ones(32, device="cuda")
        probe.add_(1.0)
        torch.cuda.synchronize()
    except Exception as exc:
        out["kernel_error"] = type(exc).__name__ + ": " + str(exc)

print(json.dumps(out))
"#;

/// What torch is installed, and what torch the engine's metadata demands.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct TorchAlignmentProbe {
    /// The installed torch, e.g. `2.11.0+rocm7.13.0`. `None` if torch is absent.
    pub installed_torch: Option<String>,
    /// The engine's pinned requirement, e.g. `torch==2.11.0+gitd0c8b1f`. `None`
    /// when the engine is not installed yet or does not pin torch.
    pub engine_requires_torch: Option<String>,
}

/// Read installed/required torch from distribution metadata, without importing
/// torch. Metadata questions must not depend on the runtime being usable — this
/// is called both before and after an engine install, and in the broken state
/// importing torch is exactly what fails.
pub(crate) fn probe_torch_alignment(
    python_executable: &Path,
    engine_distribution: &str,
) -> Result<TorchAlignmentProbe> {
    let env = vec![(
        "ROCM_CLI_PROBE_DIST".to_owned(),
        engine_distribution.to_owned(),
    )];
    let text = capture_python_stdout_with_env(
        python_executable,
        TORCH_ALIGNMENT_PROBE_SCRIPT,
        &env,
        "launch torch alignment probe",
    )?;
    serde_json::from_str(text.trim()).context("failed to parse torch alignment probe output")
}

const TORCH_ALIGNMENT_PROBE_SCRIPT: &str = r#"
import json
import os
import re
import importlib.metadata as md

out = {"installed_torch": None, "engine_requires_torch": None}

try:
    out["installed_torch"] = md.version("torch")
except Exception:
    pass

try:
    for raw in md.requires(os.environ.get("ROCM_CLI_PROBE_DIST", "vllm")) or []:
        # `Requires-Dist` entries carry environment markers after ';'. Only the
        # requirement itself matters here.
        requirement = raw.split(";")[0].strip()
        # The distribution name runs up to the first specifier, extra or space —
        # and there is usually no space at all, as in `torch==2.11.0+gitd0c8b1f`.
        matched = re.match(r"[A-Za-z0-9._-]+", requirement)
        if matched is None:
            continue
        if matched.group(0).lower().replace("_", "-") == "torch":
            out["engine_requires_torch"] = requirement
            break
except Exception:
    pass

print(json.dumps(out))
"#;

/// Split a version into its public part and its local segment.
///
/// `2.11.0+rocm7.13.0` -> `("2.11.0", Some("rocm7.13.0"))`. The local segment is
/// the build identifier: for TheRock wheels it names the ROCm build, and for the
/// engine's own index it is an opaque commit tag.
///
/// Defined in `rocm-core` because the vLLM engine has to make the same split to
/// recognise a runtime the CLI deliberately realigned; two copies of this would be
/// two places for the two sides to drift apart.
pub(crate) fn split_local_version(version: &str) -> (&str, Option<&str>) {
    rocm_core::uv::split_local_version(version)
}

/// The version pinned by a `==` requirement, e.g. `torch==2.11.0+git…` -> the
/// version. Returns `None` for any looser requirement, since only an exact pin
/// tells us which release the engine was built against.
pub(crate) fn requirement_pinned_version(requirement: &str) -> Option<&str> {
    let (_, version) = requirement.split_once("==")?;
    let version = version.trim();
    if version.is_empty() || version.contains(',') {
        return None;
    }
    Some(version)
}

/// Install one exact package version from `index_url` into `python_executable`.
///
/// Used to put the SDK's build of a package back after another installer has
/// replaced it. `--reinstall-package` is required: without it uv treats the
/// already-present distribution as satisfying the request and does nothing.
///
/// `--index-url`, not `--extra-index-url`, so the SDK index is the only place a
/// candidate can come from. This matches how `install_therock_runtime` installs
/// from the same index, and it is load-bearing rather than cosmetic: with PyPI
/// left in the candidate set, a `+rocm` build the SDK index does not publish can
/// resolve against PyPI instead, and the caller's `Unavailable` classification —
/// the whole point of which is to say "the SDK index has no such build" — never
/// gets the resolver error it keys on.
///
/// `--no-deps` because this is a surgical swap of one build for another build of
/// the *same release*: the environment already carries a resolved dependency tree
/// and re-resolving it here is free to move `torchvision`/`torchaudio` as a side
/// effect, which is the mixed stack this change is trying not to create. A build
/// that genuinely needs a different dependency is not silently ignored — the
/// `uv pip check` that runs immediately after reports it as a violation.
pub(crate) fn install_pinned_package(
    paths: &AppPaths,
    python_executable: &Path,
    index_url: &str,
    package: &str,
    requirement: &str,
) -> Result<()> {
    let uv = rocm_core::uv::ensure_uv_binary(paths)
        .context("failed to acquire uv binary for the torch alignment install")?;
    let mut args = rocm_core::uv::uv_pip_install_base(python_executable);
    args.push("--index-url".to_owned());
    args.push(index_url.to_owned());
    args.push("--no-deps".to_owned());
    args.push("--reinstall-package".to_owned());
    args.push(package.to_owned());
    args.push(requirement.to_owned());
    let borrowed = args.iter().map(String::as_str).collect::<Vec<_>>();
    // `run_command_with_env`, not `run_command`, for two reasons. The uv environment
    // carries `UV_HTTP_TIMEOUT` (uv has no `--timeout` flag) and `UV_CACHE_DIR`;
    // without the latter uv falls back to `$HOME/.cache/uv`, loses hardlinking when
    // that is on another filesystem, and silently copies the whole torch stack per
    // environment — and the e2e lanes' shared cache is threaded through the same
    // helper. It also captures stderr on every platform, where `run_command`'s
    // Windows branch reports only an exit status; the caller classifies this
    // install's outcome by matching the resolver's message, so on Windows an
    // unpublished build would otherwise be reported as a generic failure.
    run_command_with_env(
        &uv,
        &borrowed,
        &rocm_core::uv::uv_command_env(paths),
        "install the SDK build of the engine's torch",
    )
}

pub(crate) fn validate_rocm_sdk_runtime_probe(probe: &RocmSdkPythonProbe) -> Result<()> {
    if !probe.import_ok {
        bail!(
            "TheRock packages did not expose a usable rocm_sdk runtime: {}",
            probe.error.as_deref().unwrap_or("<unknown error>")
        );
    }
    let Some(rocm_sdk_root) = probe.root_path.as_ref() else {
        bail!("TheRock packages exposed rocm_sdk but did not report a runtime root path");
    };
    if !rocm_sdk_root.is_dir() {
        bail!(
            "TheRock rocm_sdk runtime root path does not exist: {}",
            rocm_sdk_root.display()
        );
    }
    let Some(rocm_sdk_bin) = probe.bin_path.as_ref() else {
        bail!("TheRock packages exposed rocm_sdk but did not report a runtime bin path");
    };
    if !rocm_sdk_bin.is_dir() {
        bail!(
            "TheRock rocm_sdk runtime bin path does not exist: {}",
            rocm_sdk_bin.display()
        );
    }
    if !probe_has_resolved_library(probe, "amdhip64") {
        bail!("TheRock rocm_sdk runtime did not expose amdhip64 through rocm_sdk.find_libraries");
    }
    if !probe_has_resolved_library(probe, "hipblas") {
        bail!("TheRock rocm_sdk runtime did not expose hipblas through rocm_sdk.find_libraries");
    }
    Ok(())
}

fn probe_has_resolved_library(probe: &RocmSdkPythonProbe, shortname: &str) -> bool {
    probe.resolved_libraries.iter().any(|library| {
        library.shortname == shortname && library.paths.iter().any(|path| path.is_file())
    })
}

const ROCM_SDK_PROBE_SCRIPT: &str = r#"
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sysconfig

out = {
    "import_ok": False,
    "rocm_sdk_version": None,
    "site_packages": sysconfig.get_paths().get("purelib"),
    "root_path": None,
    "bin_path": None,
    "cmake_path": None,
    "runtime_roots": [],
    "bin_paths": [],
    "library_paths": [],
    "default_target_family": None,
    "available_target_families": [],
    "resolved_target_family": None,
    "packages": [],
    "library_shortnames": [],
    "resolved_libraries": [],
    "error": None,
}

def add_path(key, path):
    if path is None:
        return
    value = str(path)
    if value not in out[key]:
        out[key].append(value)

def package_root(package, target_family=None):
    module_name = package.get_py_package_name(target_family)
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).parent

def add_runtime_root(root):
    if root is None:
        return
    add_path("runtime_roots", root)
    for child in [root / "bin", root / "lib", root / "lib64", root / "lib" / "rocm_sysdeps" / "lib"]:
        if child.is_dir():
            if child.name == "bin":
                add_path("bin_paths", child)
            add_path("library_paths", child)

try:
    import rocm_sdk
    from rocm_sdk import _dist_info as di

    out["import_ok"] = True
    out["rocm_sdk_version"] = getattr(rocm_sdk, "__version__", None)
    out["default_target_family"] = getattr(di, "DEFAULT_TARGET_FAMILY", None)
    out["available_target_families"] = list(getattr(di, "AVAILABLE_TARGET_FAMILIES", []))
    try:
        from rocm_sdk import _devel
        root_path = _devel.get_devel_root()
        out["root_path"] = str(root_path)
        out["bin_path"] = str(root_path / "bin")
        out["cmake_path"] = str(root_path / "lib" / "cmake")
        add_runtime_root(root_path)
    except Exception as exc:
        out["root_path_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        out["resolved_target_family"] = di.determine_target_family()
    except Exception as exc:
        out["resolved_target_family_error"] = type(exc).__name__ + ": " + str(exc)

    target_family = out["resolved_target_family"] or out["default_target_family"]
    for logical_name, target in [
        ("core", None),
        ("libraries", target_family),
        ("device", target_family),
        ("profiler", None),
    ]:
        try:
            package = di.ALL_PACKAGES[logical_name]
            if package.has_py_package(target):
                add_runtime_root(package_root(package, target))
        except Exception as exc:
            out.setdefault("package_root_errors", {})[logical_name] = type(exc).__name__ + ": " + str(exc)

    scripts_path = sysconfig.get_path("scripts")
    if scripts_path:
        scripts_path = Path(scripts_path)
        if scripts_path.is_dir():
            add_path("bin_paths", scripts_path)

    if out["root_path"] is None and out["runtime_roots"]:
        out["root_path"] = out["runtime_roots"][0]
    if out["bin_path"] is None and out["bin_paths"]:
        out["bin_path"] = out["bin_paths"][0]
    if out["cmake_path"] is None and out["root_path"] is not None:
        cmake_path = Path(out["root_path"]) / "lib" / "cmake"
        if cmake_path.is_dir():
            out["cmake_path"] = str(cmake_path)

    out["library_shortnames"] = sorted(getattr(di, "ALL_LIBRARIES", {}).keys())
    resolved_libraries = []
    for shortname in out["library_shortnames"]:
        try:
            paths = [str(path) for path in rocm_sdk.find_libraries(shortname)]
        except Exception:
            paths = []
        if paths:
            resolved_libraries.append({"shortname": shortname, "paths": paths})
    out["resolved_libraries"] = resolved_libraries

    packages = []
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower().startswith("rocm"):
            packages.append({"name": name, "version": dist.version})
    out["packages"] = sorted(packages, key=lambda item: item["name"].lower())
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)

print(json.dumps(out))
"#;

fn progress_line(message: impl AsRef<str>) {
    println!("{}", message.as_ref());
    let _ = std::io::stdout().flush();
}

fn capture_command_output(program: &Path, args: &[&str]) -> Result<Output> {
    if runtime_is_windows() {
        return capture_command_output_with_temp_files(program, args);
    }
    Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to launch {}", program.display()))
}

fn capture_command_output_with_temp_files(program: &Path, args: &[&str]) -> Result<Output> {
    let temp_dir = windows_temp_dir("rocm-cli-command")?;
    let stdout_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    let stdout = fs::read(&stdout_path).unwrap_or_default();
    let stderr = fs::read(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// Run a probe script with extra environment set and return its stdout.
///
/// The script goes to a temp file rather than `python -c`, because the loader
/// search path this exists to set must reach the child through its environment,
/// and a file keeps the invocation identical on every platform.
fn capture_python_stdout_with_env(
    python_executable: &Path,
    script: &str,
    env: &[(String, String)],
    context_text: &str,
) -> Result<String> {
    let temp_root = if runtime_is_windows() {
        windows_temp_dir("rocm-cli-python-probe")?
    } else {
        linux_temp_dir("rocm-cli-python-probe")?
    };
    let script_path = temp_root.join("probe.py");
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write {}", script_path.display()))?;

    let mut command = Command::new(python_executable);
    command.arg(&script_path);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch {}", python_executable.display()));
    let _ = fs::remove_dir_all(&temp_root);
    let output = output?;

    if !output.status.success() {
        bail!(
            "{context_text}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("{context_text}: failed to decode Python output"))
}

fn capture_python_stdout(
    python_executable: &Path,
    script: &str,
    context_text: &str,
) -> Result<String> {
    if !runtime_is_windows() {
        let output = capture_command_output(python_executable, &["-c", script])?;
        if !output.status.success() {
            bail!(
                "{context_text}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        return String::from_utf8(output.stdout)
            .with_context(|| format!("{context_text}: failed to decode Python output"));
    }

    let temp_dir = windows_temp_dir("rocm-cli-python")?;
    let script_path = temp_dir.join("probe.py");
    let wrapper_path = temp_dir.join("wrapper.py");
    let output_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write {}", script_path.display()))?;
    fs::write(
        &wrapper_path,
        r#"import contextlib
import pathlib
import runpy
import sys

out = pathlib.Path(sys.argv[1])
script = pathlib.Path(sys.argv[2])
with out.open("w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        runpy.run_path(str(script), run_name="__main__")
"#,
    )
    .with_context(|| format!("failed to write {}", wrapper_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = Command::new(python_executable)
        .arg(windows_child_path(&wrapper_path))
        .arg(windows_child_path(&output_path))
        .arg(windows_child_path(&script_path))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch {}", python_executable.display()))?;
    let text = fs::read_to_string(&output_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(&temp_dir);
    if status.success() {
        Ok(text)
    } else {
        let stderr = stderr.trim().to_owned();
        let detail = if stderr.is_empty() {
            format!("command exited with status {status}")
        } else {
            stderr
        };
        bail!("{context_text}: {detail}")
    }
}

fn run_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    if runtime_is_windows() {
        let status = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to launch {}", program.display()))?;
        if status.success() {
            return Ok(());
        }
        bail!("{context_text}: command exited with status {status}");
    }

    let output = capture_command_output(program, args)?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("command exited with status {}", output.status)
    };
    bail!("{context_text}: {detail}")
}

#[allow(dead_code)]
fn run_progress_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: command exited with status {status}");
}

fn run_command_with_env(
    program: &Path,
    args: &[&str],
    env: &[(String, String)],
    context_text: &str,
) -> Result<()> {
    let mut command = Command::new(program);
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let detail = if stderr.is_empty() {
        format!("command exited with status {}", output.status)
    } else {
        stderr
    };
    bail!("{context_text}: {detail}")
}

fn run_uv_progress_command(
    paths: &AppPaths,
    uv: &Path,
    args: &[&str],
    context_text: &str,
) -> Result<()> {
    let mut command = Command::new(uv);
    command.args(args);
    for (key, value) in &uv_command_env(paths) {
        command.env(key, value);
    }
    let status = command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| "failed to launch uv".to_string())?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: uv exited with status {status}");
}

fn managed_tools_root(paths: &AppPaths) -> PathBuf {
    managed_tools_dir(&paths.data_dir)
}

fn managed_python_manifest_path(paths: &AppPaths) -> PathBuf {
    managed_tools_root(paths)
        .join("registry")
        .join("python.json")
}

fn load_managed_python_manifest(paths: &AppPaths) -> Result<Option<ManagedPythonManifest>> {
    let path = managed_python_manifest_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn save_managed_python_manifest(paths: &AppPaths, manifest: &ManagedPythonManifest) -> Result<()> {
    let path = managed_python_manifest_path(paths);
    let parent = path
        .parent()
        .context("managed Python manifest path has no parent directory")?;
    fs::create_dir_all(parent)?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(manifest)
            .context("failed to serialize managed Python manifest")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn record_managed_python_config(paths: &AppPaths, python: &Path) -> Result<()> {
    let mut config = RocmCliConfig::load(paths).unwrap_or_default();
    config.tools.insert(
        "python".to_owned(),
        ManagedToolConfig {
            path: Some(python.to_path_buf()),
            managed: true,
        },
    );
    config.save(paths)
}

fn managed_python_bootstrap_disabled() -> bool {
    std::env::var("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP")
        .ok()
        .is_some_and(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
}

fn managed_python_version() -> String {
    std::env::var("ROCM_CLI_MANAGED_PYTHON_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MANAGED_PYTHON_VERSION.to_owned())
}

fn ensure_managed_python(paths: &AppPaths) -> Result<PythonLauncher> {
    let version = managed_python_version();
    progress_line(format!("Preparing Python {version}..."));

    let uv = ensure_uv_binary(paths)?;

    // Check the manifest first — if the recorded executable is still usable, skip the install.
    if let Ok(Some(manifest)) = load_managed_python_manifest(paths)
        && manifest.version == version
        && manifest.executable.is_file()
        && python_launcher_install_ready(&manifest.executable).is_ok()
    {
        progress_line(format!(
            "Using existing Python {version} at {}.",
            manifest.executable.display()
        ));
        let _ = record_managed_python_config(paths, &manifest.executable);
        return Ok(PythonLauncher {
            executable: manifest.executable,
            source: "managed",
        });
    }

    progress_line(format!("Installing Python {version} via uv..."));
    let status = Command::new(&uv)
        .args(["python", "install", &version])
        .envs(uv_command_env(paths))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to launch uv python install")?;
    if !status.success() {
        bail!("uv python install {version} failed with {status}");
    }

    progress_line(format!("Finding Python {version}..."));
    let output = Command::new(&uv)
        .args(["python", "find", &version])
        .envs(uv_command_env(paths))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("failed to launch uv python find")?;
    if !output.status.success() {
        bail!("uv python find {version} failed after install");
    }
    let executable = PathBuf::from(
        String::from_utf8(output.stdout)
            .context("uv python find output was not valid UTF-8")?
            .trim(),
    );
    if !executable.is_file() {
        bail!(
            "uv python find returned a path that does not exist: {}",
            executable.display()
        );
    }

    python_launcher_install_ready(&executable).with_context(|| {
        format!(
            "Python {version} at {} could not create a virtual environment",
            executable.display()
        )
    })?;

    let manifest = ManagedPythonManifest {
        executable: executable.clone(),
        version: version.clone(),
        installed_at_unix_ms: unix_time_millis(),
    };
    save_managed_python_manifest(paths, &manifest)?;
    let _ = record_managed_python_config(paths, &executable);
    progress_line(format!(
        "Python {version} is ready at {}.",
        executable.display()
    ));
    Ok(PythonLauncher {
        executable,
        source: "managed",
    })
}

/// The environment inputs [`resolve_python_launcher`] reads.
///
/// Passed in rather than read at each use site so a caller can point the
/// resolver somewhere else without touching the process environment. Tests need
/// that: `cargo test` runs every test as a thread in one process, so a test that
/// overwrote `PATH` to steer this resolver also hid every other PATH-resolved
/// binary from unrelated tests running at the same moment.
struct PythonResolverEnv {
    /// `ROCM_CLI_PYTHON`: an explicit interpreter that wins over any search.
    python_override: Option<String>,
    /// The directories to search for an interpreter, in `PATH` order.
    search_dirs: Vec<PathBuf>,
}

impl PythonResolverEnv {
    fn from_process_env() -> Self {
        Self {
            python_override: std::env::var("ROCM_CLI_PYTHON").ok(),
            search_dirs: std::env::var_os("PATH")
                .map(|value| split_runtime_path(&value))
                .unwrap_or_default(),
        }
    }
}

fn resolve_python_launcher(paths: &AppPaths) -> Result<PythonLauncher> {
    resolve_python_launcher_in(paths, &PythonResolverEnv::from_process_env())
}

fn resolve_python_launcher_in(paths: &AppPaths, env: &PythonResolverEnv) -> Result<PythonLauncher> {
    if let Some(value) = env.python_override.as_deref() {
        python_launcher_install_ready(Path::new(value))
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLauncher {
            executable: PathBuf::from(value),
            source: "env",
        });
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        match python_launcher_install_ready(&candidate) {
            Ok(()) => {
                return Ok(PythonLauncher {
                    executable: candidate,
                    source: "path",
                });
            }
            Err(_) => {
                skipped_path_python = true;
            }
        }
    }
    if skipped_path_python {
        progress_line(
            "Python from PATH cannot create a virtual environment; using ROCm CLI's managed Python.",
        );
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        if python_launcher_install_ready(&manifest.executable).is_ok() {
            return Ok(PythonLauncher {
                executable: manifest.executable,
                source: "managed",
            });
        }
        progress_line(
            "Saved managed Python cannot create a virtual environment; preparing Python again.",
        );
    }

    if managed_python_bootstrap_disabled() {
        bail!(
            "unable to locate Python, and managed Python bootstrap is disabled by ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"
        );
    }
    ensure_managed_python(paths)
}

fn python_path_candidates(search_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let program_names: &[&str] = if runtime_is_windows() {
        &["python", "python3", "py"]
    } else {
        &["python3", "python"]
    };
    program_names
        .iter()
        .flat_map(|program| resolve_program_on_path(program, search_dirs))
        .collect()
}

fn resolve_program_on_path(program: &str, search_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let candidates = program_path_candidates(program);
    search_dirs
        .iter()
        .flat_map(|dir| candidates.iter().map(move |candidate| dir.join(candidate)))
        .filter(|path| path.is_file())
        .map(|path| normalize_runtime_path_for_host(&path))
        .collect()
}

fn split_runtime_path(value: &std::ffi::OsStr) -> Vec<PathBuf> {
    runtime_path_list_split(value)
}

fn program_path_candidates(program: &str) -> Vec<String> {
    let path = Path::new(program);
    if !runtime_is_windows() || path.extension().is_some() {
        return vec![program.to_owned()];
    }
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    let mut names = vec![program.to_owned()];
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.sort();
    names.dedup();
    names
}

fn python_launcher_install_ready(program: &Path) -> Result<()> {
    let compatibility = wheel_compatibility_for_python(program)?;
    if compatibility.python_tag != "cp312" {
        bail!(
            "Python wheel tag {} is not supported; cp312 is required",
            compatibility.python_tag
        );
    }
    verify_python_can_create_venv(program)
}

fn verify_python_can_create_venv(program: &Path) -> Result<()> {
    let probe_root = python_venv_probe_temp_root()?;
    let probe_dir = probe_root.join("env");
    let args = python_venv_args(&probe_dir);
    let result = run_command(
        program,
        args.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        "probe Python virtual environment support",
    );
    let _ = fs::remove_dir_all(&probe_root);
    result
}

fn python_venv_probe_temp_root() -> Result<PathBuf> {
    if runtime_is_windows() {
        windows_temp_dir("rocm-cli-python-venv-probe")
    } else {
        linux_temp_dir("rocm-cli-python-venv-probe")
    }
}

fn parse_tarball_index_html(html: &str) -> Result<Vec<TarballIndexFile>> {
    let start = html
        .find("const files = ")
        .context("tarball index did not contain the embedded file list")?;
    let json_start = start + "const files = ".len();
    let rest = &html[json_start..];
    let end = rest
        .find("];")
        .context("tarball index did not contain the end of the embedded file list")?;
    let json = format!("{}]", &rest[..end]);
    serde_json::from_str(&json).context("failed to parse TheRock tarball index file list")
}

fn compare_version_strings(left: &str, right: &str) -> Ordering {
    match (parse_version(left), parse_version(right)) {
        (Some(left_parsed), Some(right_parsed)) => {
            left_parsed.cmp(&right_parsed).then_with(|| left.cmp(right))
        }
        _ => left.cmp(right),
    }
}

fn parse_version(value: &str) -> Option<ParsedVersion> {
    let value = value.split('+').next().unwrap_or(value);
    let mut parts = value.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_rest = parts.next()?;

    let patch_len = patch_and_rest
        .chars()
        .take_while(char::is_ascii_digit)
        .count();
    if patch_len == 0 {
        return None;
    }
    let patch = patch_and_rest[..patch_len].parse().ok()?;
    let suffix = &patch_and_rest[patch_len..];

    let (stage, stage_number) = if suffix.is_empty() {
        (VersionStage::Stable, 0)
    } else if let Some(rest) = suffix.strip_prefix("rc") {
        (VersionStage::Rc, rest.parse().ok()?)
    } else if let Some(rest) = suffix.strip_prefix('a') {
        (VersionStage::Alpha, rest.parse().ok()?)
    } else {
        return None;
    };

    Some(ParsedVersion {
        major,
        minor,
        patch,
        stage,
        stage_number,
    })
}

/// Recovery guidance appended to family/index resolution failures so a clean
/// first run can recover without the user having to guess a `--family`.
///
/// `source` is the [`FamilyResolution::source`] that produced `family`:
/// auto-detected (`host`, `managed-runtime`) versus user-supplied (`manifest`
/// from `--family`, `env` from `ROCM_CLI_THEROCK_FAMILY`). The wording differs
/// so an auto-detected miss points the user at `--family`, while a user-supplied
/// miss confirms the family they already named. Both point at the other channel
/// and, where valid for the platform, the other install format.
fn canonical_wheel_resolution_hint(channel: TheRockChannel) -> String {
    let other_channel = match channel {
        TheRockChannel::Release => "nightly",
        TheRockChannel::Nightly => "release",
    };
    let mut hint = format!(
        "No complete compatible package stack was found in the canonical {} aggregate stream. Try `--channel {other_channel}`",
        channel.as_str()
    );
    if !runtime_is_windows() {
        hint.push_str(" or `--format tarball`");
    }
    hint.push('.');
    hint
}

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
    let fingerprint = format!("{:x}", hasher.finalize());
    slugify(&format!(
        "{}-wheel-multi-arch-{version}-{}",
        channel.as_str(),
        &fingerprint[..16]
    ))
}

fn family_resolution_hint(
    source: &str,
    family: &str,
    channel: TheRockChannel,
    format: &str,
) -> String {
    let families = known_therock_families().join(", ");
    let auto_detected = matches!(source, "host" | "managed-runtime");
    let mut hint = String::new();

    if auto_detected {
        let _ = write!(
            hint,
            "no installable TheRock {} runtime was found for the auto-detected GPU family `{family}`.\n\
             Re-run with an explicit package family: `rocm install sdk --family <FAMILY>`.\n\
             Recognized families: {families}.",
            channel.as_str()
        );
    } else {
        let _ = write!(
            hint,
            "no installable TheRock {} runtime was found for the requested package family `{family}`.\n\
             Recognized families: {families}.",
            channel.as_str()
        );
    }

    let other_channel = match channel {
        TheRockChannel::Release => "nightly",
        TheRockChannel::Nightly => "release",
    };
    let alternate_format = match format {
        "wheel" if !runtime_is_windows() => Some("tarball"),
        "tarball" => Some("wheel"),
        _ => None,
    };
    match alternate_format {
        Some(alternate_format) => {
            let _ = write!(
                hint,
                "\nIf your GPU is newer than the {} packages, try `--channel {other_channel}` or `--format {alternate_format}`.",
                channel.as_str()
            );
        }
        None => {
            let _ = write!(
                hint,
                "\nIf your GPU is newer than the {} packages, try `--channel {other_channel}`.",
                channel.as_str()
            );
        }
    }

    hint
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

fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

fn runtime_manifest_path(paths: &AppPaths, runtime_key: &str) -> PathBuf {
    runtime_registry_dir(paths).join(format!("{runtime_key}.json"))
}

fn save_runtime_manifest(paths: &AppPaths, manifest: &InstalledRuntimeManifest) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime manifest registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&manifest).context("failed to serialize runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;

    let local_manifest_path = manifest.install_root.join(".rocm-cli-runtime.json");
    fs::write(
        &local_manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .context("failed to serialize local runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", local_manifest_path.display()))?;
    Ok(())
}

pub(crate) fn load_runtime_manifests(paths: &AppPaths) -> Result<Vec<InstalledRuntimeManifest>> {
    let registry_dir = runtime_registry_dir(paths);
    if !registry_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&registry_dir)
        .with_context(|| format!("failed to read {}", registry_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(manifest) = serde_json::from_slice::<InstalledRuntimeManifest>(&bytes) {
            manifests.push(manifest.normalize_host_paths());
        }
    }
    manifests.sort_by_key(|manifest| std::cmp::Reverse(manifest.installed_at_unix_ms));
    Ok(manifests)
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
    use super::*;

    // Serializes tests that replace process-global `PATH` (or other env vars) while
    // they run. Because env is shared across all test threads, any test that spawns a
    // bare-name binary (e.g. `tar`) via `PATH` lookup must also hold this lock, or it
    // can fail with ENOENT while another test has temporarily narrowed `PATH`.
    static PROCESS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn tarball_space_preflight_skips_when_the_download_size_is_unknown() {
        // No HEAD response (unroutable host) must not block an install.
        let temp = std::env::temp_dir();
        let warning = preflight_tarball_space("http://127.0.0.1:1/rocm.tar.gz", &temp, &temp)
            .expect("an unknown download size must not fail the preflight");
        assert_eq!(
            warning, None,
            "an unknown download size must not produce an extraction warning either"
        );
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

    /// The temp name keeps every extension, so a cleanup sweep over a cache
    /// directory can still tell what a leftover was going to be.
    #[test]
    fn temp_sibling_path_preserves_multi_dot_file_names() {
        let temp = temp_sibling_path(
            Path::new("/tmp/cache/sdk.tar.gz"),
            std::ffi::OsStr::new("test"),
        )
        .unwrap();
        let name = temp.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(name, "sdk.tar.gz.tmp-test");
        assert_eq!(temp.parent().unwrap(), Path::new("/tmp/cache"));
    }

    #[test]
    fn concurrent_atomic_writes_do_not_remove_a_published_destination() {
        let root = std::env::temp_dir().join(format!(
            "rocm-atomic-collision-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("sdk.tar.gz");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .enumerate()
            .map(|(writer, bytes)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    write_file_atomically_with(
                        &destination,
                        bytes,
                        |attempt| {
                            if attempt == 0 {
                                std::ffi::OsString::from("same-millisecond")
                            } else {
                                std::ffi::OsString::from(format!(
                                    "same-millisecond-{writer}-{attempt}"
                                ))
                            }
                        },
                        || {
                            before_publish.wait();
                        },
                    )
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a writer must remain published");
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
    }

    #[test]
    fn concurrent_cached_publications_use_distinct_staging_files() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-publish-collision-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.body");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|bytes| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    let staged = stage_file_for_atomic_publish(&destination, bytes)?;
                    before_publish.wait();
                    publish_staged_file(&staged, &destination)
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a cache writer must remain published");
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
        assert!(
            leftovers.is_empty(),
            "staged cache files leaked: {leftovers:?}"
        );
    }

    #[test]
    fn failed_cached_publication_preserves_destination_and_cleans_staging_file() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-publish-failure-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.body");
        fs::write(&destination, b"published").unwrap();
        let staged = stage_file_for_atomic_publish(&destination, b"replacement").unwrap();

        publish_staged_file_with(&staged, &destination, |_, _| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "simulated cache publication failure",
            ))
        })
        .expect_err("simulated cache publication failure must be returned");

        assert_eq!(fs::read(&destination).unwrap(), b"published");
        assert!(
            !staged.exists(),
            "failed publication leaked its staging file"
        );
        let _ = fs::remove_dir_all(&root);
    }

    fn cached_http_entry(generation: &str) -> CachedHttpCacheEntry {
        CachedHttpCacheEntry {
            metadata: CachedHttpMetadata {
                url: format!("https://example.invalid/{generation}"),
                etag: Some(format!("etag-{generation}")),
                last_modified: None,
                signature: Some(CachedHttpSignatureMetadata {
                    url: format!("https://example.invalid/{generation}.sig"),
                    verified_at_unix_ms: 1,
                    public_key_source: generation.to_owned(),
                }),
                fetched_at_unix_ms: 2,
            },
            body: format!("body-{generation}"),
            signature_bytes: Some(generation.as_bytes().to_vec()),
        }
    }

    #[test]
    fn concurrent_cached_http_commits_publish_one_complete_generation() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-generation-collision-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.json");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = ["first", "second"]
            .into_iter()
            .enumerate()
            .map(|(writer, generation)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    let entry = cached_http_entry(generation);
                    write_cached_http_entry_with(
                        &destination,
                        &entry,
                        |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        },
                        || {
                            before_publish.wait();
                        },
                        publish_temp_file,
                    )
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published: CachedHttpCacheEntry =
            serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert!(
            published == cached_http_entry("first") || published == cached_http_entry("second"),
            "published cache mixed generations: {published:?}"
        );
        assert!(leftovers.is_empty(), "cache commit leaked: {leftovers:?}");
    }

    #[test]
    fn failed_cached_http_commit_preserves_previous_complete_generation() {
        let root = std::env::temp_dir().join(format!(
            "rocm-cache-generation-failure-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("index.json");
        let previous = cached_http_entry("previous");
        write_cached_http_entry(&destination, &previous).unwrap();

        write_cached_http_entry_with(
            &destination,
            &cached_http_entry("replacement"),
            |attempt| OsString::from(format!("commit-failure-{attempt}")),
            || {},
            |tmp, path| {
                let staged: CachedHttpCacheEntry =
                    serde_json::from_slice(&fs::read(tmp).unwrap()).unwrap();
                assert_eq!(staged, cached_http_entry("replacement"));
                assert_eq!(path, destination);
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "simulated cache generation commit failure",
                ))
            },
        )
        .expect_err("simulated commit failure must be returned");

        let preserved: CachedHttpCacheEntry =
            serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(preserved, previous);
        assert_eq!(leftovers, vec![OsString::from("index.json")]);
    }

    #[test]
    fn failed_atomic_replace_preserves_destination_and_cleans_temp() {
        let root = std::env::temp_dir().join(format!(
            "rocm-atomic-replace-failure-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("sdk.tar.gz");
        fs::write(&destination, b"published").unwrap();

        write_file_atomically_with_publish(
            &destination,
            b"replacement",
            |attempt| OsString::from(format!("replace-failure-{attempt}")),
            || {},
            |tmp, path| {
                assert_eq!(fs::read(tmp).unwrap(), b"replacement");
                assert_eq!(path, destination);
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "simulated atomic replacement failure",
                ))
            },
        )
        .expect_err("simulated replacement failure must be returned");

        assert_eq!(fs::read(&destination).unwrap(), b"published");
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(leftovers, vec![OsString::from("sdk.tar.gz")]);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_temp_name_preserves_non_unicode_file_name_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let file_name = std::ffi::OsString::from_vec(b"sdk-\xff.tar.gz".to_vec());
        let destination = Path::new("/tmp").join(&file_name);
        let temp = temp_sibling_path(&destination, std::ffi::OsStr::new("collision")).unwrap();

        let mut expected = file_name.into_vec();
        expected.extend_from_slice(b".tmp-collision");
        assert_eq!(temp.file_name().unwrap().as_bytes(), expected);
    }

    /// Regression: a failed write must not leave a `.tmp-*` scratch file
    /// behind. The name is unique per attempt, so before this an orphan
    /// accumulated per retry — and when the failure is a full disk, those
    /// orphans are exactly what keeps it full.
    ///
    /// Provokes the failure by pointing the destination at a non-empty
    /// directory: the temp file is written, then neither the rename nor the
    /// replace fallback can succeed. Portable, unlike an out-of-space test.
    #[test]
    fn write_file_atomically_cleans_up_temp_when_the_rename_fails() {
        let root = std::env::temp_dir().join(format!(
            "rocm-atomic-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let occupied = root.join("sdk.tar.gz");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        write_file_atomically(&occupied, b"payload")
            .expect_err("renaming onto a non-empty directory should fail");

        let leftovers: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert!(
            leftovers.is_empty(),
            "failed write left temp files behind: {leftovers:?}"
        );
    }

    /// Mirrors the `/dev/shm` reproduction from the original report: a genuine
    /// ENOSPC, not a rename failure standing in for one.
    ///
    /// Ignored by default because it fills `/dev/shm`, which is shared with
    /// anything else on the host, so it is not safe to run concurrently. Run
    /// with `cargo test -p rocm -- --ignored write_file_atomically_cleans_up`.
    #[test]
    #[ignore = "fills /dev/shm to provoke ENOSPC; not safe to run concurrently"]
    fn write_file_atomically_cleans_up_temp_on_write_failure() {
        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        }
        let dir = shm.join(format!("rocm-enospc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("artifact.tar.gz");
        // Larger than the tmpfs, so the write is guaranteed to hit ENOSPC.
        let payload = vec![0u8; 256 * 1024 * 1024];

        let mut failures = Vec::new();
        for _ in 0..2 {
            write_file_atomically(&dest, &payload)
                .expect_err("writing past the end of the filesystem should fail");
            failures.push(
                fs::read_dir(&dir)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            );
        }
        let destination_exists = dest.exists();
        let _ = fs::remove_dir_all(&dir);

        for leftovers in &failures {
            assert!(
                leftovers.is_empty(),
                "failed write left files behind: {leftovers:?}"
            );
        }
        assert!(
            !destination_exists,
            "destination must not exist after failure"
        );
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
    fn release_channel_prefers_stable_versions() {
        let versions = vec![
            "7.11.0".to_owned(),
            "7.12.0".to_owned(),
            "7.13.0a20260326".to_owned(),
        ];
        assert_eq!(
            select_latest_version(&versions, TheRockChannel::Release),
            Some("7.12.0".to_owned())
        );
    }

    #[test]
    fn release_channel_rejects_prerelease_only_versions() {
        let versions = vec!["7.13.0a20260326".to_owned(), "7.14.0rc1".to_owned()];
        assert_eq!(
            select_latest_version(&versions, TheRockChannel::Release),
            None
        );
    }

    #[test]
    fn canonical_channels_use_only_their_aggregate_streams() {
        let release = canonical_source(TheRockChannel::Release);
        assert_eq!(
            release.wheel_index,
            "https://repo.amd.com/rocm/whl-multi-arch"
        );
        assert_eq!(
            release.tarball_catalog,
            "https://repo.amd.com/rocm/tarball/"
        );
        assert_eq!(release.layout_generation, "multi-arch-v2");

        let nightly = canonical_source(TheRockChannel::Nightly);
        assert_eq!(
            nightly.wheel_index,
            "https://rocm.nightlies.amd.com/whl-multi-arch"
        );
        assert_eq!(
            nightly.tarball_catalog,
            "https://rocm.nightlies.amd.com/tarball/"
        );
        assert_eq!(nightly.layout_generation, "multi-arch-v2");
    }

    #[test]
    fn nightly_accepts_future_prerelease_major_without_cli_changes() {
        let selected = select_matching_pip_package_versions(
            TheRockChannel::Nightly,
            &["10.1.0a20260822".to_owned()],
            &["2.12.0+rocm10.1.0a20260822".to_owned()],
            &["0.27.0+rocm10.1.0a20260822".to_owned()],
            &["2.12.0+rocm10.1.0a20260822".to_owned()],
            None,
        )
        .expect("future nightly major should resolve");

        assert_eq!(selected.rocm, "10.1.0a20260822");
    }

    #[test]
    fn tarball_selection_never_crosses_channels() {
        let platform = platform_tarball_token();
        let files = vec![
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-7.14.0.tar.gz"),
                mtime: 1.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.1.0a20260822.tar.gz"),
                mtime: 2.0,
            },
        ];

        assert_eq!(
            select_tarball_candidate(&files, TheRockChannel::Release, "gfx120X-all")
                .map(|(_, version)| version),
            Some("7.14.0".to_owned())
        );
        assert_eq!(
            select_tarball_candidate(&files, TheRockChannel::Nightly, "gfx120X-all")
                .map(|(_, version)| version),
            Some("10.1.0a20260822".to_owned())
        );
    }

    #[test]
    fn canonical_provenance_reports_required_dry_run_fields() {
        let source = canonical_source(TheRockChannel::Nightly);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Nightly,
            source.wheel_index,
            source.layout_generation,
            "10.1.0a20260822",
        );

        assert!(output.contains("channel: nightly"));
        assert!(output.contains("canonical_source: https://rocm.nightlies.amd.com/whl-multi-arch"));
        assert!(output.contains("selected_rocm_version: 10.1.0a20260822"));
        assert!(output.contains("build_date: 2026-08-22"));
        assert!(output.contains("source_layout_generation: multi-arch-v2"));
    }

    #[test]
    fn aggregate_device_target_maps_aliases_and_preserves_published_chips() {
        assert_eq!(
            canonical_aggregate_device_target("gfx943", "gfx94X-dcgpu").as_deref(),
            Some("gfx942")
        );
        assert_eq!(
            canonical_aggregate_device_target("gfx1151", "gfx1151").as_deref(),
            Some("gfx1151")
        );
        assert_eq!(
            canonical_aggregate_device_target("gfx1103", "gfx110X-all").as_deref(),
            Some("gfx1103")
        );
        assert_eq!(
            canonical_aggregate_device_target("gfx1201", "gfx120X-all").as_deref(),
            Some("gfx1201")
        );
        assert_eq!(
            canonical_aggregate_device_target("gfx1151", "gfx120X-all"),
            None
        );

        let composition = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec![
                "rocm[libraries,devel,device-gfx1103]==7.14.1".to_owned(),
                "torch==2.11.0+rocm7.14.1".to_owned(),
            ],
        };
        assert_eq!(
            wheel_composition_device_target(Some(&composition)),
            Some("gfx1103"),
            "update planning must recover the installed target without probing the current host"
        );
    }

    #[test]
    fn aggregate_wheel_install_requires_one_detected_device_payload() {
        assert_eq!(
            required_aggregate_device_target(Some("gfx943"), "gfx94X-dcgpu").unwrap(),
            "gfx942"
        );

        let missing = required_aggregate_device_target(None, "gfx94X-dcgpu")
            .unwrap_err()
            .to_string();
        assert!(missing.contains("refusing to install every device payload"));

        let mismatched = required_aggregate_device_target(Some("gfx1151"), "gfx120X-all")
            .unwrap_err()
            .to_string();
        assert!(mismatched.contains("does not match resolved family"));
    }

    #[test]
    fn same_version_old_composition_cache_requires_repair() {
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
        let required = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec![
                "rocm[libraries,devel,device-gfx942]==7.14.0".to_owned(),
                "torch==2.11.0+rocm7.14.0".to_owned(),
                "torchvision==0.26.0+rocm7.14.0".to_owned(),
                "torchaudio==2.11.0+rocm7.14.0".to_owned(),
            ],
        };
        let target_runtime_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &required);

        assert_eq!(old_cache.wheel_composition, None);
        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key,),
            RuntimeFreshness::RepairAvailable
        );

        old_cache.wheel_composition = Some(required.clone());
        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key,),
            RuntimeFreshness::RepairAvailable,
            "a manifest with the right packages but the legacy identity still needs side-by-side migration"
        );

        old_cache.runtime_key = target_runtime_key.clone();
        assert_eq!(
            runtime_freshness(&old_cache, "7.14.0", Some(&required), &target_runtime_key,),
            RuntimeFreshness::UpToDate
        );
    }

    #[test]
    fn completed_composition_repair_suppresses_repeat_repair() {
        let required = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx942]==7.14.0".to_owned()],
        };
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
        let manifests = vec![source.clone(), replacement];

        assert_eq!(
            runtime_freshness_with_manifests(
                &manifests,
                &source,
                "7.14.0",
                Some(&required),
                &target_runtime_key,
            ),
            RuntimeFreshness::UpToDate,
        );
    }

    #[test]
    fn unknown_aggregate_layout_is_rejected_clearly() {
        let error =
            validate_aggregate_index_layout("<html><a href=\"gfx120X-all/\">legacy</a></html>")
                .unwrap_err()
                .to_string();
        assert!(error.contains("unknown canonical TheRock aggregate index layout"));
    }

    #[test]
    fn pip_runtime_installs_pinned_target_complete_stack_from_aggregate_index() {
        let package_versions = TheRockPipPackageVersions {
            rocm: "7.13.0a20260513".to_owned(),
            torch: "2.10.0+rocm7.13.0a20260513".to_owned(),
            torchvision: "0.25.0+rocm7.13.0a20260513".to_owned(),
            torchaudio: "2.10.0+rocm7.13.0a20260513".to_owned(),
            compatibility_key: "7.13.0a20260513".to_owned(),
        };
        let package_specs =
            therock_pip_package_specs(&package_versions, "libraries,devel,device-gfx942");

        assert_eq!(
            package_specs,
            vec![
                "rocm[libraries,devel,device-gfx942]==7.13.0a20260513".to_owned(),
                "torch==2.10.0+rocm7.13.0a20260513".to_owned(),
                "torchvision==0.25.0+rocm7.13.0a20260513".to_owned(),
                "torchaudio==2.10.0+rocm7.13.0a20260513".to_owned(),
            ]
        );
    }

    /// The downloaded archive is removed once it has been unpacked; keeping it
    /// would double the disk cost of every installed SDK version.
    #[test]
    fn extracting_the_sdk_archive_removes_it() -> Result<()> {
        // Extraction spawns bare-name `tar` via `PATH`; hold the shared env lock so a
        // concurrent test that temporarily narrows `PATH` cannot make it fail to launch.
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, _paths) = test_paths("discard-archive");
        let cache = root.join("cache");
        let payload_dir = root.join("payload");
        fs::create_dir_all(&cache)?;
        fs::create_dir_all(&payload_dir)?;
        fs::write(payload_dir.join("marker.txt"), b"sdk")?;

        let archive = cache.join("therock-sdk.tar.gz");
        let tar = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&payload_dir)
            .arg("marker.txt")
            .status()?;
        if !tar.success() {
            eprintln!("skipping: tar unavailable on this host");
            let _ = fs::remove_dir_all(&root);
            return Ok(());
        }
        assert!(archive.is_file(), "archive fixture should exist");

        let target = root.join("install");
        fs::create_dir_all(&target)?;
        extract_tarball_and_discard_archive(&archive, &target)?;

        assert!(
            target.join("marker.txt").is_file(),
            "the archive contents should have been extracted"
        );
        assert!(
            !archive.exists(),
            "the archive should be removed once unpacked, found {}",
            archive.display()
        );

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn pip_runtime_selects_latest_common_rocm_suffix_not_latest_rocm_package() {
        let rocm_versions = vec![
            "7.13.0".to_owned(),
            "7.13.1".to_owned(),
            "7.14.0".to_owned(),
        ];
        let torch_versions = vec![
            "2.9.1+rocm7.13.1".to_owned(),
            "2.10.0+rocm7.13.1".to_owned(),
        ];
        let torchvision_versions = vec![
            "0.24.0+rocm7.13.1".to_owned(),
            "0.25.0+rocm7.13.1".to_owned(),
        ];
        let torchaudio_versions = vec![
            "2.9.0+rocm7.13.1".to_owned(),
            "2.10.0+rocm7.13.1".to_owned(),
        ];

        let selected = select_matching_pip_package_versions(
            TheRockChannel::Release,
            &rocm_versions,
            &torch_versions,
            &torchvision_versions,
            &torchaudio_versions,
            None,
        )
        .expect("expected compatible package set");

        assert_eq!(selected.rocm, "7.13.1");
        assert_eq!(selected.torch, "2.10.0+rocm7.13.1");
        assert_eq!(selected.torchvision, "0.25.0+rocm7.13.1");
        assert_eq!(selected.torchaudio, "2.10.0+rocm7.13.1");
    }

    #[test]
    fn pip_runtime_rejects_date_only_rocm_suffix_matches() {
        let rocm_versions = vec!["7.14.0a20260602".to_owned()];
        let torch_versions = vec!["2.10.0+rocm7.13.0a20260602".to_owned()];
        let torchvision_versions = vec!["0.25.0+rocm7.13.0a20260602".to_owned()];
        let torchaudio_versions = vec!["2.10.0+rocm7.13.0a20260602".to_owned()];

        assert!(
            select_matching_pip_package_versions(
                TheRockChannel::Release,
                &rocm_versions,
                &torch_versions,
                &torchvision_versions,
                &torchaudio_versions,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn pip_runtime_selects_requested_build_date_stack() -> Result<()> {
        let rocm_versions = vec![
            "7.13.0a20260604".to_owned(),
            "7.13.0a20260605".to_owned(),
            "7.13.0a20260606".to_owned(),
        ];
        let torch_versions = vec![
            "2.10.0+rocm7.13.0a20260605".to_owned(),
            "2.10.0+rocm7.13.0a20260606".to_owned(),
        ];
        let torchvision_versions = vec![
            "0.25.0+rocm7.13.0a20260605".to_owned(),
            "0.25.0+rocm7.13.0a20260606".to_owned(),
        ];
        let torchaudio_versions = vec![
            "2.10.0+rocm7.13.0a20260605".to_owned(),
            "2.10.0+rocm7.13.0a20260606".to_owned(),
        ];
        let selector = RuntimeVersionSelector::build_date("06052026")?;

        let selected = select_matching_pip_package_versions(
            TheRockChannel::Release,
            &rocm_versions,
            &torch_versions,
            &torchvision_versions,
            &torchaudio_versions,
            Some(&selector),
        )
        .expect("expected requested build-date package set");

        assert_eq!(selected.rocm, "7.13.0a20260605");
        assert_eq!(selected.torch, "2.10.0+rocm7.13.0a20260605");
        assert_eq!(
            selector,
            RuntimeVersionSelector::BuildDate("2026-06-05".to_owned())
        );
        Ok(())
    }

    #[test]
    fn pip_runtime_rejects_requested_build_date_without_matching_stack() -> Result<()> {
        let rocm_versions = vec!["7.13.0a20260605".to_owned()];
        let torch_versions = vec!["2.10.0+rocm7.13.0a20260606".to_owned()];
        let torchvision_versions = vec!["0.25.0+rocm7.13.0a20260606".to_owned()];
        let torchaudio_versions = vec!["2.10.0+rocm7.13.0a20260606".to_owned()];
        let selector = RuntimeVersionSelector::build_date("2026-06-05")?;

        assert!(
            select_matching_pip_package_versions(
                TheRockChannel::Release,
                &rocm_versions,
                &torch_versions,
                &torchvision_versions,
                &torchaudio_versions,
                Some(&selector),
            )
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn simple_index_parser_strips_wheel_tags_decodes_plus_and_filters_platform() {
        let compatibility = WheelCompatibility {
            python_tag: "cp312".to_owned(),
            platform_tags: vec!["win_amd64".to_owned(), "any".to_owned()],
        };
        let html = r#"
            <a href="torch-2.10.0%2Brocm7.13.0a20260513-cp312-cp312-win_amd64.whl">torch-2.10.0%2Brocm7.13.0a20260513-cp312-cp312-win_amd64.whl</a>
            <a href="torch-2.11.0%2Brocm7.13.0a20260514-cp313-cp313-win_amd64.whl">torch-2.11.0%2Brocm7.13.0a20260514-cp313-cp313-win_amd64.whl</a>
            <a href="torch-2.12.0+rocm7.13.0a20260515-cp312-cp312-linux_x86_64.whl">torch-2.12.0+rocm7.13.0a20260515-cp312-cp312-linux_x86_64.whl</a>
        "#;

        assert_eq!(
            parse_simple_index_versions(html, "torch", Some(&compatibility)),
            vec!["2.10.0+rocm7.13.0a20260513".to_owned()]
        );
    }

    #[test]
    fn python_venv_args_use_python_default_linking() {
        let args = python_venv_args(Path::new("/mnt/d/path/to/rocm"));

        assert!(!args.iter().any(|arg| arg == "--copies"));
        assert_eq!(args.last().map(String::as_str), Some("/mnt/d/path/to/rocm"));
    }

    #[test]
    fn python_venv_args_target_install_root() {
        let args = python_venv_args(Path::new("/mnt/envs/my-env"));
        assert_eq!(args, vec!["-m", "venv", "/mnt/envs/my-env"]);
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
            "rocm-cli-linked-runtimes-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
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
        // the uv cache shares a filesystem with the environment it populates.
        assert!(managed_uv_cache_dir(&paths.data_dir).starts_with(&paths.data_dir));
    }

    #[test]
    fn uv_cache_does_not_follow_a_prefix_install_root() {
        // Documents a known gap rather than an intended behavior: `--prefix` relocates
        // install_root only, while the uv cache stays keyed off the data dir. When the two
        // land on different filesystems uv falls back to copying. Tracked separately; see
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
    fn managed_python_defaults_to_312() {
        assert_eq!(DEFAULT_MANAGED_PYTHON_VERSION, "3.12");
    }

    #[test]
    fn managed_python_manifest_round_trips() -> Result<()> {
        let (root, paths) = test_paths("managed-python-manifest");
        let manifest = ManagedPythonManifest {
            executable: paths
                .data_dir
                .join("tools")
                .join("python")
                .join("python.exe"),
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };

        save_managed_python_manifest(&paths, &manifest)?;
        let loaded = load_managed_python_manifest(&paths)?.expect("manifest should load");

        fs::remove_dir_all(root).ok();
        assert_eq!(loaded.executable, manifest.executable);
        assert_eq!(loaded.version, "3.12");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn python_launcher_prefers_path_python_before_saved_managed_python() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-prefers-path");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        let path_python = write_fake_python_with_venv(&bin_dir, "python")?;
        let managed_python = paths.data_dir.join("tools").join("python").join("python");
        fs::create_dir_all(managed_python.parent().expect("managed python parent"))?;
        fs::write(&managed_python, "not used")?;
        let manifest = ManagedPythonManifest {
            executable: managed_python,
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };
        save_managed_python_manifest(&paths, &manifest)?;
        // Keep the search hermetic: including the real PATH lets a genuine cp312
        // python (present on CI) win over the fake one and breaks the
        // executable assertion. The fake alone is all this test needs.
        let launcher = resolve_python_launcher_in(
            &paths,
            &PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir],
            },
        )?;
        assert_eq!(launcher.source, "path");
        assert!(
            launcher.executable.is_absolute(),
            "PATH launcher should resolve to an absolute executable: {}",
            launcher.executable.display()
        );
        let launcher_path = launcher
            .executable
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        let expected_path = path_python
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        assert_eq!(launcher_path, expected_path);
        assert!(path_python.exists());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn python_venv_probe_temp_root_uses_windows_temp_env() -> Result<()> {
        if !runtime_is_windows() {
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, _paths) = test_paths("python-probe-temp-root");
        let temp_root = root.join("Temp");
        fs::create_dir_all(&temp_root)?;
        let old_temp = std::env::var_os("TEMP");
        let old_tmp = std::env::var_os("TMP");
        let old_localappdata = std::env::var_os("LOCALAPPDATA");
        unsafe {
            std::env::set_var("TEMP", &temp_root);
            std::env::remove_var("TMP");
            std::env::remove_var("LOCALAPPDATA");
        }
        let probe_root = python_venv_probe_temp_root();
        unsafe {
            match old_temp {
                Some(value) => std::env::set_var("TEMP", value),
                None => std::env::remove_var("TEMP"),
            }
            match old_tmp {
                Some(value) => std::env::set_var("TMP", value),
                None => std::env::remove_var("TMP"),
            }
            match old_localappdata {
                Some(value) => std::env::set_var("LOCALAPPDATA", value),
                None => std::env::remove_var("LOCALAPPDATA"),
            }
        }
        let probe_root = probe_root?;
        assert!(
            probe_root.starts_with(&temp_root),
            "probe root should stay under TEMP: {} not under {}",
            probe_root.display(),
            temp_root.display()
        );
        assert!(
            !probe_root.to_string_lossy().starts_with("/tmp/"),
            "Windows probe root must not use Unix /tmp: {}",
            probe_root.display()
        );
        fs::remove_dir_all(&probe_root).ok();
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn python_launcher_prefers_path_python_over_managed_when_venv_capable() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-path-over-managed");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        let path_python = write_fake_python_with_venv(&bin_dir, "python3")?;
        let managed_dir = paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let managed_python = write_fake_python_with_venv(&managed_dir, "python")?;
        let manifest = ManagedPythonManifest {
            executable: managed_python,
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };
        save_managed_python_manifest(&paths, &manifest)?;
        let launcher = resolve_python_launcher_in(
            &paths,
            &PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir],
            },
        )?;

        assert_eq!(launcher.source, "path");
        assert!(path_python.exists());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// The interpreter search must look only where it is told to look.
    ///
    /// This is the property that keeps the resolver out of the process
    /// environment. While it read `PATH` itself, the only way to steer it was to
    /// overwrite `PATH` for the whole process — which, under `cargo test`, also
    /// hid `tar` and every other PATH-resolved binary from the unrelated tests
    /// sharing that process.
    #[cfg(unix)]
    #[test]
    fn python_path_search_only_uses_the_given_directories() -> Result<()> {
        let (root, _paths) = test_paths("python-search-scope");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        write_fake_python_with_venv(&bin_dir, "python3")?;

        assert!(
            python_path_candidates(&[]).is_empty(),
            "an empty search list must yield no candidates even though the real PATH has a python"
        );
        let candidates = python_path_candidates(std::slice::from_ref(&bin_dir));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.starts_with(&bin_dir)),
            "the search must stay inside the given directories: {candidates:?}"
        );
        assert_eq!(candidates.len(), 1, "expected exactly the fixture python");

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[cfg(unix)]
    fn write_fake_python_with_venv(dir: &Path, name: &str) -> Result<PathBuf> {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        let script = r#"#!/bin/sh
if [ "$1" = "-c" ]; then
  echo cp312
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  /bin/mkdir -p "$3/bin"
  /bin/cat > "$3/bin/python" <<'PY'
#!/bin/sh
echo Python 3.12.10
PY
  /bin/chmod +x "$3/bin/python"
  exit 0
fi
echo Python 3.12.10
"#;
        fs::write(&path, script)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        Ok(path)
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
    fn aggregate_wheel_runtime_key_includes_package_composition() {
        let base = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx942]==7.14.0".to_owned()],
        };
        let changed = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx950]==7.14.0".to_owned()],
        };

        let base_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &base);
        let changed_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &changed);

        assert!(base_key.starts_with("release-wheel-multi-arch-7-14-0-"));
        assert_ne!(base_key, changed_key);
    }

    #[test]
    fn aggregate_wheel_resolution_hint_does_not_recommend_family_override() {
        let hint = canonical_wheel_resolution_hint(TheRockChannel::Release);
        assert!(!hint.contains("--family"));
        assert!(hint.contains("--channel nightly"));
        if !runtime_is_windows() {
            assert!(hint.contains("--format tarball"));
        }
    }

    #[test]
    fn stable_provenance_uses_neutral_build_date_wording() {
        let source = canonical_source(TheRockChannel::Release);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Release,
            source.wheel_index,
            source.layout_generation,
            "7.14.0",
        );
        assert!(output.contains("build_date: not encoded in stable version"));
        assert!(!output.contains("not published"));
    }

    #[test]
    fn metadata_cache_paths_stay_under_rocm_cli_cache() {
        let (root, paths) = test_paths("metadata-cache-paths");
        let (body, metadata) = metadata_cache_paths(&paths, "simple-index:https://example.invalid");

        assert!(body.starts_with(paths.cache_dir.join("therock").join("metadata")));
        assert_eq!(
            body.extension().and_then(|value| value.to_str()),
            Some("body")
        );
        assert_eq!(
            metadata.extension().and_then(|value| value.to_str()),
            Some("json")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_paths_are_sidecars() {
        let (root, paths) = test_paths("metadata-signature-paths");
        let (body, _) = metadata_cache_paths(&paths, "simple-index:https://example.invalid");

        assert_eq!(
            metadata_signature_url("https://example.invalid/index").as_str(),
            "https://example.invalid/index.sig"
        );
        assert_eq!(
            metadata_signature_path(&body)
                .extension()
                .and_then(|value| value.to_str()),
            Some("sig")
        );
        assert!(metadata_signature_path(&body).starts_with(&paths.cache_dir));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_policy_requires_public_key_when_enabled() {
        let (root, paths) = test_paths("metadata-signature-requires-key");
        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: None,
        };
        let temp_key = paths.cache_dir.join("metadata-key.pem");

        let error = with_metadata_public_key(&policy, &temp_key, |_path, _source| Ok(()))
            .unwrap_err()
            .to_string();

        assert!(error.contains("ROCM_CLI_METADATA_PUBLIC_KEY_PATH"));
        assert!(error.contains("ROCM_CLI_METADATA_PUBLIC_KEY_PEM"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn metadata_signature_policy_writes_inline_public_key_temporarily() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-inline-key");
        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: Some(
                "-----BEGIN PUBLIC KEY-----\nunit-test\n-----END PUBLIC KEY-----\n".to_owned(),
            ),
        };
        let temp_key = paths.cache_dir.join("metadata-key.pem");

        let observed = with_metadata_public_key(&policy, &temp_key, |path, source| {
            assert_eq!(source, "env-pem");
            assert_eq!(
                fs::read_to_string(path)?,
                "-----BEGIN PUBLIC KEY-----\nunit-test\n-----END PUBLIC KEY-----\n"
            );
            Ok(path.to_path_buf())
        })?
        .expect("inline key should be active");

        assert_eq!(observed.parent(), temp_key.parent());
        assert!(
            observed
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("metadata-key.pem.tmp-")),
            "inline key must use a reserved sibling path: {}",
            observed.display()
        );
        assert!(!observed.exists());
        assert!(!temp_key.exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn metadata_policy_uses_pinned_key_and_requires_by_default() {
        // A pinned trust root with no env inputs: verification becomes required
        // by default and the pinned PEM is used.
        let pinned = "-----BEGIN PUBLIC KEY-----\npinned\n-----END PUBLIC KEY-----\n";
        let policy = MetadataSignaturePolicy::resolve(false, None, None, Some(pinned.to_owned()));
        assert!(policy.required);
        assert!(policy.active());
        assert_eq!(policy.public_key_pem.as_deref(), Some(pinned));
        assert!(policy.public_key_path.is_none());
    }

    #[test]
    fn metadata_policy_env_key_overrides_pinned_key() {
        // An explicit env PEM wins over the pinned root (escape hatch), and does
        // not force `required` on its own.
        let pinned = "-----BEGIN PUBLIC KEY-----\npinned\n-----END PUBLIC KEY-----\n";
        let env_pem = "-----BEGIN PUBLIC KEY-----\nenv\n-----END PUBLIC KEY-----\n";
        let policy = MetadataSignaturePolicy::resolve(
            false,
            None,
            Some(env_pem.to_owned()),
            Some(pinned.to_owned()),
        );
        assert_eq!(policy.public_key_pem.as_deref(), Some(env_pem));

        let env_path = PathBuf::from("/keys/metadata.pem");
        let policy = MetadataSignaturePolicy::resolve(
            false,
            Some(env_path.clone()),
            None,
            Some(pinned.to_owned()),
        );
        assert_eq!(policy.public_key_path, Some(env_path));
        assert!(policy.public_key_pem.is_none());
    }

    #[test]
    fn metadata_policy_without_pinned_key_preserves_optin_behavior() {
        // Empty pinned sentinel + no env inputs: verification stays inactive,
        // exactly as before pinning was introduced.
        let policy = MetadataSignaturePolicy::resolve(false, None, None, None);
        assert!(!policy.required);
        assert!(!policy.active());

        // `ROCM_CLI_REQUIRE_METADATA_SIGNATURE=1` alone still activates it.
        let policy = MetadataSignaturePolicy::resolve(true, None, None, None);
        assert!(policy.required);
    }

    #[test]
    fn metadata_cache_revalidation_requires_cached_signature_when_policy_is_active() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-revalidate");
        fs::create_dir_all(&paths.cache_dir)?;
        let signature_path = paths.cache_dir.join("index.sig");
        let unsigned_metadata = CachedHttpMetadata {
            url: "https://example.invalid/index".to_owned(),
            etag: Some("etag".to_owned()),
            last_modified: None,
            signature: None,
            fetched_at_unix_ms: 1,
        };
        let signed_metadata = CachedHttpMetadata {
            signature: Some(CachedHttpSignatureMetadata {
                url: "https://example.invalid/index.sig".to_owned(),
                verified_at_unix_ms: 2,
                public_key_source: "path".to_owned(),
            }),
            ..unsigned_metadata.clone()
        };
        let inactive_policy = MetadataSignaturePolicy::default();
        let active_policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: None,
            public_key_pem: Some("key".to_owned()),
        };

        assert!(metadata_cache_can_revalidate(
            &unsigned_metadata,
            &inactive_policy,
            &signature_path
        ));
        assert!(!metadata_cache_can_revalidate(
            &unsigned_metadata,
            &active_policy,
            &signature_path
        ));
        assert!(!metadata_cache_can_revalidate(
            &signed_metadata,
            &active_policy,
            &signature_path
        ));

        fs::write(&signature_path, "signature")?;
        assert!(metadata_cache_can_revalidate(
            &signed_metadata,
            &active_policy,
            &signature_path
        ));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn metadata_signature_verification_accepts_generated_key_and_rejects_tamper() -> Result<()> {
        let (root, paths) = test_paths("metadata-signature-generated-key");
        fs::create_dir_all(&paths.cache_dir)?;
        let private_key = paths.cache_dir.join("metadata-private.pem");
        let public_key = paths.cache_dir.join("metadata-public.pem");
        let payload_path = paths.cache_dir.join("index.body");
        let signature_path = paths.cache_dir.join("index.sig");
        let temp_key = paths.cache_dir.join("metadata-public.tmp.pem");

        generate_test_signing_key(&private_key, &public_key)?;
        fs::write(&payload_path, "version = 1\n")?;
        sign_test_payload(&private_key, &payload_path, &signature_path)?;

        let policy = MetadataSignaturePolicy {
            required: true,
            public_key_path: Some(public_key),
            public_key_pem: None,
        };
        verify_cached_metadata_signature(&policy, &payload_path, &signature_path, &temp_key)?;

        fs::write(&payload_path, "version = 2\n")?;
        let error =
            verify_cached_metadata_signature(&policy, &payload_path, &signature_path, &temp_key)
                .unwrap_err()
                .to_string();

        assert!(error.contains("metadata signature verification failed"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn http_header_value_uses_last_response_header_block() {
        let headers =
            "HTTP/2 302\r\netag: old\r\n\r\nHTTP/2 200\r\nETag: new\r\nLast-Modified: today\r\n";

        assert_eq!(http_header_value(headers, "etag").as_deref(), Some("new"));
        assert_eq!(
            http_header_value(headers, "last-modified").as_deref(),
            Some("today")
        );
    }

    #[test]
    fn windows_child_path_maps_ape_drive_paths() {
        assert_eq!(
            windows_child_path(Path::new("/D/path/to/rocm-cli/file.ps1")),
            r"D:\path\to\rocm-cli\file.ps1"
        );
        assert_eq!(windows_child_path(Path::new("/c")), r"C:\");
    }

    #[test]
    fn native_http_download_and_get_round_trip_without_powershell() -> Result<()> {
        use std::net::TcpListener;
        use std::thread;

        // Serve a fixed body from a localhost HTTP/1.1 server so the request
        // exercises the native `ureq` transport that `http_get`/`download_file`
        // now use on every platform. This runs under `cargo test` on the
        // windows-latest CI job, where `runtime_is_windows()` is true and the
        // removed PowerShell/ExecutionPolicy-Bypass backend used to run — so it
        // verifies the native replacement on real Windows.
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let body = b"native-http-smoke-body".to_vec();
        let served = body.clone();
        // Two requests: one for download_file, one for http_get.
        let server = thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    served.len()
                )?;
                stream.write_all(&served)?;
                stream.flush()?;
            }
            Ok(())
        });

        let url = format!("http://127.0.0.1:{port}/artifact.bin");

        let temp = workspace_test_artifact_dir().join(format!(
            "native-http-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&temp)?;
        let destination = temp.join("artifact.bin");

        download_file(&url, &destination)?;
        assert_eq!(fs::read(&destination)?, body);

        let response = http_get(&url, &[], Some(5))?;
        assert_eq!(response.status, 200);
        assert_eq!(response.body, body);

        server.join().expect("localhost server thread panicked")?;
        let _ = fs::remove_dir_all(&temp);
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
    fn resolve_family_uses_managed_runtime_before_host_detection() -> Result<()> {
        if std::env::var("ROCM_CLI_THEROCK_FAMILY")
            .ok()
            .and_then(|value| normalize_therock_family(&value))
            .is_some()
        {
            return Ok(());
        }

        let (root, paths) = test_paths("resolve-family-managed-runtime");
        let manifest = test_runtime_manifest("active", "therock-release:gfx120X-all", 1);
        write_test_runtime_manifest(&paths, &manifest)?;

        let resolution = resolve_family(&paths, None)?;

        assert_eq!(resolution.family, "gfx120X-all");
        assert_eq!(resolution.source, "managed-runtime");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn family_resolution_hint_for_auto_detected_points_at_family_flag() {
        let hint = family_resolution_hint("host", "gfx950-dcgpu", TheRockChannel::Release, "wheel");

        assert!(hint.contains("auto-detected GPU family `gfx950-dcgpu`"));
        assert!(hint.contains("--family <FAMILY>"));
        // Lists recognized families the user can pass instead.
        assert!(hint.contains("gfx110X-all"));
        // Points at the other channel as an escape hatch.
        assert!(hint.contains("--channel nightly"));
    }

    #[test]
    fn family_resolution_hint_for_user_supplied_frames_around_requested_family() {
        let hint =
            family_resolution_hint("manifest", "gfx110X-all", TheRockChannel::Release, "wheel");

        assert!(hint.contains("requested package family `gfx110X-all`"));
        // A family the user named themselves is not described as auto-detected.
        assert!(!hint.contains("auto-detected"));
        assert!(!hint.contains("--family <FAMILY>"));
        assert!(hint.contains("--channel nightly"));
    }

    #[test]
    fn family_resolution_hint_suggests_release_channel_from_nightly() {
        let hint = family_resolution_hint("host", "gfx950-dcgpu", TheRockChannel::Nightly, "wheel");

        assert!(hint.contains("--channel release"));
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

    #[test]
    fn parses_rocm_sdk_probe_contract() -> Result<()> {
        let root_path = if cfg!(windows) {
            PathBuf::from(r"C:\venv\Lib\site-packages\_rocm_sdk_devel")
        } else {
            PathBuf::from("/tmp/venv/lib/python3.12/site-packages/_rocm_sdk_devel")
        };
        let bin_path = root_path.join("bin");
        let cmake_path = root_path.join("lib").join("cmake");
        let site_packages = root_path
            .parent()
            .expect("test root has a parent")
            .display()
            .to_string();
        let payload = serde_json::json!({
            "import_ok": true,
            "rocm_sdk_version": "7.13.0a20260423",
            "site_packages": site_packages,
            "root_path": root_path,
            "bin_path": bin_path,
            "cmake_path": cmake_path,
            "runtime_roots": [root_path],
            "bin_paths": [bin_path],
            "library_paths": [root_path.join("lib")],
            "default_target_family": "gfx1151",
            "available_target_families": ["gfx1151"],
            "resolved_target_family": "gfx1151",
            "packages": [{"name": "rocm", "version": "7.13.0a20260423"}],
            "library_shortnames": ["amdhip64", "hipblas"],
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [root_path.join("bin").join("amdhip64_7.dll")]},
                {"shortname": "hipblas", "paths": [root_path.join("bin").join("hipblas.dll")]}
            ],
            "error": null
        })
        .to_string();
        let probe = parse_rocm_sdk_probe(&payload)?;

        assert!(probe.import_ok);
        assert_eq!(probe.rocm_sdk_version.as_deref(), Some("7.13.0a20260423"));
        assert_eq!(probe.resolved_target_family.as_deref(), Some("gfx1151"));
        assert_eq!(
            probe
                .root_path
                .as_deref()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str()),
            Some("_rocm_sdk_devel")
        );
        assert_eq!(
            probe
                .bin_path
                .as_deref()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str()),
            Some("bin")
        );
        assert_eq!(probe.available_target_families, vec!["gfx1151"]);
        assert_eq!(probe.packages[0].name, "rocm");
        assert!(probe.library_shortnames.contains(&"amdhip64".to_owned()));
        assert_eq!(probe.resolved_libraries.len(), 2);
        Ok(())
    }

    #[test]
    fn runtime_only_rocm_sdk_probe_validates_without_devel_root() -> Result<()> {
        let (root, _paths) = test_paths("runtime-only-probe");
        let site_packages = root.join("venv").join("Lib").join("site-packages");
        let core_root = site_packages.join("_rocm_sdk_core");
        let core_bin = core_root.join("bin");
        let libraries_root = site_packages.join("_rocm_sdk_libraries_gfx120X_all");
        let libraries_bin = libraries_root.join("bin");
        fs::create_dir_all(&core_bin)?;
        fs::create_dir_all(&libraries_bin)?;
        let amdhip = core_bin.join("amdhip64_7.dll");
        let hipblas = libraries_bin.join("hipblas.dll");
        fs::write(&amdhip, b"test")?;
        fs::write(&hipblas, b"test")?;
        let payload = serde_json::json!({
            "import_ok": true,
            "rocm_sdk_version": "7.13.0a20260416",
            "site_packages": site_packages,
            "root_path": core_root,
            "bin_path": core_bin,
            "cmake_path": null,
            "runtime_roots": [core_root, libraries_root],
            "bin_paths": [core_bin, libraries_bin],
            "library_paths": [core_bin, libraries_bin],
            "default_target_family": "gfx120X-all",
            "available_target_families": ["gfx120X-all"],
            "resolved_target_family": "gfx120X-all",
            "root_path_error": "ModuleNotFoundError: rocm_sdk_devel is not installed",
            "packages": [
                {"name": "rocm", "version": "7.13.0a20260416"},
                {"name": "rocm-sdk-core", "version": "7.13.0a20260416"},
                {"name": "rocm-sdk-libraries-gfx120X-all", "version": "7.13.0a20260416"}
            ],
            "library_shortnames": ["amdhip64", "hipblas"],
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip]},
                {"shortname": "hipblas", "paths": [hipblas]}
            ],
            "error": null
        })
        .to_string();

        let probe = parse_rocm_sdk_probe(&payload)?;
        validate_rocm_sdk_runtime_probe(&probe)?;
        let _ = fs::remove_dir_all(root);

        assert!(probe.import_ok);
        assert_eq!(probe.runtime_roots.len(), 2);
        assert_eq!(probe.bin_paths.len(), 2);
        assert_eq!(probe.resolved_target_family.as_deref(), Some("gfx120X-all"));
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn install_sdk_rejects_tarball_on_windows_before_resolution() {
        let root = workspace_test_artifact_dir()
            .join(format!("rocm-cli-therock-test-{}", unix_time_millis()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };

        let error = install_sdk(&paths, "release", "tarball", None, None, None, true)
            .unwrap_err()
            .to_string();

        assert!(error.contains("tarball installs are not supported on Windows"));
        assert!(error.contains("rocm install sdk --format wheel"));
    }

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-cli-therock-test-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("therock")
    }

    fn generate_test_signing_key(private_key: &Path, public_key: &Path) -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        fs::write(private_key, private_pem.as_bytes())?;
        fs::write(public_key, public_pem.as_bytes())?;
        Ok(())
    }

    fn sign_test_payload(private_key: &Path, payload: &Path, signature: &Path) -> Result<()> {
        let private_pem = fs::read_to_string(private_key)?;
        let payload_bytes = fs::read(payload)?;
        let produced = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload_bytes)?;
        fs::write(signature, produced)?;
        Ok(())
    }

    fn test_runtime_manifest(
        runtime_key: &str,
        runtime_id: &str,
        installed_at_unix_ms: u128,
    ) -> InstalledRuntimeManifest {
        InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: runtime_id.to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: runtime_id
                .split_once(':')
                .map_or_else(|| "gfx120X-all".to_owned(), |(_, family)| family.to_owned()),
            family_source: "test".to_owned(),
            version: "7.13.0a20260416".to_owned(),
            install_root: PathBuf::from("runtime-root"),
            selected_artifact_url: "https://example.invalid/rocm".to_owned(),
            index_url: Some("https://example.invalid/simple".to_owned()),
            tarball_file_name: None,
            python_launcher: Some("python".to_owned()),
            python_executable: Some("python".to_owned()),
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            installed_at_unix_ms,
        }
    }

    fn write_test_runtime_manifest(
        paths: &AppPaths,
        manifest: &InstalledRuntimeManifest,
    ) -> Result<()> {
        let path = runtime_manifest_path(paths, &manifest.runtime_key);
        fs::create_dir_all(path.parent().expect("manifest path should have parent"))?;
        fs::write(path, serde_json::to_vec_pretty(manifest)?)?;
        Ok(())
    }

    #[test]
    fn runtime_version_display_mentions_embedded_build_date() {
        assert_eq!(
            runtime_version_display("7.14.0a20260601"),
            "7.14.0a20260601 (build 2026-06-01)"
        );
        assert_eq!(
            runtime_version_display("2.11.0+rocm7.13.0a20260416"),
            "2.11.0+rocm7.13.0a20260416 (build 2026-04-16)"
        );
        assert_eq!(runtime_version_display("7.14.0"), "7.14.0");
        assert_eq!(
            runtime_version_build_date("7.14.0a20260230"),
            None,
            "invalid calendar dates should not be displayed"
        );
    }

    #[test]
    fn runtime_device_probe_without_kernel_error_field_still_parses() {
        // Output produced before the kernel probe existed must not become a parse
        // failure: an older runtime's probe is still a valid "no kernel attempted".
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":true,"torch_version":"2.11.0","hip_version":"7.13",
                "device_count":8,"error":null}"#,
        )
        .expect("probe without kernel_error should parse");

        assert_eq!(probe.device_count, Some(8));
        assert_eq!(probe.error, None);
        assert_eq!(probe.kernel_error, None);
    }

    #[test]
    fn runtime_device_probe_keeps_kernel_failures_out_of_the_enumeration_error() {
        // The distinction the caller acts on: devices were found, so this is not a
        // "no devices" runtime, but the GPU cannot run work.
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":true,"torch_version":"2.11.0","hip_version":"7.13",
                "device_count":8,"error":null,
                "kernel_error":"RuntimeError: HIP error: invalid device function"}"#,
        )
        .expect("probe with kernel_error should parse");

        assert_eq!(probe.device_count, Some(8));
        assert_eq!(probe.error, None);
        assert_eq!(
            probe.kernel_error.as_deref(),
            Some("RuntimeError: HIP error: invalid device function")
        );
    }

    #[test]
    fn runtime_device_probe_reports_import_failures_only_as_enumeration_errors() {
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":false,"torch_version":null,"hip_version":null,
                "device_count":null,"error":"ImportError: no module named torch",
                "kernel_error":null}"#,
        )
        .expect("failed-import probe should parse");

        assert!(!probe.import_ok);
        assert_eq!(probe.device_count, None);
        assert_eq!(probe.kernel_error, None);
        assert_eq!(
            probe.error.as_deref(),
            Some("ImportError: no module named torch")
        );
    }

    #[test]
    fn runtime_device_probe_script_guards_the_kernel_behind_successful_enumeration() {
        // The script is the contract: a kernel must never be launched when the
        // import or enumeration already failed, or when there is no device to launch
        // it on. Getting this wrong turns a "no devices" runtime into a crash.
        assert!(
            RUNTIME_DEVICE_PROBE_SCRIPT
                .contains(r#"if out["error"] is None and (out["device_count"] or 0) > 0:"#),
            "the kernel attempt must be gated on a clean enumeration with devices"
        );

        let guard = RUNTIME_DEVICE_PROBE_SCRIPT
            .split_once(r#"if out["error"] is None"#)
            .expect("script should contain the kernel guard")
            .0;
        assert!(
            !guard.contains("device=\"cuda\"") && !guard.contains("synchronize"),
            "no kernel work may run before the guard"
        );
    }

    #[test]
    fn runtime_device_probe_script_records_kernel_failures_in_their_own_field() {
        let kernel_section = RUNTIME_DEVICE_PROBE_SCRIPT
            .split_once(r#"if out["error"] is None"#)
            .expect("script should contain the kernel guard")
            .1;

        // Allocate, mutate, and synchronize: an unusable GPU commonly survives the
        // allocation and only faults once work is actually launched and awaited.
        assert!(kernel_section.contains(r#"torch.ones(32, device="cuda")"#));
        assert!(kernel_section.contains("probe.add_(1.0)"));
        assert!(kernel_section.contains("torch.cuda.synchronize()"));

        assert!(
            kernel_section.contains(r#"out["kernel_error"] = type(exc).__name__"#),
            "a kernel failure must be recorded in kernel_error"
        );
        assert!(
            !kernel_section.contains(r#"out["error"] ="#),
            "the kernel attempt must never overwrite the enumeration error"
        );
        assert!(
            !kernel_section.contains(r#"out["device_count"] ="#),
            "a kernel failure must preserve the enumerated device count"
        );
    }
}
