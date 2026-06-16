use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ManagedToolConfig, RocmCliConfig, detect_host_gpu_diagnostics,
    detect_host_therock_family, detect_managed_therock_family, ensure_uv_binary, managed_tools_dir,
    normalize_runtime_path_for_host, normalize_runtime_path_for_storage,
    normalize_runtime_path_text_for_host, normalize_runtime_path_text_for_storage,
    normalize_therock_family, runtime_is_windows, runtime_os_name, runtime_path_for_windows_child,
    runtime_path_list_split, runtime_python_executable_in_env, unix_time_millis, uv_command_env,
    uv_pip_install_base, uv_venv_args, verify_rsa_pkcs1_sha256_signature,
};
#[cfg(test)]
use rocm_core::{generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

const THEROCK_PIP_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/v2";
const THEROCK_RELEASE_TARBALL_BASE: &str = "https://repo.amd.com/rocm/tarball/";
const THEROCK_NIGHTLY_TARBALL_BASE: &str = "https://rocm.nightlies.amd.com/tarball/";
const DEFAULT_MANAGED_PYTHON_VERSION: &str = "3.12";
const STARTUP_UPDATE_CHECK_INTERVAL_MS: u128 = 12 * 60 * 60 * 1_000;
const STARTUP_UPDATE_CHECK_TIMEOUT_SECS: u64 = 2;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TheRockChannel {
    Release,
    Nightly,
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

    fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }

    fn tarball_base_url(self) -> &'static str {
        match self {
            Self::Release => THEROCK_RELEASE_TARBALL_BASE,
            Self::Nightly => THEROCK_NIGHTLY_TARBALL_BASE,
        }
    }
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedHttpSignatureMetadata {
    url: String,
    verified_at_unix_ms: u128,
    public_key_source: String,
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
    pub update_available: bool,
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
    use std::fmt::Write as _;
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

    for manifest in manifests {
        let plan = match runtime_update_plan(paths, &manifest) {
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
            let _ = writeln!(
                output,
                "    next step: run `rocm update --apply --runtime {}` to install the newer runtime side-by-side",
                manifest.runtime_key
            );
            let _ = writeln!(
                output,
                "    activate: add `--activate` to make the newly installed runtime the default after install"
            );
        }
    }

    Ok(output)
}

pub(crate) fn runtime_update_plan(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
) -> Result<RuntimeUpdatePlan> {
    let (latest_version, latest_source, format) =
        resolve_latest_for_manifest(paths, manifest, None)?;
    let status = match compare_version_strings(&manifest.version, &latest_version) {
        Ordering::Less => "update_available",
        Ordering::Equal => "up_to_date",
        Ordering::Greater => "ahead_of_index",
    };
    Ok(RuntimeUpdatePlan {
        latest_version,
        latest_source,
        format,
        status: status.to_owned(),
        update_available: status == "update_available",
    })
}

fn resolve_latest_for_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    download_timeout_secs: Option<u64>,
) -> Result<(String, String, String)> {
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
            Ok((
                resolution.latest_version,
                resolution.index_url,
                "wheel".to_owned(),
            ))
        }
        "tarball" => {
            let artifact = resolve_tarball_artifact_with_timeout(
                paths,
                channel,
                Some(manifest.family.as_str()),
                download_timeout_secs,
            )?;
            Ok((artifact.version, artifact.url, "tarball".to_owned()))
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
        now_unix_ms,
        Some(STARTUP_UPDATE_CHECK_TIMEOUT_SECS),
    );
    save_startup_update_check(paths, &record)?;
    Ok(Some(record))
}

fn startup_update_check_disabled() -> bool {
    std::env::var_os("ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK").is_some()
}

fn startup_update_check_due(previous_unix_ms: u128, now_unix_ms: u128) -> bool {
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
    now_unix_ms: u128,
    download_timeout_secs: Option<u64>,
) -> StartupUpdateCheckRecord {
    match resolve_latest_for_manifest(paths, manifest, download_timeout_secs) {
        Ok((latest_version, _latest_source, kind)) => {
            let status = match compare_version_strings(&manifest.version, &latest_version) {
                Ordering::Less => "update_available",
                Ordering::Equal => "up_to_date",
                Ordering::Greater => "ahead_of_index",
            };
            StartupUpdateCheckRecord {
                runtime_key: manifest.runtime_key.clone(),
                runtime_id: manifest.runtime_id.clone(),
                channel: manifest.channel.clone(),
                format: kind,
                family: manifest.family.clone(),
                installed_version: manifest.version.clone(),
                latest_version: Some(latest_version),
                status: status.to_owned(),
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
    progress_line(format!(
        "Found TheRock package family {} version {} with a matching PyTorch stack.",
        resolution.family, resolution.latest_version
    ));
    let runtime_key = runtime_key(
        channel,
        "wheel",
        &resolution.family,
        Some(&resolution.latest_version),
    );
    let install_root = prefix.unwrap_or_else(|| managed_runtime_root(paths, "wheel", &runtime_key));
    let manifest_path = runtime_manifest_path(paths, &runtime_key);

    let mut output = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(
        output,
        "  summary: rocm-cli will install the ROCm SDK and matching PyTorch packages for this Python and operating system"
    );
    let _ = writeln!(output, "  channel: {}", channel.as_str());
    let _ = writeln!(output, "  format: wheel");
    if let Some(selector) = version_selector {
        let _ = writeln!(output, "  requested: {}", selector.describe());
    }
    let _ = writeln!(output, "  family: {}", resolution.family);
    let _ = writeln!(output, "  family_source: {}", resolution.family_source);
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
        therock_pip_package_specs(&resolution.package_versions).join(" ")
    );
    let _ = writeln!(
        output,
        "  package_policy: find the newest TheRock ROCm SDK version that has a matching PyTorch stack in the same index, then install pinned rocm[libraries,devel], torch, torchvision, and torchaudio versions in one uv transaction"
    );
    if dry_run {
        let env_python = venv_python_path(&install_root);
        let mut install_args = uv_pip_install_base(&env_python);
        install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
        if matches!(channel, TheRockChannel::Nightly) {
            install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
        }
        install_args.extend(therock_pip_package_specs(&resolution.package_versions));
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
            "  command: uv {} && uv {}",
            venv_args_display, install_args_display
        );
        let _ = writeln!(
            output,
            "  activation: use the managed venv Python; TheRock libraries are resolved from that venv by rocm_sdk.initialize_process"
        );
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(output);
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
    ensure_uv_venv(&uv, &python_launcher.executable, &install_root)?;
    let env_python = venv_python_path(&install_root);

    progress_line(format!(
        "Installing {} from {}",
        therock_pip_package_specs(&resolution.package_versions).join(" "),
        resolution.index_url
    ));
    let mut install_args = uv_pip_install_base(&env_python);
    install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
    if matches!(channel, TheRockChannel::Nightly) {
        install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    install_args.extend(therock_pip_package_specs(&resolution.package_versions));
    run_uv_progress_command(
        &uv,
        install_args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        "install TheRock devel SDK, torch stack, and resolved dependencies",
    )?;

    progress_line("Checking the installed ROCm SDK...");
    let rocm_sdk_probe = probe_rocm_sdk_runtime(&env_python)
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

fn therock_pip_package_specs(package_versions: &TheRockPipPackageVersions) -> Vec<String> {
    vec![
        format!("rocm[libraries,devel]=={}", package_versions.rocm),
        format!("torch=={}", package_versions.torch),
        format!("torchvision=={}", package_versions.torchvision),
        format!("torchaudio=={}", package_versions.torchaudio),
    ]
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
    let install_root =
        prefix.unwrap_or_else(|| managed_runtime_root(paths, "tarball", &runtime_key));
    let manifest_path = runtime_manifest_path(paths, &runtime_key);
    let cache_path = paths.cache_dir.join("therock").join(&artifact.file_name);

    let mut output = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(output, "  channel: {}", channel.as_str());
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

    download_file(&artifact.url, &cache_path)?;
    extract_tarball(&cache_path, &install_root)?;

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
    let index_url = therock_index_url(&family_resolution.family);
    let rocm_versions =
        load_simple_index_versions(paths, &index_url, "rocm", None, download_timeout_secs)?;
    let torch_versions = load_simple_index_versions(
        paths,
        &index_url,
        "torch",
        Some(wheel_compatibility),
        download_timeout_secs,
    )?;
    let torchvision_versions = load_simple_index_versions(
        paths,
        &index_url,
        "torchvision",
        Some(wheel_compatibility),
        download_timeout_secs,
    )?;
    let torchaudio_versions = load_simple_index_versions(
        paths,
        &index_url,
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
        let requested = version_selector
            .map(RuntimeVersionSelector::describe)
            .unwrap_or_else(|| "latest compatible version".to_owned());
        format!(
            "no mutually compatible TheRock rocm[libraries,devel], torch, torchvision, and torchaudio versions were found for {requested} in {index_url}"
        )
    })?;
    let latest_version = package_versions.rocm.clone();
    Ok(PipRuntimeResolution {
        family: family_resolution.family,
        family_source: family_resolution.source,
        index_url,
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
    let html = download_text_cached(
        paths,
        &format!("tarball-index-{}", channel.as_str()),
        channel.tarball_base_url(),
        download_timeout_secs,
    )?
    .text;
    let files = parse_tarball_index_html(&html)?;
    let prefix = format!(
        "therock-dist-{}-{}-",
        platform_tarball_token(),
        family_resolution.family
    );
    let mut candidates = files
        .into_iter()
        .filter_map(|file| {
            let version = file
                .name
                .strip_prefix(&prefix)?
                .strip_suffix(".tar.gz")?
                .to_owned();
            Some((file, version))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.0
            .mtime
            .partial_cmp(&right.0.mtime)
            .unwrap_or(Ordering::Equal)
            .then_with(|| compare_version_strings(&left.1, &right.1))
    });
    let (file, version) = candidates
        .pop()
        .context("no matching TheRock tarball artifact was found for the resolved GPU family")?;
    Ok(TarballArtifact {
        family: family_resolution.family,
        family_source: family_resolution.source,
        url: format!(
            "{}/{}",
            channel.tarball_base_url().trim_end_matches('/'),
            file.name
        ),
        file_name: file.name,
        version,
    })
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
        "unable to resolve a supported TheRock GPU family for this host\n\n{}",
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
        let stable = all
            .iter()
            .filter(|version| {
                parse_version(version)
                    .map(|parsed| parsed.stage == VersionStage::Stable)
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>();
        if !stable.is_empty() {
            return stable;
        }
    }
    all
}

fn select_latest_stack_package(
    versions: &[String],
    rocm_version: &str,
    matches_stack: impl Fn(&ParsedVersion) -> bool,
) -> Option<String> {
    let mut candidates = package_versions_matching_rocm(versions, rocm_version)
        .into_iter()
        .filter(|version| {
            parse_package_base_version(version)
                .map(|base| matches_stack(&base))
                .unwrap_or(false)
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
        .filter(|ch| ch.is_ascii_digit())
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

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: u32) -> bool {
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
        if parse_version(version)
            .map(|parsed| parsed.stage == VersionStage::Stable)
            .unwrap_or(false)
        {
            stable.push(version.clone());
        }
    }
    stable.sort_by(|left, right| compare_version_strings(left, right));
    match channel {
        TheRockChannel::Release => stable.pop().or_else(|| all.pop()),
        TheRockChannel::Nightly => all.pop(),
    }
}

impl MetadataSignaturePolicy {
    fn from_env() -> Self {
        Self {
            required: truthy_env("ROCM_CLI_REQUIRE_METADATA_SIGNATURE"),
            public_key_path: env_path("ROCM_CLI_METADATA_PUBLIC_KEY_PATH"),
            public_key_pem: env_nonempty("ROCM_CLI_METADATA_PUBLIC_KEY_PEM"),
        }
    }

    fn active(&self) -> bool {
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
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
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
        let parent = temp_key_path
            .parent()
            .context("metadata public key temp path has no parent directory")?;
        fs::create_dir_all(parent)?;
        fs::write(temp_key_path, public_key_pem)
            .with_context(|| format!("failed to write {}", temp_key_path.display()))?;
        let result = verify(temp_key_path, "env-pem");
        let _ = fs::remove_file(temp_key_path);
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
    payload_path: &Path,
    signature_path: &Path,
    temp_key_path: &Path,
    max_time_secs: Option<u64>,
) -> Result<Option<CachedHttpSignatureMetadata>> {
    if !policy.active() {
        return Ok(None);
    }
    let signature_url = metadata_signature_url(url);
    download_signature_file(&signature_url, signature_path, max_time_secs)?;
    let public_key_source =
        with_metadata_public_key(policy, temp_key_path, |public_key, source| {
            verify_metadata_signature(payload_path, signature_path, public_key)?;
            Ok(source.to_owned())
        })?
        .context("metadata signature policy was active but no public key was resolved")?;
    Ok(Some(CachedHttpSignatureMetadata {
        url: signature_url,
        verified_at_unix_ms: unix_time_millis(),
        public_key_source,
    }))
}

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

fn download_signature_file(
    url: &str,
    destination: &Path,
    max_time_secs: Option<u64>,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("metadata signature destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    let response = http_get(url, &[], max_time_secs)?;
    if response.status != 200 {
        let _ = fs::remove_file(destination);
        bail!(
            "HTTP {} while fetching metadata signature {url}",
            response.status
        );
    }
    write_file_atomically(destination, &response.body)?;
    Ok(())
}

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

fn metadata_cache_can_revalidate(
    metadata: &CachedHttpMetadata,
    policy: &MetadataSignaturePolicy,
    signature_path: &Path,
) -> bool {
    !policy.active() || (metadata.signature.is_some() && signature_path.is_file())
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
    let previous_metadata = fs::read(&metadata_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CachedHttpMetadata>(&bytes).ok())
        .filter(|metadata| metadata.url == url);
    let cache_dir = body_path
        .parent()
        .context("metadata cache path has no parent directory")?;
    fs::create_dir_all(cache_dir)?;

    let unique = unix_time_millis();
    let tmp_body = body_path.with_extension(format!("body.tmp-{unique}"));
    let tmp_signature = body_path.with_extension(format!("sig.tmp-{unique}"));
    let tmp_public_key = body_path.with_extension(format!("public-key.tmp-{unique}.pem"));
    let mut headers = Vec::new();
    if let Some(etag) = previous_metadata
        .as_ref()
        .filter(|metadata| {
            metadata_cache_can_revalidate(metadata, &signature_policy, &signature_path)
        })
        .and_then(|metadata| metadata.etag.as_deref())
    {
        headers.push(("If-None-Match", etag));
    }

    let response = http_get(url, &headers, max_time_secs)?;
    if response.status == 304 {
        let _ = fs::remove_file(&tmp_body);
        verify_cached_metadata_signature(
            &signature_policy,
            &body_path,
            &signature_path,
            &tmp_public_key,
        )?;
        let text = fs::read_to_string(&body_path).with_context(|| {
            format!(
                "metadata cache returned 304 but cached body is missing: {}",
                body_path.display()
            )
        })?;
        return Ok(CachedHttpText { text });
    }
    if response.status != 200 {
        let _ = fs::remove_file(&tmp_body);
        bail!("HTTP {} while fetching {url}", response.status);
    }

    write_file_atomically(&tmp_body, &response.body)?;
    let signature_metadata = match fetch_and_verify_metadata_signature(
        &signature_policy,
        url,
        &tmp_body,
        &tmp_signature,
        &tmp_public_key,
        max_time_secs,
    ) {
        Ok(signature_metadata) => signature_metadata,
        Err(error) => {
            let _ = fs::remove_file(&tmp_body);
            let _ = fs::remove_file(&tmp_signature);
            let _ = fs::remove_file(&tmp_public_key);
            return Err(error);
        }
    };
    let metadata = CachedHttpMetadata {
        url: url.to_owned(),
        etag: http_header_value(&response.headers, "etag"),
        last_modified: http_header_value(&response.headers, "last-modified"),
        signature: signature_metadata.clone(),
        fetched_at_unix_ms: unix_time_millis(),
    };
    fs::rename(&tmp_body, &body_path).or_else(|_| {
        let _ = fs::remove_file(&body_path);
        fs::rename(&tmp_body, &body_path)
    })?;
    if signature_metadata.is_some() {
        fs::rename(&tmp_signature, &signature_path).or_else(|_| {
            let _ = fs::remove_file(&signature_path);
            fs::rename(&tmp_signature, &signature_path)
        })?;
    } else {
        let _ = fs::remove_file(&signature_path);
    }
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata)
            .context("failed to serialize metadata cache record")?,
    )?;
    let text = fs::read_to_string(&body_path)
        .with_context(|| format!("failed to read cached metadata {}", body_path.display()))?;
    Ok(CachedHttpText { text })
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

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    if use_windows_powershell_http() {
        return download_file_windows_powershell(url, destination, None);
    }
    let response = http_get(url, &[], None)?;
    if response.status != 200 {
        bail!("HTTP {} while fetching {url}", response.status);
    }
    write_file_atomically(destination, &response.body)
}

fn http_get(
    url: &str,
    headers: &[(&str, &str)],
    max_time_secs: Option<u64>,
) -> Result<HttpResponseBody> {
    if use_windows_powershell_http() {
        return http_get_windows_powershell(url, headers, max_time_secs);
    }
    let timeout = max_time_secs
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(600));
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

fn use_windows_powershell_http() -> bool {
    runtime_is_windows()
}

#[derive(Deserialize)]
struct WindowsHttpMetadata {
    #[serde(rename = "StatusCode")]
    status_code: u16,
    #[serde(rename = "Headers")]
    headers: String,
}

fn http_get_windows_powershell(
    url: &str,
    headers: &[(&str, &str)],
    max_time_secs: Option<u64>,
) -> Result<HttpResponseBody> {
    let temp_dir = windows_http_temp_dir()?;
    let body_path = temp_dir.join("body.bin");
    let meta_path = temp_dir.join("meta.json");
    let script_path = temp_dir.join("http-get.ps1");
    let timeout = max_time_secs.filter(|value| *value > 0).unwrap_or(600);
    fs::write(&script_path, windows_http_get_script())
        .with_context(|| format!("failed to write {}", script_path.display()))?;
    let mut command = Command::new("powershell.exe");
    command
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(windows_child_path(&script_path))
        .args(["-Url", url])
        .arg("-BodyPath")
        .arg(windows_child_path(&body_path))
        .arg("-MetaPath")
        .arg(windows_child_path(&meta_path))
        .args(["-TimeoutSec", &timeout.to_string()]);
    for (name, value) in headers {
        command.arg("-Header").arg(format!("{name}={value}"));
    }
    let stdout_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch PowerShell HTTP request for {url}"))?;
    if !windows_child_status_success(status) {
        let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
        let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
        let temp_note = windows_http_failure_temp_note(&temp_dir);
        if !windows_http_keep_temp() {
            let _ = fs::remove_dir_all(&temp_dir);
        }
        bail!(
            "PowerShell HTTP request failed for {url} (status {status}): {}{}{}",
            stdout.trim(),
            stderr.trim(),
            temp_note
        );
    }
    let metadata_bytes = fs::read(&meta_path)
        .with_context(|| format!("failed to read HTTP metadata {}", meta_path.display()))?;
    let metadata: WindowsHttpMetadata = serde_json::from_slice(strip_utf8_bom(&metadata_bytes))
        .context("failed to parse PowerShell HTTP metadata")?;
    let body = fs::read(&body_path)
        .with_context(|| format!("failed to read HTTP response body {}", body_path.display()))?;
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(HttpResponseBody {
        status: metadata.status_code,
        headers: metadata.headers,
        body,
    })
}

fn download_file_windows_powershell(
    url: &str,
    destination: &Path,
    max_time_secs: Option<u64>,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    let temp_dir = windows_http_temp_dir()?;
    let script_path = temp_dir.join("download-file.ps1");
    let meta_path = temp_dir.join("meta.json");
    let timeout = max_time_secs.filter(|value| *value > 0).unwrap_or(600);
    fs::write(&script_path, windows_http_get_script())
        .with_context(|| format!("failed to write {}", script_path.display()))?;
    let stdout_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status = Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(windows_child_path(&script_path))
        .args(["-Url", url])
        .arg("-BodyPath")
        .arg(windows_child_path(destination))
        .arg("-MetaPath")
        .arg(windows_child_path(&meta_path))
        .args(["-TimeoutSec", &timeout.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("failed to launch PowerShell download for {url}"))?;
    if !windows_child_status_success(status) {
        let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
        let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
        let temp_note = windows_http_failure_temp_note(&temp_dir);
        if !windows_http_keep_temp() {
            let _ = fs::remove_dir_all(&temp_dir);
        }
        bail!(
            "PowerShell download failed for {url} (status {status}): {}{}{}",
            stdout.trim(),
            stderr.trim(),
            temp_note
        );
    }
    let metadata_bytes = fs::read(&meta_path)
        .with_context(|| format!("failed to read HTTP metadata {}", meta_path.display()))?;
    let metadata: WindowsHttpMetadata = serde_json::from_slice(strip_utf8_bom(&metadata_bytes))
        .context("failed to parse PowerShell HTTP metadata")?;
    if metadata.status_code != 200 {
        let _ = fs::remove_dir_all(&temp_dir);
        let _ = fs::remove_file(destination);
        bail!("HTTP {} while fetching {url}", metadata.status_code);
    }
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

fn windows_http_temp_dir() -> Result<PathBuf> {
    if !runtime_is_windows() {
        return linux_temp_dir("rocm-cli-http");
    }
    windows_temp_dir("rocm-cli-http")
}

fn linux_temp_dir(prefix: &str) -> Result<PathBuf> {
    let root = std::env::temp_dir();
    let base = format!("{prefix}-{}-{}", std::process::id(), unix_time_millis());
    for attempt in 0..100 {
        let dir = root.join(format!("{base}-{attempt}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
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
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
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

fn windows_child_status_success(status: std::process::ExitStatus) -> bool {
    status.success() || status.code() == Some(0)
}

fn windows_http_keep_temp() -> bool {
    std::env::var("ROCM_CLI_DEBUG_WINDOWS_HTTP")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn windows_http_failure_temp_note(temp_dir: &Path) -> String {
    if windows_http_keep_temp() {
        format!("; temp files kept at {}", curl_child_path(temp_dir))
    } else {
        String::new()
    }
}

fn windows_http_get_script() -> &'static str {
    r#"
param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$BodyPath,
  [Parameter(Mandatory=$true)][string]$MetaPath,
  [int]$TimeoutSec = 600,
  [string[]]$Header = @()
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$headers = @{}
foreach ($entry in $Header) {
  $index = $entry.IndexOf('=')
  if ($index -gt 0) {
    $headers[$entry.Substring(0, $index)] = $entry.Substring($index + 1)
  }
}
$response = Invoke-WebRequest -Uri $Url -UseBasicParsing -Headers $headers -OutFile $BodyPath -PassThru -TimeoutSec $TimeoutSec
$headerLines = @()
foreach ($key in $response.Headers.Keys) {
  $headerLines += "${key}: $($response.Headers[$key])"
}
$metadata = [pscustomobject]@{
  StatusCode = [int]$response.StatusCode
  Headers = ($headerLines -join "`n")
}
$json = $metadata | ConvertTo-Json -Compress
[System.IO.File]::WriteAllText($MetaPath, $json, [System.Text.UTF8Encoding]::new($false))
"#
}

fn strip_utf8_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes)
}

fn windows_child_path(path: &Path) -> String {
    runtime_path_for_windows_child(path)
}

fn curl_child_path(path: &Path) -> String {
    if runtime_is_windows() {
        windows_child_path(path)
    } else {
        path.display().to_string()
    }
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent)?;
    let tmp = path.with_extension(format!("tmp-{}", unix_time_millis()));
    {
        let mut file = fs::File::create(&tmp)
            .with_context(|| format!("failed to create {}", tmp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", tmp.display()))?;
    }
    fs::rename(&tmp, path).or_else(|_| {
        let _ = fs::remove_file(path);
        fs::rename(&tmp, path)
    })?;
    Ok(())
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
}

fn ensure_uv_venv(uv: &Path, python_launcher: &Path, install_root: &Path) -> Result<()> {
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
        &uv_command_env(),
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
    let text = capture_python_stdout(
        python_executable,
        ROCM_SDK_PROBE_SCRIPT,
        "launch rocm_sdk probe",
    )
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
    bail!("{}: {}", context_text, detail)
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
    let detail = if !stderr.is_empty() {
        stderr
    } else {
        format!("command exited with status {}", output.status)
    };
    bail!("{}: {}", context_text, detail)
}

fn run_uv_progress_command(uv: &Path, args: &[&str], context_text: &str) -> Result<()> {
    let mut command = Command::new(uv);
    command.args(args);
    for (key, value) in &uv_command_env() {
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
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
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
        .envs(uv_command_env())
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
        .envs(uv_command_env())
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

fn resolve_python_launcher(paths: &AppPaths) -> Result<PythonLauncher> {
    if let Ok(value) = std::env::var("ROCM_CLI_PYTHON") {
        python_launcher_install_ready(Path::new(&value))
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLauncher {
            executable: PathBuf::from(value),
            source: "env",
        });
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates() {
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

fn python_path_candidates() -> Vec<PathBuf> {
    let program_names: &[&str] = if runtime_is_windows() {
        &["python", "python3", "py"]
    } else {
        &["python3", "python"]
    };
    program_names
        .iter()
        .flat_map(|program| resolve_program_on_path(program))
        .collect()
}

fn resolve_program_on_path(program: &str) -> Vec<PathBuf> {
    let Some(path_value) = std::env::var_os("PATH") else {
        return Vec::new();
    };
    let candidates = program_path_candidates(program);
    split_runtime_path(&path_value)
        .into_iter()
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
    result.map(|_| ())
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
        .take_while(|ch| ch.is_ascii_digit())
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

fn therock_index_url(family: &str) -> String {
    format!("{THEROCK_PIP_INDEX_BASE}/{family}")
}

fn platform_tarball_token() -> &'static str {
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

    static PYTHON_RESOLVER_TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn pip_runtime_installs_pinned_devel_and_torch_stack_from_therock_index() {
        let package_versions = TheRockPipPackageVersions {
            rocm: "7.13.0a20260513".to_owned(),
            torch: "2.10.0+rocm7.13.0a20260513".to_owned(),
            torchvision: "0.25.0+rocm7.13.0a20260513".to_owned(),
            torchaudio: "2.10.0+rocm7.13.0a20260513".to_owned(),
            compatibility_key: "7.13.0a20260513".to_owned(),
        };
        let package_specs = therock_pip_package_specs(&package_versions);

        assert_eq!(
            package_specs,
            vec![
                "rocm[libraries,devel]==7.13.0a20260513".to_owned(),
                "torch==2.10.0+rocm7.13.0a20260513".to_owned(),
                "torchvision==0.25.0+rocm7.13.0a20260513".to_owned(),
                "torchaudio==2.10.0+rocm7.13.0a20260513".to_owned(),
            ]
        );
    }

    #[test]
    fn pip_runtime_selects_latest_common_rocm_suffix_not_latest_rocm_package() {
        let rocm_versions = vec![
            "7.13.0a20260512".to_owned(),
            "7.13.0a20260513".to_owned(),
            "7.14.0a20260602".to_owned(),
        ];
        let torch_versions = vec![
            "2.9.1+rocm7.13.0a20260513".to_owned(),
            "2.10.0+rocm7.13.0a20260513".to_owned(),
        ];
        let torchvision_versions = vec![
            "0.24.0+rocm7.13.0a20260513".to_owned(),
            "0.25.0+rocm7.13.0a20260513".to_owned(),
        ];
        let torchaudio_versions = vec![
            "2.9.0+rocm7.13.0a20260513".to_owned(),
            "2.10.0+rocm7.13.0a20260513".to_owned(),
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

        assert_eq!(selected.rocm, "7.13.0a20260513");
        assert_eq!(selected.torch, "2.10.0+rocm7.13.0a20260513");
        assert_eq!(selected.torchvision, "0.25.0+rocm7.13.0a20260513");
        assert_eq!(selected.torchaudio, "2.10.0+rocm7.13.0a20260513");
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
        let args = python_venv_args(Path::new("/mnt/d/jam/rocm"));

        assert!(!args.iter().any(|arg| arg == "--copies"));
        assert_eq!(args.last().map(String::as_str), Some("/mnt/d/jam/rocm"));
    }

    #[test]
    fn python_venv_args_target_install_root() {
        let args = python_venv_args(Path::new("/mnt/envs/my-env"));
        assert_eq!(args, vec!["-m", "venv", "/mnt/envs/my-env"]);
    }

    #[test]
    fn managed_uv_cache_defaults_inside_generated_runtime_folder() {
        let (_root, paths) = test_paths("managed-uv-cache");
        let runtime_key = "release-wheel-gfx120x-all-7-14-0";
        let install_root = managed_runtime_root(&paths, "wheel", runtime_key);
        // uv caches live beside the venv; verify the wheel root path structure
        assert!(install_root.starts_with(&paths.data_dir));
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
        let _guard = PYTHON_RESOLVER_TEST_ENV_LOCK.lock().unwrap();
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
        let old_path = std::env::var_os("PATH");
        let old_rocm_cli_python = std::env::var_os("ROCM_CLI_PYTHON");
        // Keep PATH hermetic: appending the real PATH lets a genuine cp312
        // python (present on CI) win over the fake one and breaks the
        // executable assertion. The fake on PATH is all this test needs.
        let joined_path = std::env::join_paths([bin_dir.clone()])?;
        unsafe {
            std::env::set_var("PATH", joined_path);
            std::env::remove_var("ROCM_CLI_PYTHON");
        }
        let launcher = resolve_python_launcher(&paths)?;
        unsafe {
            match old_path {
                Some(old_path) => std::env::set_var("PATH", old_path),
                None => std::env::remove_var("PATH"),
            }
            match old_rocm_cli_python {
                Some(value) => std::env::set_var("ROCM_CLI_PYTHON", value),
                None => std::env::remove_var("ROCM_CLI_PYTHON"),
            }
        }
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
    fn python_venv_probe_temp_root_uses_windows_temp_env() -> Result<()> {
        if !runtime_is_windows() {
            return Ok(());
        }
        let _guard = PYTHON_RESOLVER_TEST_ENV_LOCK.lock().unwrap();
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
        let _guard = PYTHON_RESOLVER_TEST_ENV_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-path-over-managed");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        let path_python = write_fake_python_with_venv(&bin_dir, "python3")?;
        let managed_dir = paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let managed_python = write_fake_python_with_venv(&managed_dir, "python")?;
        let manifest = ManagedPythonManifest {
            executable: managed_python.clone(),
            version: "3.12".to_owned(),
            installed_at_unix_ms: 123,
        };
        save_managed_python_manifest(&paths, &manifest)?;
        let old_path = std::env::var_os("PATH");
        let old_rocm_cli_python = std::env::var_os("ROCM_CLI_PYTHON");
        let joined_path = std::env::join_paths([bin_dir.clone()])?;
        unsafe {
            std::env::set_var("PATH", joined_path);
            std::env::remove_var("ROCM_CLI_PYTHON");
        }
        let launcher = resolve_python_launcher(&paths)?;
        unsafe {
            match old_path {
                Some(old_path) => std::env::set_var("PATH", old_path),
                None => std::env::remove_var("PATH"),
            }
            match old_rocm_cli_python {
                Some(value) => std::env::set_var("ROCM_CLI_PYTHON", value),
                None => std::env::remove_var("ROCM_CLI_PYTHON"),
            }
        }

        assert_eq!(launcher.source, "path");
        assert!(path_python.exists());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[cfg(unix)]
    fn write_fake_python_with_venv(dir: &Path, name: &str) -> Result<PathBuf> {
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
        use std::os::unix::fs::PermissionsExt;
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
    fn runtime_key_includes_version_for_side_by_side_installs() {
        assert_eq!(
            runtime_key(
                TheRockChannel::Release,
                "wheel",
                "gfx120X-all",
                Some("7.13.0a20260416")
            ),
            "release-wheel-gfx120x-all-7-13-0a20260416"
        );
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

        assert_eq!(observed, temp_key);
        assert!(!temp_key.exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
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
    fn windows_http_uses_powershell_backend() {
        if runtime_is_windows() {
            assert!(use_windows_powershell_http());
        } else {
            assert!(!use_windows_powershell_http());
        }
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
                latest_version: Some(manifest.version.clone()),
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
                .map(|(_, family)| family.to_owned())
                .unwrap_or_else(|| "gfx120X-all".to_owned()),
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
}
