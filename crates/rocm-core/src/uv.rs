// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Acquisition and invocation helpers for the [`uv`](https://github.com/astral-sh/uv)
//! package manager.
//!
//! `uv` replaces the previous `python -m venv` + `python -m ensurepip` +
//! `python -m pip install` flow used to provision managed runtimes. This module owns
//! downloading a standalone `uv` binary into the managed cache and the small set of
//! argument/environment helpers shared by `apps/rocm` and the engine crates (both of
//! which depend only on `rocm-core`).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::runtime::{
    managed_tools_dir, managed_uv_cache_dir, runtime_is_windows, runtime_os_name,
};
use crate::{AppPaths, download_file_to_path, unix_time_millis};

/// Default network timeout, in seconds, applied to `uv` HTTP operations.
pub const DEFAULT_UV_TIMEOUT_SECS: u64 = 600;

/// Environment variable consulted to point at a preinstalled `uv` binary, bypassing the
/// managed download (used by orchestrators and for offline/air-gapped hosts).
pub const UV_BINARY_ENV: &str = "ROCM_CLI_UV_BINARY";

/// Environment variable used to pin the downloaded `uv` release (e.g. `0.8.4`). Defaults
/// to the latest published release.
pub const UV_VERSION_ENV: &str = "ROCM_CLI_UV_VERSION";

/// Environment variable used to tune the `uv` network timeout, in seconds. Falls back to
/// the legacy `ROCM_CLI_PIP_TIMEOUT_SECS` for compatibility.
pub const UV_TIMEOUT_ENV: &str = "ROCM_CLI_UV_TIMEOUT_SECS";
const LEGACY_PIP_TIMEOUT_ENV: &str = "ROCM_CLI_PIP_TIMEOUT_SECS";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedUvManifest {
    version: String,
    asset: String,
    source_url: String,
    executable: PathBuf,
    installed_at_unix_ms: u128,
}

/// The platform-specific file name of the `uv` executable.
pub const fn uv_binary_name() -> &'static str {
    if runtime_is_windows() { "uv.exe" } else { "uv" }
}

/// The network timeout applied to `uv` operations, honoring `ROCM_CLI_UV_TIMEOUT_SECS`
/// then the legacy `ROCM_CLI_PIP_TIMEOUT_SECS`.
pub fn uv_http_timeout_secs() -> u64 {
    env_secs(UV_TIMEOUT_ENV)
        .or_else(|| env_secs(LEGACY_PIP_TIMEOUT_ENV))
        .unwrap_or(DEFAULT_UV_TIMEOUT_SECS)
}

/// Environment variable `uv` reads to locate its content-addressed cache.
pub const UV_CACHE_DIR_ENV: &str = "UV_CACHE_DIR";

/// Environment variable used to place the `uv` cache explicitly.
///
/// Namespaced like the other rocm-cli knobs in this module so that a `UV_CACHE_DIR` a
/// developer exported for unrelated Python work is distinguishable from a deliberate
/// choice about rocm-cli.
pub const UV_CACHE_DIR_OVERRIDE_ENV: &str = "ROCM_CLI_UV_CACHE_DIR";

/// Where the `uv` cache for a spawned command comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UvCacheSource {
    /// Derived from the managed data directory.
    Managed(PathBuf),
    /// Set explicitly via [`UV_CACHE_DIR_OVERRIDE_ENV`].
    Override(PathBuf),
    /// Inherited from an ambient [`UV_CACHE_DIR_ENV`] in the environment.
    Inherited(PathBuf),
}

impl UvCacheSource {
    /// The cache directory this source resolves to.
    pub fn path(&self) -> &Path {
        match self {
            Self::Managed(path) | Self::Override(path) | Self::Inherited(path) => path,
        }
    }

    /// Whether the managed colocation was bypassed, so callers can report it.
    pub const fn is_override(&self) -> bool {
        !matches!(self, Self::Managed(_))
    }
}

/// Resolve the `uv` cache directory from the managed paths and the two override
/// variables.
///
/// Precedence: [`UV_CACHE_DIR_OVERRIDE_ENV`] (a deliberate rocm-cli choice), then an
/// ambient [`UV_CACHE_DIR_ENV`] (kept so the e2e harness can share one cache across
/// scenarios), then the managed location beside the environments `uv` populates.
///
/// Blank and whitespace-only values are ignored in both cases, matching `uv_version` and
/// `env_secs` in this module — `UV_CACHE_DIR="   "` is a leftover, not a choice.
pub fn uv_cache_source(paths: &AppPaths) -> UvCacheSource {
    resolve_uv_cache_source(
        paths,
        std::env::var_os(UV_CACHE_DIR_OVERRIDE_ENV).as_deref(),
        std::env::var_os(UV_CACHE_DIR_ENV).as_deref(),
    )
}

fn resolve_uv_cache_source(
    paths: &AppPaths,
    override_dir: Option<&OsStr>,
    inherited_dir: Option<&OsStr>,
) -> UvCacheSource {
    if let Some(value) = meaningful_cache_dir(override_dir) {
        return UvCacheSource::Override(value);
    }
    if let Some(value) = meaningful_cache_dir(inherited_dir) {
        return UvCacheSource::Inherited(value);
    }
    UvCacheSource::Managed(managed_uv_cache_dir(&paths.data_dir))
}

/// A cache path is meaningful only when it is present and not blank. `OsStr` has no
/// `trim`, so trimming is done on the lossy view and the original value is kept when it
/// survives — a path is never silently rewritten.
fn meaningful_cache_dir(value: Option<&OsStr>) -> Option<PathBuf> {
    let value = value?;
    let trimmed = value.to_string_lossy();
    let trimmed = trimmed.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

/// Environment pairs to apply when spawning `uv`.
///
/// Network behavior is configured consistently (uv reads `UV_HTTP_TIMEOUT` rather than
/// accepting a `--timeout` flag) and the cache lives beside the managed environments it
/// populates.
///
/// Without a cache inside the managed root, `uv` caches under `$HOME/.cache/uv`; when
/// that is on a different filesystem from the data directory, `uv` cannot hardlink and
/// silently copies every file, so each environment carries a full duplicate of the SDK
/// and torch stack.
///
/// Note this colocates with the *data directory*, not with a `--prefix` install root; see
/// the `--prefix` caveat in `docs/manual-testing.md`.
pub fn uv_command_env(paths: &AppPaths) -> Vec<(String, String)> {
    uv_command_env_for_cache(uv_cache_source(paths))
}

fn uv_command_env_for_cache(cache: UvCacheSource) -> Vec<(String, String)> {
    vec![
        (
            "UV_HTTP_TIMEOUT".to_owned(),
            uv_http_timeout_secs().to_string(),
        ),
        (
            UV_CACHE_DIR_ENV.to_owned(),
            cache.path().to_string_lossy().into_owned(),
        ),
    ]
}

/// Arguments for `uv venv`, creating an environment at `env_root` using `python`.
pub fn uv_venv_args(python: &Path, env_root: &Path) -> Vec<String> {
    vec![
        "venv".to_owned(),
        "--python".to_owned(),
        python.to_string_lossy().into_owned(),
        env_root.to_string_lossy().into_owned(),
    ]
}

/// Base arguments for `uv pip install` targeting the interpreter `venv_python`. Callers
/// append index/cache/package arguments.
pub fn uv_pip_install_base(venv_python: &Path) -> Vec<String> {
    vec![
        "pip".to_owned(),
        "install".to_owned(),
        "--python".to_owned(),
        venv_python.to_string_lossy().into_owned(),
    ]
}

/// Arguments for `uv pip freeze` targeting the interpreter `venv_python`.
pub fn uv_pip_freeze_args(venv_python: &Path) -> Vec<String> {
    vec![
        "pip".to_owned(),
        "freeze".to_owned(),
        "--python".to_owned(),
        venv_python.to_string_lossy().into_owned(),
    ]
}

/// Arguments for `uv pip check` targeting the interpreter `venv_python`.
///
/// `--color never` is not cosmetic: this output is parsed, and `uv` colorizes its
/// findings whenever `FORCE_COLOR` / `CLICOLOR_FORCE` is exported — even when stderr is
/// a pipe rather than a terminal. The escape prefix would push every finding past the
/// parser, which reads as "the check could not run" and silently restores the very
/// behavior the check exists to catch.
pub fn uv_pip_check_args(venv_python: &Path) -> Vec<String> {
    vec![
        "pip".to_owned(),
        "check".to_owned(),
        "--color".to_owned(),
        "never".to_owned(),
        "--python".to_owned(),
        venv_python.to_string_lossy().into_owned(),
    ]
}

/// One unsatisfied `Requires-Dist` in an environment, as reported by `uv pip check`.
///
/// The managed runtime venv has more than one owner: `rocm install sdk` writes the
/// TheRock torch stack into it, and engines installed against that runtime write their
/// own pinned dependencies over the top. Either side can leave the other's requirements
/// unmet without removing any package, so "the distribution is importable" is not
/// evidence that it can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyViolation {
    /// Distribution whose `Requires-Dist` is unsatisfied (the requirer, not the
    /// package at the wrong version).
    pub requiring: String,
    /// The line `uv` reported, kept verbatim so the user sees the requirement and the
    /// installed version exactly as the resolver saw them.
    pub detail: String,
}

/// Opening frame of every line `uv pip check` emits about a distribution.
const UV_CHECK_VIOLATION_PREFIX: &str = "The package `";

/// What follows the requirer's name on an unsatisfied-requirement line specifically.
///
/// The trailing backtick is the whole point: `uv` reports several other conditions under
/// the same opening frame — a missing `WHEEL`/`METADATA` file, an unsatisfied
/// `Requires-Python`, a package with multiple installed distributions — and none of them
/// is a `Requires-Dist` a reinstall of the requirer can resolve. Only a requirement
/// `uv` quotes is one.
const UV_CHECK_REQUIREMENT_INFIX: &str = " requires `";

/// Report the unsatisfied `Requires-Dist` entries in the environment at `venv_python`.
///
/// An empty vector means every installed distribution's requirements are met. `uv pip
/// check` writes its findings to stderr and exits non-zero when it finds any, so a
/// non-zero exit with parsed findings is a successful check — but a non-zero exit with
/// nothing parsed is a failed invocation (missing interpreter, unusable `uv`) and is
/// surfaced as an error rather than mistaken for a clean environment.
pub fn check_dependencies(
    paths: &AppPaths,
    venv_python: &Path,
) -> Result<Vec<DependencyViolation>> {
    let uv = ensure_uv_binary(paths).context("failed to acquire uv binary for dependency check")?;
    let output = Command::new(&uv)
        .args(uv_pip_check_args(venv_python))
        .envs(uv_command_env(paths))
        .output()
        .with_context(|| format!("failed to run {} pip check", uv.display()))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let violations = parse_dependency_violations(&stderr);
    if output.status.success() {
        return Ok(Vec::new());
    }
    if violations.is_empty() {
        bail!(
            "`uv pip check` failed for {}: {}",
            venv_python.display(),
            stderr.trim()
        );
    }
    Ok(violations)
}

/// The violations whose requirer is `distribution`, matched case-insensitively.
///
/// Environments that host an inference engine routinely carry unrelated upstream
/// conflicts between third-party packages. Callers filter to the distribution they own
/// so those do not bury the one finding they can act on.
pub fn violations_requiring<'a>(
    violations: &'a [DependencyViolation],
    distribution: &str,
) -> Vec<&'a DependencyViolation> {
    violations
        .iter()
        .filter(|violation| violation.requiring.eq_ignore_ascii_case(distribution))
        .collect()
}

/// Pull the unsatisfied-requirement lines out of a `uv pip check` stderr body.
///
/// `uv` frames one as ``The package `<requirer>` requires `<spec>`, but `<version>` is
/// installed`` (or `` but it's not installed``), among progress lines (`Using Python
/// ...`, `Checked N packages ...`, `Found N incompatibilities`) and reports of other
/// conditions that share the opening frame. Both halves of the frame are required, so
/// neither a change in the surrounding chatter nor a differently-shaped `uv` diagnostic
/// can invent a violation.
fn parse_dependency_violations(stderr: &str) -> Vec<DependencyViolation> {
    stderr
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            let requirer = line.strip_prefix(UV_CHECK_VIOLATION_PREFIX)?;
            let (requirer, rest) = requirer.split_once('`')?;
            rest.starts_with(UV_CHECK_REQUIREMENT_INFIX)
                .then(|| DependencyViolation {
                    requiring: requirer.to_owned(),
                    detail: line.to_owned(),
                })
        })
        .collect()
}

/// Ensure a usable `uv` binary is available, downloading and caching one if needed.
/// Returns the path to the executable.
pub fn ensure_uv_binary(paths: &AppPaths) -> Result<PathBuf> {
    if let Some(path) = uv_binary_override() {
        return Ok(path);
    }

    let version = uv_version();
    let asset = uv_asset_name()?;
    let install_dir = managed_tools_dir(&paths.data_dir)
        .join("uv")
        .join(slug(&version));
    let binary_name = uv_binary_name();

    if let Some(existing) = find_binary_in(&install_dir, binary_name)
        && uv_binary_is_usable(&existing)
    {
        return Ok(existing);
    }

    let url = uv_download_url(&version, &asset);
    let archive_path = paths
        .cache_dir
        .join("tools")
        .join("uv")
        .join(slug(&version))
        .join(&asset);
    eprintln!("Downloading uv ({version}) from {url}");
    download_file_to_path(
        &url,
        &archive_path,
        Duration::from_secs(uv_http_timeout_secs()),
    )
    .with_context(|| format!("failed to download uv from {url}"))?;

    let staging = install_dir.with_extension(format!("tmp-{}", unix_time_millis()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("failed to create {}", staging.display()))?;
    extract_archive(&archive_path, &staging)
        .with_context(|| format!("failed to extract uv archive {}", archive_path.display()))?;

    let staged_binary = find_binary_in(&staging, binary_name).with_context(|| {
        format!(
            "uv archive {} did not contain a `{binary_name}` executable",
            archive_path.display()
        )
    })?;
    make_executable(&staged_binary)?;

    let _ = std::fs::remove_dir_all(&install_dir);
    if let Some(parent) = install_dir.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::rename(&staging, &install_dir).or_else(|_| {
        let _ = std::fs::remove_dir_all(&install_dir);
        std::fs::rename(&staging, &install_dir)
    })?;

    let binary = find_binary_in(&install_dir, binary_name).with_context(|| {
        format!(
            "uv executable missing after install at {}",
            install_dir.display()
        )
    })?;
    if !uv_binary_is_usable(&binary) {
        bail!("downloaded uv at {} is not runnable", binary.display());
    }

    let manifest = ManagedUvManifest {
        version,
        asset,
        source_url: url,
        executable: binary.clone(),
        installed_at_unix_ms: unix_time_millis(),
    };
    write_uv_manifest(paths, &manifest);
    let _ = std::fs::remove_file(&archive_path);

    Ok(binary)
}

fn uv_binary_override() -> Option<PathBuf> {
    let value = std::env::var_os(UV_BINARY_ENV)?;
    if value.is_empty() {
        return None;
    }
    let path = PathBuf::from(value);
    path.is_file().then_some(path)
}

fn uv_version() -> String {
    std::env::var(UV_VERSION_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "latest".to_owned())
}

fn uv_asset_name() -> Result<String> {
    let triple = match (runtime_os_name(), std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        ("windows", "aarch64") => "aarch64-pc-windows-msvc",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        (os, arch) => bail!("unsupported platform for uv download: {os}/{arch}"),
    };
    let extension = if runtime_is_windows() {
        "zip"
    } else {
        "tar.gz"
    };
    Ok(format!("uv-{triple}.{extension}"))
}

fn uv_download_url(version: &str, asset: &str) -> String {
    if version == "latest" {
        format!("https://github.com/astral-sh/uv/releases/latest/download/{asset}")
    } else {
        format!("https://github.com/astral-sh/uv/releases/download/{version}/{asset}")
    }
}

fn extract_archive(archive_path: &Path, target_dir: &Path) -> Result<()> {
    // System `tar` handles both `.tar.gz` (-xf auto-detects gzip) and `.zip` (bsdtar on
    // Windows 10+), avoiding extra archive crates in rocm-core.
    let status = Command::new("tar")
        .arg("-xf")
        .arg(archive_path)
        .arg("-C")
        .arg(target_dir)
        .status()
        .with_context(|| format!("failed to launch tar to extract {}", archive_path.display()))?;
    if !status.success() {
        bail!(
            "tar exited with {status} while extracting {}",
            archive_path.display()
        );
    }
    Ok(())
}

fn find_binary_in(dir: &Path, name: &str) -> Option<PathBuf> {
    let direct = dir.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && let Some(found) = find_binary_in(&path, name)
        {
            return Some(found);
        } else if path.is_file() && path.file_name().and_then(|value| value.to_str()) == Some(name)
        {
            return Some(path);
        }
    }
    None
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("failed to mark {} executable", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn uv_binary_is_usable(path: &Path) -> bool {
    Command::new(path)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn write_uv_manifest(paths: &AppPaths, manifest: &ManagedUvManifest) {
    let registry = managed_tools_dir(&paths.data_dir).join("registry");
    if std::fs::create_dir_all(&registry).is_err() {
        return;
    }
    if let Ok(bytes) = serde_json::to_vec_pretty(manifest) {
        let _ = std::fs::write(registry.join("uv.json"), bytes);
    }
}

fn env_secs(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn slug(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::case_sensitive_file_extension_comparisons)] // asset names are internally generated and always lowercase
    fn asset_name_has_archive_extension() {
        // Whatever the host, the asset is one of the two known archive kinds.
        let asset = uv_asset_name().expect("supported host platform for tests");
        assert!(
            asset.ends_with(".tar.gz") || asset.ends_with(".zip"),
            "{asset}"
        );
        assert!(asset.starts_with("uv-"), "{asset}");
    }

    #[test]
    fn download_url_pins_explicit_version() {
        assert_eq!(
            uv_download_url("0.8.4", "uv-x86_64-unknown-linux-gnu.tar.gz"),
            "https://github.com/astral-sh/uv/releases/download/0.8.4/uv-x86_64-unknown-linux-gnu.tar.gz"
        );
    }

    #[test]
    fn download_url_uses_latest_redirect() {
        assert_eq!(
            uv_download_url("latest", "uv-x86_64-unknown-linux-gnu.tar.gz"),
            "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
        );
    }

    #[test]
    fn venv_args_target_python_and_root() {
        let args = uv_venv_args(Path::new("/py/bin/python3"), Path::new("/envs/run"));
        assert_eq!(
            args,
            vec!["venv", "--python", "/py/bin/python3", "/envs/run"]
        );
    }

    #[test]
    fn pip_check_args_target_venv_python_and_disable_color() {
        let args = uv_pip_check_args(Path::new("/envs/run/bin/python"));
        assert_eq!(
            args,
            vec![
                "pip",
                "check",
                "--color",
                "never",
                "--python",
                "/envs/run/bin/python"
            ],
            "the output is parsed, so color must be off even under FORCE_COLOR"
        );
    }

    #[test]
    fn a_colorized_finding_is_never_silently_dropped() {
        // `--color never` is what keeps this out of the real output, but the parser must
        // not be the only thing standing between a colorized line and a false all-clear:
        // an escape-prefixed finding must not parse as a clean environment.
        let stderr = "\u{1b}[1mThe package `vllm` requires `torch==2.10.0+git8514f05`, but `2.9.1` is installed\u{1b}[0m\n";

        assert!(
            parse_dependency_violations(stderr).is_empty(),
            "a colorized line does not parse — check_dependencies reports the non-zero \
             exit as a failed invocation rather than a clean environment"
        );
    }

    #[test]
    fn an_absent_requirement_is_a_violation() {
        // Verbatim `uv pip check` output for a dependency that is missing outright
        // rather than present at the wrong version — also an unsatisfied Requires-Dist.
        let stderr =
            "The package `vllm` requires `torch==2.10.0+git8514f05`, but it's not installed\n";

        let violations = parse_dependency_violations(stderr);

        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].requiring, "vllm");
    }

    #[test]
    fn other_uv_diagnostics_sharing_the_frame_are_not_violations() {
        // Verbatim shapes `uv pip check` emits under the same `The package `x`` opening.
        // None is a Requires-Dist a reinstall of the requirer would resolve, so treating
        // them as violations would force a futile multi-gigabyte reinstall and print a
        // remedy that cannot clear the condition.
        let stderr = "\
The package `vllm` is broken or incomplete (unable to read `WHEEL` file). Consider recreating the virtualenv, or removing the package directory at: /rocm/site-packages/vllm-0.26.0.dist-info.
The package `vllm` is broken or incomplete (unable to read `METADATA`). Consider recreating the virtualenv, or removing the package directory at: /rocm/site-packages/vllm-0.26.0.dist-info.
The package `vllm` requires Python >=3.99, but `3.12.9` is installed
The package `vllm` has multiple installed distributions: /rocm/site-packages/vllm-0.26.0.dist-info
";

        assert!(
            parse_dependency_violations(stderr).is_empty(),
            "{:?}",
            parse_dependency_violations(stderr)
        );
    }

    #[test]
    fn dependency_violations_come_only_from_framed_findings() {
        // Verbatim `uv pip check` stderr for an environment where the SDK torch stack
        // was written over the engine's pinned torch, plus an unrelated
        // upstream conflict of the kind these environments routinely carry.
        let stderr = "\
Using Python 3.12.9 environment at: /rocm/runtimes/wheel/nightly-gfx94x
Checked 251 packages in 12ms
Found 2 incompatibilities
The package `vllm` requires `torch==2.10.0+git8514f05`, but `2.9.1+rocm7.14.0a20260611` is installed
The package `tilelang` requires `cloudpickle>=3.0`, but `2.2.1` is installed
";

        let violations = parse_dependency_violations(stderr);

        assert_eq!(violations.len(), 2, "{violations:?}");
        assert_eq!(violations[0].requiring, "vllm");
        assert!(
            violations[0].detail.contains("torch==2.10.0+git8514f05"),
            "the requirement and the installed version are kept verbatim: {}",
            violations[0].detail
        );
        assert_eq!(violations[1].requiring, "tilelang");
    }

    #[test]
    fn a_compatible_environment_reports_no_violations() {
        let stderr = "\
Using Python 3.12.9 environment at: /rocm/runtimes/wheel/nightly-gfx94x
Checked 251 packages in 12ms
All installed packages are compatible
";

        assert!(parse_dependency_violations(stderr).is_empty());
        assert!(parse_dependency_violations("").is_empty());
    }

    #[test]
    fn unframed_output_never_invents_a_violation() {
        // An invocation failure (bad interpreter) must not read as a finding; the
        // caller distinguishes it from a clean environment by the empty result.
        let stderr = "error: Failed to inspect Python interpreter from /nope/bin/python\n";

        assert!(parse_dependency_violations(stderr).is_empty());
    }

    #[test]
    fn violations_are_filtered_to_the_distribution_that_owns_them() {
        let violations = vec![
            DependencyViolation {
                requiring: "vLLM".to_owned(),
                detail: "The package `vLLM` requires `torch==2.10.0`, but `2.9.1` is installed"
                    .to_owned(),
            },
            DependencyViolation {
                requiring: "tilelang".to_owned(),
                detail:
                    "The package `tilelang` requires `cloudpickle>=3.0`, but `2.2.1` is installed"
                        .to_owned(),
            },
        ];

        let owned = violations_requiring(&violations, "vllm");

        assert_eq!(owned.len(), 1, "match is case-insensitive: {owned:?}");
        assert_eq!(owned[0].requiring, "vLLM");
        assert!(violations_requiring(&violations, "torch").is_empty());
    }

    #[test]
    fn pip_install_base_targets_venv_python() {
        let args = uv_pip_install_base(Path::new("/envs/run/bin/python"));
        assert_eq!(
            args,
            vec!["pip", "install", "--python", "/envs/run/bin/python"]
        );
    }

    fn test_paths(root: &str) -> AppPaths {
        AppPaths {
            config_dir: PathBuf::from(root).join("config"),
            data_dir: PathBuf::from(root),
            cache_dir: PathBuf::from(root).join("cache"),
        }
    }

    fn cache_dir_in(env: &[(String, String)]) -> Option<String> {
        env.iter()
            .find(|(key, _)| key == UV_CACHE_DIR_ENV)
            .map(|(_, value)| value.clone())
    }

    #[test]
    fn command_env_cache_dir_is_derived_from_data_dir() {
        let paths = test_paths("/managed/root");
        let env = uv_command_env_for_cache(resolve_uv_cache_source(&paths, None, None));
        assert_eq!(
            cache_dir_in(&env),
            Some(
                managed_uv_cache_dir(&paths.data_dir)
                    .to_string_lossy()
                    .into_owned()
            )
        );
        assert!(env.iter().any(|(key, _)| key == "UV_HTTP_TIMEOUT"));
    }

    #[test]
    fn command_env_keeps_an_inherited_cache_dir() {
        // The e2e harness sets UV_CACHE_DIR to share one cache across scenarios; a user can
        // do the same. Such a choice must be inherited, not overridden.
        let paths = test_paths("/managed/root");
        let source = resolve_uv_cache_source(&paths, None, Some(OsStr::new("/shared/uv-cache")));
        assert_eq!(
            source,
            UvCacheSource::Inherited(PathBuf::from("/shared/uv-cache"))
        );
        assert_eq!(
            cache_dir_in(&uv_command_env_for_cache(source)),
            Some("/shared/uv-cache".to_owned())
        );
    }

    #[test]
    fn namespaced_override_wins_over_an_ambient_uv_cache_dir() {
        // ROCM_CLI_UV_CACHE_DIR is the rocm-cli knob; a bare UV_CACHE_DIR may just be
        // exported for unrelated Python work, so the namespaced one takes precedence.
        let paths = test_paths("/managed/root");
        let source = resolve_uv_cache_source(
            &paths,
            Some(OsStr::new("/chosen/uv-cache")),
            Some(OsStr::new("/ambient/uv-cache")),
        );
        assert_eq!(
            source,
            UvCacheSource::Override(PathBuf::from("/chosen/uv-cache"))
        );
        assert!(source.is_override());
    }

    #[test]
    fn blank_cache_overrides_fall_back_to_the_managed_location() {
        // The boundary the env read actually has to survive: unset, empty, and
        // whitespace-only all mean "no choice was made".
        let paths = test_paths("/managed/root");
        let managed = UvCacheSource::Managed(managed_uv_cache_dir(&paths.data_dir));
        for blank in ["", "   ", "\t\n"] {
            assert_eq!(
                resolve_uv_cache_source(&paths, Some(OsStr::new(blank)), None),
                managed,
                "blank ROCM_CLI_UV_CACHE_DIR {blank:?} should not count as an override"
            );
            assert_eq!(
                resolve_uv_cache_source(&paths, None, Some(OsStr::new(blank))),
                managed,
                "blank UV_CACHE_DIR {blank:?} should not count as an override"
            );
        }
        assert!(!managed.is_override());
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_from_a_cache_override() {
        let paths = test_paths("/managed/root");
        assert_eq!(
            resolve_uv_cache_source(&paths, Some(OsStr::new("  /chosen/uv-cache \n")), None),
            UvCacheSource::Override(PathBuf::from("/chosen/uv-cache"))
        );
    }

    #[test]
    fn managed_cache_dir_tracks_rocm_cli_data_dir() {
        // AppPaths::with_managed_root is how ROCM_CLI_DATA_DIR reaches the cache, so
        // exercise that path rather than two hand-built AppPaths.
        let moved = test_paths("/home/user/.rocm").with_managed_root("/mnt/big/rocm", false);
        let source = resolve_uv_cache_source(&moved, None, None);
        assert_eq!(
            source,
            UvCacheSource::Managed(managed_uv_cache_dir(&moved.data_dir))
        );
        assert!(
            source.path().starts_with("/mnt/big/rocm"),
            "cache {} should sit under the relocated data dir",
            source.path().display()
        );
    }

    #[test]
    fn slug_sanitizes_unexpected_characters() {
        assert_eq!(slug("0.8.4"), "0.8.4");
        assert_eq!(slug("latest"), "latest");
        assert_eq!(slug("weird/version space"), "weird-version-space");
    }
}
