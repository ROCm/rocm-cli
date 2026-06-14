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
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::runtime::{managed_tools_dir, runtime_is_windows, runtime_os_name};
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
pub fn uv_binary_name() -> &'static str {
    if runtime_is_windows() { "uv.exe" } else { "uv" }
}

/// The network timeout applied to `uv` operations, honoring `ROCM_CLI_UV_TIMEOUT_SECS`
/// then the legacy `ROCM_CLI_PIP_TIMEOUT_SECS`.
pub fn uv_http_timeout_secs() -> u64 {
    env_secs(UV_TIMEOUT_ENV)
        .or_else(|| env_secs(LEGACY_PIP_TIMEOUT_ENV))
        .unwrap_or(DEFAULT_UV_TIMEOUT_SECS)
}

/// Environment pairs to apply when spawning `uv` so network behavior is configured
/// consistently (uv reads `UV_HTTP_TIMEOUT` rather than accepting a `--timeout` flag).
pub fn uv_command_env() -> Vec<(String, String)> {
    vec![(
        "UV_HTTP_TIMEOUT".to_owned(),
        uv_http_timeout_secs().to_string(),
    )]
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
        .map(|status| status.success())
        .unwrap_or(false)
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
    fn pip_install_base_targets_venv_python() {
        let args = uv_pip_install_base(Path::new("/envs/run/bin/python"));
        assert_eq!(
            args,
            vec!["pip", "install", "--python", "/envs/run/bin/python"]
        );
    }

    #[test]
    fn slug_sanitizes_unexpected_characters() {
        assert_eq!(slug("0.8.4"), "0.8.4");
        assert_eq!(slug("latest"), "latest");
        assert_eq!(slug("weird/version space"), "weird-version-space");
    }
}
