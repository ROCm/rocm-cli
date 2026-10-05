// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::host_gpu::capture_optional_path_command_with_env;
use crate::{
    AppPaths, OPTIONAL_COMMAND_TIMEOUT, RocmCliConfig, extract_first_gfx_token,
    normalize_runtime_path_for_host, normalize_therock_family, runtime_is_linux,
    runtime_is_windows, runtime_python_executable_in_env,
};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

pub fn detect_managed_therock_family(paths: &AppPaths) -> Option<String> {
    newer_therock_family(
        newest_therock_family_in_manifest_dir(&paths.data_dir.join("runtimes").join("registry")),
        newest_therock_family_in_engine_manifests(paths),
    )
    .map(|(_, family)| family)
}

fn newest_therock_family_in_engine_manifests(paths: &AppPaths) -> Option<(u128, String)> {
    let engines_dir = paths.data_dir.join("engines");
    let entries = fs::read_dir(engines_dir).ok()?;
    let mut best = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        best = newer_therock_family(
            best,
            newest_therock_family_in_manifest_dir(&path.join("manifests")),
        );
    }
    best
}

fn newest_therock_family_in_manifest_dir(path: &Path) -> Option<(u128, String)> {
    let entries = fs::read_dir(path).ok()?;
    let mut best = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(record) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())
        else {
            continue;
        };
        let Some(family) = record.therock_family() else {
            continue;
        };
        best = newer_therock_family(
            best,
            Some((record.installed_at_unix_ms.unwrap_or(0), family)),
        );
    }
    best
}

pub(crate) fn detect_managed_therock_sdk_gfx_target(paths: &AppPaths) -> Option<String> {
    managed_therock_sdk_probe_candidates(&paths.data_dir.join("runtimes").join("registry"))
        .into_iter()
        .find_map(|candidate| {
            let tool = managed_sdk_tool_path(&candidate.bin_path, "rocm_agent_enumerator")?;
            let mut envs = Vec::new();
            if let Some(ld_library_path) = managed_sdk_ld_library_path(&candidate) {
                envs.push(("LD_LIBRARY_PATH", ld_library_path));
            }
            capture_optional_path_command_with_env(&tool, &[], &envs, OPTIONAL_COMMAND_TIMEOUT)
                .and_then(|output| extract_first_gfx_token(&output))
        })
}

#[derive(Debug, Clone, Default)]
pub struct ManagedRuntimeEnvironment {
    pub rocm_root: Option<PathBuf>,
    pub path_entries: Vec<PathBuf>,
    pub library_entries: Vec<PathBuf>,
}

pub fn active_managed_therock_environment(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<ManagedRuntimeEnvironment>> {
    Ok(select_active_managed_therock_record(paths, config)
        .map(|record| managed_therock_environment_from_record(&record)))
}

/// Channel (`"release"`/`"nightly"`) of the active managed TheRock runtime.
///
/// Reflects the channel recorded at install time. Returns `None` when there is no managed
/// runtime (system or legacy ROCm) or the registry record predates channel recording.
pub fn active_managed_therock_channel(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    Ok(select_active_managed_therock_record(paths, config).and_then(|record| record.channel))
}

/// The interpreter a framework probe should run, plus the loader path its torch
/// needs.
///
/// The library paths are not decoration. A TheRock runtime's torch resolves HIP
/// from a sibling `_rocm_sdk_core` package rather than from its own `torch/lib`,
/// so running the interpreter without them fails the import outright with
/// `libroctx64.so.4: cannot open shared object file`. That reads as a broken
/// runtime rather than as an unconfigured probe, which is a worse answer than
/// the silence it would replace.
#[derive(Debug, Clone)]
pub struct FrameworkInterpreter {
    pub python: PathBuf,
    pub library_paths: Vec<PathBuf>,
}

/// The active managed runtime's Python interpreter, when there is one.
///
/// `None` on an unmanaged host, and also when the runtime records an interpreter
/// that is no longer on disk — a caller that cannot spawn the interpreter is
/// better served by the ambient one than by a path that fails to execute.
///
/// Not the only "which runtime is active" selector on this path. `rocm examine`'s
/// *human* report resolves the runtime through `current_runtime_manifest`, which
/// wants an active key or exactly one matching `default_runtime_id`, whereas the
/// record chosen here falls back to the most recently installed one. With no
/// active key set the two can disagree, so `examine --json` can report a
/// framework read from a runtime the human form calls
/// `active_runtime_status: unset`. The divergence predates this function; it is
/// recorded because reading the framework through the runtime is what first made
/// it observable.
pub fn active_managed_framework_interpreter(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Option<FrameworkInterpreter> {
    let record = select_active_managed_therock_record(paths, config)?;
    let python = managed_therock_python_executable(&record)?;
    Some(FrameworkInterpreter {
        python,
        library_paths: managed_therock_environment_from_record(&record).library_entries,
    })
}

/// Prefer the interpreter the installer recorded over the conventional location
/// inside the install root: an imported or read-only runtime can record an
/// interpreter that does not sit under `install_root` at all.
fn managed_therock_python_executable(record: &TheRockFamilyManifest) -> Option<PathBuf> {
    record
        .python_executable
        .as_deref()
        // Registry paths are used verbatim as stored, so this one still needs
        // host-normalizing; the derived form below is normalized already.
        .map(normalize_runtime_path_for_host)
        .into_iter()
        .chain(
            record
                .install_root
                .as_deref()
                .map(runtime_python_executable_in_env),
        )
        .find(|candidate| candidate.is_file())
}

/// Version (e.g. `"10.0.0"`) of the active managed TheRock runtime.
///
/// Reflects the version recorded at install time. Returns `None` when there is no managed
/// runtime (system or legacy ROCm) or the registry record is malformed or hand-edited.
pub fn active_managed_therock_version(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    Ok(select_active_managed_therock_record(paths, config).and_then(|record| record.version))
}

/// Pick the active managed TheRock runtime record: the one matching
/// `config.active_runtime_key`, falling back to the most recently installed.
fn select_active_managed_therock_record(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Option<TheRockFamilyManifest> {
    let registry_dir = paths.data_dir.join("runtimes").join("registry");
    let mut records = managed_therock_environment_records(&registry_dir);
    if records.is_empty() {
        return None;
    }

    if let Some(active_key) = config.active_runtime_key.as_deref()
        && let Some((_, record)) = records.iter().find(|(_, record)| {
            record
                .runtime_key
                .as_deref()
                .is_some_and(|key| key.eq_ignore_ascii_case(active_key))
                || record
                    .runtime_id
                    .as_deref()
                    .is_some_and(|id| id.eq_ignore_ascii_case(active_key))
        })
    {
        return Some(record.clone());
    }

    records.sort_by_key(|(_, record)| std::cmp::Reverse(record.installed_at_unix_ms.unwrap_or(0)));
    records.into_iter().next().map(|(_, record)| record)
}

pub fn prepend_runtime_paths(
    entries: &[PathBuf],
    current: Option<OsString>,
) -> Result<Option<OsString>> {
    let mut parts = Vec::new();
    for entry in entries {
        push_existing_runtime_path(&mut parts, entry.clone());
    }
    if let Some(current) = current
        && !current.is_empty()
    {
        for entry in std::env::split_paths(&current) {
            push_existing_runtime_path(&mut parts, entry);
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        std::env::join_paths(parts)
            .map(Some)
            .context("failed to join runtime environment paths")
    }
}

pub(crate) fn managed_therock_environment_records(
    registry_dir: &Path,
) -> Vec<(PathBuf, TheRockFamilyManifest)> {
    let Ok(entries) = fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                return None;
            }
            let record = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())?;
            (record.looks_like_therock()
                && record.rocm_sdk.as_ref().is_some_and(|sdk| sdk.import_ok))
            .then_some((path, record))
        })
        .collect()
}

fn managed_therock_environment_from_record(
    record: &TheRockFamilyManifest,
) -> ManagedRuntimeEnvironment {
    let mut env = ManagedRuntimeEnvironment::default();
    let sdk = record.rocm_sdk.as_ref();
    env.rocm_root = sdk
        .and_then(|sdk| sdk.root_path.clone())
        .or_else(|| record.install_root.clone());

    if let Some(sdk) = sdk {
        if let Some(bin_path) = sdk.bin_path.as_ref() {
            push_existing_runtime_path(&mut env.path_entries, bin_path.clone());
        }
        for path in &sdk.bin_paths {
            push_existing_runtime_path(&mut env.path_entries, path.clone());
        }
        for path in &sdk.library_paths {
            push_existing_runtime_path(&mut env.library_entries, path.clone());
        }
        if let Some(root_path) = sdk.root_path.as_ref() {
            collect_runtime_environment_paths(root_path, &mut env);
        }
        for root_path in &sdk.runtime_roots {
            collect_runtime_environment_paths(root_path, &mut env);
        }
    }
    if let Some(install_root) = record.install_root.as_ref() {
        collect_runtime_environment_paths(install_root, &mut env);
    }
    if runtime_is_linux() {
        push_existing_runtime_path(&mut env.library_entries, PathBuf::from("/usr/lib/wsl/lib"));
    }
    env
}

fn collect_runtime_environment_paths(root: &Path, env: &mut ManagedRuntimeEnvironment) {
    for path in [
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ] {
        if !path.is_dir() {
            continue;
        }
        if path.file_name().and_then(|value| value.to_str()) == Some("bin") {
            push_existing_runtime_path(&mut env.path_entries, path.clone());
        }
        push_existing_runtime_path(&mut env.library_entries, path);
    }
}

fn push_existing_runtime_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.exists() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

fn managed_therock_sdk_probe_candidates(registry_dir: &Path) -> Vec<TheRockSdkProbeCandidate> {
    let Ok(entries) = fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(record) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())
        else {
            continue;
        };
        if !record.looks_like_therock() {
            continue;
        }
        let Some(sdk) = record.rocm_sdk else {
            continue;
        };
        if !sdk.import_ok {
            continue;
        }
        let Some(root_path) = sdk.root_path else {
            continue;
        };
        let Some(bin_path) = sdk.bin_path else {
            continue;
        };
        candidates.push(TheRockSdkProbeCandidate {
            installed_at_unix_ms: record.installed_at_unix_ms.unwrap_or(0),
            site_packages: sdk.site_packages,
            root_path,
            bin_path,
        });
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.installed_at_unix_ms));
    candidates
}

pub(crate) fn managed_sdk_tool_path(bin_path: &Path, tool: &str) -> Option<PathBuf> {
    let mut names = vec![tool.to_owned()];
    if runtime_is_windows() {
        names.push(format!("{tool}.exe"));
    }
    names.push(format!("{tool}.cmd"));
    names.push(format!("{tool}.bat"));
    names
        .into_iter()
        .map(|name| bin_path.join(name))
        .find(|path| path.is_file())
}

fn managed_sdk_ld_library_path(candidate: &TheRockSdkProbeCandidate) -> Option<OsString> {
    let mut paths = Vec::new();
    collect_sdk_library_paths(&candidate.root_path, &mut paths);
    if let Some(site_packages) = candidate.site_packages.as_deref()
        && let Ok(entries) = fs::read_dir(site_packages)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with("_rocm_sdk_") {
                collect_sdk_library_paths(&path, &mut paths);
            }
        }
    }
    let wsl_lib = PathBuf::from("/usr/lib/wsl/lib");
    if wsl_lib.is_dir() {
        paths.push(wsl_lib);
    }
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH")
        && !existing.is_empty()
    {
        paths.extend(std::env::split_paths(&existing));
    }
    if paths.is_empty() {
        None
    } else {
        std::env::join_paths(paths).ok()
    }
}

fn collect_sdk_library_paths(root: &Path, paths: &mut Vec<PathBuf>) {
    for path in [
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ] {
        if path.is_dir() {
            paths.push(path);
        }
    }
}

fn newer_therock_family(
    left: Option<(u128, String)>,
    right: Option<(u128, String)>,
) -> Option<(u128, String)> {
    match (left, right) {
        (Some(left), Some(right)) if left.0 > right.0 => Some(left),
        (Some(_) | None, Some(right)) => Some(right),
        (Some(left), None) => Some(left),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TheRockFamilyManifest {
    #[serde(default)]
    runtime_key: Option<String>,
    #[serde(default)]
    runtime_id: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    therock_family: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    pub(crate) rocm_sdk: Option<TheRockSdkProbeManifest>,
    #[serde(default)]
    pub(crate) install_root: Option<PathBuf>,
    /// Recorded by the installer. Absent on records written before it was, and
    /// on runtimes whose interpreter was never resolved.
    #[serde(default)]
    python_executable: Option<PathBuf>,
    #[serde(default)]
    pub(crate) installed_at_unix_ms: Option<u128>,
}

impl TheRockFamilyManifest {
    fn therock_family(&self) -> Option<String> {
        if !self.looks_like_therock() {
            return None;
        }
        self.therock_family
            .as_deref()
            .or(self.family.as_deref())
            .and_then(normalize_therock_family)
    }

    fn looks_like_therock(&self) -> bool {
        self.therock_family.is_some()
            || self
                .runtime_id
                .as_deref()
                .is_some_and(|runtime_id| runtime_id.to_ascii_lowercase().starts_with("therock-"))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TheRockSdkProbeManifest {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    site_packages: Option<PathBuf>,
    #[serde(default)]
    root_path: Option<PathBuf>,
    #[serde(default)]
    pub(crate) bin_path: Option<PathBuf>,
    #[serde(default)]
    runtime_roots: Vec<PathBuf>,
    #[serde(default)]
    pub(crate) bin_paths: Vec<PathBuf>,
    #[serde(default)]
    library_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct TheRockSdkProbeCandidate {
    installed_at_unix_ms: u128,
    site_packages: Option<PathBuf>,
    root_path: PathBuf,
    bin_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_python_executable_name;

    fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-core-{name}-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("core")
    }

    fn write_fake_rocm_agent_enumerator(bin_dir: &Path, target: &str) -> Result<()> {
        if cfg!(windows) {
            let path = bin_dir.join("rocm_agent_enumerator.cmd");
            fs::write(path, format!("@echo off\r\necho {target}\r\n"))?;
        } else {
            let path = bin_dir.join("rocm_agent_enumerator");
            fs::write(&path, format!("#!/bin/sh\necho {target}\n"))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            }
        }
        Ok(())
    }

    #[test]
    fn managed_therock_family_uses_runtime_manifest_not_host_mapping() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-therock-family");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("newest.json"),
            r#"{
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx1201",
                "installed_at_unix_ms": 20
            }"#,
        )?;
        fs::write(
            registry.join("older.json"),
            r#"{
                "runtime_id": "therock-release:gfx110X-all",
                "family": "gfx1103",
                "installed_at_unix_ms": 10
            }"#,
        )?;
        fs::write(
            registry.join("not-therock.json"),
            r#"{
                "runtime_id": "other-runtime",
                "family": "gfx1030",
                "installed_at_unix_ms": 30
            }"#,
        )?;

        assert_eq!(
            detect_managed_therock_family(&paths),
            Some("gfx120X-all".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_therock_family_falls_back_to_engine_env_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("engine-therock-family");
        let manifests = paths.engine_manifests_dir("vllm");
        fs::create_dir_all(&manifests)?;
        fs::write(
            manifests.join("env.json"),
            r#"{
                "runtime_id": "therock-release",
                "therock_family": "gfx1151",
                "installed_at_unix_ms": 15
            }"#,
        )?;

        assert_eq!(
            detect_managed_therock_family(&paths),
            Some("gfx1151".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_therock_family_is_none_without_therock_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("no-therock-family");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("other.json"),
            r#"{
                "runtime_id": "other-runtime",
                "family": "gfx1201",
                "installed_at_unix_ms": 99
            }"#,
        )?;

        assert_eq!(detect_managed_therock_family(&paths), None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_detects_gfx_from_therock_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-gfx");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx1201")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(
            detect_managed_therock_sdk_gfx_target(&paths),
            Some("gfx1201".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_skips_non_therock_manifests() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-skip-non-therock");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx9999")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "external-runtime",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(detect_managed_therock_sdk_gfx_target(&paths), None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_reads_recorded_channel() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-nightly:gfx120X-all",
                "family": "gfx120X-all",
                "channel": "nightly",
                "installed_at_unix_ms": 10,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;

        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("nightly".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_is_none_without_runtime() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-none");
        let config = RocmCliConfig::default();
        assert_eq!(active_managed_therock_channel(&paths, &config)?, None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_falls_back_to_most_recent() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-recent");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_channel_record(&registry, "older", "release", 10)?;
        write_therock_channel_record(&registry, "newer", "nightly", 20)?;

        // No active_runtime_key set: the most recently installed runtime wins.
        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("nightly".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_prefers_active_runtime_key() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-active-key");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_channel_record(&registry, "older", "release", 10)?;
        write_therock_channel_record(&registry, "newer", "nightly", 20)?;

        // The active key points at the older runtime, overriding recency.
        let config = RocmCliConfig {
            active_runtime_key: Some("therock-release:older".to_owned()),
            ..RocmCliConfig::default()
        };
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("release".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// Write a registry record for a managed TheRock runtime, always planting a
    /// real interpreter at the conventional location under `install_root` so the
    /// DERIVED candidate exists on disk.
    ///
    /// `recorded_python` decides the record's `python_executable` key:
    ///
    /// - `None` writes no such key at all -- the shape of a record from before
    ///   the installer recorded one, where only the derived candidate exists.
    /// - `Some(path)` records that path verbatim, whether or not anything is
    ///   there. Planting it is the caller's business, so a caller can exercise
    ///   either side of the gate in `managed_therock_python_executable` -- which
    ///   is `is_file()`, not `exists()`.
    fn write_therock_runtime_with_interpreter(
        registry: &Path,
        install_root: &Path,
        name: &str,
        recorded_python: Option<&Path>,
    ) -> Result<()> {
        let bin = install_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        fs::create_dir_all(&bin)?;
        fs::write(
            bin.join(runtime_python_executable_name()),
            b"#!/bin/sh\nexit 0\n",
        )?;
        let mut record = serde_json::json!({
            "runtime_id": format!("therock-release:{name}"),
            "runtime_key": name,
            "family": "gfx94X-dcgpu",
            "channel": "release",
            "installed_at_unix_ms": 10,
            "install_root": install_root,
            "rocm_sdk": { "import_ok": true },
        });
        if let Some(python) = recorded_python {
            record["python_executable"] = serde_json::json!(python);
        }
        fs::write(
            registry.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&record)?,
        )?;
        Ok(())
    }

    #[test]
    fn the_framework_interpreter_is_none_without_a_managed_runtime() -> Result<()> {
        // An unmanaged host must fall back to the ambient `PATH` probe, which is
        // what `None` selects for the caller.
        let (root, paths) = temp_app_paths("framework-interpreter-none");
        assert!(active_managed_framework_interpreter(&paths, &RocmCliConfig::default()).is_none());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn the_framework_interpreter_is_derived_when_none_was_recorded() -> Result<()> {
        // Records written before the installer recorded `python_executable` must
        // still resolve, from the conventional location under `install_root`.
        let (root, paths) = temp_app_paths("framework-interpreter-derived");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        let install_root = paths.data_dir.join("rt-derived");
        write_therock_runtime_with_interpreter(&registry, &install_root, "derived", None)?;

        let interpreter = active_managed_framework_interpreter(&paths, &RocmCliConfig::default())
            .expect("a managed runtime with an interpreter on disk must resolve");
        assert_eq!(
            interpreter.python,
            runtime_python_executable_in_env(&install_root)
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn a_recorded_interpreter_outside_the_install_root_wins_over_the_derived_one() -> Result<()> {
        // The branch the e2e scenario's relaxed assertion rests on: an imported
        // or read-only runtime can record an interpreter that does not sit under
        // `install_root`, and preferring the derived path would hand back the
        // wrong interpreter whenever both exist.
        let (root, paths) = temp_app_paths("framework-interpreter-recorded");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        let install_root = paths.data_dir.join("rt-recorded");
        let elsewhere = paths.data_dir.join("outside").join("venv");
        let recorded = runtime_python_executable_in_env(&elsewhere);
        fs::create_dir_all(recorded.parent().expect("an interpreter has a parent"))?;
        fs::write(&recorded, b"#!/bin/sh\nexit 0\n")?;
        // `write_therock_runtime_with_interpreter` also plants the derived
        // interpreter under `install_root`, so both candidates are on disk and
        // the `exists()` gate cannot decide this for us.
        write_therock_runtime_with_interpreter(
            &registry,
            &install_root,
            "recorded",
            Some(&recorded),
        )?;

        let interpreter = active_managed_framework_interpreter(&paths, &RocmCliConfig::default())
            .expect("a managed runtime with an interpreter on disk must resolve");
        assert_eq!(
            interpreter.python, recorded,
            "the recorded interpreter must win over the one derived from install_root"
        );
        assert!(
            !interpreter.python.starts_with(&install_root),
            "the point of the recorded path is that it need not sit under install_root"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn a_recorded_interpreter_that_is_gone_does_not_resolve() -> Result<()> {
        // Handing back a path that cannot be spawned would turn "no torch" into
        // a spawn failure; the ambient probe is the better answer, so this must
        // be `None` rather than the missing path.
        //
        // The record really carries a `python_executable`, which is the whole
        // point: with none recorded this would drive the gate on the DERIVED
        // candidate and stay green however the recorded half of it is mangled.
        let (root, paths) = temp_app_paths("framework-interpreter-missing");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        let install_root = paths.data_dir.join("rt-gone");
        let recorded = runtime_python_executable_in_env(&paths.data_dir.join("gone").join("venv"));
        write_therock_runtime_with_interpreter(&registry, &install_root, "gone", Some(&recorded))?;
        // The helper plants the derived interpreter; take it away so that BOTH
        // candidates are recorded-or-derived paths that are not on disk, and
        // neither can carry the result.
        fs::remove_dir_all(&install_root)?;
        assert!(!recorded.exists(), "the recorded interpreter must be gone");

        assert!(
            active_managed_framework_interpreter(&paths, &RocmCliConfig::default()).is_none(),
            "a recorded interpreter that is not on disk must not be handed back"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn a_recorded_interpreter_that_is_a_directory_is_passed_over() -> Result<()> {
        // `exists()` would accept a directory and hand back something that
        // cannot be spawned, which is the failure the gate exists to prevent;
        // `is_file()` passes over it and the derived interpreter answers
        // instead. Relaxing the predicate is otherwise invisible to the suite.
        let (root, paths) = temp_app_paths("framework-interpreter-dir");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        let install_root = paths.data_dir.join("rt-dir");
        let recorded = runtime_python_executable_in_env(&paths.data_dir.join("dir").join("venv"));
        fs::create_dir_all(&recorded)?;
        write_therock_runtime_with_interpreter(&registry, &install_root, "dir", Some(&recorded))?;

        let interpreter = active_managed_framework_interpreter(&paths, &RocmCliConfig::default())
            .expect("the derived interpreter is on disk, so something must resolve");
        assert_eq!(
            interpreter.python,
            runtime_python_executable_in_env(&install_root),
            "a recorded path that is a directory must not be preferred over a real interpreter"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    fn write_therock_channel_record(
        registry: &Path,
        name: &str,
        channel: &str,
        installed_at_unix_ms: u64,
    ) -> Result<()> {
        fs::write(
            registry.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": format!("therock-{channel}:{name}"),
                "family": "gfx120X-all",
                "channel": channel,
                "installed_at_unix_ms": installed_at_unix_ms,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;
        Ok(())
    }

    #[test]
    fn active_managed_therock_version_reads_recorded_version() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-version");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx120X-all",
                "version": "10.0.0",
                "installed_at_unix_ms": 10,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;

        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_version(&paths, &config)?,
            Some("10.0.0".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_version_is_none_without_runtime() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-version-none");
        let config = RocmCliConfig::default();
        assert_eq!(active_managed_therock_version(&paths, &config)?, None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_version_falls_back_to_most_recent() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-version-recent");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_version_record(&registry, "older", "7.13.0", 10)?;
        write_therock_version_record(&registry, "newer", "10.0.0", 20)?;

        // No active_runtime_key set: the most recently installed runtime wins.
        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_version(&paths, &config)?,
            Some("10.0.0".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_version_prefers_active_runtime_key() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-version-active-key");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_version_record(&registry, "older", "7.13.0", 10)?;
        write_therock_version_record(&registry, "newer", "10.0.0", 20)?;

        // The active key points at the older runtime, overriding recency.
        let config = RocmCliConfig {
            active_runtime_key: Some("therock-release:older".to_owned()),
            ..RocmCliConfig::default()
        };
        assert_eq!(
            active_managed_therock_version(&paths, &config)?,
            Some("7.13.0".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    fn write_therock_version_record(
        registry: &Path,
        name: &str,
        version: &str,
        installed_at_unix_ms: u64,
    ) -> Result<()> {
        fs::write(
            registry.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": format!("therock-release:{name}"),
                "family": "gfx120X-all",
                "version": version,
                "installed_at_unix_ms": installed_at_unix_ms,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;
        Ok(())
    }
}
