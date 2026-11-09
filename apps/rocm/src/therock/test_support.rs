// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Test-only fixtures shared by the `therock` child modules and the tests that
//! stay in `therock.rs`. Distinct from the crate-level `test_support`, which
//! holds the fixtures every module's tests share.

use super::source::FamilyResolution;
use crate::runtime_registry::{InstalledRuntimeManifest, runtime_manifest_path};
use anyhow::Result;
use rocm_core::AppPaths;
use std::fs;
use std::path::{Path, PathBuf};

/// Process environment variables set for one test and restored on drop. Every
/// caller holds `PROCESS_ENV_TEST_LOCK` for as long as it holds this.
pub(super) struct RestoredEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl RestoredEnv {
    /// Sets (`Some`) or removes (`None`) each variable, remembering what it was.
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    pub(super) fn set(vars: &[(&'static str, Option<&Path>)]) -> Self {
        let saved = vars
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in vars {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self(saved)
    }
}

impl Drop for RestoredEnv {
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

pub(super) fn family_for_test(family: &str, raw_arch: Option<&str>) -> FamilyResolution {
    FamilyResolution {
        family: family.to_owned(),
        source: "manifest".to_owned(),
        raw_arch: raw_arch.map(str::to_owned),
    }
}

pub(super) fn test_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = workspace_test_artifact_dir().join(format!(
        "rocm-cli-therock-test-{name}-{}",
        crate::test_support::unique_suffix()
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

pub(super) fn workspace_test_artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("therock")
}

pub(super) fn test_runtime_manifest(
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
        source_layout_generation: None,
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
        devel: true,
        installed_at_unix_ms,
    }
}

pub(super) fn write_test_runtime_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
) -> Result<()> {
    let path = runtime_manifest_path(paths, &manifest.runtime_key);
    fs::create_dir_all(path.parent().expect("manifest path should have parent"))?;
    fs::write(path, serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| test_paths("same-label").0);
    }
}
