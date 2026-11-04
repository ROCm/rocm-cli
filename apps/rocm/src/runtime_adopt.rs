// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Registering a runtime that was not installed by `rocm install sdk`:
//! importing a manifest written elsewhere, adopting an existing Python
//! environment by probing it, and recovering the registration of the runtime
//! `rocm setup` created.
//!
//! Relocated from `runtimes_cmd.rs` with no behavior change. `runtimes_cmd.rs`
//! calls into this module for `rocm runtimes import` and `rocm runtimes adopt`,
//! and re-exports `newest_installed_runtime_manifest`,
//! `recover_setup_runtime_registration` and `setup_runtime_paths` for the
//! modules that import them from there. The path comparison both adopt and
//! uninstall need lives in `runtime_registry.rs`, so this module does not
//! depend on `runtimes_cmd.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, managed_pip_cache_dir, runtime_python_executable_in_env};

use crate::runtime_registry::{
    paths_equivalent, runtime_manifest_path, runtime_registry_dir,
    validate_runtime_manifest_for_activation,
};
use crate::therock;

pub(crate) fn import_runtime_manifest(
    paths: &AppPaths,
    manifest_path: &Path,
    replace: bool,
) -> Result<therock::InstalledRuntimeManifest> {
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let mut manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_key",
            manifest_path.display()
        );
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!(
            "runtime manifest {} has an empty runtime_id",
            manifest_path.display()
        );
    }

    manifest.read_only = true;
    manifest.imported_from = Some(
        manifest_path
            .canonicalize()
            .unwrap_or_else(|_| manifest_path.to_path_buf()),
    );
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("imported runtime `{}` is not usable", manifest.runtime_key))?;

    write_runtime_registry_manifest(paths, &manifest, replace)?;

    Ok(manifest)
}

pub(crate) fn newest_installed_runtime_manifest(
    paths: &AppPaths,
) -> Result<Option<therock::InstalledRuntimeManifest>> {
    // `load_runtime_manifests` already returns newest install first with ties
    // broken on the runtime key, so re-sorting here would be a no-op.
    Ok(therock::load_runtime_manifests(paths)?.into_iter().next())
}

#[derive(Debug, Clone)]
pub(crate) struct AdoptRuntimeRequest {
    pub(crate) python_executable: PathBuf,
    pub(crate) install_root: PathBuf,
    pub(crate) runtime_id: String,
    pub(crate) runtime_key: String,
    pub(crate) replace: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct AdoptRuntimeOptions {
    pub(crate) python_input: PathBuf,
    pub(crate) install_root: Option<PathBuf>,
    pub(crate) runtime_id: Option<String>,
    pub(crate) runtime_key: Option<String>,
    pub(crate) channel: Option<String>,
    pub(crate) replace: bool,
}

pub(crate) fn adopt_runtime_from_python_options(
    paths: &AppPaths,
    options: AdoptRuntimeOptions,
) -> Result<therock::InstalledRuntimeManifest> {
    let (python_executable, inferred_root) = resolve_adopt_python_input(&options.python_input)?;
    let probe = therock::probe_rocm_sdk_runtime(&python_executable)
        .with_context(|| format!("failed to probe {}", python_executable.display()))?;
    let request = infer_adopt_runtime_request(
        python_executable,
        options.install_root.or(inferred_root),
        options.runtime_id,
        options.runtime_key,
        options.channel,
        options.replace,
        &probe,
    )?;
    adopt_runtime_from_probe(paths, request, probe)
}

fn infer_adopt_runtime_request(
    python_executable: PathBuf,
    install_root: Option<PathBuf>,
    runtime_id: Option<String>,
    runtime_key: Option<String>,
    channel: Option<String>,
    replace: bool,
    probe: &therock::RocmSdkPythonProbe,
) -> Result<AdoptRuntimeRequest> {
    let install_root = install_root.with_context(|| {
        format!(
            "could not infer the Python environment folder from {}; pass --root",
            python_executable.display()
        )
    })?;
    let runtime_id = match runtime_id {
        Some(value) if !value.trim().is_empty() => {
            if let Some(channel) = channel.as_deref() {
                let (parsed_channel, _) = parse_therock_runtime_id(&value)?;
                let requested_channel = normalize_adopt_channel(channel)?;
                if parsed_channel != requested_channel {
                    bail!(
                        "--channel {requested_channel} does not match runtime id channel {parsed_channel}"
                    );
                }
            }
            value
        }
        Some(_) => bail!("runtime_id must not be empty"),
        None => {
            let channel = normalize_adopt_channel(channel.as_deref().unwrap_or("release"))?;
            let family = probe
                .resolved_target_family
                .as_deref()
                .or(probe.default_target_family.as_deref())
                .filter(|value| !value.trim().is_empty())
                .context(
                    "rocm_sdk probe did not report a GPU package; pass --runtime-id explicitly",
                )?;
            format!("therock-{channel}:{family}")
        }
    };
    let runtime_key = match runtime_key {
        Some(value) if !value.trim().is_empty() => value,
        Some(_) => bail!("runtime_key must not be empty"),
        None => {
            let (channel, family) = parse_therock_runtime_id(&runtime_id)?;
            let version = probe
                .rocm_sdk_version
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .context("rocm_sdk probe did not report a version; cannot name adopted runtime")?;
            format!(
                "adopted-{channel}-pip-{}-{}",
                runtime_key_component(&family),
                runtime_key_component(version)
            )
        }
    };
    Ok(AdoptRuntimeRequest {
        python_executable,
        install_root,
        runtime_id,
        runtime_key,
        replace,
    })
}

fn normalize_adopt_channel(channel: &str) -> Result<String> {
    match channel.trim().to_ascii_lowercase().as_str() {
        "release" => Ok("release".to_owned()),
        "nightly" => Ok("nightly".to_owned()),
        other => bail!("adopt channel must be release or nightly, got `{other}`"),
    }
}

fn runtime_key_component(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    let trimmed = output.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "runtime".to_owned()
    } else {
        trimmed
    }
}

fn resolve_adopt_python_input(input: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let absolute = if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(input)
    };
    if absolute.is_dir() {
        let env_root = absolute.canonicalize().with_context(|| {
            format!(
                "failed to resolve Python environment folder {}",
                absolute.display()
            )
        })?;
        let python = runtime_python_executable_in_env(&env_root);
        if !python.is_file() {
            bail!(
                "Python executable is missing in {}",
                python.parent().unwrap_or(env_root.as_path()).display()
            );
        }
        return Ok((python, Some(env_root)));
    }
    if absolute.is_file() {
        let inferred_root = infer_python_env_root(&absolute);
        return Ok((absolute, inferred_root));
    }
    bail!(
        "Python executable or folder is missing: {}",
        absolute.display()
    );
}

fn infer_python_env_root(python_executable: &Path) -> Option<PathBuf> {
    let bin_dir = python_executable.parent()?;
    let bin_name = bin_dir.file_name()?.to_string_lossy();
    if bin_name.eq_ignore_ascii_case("Scripts") || bin_name == "bin" {
        return bin_dir.parent().map(Path::to_path_buf);
    }
    bin_dir.parent().map(Path::to_path_buf)
}

pub(crate) fn adopt_runtime_from_probe(
    paths: &AppPaths,
    request: AdoptRuntimeRequest,
    probe: therock::RocmSdkPythonProbe,
) -> Result<therock::InstalledRuntimeManifest> {
    if request.runtime_key.trim().is_empty() {
        bail!("runtime_key must not be empty");
    }
    if request.runtime_id.trim().is_empty() {
        bail!("runtime_id must not be empty");
    }
    let python_executable = absolute_existing_file_path_preserving_symlink(
        &request.python_executable,
        "runtime Python executable",
    )?;
    let install_root = request.install_root.canonicalize().with_context(|| {
        format!(
            "runtime install root is missing: {}",
            request.install_root.display()
        )
    })?;
    if !install_root.is_dir() {
        bail!(
            "runtime install root is missing: {}",
            install_root.display()
        );
    }
    let (channel, family) = parse_therock_runtime_id(&request.runtime_id)?;
    therock::validate_rocm_sdk_runtime_probe(&probe)?;
    let version = probe
        .rocm_sdk_version
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .context("rocm_sdk probe did not report a version; cannot adopt runtime explicitly")?
        .to_owned();

    let manifest = therock::InstalledRuntimeManifest {
        runtime_key: request.runtime_key,
        runtime_id: request.runtime_id,
        channel,
        format: "wheel".to_owned(),
        family,
        family_source: "runtime_id".to_owned(),
        version,
        install_root: install_root.clone(),
        selected_artifact_url: "adopted-read-only".to_owned(),
        source_layout_generation: None,
        index_url: None,
        tarball_file_name: None,
        python_launcher: None,
        python_executable: Some(python_executable.display().to_string()),
        pip_cache_dir: None,
        // The probe only reports a CMake path when the `devel` packages are
        // present, so it tells us what this pre-existing environment has.
        devel: probe.cmake_path.is_some(),
        rocm_sdk: Some(probe),
        // Adoption does not install torch, so the build is derived from the SDK
        // version instead.
        sdk_torch: None,
        wheel_composition: None,
        read_only: true,
        imported_from: Some(install_root),
        installed_at_unix_ms: rocm_core::unix_time_millis(),
    };
    validate_runtime_manifest_for_activation(&manifest)
        .with_context(|| format!("adopted runtime `{}` is not usable", manifest.runtime_key))?;
    write_runtime_registry_manifest(paths, &manifest, request.replace)?;
    Ok(manifest)
}

fn absolute_existing_file_path_preserving_symlink(path: &Path, label: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(path)
    };
    if !absolute.is_file() {
        bail!("{label} is missing: {}", absolute.display());
    }
    Ok(absolute)
}

fn parse_therock_runtime_id(runtime_id: &str) -> Result<(String, String)> {
    let (prefix, family) = runtime_id.split_once(':').with_context(|| {
        format!("runtime_id `{runtime_id}` must include a TheRock family suffix after ':'")
    })?;
    let family = family.trim();
    if family.is_empty() {
        bail!("runtime_id `{runtime_id}` has an empty TheRock family suffix");
    }
    let channel = match prefix.trim() {
        "therock-release" => "release",
        "therock-nightly" => "nightly",
        other => bail!(
            "runtime_id `{runtime_id}` must start with therock-release: or therock-nightly:, got `{other}`"
        ),
    };
    Ok((channel.to_owned(), family.to_owned()))
}

fn write_runtime_registry_manifest(
    paths: &AppPaths,
    manifest: &therock::InstalledRuntimeManifest,
    replace: bool,
) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    if registry_path.exists() && !replace {
        bail!(
            "runtime registry entry already exists: {}; pass --replace to overwrite it",
            registry_path.display()
        );
    }
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&manifest)
            .context("failed to serialize runtime registry manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;
    Ok(())
}

pub(crate) fn recover_setup_runtime_registration(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    let Some(setup_root) = config
        .setup
        .therock_venv
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
    else {
        return Ok(None);
    };
    if !setup_root.is_dir() {
        return Ok(None);
    }

    let local_manifest_path = setup_root.join(".rocm-cli-runtime.json");
    if !local_manifest_path.is_file() {
        return Ok(None);
    }

    let bytes = fs::read(&local_manifest_path)
        .with_context(|| format!("failed to read {}", local_manifest_path.display()))?;
    let manifest: therock::InstalledRuntimeManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", local_manifest_path.display()))?;
    if manifest.runtime_key.trim().is_empty() {
        bail!(
            "setup runtime manifest {} has an empty runtime_key",
            local_manifest_path.display()
        );
    }
    if !paths_equivalent(&manifest.install_root, setup_root) {
        bail!(
            "setup runtime manifest {} points at {}, but setup is configured for {}",
            local_manifest_path.display(),
            manifest.install_root.display(),
            setup_root.display()
        );
    }
    validate_runtime_manifest_for_activation(&manifest).with_context(|| {
        format!(
            "setup runtime `{}` from {} is not usable",
            manifest.runtime_key,
            local_manifest_path.display()
        )
    })?;

    if !runtime_manifest_path(paths, &manifest.runtime_key).is_file() {
        write_runtime_registry_manifest(paths, &manifest, false).with_context(|| {
            format!(
                "failed to restore setup runtime `{}` into {}",
                manifest.runtime_key,
                runtime_registry_dir(paths).display()
            )
        })?;
    }
    Ok(Some(manifest.runtime_key))
}

/// The folder setup was configured for and its pip cache — the pair both
/// `examine` forms report. Shared so the two cannot name different folders.
///
/// An empty configured path counts as unset, matching
/// `recover_setup_runtime_registration`'s own guard: `config.json` is
/// hand-editable, and `managed_pip_cache_dir("")` is a bogus relative path.
pub(crate) fn setup_runtime_paths(config: &RocmCliConfig) -> Option<(&Path, PathBuf)> {
    let root = config
        .setup
        .therock_venv
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())?;
    Some((root, managed_pip_cache_dir(root)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::test_paths;

    #[test]
    fn runtime_import_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-import-replace");
        let manifest = write_test_pip_runtime(
            &paths,
            "external-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let exported_manifest = root.join("external-runtime.json");
        fs::write(&exported_manifest, serde_json::to_vec_pretty(&manifest)?)?;

        let error = import_runtime_manifest(&paths, &exported_manifest, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("already exists"));
        import_runtime_manifest(&paths, &exported_manifest, true)?;

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// The other half of the same derivation: an adopted environment that does
    /// expose a CMake path has the toolchain, and the manifest must say so or
    /// the next `rocm update` would quietly reinstall without it.
    #[test]
    fn runtime_adopt_records_devel_when_the_probe_finds_cmake() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-devel");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root
            .join("Lib")
            .join("site-packages")
            .join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        let cmake_path = sdk_root.join("lib").join("cmake");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::create_dir_all(&cmake_path)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                cmake_path: Some(cmake_path),
                runtime_roots: vec![sdk_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_bin],
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert!(
            adopted.devel,
            "a probe reporting a CMake path must record the toolchain"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_infers_ids_from_probe() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable.clone(),
            Some(external_root.clone()),
            None,
            None,
            None,
            false,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.13.0a20260423".to_owned()),
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.python_executable, python_executable);
        assert_eq!(request.install_root, external_root);
        assert_eq!(request.runtime_id, "therock-release:gfx120X-all");
        assert_eq!(
            request.runtime_key,
            "adopted-release-pip-gfx120x-all-7-13-0a20260423"
        );
        assert!(!request.replace);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_request_honors_nightly_channel() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer-nightly");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        fs::create_dir_all(&scripts_dir)?;
        fs::write(&python_executable, "python")?;

        let request = infer_adopt_runtime_request(
            python_executable,
            Some(external_root),
            None,
            None,
            Some("nightly".to_owned()),
            true,
            &therock::RocmSdkPythonProbe {
                rocm_sdk_version: Some("7.14.0a20260531".to_owned()),
                default_target_family: Some("gfx1151".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(request.runtime_id, "therock-nightly:gfx1151");
        assert_eq!(
            request.runtime_key,
            "adopted-nightly-pip-gfx1151-7-14-0a20260531"
        );
        assert!(request.replace);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn runtime_adopt_preserves_venv_python_symlink_path() -> Result<()> {
        use std::os::unix::fs::symlink;

        let (root, paths) = test_paths("runtime-adopt-python-symlink");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join("bin");
        let system_python = root.join("python3.12");
        let python_executable = scripts_dir.join("python");
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join("libamdhip64.so");
        let hipblas = sdk_bin.join("libhipblas.so");
        fs::write(&system_python, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;
        symlink(&system_python, &python_executable)?;

        let adopted = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable: python_executable.clone(),
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-symlink-python".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                runtime_roots: vec![sdk_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_bin],
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                resolved_target_family: Some("gfx120X-all".to_owned()),
                ..therock::RocmSdkPythonProbe::default()
            },
        )?;

        assert_eq!(
            adopted.python_executable.as_deref(),
            Some(python_executable.as_path().to_string_lossy().as_ref())
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_rejects_runtime_id_without_family() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-no-family");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::write(&python_executable, "python")?;

        let error = adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: false,
            },
            therock::RocmSdkPythonProbe {
                import_ok: true,
                rocm_sdk_version: Some("7.13.0".to_owned()),
                root_path: Some(sdk_root),
                bin_path: Some(sdk_bin),
                ..therock::RocmSdkPythonProbe::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("must include a TheRock family suffix"));
        assert!(!runtime_registry_dir(&paths).exists());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn runtime_adopt_refuses_to_overwrite_without_replace() -> Result<()> {
        let (root, paths) = test_paths("runtime-adopt-replace");
        let external_root = root.join("external-therock-venv");
        let scripts_dir = external_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let python_executable = scripts_dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let sdk_root = external_root.join("rocm_sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&scripts_dir)?;
        fs::create_dir_all(&sdk_bin)?;
        let amdhip = sdk_bin.join(if cfg!(windows) {
            "amdhip64_7.dll"
        } else {
            "libamdhip64.so"
        });
        let hipblas = sdk_bin.join(if cfg!(windows) {
            "hipblas.dll"
        } else {
            "libhipblas.so"
        });
        fs::write(&python_executable, "python")?;
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let probe = therock::RocmSdkPythonProbe {
            import_ok: true,
            rocm_sdk_version: Some("7.13.0".to_owned()),
            root_path: Some(sdk_root.clone()),
            bin_path: Some(sdk_bin.clone()),
            runtime_roots: vec![sdk_root],
            bin_paths: vec![sdk_bin.clone()],
            library_paths: vec![sdk_bin],
            resolved_libraries: vec![
                therock::RocmSdkLibraryProbe {
                    shortname: "amdhip64".to_owned(),
                    paths: vec![amdhip],
                },
                therock::RocmSdkLibraryProbe {
                    shortname: "hipblas".to_owned(),
                    paths: vec![hipblas],
                },
            ],
            ..therock::RocmSdkPythonProbe::default()
        };
        let request = AdoptRuntimeRequest {
            python_executable: python_executable.clone(),
            install_root: external_root.clone(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
            replace: false,
        };

        adopt_runtime_from_probe(&paths, request.clone(), probe.clone())?;
        let error = adopt_runtime_from_probe(&paths, request, probe.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already exists"));

        adopt_runtime_from_probe(
            &paths,
            AdoptRuntimeRequest {
                python_executable,
                install_root: external_root,
                runtime_id: "therock-release:gfx120X-all".to_owned(),
                runtime_key: "adopted-release-pip-gfx120x-all-7-13-0".to_owned(),
                replace: true,
            },
            probe,
        )?;

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
