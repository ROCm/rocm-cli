// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Registering a runtime that was not installed by `rocm install sdk`:
//! importing a manifest written elsewhere, adopting an existing Python
//! environment by probing it, and recovering the registration of the runtime
//! `rocm setup` created. It also holds the one rule for picking "the one ready
//! runtime" when nothing else says which to use
//! ([`select_single_ready_runtime`]), which sits here because it recovers the
//! setup registration first; `rocm comfyui install` and the engine paths both
//! ask it and each words the zero and several cases itself.
//!
//! The rest was relocated from `runtimes_cmd.rs` with no behavior change. `runtimes_cmd.rs`
//! calls into this module for `rocm runtimes import` and `rocm runtimes adopt`,
//! and re-exports `newest_installed_runtime_manifest`,
//! `recover_setup_runtime_registration` and `setup_runtime_paths` for the
//! modules that import them from there. The path comparison both adopt and
//! uninstall need lives in `runtime_registry.rs`, so this module does not
//! depend on `runtimes_cmd.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, managed_pip_cache_dir, runtime_python_env_root_for_executable,
    runtime_python_executable_in_env,
};

use crate::runtime_registry::{
    ManifestOverwrite, paths_equivalent, runtime_manifest_path, runtime_registry_dir,
    sort_manifests_newest_install_first, validate_runtime_manifest_for_activation,
    write_runtime_manifest,
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

    write_runtime_manifest(
        paths,
        &manifest,
        ManifestOverwrite::for_replace_flag(replace),
    )?;

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
        // `None` for an interpreter outside a `bin`/`Scripts` folder: the
        // request then asks for `--root` rather than adopting a guessed folder.
        let inferred_root = runtime_python_env_root_for_executable(&absolute);
        return Ok((absolute, inferred_root));
    }
    bail!(
        "Python executable or folder is missing: {}",
        absolute.display()
    );
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
    write_runtime_manifest(
        paths,
        &manifest,
        ManifestOverwrite::for_replace_flag(request.replace),
    )?;
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

/// The runtime `rocm setup` created, as the manifest in its install tree
/// describes it, and whether the runtime registry already holds its entry.
struct SetupRuntime {
    manifest: therock::InstalledRuntimeManifest,
    registered: bool,
}

/// Find the runtime `rocm setup` created and check that it is usable, without
/// changing anything. `None` when setup named no folder or the folder holds no
/// manifest; an error when it does and the runtime cannot be used.
fn find_setup_runtime(paths: &AppPaths, config: &RocmCliConfig) -> Result<Option<SetupRuntime>> {
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

    let registered = runtime_manifest_path(paths, &manifest.runtime_key).is_file();
    Ok(Some(SetupRuntime {
        manifest,
        registered,
    }))
}

pub(crate) fn recover_setup_runtime_registration(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    let Some(setup) = find_setup_runtime(paths, config)? else {
        return Ok(None);
    };
    if !setup.registered {
        write_runtime_manifest(paths, &setup.manifest, ManifestOverwrite::Refuse).with_context(
            || {
                format!(
                    "failed to restore setup runtime `{}` into {}",
                    setup.manifest.runtime_key,
                    runtime_registry_dir(paths).display()
                )
            },
        )?;
    }
    Ok(Some(setup.manifest.runtime_key))
}

/// Whether finding the runtime `rocm setup` created may write its lost registry
/// entry back while a runtime is being picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetupRecovery {
    /// Write the entry back, as `rocm serve` and `rocm engines install` do.
    Persist,
    /// Count the setup runtime as if its entry were there, and write nothing:
    /// a `--dry-run` preview must name the runtime the real run will use
    /// without changing any file.
    PreviewOnly,
}

impl SetupRecovery {
    /// The recovery a command with a `--dry-run` flag uses.
    pub(crate) const fn for_dry_run(dry_run: bool) -> Self {
        if dry_run {
            Self::PreviewOnly
        } else {
            Self::Persist
        }
    }
}

/// What picking "the one ready runtime" found. The caller decides how to report
/// the two cases that are not a pick, because the advice differs by command
/// (`rocm comfyui install` and `rocm serve` name different remedies).
#[derive(Debug)]
pub(crate) enum SingleReadyRuntime {
    /// Exactly one registered runtime is ready.
    One(Box<therock::InstalledRuntimeManifest>),
    /// No registered runtime is ready. `registered` is everything the registry
    /// holds, unusable entries included and possibly none, so a caller can say
    /// why each one does not count.
    Zero {
        registered: Vec<therock::InstalledRuntimeManifest>,
    },
    /// More than one is ready, so there is nothing to pick without a choice.
    Many {
        ready: Vec<therock::InstalledRuntimeManifest>,
    },
}

/// Pick the one ready runtime when nothing else says which to use: no
/// `--runtime-id`, no active runtime, no configured default. This is the only
/// place that rule lives; `rocm comfyui install` and the engine paths
/// (`rocm serve`, `rocm engines install`) both call it.
///
/// Ready means `validate_runtime_manifest_for_activation` accepts the runtime,
/// the same check `rocm runtimes activate` applies. Setup is recovered first
/// (`recover_setup_runtime_registration`), so the runtime `rocm setup` created
/// counts even if its registry entry was lost. With [`SetupRecovery::Persist`]
/// that writes the entry back; with [`SetupRecovery::PreviewOnly`] nothing is
/// written, and the setup runtime is counted from its own manifest as the
/// registry would hold it once restored, so the answer is the one `Persist`
/// gives.
pub(crate) fn select_single_ready_runtime(
    paths: &AppPaths,
    config: &RocmCliConfig,
    recovery: SetupRecovery,
) -> Result<SingleReadyRuntime> {
    Ok(classify_single_ready_runtime(
        registered_runtimes_with_setup(paths, config, recovery)?,
    ))
}

/// Every runtime the registry holds, with the one `rocm setup` created counted
/// even if its registry entry was lost. With [`SetupRecovery::Persist`] the
/// entry is written back first (`recover_setup_runtime_registration`); with
/// [`SetupRecovery::PreviewOnly`] nothing is written, and the setup runtime is
/// added from its own manifest as the registry would hold it once restored, so
/// both return the same list. An error when the setup runtime exists but
/// cannot be used.
pub(crate) fn registered_runtimes_with_setup(
    paths: &AppPaths,
    config: &RocmCliConfig,
    recovery: SetupRecovery,
) -> Result<Vec<therock::InstalledRuntimeManifest>> {
    match recovery {
        SetupRecovery::Persist => {
            recover_setup_runtime_registration(paths, config)?;
            therock::load_runtime_manifests(paths)
        }
        SetupRecovery::PreviewOnly => {
            let setup = find_setup_runtime(paths, config)?;
            let mut registered = therock::load_runtime_manifests(paths)?;
            if let Some(setup) = setup
                && !setup.registered
            {
                // What `load_runtime_manifests` would return for the entry that
                // `Persist` would have written.
                registered.push(
                    setup
                        .manifest
                        .normalize_storage_paths()
                        .normalize_host_paths(),
                );
                sort_manifests_newest_install_first(&mut registered);
            }
            Ok(registered)
        }
    }
}

/// The pure half of [`select_single_ready_runtime`]: which of `registered` are
/// ready, and whether that is none, one or several.
pub(crate) fn classify_single_ready_runtime(
    registered: Vec<therock::InstalledRuntimeManifest>,
) -> SingleReadyRuntime {
    let ready: Vec<_> = registered
        .iter()
        .filter(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok())
        .cloned()
        .collect();
    if ready.is_empty() {
        return SingleReadyRuntime::Zero { registered };
    }
    match <[therock::InstalledRuntimeManifest; 1]>::try_from(ready) {
        Ok([manifest]) => SingleReadyRuntime::One(Box::new(manifest)),
        Err(ready) => SingleReadyRuntime::Many { ready },
    }
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

    /// `rocm runtimes adopt --python <file>` without `--root` takes the
    /// environment folder from the shared rule (see
    /// `rocm_core::runtime_python_env_root_for_executable`): only an
    /// interpreter in a `bin` or `Scripts` folder, in any case, names one. An
    /// interpreter in an unrelated folder yields no root, and the request is
    /// then refused with `--root` guidance rather than adopting a guessed
    /// folder.
    #[test]
    fn adopt_python_file_infers_a_root_only_from_a_bin_or_scripts_folder() -> Result<()> {
        let (root, _) = test_paths("runtime-adopt-infer-root");
        for (folder, expected_root) in [
            ("bin", true),
            ("Scripts", true),
            ("SCRIPTS", true),
            ("tools", false),
        ] {
            let env_root = root.join(format!("env-{folder}"));
            let python = env_root.join(folder).join("python");
            fs::create_dir_all(python.parent().unwrap())?;
            fs::write(&python, "python")?;

            let (resolved, inferred) = resolve_adopt_python_input(&python)?;

            assert_eq!(resolved, python, "{folder}");
            assert_eq!(
                inferred.as_deref(),
                expected_root.then_some(env_root.as_path()),
                "{folder}"
            );
        }

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    fn pip_runtime(paths: &AppPaths, key: &str) -> Result<therock::InstalledRuntimeManifest> {
        write_test_pip_runtime(paths, key, "therock-release:gfx120X-all", "7.14.0", 20)
    }

    /// Readiness is the activation check and nothing looser: a runtime whose
    /// interpreter is gone is registered but does not count, and neither is it
    /// "the only one".
    #[test]
    fn single_ready_runtime_counts_only_what_activation_accepts() -> Result<()> {
        let (root, paths) = test_paths("single-ready-classify");
        let ready = pip_runtime(&paths, "runtime-a")?;
        let broken = pip_runtime(&paths, "runtime-b")?;
        fs::remove_file(
            broken
                .python_executable
                .as_deref()
                .context("the fixture records an interpreter")?,
        )?;
        assert!(validate_runtime_manifest_for_activation(&ready).is_ok());
        assert!(validate_runtime_manifest_for_activation(&broken).is_err());

        match classify_single_ready_runtime(vec![ready.clone(), broken.clone()]) {
            SingleReadyRuntime::One(manifest) => assert_eq!(manifest.runtime_key, "runtime-a"),
            other => panic!("expected the one ready runtime, got {other:?}"),
        }
        match classify_single_ready_runtime(vec![broken.clone()]) {
            SingleReadyRuntime::Zero { registered } => {
                assert_eq!(registered.len(), 1, "unusable entries stay listed");
            }
            other => panic!("expected no ready runtime, got {other:?}"),
        }
        match classify_single_ready_runtime(Vec::new()) {
            SingleReadyRuntime::Zero { registered } => assert!(registered.is_empty()),
            other => panic!("expected no ready runtime, got {other:?}"),
        }
        let second = pip_runtime(&paths, "runtime-c")?;
        match classify_single_ready_runtime(vec![ready, broken, second]) {
            SingleReadyRuntime::Many { ready } => {
                let keys: Vec<_> = ready.iter().map(|m| m.runtime_key.as_str()).collect();
                assert_eq!(keys, ["runtime-a", "runtime-c"], "only ready ones listed");
            }
            other => panic!("expected several ready runtimes, got {other:?}"),
        }
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// What no inferred root costs the user: the request is refused and asks
    /// for `--root`, and an explicit `--root` still adopts the same
    /// interpreter.
    #[test]
    fn adopt_request_without_an_inferred_root_asks_for_root() {
        let python = PathBuf::from("python");
        let request = |install_root: Option<PathBuf>| {
            infer_adopt_runtime_request(
                python.clone(),
                install_root,
                None,
                None,
                None,
                false,
                &therock::RocmSdkPythonProbe {
                    rocm_sdk_version: Some("7.13.0".to_owned()),
                    resolved_target_family: Some("gfx120X-all".to_owned()),
                    ..therock::RocmSdkPythonProbe::default()
                },
            )
        };

        let error = request(None).unwrap_err().to_string();
        assert!(error.contains("pass --root"), "{error}");

        let explicit = request(Some(PathBuf::from("/opt/rocm-env"))).unwrap();
        assert_eq!(explicit.install_root, PathBuf::from("/opt/rocm-env"));
    }

    /// The selection recovers the setup runtime first, so a runtime the registry
    /// lost is the one found, and finding it is what writes its entry back.
    #[test]
    fn select_single_ready_runtime_recovers_setup_before_counting() -> Result<()> {
        let (root, paths) = test_paths("single-ready-recovers-setup");
        let lost = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "runtime-s",
        )?;
        let config = RocmCliConfig::load(&paths)?;
        assert!(crate::test_support::runtime_registry_entry_names(&paths).is_empty());

        match select_single_ready_runtime(&paths, &config, SetupRecovery::Persist)? {
            SingleReadyRuntime::One(manifest) => assert_eq!(manifest.runtime_key, "runtime-s"),
            other => panic!("expected the recovered setup runtime, got {other:?}"),
        }
        assert_eq!(
            crate::test_support::runtime_registry_entry_names(&paths),
            vec![format!("{}.json", lost.runtime_key)]
        );

        // Without setup naming a folder there is nothing to recover and the
        // registry is not touched.
        let (other_root, other_paths) = test_paths("single-ready-no-setup");
        match select_single_ready_runtime(
            &other_paths,
            &RocmCliConfig::default(),
            SetupRecovery::Persist,
        )? {
            SingleReadyRuntime::Zero { registered } => assert!(registered.is_empty()),
            other => panic!("expected no ready runtime, got {other:?}"),
        }
        assert!(crate::test_support::runtime_registry_entry_names(&other_paths).is_empty());
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(other_root);
        Ok(())
    }

    /// The keys `select_single_ready_runtime` found ready, in the order it lists
    /// them, or the message of its refusal to recover.
    fn single_ready_outcome(
        paths: &AppPaths,
        config: &RocmCliConfig,
        recovery: SetupRecovery,
    ) -> Result<Vec<String>, String> {
        match select_single_ready_runtime(paths, config, recovery) {
            Ok(SingleReadyRuntime::One(manifest)) => Ok(vec![manifest.runtime_key]),
            Ok(SingleReadyRuntime::Many { ready }) => {
                Ok(ready.into_iter().map(|m| m.runtime_key).collect())
            }
            Ok(SingleReadyRuntime::Zero { registered }) => {
                Err(format!("none ready of {}", registered.len()))
            }
            Err(error) => Err(format!("{error:#}")),
        }
    }

    /// A preview counts the runtime setup created from its own manifest and
    /// writes nothing; the real run then finds the same runtime and writes the
    /// entry. Both orders of install time are covered: the ready runtimes are
    /// listed newest first, and a preview must list them the way the registry
    /// will once the entry is back.
    #[test]
    fn preview_counts_the_setup_runtime_without_writing_and_agrees_with_persist() -> Result<()> {
        for (label, other_installed_at) in [("other-newer", 30), ("other-older", 10)] {
            let (root, paths) = test_paths(&format!("single-ready-preview-{label}"));
            let lost = crate::test_support::write_test_setup_runtime_missing_from_registry(
                &paths,
                "runtime-setup",
            )?;
            write_test_pip_runtime(
                &paths,
                "runtime-other",
                "therock-release:gfx120X-all",
                "7.14.0",
                other_installed_at,
            )?;
            let config = RocmCliConfig::load(&paths)?;
            let registry_before = crate::test_support::runtime_registry_entry_names(&paths);
            assert_eq!(registry_before, ["runtime-other.json"], "{label}");

            let preview = single_ready_outcome(&paths, &config, SetupRecovery::PreviewOnly)
                .expect("the setup runtime and the other one are both ready");

            assert_eq!(preview.len(), 2, "{label}: {preview:?}");
            assert!(preview.contains(&lost.runtime_key), "{label}: {preview:?}");
            assert_eq!(
                crate::test_support::runtime_registry_entry_names(&paths),
                registry_before,
                "{label}: a preview must not write the lost entry"
            );
            let persisted = single_ready_outcome(&paths, &config, SetupRecovery::Persist)
                .expect("recovery finds the same two");
            assert_eq!(preview, persisted, "{label}: preview and real run disagree");
            assert_eq!(
                crate::test_support::runtime_registry_entry_names(&paths),
                [
                    "runtime-other.json".to_owned(),
                    "runtime-setup.json".to_owned()
                ],
                "{label}: the real run writes the entry back"
            );
            let _ = fs::remove_dir_all(root);
        }
        Ok(())
    }

    /// The setup runtime alone is "the one ready runtime" in a preview, and a
    /// registry that already holds it is left as it is.
    #[test]
    fn preview_picks_a_lost_setup_runtime_and_ignores_a_registered_one() -> Result<()> {
        let (root, paths) = test_paths("single-ready-preview-one");
        crate::test_support::write_test_setup_runtime_missing_from_registry(&paths, "runtime-s")?;
        let config = RocmCliConfig::load(&paths)?;

        assert_eq!(
            single_ready_outcome(&paths, &config, SetupRecovery::PreviewOnly),
            Ok(vec!["runtime-s".to_owned()])
        );
        assert!(crate::test_support::runtime_registry_entry_names(&paths).is_empty());

        // Once the entry exists a preview and a real run see one runtime, and
        // neither adds a second copy of it.
        assert_eq!(
            single_ready_outcome(&paths, &config, SetupRecovery::Persist),
            Ok(vec!["runtime-s".to_owned()])
        );
        assert_eq!(
            single_ready_outcome(&paths, &config, SetupRecovery::PreviewOnly),
            Ok(vec!["runtime-s".to_owned()])
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// A setup runtime that is no longer usable is reported by a preview as the
    /// real run reports it, and neither writes anything.
    #[test]
    fn preview_reports_an_unusable_setup_runtime_as_persist_does() -> Result<()> {
        let (root, paths) = test_paths("single-ready-preview-unusable");
        let lost = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "runtime-s",
        )?;
        fs::remove_file(
            lost.python_executable
                .as_deref()
                .context("the fixture records an interpreter")?,
        )?;
        let config = RocmCliConfig::load(&paths)?;

        let preview = single_ready_outcome(&paths, &config, SetupRecovery::PreviewOnly)
            .expect_err("an unusable setup runtime is an error");
        let persist = single_ready_outcome(&paths, &config, SetupRecovery::Persist)
            .expect_err("an unusable setup runtime is an error");

        assert!(preview.contains("is not usable"), "{preview}");
        assert_eq!(preview, persist);
        assert!(crate::test_support::runtime_registry_entry_names(&paths).is_empty());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn a_dry_run_previews_and_a_real_run_persists() {
        assert_eq!(SetupRecovery::for_dry_run(true), SetupRecovery::PreviewOnly);
        assert_eq!(SetupRecovery::for_dry_run(false), SetupRecovery::Persist);
    }
}
