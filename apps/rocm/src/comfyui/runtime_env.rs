// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which installed ROCm runtime ComfyUI uses, and the process environment that
//! runtime needs.
//!
//! [`select_runtime`] turns an explicit `--runtime-id` or the configured
//! default into the one ready managed Python runtime an install runs against,
//! and explains how to pick one when none or several qualify. A choice that
//! names the runtime `rocm setup` created finds it even when its registry entry
//! was lost (see `pick_from_registry`), so the `--runtime-id` an install preview
//! prints works from the state the preview left. The
//! `runtime_environment_*` functions build the `VIRTUAL_ENV`, `ROCM_PATH`,
//! `PATH` and library-path settings a ComfyUI child process runs with, and
//! [`apply_runtime_environment`] puts them on a command.
//!
//! The types these functions return ([`SelectedRuntime`],
//! [`ComfyUiRuntimeEnvironment`]) stay in the parent module with the other
//! ComfyUI types.

use super::{ComfyUiManifest, ComfyUiRuntimeEnvironment, SelectedRuntime};
use crate::runtime_adopt::{
    SetupRecovery, SingleReadyRuntime, recover_setup_runtime_registration,
    registered_runtimes_with_setup, select_single_ready_runtime,
};
use crate::runtimes_cmd::runtime_usability_status;
use crate::therock;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, runtime_is_linux, runtime_is_windows, runtime_path_list_join,
    runtime_path_list_split, runtime_paths_equivalent, runtime_python_env_root_for_executable,
};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn select_runtime(
    paths: &AppPaths,
    config: &RocmCliConfig,
    selector: Option<&str>,
    recovery: SetupRecovery,
) -> Result<SelectedRuntime> {
    let manifest = match selector.map(str::trim).filter(|value| !value.is_empty()) {
        Some(selector) => pick_from_registry(paths, config, recovery, |manifests| {
            select_runtime_by_selector(manifests, selector)
        })?,
        None => select_default_runtime(paths, config, recovery)?,
    };
    if runtime_usability_status(&manifest) != "ready" {
        bail!(
            "The selected ROCm runtime is not ready: {}",
            runtime_usability_status(&manifest)
        );
    }
    if manifest.format != "wheel" {
        bail!("ComfyUI installs require a rocm-cli managed Python ROCm runtime.");
    }
    // Defence in depth, not a reachable branch on a settled filesystem: the
    // `ready` check above ran `validate_wheel_runtime_manifest`, which already
    // proved `python_executable` is `Some` and `is_file()`. This re-check only
    // fires if the interpreter disappears in the window between the two calls,
    // so keep it (a missing interpreter must not reach the installer) and do
    // not spend further polish on its wording.
    let python = manifest
        .python_executable
        .as_deref()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .with_context(|| {
            "The selected ROCm runtime has no Python executable. Pick another in `/runtimes` or pass `--runtime-id <key>`. See `rocm runtimes list`."
                .to_string()
        })?;
    Ok(SelectedRuntime { manifest, python })
}

/// Render a bare `key, key` list. Sound only where the caller has already
/// filtered `manifests` by `runtime_usability_status(...) == "ready"`, because
/// that is the only filter that makes every listed key a valid answer to the
/// `rocm runtimes activate <key>` advice these messages carry; a per-key status
/// would then be noise. Matching a selector is *not* such a filter: two
/// manifests can share a `runtime_id` with one of them unusable, so an
/// id-matched set can contain a key that cannot be activated. Every branch
/// whose set is not ready-filtered — the whole-registry ones and the two
/// id-ambiguity ones — uses `format_runtime_statuses` instead; see its doc
/// comment for why the two labels differ.
fn format_available_runtime_keys<'a>(
    manifests: impl IntoIterator<Item = &'a therock::InstalledRuntimeManifest>,
) -> String {
    manifests
        .into_iter()
        .map(|manifest| manifest.runtime_key.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render each known runtime as `` `key` (status) ``, surfacing the per-manifest
/// reason `runtime_usability_status` already computed (e.g.
/// `unusable (install root is missing: …)`) instead of dropping it.
///
/// Used by every branch whose set is not ready-filtered: the whole-registry
/// ones ("none ready" and the two not-found branches) and the two branches that
/// report an ambiguous `runtime_id`. None of them can promise the listed
/// runtimes are pickable — a shared `runtime_id` says nothing about readiness —
/// so printing bare keys under `Available:` would invite the user to
/// `rocm runtimes activate` an unusable one and land on a second error. The
/// label split is deliberate and load-bearing: `Available:` means "any of these
/// works", `Runtimes found:` means "here is everything registered, with why",
/// and `Runtimes matched:` means "here is what the selector matched, with why".
fn format_runtime_statuses<'a>(
    manifests: impl IntoIterator<Item = &'a therock::InstalledRuntimeManifest>,
) -> String {
    manifests
        .into_iter()
        .map(|manifest| {
            format!(
                "`{}` ({})",
                manifest.runtime_key,
                runtime_usability_status(manifest)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Apply `pick` (a choice that names a runtime: `--runtime-id`, the active
/// runtime or the configured default) to the registry. A runtime that `rocm
/// setup` created and whose registry entry was lost is not a refusal either:
/// when the registry alone has no answer, `pick` runs once more over the
/// registry plus that runtime, the way the engine paths resolve an explicit
/// `--runtime-id`. Which write that implies is `recovery`'s: a preview adds the
/// runtime to the list without writing its entry, so the `--runtime-id` an
/// install preview prints is a command that works from the state the preview
/// left. If setup's runtime cannot be used, the registry's own answer stands.
///
/// The two failures are not alike. Not finding a usable setup runtime leaves the
/// registry's refusal, because the user named something else or setup left
/// nothing. Failing to write the entry of a setup runtime that was found and
/// picked is not a refusal about a missing runtime, and swallowing it would
/// answer "Install ROCm first" about a runtime that is there, so that error is
/// returned as it is. The runtime is therefore found and picked without writing,
/// and written back only for a pick that is going to be used.
fn pick_from_registry(
    paths: &AppPaths,
    config: &RocmCliConfig,
    recovery: SetupRecovery,
    pick: impl for<'a> Fn(
        &'a [therock::InstalledRuntimeManifest],
    ) -> Result<&'a therock::InstalledRuntimeManifest>,
) -> Result<therock::InstalledRuntimeManifest> {
    let registered = therock::load_runtime_manifests(paths)?;
    let refusal = if registered.is_empty() {
        install_rocm_first()
    } else {
        match pick(&registered) {
            Ok(manifest) => return Ok(manifest.clone()),
            Err(error) => error,
        }
    };
    // Found without writing, whichever recovery was asked for: the write comes
    // after the pick, so its failure is reported as what it is.
    let with_setup = match registered_runtimes_with_setup(paths, config, SetupRecovery::PreviewOnly)
    {
        Ok(with_setup) if with_setup.len() > registered.len() => with_setup,
        _ => return Err(refusal),
    };
    let picked = pick(&with_setup)?.clone();
    if recovery == SetupRecovery::Persist {
        recover_setup_runtime_registration(paths, config)?;
    }
    Ok(picked)
}

fn install_rocm_first() -> anyhow::Error {
    anyhow::anyhow!(
        "Install ROCm first: run `rocm` and pick \"Set up this system\", then install ComfyUI."
    )
}

fn select_default_runtime(
    paths: &AppPaths,
    config: &RocmCliConfig,
    recovery: SetupRecovery,
) -> Result<therock::InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref() {
        return pick_from_registry(paths, config, recovery, |manifests| {
            select_runtime_by_selector(manifests, active_key)
        });
    }
    match config.default_runtime_id.as_deref() {
        Some(default_runtime_id) => pick_from_registry(paths, config, recovery, |manifests| {
            select_runtime_by_default_id(manifests, default_runtime_id)
        }),
        // Nothing says which runtime to use: the rule the engine paths share.
        None => {
            single_ready_runtime_or_refusal(select_single_ready_runtime(paths, config, recovery)?)
        }
    }
}

fn select_runtime_by_default_id<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    default_runtime_id: &str,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    let matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [manifest] => Ok(*manifest),
        [] => {
            // Whole-registry list, so it carries statuses: a not-found default
            // is no evidence the survivors are usable, and advising `activate`
            // on an unusable one just produces a second error.
            let statuses = format_runtime_statuses(manifests);
            bail!(
                "The configured default ROCm runtime was not found. Pick one in `/runtimes`, re-activate one with `rocm runtimes activate <key>`, or pass `--runtime-id <key>`. Runtimes found: {statuses}. See `rocm runtimes list`."
            )
        }
        _ => {
            // `matches` is filtered by `runtime_id` alone, so it can hold an
            // unusable runtime next to a ready one. Statuses, not bare keys:
            // `rocm runtimes activate` on the unusable match is a second error.
            let statuses = format_runtime_statuses(matches.iter().copied());
            bail!(
                "More than one ROCm runtime matches the configured default. Pick one in `/runtimes`, set a default with `rocm runtimes activate <key>`, or pass `--runtime-id <key>`. Runtimes matched: {statuses}. See `rocm runtimes list`."
            )
        }
    }
}

/// ComfyUI's words for what [`select_single_ready_runtime`] found. The rule
/// itself, including which runtimes count as ready, is not decided here.
fn single_ready_runtime_or_refusal(
    found: SingleReadyRuntime,
) -> Result<therock::InstalledRuntimeManifest> {
    match found {
        SingleReadyRuntime::One(manifest) => Ok(*manifest),
        SingleReadyRuntime::Zero { registered } if registered.is_empty() => {
            Err(install_rocm_first())
        }
        SingleReadyRuntime::Zero { registered } => {
            let statuses = format_runtime_statuses(&registered);
            bail!(
                "No ROCm runtime is ready. Fix one or install another: run `rocm` and pick \"Set up this system\" (or run `rocm install sdk`), then install ComfyUI. Runtimes found: {statuses}. See `rocm runtimes list`."
            )
        }
        SingleReadyRuntime::Many { ready } => {
            let available = format_available_runtime_keys(&ready);
            bail!(
                "Multiple ROCm runtimes are ready. Pick one in `/runtimes`, set a default with `rocm runtimes activate <key>`, or pass `--runtime-id <key>`. Available: {available}. See `rocm runtimes list`."
            )
        }
    }
}

fn select_runtime_by_selector<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(manifest) = manifests
        .iter()
        .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
    {
        return Ok(manifest);
    }
    let matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [manifest] => Ok(*manifest),
        [] => {
            // Same reasoning as the configured-default not-found branch: this
            // enumerates every registered runtime, so each entry carries its
            // status rather than implying all of them are pickable.
            let statuses = format_runtime_statuses(manifests);
            bail!(
                "ROCm runtime not found: `{selector}`. Pick one in `/runtimes`, re-activate one with `rocm runtimes activate <key>`, or pass the exact runtime key with `--runtime-id <key>`. Runtimes found: {statuses}. See `rocm runtimes list`."
            )
        }
        _ => {
            // Same reasoning as the ambiguous configured-default branch: an
            // id match is not a readiness check, so each entry carries its
            // status rather than reading as pickable.
            let statuses = format_runtime_statuses(matches.iter().copied());
            bail!(
                "More than one ROCm runtime matches `{selector}`. Pick one in `/runtimes`, set a default with `rocm runtimes activate <key>`, or pass the exact runtime key with `--runtime-id <key>`. Runtimes matched: {statuses}. See `rocm runtimes list`."
            )
        }
    }
}

pub(super) fn runtime_environment_for_manifest(
    paths: &AppPaths,
    manifest: &ComfyUiManifest,
) -> Result<ComfyUiRuntimeEnvironment> {
    let runtimes = therock::load_runtime_manifests(paths)?;
    if let Some(runtime) = runtimes.iter().find(|runtime| {
        runtime
            .runtime_key
            .eq_ignore_ascii_case(&manifest.runtime_key)
            || runtime
                .runtime_id
                .eq_ignore_ascii_case(&manifest.runtime_id)
    }) {
        return Ok(runtime_environment_from_runtime(
            runtime,
            &manifest.python_executable,
        ));
    }

    let mut env = ComfyUiRuntimeEnvironment {
        venv_root: runtime_python_env_root_for_executable(&manifest.python_executable),
        rocm_root: Some(manifest.runtime_root.clone()),
        path_entries: Vec::new(),
        library_entries: Vec::new(),
    };
    if let Some(bin_dir) = manifest.python_executable.parent() {
        push_existing_path(&mut env.path_entries, bin_dir.to_path_buf());
    }
    collect_rocm_runtime_paths(&manifest.runtime_root, &mut env);
    Ok(env)
}

pub(super) fn runtime_environment_from_runtime(
    manifest: &therock::InstalledRuntimeManifest,
    python: &Path,
) -> ComfyUiRuntimeEnvironment {
    let mut env = ComfyUiRuntimeEnvironment {
        venv_root: runtime_python_env_root_for_executable(python),
        rocm_root: manifest
            .rocm_sdk
            .as_ref()
            .and_then(|sdk| sdk.root_path.clone())
            .or_else(|| Some(manifest.install_root.clone())),
        path_entries: Vec::new(),
        library_entries: Vec::new(),
    };
    if let Some(bin_dir) = python.parent() {
        push_existing_path(&mut env.path_entries, bin_dir.to_path_buf());
    }
    if let Some(sdk) = manifest.rocm_sdk.as_ref() {
        if let Some(bin_path) = sdk.bin_path.as_ref() {
            push_existing_path(&mut env.path_entries, bin_path.clone());
        }
        for bin_path in &sdk.bin_paths {
            push_existing_path(&mut env.path_entries, bin_path.clone());
        }
        for library_path in &sdk.library_paths {
            push_existing_path(&mut env.library_entries, library_path.clone());
        }
        if let Some(root_path) = sdk.root_path.as_ref() {
            collect_rocm_runtime_paths(root_path, &mut env);
        }
        for root_path in &sdk.runtime_roots {
            collect_rocm_runtime_paths(root_path, &mut env);
        }
    }
    collect_rocm_runtime_paths(&manifest.install_root, &mut env);
    if runtime_is_linux() {
        push_existing_path(&mut env.library_entries, PathBuf::from("/usr/lib/wsl/lib"));
    }
    env
}

fn collect_rocm_runtime_paths(root: &Path, env: &mut ComfyUiRuntimeEnvironment) {
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
            push_existing_path(&mut env.path_entries, path.clone());
        }
        push_existing_path(&mut env.library_entries, path);
    }
}

pub(super) fn apply_runtime_environment(
    command: &mut Command,
    env: &ComfyUiRuntimeEnvironment,
) -> Result<()> {
    for (key, value) in runtime_environment_assignments(env)? {
        command.env(key, value);
    }
    Ok(())
}

pub(super) fn runtime_environment_assignments(
    env: &ComfyUiRuntimeEnvironment,
) -> Result<Vec<(&'static str, OsString)>> {
    let mut values = Vec::new();
    if let Some(venv_root) = env.venv_root.as_ref() {
        values.push(("VIRTUAL_ENV", venv_root.as_os_str().to_owned()));
    }
    if let Some(rocm_root) = env.rocm_root.as_ref() {
        values.push(("ROCM_PATH", rocm_root.as_os_str().to_owned()));
    }
    let mut path_entries = env.path_entries.clone();
    if runtime_is_windows() {
        path_entries.extend(env.library_entries.iter().cloned());
    }
    if let Some(path) = prepend_env_paths(&path_entries, std::env::var_os("PATH"))? {
        values.push(("PATH", path));
    }
    // The amd-smi link a managed runtime's torch needs to count devices (see
    // `rocm_core::ensure_amd_smi_compat`), after the recorded directories.
    let mut library_entries = env.library_entries.clone();
    if let Some(shim) = env.venv_root.as_ref().and_then(|venv_root| {
        rocm_core::ensure_amd_smi_compat(
            &venv_root.join(rocm_core::RUNTIME_COMPAT_DIR_NAME),
            &env.library_entries,
        )
    }) {
        library_entries.push(shim);
    }
    if runtime_is_linux()
        && let Some(ld_library_path) =
            prepend_env_paths(&library_entries, std::env::var_os("LD_LIBRARY_PATH"))?
    {
        values.push(("LD_LIBRARY_PATH", ld_library_path));
    }
    Ok(values)
}

fn prepend_env_paths(entries: &[PathBuf], current: Option<OsString>) -> Result<Option<OsString>> {
    let mut parts = Vec::new();
    for entry in entries {
        push_existing_path(&mut parts, entry.clone());
    }
    if let Some(current) = current
        && !current.is_empty()
    {
        for entry in split_runtime_paths(&current) {
            push_existing_path(&mut parts, entry);
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        join_runtime_paths(parts)
    }
}

fn split_runtime_paths(paths: &OsString) -> Vec<PathBuf> {
    runtime_path_list_split(paths)
}

fn join_runtime_paths(paths: Vec<PathBuf>) -> Result<Option<OsString>> {
    runtime_path_list_join(paths).map(Some)
}

fn push_existing_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.is_dir() {
        return;
    }
    if !paths.iter().any(|seen| same_path_text(seen, &path)) {
        paths.push(path);
    }
}

pub(super) fn same_path_text(left: &Path, right: &Path) -> bool {
    runtime_paths_equivalent(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comfyui::render_status;
    use crate::comfyui::tests::{ready_runtime_manifest, test_paths, write_runtime_manifest};
    use crate::runtime_adopt::classify_single_ready_runtime;
    use crate::test_support::{runtime_registry_entry_names, write_test_pip_runtime};
    use rocm_core::{runtime_python_executable_in_env, runtime_rocm_library_filename};
    use std::fs;

    /// `select_runtime` as a real install runs it: a runtime that only setup
    /// recovery can make visible has its registry entry written back. The
    /// preview mode has its own tests.
    fn persisting_select_runtime(
        paths: &AppPaths,
        config: &RocmCliConfig,
        selector: Option<&str>,
    ) -> Result<SelectedRuntime> {
        select_runtime(paths, config, selector, SetupRecovery::Persist)
    }

    #[test]
    fn default_runtime_selection_uses_single_ready_runtime() -> Result<()> {
        let paths = test_paths("comfyui-single-ready-runtime");
        let runtime_root = paths.data_dir.join("runtimes").join("default");
        let python = runtime_python_executable_in_env(&runtime_root);
        let python_bin = python
            .parent()
            .context("runtime Python path has no parent")?
            .to_path_buf();
        let sdk_root = runtime_root.join("sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&python_bin)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::write(runtime_root.join(".rocm-cli-runtime.json"), "{}")?;
        fs::write(&python, "python")?;
        let amdhip = sdk_bin.join(runtime_rocm_library_filename("amdhip64"));
        let hipblas = sdk_bin.join(runtime_rocm_library_filename("hipblas"));
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        let manifest = therock::InstalledRuntimeManifest {
            runtime_key: "release-pip-gfx120x-all-7-13-0a20260511".to_owned(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "test".to_owned(),
            version: "7.13.0a20260511".to_owned(),
            install_root: runtime_root,
            selected_artifact_url: "https://example.invalid/simple".to_owned(),
            source_layout_generation: None,
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: Some(python.display().to_string()),
            pip_cache_dir: None,
            rocm_sdk: Some(therock::RocmSdkPythonProbe {
                import_ok: true,
                root_path: Some(sdk_root),
                bin_path: Some(sdk_bin),
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
            }),
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms: 100,
        };

        let manifests = [manifest];
        let selected =
            single_ready_runtime_or_refusal(classify_single_ready_runtime(manifests.to_vec()))?;
        assert_eq!(
            selected.runtime_key,
            "release-pip-gfx120x-all-7-13-0a20260511"
        );
        Ok(())
    }

    /// Every ambiguity / not-found remediation message must steer the user to a
    /// concrete next step: the TUI picker, the durable activate command, the
    /// per-invocation flag, and the discovery command. Asserted centrally so a
    /// future reworded branch that silently drops one is caught.
    fn assert_actionable(message: &str) {
        for needle in [
            "/runtimes",
            "rocm runtimes activate",
            "--runtime-id <key>",
            "rocm runtimes list",
        ] {
            assert!(
                message.contains(needle),
                "actionable error must mention {needle:?}, got: {message}"
            );
        }
    }

    /// Build a runtime that registers but is not usable, so the "no ready
    /// runtime" path has a per-manifest reason to surface. A missing install
    /// root fails `validate_runtime_manifest_for_activation`'s first on-disk
    /// check, yielding `unusable (install root is missing: …)`.
    fn unusable_runtime_manifest(
        paths: &AppPaths,
        runtime_key: &str,
    ) -> Result<therock::InstalledRuntimeManifest> {
        let mut manifest = ready_runtime_manifest(paths, runtime_key)?;
        manifest.install_root = paths
            .data_dir
            .join("missing-runtime-root")
            .join(runtime_key);
        Ok(manifest)
    }

    #[test]
    fn ambiguous_ready_runtimes_name_flag_and_list_keys() -> Result<()> {
        let paths = test_paths("comfyui-ambiguous-ready-runtimes");
        let release = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let nightly = ready_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;

        let manifests = [release, nightly];
        let error =
            single_ready_runtime_or_refusal(classify_single_ready_runtime(manifests.to_vec()))
                .expect_err("two ready runtimes should be ambiguous");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("Multiple ROCm runtimes are ready"),
            "error should be the multi-ready branch, got: {message}"
        );
        assert!(
            message.contains("release-wheel-gfx94x-dcgpu-7-13-0")
                && message.contains("nightly-wheel-gfx94x-dcgpu-7-14-0"),
            "error should list the available runtime keys, got: {message}"
        );
        // This set is ready-filtered, so `format_available_runtime_keys`' bare
        // `key, key` is the deliberate rendering (see its doc comment): a
        // per-key status here would annotate every entry `(ready)` and say
        // nothing. Without this, swapping in `format_runtime_statuses` would
        // still pass the key assertions above.
        assert!(
            !message.contains("(ready)"),
            "a ready-filtered list must stay bare keys, not per-key statuses, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn no_ready_runtime_lists_each_key_and_reason() -> Result<()> {
        // The "none ready" branch used to duplicate the zero-runtime advice and
        // drop the reason each runtime is unusable. It must now name the key and
        // surface the reason `runtime_usability_status` already computed.
        let paths = test_paths("comfyui-no-ready-runtime");
        let unusable = unusable_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;

        let manifests = [unusable];
        let error =
            single_ready_runtime_or_refusal(classify_single_ready_runtime(manifests.to_vec()))
                .expect_err("an unusable runtime is not ready");
        let message = error.to_string();

        assert!(
            message.contains("release-wheel-gfx94x-dcgpu-7-13-0"),
            "error should name the unusable runtime key, got: {message}"
        );
        assert!(
            message.contains("install root is missing"),
            "error should surface the per-manifest unusable reason, got: {message}"
        );
        assert!(
            message.contains("rocm runtimes list"),
            "error should point to `rocm runtimes list`, got: {message}"
        );
        // The setup pointer must name a destination that actually exists. The
        // only ROCm-setup entry point a user can see is the bare-`rocm`
        // launcher row labelled "Set up this system"
        // (`crates/rocm-dash-tui/src/ui/launcher.rs`, rendered verbatim from
        // `ROWS`); there has never been a screen called "Set Up ROCm".
        //
        // The positive assertion already catches a plain revert to either wrong
        // wording on its own: neither the invented label this branch emitted
        // until now ("install another from Set Up ROCm") nor the superseded
        // pre-`31957ed` wording ("Set up ROCm first from Set Up ROCm") contains
        // `pick "Set up this system"`, so under either revert the positive
        // needle stops matching and reddens. Measured, not assumed: reverting
        // the production string to the invented label fails the positive
        // assertion, and fails the negative one too once the positive is
        // neutralized.
        //
        // The negative assertion is therefore not redundant, but its value is
        // the case the positive one cannot see: a message carrying *both*
        // labels. Re-adding "Set Up ROCm" alongside the real row still
        // satisfies the positive needle, so only the negative one reddens —
        // measured the same way. That is the drift worth guarding here, since
        // the invented screen name is likelier to creep back as an "extra"
        // pointer than as a wholesale revert.
        //
        // Nor can `assert_actionable` cover this branch — it is deliberately
        // excluded, since the none-ready message carries only one of its four
        // needles (`rocm runtimes list`); the other three are remedies for
        // choosing between runtimes, which do not apply when none is usable.
        assert!(
            message.contains("pick \"Set up this system\""),
            "error should still point at the launcher's \"Set up this system\" row, got: {message}"
        );
        assert!(
            !message.contains("Set Up ROCm"),
            "the setup pointer must not name the nonexistent `Set Up ROCm` screen, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn zero_runtime_setup_pointers_name_the_real_launcher_row() -> Result<()> {
        // The none-ready pointer above is one of three siblings that send a
        // user off to install ROCm; the other two fire when the registry is
        // empty rather than merely unusable. Until now only the none-ready one
        // was pinned, so these two could drift back to the invented `Set Up
        // ROCm` screen name without reddening anything. Same needles, same
        // reasoning as `no_ready_runtime_lists_each_key_and_reason`: the
        // positive one catches a wholesale revert, the negative one catches the
        // invented label creeping back *alongside* the real row.
        //
        // `test_paths` never creates the runtime registry directory, so
        // `load_runtime_manifests` returns empty and both zero-runtime branches
        // are the ones under test.
        let paths = test_paths("comfyui-zero-runtime-setup-pointers");

        // `render_status`' "ROCm" note (the zero-runtime arm of `render_status`).
        let status = render_status(&paths, &RocmCliConfig::default())?;
        assert!(
            status.contains("pick \"Set up this system\""),
            "status should point at the launcher's \"Set up this system\" row, got: {status}"
        );
        assert!(
            !status.contains("Set Up ROCm"),
            "status must not name the nonexistent `Set Up ROCm` screen, got: {status}"
        );

        // `select_runtime`'s bail when nothing is registered at all.
        let error = persisting_select_runtime(&paths, &RocmCliConfig::default(), None)
            .expect_err("no registered runtime means no selection");
        let message = error.to_string();
        assert!(
            message.contains("pick \"Set up this system\""),
            "the zero-runtime refusal should point at the launcher's \"Set up this system\" row, got: {message}"
        );
        assert!(
            !message.contains("Set Up ROCm"),
            "the zero-runtime refusal must not name the nonexistent `Set Up ROCm` screen, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn configured_default_not_found_is_actionable() -> Result<()> {
        // This branch enumerates the WHOLE registry, so it is planted with one
        // ready and one unusable runtime: a bare key list here would advise
        // `rocm runtimes activate` on the unusable one and land the user on a
        // second error, so each entry must carry its status.
        let paths = test_paths("comfyui-default-not-found");
        let ready = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let unusable = unusable_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;

        let manifests = [ready, unusable];
        let error = select_runtime_by_default_id(&manifests, "no-such-runtime-id")
            .expect_err("configured default id matches nothing");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("The configured default ROCm runtime was not found"),
            "error should be the configured-default-not-found branch, got: {message}"
        );
        assert!(
            message.contains("`release-wheel-gfx94x-dcgpu-7-13-0` (ready)"),
            "the usable runtime must be listed as ready, got: {message}"
        );
        assert!(
            message.contains("`nightly-wheel-gfx94x-dcgpu-7-14-0` (unusable")
                && message.contains("install root is missing"),
            "the unusable runtime must carry its reason, not read as pickable, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn multiple_configured_default_matches_is_actionable() -> Result<()> {
        // `ready_runtime_manifest` hardcodes the same runtime_id for every key,
        // so a configured default resolving to that id matches both entries and
        // exercises `select_default_runtime`'s own ambiguity arm (distinct from
        // the single-ready arm the other test hits).
        let paths = test_paths("comfyui-default-ambiguous");
        let release = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let nightly = ready_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;

        let manifests = [release, nightly];
        let error = select_runtime_by_default_id(&manifests, "therock-release:gfx120X-all")
            .expect_err("two manifests share the configured default id");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("More than one ROCm runtime matches the configured default"),
            "error should be the ambiguous-configured-default branch, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn ambiguous_configured_default_marks_the_unusable_match() -> Result<()> {
        // The ambiguity set is filtered by `runtime_id` alone, and a shared
        // runtime_id says nothing about readiness: one of the two matches here
        // has a missing install root. A bare key list would tell the user to
        // `rocm runtimes activate` it and land them on a second error, so each
        // match must carry the status `runtime_usability_status` computes.
        let paths = test_paths("comfyui-default-ambiguous-unusable");
        let ready = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let unusable = unusable_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;
        assert_eq!(
            ready.runtime_id, unusable.runtime_id,
            "the two manifests must share a runtime_id to reach the ambiguity arm"
        );

        let manifests = [ready, unusable];
        let error = select_runtime_by_default_id(&manifests, "therock-release:gfx120X-all")
            .expect_err("two manifests share the configured default id");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("More than one ROCm runtime matches the configured default"),
            "error should be the ambiguous-configured-default branch, got: {message}"
        );
        assert!(
            message.contains("`release-wheel-gfx94x-dcgpu-7-13-0` (ready)"),
            "the usable match must be listed as ready, got: {message}"
        );
        assert!(
            message.contains("`nightly-wheel-gfx94x-dcgpu-7-14-0` (unusable")
                && message.contains("install root is missing"),
            "the unusable match must carry its reason, not read as pickable, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn unknown_selector_is_actionable() -> Result<()> {
        // Whole-registry list again — see `configured_default_not_found_is_actionable`.
        let paths = test_paths("comfyui-selector-not-found");
        let ready = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let unusable = unusable_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;

        let manifests = [ready, unusable];
        let error = select_runtime_by_selector(&manifests, "no-such-key")
            .expect_err("selector matches no runtime");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("ROCm runtime not found: `no-such-key`"),
            "error should be the unknown-selector branch, got: {message}"
        );
        assert!(
            message.contains("`release-wheel-gfx94x-dcgpu-7-13-0` (ready)"),
            "the usable runtime must be listed as ready, got: {message}"
        );
        assert!(
            message.contains("`nightly-wheel-gfx94x-dcgpu-7-14-0` (unusable")
                && message.contains("install root is missing"),
            "the unusable runtime must carry its reason, not read as pickable, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn ambiguous_selector_is_actionable() -> Result<()> {
        // Both manifests share runtime_id `therock-release:gfx120X-all`; neither
        // runtime_key equals it, so selection falls through to the id filter and
        // matches both, hitting `select_runtime_by_selector`'s ambiguity arm.
        let paths = test_paths("comfyui-selector-ambiguous");
        let release = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let nightly = ready_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;

        let manifests = [release, nightly];
        let error = select_runtime_by_selector(&manifests, "therock-release:gfx120X-all")
            .expect_err("selector id matches two runtimes");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("More than one ROCm runtime matches `therock-release:gfx120X-all`"),
            "error should be the ambiguous-selector branch, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn ambiguous_selector_marks_the_unusable_match() -> Result<()> {
        // Selector twin of `ambiguous_configured_default_marks_the_unusable_match`:
        // the selector falls through to the `runtime_id` filter, which is not a
        // readiness check, so the unusable match must be rendered with its
        // reason rather than as a bare activatable key.
        let paths = test_paths("comfyui-selector-ambiguous-unusable");
        let ready = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        let unusable = unusable_runtime_manifest(&paths, "nightly-wheel-gfx94x-dcgpu-7-14-0")?;
        assert_eq!(
            ready.runtime_id, unusable.runtime_id,
            "the two manifests must share a runtime_id to reach the ambiguity arm"
        );

        let manifests = [ready, unusable];
        let error = select_runtime_by_selector(&manifests, "therock-release:gfx120X-all")
            .expect_err("selector id matches two runtimes");
        let message = error.to_string();

        assert_actionable(&message);
        assert!(
            message.contains("More than one ROCm runtime matches `therock-release:gfx120X-all`"),
            "error should be the ambiguous-selector branch, got: {message}"
        );
        assert!(
            message.contains("`release-wheel-gfx94x-dcgpu-7-13-0` (ready)"),
            "the usable match must be listed as ready, got: {message}"
        );
        assert!(
            message.contains("`nightly-wheel-gfx94x-dcgpu-7-14-0` (unusable")
                && message.contains("install root is missing"),
            "the unusable match must carry its reason, not read as pickable, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn selected_unusable_runtime_reports_not_ready_reason() -> Result<()> {
        // Naming a runtime explicitly by selector bypasses the readiness filter
        // in `select_default_runtime` / `select_single_ready_runtime`, so
        // `select_runtime` re-checks it and bails with the reworked
        // "The selected ROCm runtime is not ready: <reason>" message. This is
        // also why the sibling "has no Python executable" arm is unreachable:
        // `validate_wheel_runtime_manifest` already rejects a ready wheel runtime
        // whose python is missing, so any such runtime lands on this not-ready
        // arm — carrying the reason — rather than on the no-python arm.
        let paths = test_paths("comfyui-selected-unusable");
        let unusable = unusable_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        write_runtime_manifest(&paths, &unusable)?;

        let error = persisting_select_runtime(
            &paths,
            &RocmCliConfig::default(),
            Some("release-wheel-gfx94x-dcgpu-7-13-0"),
        )
        .expect_err("an explicitly selected unusable runtime is not ready");
        let message = error.to_string();

        assert!(
            message.contains("The selected ROCm runtime is not ready"),
            "error should use the reworked `ROCm runtime` noun, got: {message}"
        );
        assert!(
            message.contains("install root is missing"),
            "error should surface the per-manifest unusable reason, got: {message}"
        );
        Ok(())
    }

    #[test]
    fn runtime_environment_preloads_managed_rocm_paths() -> Result<()> {
        let paths = test_paths("comfyui-runtime-env");
        let env_root = paths.data_dir.join("envs").join("therock");
        let python = runtime_python_executable_in_env(&env_root);
        let python_bin = python
            .parent()
            .context("runtime Python path has no parent")?
            .to_path_buf();
        let sdk_root = paths.data_dir.join("runtimes").join("sdk");
        let sdk_bin = sdk_root.join("bin");
        let sdk_lib = sdk_root.join("lib");
        let runtime_root = paths.data_dir.join("runtimes").join("default");
        let runtime_bin = runtime_root.join("bin");
        let runtime_lib = runtime_root.join("lib");
        let runtime_sysdeps = runtime_root.join("lib").join("rocm_sysdeps").join("lib");
        for dir in [
            &python_bin,
            &sdk_bin,
            &sdk_lib,
            &runtime_bin,
            &runtime_lib,
            &runtime_sysdeps,
        ] {
            fs::create_dir_all(dir)?;
        }

        let manifest = therock::InstalledRuntimeManifest {
            runtime_key: "therock-release-gfx120x-all".to_owned(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "test".to_owned(),
            version: "7.13.0a20260511".to_owned(),
            install_root: runtime_root.clone(),
            selected_artifact_url: "https://example.invalid/simple".to_owned(),
            source_layout_generation: None,
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: Some(python.display().to_string()),
            pip_cache_dir: None,
            rocm_sdk: Some(therock::RocmSdkPythonProbe {
                import_ok: true,
                root_path: Some(sdk_root.clone()),
                bin_path: Some(sdk_bin.clone()),
                runtime_roots: vec![runtime_root],
                bin_paths: vec![sdk_bin.clone()],
                library_paths: vec![sdk_lib.clone()],
                ..Default::default()
            }),
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms: 100,
        };

        let env = runtime_environment_from_runtime(&manifest, &python);
        let mut command = Command::new(&python);
        apply_runtime_environment(&mut command, &env)?;

        assert_eq!(
            command_env_value(&command, "VIRTUAL_ENV").as_deref(),
            Some(env_root.as_os_str())
        );
        assert_eq!(
            command_env_value(&command, "ROCM_PATH").as_deref(),
            Some(sdk_root.as_os_str())
        );
        let path = command_env_value(&command, "PATH").context("PATH should be set")?;
        let path_entries = split_runtime_paths(&path);
        assert!(path_entries.contains(&python_bin));
        assert!(path_entries.contains(&sdk_bin));
        assert!(path_entries.contains(&runtime_bin));

        if runtime_is_windows() {
            assert!(path_entries.contains(&sdk_lib));
            assert!(path_entries.contains(&runtime_lib));
        } else if runtime_is_linux() {
            let ld_library_path =
                command_env_value(&command, "LD_LIBRARY_PATH").context("LD_LIBRARY_PATH")?;
            let library_entries = split_runtime_paths(&ld_library_path);
            assert!(library_entries.contains(&sdk_lib));
            assert!(library_entries.contains(&runtime_lib));
            assert!(library_entries.contains(&runtime_sysdeps));
        }
        Ok(())
    }

    /// Zero runtimes registered: the refusal points at the launcher's setup row,
    /// and selecting wrote nothing.
    #[test]
    fn comfyui_selection_with_no_registered_runtime_says_to_install_rocm_and_writes_nothing()
    -> Result<()> {
        let paths = test_paths("comfyui-selection-none-registered");

        let message = persisting_select_runtime(&paths, &RocmCliConfig::default(), None)
            .expect_err("no registered runtime means no selection")
            .to_string();

        assert!(
            message.contains("Install ROCm first")
                && message.contains("pick \"Set up this system\""),
            "unexpected refusal: {message}"
        );
        assert_eq!(runtime_registry_entry_names(&paths), Vec::<String>::new());
        Ok(())
    }

    /// One ready runtime and nothing chosen: it is the selection.
    #[test]
    fn comfyui_selection_with_one_ready_runtime_selects_it() -> Result<()> {
        let paths = test_paths("comfyui-selection-one-ready");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;

        let selected = persisting_select_runtime(&paths, &RocmCliConfig::default(), None)?;

        assert_eq!(selected.manifest.runtime_key, manifest.runtime_key);
        assert_eq!(
            Some(selected.python.display().to_string()),
            manifest.python_executable
        );
        Ok(())
    }

    /// Registered but none ready: the refusal lists the runtime with the reason
    /// it is unusable, and fixing that reason (here: putting the interpreter
    /// back) is what clears it.
    #[test]
    fn comfyui_selection_with_only_an_unusable_runtime_clears_once_it_is_fixed() -> Result<()> {
        let paths = test_paths("comfyui-selection-none-ready");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let python = PathBuf::from(
            manifest
                .python_executable
                .as_deref()
                .context("the fixture records an interpreter")?,
        );
        fs::remove_file(&python)?;

        let message = persisting_select_runtime(&paths, &RocmCliConfig::default(), None)
            .expect_err("an unusable runtime is not ready")
            .to_string();

        assert!(
            message.contains("No ROCm runtime is ready")
                && message.contains("rocm runtimes list")
                && message.contains(&format!("`{}` (unusable", manifest.runtime_key)),
            "unexpected refusal: {message}"
        );
        assert_eq!(
            runtime_registry_entry_names(&paths),
            vec![format!("{}.json", manifest.runtime_key)]
        );

        fs::write(&python, "python")?;
        let selected = persisting_select_runtime(&paths, &RocmCliConfig::default(), None)?;
        assert_eq!(selected.manifest.runtime_key, manifest.runtime_key);
        Ok(())
    }

    /// Several ready runtimes and none chosen: the refusal names the remedies,
    /// and each one clears it. `rocm runtimes activate <key>` is run as
    /// `activate_runtime`, its implementation, and `--runtime-id` as the selector.
    #[test]
    fn comfyui_selection_with_several_ready_runtimes_names_remedies_that_clear_it() -> Result<()> {
        let paths = test_paths("comfyui-selection-several-ready");
        let first = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let second = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
            "therock-release:gfx120X-all",
            "7.14.0",
            20,
        )?;
        let mut config = RocmCliConfig::default();

        let message = persisting_select_runtime(&paths, &config, None)
            .expect_err("several ready runtimes and no choice must not select")
            .to_string();

        assert!(
            message.contains("Multiple ROCm runtimes are ready")
                && message.contains("rocm runtimes activate <key>")
                && message.contains("--runtime-id <key>")
                && message.contains(&first.runtime_key)
                && message.contains(&second.runtime_key),
            "unexpected refusal: {message}"
        );

        let selected = persisting_select_runtime(&paths, &config, Some(&first.runtime_key))?;
        assert_eq!(selected.manifest.runtime_key, first.runtime_key);

        crate::runtimes_cmd::activate_runtime(&paths, &mut config, &second.runtime_key)?;
        let selected = persisting_select_runtime(&paths, &config, None)?;
        assert_eq!(selected.manifest.runtime_key, second.runtime_key);
        Ok(())
    }

    /// The setup-recovery half of the rule: a ready runtime that only
    /// `recover_setup_runtime_registration` can make visible is found by the
    /// ComfyUI path, as it is by the engine paths, and finding it re-files its
    /// registry entry, so the selection names a key `rocm runtimes list` shows.
    #[test]
    fn comfyui_selection_recovers_the_setup_runtime_the_registry_lost() -> Result<()> {
        let paths = test_paths("comfyui-selection-setup-recovery");
        let manifest = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
        )?;
        let config = RocmCliConfig::load(&paths)?;
        assert!(
            config.setup.therock_venv.is_some(),
            "setup must name the folder for this test to mean anything"
        );
        assert_eq!(runtime_registry_entry_names(&paths), Vec::<String>::new());

        let selected = persisting_select_runtime(&paths, &config, None)?;

        assert_eq!(selected.manifest.runtime_key, manifest.runtime_key);
        assert_eq!(
            runtime_registry_entry_names(&paths),
            vec![format!("{}.json", manifest.runtime_key)]
        );
        Ok(())
    }

    /// Recovering the setup runtime can fail, and the ComfyUI path now reports
    /// that failure as the engine paths do, rather than carrying on as if setup
    /// had named nothing: a setup folder whose runtime is no longer usable is
    /// named, with the reason, and nothing is written to the registry.
    #[test]
    fn comfyui_selection_reports_a_setup_runtime_that_is_no_longer_usable() -> Result<()> {
        let paths = test_paths("comfyui-selection-setup-unusable");
        let manifest = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
        )?;
        fs::remove_file(
            manifest
                .python_executable
                .as_deref()
                .context("the fixture records an interpreter")?,
        )?;
        let config = RocmCliConfig::load(&paths)?;

        let message = persisting_select_runtime(&paths, &config, None)
            .expect_err("an unusable setup runtime is an error")
            .to_string();

        assert!(
            message.contains("is not usable") && message.contains(&manifest.runtime_key),
            "got: {message}"
        );
        assert_eq!(runtime_registry_entry_names(&paths), Vec::<String>::new());
        Ok(())
    }

    /// A choice that names the runtime setup created, by `--runtime-id` (key or
    /// id), the active runtime or the configured default, finds it even though
    /// its registry entry was lost. That is what makes the `--runtime-id` an
    /// install preview prints a command that works from the state the preview
    /// left: a preview adds the runtime to what it searches and writes nothing,
    /// a real run writes the entry back. Each row asserts the pick and the
    /// registry together.
    #[test]
    fn a_choice_naming_the_setup_runtime_finds_it_after_its_registry_entry_was_lost() -> Result<()>
    {
        type Choice = fn(&therock::InstalledRuntimeManifest, &mut RocmCliConfig) -> Option<String>;
        let choices: [(&str, Choice); 4] = [
            ("explicit key", |manifest, _| {
                Some(manifest.runtime_key.clone())
            }),
            ("explicit id", |manifest, _| {
                Some(manifest.runtime_id.clone())
            }),
            ("active runtime", |manifest, config| {
                config.active_runtime_key = Some(manifest.runtime_key.clone());
                None
            }),
            ("configured default", |manifest, config| {
                config.default_runtime_id = Some(manifest.runtime_id.clone());
                None
            }),
        ];
        for (label, choice) in choices {
            for recovery in [SetupRecovery::PreviewOnly, SetupRecovery::Persist] {
                let paths = test_paths(&format!(
                    "comfyui-choice-setup-{}-{recovery:?}",
                    label.replace(' ', "-")
                ));
                let manifest = crate::test_support::write_test_setup_runtime_missing_from_registry(
                    &paths,
                    "release-pip-gfx120x-all-7-14-0",
                )?;
                let mut config = RocmCliConfig::load(&paths)?;
                let selector = choice(&manifest, &mut config);

                let selected = select_runtime(&paths, &config, selector.as_deref(), recovery)
                    .with_context(|| format!("{label}, {recovery:?}"))?;

                assert_eq!(
                    selected.manifest.runtime_key, manifest.runtime_key,
                    "{label}, {recovery:?}"
                );
                let expected_entries = match recovery {
                    SetupRecovery::PreviewOnly => Vec::new(),
                    SetupRecovery::Persist => vec![format!("{}.json", manifest.runtime_key)],
                };
                assert_eq!(
                    runtime_registry_entry_names(&paths),
                    expected_entries,
                    "{label}, {recovery:?}"
                );
            }
        }
        Ok(())
    }

    /// Finding the setup runtime for a named choice must not hide that its
    /// registry entry cannot be written back. The preview of such an install
    /// succeeds, because it writes nothing; the printed command is the real run,
    /// and it has to say why it cannot restore the entry rather than answer as if
    /// there were no runtime ("Install ROCm first"). Each row asserts the error
    /// and the registry together: the failing run leaves the registry as it
    /// found it, and the runtime was still found (the preview, and the
    /// unwritable registry's own refusal for a name that is not setup's).
    #[test]
    fn a_choice_naming_the_setup_runtime_reports_a_registry_that_cannot_take_it_back() -> Result<()>
    {
        type Choice = fn(&therock::InstalledRuntimeManifest, &mut RocmCliConfig) -> Option<String>;
        let choices: [(&str, Choice); 5] = [
            ("explicit key", |manifest, _| {
                Some(manifest.runtime_key.clone())
            }),
            ("explicit id", |manifest, _| {
                Some(manifest.runtime_id.clone())
            }),
            ("active runtime", |manifest, config| {
                config.active_runtime_key = Some(manifest.runtime_key.clone());
                None
            }),
            ("configured default", |manifest, config| {
                config.default_runtime_id = Some(manifest.runtime_id.clone());
                None
            }),
            // Not a named choice, but the same persistence, reported the same way.
            ("no choice", |_, _| None),
        ];
        for (label, choice) in choices {
            let paths = test_paths(&format!(
                "comfyui-choice-unwritable-{}",
                label.replace(' ', "-")
            ));
            let manifest = crate::test_support::write_test_setup_runtime_missing_from_registry(
                &paths,
                "release-pip-gfx120x-all-7-14-0",
            )?;
            let mut config = RocmCliConfig::load(&paths)?;
            let selector = choice(&manifest, &mut config);
            let registry = crate::test_support::make_runtime_registry_unwritable(&paths)?;

            let previewed = select_runtime(
                &paths,
                &config,
                selector.as_deref(),
                SetupRecovery::PreviewOnly,
            )
            .with_context(|| format!("{label}: the preview writes nothing"))?;
            assert_eq!(
                previewed.manifest.runtime_key, manifest.runtime_key,
                "{label}"
            );

            let error =
                select_runtime(&paths, &config, selector.as_deref(), SetupRecovery::Persist)
                    .expect_err("a registry that cannot be written cannot take the entry back");
            let message = format!("{error:#}");
            assert!(
                message.contains(&format!(
                    "failed to restore setup runtime `{}`",
                    manifest.runtime_key
                )),
                "{label}: got: {message}"
            );
            assert!(
                !message.contains("Install ROCm first") && !message.contains("not found"),
                "{label}: the persistence error must not be replaced by a refusal about a missing runtime: {message}"
            );
            assert_eq!(
                fs::read_to_string(&registry)?,
                "not a directory",
                "{label}: a failed restore leaves the registry as it was"
            );
        }
        Ok(())
    }

    /// The same, with the registry readable and holding another runtime, which is
    /// the state a read-only registry directory leaves. The refusal that stands
    /// without the fix is then the registry's "not found", so the error is
    /// compared with that. Where the directory stays writable regardless (an
    /// administrator), the write succeeds and the test checks the restore
    /// instead, so it cannot pass by writing nothing.
    #[cfg(unix)]
    #[test]
    fn a_read_only_registry_directory_is_reported_not_hidden_behind_a_not_found() -> Result<()> {
        use crate::runtimes_cmd::runtime_registry_dir;
        use std::os::unix::fs::PermissionsExt;

        let paths = test_paths("comfyui-choice-read-only-registry");
        let setup = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
        )?;
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig::load(&paths)?;
        let registry = runtime_registry_dir(&paths);
        let original = fs::metadata(&registry)?.permissions();
        fs::set_permissions(&registry, fs::Permissions::from_mode(0o555))?;

        let selected = select_runtime(
            &paths,
            &config,
            Some(&setup.runtime_key),
            SetupRecovery::Persist,
        );

        fs::set_permissions(&registry, original)?;
        match selected {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains(&format!(
                        "failed to restore setup runtime `{}`",
                        setup.runtime_key
                    )) && !message.contains("not found"),
                    "got: {message}"
                );
                assert!(
                    !runtime_registry_entry_names(&paths)
                        .contains(&format!("{}.json", setup.runtime_key)),
                    "a failed restore writes no entry"
                );
            }
            Ok(selected) => {
                assert_eq!(selected.manifest.runtime_key, setup.runtime_key);
                assert!(
                    runtime_registry_entry_names(&paths)
                        .contains(&format!("{}.json", setup.runtime_key)),
                    "the directory stayed writable, so the entry is restored"
                );
            }
        }
        Ok(())
    }

    /// What the recovery must not turn into: a choice that names some other
    /// runtime is answered by the registry whether or not setup's runtime is
    /// usable, a name that matches nothing is still refused (with the setup
    /// runtime among those found when it is usable), and an unusable setup
    /// runtime leaves the registry's own refusal standing instead of replacing
    /// it with an error about a runtime the user did not ask for.
    #[test]
    fn recovering_the_setup_runtime_does_not_change_the_answer_for_other_choices() -> Result<()> {
        let paths = test_paths("comfyui-choice-other-than-setup");
        let setup = crate::test_support::write_test_setup_runtime_missing_from_registry(
            &paths,
            "release-pip-gfx120x-all-7-14-0",
        )?;
        let other = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig::load(&paths)?;
        let registry = runtime_registry_entry_names(&paths);
        assert_eq!(registry, [format!("{}.json", other.runtime_key)]);

        let picked = select_runtime(
            &paths,
            &config,
            Some(&other.runtime_key),
            SetupRecovery::PreviewOnly,
        )?;
        assert_eq!(picked.manifest.runtime_key, other.runtime_key);

        let miss = select_runtime(
            &paths,
            &config,
            Some("no-such-runtime"),
            SetupRecovery::PreviewOnly,
        )
        .expect_err("nothing is called that")
        .to_string();
        assert!(
            miss.contains("ROCm runtime not found: `no-such-runtime`")
                && miss.contains(&setup.runtime_key),
            "{miss}"
        );
        assert_eq!(runtime_registry_entry_names(&paths), registry);

        fs::remove_file(
            setup
                .python_executable
                .as_deref()
                .context("the fixture records an interpreter")?,
        )?;
        for recovery in [SetupRecovery::PreviewOnly, SetupRecovery::Persist] {
            let picked = select_runtime(&paths, &config, Some(&other.runtime_key), recovery)
                .with_context(|| format!("an unusable setup runtime, {recovery:?}"))?;
            assert_eq!(picked.manifest.runtime_key, other.runtime_key);
            let miss = select_runtime(&paths, &config, Some("no-such-runtime"), recovery)
                .expect_err("nothing is called that")
                .to_string();
            assert!(
                miss.contains("ROCm runtime not found: `no-such-runtime`")
                    && !miss.contains("is not usable"),
                "{recovery:?}: {miss}"
            );
            assert_eq!(
                runtime_registry_entry_names(&paths),
                registry,
                "{recovery:?}"
            );
        }
        Ok(())
    }

    /// A registry directory that stops writing, as far as this process can be
    /// stopped: a read-only directory, which does not stop `root`. `None` where it
    /// would not bite (the process can write anyway, or the host has no such
    /// modes), so a caller skips the row instead of passing for the wrong reason.
    /// Restores the mode on drop.
    #[cfg(unix)]
    struct ReadOnlyRegistry {
        path: PathBuf,
        original: fs::Permissions,
    }

    #[cfg(unix)]
    impl Drop for ReadOnlyRegistry {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.path, self.original.clone());
        }
    }

    #[cfg(unix)]
    fn make_registry_read_only(paths: &AppPaths) -> Option<ReadOnlyRegistry> {
        use std::os::unix::fs::PermissionsExt;

        let path = crate::runtimes_cmd::runtime_registry_dir(paths);
        let original = fs::metadata(&path).ok()?.permissions();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o555)).ok()?;
        let guard = ReadOnlyRegistry { path, original };
        let probe = guard.path.join(".write-probe");
        if fs::write(&probe, "").is_ok() {
            let _ = fs::remove_file(probe);
            return None;
        }
        Some(guard)
    }

    #[cfg(not(unix))]
    struct ReadOnlyRegistry;

    #[cfg(not(unix))]
    fn make_registry_read_only(_paths: &AppPaths) -> Option<ReadOnlyRegistry> {
        None
    }

    /// Every state `rocm comfyui install` can pick a runtime from, one row each:
    /// what the user chose (nothing, a registered runtime by name, the setup
    /// runtime by name) against how many ready runtimes the registry holds (0, 1,
    /// 2), what the setup folder holds (nothing, a usable runtime whose registry
    /// entry was lost, a runtime that is no longer usable), whether the registry
    /// takes a new entry, and whether the install is a preview or a real run.
    ///
    /// The rules the rows pin: a choice that names a registered runtime is
    /// answered by the registry whatever setup holds. With nothing chosen the
    /// ready runtimes are counted, and a setup runtime that cannot be used is not
    /// one of them: it does not stop a registered ready runtime from being picked
    /// (one) or refused as ambiguous (two), and is reported only when it was the
    /// last hope (none registered). A setup runtime that can be used is counted,
    /// and a failure to write its entry back is reported as that, whatever else
    /// the count would have said. Each row asserts the message and the registry
    /// together.
    #[test]
    fn comfyui_runtime_selection_state_matrix() -> Result<()> {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Chosen {
            Nothing,
            Registered,
            SetupRuntime,
        }
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum SetupFolder {
            Absent,
            Usable,
            Unusable,
        }
        struct Expected {
            selected: Option<&'static str>,
            error_has: Vec<String>,
            error_lacks: &'static [&'static str],
            setup_entry_written: bool,
        }

        const SETUP_KEY: &str = "runtime-s";
        const MULTIPLE: &str = "Multiple ROCm runtimes are ready";
        let registered_keys = ["runtime-a", "runtime-b"];
        let restore_failed = format!("failed to restore setup runtime `{SETUP_KEY}`");
        let mut rows_run = 0;
        let mut rows_skipped = 0;

        for chosen in [Chosen::Nothing, Chosen::Registered, Chosen::SetupRuntime] {
            for ready in 0..=2usize {
                for setup in [
                    SetupFolder::Absent,
                    SetupFolder::Usable,
                    SetupFolder::Unusable,
                ] {
                    for registry_writable in [true, false] {
                        for recovery in [SetupRecovery::Persist, SetupRecovery::PreviewOnly] {
                            if chosen == Chosen::Registered && ready == 0 {
                                continue; // nothing registered to name
                            }
                            let row = format!(
                                "{chosen:?}, {ready} ready, setup {setup:?}, writable {registry_writable}, {recovery:?}"
                            );
                            let paths = test_paths(&format!(
                                "comfyui-selection-matrix-{}",
                                rows_run + rows_skipped
                            ));
                            for (index, key) in registered_keys.iter().take(ready).enumerate() {
                                write_test_pip_runtime(
                                    &paths,
                                    key,
                                    "therock-release:gfx120X-all",
                                    "7.14.0",
                                    10 + index as u128,
                                )?;
                            }
                            if setup != SetupFolder::Absent {
                                let manifest =
                                    crate::test_support::write_test_setup_runtime_missing_from_registry(
                                        &paths, SETUP_KEY,
                                    )?;
                                if setup == SetupFolder::Unusable {
                                    fs::remove_file(
                                        manifest
                                            .python_executable
                                            .as_deref()
                                            .context("the fixture records an interpreter")?,
                                    )?;
                                }
                            }
                            let config = RocmCliConfig::load(&paths)?;
                            // A registry that cannot take an entry: a plain file for
                            // an empty one (stops `root` too), a read-only directory
                            // where it holds runtimes.
                            let registry_path = crate::runtimes_cmd::runtime_registry_dir(&paths);
                            let _read_only;
                            if !registry_writable {
                                if ready == 0 {
                                    crate::test_support::make_runtime_registry_unwritable(&paths)?;
                                } else {
                                    let Some(guard) = make_registry_read_only(&paths) else {
                                        rows_skipped += 1;
                                        continue;
                                    };
                                    _read_only = guard;
                                }
                            }
                            rows_run += 1;

                            let persists = recovery == SetupRecovery::Persist;
                            let setup_usable = setup == SetupFolder::Usable;
                            let write_fails = persists && !registry_writable && setup_usable;
                            let writes = persists && registry_writable && setup_usable;
                            let ready_keys = registered_keys[..ready].to_vec();
                            let expected = match chosen {
                                Chosen::Nothing => match (setup, ready) {
                                    (SetupFolder::Usable, _) if write_fails => Expected {
                                        selected: None,
                                        error_has: vec![restore_failed.clone()],
                                        error_lacks: &["Install ROCm first", MULTIPLE],
                                        setup_entry_written: false,
                                    },
                                    (SetupFolder::Usable, 0) => Expected {
                                        selected: Some(SETUP_KEY),
                                        error_has: vec![],
                                        error_lacks: &[],
                                        setup_entry_written: writes,
                                    },
                                    (SetupFolder::Usable, _) => Expected {
                                        selected: None,
                                        error_has: std::iter::once(MULTIPLE.to_owned())
                                            .chain(ready_keys.iter().map(|key| (*key).to_owned()))
                                            .chain([SETUP_KEY.to_owned()])
                                            .collect(),
                                        error_lacks: &["Install ROCm first"],
                                        setup_entry_written: writes,
                                    },
                                    (SetupFolder::Unusable, 0) => Expected {
                                        selected: None,
                                        error_has: vec![
                                            format!("setup runtime `{SETUP_KEY}`"),
                                            "is not usable".to_owned(),
                                        ],
                                        error_lacks: &["Install ROCm first"],
                                        setup_entry_written: false,
                                    },
                                    (SetupFolder::Absent, 0) => Expected {
                                        selected: None,
                                        error_has: vec!["Install ROCm first".to_owned()],
                                        error_lacks: &["is not usable"],
                                        setup_entry_written: false,
                                    },
                                    (_, 1) => Expected {
                                        selected: Some("runtime-a"),
                                        error_has: vec![],
                                        error_lacks: &[],
                                        setup_entry_written: false,
                                    },
                                    (_, _) => Expected {
                                        selected: None,
                                        error_has: std::iter::once(MULTIPLE.to_owned())
                                            .chain(ready_keys.iter().map(|key| (*key).to_owned()))
                                            .collect(),
                                        error_lacks: &["is not usable", SETUP_KEY],
                                        setup_entry_written: false,
                                    },
                                },
                                Chosen::Registered => Expected {
                                    selected: Some("runtime-a"),
                                    error_has: vec![],
                                    error_lacks: &[],
                                    setup_entry_written: false,
                                },
                                Chosen::SetupRuntime => match setup {
                                    SetupFolder::Usable if write_fails => Expected {
                                        selected: None,
                                        error_has: vec![restore_failed.clone()],
                                        error_lacks: &["Install ROCm first", "not found"],
                                        setup_entry_written: false,
                                    },
                                    SetupFolder::Usable => Expected {
                                        selected: Some(SETUP_KEY),
                                        error_has: vec![],
                                        error_lacks: &[],
                                        setup_entry_written: writes,
                                    },
                                    SetupFolder::Absent | SetupFolder::Unusable => Expected {
                                        selected: None,
                                        error_has: vec![if ready == 0 {
                                            "Install ROCm first".to_owned()
                                        } else {
                                            format!("ROCm runtime not found: `{SETUP_KEY}`")
                                        }],
                                        error_lacks: &["is not usable"],
                                        setup_entry_written: false,
                                    },
                                },
                            };

                            let selector = match chosen {
                                Chosen::Nothing => None,
                                Chosen::Registered => Some("runtime-a"),
                                Chosen::SetupRuntime => Some(SETUP_KEY),
                            };
                            let outcome = select_runtime(&paths, &config, selector, recovery);

                            match (outcome, expected.selected) {
                                (Ok(selected), Some(key)) => {
                                    assert_eq!(selected.manifest.runtime_key, key, "{row}");
                                }
                                (Ok(selected), None) => panic!(
                                    "{row}: expected a refusal, picked {}",
                                    selected.manifest.runtime_key
                                ),
                                (Err(error), Some(key)) => {
                                    panic!("{row}: expected {key}, got: {error:#}")
                                }
                                (Err(error), None) => {
                                    let message = format!("{error:#}");
                                    for needle in &expected.error_has {
                                        assert!(
                                            message.contains(needle.as_str()),
                                            "{row}: `{needle}` missing from: {message}"
                                        );
                                    }
                                    for needle in expected.error_lacks {
                                        assert!(
                                            !message.contains(needle),
                                            "{row}: `{needle}` must not appear in: {message}"
                                        );
                                    }
                                }
                            }

                            let mut entries: Vec<String> =
                                ready_keys.iter().map(|key| format!("{key}.json")).collect();
                            if expected.setup_entry_written {
                                entries.push(format!("{SETUP_KEY}.json"));
                            }
                            entries.sort();
                            assert_eq!(runtime_registry_entry_names(&paths), entries, "{row}");
                            if !registry_writable && ready == 0 {
                                assert_eq!(
                                    fs::read_to_string(&registry_path)?,
                                    "not a directory",
                                    "{row}: a failed restore leaves the registry as it was"
                                );
                            }
                        }
                    }
                }
            }
        }
        // The rows that need a registry holding runtimes that refuses a write skip
        // themselves where the process can write anyway; the rest must have run.
        assert_eq!(rows_run + rows_skipped, 96, "the matrix has 96 rows");
        assert!(rows_run >= 60, "ran {rows_run}, skipped {rows_skipped}");
        Ok(())
    }

    /// The two paths that pick "the one ready runtime" must not drift apart:
    /// for each registry state below, ComfyUI selects a runtime exactly when
    /// the engine paths do, and the same one. Both reach the rule through
    /// `select_single_ready_runtime`, so a copy of it re-introduced in either
    /// caller (a different readiness test, a missing setup recovery) shows up as
    /// a disagreement on one of these states.
    #[test]
    fn comfyui_and_engine_paths_agree_on_the_single_ready_runtime() -> Result<()> {
        use crate::engine_runtime_select::{
            resolve_engine_selection, validate_engine_selection_runtime,
        };

        #[derive(Clone, Copy)]
        enum State {
            Empty,
            OneUnusable,
            OneReady,
            TwoReady,
            ReadyAndUnusable,
            SetupRuntimeLost,
            SetupRuntimeLostBesideAReadyOne,
            UnusableSetupRuntimeBesideAReadyOne,
        }
        for (label, state, expected) in [
            ("empty", State::Empty, None),
            ("one-unusable", State::OneUnusable, None),
            ("one-ready", State::OneReady, Some("runtime-a")),
            ("two-ready", State::TwoReady, None),
            (
                "ready-and-unusable",
                State::ReadyAndUnusable,
                Some("runtime-a"),
            ),
            (
                "setup-runtime-lost",
                State::SetupRuntimeLost,
                Some("runtime-s"),
            ),
            (
                "setup-runtime-lost-beside-a-ready-one",
                State::SetupRuntimeLostBesideAReadyOne,
                None,
            ),
            (
                "unusable-setup-runtime-beside-a-ready-one",
                State::UnusableSetupRuntimeBesideAReadyOne,
                Some("runtime-a"),
            ),
        ] {
            // Each path gets its own copy of the state: a recovery run by one
            // writes the registry, which would otherwise hand the other path a
            // state it could not have found for itself.
            let plant_state = |paths: &AppPaths| -> Result<()> {
                let plant = |key: &str| {
                    write_test_pip_runtime(paths, key, "therock-release:gfx120X-all", "7.14.0", 20)
                };
                let break_runtime = |manifest: &therock::InstalledRuntimeManifest| -> Result<()> {
                    fs::remove_file(
                        manifest
                            .python_executable
                            .as_deref()
                            .context("the fixture records an interpreter")?,
                    )?;
                    Ok(())
                };
                match state {
                    State::Empty => {}
                    State::OneUnusable => break_runtime(&plant("runtime-a")?)?,
                    State::OneReady => {
                        plant("runtime-a")?;
                    }
                    State::TwoReady => {
                        plant("runtime-a")?;
                        plant("runtime-b")?;
                    }
                    State::ReadyAndUnusable => {
                        plant("runtime-a")?;
                        break_runtime(&plant("runtime-b")?)?;
                    }
                    State::SetupRuntimeLost => {
                        crate::test_support::write_test_setup_runtime_missing_from_registry(
                            paths,
                            "runtime-s",
                        )?;
                    }
                    State::SetupRuntimeLostBesideAReadyOne => {
                        plant("runtime-a")?;
                        crate::test_support::write_test_setup_runtime_missing_from_registry(
                            paths,
                            "runtime-s",
                        )?;
                    }
                    State::UnusableSetupRuntimeBesideAReadyOne => {
                        plant("runtime-a")?;
                        break_runtime(
                            &crate::test_support::write_test_setup_runtime_missing_from_registry(
                                paths,
                                "runtime-s",
                            )?,
                        )?;
                    }
                }
                Ok(())
            };
            let comfyui_paths = test_paths(&format!("comfyui-engine-agree-{label}-comfyui"));
            plant_state(&comfyui_paths)?;
            let comfyui = persisting_select_runtime(
                &comfyui_paths,
                &RocmCliConfig::load(&comfyui_paths)?,
                None,
            )
            .ok()
            .map(|selected| selected.manifest.runtime_key);
            let engine_paths = test_paths(&format!("comfyui-engine-agree-{label}-engine"));
            plant_state(&engine_paths)?;
            let engine = validate_engine_selection_runtime(
                &engine_paths,
                resolve_engine_selection(&RocmCliConfig::load(&engine_paths)?, "vllm", None, None),
            )?
            .runtime_id;

            assert_eq!(comfyui, engine, "{label}: the two paths disagree");
            assert_eq!(comfyui.as_deref(), expected, "{label}");
        }
        Ok(())
    }

    fn command_env_value(command: &Command, key: &str) -> Option<OsString> {
        command
            .get_envs()
            .find(|(name, _)| name.to_string_lossy() == key)
            .and_then(|(_, value)| value.map(OsString::from))
    }

    /// The environment folder ComfyUI exports as `VIRTUAL_ENV`, by the folder
    /// the interpreter sits in. Today only a `bin` or `Scripts` folder
    /// (any case) identifies a venv; anything else leaves `VIRTUAL_ENV` unset
    /// rather than pointing it at a guessed folder.
    #[test]
    fn runtime_environment_exports_virtual_env_only_for_a_bin_or_scripts_folder() -> Result<()> {
        let cases: [(&str, Option<&str>); 7] = [
            ("/opt/venv/bin/python", Some("/opt/venv")),
            ("/opt/venv/Scripts/python.exe", Some("/opt/venv")),
            ("/opt/venv/SCRIPTS/python.exe", Some("/opt/venv")),
            ("/opt/venv/BIN/python", Some("/opt/venv")),
            ("/opt/venv/tools/python", None),
            ("python", None),
            ("/python", None),
        ];
        for (python, expected) in cases {
            let python = PathBuf::from(python);
            let manifest = minimal_runtime_manifest(&python);

            let env = runtime_environment_from_runtime(&manifest, &python);
            let mut command = Command::new(&python);
            apply_runtime_environment(&mut command, &env)?;

            assert_eq!(
                command_env_value(&command, "VIRTUAL_ENV").as_deref(),
                expected.map(std::ffi::OsStr::new),
                "{}",
                python.display()
            );
        }
        Ok(())
    }

    /// A second amd-smi library in the ComfyUI process makes torch count zero
    /// devices; the launch environment must carry the link to the SDK's own.
    #[cfg(target_os = "linux")]
    #[test]
    fn runtime_environment_carries_the_amd_smi_link_dir() -> Result<()> {
        let paths = test_paths("comfyui-amd-smi-link");
        let venv_root = paths.data_dir.join("venv");
        let sdk_lib = paths.data_dir.join("sdk-lib");
        fs::create_dir_all(&sdk_lib)?;
        fs::write(sdk_lib.join("libamd_smi.so.26"), b"sdk")?;
        let env = ComfyUiRuntimeEnvironment {
            venv_root: Some(venv_root.clone()),
            library_entries: vec![sdk_lib.clone()],
            ..Default::default()
        };

        let assignments = runtime_environment_assignments(&env)?;

        let compat = venv_root.join(rocm_core::RUNTIME_COMPAT_DIR_NAME);
        let ld_library_path = assignments
            .iter()
            .find(|(key, _)| *key == "LD_LIBRARY_PATH")
            .map(|(_, value)| value.clone())
            .context("LD_LIBRARY_PATH was not assigned")?;
        let entries: Vec<PathBuf> = std::env::split_paths(&ld_library_path).collect();
        assert!(entries.contains(&compat), "{entries:?}");
        assert_eq!(
            fs::read_link(compat.join("libamd_smi.so"))?,
            sdk_lib.join("libamd_smi.so.26")
        );
        Ok(())
    }

    fn minimal_runtime_manifest(python: &Path) -> therock::InstalledRuntimeManifest {
        therock::InstalledRuntimeManifest {
            runtime_key: "therock-release-gfx120x-all".to_owned(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "test".to_owned(),
            version: "7.13.0a20260511".to_owned(),
            install_root: PathBuf::from("/nonexistent/runtime"),
            selected_artifact_url: "https://example.invalid/simple".to_owned(),
            source_layout_generation: None,
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: Some(python.display().to_string()),
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms: 100,
        }
    }
}
