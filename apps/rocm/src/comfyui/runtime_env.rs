// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which installed ROCm runtime ComfyUI uses, and the process environment that
//! runtime needs.
//!
//! [`select_runtime`] turns an explicit `--runtime-id` or the configured
//! default into the one ready managed Python runtime an install runs against,
//! and explains how to pick one when none or several qualify. The
//! `runtime_environment_*` functions build the `VIRTUAL_ENV`, `ROCM_PATH`,
//! `PATH` and library-path settings a ComfyUI child process runs with, and
//! [`apply_runtime_environment`] puts them on a command.
//!
//! The types these functions return ([`SelectedRuntime`],
//! [`ComfyUiRuntimeEnvironment`]) stay in the parent module with the other
//! ComfyUI types.

use super::{ComfyUiManifest, ComfyUiRuntimeEnvironment, SelectedRuntime};
use crate::runtimes_cmd::runtime_usability_status;
use crate::therock;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, runtime_is_linux, runtime_is_windows, runtime_path_list_join,
    runtime_path_list_split, runtime_paths_equivalent,
};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn select_runtime(
    paths: &AppPaths,
    config: &RocmCliConfig,
    selector: Option<&str>,
) -> Result<SelectedRuntime> {
    let manifests = therock::load_runtime_manifests(paths)?;
    if manifests.is_empty() {
        bail!(
            "Install ROCm first: run `rocm` and pick \"Set up this system\", then install ComfyUI."
        );
    }
    let manifest = match selector.map(str::trim).filter(|value| !value.is_empty()) {
        Some(selector) => select_runtime_by_selector(&manifests, selector)?.clone(),
        None => select_default_runtime(config, &manifests)?.clone(),
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

fn select_default_runtime<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref() {
        return select_runtime_by_selector(manifests, active_key);
    }
    let Some(default_runtime_id) = config.default_runtime_id.as_deref() else {
        return select_single_ready_runtime(manifests);
    };
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

fn select_single_ready_runtime(
    manifests: &[therock::InstalledRuntimeManifest],
) -> Result<&therock::InstalledRuntimeManifest> {
    let ready = manifests
        .iter()
        .filter(|manifest| runtime_usability_status(manifest) == "ready")
        .collect::<Vec<_>>();
    match ready.as_slice() {
        [manifest] => Ok(*manifest),
        [] => {
            let statuses = format_runtime_statuses(manifests);
            bail!(
                "No ROCm runtime is ready. Fix one or install another: run `rocm` and pick \"Set up this system\" (or run `rocm install sdk`), then install ComfyUI. Runtimes found: {statuses}. See `rocm runtimes list`."
            )
        }
        _ => {
            let available = format_available_runtime_keys(ready.iter().copied());
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
        venv_root: infer_python_env_root(&manifest.python_executable),
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
        venv_root: infer_python_env_root(python),
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

fn infer_python_env_root(python: &Path) -> Option<PathBuf> {
    let bin_dir = python.parent()?;
    let name = bin_dir.file_name()?.to_string_lossy().to_ascii_lowercase();
    if matches!(name.as_str(), "scripts" | "bin") {
        return bin_dir.parent().map(Path::to_path_buf);
    }
    None
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
    if runtime_is_linux()
        && let Some(ld_library_path) =
            prepend_env_paths(&env.library_entries, std::env::var_os("LD_LIBRARY_PATH"))?
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
    use rocm_core::{runtime_python_executable_in_env, runtime_rocm_library_filename};
    use std::fs;

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
        let selected = select_default_runtime(&RocmCliConfig::default(), &manifests)?;
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
        let error = select_default_runtime(&RocmCliConfig::default(), &manifests)
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
        let error = select_default_runtime(&RocmCliConfig::default(), &manifests)
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
        let error = select_runtime(&paths, &RocmCliConfig::default(), None)
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
        let config = RocmCliConfig {
            default_runtime_id: Some("no-such-runtime-id".to_owned()),
            ..Default::default()
        };
        let error = select_default_runtime(&config, &manifests)
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
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..Default::default()
        };
        let error = select_default_runtime(&config, &manifests)
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
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..Default::default()
        };
        let error = select_default_runtime(&config, &manifests)
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

        let error = select_runtime(
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
