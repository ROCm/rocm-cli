// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Managed Python for TheRock installs: finding an interpreter that can serve
//! the install (an explicit `ROCM_CLI_PYTHON`, one on `PATH`, or the managed one
//! `uv` provisions), the interpreter the layout requires (`python_requirement`),
//! the wheel tags that interpreter reports, and creating the runtime's virtual
//! environment.
//!
//! Builds on `io`, `process` and `source` (`python_requirement` takes a
//! `SourceLayout`); nothing in those depends on this module. The one item it
//! reads from `therock.rs` is `venv_python_path`, which stays there for good
//! with the other keys and roots helpers.

use super::io::{linux_temp_dir, windows_temp_dir};
use super::process::{
    capture_python_stdout, progress_line, progress_output_suppressed, run_command,
    run_command_with_env,
};
use super::source::SourceLayout;
use super::venv_python_path;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ManagedToolConfig, RocmCliConfig, ensure_uv_binary, managed_tools_dir,
    normalize_runtime_path_for_host, runtime_is_windows, runtime_os_name, runtime_path_list_split,
    unix_time_millis, uv_command_env, uv_venv_args,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_MANAGED_PYTHON_VERSION: &str = "3.12";

/// Interpreter `uv` provisions for a [`SourceLayout::Next`] install. See
/// [`python_requirement`] for why this differs from the canonical default.
const NEXT_MANAGED_PYTHON_VERSION: &str = "3.14";

#[derive(Debug, Clone)]
pub(super) struct WheelCompatibility {
    pub(super) python_tag: String,
    pub(super) platform_tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct PythonLauncher {
    pub(super) executable: PathBuf,
    pub(super) source: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedPythonManifest {
    executable: PathBuf,
    version: String,
    installed_at_unix_ms: u128,
}

pub(super) fn wheel_compatibility_for_python(
    python_executable: &Path,
) -> Result<WheelCompatibility> {
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

pub(super) fn ensure_uv_venv(
    paths: &AppPaths,
    uv: &Path,
    python_launcher: &Path,
    install_root: &Path,
) -> Result<()> {
    let env_python = venv_python_path(install_root);
    if env_python.is_file() {
        let reusable = run_command(
            &env_python,
            &["--version"],
            "verify existing managed TheRock runtime Python",
        )
        .is_ok()
            && wheel_compatibility_for_python(&env_python).is_ok_and(|existing| {
                wheel_compatibility_for_python(python_launcher)
                    .is_ok_and(|required| existing.python_tag == required.python_tag)
            });
        if reusable {
            return Ok(());
        }
        // Same `install_root` can be reused across reinstalls (it's keyed on
        // resolved version/build date, not on the interpreter), so an existing
        // venv left over from a lower-tag requirement must not be mistaken for
        // a compatible one just because it still runs.
    }
    // Anything already in the folder goes through the ownership-checked
    // rebuild, not only a folder whose Python is present: a venv with no
    // Python (an earlier rebuild that failed part-way, a deleted interpreter)
    // handed to a plain `uv venv` is deleted whole by uv 0.7, built over with
    // its stale packages by 0.9, and refused for good by 0.10.
    //
    // The folder itself is never deleted: with `--prefix` it is whatever
    // the user named, and even a runtime ROCm CLI created holds data that
    // is not the venv's (ComfyUI keeps its models, workflows and outputs
    // under `<install_root>/apps/comfyui`). Only the venv's own parts go.
    let rebuilding = match venv_folder_state(paths, install_root)? {
        VenvFolder::Empty => false,
        VenvFolder::Owned => {
            let kept = clear_owned_venv_for_rebuild(paths, install_root)?;
            progress_line(venv_rebuild_line(install_root, env_python.is_file(), &kept));
            true
        }
    };
    let args = uv_venv_command_args(python_launcher, install_root, rebuilding);
    run_command_with_env(
        uv,
        args.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        &uv_venv_command_env(paths),
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

/// The `uv venv` arguments for an install root, shared by the install and its
/// `--dry-run` preview so the preview shows the command that will run.
///
/// Over a folder that already has entries, `--allow-existing` is the only
/// form that is safe on every uv: without it uv 0.7 deletes an existing venv
/// whole and uv 0.10 refuses the folder; `--clear` deletes the whole folder.
/// Only a missing or empty folder gets the plain form.
pub(super) fn uv_venv_command_args(
    python_launcher: &Path,
    install_root: &Path,
    folder_has_entries: bool,
) -> Vec<String> {
    let mut args = uv_venv_args(python_launcher, install_root);
    if folder_has_entries {
        args.push("--allow-existing".to_owned());
    }
    args
}

/// The environment for `uv venv` into an install root.
///
/// `UV_VENV_CLEAR=1` inherited from the user's shell makes uv delete an
/// existing folder before creating the venv, which is the `--clear` this
/// function's caller deliberately does not pass: the folder may be a `--prefix`
/// the user named, holding their own files. Pinning it off keeps that decision
/// here instead of in whatever the shell happened to export.
fn uv_venv_command_env(paths: &AppPaths) -> Vec<(String, String)> {
    let mut env = uv_command_env(paths);
    env.push(("UV_VENV_CLEAR".to_owned(), "0".to_owned()));
    env
}

/// The entries a Python virtual environment is made of, as `uv venv` and
/// `python -m venv` lay one out on Linux (`bin`, `lib`, `lib64`, `include`) and
/// Windows (`Scripts`, `Lib`, `Include`). Clearing these and nothing else is
/// what lets a broken venv be rebuilt without taking the rest of its folder.
///
/// `pyvenv.cfg` is first, so a clear that fails part-way (a file it may not
/// delete, a busy one on Windows) never leaves a folder that still looks like
/// a venv with nothing behind it. Whatever it does leave, the next run sends
/// back through [`venv_folder_state`].
///
/// `share` is deliberately absent: packages install data there, but so can a
/// person, and nothing distinguishes the two. A stale `share` costs nothing,
/// because the reinstall that follows overwrites what it needs.
const VENV_OWNED_ENTRIES: &[&str] = &[
    "pyvenv.cfg",
    "CACHEDIR.TAG",
    "bin",
    "Scripts",
    "lib",
    "lib64",
    "Lib",
    "include",
    "Include",
];

/// The in-tree runtime manifest `save_runtime_manifest` writes beside every
/// install. Only the fields ownership needs: an older manifest that no longer
/// deserializes as a full [`InstalledRuntimeManifest`] still proves who made
/// the folder.
#[derive(Debug, Deserialize)]
struct RuntimeOwnershipMarker {
    runtime_key: String,
    #[serde(default)]
    format: Option<String>,
    install_root: PathBuf,
}

/// What a folder's own contents say about whether ROCm CLI made it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MarkerFinding {
    Missing,
    Unreadable,
    NamesOtherFolder(PathBuf),
    /// A manifest for this folder from a non-wheel install: a tarball
    /// extracts its own `bin`, `lib` and `include` here, which are the SDK,
    /// not a venv, so they are not the venv's to clear.
    OtherFormat(String),
    WheelHere,
}

/// What [`ensure_uv_venv`] may do with an install root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VenvFolder {
    /// Missing or without a single entry: a plain `uv venv` is safe.
    Empty,
    /// Has entries, and ROCm CLI can show it made the folder for a wheel
    /// runtime: its venv files may be cleared and rebuilt in place.
    Owned,
}

/// Decide, without changing anything, whether a venv may be (re)built in
/// `install_root`. Refuses a folder that has entries unless it is somewhere
/// ROCm CLI may delete from AND it can show it made the folder:
///
/// 1. not a protected system location, by its given spelling or by where it
///    really resolves (`rocm_core::runtime_install_root_is_protected`); and
/// 2. either ROCm CLI's in-tree runtime manifest from a wheel install,
///    naming this very folder, or a place inside ROCm CLI's own managed
///    runtimes folder. The second covers an install that failed after
///    creating the venv but before writing its manifest: nobody else puts a
///    venv there.
///
/// `pyvenv.cfg` is not a condition. Anyone's venv has one, so it proves
/// nothing about who made it; and a clear that removed it before failing must
/// still be retried.
///
/// A `--prefix` folder is used verbatim, so without these checks a home
/// directory whose `~/bin/python` fails `--version` was deleted whole.
fn venv_folder_state(paths: &AppPaths, install_root: &Path) -> Result<VenvFolder> {
    if !folder_has_entries(install_root)? {
        return Ok(VenvFolder::Empty);
    }
    refuse_protected_venv_rebuild(install_root)?;
    let marker = runtime_ownership_marker_finding(install_root);
    let in_managed_root = install_root_is_in_managed_wheel_root(paths, install_root);
    if marker == MarkerFinding::WheelHere || in_managed_root {
        return Ok(VenvFolder::Owned);
    }
    bail!(unowned_venv_rebuild_refusal(install_root, &marker))
}

pub(super) fn folder_has_entries(path: &Path) -> Result<bool> {
    match fs::read_dir(path) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

/// Clear the venv files of a folder [`venv_folder_state`] reports as
/// [`VenvFolder::Owned`], keeping everything else. Returns the names kept.
/// Re-checks ownership itself, so no caller can reach the deletes without it.
fn clear_owned_venv_for_rebuild(paths: &AppPaths, install_root: &Path) -> Result<Vec<String>> {
    if venv_folder_state(paths, install_root)? != VenvFolder::Owned {
        return Ok(Vec::new());
    }
    for name in VENV_OWNED_ENTRIES {
        remove_venv_entry(&install_root.join(name))?;
    }
    let mut kept = fs::read_dir(install_root)
        .with_context(|| format!("failed to read {}", install_root.display()))?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("failed to read {}", install_root.display()))?;
    kept.sort();
    Ok(kept)
}

/// The protected-location half of [`venv_folder_state`], checked on the path
/// as given and again on where it really resolves: a link inside the home
/// folder can point into `/usr`, and the lexical check alone answers for the
/// link, not for what a delete would reach.
fn refuse_protected_venv_rebuild(install_root: &Path) -> Result<()> {
    let resolved = install_root.canonicalize().ok();
    let protected = std::iter::once(install_root)
        .chain(resolved.as_deref())
        .find(|candidate| rocm_core::runtime_install_root_is_protected(candidate));
    if let Some(protected) = protected {
        bail!(
            "refusing to rebuild the Python environment at {}: {} is a protected system location. Nothing was changed. Choose an empty folder with --prefix instead, for example `rocm install sdk --prefix <empty-folder>`",
            install_root.display(),
            protected.display()
        );
    }
    Ok(())
}

fn runtime_ownership_marker_finding(install_root: &Path) -> MarkerFinding {
    let path = install_root.join(".rocm-cli-runtime.json");
    if !path.is_file() {
        return MarkerFinding::Missing;
    }
    let Some(marker) = fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<RuntimeOwnershipMarker>(&bytes).ok())
        .filter(|marker| !marker.runtime_key.trim().is_empty())
    else {
        return MarkerFinding::Unreadable;
    };
    let recorded = normalize_runtime_path_for_host(&marker.install_root);
    if !paths_name_same_folder(&recorded, install_root) {
        return MarkerFinding::NamesOtherFolder(recorded);
    }
    match marker.format.as_deref() {
        Some("wheel") => MarkerFinding::WheelHere,
        other => MarkerFinding::OtherFormat(other.unwrap_or("unknown").to_owned()),
    }
}

fn paths_name_same_folder(left: &Path, right: &Path) -> bool {
    if rocm_core::runtime_paths_equivalent(left, right) {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => rocm_core::runtime_paths_equivalent(&left, &right),
        _ => false,
    }
}

/// Strictly inside `<data_dir>/runtimes/wheel`, where every folder is one
/// `install sdk` made for itself.
fn install_root_is_in_managed_wheel_root(paths: &AppPaths, install_root: &Path) -> bool {
    let managed =
        rocm_core::resolve_path_through_symlinks(&paths.data_dir.join("runtimes").join("wheel"));
    let root = rocm_core::resolve_path_through_symlinks(install_root);
    rocm_core::runtime_path_is_same_or_inside(&root, &managed)
        && !rocm_core::runtime_paths_equivalent(&root, &managed)
}

fn unowned_venv_rebuild_refusal(install_root: &Path, marker: &MarkerFinding) -> String {
    let python = venv_python_path(install_root);
    let mut found = vec![if python.is_file() {
        format!(
            "a Python at {} that does not run or does not match",
            python.display()
        )
    } else if install_root.join("pyvenv.cfg").is_file() {
        format!("pyvenv.cfg but no Python at {}", python.display())
    } else {
        "files that are not a Python environment".to_owned()
    }];
    found.push(match marker {
        MarkerFinding::Missing => "no .rocm-cli-runtime.json from a ROCm CLI install".to_owned(),
        MarkerFinding::Unreadable => "a .rocm-cli-runtime.json ROCm CLI cannot read".to_owned(),
        MarkerFinding::NamesOtherFolder(other) => format!(
            "a .rocm-cli-runtime.json that belongs to {}",
            other.display()
        ),
        MarkerFinding::OtherFormat(format) => {
            format!("a .rocm-cli-runtime.json from a {format} install, not a Python environment")
        }
        // Unreachable for a refusal, but stated rather than hidden.
        MarkerFinding::WheelHere => "a .rocm-cli-runtime.json for this folder".to_owned(),
    });
    format!(
        "refusing to create the Python environment at {}: the folder is not empty and ROCm CLI cannot show it created it (found {}), so it will not delete or overwrite what is in it. Nothing was changed. Choose an empty folder, or one ROCm CLI created, with --prefix, for example `rocm install sdk --prefix <empty-folder>`",
        install_root.display(),
        found.join("; ")
    )
}

/// Remove one venv entry without following a link out of the folder: `lib64`
/// is a symlink to `lib` in a uv venv, and a link must go, not its target.
fn remove_venv_entry(path: &Path) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    let result = if metadata.file_type().is_symlink() {
        fs::remove_file(path).or_else(|_| fs::remove_dir(path))
    } else if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    result.with_context(|| format!("failed to remove {}", path.display()))
}

/// The progress line for a rebuild, from the same list the rebuild returned,
/// so what it says was kept is what is still on disk.
fn venv_rebuild_line(install_root: &Path, had_python: bool, kept: &[String]) -> String {
    let kept = if kept.is_empty() {
        "nothing else was in the folder".to_owned()
    } else {
        format!("kept: {}", kept.join(", "))
    };
    let state = if had_python {
        "no longer runs or does not match the required interpreter"
    } else {
        "is incomplete"
    };
    format!(
        "Existing Python environment at {} {state}; rebuilding only its venv files ({kept}).",
        install_root.display()
    )
}

/// The `--dry-run` line for the venv step, from the same decision the
/// install takes, so the preview predicts a refusal instead of showing a
/// command that will not run.
pub(super) fn venv_step_preview(paths: &AppPaths, install_root: &Path) -> String {
    match venv_folder_state(paths, install_root) {
        Ok(VenvFolder::Empty) => "create a new Python environment in an empty folder".to_owned(),
        Ok(VenvFolder::Owned) => "reuse the Python environment ROCm CLI created here if it runs and matches; otherwise rebuild only its venv files and keep everything else in the folder".to_owned(),
        Err(error) => format!("the install will refuse this folder: {error}"),
    }
}

fn python_venv_args(install_root: &Path) -> Vec<String> {
    vec![
        "-m".to_owned(),
        "venv".to_owned(),
        install_root.to_string_lossy().to_string(),
    ]
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

/// The interpreter a runtime's wheels require.
///
/// ROCm >= [`THEROCK_NEXT_MIN_MAJOR`] publishes its framework wheels (notably
/// vLLM, flash-attn and amd-aiter) for one CPython version only, and it is not
/// the one the canonical stream uses. Both fields move together, so they travel
/// together: `version` is what `uv` provisions, `tag` is what an already-present
/// interpreter is checked against.
///
/// [`THEROCK_NEXT_MIN_MAJOR`]: super::source::THEROCK_NEXT_MIN_MAJOR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PythonRequirement {
    tag: &'static str,
    version: &'static str,
}

/// The interpreter required to install from `layout`.
///
/// Keyed on the layout rather than on a resolved ROCm version because the
/// interpreter has to be chosen *before* resolution: `resolve_pip_runtime` picks
/// wheels using the interpreter's own tags, so the resolved version is not
/// available yet. The layout is an honest stand-in only when it was decided
/// from an explicit stable pin (`next_layout_requested`); a fresh install with
/// no pin, or a nightly build-date/prerelease pin, needs
/// [`canonical_auto_select_needs_next_python`] to check whether the channel's
/// own "latest" has rotated past [`THEROCK_NEXT_MIN_MAJOR`] on its own.
///
/// [`canonical_auto_select_needs_next_python`]: super::canonical_auto_select_needs_next_python
/// [`THEROCK_NEXT_MIN_MAJOR`]: super::source::THEROCK_NEXT_MIN_MAJOR
pub(super) const fn python_requirement(layout: SourceLayout) -> PythonRequirement {
    match layout {
        SourceLayout::Next => PythonRequirement {
            tag: "cp314",
            version: NEXT_MANAGED_PYTHON_VERSION,
        },
        SourceLayout::Canonical => PythonRequirement {
            tag: "cp312",
            version: DEFAULT_MANAGED_PYTHON_VERSION,
        },
    }
}

pub(super) fn managed_python_version(requirement: PythonRequirement) -> String {
    std::env::var("ROCM_CLI_MANAGED_PYTHON_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| requirement.version.to_owned())
}

fn ensure_managed_python(
    paths: &AppPaths,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    let version = managed_python_version(requirement);
    progress_line(format!("Preparing Python {version}..."));

    let uv = ensure_uv_binary(paths)?;

    // Check the manifest first — if the recorded executable is still usable, skip the install.
    // ponytail: this manifest holds one version at a time, so alternating between a
    // canonical (3.12) and a `Next` (3.14) install re-runs `uv python install` on every
    // switch instead of keeping both cached — `uv` itself still short-circuits the
    // re-download, so this costs a subprocess round trip, not a real reinstall. Upgrade
    // to a version-keyed manifest if that alternation turns out to be a common pattern.
    if let Ok(Some(manifest)) = load_managed_python_manifest(paths)
        && manifest.version == version
        && manifest.executable.is_file()
        && python_launcher_install_ready(&manifest.executable, requirement).is_ok()
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
    let install_stdio = || {
        if progress_output_suppressed() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        }
    };
    let install_output = Command::new(&uv)
        .args(["python", "install", &version])
        .envs(uv_command_env(paths))
        .stdin(Stdio::null())
        .stdout(install_stdio())
        .stderr(install_stdio())
        .output()
        .context("failed to launch uv python install")?;
    if !install_output.status.success() {
        let status = install_output.status;
        let stderr = String::from_utf8_lossy(&install_output.stderr);
        let stderr = stderr.trim();
        if stderr.is_empty() {
            bail!("uv python install {version} failed with {status}");
        }
        bail!("uv python install {version} failed with {status}: {stderr}");
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

    python_launcher_install_ready(&executable, requirement).with_context(|| {
        format!(
            "Python {version} at {} is not usable for this ROCm install",
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
/// binary from unrelated tests running at the same moment. In test builds, a
/// test can also install one as a thread-local override (see
/// `OverridePythonResolverEnv`) instead of constructing one to pass directly
/// to [`resolve_python_launcher_in`].
#[cfg_attr(test, derive(Clone))]
pub(super) struct PythonResolverEnv {
    /// `ROCM_CLI_PYTHON`: an explicit interpreter that wins over any search.
    pub(super) python_override: Option<String>,
    /// The directories to search for an interpreter, in `PATH` order.
    pub(super) search_dirs: Vec<PathBuf>,
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

// Lets a test steer `resolve_python_launcher` at a fixed `PythonResolverEnv`
// instead of the real process environment, scoped to the calling thread only.
//
// Unlike mutating process-wide `PATH`, a thread-local cannot race a sibling
// test thread that spawns a PATH-resolved binary (e.g. bare-name `tar`)
// during the window a test is steering resolution (see issue #650, where
// that mutation intermittently failed unrelated tests under `cargo test`'s
// one-process-many-threads model).
//
// Because the override is thread-local, it only applies when
// `resolve_python_launcher` runs on the thread that installed it. If the code
// under test ever resolves from a spawned thread (or `thread::scope`), that
// thread silently falls back to the real process environment.
#[cfg(test)]
thread_local! {
    static PYTHON_RESOLVER_ENV_OVERRIDE: std::cell::RefCell<Option<PythonResolverEnv>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard installing a fixed [`PythonResolverEnv`] for [`resolve_python_launcher`]
/// on the calling thread only, restoring the prior override on drop (so nested
/// use composes correctly, mirroring [`SuppressProgressOutput`]; the
/// `nested_python_resolver_env_override_restores_the_outer_one_on_drop` test
/// covers the property this claims).
///
/// [`SuppressProgressOutput`]: super::process::SuppressProgressOutput
#[cfg(test)]
pub(super) struct OverridePythonResolverEnv {
    previous: Option<PythonResolverEnv>,
}

#[cfg(test)]
impl OverridePythonResolverEnv {
    pub(super) fn new(env: PythonResolverEnv) -> Self {
        let previous = PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| cell.replace(Some(env)));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for OverridePythonResolverEnv {
    fn drop(&mut self) {
        let previous = self.previous.take();
        PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| *cell.borrow_mut() = previous);
    }
}

#[cfg(test)]
fn python_resolver_env_override() -> Option<PythonResolverEnv> {
    PYTHON_RESOLVER_ENV_OVERRIDE.with(|cell| cell.borrow().clone())
}

#[cfg(not(test))]
const fn python_resolver_env_override() -> Option<PythonResolverEnv> {
    None
}

pub(super) fn resolve_python_launcher(
    paths: &AppPaths,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    let env = python_resolver_env_override().unwrap_or_else(PythonResolverEnv::from_process_env);
    resolve_python_launcher_in(paths, &env, requirement)
}

fn resolve_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    if let Some(value) = env.python_override.as_deref() {
        python_launcher_install_ready(Path::new(value), requirement)
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLauncher {
            executable: PathBuf::from(value),
            source: "env",
        });
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        match python_launcher_install_ready(&candidate, requirement) {
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
        progress_line(format!(
            "Python from PATH is not usable for this ROCm install ({} is required); using ROCm CLI's managed Python.",
            requirement.tag
        ));
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        if python_launcher_install_ready(&manifest.executable, requirement).is_ok() {
            return Ok(PythonLauncher {
                executable: manifest.executable,
                source: "managed",
            });
        }
        progress_line(format!(
            "Saved managed Python is not usable for this ROCm install ({} is required); preparing Python again.",
            requirement.tag
        ));
    }

    if managed_python_bootstrap_disabled() {
        bail!(
            "unable to locate Python, and managed Python bootstrap is disabled by ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"
        );
    }
    ensure_managed_python(paths, requirement)
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

pub(super) fn split_runtime_path(value: &std::ffi::OsStr) -> Vec<PathBuf> {
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

pub(super) fn python_launcher_install_ready(
    program: &Path,
    requirement: PythonRequirement,
) -> Result<()> {
    let compatibility = wheel_compatibility_for_python(program)?;
    if compatibility.python_tag != requirement.tag {
        bail!(
            "Python wheel tag {} is not supported by this ROCm install; {} is required",
            compatibility.python_tag,
            requirement.tag
        );
    }
    verify_python_can_create_venv(program).context("Python could not create a virtual environment")
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

#[cfg(test)]
mod tests {
    use super::super::managed_runtime_root;
    use super::super::process::PROGRESS_LINE_SINK;
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::test_paths;

    #[cfg(unix)]
    fn write_fake_python_with_venv(dir: &Path, name: &str) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
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
        write_fake_executable(&path, script)?;
        Ok(path)
    }

    #[cfg(unix)]
    fn write_fake_python_reporting(
        dir: &Path,
        name: &str,
        tag: &str,
        version: &str,
    ) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join(name);
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then\n  echo {tag}\n  exit 0\nfi\necho {version}\n"
        );
        write_fake_executable(&path, script)?;
        Ok(path)
    }

    #[cfg(unix)]
    fn write_fake_uv_creating_venv(
        dir: &Path,
        name: &str,
        tag: &str,
        version: &str,
    ) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join(name);
        let script = format!(
            r#"#!/bin/sh
if [ "$1" = "venv" ] && [ "$2" = "--python" ]; then
  /bin/mkdir -p "$4/bin"
  /bin/cat > "$4/bin/python" <<PY
#!/bin/sh
if [ "\$1" = "-c" ]; then
  echo {tag}
  exit 0
fi
echo {version}
PY
  /bin/chmod +x "$4/bin/python"
  exit 0
fi
exit 1
"#
        );
        write_fake_executable(&path, script)?;
        Ok(path)
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

    #[test]
    fn managed_python_defaults_to_312() {
        assert_eq!(DEFAULT_MANAGED_PYTHON_VERSION, "3.12");
    }

    /// ROCm 10.x framework wheels (vLLM, flash-attn, amd-aiter) are published
    /// cp314-only, so the next layout must provision 3.14 while everything else
    /// stays on 3.12.
    #[test]
    fn python_requirement_follows_the_source_layout() {
        let canonical = python_requirement(SourceLayout::Canonical);
        assert_eq!(canonical.tag, "cp312");
        assert_eq!(canonical.version, "3.12");

        let next = python_requirement(SourceLayout::Next);
        assert_eq!(next.tag, "cp314");
        assert_eq!(next.version, "3.14");
    }

    /// A tag mismatch and a broken venv are different failures with different
    /// fixes, so the gate must not report one as the other.
    #[cfg(unix)]
    #[test]
    fn python_gate_rejects_the_wrong_tag_without_blaming_the_venv() -> Result<()> {
        let (root, _paths) = test_paths("python-gate-tag-mismatch");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        // A real, working cp312 interpreter: the venv probe would succeed.
        let python = write_fake_python_with_venv(&bin_dir, "python")?;

        python_launcher_install_ready(&python, python_requirement(SourceLayout::Canonical))
            .expect("a cp312 python satisfies a canonical install");

        let err = python_launcher_install_ready(&python, python_requirement(SourceLayout::Next))
            .expect_err("a cp312 python cannot serve a next-layout install");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("cp314"), "{rendered}");
        assert!(
            !rendered.contains("virtual environment"),
            "a tag mismatch must not be reported as a venv failure: {rendered}"
        );
        Ok(())
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
            python_requirement(SourceLayout::Canonical),
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
            python_requirement(SourceLayout::Canonical),
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

    /// `resolve_python_launcher` must consult an installed
    /// `OverridePythonResolverEnv` instead of the real process environment.
    /// That is what lets a test steer resolution without overwriting
    /// process-wide `PATH`, which raced sibling test threads spawning
    /// PATH-resolved binaries (issue #650).
    #[cfg(unix)]
    #[test]
    fn python_resolver_env_override_is_consulted_by_resolve_python_launcher() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            // No wheel platform tag for this host (e.g. macOS): every python fails
            // the wheel-compatibility check, so resolution always falls through to
            // the managed/uv path regardless of PATH. Nothing to assert here.
            return Ok(());
        }
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let (root, paths) = test_paths("python-resolver-env-override-hermetic");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir)?;
        write_fake_python_with_venv(&bin_dir, "python3")?;

        let launcher = {
            let _guard = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![bin_dir.clone()],
            });
            resolve_python_launcher(&paths, python_requirement(SourceLayout::Canonical))?
        };

        assert_eq!(
            launcher.source, "path",
            "resolve_python_launcher must consult the test override, not the real environment"
        );
        assert!(launcher.executable.starts_with(&bin_dir));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn nested_python_resolver_env_override_restores_the_outer_one_on_drop() {
        // Mirrors the bookkeeping check `suppress_progress_output_contract` (in
        // `therock::process`) makes for `SuppressProgressOutput`: proves the
        // same restore-previous composition for `OverridePythonResolverEnv`,
        // which its doc comment claims but no other test exercises.
        // Thread-local, so this doesn't race with other tests' threads.
        assert!(python_resolver_env_override().is_none());
        {
            let _outer = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: Some("outer".to_owned()),
                search_dirs: vec![],
            });
            assert_eq!(
                python_resolver_env_override()
                    .unwrap()
                    .python_override
                    .as_deref(),
                Some("outer")
            );
            {
                let _inner = OverridePythonResolverEnv::new(PythonResolverEnv {
                    python_override: Some("inner".to_owned()),
                    search_dirs: vec![],
                });
                assert_eq!(
                    python_resolver_env_override()
                        .unwrap()
                        .python_override
                        .as_deref(),
                    Some("inner")
                );
            }
            assert_eq!(
                python_resolver_env_override()
                    .unwrap()
                    .python_override
                    .as_deref(),
                Some("outer"),
                "dropping the inner override must restore the outer one, not clear it"
            );
        }
        assert!(
            python_resolver_env_override().is_none(),
            "dropping the outer override must restore the pre-override state"
        );
    }

    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_recreates_on_interpreter_tag_mismatch() {
        // `install_root` is keyed on resolved version/build date, not on the
        // interpreter, so a reinstall that now requires cp314 must not reuse a
        // stale cp312 venv left over from before this tag check existed.
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-ensure-uv-venv-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();
        let root = root.canonicalize().unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };

        let install_root = root.join("install");
        let stale_bin = install_root.join("bin");
        fs::create_dir_all(&stale_bin).unwrap();
        write_fake_python_reporting(&stale_bin, "python", "cp312", "Python 3.12.10").unwrap();
        // A venv a previous `install sdk` made: rebuilding is only allowed on
        // one ROCm CLI can show it created.
        plant_rocm_owned_venv_markers(&install_root);

        let python_launcher =
            write_fake_python_reporting(&root, "python3.14", "cp314", "Python 3.14.5").unwrap();
        let uv = write_fake_uv_creating_venv(&root, "uv", "cp314", "Python 3.14.5").unwrap();

        ensure_uv_venv(&paths, &uv, &python_launcher, &install_root).unwrap();

        let env_python = venv_python_path(&install_root);
        let compat = wheel_compatibility_for_python(&env_python).unwrap();
        assert_eq!(
            compat.python_tag, "cp314",
            "stale cp312 venv must be recreated, not silently reused"
        );

        fs::remove_dir_all(root).ok();
    }

    // ── ensure_uv_venv: rebuilding a broken venv (#533) ─────────────
    //
    // The oracle is the filesystem: a refusal must leave the tree
    // byte-for-byte as it was, and a rebuild must leave every planted entry
    // that is not part of the venv. Expected outcomes and the paths that must
    // survive are written out literally rather than derived from the code
    // under test, so a change to that code cannot quietly move the goalposts.

    /// A canonical sandbox with `AppPaths` inside it and a sentinel file
    /// outside every install root the test will use.
    #[cfg(unix)]
    fn venv_rebuild_sandbox(label: &str) -> (PathBuf, AppPaths) {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-venv-rebuild-{label}-{}-{}-{}",
            std::process::id(),
            unix_time_millis(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();
        let root = root.canonicalize().unwrap();
        fs::write(root.join("sentinel"), b"outside every install root").unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    #[cfg(unix)]
    fn plant_failing_venv_python(install_root: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let python = venv_python_path(install_root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&python);
        fs::write(&python, "#!/bin/sh\nexit 1\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// What `uv venv` and `save_runtime_manifest` leave in a runtime ROCm CLI
    /// installed: `pyvenv.cfg` plus the in-tree runtime manifest of a wheel
    /// install naming the folder it sits in.
    #[cfg(unix)]
    fn plant_rocm_owned_venv_markers(install_root: &Path) {
        fs::create_dir_all(install_root).unwrap();
        fs::write(install_root.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        write_runtime_marker(install_root, install_root, "wheel");
    }

    #[cfg(unix)]
    fn write_runtime_marker(install_root: &Path, recorded_root: &Path, format: &str) {
        let marker = serde_json::json!({
            "runtime_key": "release-wheel-gfx120x-7-10-0",
            "runtime_id": "therock-release:gfx120X-all",
            "format": format,
            "install_root": recorded_root,
        });
        fs::write(
            install_root.join(".rocm-cli-runtime.json"),
            serde_json::to_vec_pretty(&marker).unwrap(),
        )
        .unwrap();
    }

    /// A fake `uv` that records every invocation and, for `venv`, behaves the
    /// way uv 0.10 does about an existing folder at `$4` (the env root
    /// `uv_venv_args` passes): it refuses one with entries unless
    /// `--allow-existing` is given, and wipes it on `--clear` or an inherited
    /// `UV_VENV_CLEAR=1`. Then it creates a working interpreter there. So a
    /// code path that leans on uv to cope with an existing folder fails here
    /// the way it would against a real uv.
    #[cfg(unix)]
    fn write_logging_fake_uv(dir: &Path) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let log = dir.join("uv-calls.log");
        let path = dir.join("uv");
        let script = format!(
            r#"#!/bin/sh
echo "$@" >> "{log}"
echo "UV_VENV_CLEAR=${{UV_VENV_CLEAR-unset}}" >> "{log}"
if [ "$1" = "venv" ] && [ "$2" = "--python" ]; then
  target="$4"
  allow=0; clear=0
  for arg in "$@"; do
    [ "$arg" = "--allow-existing" ] && allow=1
    [ "$arg" = "--clear" ] && clear=1
  done
  [ "$allow" = 0 ] && [ "${{UV_VENV_CLEAR-}}" = 1 ] && clear=1
  if [ -d "$target" ] && [ -n "$(/bin/ls -A "$target")" ]; then
    if [ "$clear" = 1 ]; then
      /bin/rm -rf "$target"
    elif [ "$allow" = 0 ]; then
      echo "error: A directory already exists at: $target" >&2
      exit 2
    fi
  fi
  /bin/mkdir -p "$target/bin"
  printf '#!/bin/sh\nif [ "$1" = "-c" ]; then echo cp312; exit 0; fi\necho Python 3.12.10\n' > "$target/bin/python"
  /bin/chmod +x "$target/bin/python"
  echo "home = /usr/bin" > "$target/pyvenv.cfg"
  exit 0
fi
exit 1
"#,
            log = log.display()
        );
        fs::write(&path, script).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        (path, log)
    }

    /// Every entry under `root`, with file contents and link targets, without
    /// following links: equality before and after is "nothing was changed".
    #[cfg(unix)]
    fn tree_snapshot(root: &Path) -> Vec<(PathBuf, String)> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(path) = stack.pop() {
            let Ok(meta) = fs::symlink_metadata(&path) else {
                continue;
            };
            let what = if meta.file_type().is_symlink() {
                format!("link -> {}", fs::read_link(&path).unwrap().display())
            } else if meta.is_dir() {
                for entry in fs::read_dir(&path).unwrap() {
                    stack.push(entry.unwrap().path());
                }
                "dir".to_owned()
            } else {
                format!("file {:?}", fs::read(&path).unwrap())
            };
            out.push((path, what));
        }
        out.sort();
        out
    }

    #[cfg(unix)]
    fn cp312_launcher(root: &Path) -> PathBuf {
        write_fake_python_reporting(root, "python3.12", "cp312", "Python 3.12.10").unwrap()
    }

    /// The reported case: a home-like folder whose `~/bin/python` fails
    /// `--version` was deleted whole, then the rebuild failed anyway.
    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_refuses_to_rebuild_a_folder_rocm_cli_did_not_create() {
        let (root, paths) = venv_rebuild_sandbox("unowned");
        let home = root.join("home");
        plant_failing_venv_python(&home);
        fs::create_dir_all(home.join("Documents")).unwrap();
        fs::write(home.join("Documents").join("thesis.txt"), b"years of work").unwrap();
        fs::write(home.join(".bashrc"), b"export PATH=$HOME/bin:$PATH\n").unwrap();
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let launcher = cp312_launcher(&root);
        let before = tree_snapshot(&root);

        let error = ensure_uv_venv(&paths, &uv, &launcher, &home)
            .expect_err("a folder ROCm CLI did not create must not be rebuilt")
            .to_string();

        assert_eq!(
            tree_snapshot(&root),
            before,
            "a refusal must change nothing"
        );
        assert!(!uv_log.exists(), "uv must not run after a refusal");
        assert!(
            error.contains(&format!(
                "refusing to create the Python environment at {}",
                home.display()
            )),
            "{error}"
        );
        assert!(
            error.contains("that does not run or does not match"),
            "{error}"
        );
        assert!(error.contains("no .rocm-cli-runtime.json"), "{error}");
        assert!(error.contains("Nothing was changed"), "{error}");
        assert!(
            error.contains("rocm install sdk --prefix <empty-folder>"),
            "{error}"
        );

        fs::remove_dir_all(root).ok();
    }

    /// The refusal tells the user to choose an empty folder, or one ROCm CLI
    /// created. Both must then actually get past it, or the advice is a dead
    /// end; and taking it must not touch the folder that was refused.
    #[test]
    #[cfg(unix)]
    fn following_the_refusal_advice_gets_past_it() {
        let (root, paths) = venv_rebuild_sandbox("advice");
        let refused = root.join("home");
        plant_failing_venv_python(&refused);
        fs::write(refused.join("notes.txt"), b"mine").unwrap();
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let launcher = cp312_launcher(&root);
        let error = ensure_uv_venv(&paths, &uv, &launcher, &refused)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Choose an empty folder, or one ROCm CLI created"));
        let refused_before = tree_snapshot(&refused);

        let empty = root.join("empty-folder");
        fs::create_dir_all(&empty).unwrap();
        ensure_uv_venv(&paths, &uv, &launcher, &empty)
            .expect("an empty folder is what the refusal recommends");
        assert!(venv_python_path(&empty).is_file());
        // A fresh venv into a folder that exists is where an inherited
        // UV_VENV_CLEAR=1 would make uv delete the folder first.
        assert!(
            fs::read_to_string(&uv_log)
                .unwrap()
                .lines()
                .any(|line| line == "UV_VENV_CLEAR=0"),
            "uv venv must run with UV_VENV_CLEAR pinned off"
        );

        let created = root.join("created-by-rocm-cli");
        plant_failing_venv_python(&created);
        plant_rocm_owned_venv_markers(&created);
        ensure_uv_venv(&paths, &uv, &launcher, &created)
            .expect("a folder ROCm CLI created is what the refusal recommends");
        assert!(venv_python_path(&created).is_file());

        assert_eq!(tree_snapshot(&refused), refused_before);
        assert!(root.join("sentinel").is_file());
        fs::remove_dir_all(root).ok();
    }

    /// ComfyUI keeps the user's models, workflows and outputs under
    /// `<install_root>/apps/comfyui`, so even a runtime ROCm CLI made carries
    /// user data. A rebuild replaces the venv's files and nothing else, and
    /// the line it prints names what it kept — checked against the disk.
    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_rebuilds_an_owned_venv_in_place_and_keeps_comfyui_data() {
        let (root, paths) = venv_rebuild_sandbox("owned");
        let install_root = root.join("prefix");
        plant_failing_venv_python(&install_root);
        plant_rocm_owned_venv_markers(&install_root);
        let site = install_root
            .join("lib")
            .join("python3.12")
            .join("site-packages");
        fs::create_dir_all(&site).unwrap();
        fs::write(site.join("stale.py"), b"old").unwrap();
        std::os::unix::fs::symlink("lib", install_root.join("lib64")).unwrap();
        fs::create_dir_all(install_root.join("include")).unwrap();
        fs::write(install_root.join("CACHEDIR.TAG"), b"Signature").unwrap();
        let models = install_root
            .join("apps")
            .join("comfyui")
            .join("source")
            .join("models");
        fs::create_dir_all(&models).unwrap();
        fs::write(models.join("x.safetensors"), b"weights").unwrap();
        let marker_before = fs::read(install_root.join(".rocm-cli-runtime.json")).unwrap();
        let (uv, uv_log) = write_logging_fake_uv(&root);

        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        ensure_uv_venv(&paths, &uv, &cp312_launcher(&root), &install_root)
            .expect("a venv ROCm CLI created is rebuilt");
        let printed = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());

        assert_eq!(
            fs::read(models.join("x.safetensors")).unwrap(),
            b"weights",
            "ComfyUI data must survive a venv rebuild"
        );
        assert_eq!(
            fs::read(install_root.join(".rocm-cli-runtime.json")).unwrap(),
            marker_before
        );
        assert!(!site.join("stale.py").exists(), "the old venv's files go");
        assert!(fs::symlink_metadata(install_root.join("lib64")).is_err());
        assert!(!install_root.join("include").exists());
        assert!(!install_root.join("CACHEDIR.TAG").exists());
        assert!(
            venv_python_path(&install_root).is_file(),
            "and a new venv is built"
        );
        assert!(root.join("sentinel").is_file());

        let calls = fs::read_to_string(&uv_log).unwrap();
        assert!(
            calls.lines().any(|line| line.starts_with("venv ")
                && line.ends_with(" --allow-existing")
                && !line.contains("--clear")),
            "the rebuild must not let uv clear the folder: {calls}"
        );
        assert!(
            calls
                .lines()
                .all(|line| !line.starts_with("UV_VENV_CLEAR=") || line == "UV_VENV_CLEAR=0"),
            "uv venv must run with UV_VENV_CLEAR pinned off: {calls}"
        );

        // The printed line and the disk agree on what was kept.
        let line = printed
            .iter()
            .find(|line| line.contains("rebuilding only its venv files"))
            .unwrap_or_else(|| panic!("no rebuild line in {printed:?}"));
        assert!(
            line.contains("kept: .rocm-cli-runtime.json, apps"),
            "{line}"
        );
        for kept in [".rocm-cli-runtime.json", "apps"] {
            assert!(install_root.join(kept).exists(), "{kept} was reported kept");
        }

        fs::remove_dir_all(root).ok();
    }

    /// A rebuild that fails part-way must not leave a folder the next run
    /// treats as fresh. Here a read-only directory inside `lib` stops the
    /// clear after `pyvenv.cfg` and `bin` are gone; the next run, with the
    /// directory deletable again, has no Python to test, and must still go
    /// through the ownership-checked rebuild and `--allow-existing` rather
    /// than a plain `uv venv`, which uv 0.7 answers by deleting the folder.
    #[test]
    #[cfg(unix)]
    fn a_rebuild_that_fails_part_way_is_finished_by_the_next_run_keeping_comfyui_data() {
        let (root, paths) = venv_rebuild_sandbox("partial");
        let install_root = root.join("prefix");
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let launcher = cp312_launcher(&root);
        let models = partial_clear_fixture(&install_root);

        let first = ensure_uv_venv(&paths, &uv, &launcher, &install_root);
        let locked = install_root.join("lib").join("locked");
        restore_partial_clear_fixture(&install_root, first.is_ok());
        assert_eq!(
            fs::read(models.join("x.safetensors")).unwrap(),
            b"weights",
            "the failed first run must not touch ComfyUI data"
        );
        assert!(!install_root.join("pyvenv.cfg").exists());
        assert!(!venv_python_path(&install_root).exists());
        fs::remove_file(&uv_log).ok();

        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        ensure_uv_venv(&paths, &uv, &launcher, &install_root)
            .expect("the next run finishes the rebuild");
        let printed = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());

        assert_eq!(fs::read(models.join("x.safetensors")).unwrap(), b"weights");
        assert!(!locked.exists(), "the rest of the old venv is cleared");
        assert!(venv_python_path(&install_root).is_file());
        let calls = fs::read_to_string(&uv_log).unwrap();
        assert!(
            calls
                .lines()
                .any(|line| line.starts_with("venv ") && line.ends_with(" --allow-existing")),
            "the second run must not hand uv a plain `uv venv`: {calls}"
        );
        assert!(
            printed
                .iter()
                .any(|line| line.contains("is incomplete; rebuilding only its venv files")),
            "{printed:?}"
        );
        fs::remove_dir_all(root).ok();
    }

    /// An owned venv with a broken Python, ComfyUI data, and a read-only
    /// directory inside `lib` that stops a clear part-way.
    #[cfg(unix)]
    fn partial_clear_fixture(install_root: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        plant_failing_venv_python(install_root);
        plant_rocm_owned_venv_markers(install_root);
        let models = install_root
            .join("apps")
            .join("comfyui")
            .join("source")
            .join("models");
        fs::create_dir_all(&models).unwrap();
        fs::write(models.join("x.safetensors"), b"weights").unwrap();
        let locked = install_root.join("lib").join("locked");
        fs::create_dir_all(&locked).unwrap();
        fs::write(locked.join("held.py"), b"held").unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o555)).unwrap();
        models
    }

    /// Make the read-only directory deletable again. A process that may
    /// delete it anyway (root) cleared everything on the first run, so the
    /// state a failed clear leaves is built by hand instead.
    #[cfg(unix)]
    fn restore_partial_clear_fixture(install_root: &Path, first_run_succeeded: bool) {
        use std::os::unix::fs::PermissionsExt;
        let locked = install_root.join("lib").join("locked");
        if first_run_succeeded {
            for name in ["bin", "pyvenv.cfg"] {
                let _ = remove_venv_entry(&install_root.join(name));
            }
            fs::create_dir_all(&locked).unwrap();
            fs::write(locked.join("held.py"), b"held").unwrap();
        } else {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    /// The same partial-clear sequence against real uv releases, which differ
    /// in exactly the way that matters here: given a plain `uv venv` over a
    /// leftover venv, 0.7 deletes the folder, 0.9 builds over it, and 0.10
    /// refuses. Opt-in, since CI has no pinned uv set:
    /// `ROCM_CLI_TEST_UV_BINARIES=/path/uv-0.7:/path/uv-0.10` plus
    /// `ROCM_CLI_TEST_PYTHON=/usr/bin/python3`.
    #[test]
    #[cfg(unix)]
    fn a_partial_clear_is_finished_safely_by_every_listed_real_uv() {
        let (Some(binaries), Some(python)) = (
            std::env::var_os("ROCM_CLI_TEST_UV_BINARIES"),
            std::env::var_os("ROCM_CLI_TEST_PYTHON"),
        ) else {
            eprintln!("skipped: set ROCM_CLI_TEST_UV_BINARIES and ROCM_CLI_TEST_PYTHON");
            return;
        };
        let python = PathBuf::from(python);
        for real_uv in std::env::split_paths(&binaries) {
            let (root, paths) = venv_rebuild_sandbox("real-uv");
            let log = root.join("uv-calls.log");
            let uv = root.join("uv");
            fs::write(
                &uv,
                format!(
                    "#!/bin/sh\necho \"$@\" >> \"{}\"\nexec \"{}\" \"$@\"\n",
                    log.display(),
                    real_uv.display()
                ),
            )
            .unwrap();
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();
            }
            let install_root = root.join("prefix");
            ensure_uv_venv(&paths, &uv, &python, &install_root)
                .unwrap_or_else(|e| panic!("{}: initial venv: {e:#}", real_uv.display()));
            let models = partial_clear_fixture(&install_root);

            let first = ensure_uv_venv(&paths, &uv, &python, &install_root);
            restore_partial_clear_fixture(&install_root, first.is_ok());
            fs::remove_file(&log).ok();
            let second = ensure_uv_venv(&paths, &uv, &python, &install_root);

            let label = real_uv.display();
            assert!(second.is_ok(), "{label}: {second:?}");
            assert_eq!(
                fs::read(models.join("x.safetensors")).ok().as_deref(),
                Some(&b"weights"[..]),
                "{label}: ComfyUI data lost"
            );
            assert!(
                run_command(&venv_python_path(&install_root), &["--version"], "probe").is_ok(),
                "{label}: the rebuilt venv does not run"
            );
            let calls = fs::read_to_string(&log).unwrap();
            assert!(
                calls
                    .lines()
                    .any(|line| line.starts_with("venv ") && line.ends_with(" --allow-existing")),
                "{label}: {calls}"
            );
            fs::remove_dir_all(root).ok();
        }
    }

    /// A venv entry that is a link is unlinked, never followed: `lib64` here
    /// points out of the folder, and what it points at is not the venv's.
    #[test]
    #[cfg(unix)]
    fn venv_rebuild_unlinks_a_linked_venv_entry_without_following_it() {
        let (root, paths) = venv_rebuild_sandbox("link");
        let install_root = root.join("prefix");
        plant_failing_venv_python(&install_root);
        plant_rocm_owned_venv_markers(&install_root);
        let outside = root.join("outside-the-prefix");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("keep.txt"), b"not the venv's").unwrap();
        std::os::unix::fs::symlink(&outside, install_root.join("lib64")).unwrap();
        let (uv, _) = write_logging_fake_uv(&root);

        ensure_uv_venv(&paths, &uv, &cp312_launcher(&root), &install_root).unwrap();

        assert!(fs::symlink_metadata(install_root.join("lib64")).is_err());
        assert_eq!(
            fs::read(outside.join("keep.txt")).unwrap(),
            b"not the venv's"
        );
        fs::remove_dir_all(root).ok();
    }

    /// A marker copied from another runtime does not make this folder ROCm
    /// CLI's: it must name the folder it sits in.
    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_refuses_a_runtime_marker_that_names_another_folder() {
        let (root, paths) = venv_rebuild_sandbox("other-marker");
        let install_root = root.join("prefix");
        plant_failing_venv_python(&install_root);
        fs::write(install_root.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        write_runtime_marker(&install_root, &root.join("elsewhere"), "wheel");
        fs::write(install_root.join("work.txt"), b"mine").unwrap();
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let before = tree_snapshot(&install_root);

        let error = ensure_uv_venv(&paths, &uv, &cp312_launcher(&root), &install_root)
            .unwrap_err()
            .to_string();

        assert_eq!(tree_snapshot(&install_root), before);
        assert!(!uv_log.exists());
        assert!(error.contains("belongs to"), "{error}");
        assert!(
            error.contains(&root.join("elsewhere").display().to_string()),
            "{error}"
        );
        fs::remove_dir_all(root).ok();
    }

    /// The managed runtimes folder is ROCm CLI's own, so a venv there without
    /// a manifest yet (an install that failed after `uv venv`) is still
    /// rebuilt rather than stranded.
    #[test]
    #[cfg(unix)]
    fn ensure_uv_venv_rebuilds_a_manifestless_venv_in_the_managed_runtimes_folder() {
        let (root, paths) = venv_rebuild_sandbox("managed");
        let install_root = managed_runtime_root(&paths, "wheel", "release-wheel-x");
        plant_failing_venv_python(&install_root);
        fs::write(install_root.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        let (uv, _) = write_logging_fake_uv(&root);

        ensure_uv_venv(&paths, &uv, &cp312_launcher(&root), &install_root)
            .expect("a venv in the managed runtimes folder is ROCm CLI's");
        assert!(venv_python_path(&install_root).is_file());

        // The managed folder itself is not a runtime and is never rebuilt.
        let wheel_root = paths.data_dir.join("runtimes").join("wheel");
        assert!(!install_root_is_in_managed_wheel_root(&paths, &wheel_root));
        fs::remove_dir_all(root).ok();
    }

    /// The protected-location check runs before ownership is looked at, on
    /// the path as given and on where it resolves. `/proc/self/cwd` is
    /// protected only by its spelling (it resolves to this test's working
    /// directory); a link to `/usr/share` only by where it resolves. Only
    /// `venv_folder_state` is called, which never deletes.
    #[test]
    #[cfg(target_os = "linux")]
    fn venv_rebuild_refuses_a_protected_location_by_spelling_and_by_resolution() {
        let (root, paths) = venv_rebuild_sandbox("protected");

        let spelled = Path::new("/proc/self/cwd");
        let error = venv_folder_state(&paths, spelled).unwrap_err().to_string();
        assert!(
            error.contains(": /proc/self/cwd is a protected system location"),
            "{error}"
        );

        let link = root.join("looks-like-mine");
        std::os::unix::fs::symlink("/usr/share", &link).unwrap();
        let error = venv_folder_state(&paths, &link).unwrap_err().to_string();
        assert!(
            error.contains(": /usr/share is a protected system location"),
            "{error}"
        );
        assert!(error.contains("Nothing was changed"), "{error}");

        fs::remove_dir_all(root).ok();
    }

    /// The `--dry-run` preview takes the install's own decision: it names the
    /// refusal for a folder the install would refuse, and shows
    /// `--allow-existing` for one it would rebuild.
    #[test]
    #[cfg(unix)]
    fn the_dry_run_preview_predicts_what_the_venv_step_will_do() {
        let (root, paths) = venv_rebuild_sandbox("preview");
        let launcher = root.join("python3");
        let empty = root.join("missing");
        assert_eq!(
            venv_step_preview(&paths, &empty),
            "create a new Python environment in an empty folder"
        );
        assert!(
            !uv_venv_command_args(&launcher, &empty, false)
                .contains(&"--allow-existing".to_owned())
        );

        let home = root.join("home");
        plant_failing_venv_python(&home);
        let preview = venv_step_preview(&paths, &home);
        assert!(
            preview.starts_with("the install will refuse this folder: refusing to create"),
            "{preview}"
        );
        assert!(home.join("bin").exists(), "a preview changes nothing");

        let owned = root.join("owned");
        plant_failing_venv_python(&owned);
        plant_rocm_owned_venv_markers(&owned);
        assert!(
            venv_step_preview(&paths, &owned).starts_with("reuse the Python environment"),
            "{}",
            venv_step_preview(&paths, &owned)
        );
        assert!(
            venv_python_path(&owned).is_file(),
            "a preview changes nothing"
        );
        assert!(
            uv_venv_command_args(&launcher, &owned, true)
                .ends_with(&["--allow-existing".to_owned()])
        );
        fs::remove_dir_all(root).ok();
    }

    /// Every shape a folder can have when `ensure_uv_venv` meets it with
    /// entries in it: what is left of a venv, what its runtime marker says,
    /// and whether it sits in the managed runtimes folder. The expected
    /// outcome of each is written out, not computed, and every case also holds
    /// user data at paths listed literally here; a rebuild must keep all of
    /// them and a refusal must change nothing.
    #[test]
    #[cfg(unix)]
    fn venv_rebuild_deletes_only_venv_files_and_only_from_an_owned_folder() {
        #[derive(Debug, Clone, Copy)]
        enum Venv {
            /// `pyvenv.cfg`, `lib`, and a Python that fails `--version`.
            BrokenPython,
            /// `pyvenv.cfg` and `lib`, no Python.
            NoPython,
            /// Only `lib`: a clear that removed `pyvenv.cfg` and `bin` and
            /// then failed.
            Leftover,
        }
        #[derive(Debug, Clone, Copy)]
        enum Marker {
            Missing,
            Garbage,
            OtherFolder,
            TarballHere,
            WheelHere,
        }
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Expect {
            Rebuilt,
            Refused,
        }
        use Expect::{Rebuilt, Refused};
        use Marker::{Garbage, Missing, OtherFolder, TarballHere, WheelHere};
        use Venv::{BrokenPython, Leftover, NoPython};
        #[rustfmt::skip]
        let table = [
            // venv shape   marker       managed  expected
            (BrokenPython, Missing,     false,   Refused),
            (BrokenPython, Garbage,     false,   Refused),
            (BrokenPython, OtherFolder, false,   Refused),
            (BrokenPython, TarballHere, false,   Refused),
            (BrokenPython, WheelHere,   false,   Rebuilt),
            (BrokenPython, Missing,     true,    Rebuilt),
            (BrokenPython, Garbage,     true,    Rebuilt),
            (BrokenPython, OtherFolder, true,    Rebuilt),
            (BrokenPython, TarballHere, true,    Rebuilt),
            (BrokenPython, WheelHere,   true,    Rebuilt),
            (NoPython,     Missing,     false,   Refused),
            (NoPython,     Garbage,     false,   Refused),
            (NoPython,     OtherFolder, false,   Refused),
            (NoPython,     TarballHere, false,   Refused),
            (NoPython,     WheelHere,   false,   Rebuilt),
            (NoPython,     Missing,     true,    Rebuilt),
            (NoPython,     Garbage,     true,    Rebuilt),
            (NoPython,     OtherFolder, true,    Rebuilt),
            (NoPython,     TarballHere, true,    Rebuilt),
            (NoPython,     WheelHere,   true,    Rebuilt),
            (Leftover,     Missing,     false,   Refused),
            (Leftover,     Garbage,     false,   Refused),
            (Leftover,     OtherFolder, false,   Refused),
            (Leftover,     TarballHere, false,   Refused),
            (Leftover,     WheelHere,   false,   Rebuilt),
            (Leftover,     Missing,     true,    Rebuilt),
            (Leftover,     Garbage,     true,    Rebuilt),
            (Leftover,     OtherFolder, true,    Rebuilt),
            (Leftover,     TarballHere, true,    Rebuilt),
            (Leftover,     WheelHere,   true,    Rebuilt),
        ];
        // Never part of a venv, so a rebuild must leave each exactly as it was.
        const USER_PATHS: [&str; 4] = [
            "Documents/thesis.txt",
            "apps/comfyui/source/models/x",
            "share/notes/todo.txt",
            "work.py",
        ];
        for (venv, marker, managed, expected) in table {
            let shape = format!("{venv:?} {marker:?} managed={managed}");
            let (root, paths) = venv_rebuild_sandbox("shapes");
            let install_root = if managed {
                managed_runtime_root(&paths, "wheel", "release-wheel-k")
            } else {
                root.join("prefix")
            };
            fs::create_dir_all(install_root.join("lib").join("python3.12")).unwrap();
            fs::write(
                install_root.join("lib").join("python3.12").join("stale.py"),
                b"old",
            )
            .unwrap();
            match venv {
                BrokenPython => {
                    plant_failing_venv_python(&install_root);
                    fs::write(install_root.join("pyvenv.cfg"), "home = /x\n").unwrap();
                }
                NoPython => fs::write(install_root.join("pyvenv.cfg"), "home = /x\n").unwrap(),
                Leftover => {}
            }
            match marker {
                Missing => {}
                Garbage => fs::write(install_root.join(".rocm-cli-runtime.json"), "{}").unwrap(),
                OtherFolder => write_runtime_marker(&install_root, &root.join("other"), "wheel"),
                TarballHere => write_runtime_marker(&install_root, &install_root, "tarball"),
                WheelHere => write_runtime_marker(&install_root, &install_root, "wheel"),
            }
            for user_path in USER_PATHS {
                let path = install_root.join(user_path);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, format!("user data: {user_path}")).unwrap();
            }
            let (uv, uv_log) = write_logging_fake_uv(&root);
            let launcher = cp312_launcher(&root);
            let before = tree_snapshot(&install_root);

            let result = ensure_uv_venv(&paths, &uv, &launcher, &install_root);

            assert!(root.join("sentinel").is_file(), "{shape}");
            match expected {
                Rebuilt => {
                    assert!(result.is_ok(), "{shape}: {result:?}");
                    for user_path in USER_PATHS {
                        assert_eq!(
                            fs::read_to_string(install_root.join(user_path)).ok(),
                            Some(format!("user data: {user_path}")),
                            "{shape}: lost {user_path}"
                        );
                    }
                    assert!(
                        !install_root
                            .join("lib")
                            .join("python3.12")
                            .join("stale.py")
                            .exists(),
                        "{shape}: the old venv was not cleared"
                    );
                    assert!(venv_python_path(&install_root).is_file(), "{shape}");
                }
                Refused => {
                    assert!(result.is_err(), "{shape}: rebuilt a folder it does not own");
                    assert_eq!(
                        tree_snapshot(&install_root),
                        before,
                        "{shape}: a refusal changed the folder"
                    );
                    assert!(!uv_log.exists(), "{shape}");
                }
            }
            fs::remove_dir_all(root).ok();
        }
    }
}
