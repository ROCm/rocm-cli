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
use crate::shell_quote::shell_quote;
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

/// Where an interpreter came from. `as_str` is what the install plan prints on
/// its `python_source:` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PythonSource {
    /// `ROCM_CLI_PYTHON`.
    Env,
    /// Found on `PATH`.
    Path,
    /// The managed Python ROCm CLI prepares.
    Managed,
    /// The interpreter an installed runtime's manifest recorded.
    Manifest,
}

impl PythonSource {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Path => "path",
            Self::Managed => "managed",
            Self::Manifest => "manifest",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PythonLauncher {
    pub(super) executable: PathBuf,
    pub(super) source: PythonSource,
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
    // Same `install_root` can be reused across reinstalls (it's keyed on
    // resolved version/build date, not on the interpreter), so an existing
    // venv left over from a lower-tag requirement must not be mistaken for a
    // compatible one just because it still runs.
    let Err(not_reusable) = existing_venv_reuse(install_root, python_launcher) else {
        return Ok(());
    };
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
    let rebuilding = match venv_folder_state(paths, install_root, &not_reusable)? {
        VenvFolder::Empty => false,
        VenvFolder::Owned => {
            let kept = clear_owned_venv_for_rebuild(paths, install_root, &not_reusable)?;
            progress_line(venv_rebuild_line(install_root, &not_reusable, &kept));
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

/// Whether the venv already in `install_root` can be used as it is: `Ok` when
/// it can, otherwise what was found instead, worded to follow "found" in the
/// refusal and to name the condition that failed, so the user can act on it.
///
/// A Python that runs and reports the right wheel tag is not enough. On Linux
/// `bin/python` is a symlink to the base interpreter, and without `pyvenv.cfg`
/// beside it that is exactly what it runs as: same tag, but `sys.prefix` is the
/// base install, so `uv pip install --python <root>/bin/python` would write
/// ROCm and torch into the user's own Python (a pyenv or conda one is
/// writable) and record the runtime as working. So reuse also needs
/// `pyvenv.cfg`, and the interpreter's own `sys.prefix` to be this folder.
///
/// This runs the folder's Python, so only the install calls it; a `--dry-run`
/// preview reads the folder instead ([`dry_run_venv_plan`]).
fn existing_venv_reuse(install_root: &Path, python_launcher: &Path) -> Result<(), String> {
    if let Some(finding) = venv_shape_finding(install_root) {
        return Err(finding);
    }
    let env_python = venv_python_path(install_root);
    if run_command(
        &env_python,
        &["--version"],
        "verify existing managed TheRock runtime Python",
    )
    .is_err()
    {
        return Err(format!(
            "a Python at {} that does not run",
            env_python.display()
        ));
    }
    let prefix = capture_python_stdout(
        &env_python,
        "import sys; print(sys.prefix)",
        "inspect existing managed TheRock runtime prefix",
    )
    .map_err(|_| {
        format!(
            "a Python at {} that does not report its sys.prefix",
            env_python.display()
        )
    })?;
    let prefix = prefix.trim();
    if !paths_name_same_folder(Path::new(prefix), install_root) {
        return Err(format!(
            "a Python at {} whose sys.prefix is {prefix}, not this folder",
            env_python.display()
        ));
    }
    let existing = wheel_compatibility_for_python(&env_python).map_err(|_| {
        format!(
            "a Python at {} whose wheel tag could not be read",
            env_python.display()
        )
    })?;
    let required = wheel_compatibility_for_python(python_launcher).map_err(|_| {
        format!(
            "a Python at {} that could not be compared with {}, whose wheel tag could not be read",
            env_python.display(),
            python_launcher.display()
        )
    })?;
    if existing.python_tag != required.python_tag {
        return Err(format!(
            "a Python at {} that reports wheel tag {} where {} is required",
            env_python.display(),
            existing.python_tag,
            required.python_tag
        ));
    }
    Ok(())
}

/// What the folder's files alone say against reusing its venv, or `None` when
/// it has both a Python and `pyvenv.cfg`, the shape a reusable venv has. Reads
/// the folder and runs nothing, so a `--dry-run` preview can ask it too: a
/// `Some` here is certain, while `None` leaves the answer to running the
/// Python, which only the install does.
fn venv_shape_finding(install_root: &Path) -> Option<String> {
    let python = venv_python_path(install_root);
    let has_cfg = install_root.join("pyvenv.cfg").is_file();
    match (python.is_file(), has_cfg) {
        (true, true) => None,
        (true, false) => Some(format!(
            "a Python at {} with no pyvenv.cfg beside it",
            python.display()
        )),
        (false, true) => Some(format!("pyvenv.cfg but no Python at {}", python.display())),
        (false, false) => Some("files that are not a Python environment".to_owned()),
    }
}

/// The `uv venv` arguments for an install root, shared by the install and its
/// `--dry-run` preview so the preview shows the command that will run.
///
/// Over a folder that already has entries, `--allow-existing` is the only
/// form that is safe on every uv: without it uv 0.7 deletes an existing venv
/// whole and uv 0.10 refuses the folder; `--clear` deletes the whole folder.
/// Only a missing or empty folder gets the plain form.
fn uv_venv_command_args(
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
/// The interpreter goes first and `pyvenv.cfg` last. A clear that stops
/// part-way (a file it may not delete, a busy one on Windows) then leaves no
/// Python behind, or a Python whose venv is still marked as one; either way the
/// next run cannot mistake it for a working venv
/// ([`existing_venv_is_reusable`]) and sends the folder back through
/// [`venv_folder_state`], which finishes the clear.
///
/// `share` is deliberately absent: packages install data there, but so can a
/// person, and nothing distinguishes the two. A stale `share` costs nothing,
/// because the reinstall that follows overwrites what it needs.
const VENV_OWNED_ENTRIES: &[&str] = &[
    "bin",
    "Scripts",
    "lib",
    "lib64",
    "Lib",
    "include",
    "Include",
    "CACHEDIR.TAG",
    "pyvenv.cfg",
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
///
/// `python_finding` is what was found instead of a reusable venv (see
/// [`existing_venv_reuse`]); a refusal names it. Reads the folder and runs
/// nothing.
fn venv_folder_state(
    paths: &AppPaths,
    install_root: &Path,
    python_finding: &str,
) -> Result<VenvFolder> {
    if !folder_has_entries(install_root)? {
        return Ok(VenvFolder::Empty);
    }
    refuse_protected_venv_rebuild(install_root)?;
    let marker = runtime_ownership_marker_finding(install_root);
    let in_managed_root = install_root_is_in_managed_wheel_root(paths, install_root);
    if marker == MarkerFinding::WheelHere || in_managed_root {
        return Ok(VenvFolder::Owned);
    }
    bail!(unowned_venv_rebuild_refusal(
        install_root,
        python_finding,
        &marker
    ))
}

fn folder_has_entries(path: &Path) -> Result<bool> {
    match fs::read_dir(path) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

/// Clear the venv files of a folder [`venv_folder_state`] reports as
/// [`VenvFolder::Owned`], keeping everything else. Returns the names kept.
/// Re-checks ownership itself, so no caller can reach the deletes without it.
fn clear_owned_venv_for_rebuild(
    paths: &AppPaths,
    install_root: &Path,
    python_finding: &str,
) -> Result<Vec<String>> {
    if venv_folder_state(paths, install_root, python_finding)? != VenvFolder::Owned {
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

fn unowned_venv_rebuild_refusal(
    install_root: &Path,
    python_finding: &str,
    marker: &MarkerFinding,
) -> String {
    let mut found = vec![python_finding.to_owned()];
    found.push(match marker {
        MarkerFinding::Missing => {
            "no finished ROCm CLI install recorded here (no .rocm-cli-runtime.json)".to_owned()
        }
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
        "refusing to create the Python environment at {}: the folder is not empty and ROCm CLI cannot show it holds a ROCm CLI install (found {}), so it will not delete or overwrite what is in it. Nothing was changed. Choose an empty folder, or one where a ROCm CLI install finished, with --prefix, for example `rocm install sdk --prefix <empty-folder>`",
        install_root.display(),
        found.join("; ")
    )
}

/// Remove one venv entry without following a link out of the folder: `lib64`
/// is a symlink to `lib` in a uv venv, and a link must go, not its target.
///
/// Only an entry that is not there counts as removed. Any other failure to
/// look at it (permissions, I/O) is an error, so a clear that could not see an
/// entry stops instead of handing `uv venv --allow-existing` a stale one.
fn remove_venv_entry(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
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
/// `python_finding` is the condition that stopped reuse, as
/// [`existing_venv_reuse`] reported it before the clear.
fn venv_rebuild_line(install_root: &Path, python_finding: &str, kept: &[String]) -> String {
    let kept = if kept.is_empty() {
        "nothing else was in the folder".to_owned()
    } else {
        format!("kept: {}", kept.join(", "))
    };
    format!(
        "Existing Python environment at {} cannot be used as it is (found {python_finding}); rebuilding only its venv files ({kept}).",
        install_root.display()
    )
}

/// What a `--dry-run` preview prints about the venv step.
#[derive(Debug)]
pub(super) struct DryRunVenvPlan {
    /// The `python_env:` line's value.
    pub(super) python_env: String,
    /// The `key: value` lines printed after `mode: dry-run`, `command` first.
    pub(super) lines: Vec<(&'static str, String)>,
}

/// The `--dry-run` lines for the venv step and the install command.
///
/// A preview runs nothing from the install folder. `install_sdk_dry_run` is a
/// model-callable tool that runs without approval and takes a model-chosen
/// `--prefix`, so a folder holding a hostile `bin/python` (a downloaded or
/// cloned tree) must not be executed by it. So this reads the folder and stops
/// there: [`venv_shape_finding`] and [`venv_folder_state`] only look at files.
///
/// What only running the folder's Python could tell, whether a venv-shaped
/// folder is reused, it says it could not tell, and prints what happens either
/// way: `command` for reuse, and `command_if_rebuilt` or
/// `refusal_if_not_reused` for the other outcome. Where the files decide alone
/// it prints one outcome: a fresh venv, a rebuild in place (`--allow-existing`,
/// after removing the venv's own entries, which the command does not show), or
/// a refusal, which gets no command at all since nothing would run.
///
/// The commands carry `UV_VENV_CLEAR=0`, which the install pins for `uv venv`,
/// and every argument is shell-quoted: the install root comes from a
/// `--prefix` the assistant can choose, and the line is one a user is invited
/// to paste.
pub(super) fn dry_run_venv_plan(
    paths: &AppPaths,
    venv_python_arg: &Path,
    install_root: &Path,
    required_tag: &str,
    install_args: &[String],
) -> DryRunVenvPlan {
    let display = |args: &[String]| {
        args.iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let install = format!("uv {}", display(install_args));
    let with_venv = |rebuilding: bool| {
        format!(
            "UV_VENV_CLEAR=0 uv {} && uv {}",
            display(&uv_venv_command_args(
                venv_python_arg,
                install_root,
                rebuilding
            )),
            display(install_args)
        )
    };
    let cleared = VENV_OWNED_ENTRIES.join(", ");
    let shape = venv_shape_finding(install_root);
    let finding = shape.clone().unwrap_or_else(|| {
        format!(
            "a Python at {} that this preview did not run",
            venv_python_path(install_root).display()
        )
    });
    let reuse = format!(
        "reuse the Python environment here if its Python runs, reports wheel tag {required_tag} and has this folder as its sys.prefix; this preview runs nothing from the folder, so it cannot tell whether it does"
    );
    let rebuild = format!(
        "rebuild only the venv files ROCm CLI created here: remove {cleared} first, which the command does not show, then create the venv with --allow-existing; everything else in the folder is kept"
    );
    match (shape, venv_folder_state(paths, install_root, &finding)) {
        (_, Ok(VenvFolder::Empty)) => DryRunVenvPlan {
            python_env: "create a new Python environment in an empty folder".to_owned(),
            lines: vec![("command", with_venv(false))],
        },
        (Some(found), Ok(VenvFolder::Owned)) => DryRunVenvPlan {
            python_env: format!("{rebuild} (found {found})"),
            lines: vec![("command", with_venv(true))],
        },
        (None, Ok(VenvFolder::Owned)) => DryRunVenvPlan {
            python_env: format!("{reuse}; if not, {rebuild}"),
            lines: vec![
                ("command", install),
                ("command_if_rebuilt", with_venv(true)),
            ],
        },
        (Some(_), Err(refusal)) => DryRunVenvPlan {
            python_env: "the install will refuse this folder".to_owned(),
            lines: vec![
                (
                    "command",
                    "none; the install will refuse this folder".to_owned(),
                ),
                ("refusal", format!("{refusal:#}")),
            ],
        },
        (None, Err(refusal)) => DryRunVenvPlan {
            python_env: format!("{reuse}; if not, the install will refuse this folder"),
            lines: vec![
                ("command", install),
                ("refusal_if_not_reused", format!("{refusal:#}")),
            ],
        },
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

fn managed_python_version_override() -> Option<String> {
    std::env::var("ROCM_CLI_MANAGED_PYTHON_VERSION").ok()
}

/// The managed Python version a bootstrap prepares under the process
/// environment. A decision made from an injected [`PythonResolverEnv`] asks
/// [`PythonResolverEnv::managed_python_version`] instead.
pub(super) fn managed_python_version(requirement: PythonRequirement) -> String {
    PythonResolverEnv::from_process_env().managed_python_version(requirement)
}

fn ensure_managed_python(
    paths: &AppPaths,
    requirement: PythonRequirement,
    version: String,
) -> Result<PythonLauncher> {
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
            source: PythonSource::Managed,
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
        source: PythonSource::Managed,
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
    /// `ROCM_CLI_MANAGED_PYTHON_VERSION`: the version a managed bootstrap
    /// prepares in place of the layout's own, when set and not blank.
    pub(super) managed_python_version: Option<String>,
    /// `ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP`: whether a host with no usable
    /// interpreter is refused instead of bootstrapped.
    pub(super) managed_bootstrap_disabled: bool,
}

impl PythonResolverEnv {
    pub(super) fn from_process_env() -> Self {
        Self {
            python_override: std::env::var("ROCM_CLI_PYTHON").ok(),
            search_dirs: std::env::var_os("PATH")
                .map(|value| split_runtime_path(&value))
                .unwrap_or_default(),
            managed_python_version: managed_python_version_override(),
            managed_bootstrap_disabled: managed_python_bootstrap_disabled(),
        }
    }

    /// The managed Python version a bootstrap under this environment prepares.
    fn managed_python_version(&self, requirement: PythonRequirement) -> String {
        self.managed_python_version
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| requirement.version.to_owned())
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

/// The interpreter an install of the wheel runtime goes on with, or, for a dry
/// run, the one it would.
///
/// The single place the install decides between the two, so a dry run cannot
/// reach [`resolve_python_launcher`]'s preparation by a path of its own.
///
/// It also prints the line that says which interpreter was chosen, once the
/// choice is made. A preview's line is built from the same value the decision
/// returned ([`PlannedPython::progress_text`]), so a line that says what a real
/// install would do cannot be printed on a path that ends in a refusal: the
/// refusal returns before any such line exists.
///
/// `prefix` is the `--prefix` the install was given. A preview runs nothing
/// from it, so an interpreter the search would have to run from inside it
/// refuses the preview instead (see [`refuse_preview_python_in_prefix`]).
pub(super) fn python_for_install(
    paths: &AppPaths,
    requirement: PythonRequirement,
    dry_run: bool,
    prefix: Option<&Path>,
) -> Result<PlannedPython> {
    let python = if dry_run {
        plan_python_launcher(paths, requirement, prefix)?
    } else {
        PlannedPython::Ready(resolve_python_launcher(paths, requirement)?)
    };
    progress_line(python.progress_text());
    Ok(python)
}

/// The interpreter a *preview* of the install, or an update check, reports,
/// found without preparing one.
///
/// [`resolve_python_launcher`] ends by downloading a managed `uv`, having it
/// install Python, and saving that Python's path in the configuration when no
/// interpreter on the host will do. A dry run promises to change nothing, so it
/// asks this instead: an interpreter that is usable now comes back as
/// [`PlannedPython::Ready`], exactly as the real install would pick it, and a
/// host that has none comes back as [`PlannedPython::WouldPrepareManaged`], the
/// step a real install would take, left untaken. Every error the real
/// resolution raises before that step (an unusable `ROCM_CLI_PYTHON`, a
/// disabled managed bootstrap) is raised here too, so a preview never succeeds
/// on a host where the install would refuse.
pub(super) fn plan_python_launcher(
    paths: &AppPaths,
    requirement: PythonRequirement,
    prefix: Option<&Path>,
) -> Result<PlannedPython> {
    let env = python_resolver_env_override().unwrap_or_else(PythonResolverEnv::from_process_env);
    plan_python_launcher_in(paths, &env, requirement, prefix)
}

fn plan_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
    prefix: Option<&Path>,
) -> Result<PlannedPython> {
    let lookup =
        find_python_launcher_in(paths, env, requirement, LookupPurpose::Preview { prefix })?;
    Ok(match lookup {
        PythonLookup::Found(launcher) => PlannedPython::Ready(launcher),
        PythonLookup::NeedsManagedBootstrap => {
            let version = env.managed_python_version(requirement);
            // `ROCM_CLI_MANAGED_PYTHON_VERSION` can name a version other than the
            // one the wheels need. A real install prepares it, finds it reports
            // the wrong tag and refuses (see `ensure_managed_python`), so the
            // preview refuses by the same rule rather than announce a plan the
            // install would not carry out. A version whose tag cannot be read
            // off its name is announced as unchecked: only installing it would
            // tell.
            let tag = python_tag_for_version(&version);
            if let Some(tag) = tag.as_deref() {
                ensure_python_tag_matches(tag, requirement).with_context(|| {
                    format!(
                        "a real install would prepare managed Python {version}, whose wheel tag it would then refuse"
                    )
                })?;
            }
            PlannedPython::WouldPrepareManaged {
                version,
                requirement,
                tag_check: if tag.is_some() {
                    TagCheck::Checked
                } else {
                    TagCheck::Unchecked
                },
            }
        }
    })
}

/// What [`plan_python_launcher`] found.
#[derive(Debug, Clone)]
pub(super) enum PlannedPython {
    /// An interpreter that is usable as it stands.
    Ready(PythonLauncher),
    /// No interpreter is usable. A real install would prepare managed Python
    /// `version`; this preview has not. `tag_check` says whether the wheel tag
    /// `version` reports could be read off its name and matched against the
    /// requirement; when it could not, only installing it would say.
    WouldPrepareManaged {
        version: String,
        requirement: PythonRequirement,
        tag_check: TagCheck,
    },
}

/// Whether a version a preview would prepare could be matched against the
/// wheel tag the layout requires before it is installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TagCheck {
    /// The tag was read off the version's name and matches the requirement.
    Checked,
    /// The tag cannot be read off the version's name (another implementation, a
    /// range, a wildcard that leaves the minor open): only installing it would
    /// tell.
    Unchecked,
}

impl PlannedPython {
    /// The `python_launcher:` line's value: the interpreter's path, or, when
    /// none exists yet, a statement that none was prepared rather than a path
    /// that would be a guess.
    pub(super) fn launcher_display(&self) -> String {
        match self {
            Self::Ready(launcher) => launcher.executable.display().to_string(),
            Self::WouldPrepareManaged { .. } => "not prepared by this preview".to_owned(),
        }
    }

    /// The `python_source:` line's value.
    pub(super) const fn source(&self) -> PythonSource {
        match self {
            Self::Ready(launcher) => launcher.source,
            Self::WouldPrepareManaged { .. } => PythonSource::Managed,
        }
    }

    /// What `uv venv --python` is handed in the printed command: the
    /// interpreter's path, or the version that would be prepared, which `uv`
    /// resolves to the managed Python the real install prepares.
    pub(super) fn venv_python_arg(&self) -> PathBuf {
        match self {
            Self::Ready(launcher) => launcher.executable.clone(),
            Self::WouldPrepareManaged { version, .. } => PathBuf::from(version),
        }
    }

    /// The `python_note:` line, present only when something was left undone.
    /// Written from the same value the branch was taken on, so it cannot name a
    /// version the install will not prepare.
    pub(super) fn note(&self) -> Option<String> {
        match self {
            Self::Ready(_) => None,
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => Some(format!(
                "no usable Python was found; a real install would use uv to install managed Python {version}{} and save its path in the configuration, which this preview did not do",
                unchecked_tag_caveat(*tag_check, *requirement)
            )),
        }
    }

    /// The `python_note:` line of an update check, in the same terms as
    /// [`Self::note`] but for the step an update would take. An update check
    /// needs the interpreter's wheel tag to read the index, and a host with no
    /// interpreter gets it from the one the layout requires, so this line is how
    /// the check says it did not go and prepare one. Built from the decision, as
    /// the preview's is.
    pub(super) fn update_note(&self) -> Option<String> {
        match self {
            Self::Ready(_) => None,
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => {
                let unchecked = unchecked_tag_caveat(*tag_check, *requirement);
                Some(format!(
                    "no usable Python was found; updating would use uv to install managed Python {version}{unchecked} and save its path in the configuration, which this check did not do"
                ))
            }
        }
    }

    /// The progress line that names the chosen interpreter: the one an install
    /// goes on with, or, for a preview on a host with no usable Python, what a
    /// real install would prepare. It is the only line that predicts the
    /// preparation, and it exists only for a decision that has already passed the
    /// bootstrap policy and the version check, so the two cannot disagree. Built
    /// from the same fields as [`Self::note`], so the two cannot differ about the
    /// version or whether its tag was checked.
    pub(super) fn progress_text(&self) -> String {
        match self {
            Self::Ready(launcher) => match launcher.source {
                PythonSource::Path => {
                    format!("Using Python from PATH: {}.", launcher.executable.display())
                }
                PythonSource::Env => format!(
                    "Using Python from ROCM_CLI_PYTHON: {}.",
                    launcher.executable.display()
                ),
                PythonSource::Managed => format!(
                    "Using ROCm CLI's portable Python: {}.",
                    launcher.executable.display()
                ),
                PythonSource::Manifest => {
                    format!("Using Python from {}.", launcher.executable.display())
                }
            },
            Self::WouldPrepareManaged {
                version,
                requirement,
                tag_check,
            } => format!(
                "No usable Python found; a real install would prepare Python {version}{}. This preview does not.",
                unchecked_tag_caveat(*tag_check, *requirement)
            ),
        }
    }

    /// The wheel tags the install will select packages by. An interpreter that
    /// exists is asked; one that does not yet is the one the layout requires,
    /// because a real install refuses to go on with any other (see
    /// `python_launcher_install_ready`).
    pub(super) fn wheel_compatibility(&self) -> Result<WheelCompatibility> {
        match self {
            Self::Ready(launcher) => wheel_compatibility_for_python(&launcher.executable),
            Self::WouldPrepareManaged { requirement, .. } => Ok(WheelCompatibility {
                python_tag: requirement.tag.to_owned(),
                platform_tags: current_platform_wheel_tags()?,
            }),
        }
    }

    /// The interpreter a real install runs with. A preview never reaches here
    /// with an interpreter it has not prepared.
    pub(super) fn into_launcher(self) -> Result<PythonLauncher> {
        match self {
            Self::Ready(launcher) => Ok(launcher),
            Self::WouldPrepareManaged { version, .. } => bail!(
                "internal error: Python {version} was not prepared before the install needed it"
            ),
        }
    }
}

/// The parenthetical a preview adds after the Python it would prepare when that
/// version's wheel tag could not be read off its name, and nothing when it
/// could. One function for [`PlannedPython::note`] and
/// [`PlannedPython::progress_text`], so the two cannot word the same caveat
/// differently.
fn unchecked_tag_caveat(tag_check: TagCheck, requirement: PythonRequirement) -> String {
    match tag_check {
        TagCheck::Checked => String::new(),
        TagCheck::Unchecked => format!(
            " (its wheel tag cannot be checked until it is installed; {} is required)",
            requirement.tag
        ),
    }
}

/// Whether a search for an interpreter is for an install or a preview of one.
/// It changes only the wording of the two progress lines about a Python that
/// turned out unusable: an install says what it goes on to do, a preview states
/// the finding and leaves the prediction to the decision's own line.
///
/// A preview also carries the install's `--prefix`, from which it runs nothing.
#[derive(Clone, Copy, Debug)]
enum LookupPurpose<'a> {
    Install,
    Preview { prefix: Option<&'a Path> },
}

impl LookupPurpose<'_> {
    /// Refuses a preview whose search would next run `candidate` (found
    /// through `source`) from inside the install's `--prefix`.
    ///
    /// A preview runs nothing from the install folder: `install_sdk_dry_run`
    /// is a model-callable tool with a model-chosen `--prefix`, and a folder
    /// can hold a hostile `bin/python`. The search runs each interpreter it
    /// meets, and skipping one would make the preview name a different
    /// interpreter from the one the install would pick, so it stops and says
    /// why instead. The interpreter's own folder is resolved but not the
    /// interpreter: a venv's `bin/python` links to the base Python, and run
    /// through that link it still reads the venv's `pyvenv.cfg` and imports
    /// from its `site-packages`. An install is not affected.
    fn refuse_preview_python_in_prefix(self, candidate: &Path, source: &str) -> Result<()> {
        let Self::Preview {
            prefix: Some(prefix),
        } = self
        else {
            return Ok(());
        };
        let location = match (candidate.parent(), candidate.file_name()) {
            (Some(parent), Some(name)) if !parent.as_os_str().is_empty() => {
                rocm_core::resolve_path_through_symlinks(parent).join(name)
            }
            _ => return Ok(()),
        };
        let prefix = rocm_core::resolve_path_through_symlinks(prefix);
        if rocm_core::runtime_path_is_same_or_inside(&location, &prefix) {
            bail!(
                "this preview did not check the Python from {source} at {}: it is inside the install folder {}, and a preview runs nothing from the install folder. A real install would run it. Preview again with that Python out of the way (off PATH, or ROCM_CLI_PYTHON unset), or choose another --prefix",
                candidate.display(),
                prefix.display()
            );
        }
        Ok(())
    }

    /// The progress line for an interpreter (`subject`) that exists but cannot
    /// serve the install. Both purposes state the finding; an install then says
    /// what it goes on to do (`install_next_step`) and a preview stops.
    fn unusable_python_line(
        self,
        subject: &str,
        requirement: PythonRequirement,
        install_next_step: &str,
    ) -> String {
        let finding = format!(
            "{subject} is not usable for this ROCm install ({} is required)",
            requirement.tag
        );
        match self {
            Self::Install => format!("{finding}; {install_next_step}."),
            Self::Preview { .. } => format!("{finding}."),
        }
    }
}

/// Where the search for an interpreter ended.
enum PythonLookup {
    /// An interpreter that is usable as it stands.
    Found(PythonLauncher),
    /// Nothing is usable and the managed bootstrap is allowed: the next step is
    /// [`ensure_managed_python`], which downloads and writes.
    NeedsManagedBootstrap,
}

fn resolve_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
) -> Result<PythonLauncher> {
    match find_python_launcher_in(paths, env, requirement, LookupPurpose::Install)? {
        PythonLookup::Found(launcher) => Ok(launcher),
        PythonLookup::NeedsManagedBootstrap => {
            ensure_managed_python(paths, requirement, env.managed_python_version(requirement))
        }
    }
}

/// Everything [`resolve_python_launcher_in`] does short of preparing a managed
/// Python: it reads the host and the saved manifest and writes nothing.
fn find_python_launcher_in(
    paths: &AppPaths,
    env: &PythonResolverEnv,
    requirement: PythonRequirement,
    purpose: LookupPurpose<'_>,
) -> Result<PythonLookup> {
    if let Some(value) = env.python_override.as_deref() {
        // A bare name is run through `PATH`, so it is checked where it resolves.
        let named = Path::new(value);
        let runs = if named
            .parent()
            .is_some_and(|parent| !parent.as_os_str().is_empty())
        {
            Some(named.to_path_buf())
        } else {
            resolve_program_on_path(value, &env.search_dirs)
                .into_iter()
                .next()
        };
        if let Some(runs) = runs.as_deref() {
            purpose.refuse_preview_python_in_prefix(runs, "ROCM_CLI_PYTHON")?;
        }
        python_launcher_install_ready(Path::new(value), requirement)
            .with_context(|| format!("ROCM_CLI_PYTHON is not usable for ROCm setup: {value}"))?;
        return Ok(PythonLookup::Found(PythonLauncher {
            executable: PathBuf::from(value),
            source: PythonSource::Env,
        }));
    }

    let mut skipped_path_python = false;
    for candidate in python_path_candidates(&env.search_dirs) {
        purpose.refuse_preview_python_in_prefix(&candidate, "PATH")?;
        match python_launcher_install_ready(&candidate, requirement) {
            Ok(()) => {
                return Ok(PythonLookup::Found(PythonLauncher {
                    executable: candidate,
                    source: PythonSource::Path,
                }));
            }
            Err(_) => {
                skipped_path_python = true;
            }
        }
    }
    // A preview states only what it found here. What a real install would then
    // do depends on the saved Python, the bootstrap policy and the version, none
    // of which is known yet, so the prediction waits for
    // `PlannedPython::progress_text`, after the decision. An install's wording is
    // unchanged.
    if skipped_path_python {
        progress_line(purpose.unusable_python_line(
            "Python from PATH",
            requirement,
            "using ROCm CLI's managed Python",
        ));
    }

    if let Some(manifest) = load_managed_python_manifest(paths)?
        && manifest.executable.is_file()
    {
        purpose
            .refuse_preview_python_in_prefix(&manifest.executable, "the saved managed Python")?;
        if python_launcher_install_ready(&manifest.executable, requirement).is_ok() {
            return Ok(PythonLookup::Found(PythonLauncher {
                executable: manifest.executable,
                source: PythonSource::Managed,
            }));
        }
        progress_line(purpose.unusable_python_line(
            "Saved managed Python",
            requirement,
            "preparing Python again",
        ));
    }

    if env.managed_bootstrap_disabled {
        bail!(
            "unable to locate Python, and managed Python bootstrap is disabled by ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP"
        );
    }
    Ok(PythonLookup::NeedsManagedBootstrap)
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
    ensure_python_tag_matches(&compatibility.python_tag, requirement)?;
    verify_python_can_create_venv(program).context("Python could not create a virtual environment")
}

/// The one rule an interpreter must meet to serve an install: its wheel tag is
/// the one the layout requires. [`python_launcher_install_ready`] applies it to
/// an interpreter that exists, and a preview applies it to the managed Python it
/// would prepare, so the two cannot disagree about what the install accepts.
fn ensure_python_tag_matches(python_tag: &str, requirement: PythonRequirement) -> Result<()> {
    if python_tag != requirement.tag {
        bail!(
            "Python wheel tag {python_tag} is not supported by this ROCm install; {} is required",
            requirement.tag
        );
    }
    Ok(())
}

/// The wheel tag a managed Python requested as `version` reports, or `None`
/// when it cannot be told without installing it.
///
/// `uv` is what reads `version` (it is passed through to `uv python install`),
/// and this repo has no parser of its own to share, so this reads the spellings
/// `uv` accepts for a CPython release that name a major and a minor: `3.13`,
/// `3.13.1`, `3.13.*`, `3.13t` (free-threaded, which still reports `cp313`), and
/// any of those after `cpython-` or `cpython@`. Whatever follows the minor is
/// not validated, because only the major and minor decide the wheel tag.
/// Anything that leaves the minor open or names something else (another
/// implementation, a range, a wildcard such as `3.*`, a pre-release spelled on
/// the minor) yields `None`.
///
/// This reader is a hand-rolled approximation of what `uv` accepts and can drift
/// from it in either direction: a spelling `uv` takes may read as `None`, and one
/// `uv` would reject may read as a tag. Both are benign. A `None` is announced
/// as unchecked (the preview still prints a plan and says the tag cannot be
/// checked until the Python is installed), never refused, and a tag read off a
/// spelling `uv` then rejects is only a preview that predicted an install that
/// fails at `uv python install`, with `uv`'s own error, as it always did.
fn python_tag_for_version(version: &str) -> Option<String> {
    let version = version.trim();
    let version = ["cpython-", "cpython@"]
        .iter()
        .find_map(|prefix| version.strip_prefix(prefix))
        .unwrap_or(version);
    let mut parts = version.split('.');
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next()?;
    // Only a free-threaded marker may follow the minor's digits.
    let minor = minor
        .strip_suffix('t')
        .unwrap_or(minor)
        .parse::<u32>()
        .ok()?;
    Some(format!("cp{major}{minor}"))
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
    // Only the Unix tests use these two; the tests that run on every platform
    // import the sink in their own bodies.
    #[cfg(unix)]
    use super::super::managed_runtime_root;
    #[cfg(unix)]
    use super::super::process::PROGRESS_LINE_SINK;
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::{RestoredEnv, test_paths};

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
                ..PythonResolverEnv::from_process_env()
            },
            python_requirement(SourceLayout::Canonical),
        )?;
        assert_eq!(launcher.source, PythonSource::Path);
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
                ..PythonResolverEnv::from_process_env()
            },
            python_requirement(SourceLayout::Canonical),
        )?;

        assert_eq!(launcher.source, PythonSource::Path);
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
                ..PythonResolverEnv::from_process_env()
            });
            resolve_python_launcher(&paths, python_requirement(SourceLayout::Canonical))?
        };

        assert_eq!(
            launcher.source,
            PythonSource::Path,
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
                ..PythonResolverEnv::from_process_env()
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
                    ..PythonResolverEnv::from_process_env()
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

    // ---- a dry run prepares no Python ----------------------------------------

    /// A `uv` that records its arguments in `marker` and then fails. A host that
    /// reaches it has gone on to prepare a managed Python; failing keeps a test
    /// that does reach it from going on to the network.
    #[cfg(unix)]
    fn write_recording_uv(dir: &Path, marker: &Path) -> Result<PathBuf> {
        use rocm_core::fake_executable::write_fake_executable;
        let path = dir.join("uv");
        write_fake_executable(
            &path,
            format!("#!/bin/sh\necho \"$@\" >> '{}'\nexit 1\n", marker.display()),
        )?;
        Ok(path)
    }

    /// Every path under the three folders a command may write to, sorted. Empty
    /// when none of them exists, so "nothing was created" and "something empty
    /// was created" are different answers.
    fn snapshot(paths: &AppPaths) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                out.push(entry.path());
                walk(&entry.path(), out);
            }
        }
        let mut out = Vec::new();
        for dir in [&paths.config_dir, &paths.data_dir, &paths.cache_dir] {
            if dir.exists() {
                out.push(dir.clone());
            }
            walk(dir, &mut out);
        }
        out.sort();
        out
    }

    /// A host with no usable Python: an empty search folder and no saved Python,
    /// unless the test adds one. `uv` is a recording stand-in, so a code path that
    /// prepares Python is visible as a marker instead of a download.
    #[cfg(unix)]
    struct PythonHost {
        root: PathBuf,
        paths: AppPaths,
        search_dir: PathBuf,
        uv_marker: PathBuf,
        _env: RestoredEnv,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    #[cfg(unix)]
    impl PythonHost {
        fn new(name: &str) -> Result<Self> {
            let lock = PROCESS_ENV_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (root, paths) = test_paths(name);
            let search_dir = root.join("search");
            fs::create_dir_all(&search_dir)?;
            let uv_marker = root.join("uv-was-run");
            let tools = root.join("tools");
            fs::create_dir_all(&tools)?;
            let uv = write_recording_uv(&tools, &uv_marker)?;
            let env = RestoredEnv::set(&[
                ("ROCM_CLI_UV_BINARY", Some(&uv)),
                ("ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP", None),
                ("ROCM_CLI_MANAGED_PYTHON_VERSION", None),
            ]);
            Ok(Self {
                root,
                paths,
                search_dir,
                uv_marker,
                _env: env,
                _lock: lock,
            })
        }

        /// What an install (`dry_run: false`) or its preview (`true`) goes on
        /// with, searching only this host's folder.
        fn python_for_install(&self, dry_run: bool) -> Result<PlannedPython> {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![self.search_dir.clone()],
                ..PythonResolverEnv::from_process_env()
            });
            python_for_install(
                &self.paths,
                python_requirement(SourceLayout::Canonical),
                dry_run,
                None,
            )
        }

        /// What an update check goes on with, searching only this host's folder.
        fn plan_for_update_check(&self) -> Result<PlannedPython> {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: vec![self.search_dir.clone()],
                ..PythonResolverEnv::from_process_env()
            });
            plan_python_launcher(
                &self.paths,
                python_requirement(SourceLayout::Canonical),
                None,
            )
        }

        fn uv_was_run(&self) -> bool {
            self.uv_marker.exists()
        }

        fn save_managed_python(&self, executable: PathBuf) -> Result<Vec<u8>> {
            save_managed_python_manifest(
                &self.paths,
                &ManagedPythonManifest {
                    executable,
                    version: DEFAULT_MANAGED_PYTHON_VERSION.to_owned(),
                    installed_at_unix_ms: 123,
                },
            )?;
            Ok(fs::read(managed_python_manifest_path(&self.paths))?)
        }
    }

    #[cfg(unix)]
    impl Drop for PythonHost {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).ok();
        }
    }

    /// The bug: a dry run on a host with no usable Python went on to prepare one.
    /// Preparing means a managed `uv` and Python are downloaded and the Python's
    /// path is saved in the configuration, all before a plan is printed.
    ///
    /// Asserts the preview's answer and the state it leaves together (the message
    /// claims nothing was prepared, so the folders have to agree), and that the
    /// same host DOES reach `uv` when the install is real, so an empty marker
    /// means the preview stayed out of it rather than that the stand-in is not
    /// wired up. The Gherkin scenario `runtime-23` drives the same case through
    /// the built binary.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_with_no_usable_python_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-none")?;
        assert!(
            snapshot(&host.paths).is_empty(),
            "the fixture must start with the config, data and cache folders absent"
        );

        let planned = host.python_for_install(true)?;

        let PlannedPython::WouldPrepareManaged {
            ref version,
            requirement,
            tag_check,
        } = planned
        else {
            panic!("a host with no Python must be planned as needing one, got {planned:?}");
        };
        assert_eq!(version, DEFAULT_MANAGED_PYTHON_VERSION);
        assert_eq!(requirement, python_requirement(SourceLayout::Canonical));
        assert_eq!(
            tag_check,
            TagCheck::Checked,
            "the default version's tag is readable"
        );
        let note = planned
            .note()
            .expect("a preview that skipped a step says so");
        assert!(
            note.contains("would use uv to install managed Python 3.12")
                && note.ends_with("which this preview did not do"),
            "{note}"
        );
        // What the note says was not done, was not done.
        assert!(!host.uv_was_run(), "the dry run ran uv");
        assert!(
            !host.paths.config_path().exists(),
            "the dry run saved a Python path in the configuration"
        );
        assert!(
            !managed_python_manifest_path(&host.paths).exists(),
            "the dry run saved a managed Python manifest"
        );
        assert_eq!(
            snapshot(&host.paths),
            Vec::<PathBuf>::new(),
            "the dry run created files or folders under the config, data or cache folder"
        );

        // Control: the real install on the same host does reach `uv`.
        assert!(host.python_for_install(false).is_err());
        assert!(
            host.uv_was_run(),
            "the stand-in uv is not reachable, so the checks above prove nothing"
        );
        Ok(())
    }

    /// The preview's decision, made from an injected environment, on every
    /// platform.
    ///
    /// The tests around it drive the decision with shell-script interpreters and
    /// run on Unix only. The decision itself (what a host with no usable Python
    /// is told, which managed version is announced or refused, what the bootstrap
    /// policy does) needs no interpreter: a [`PythonResolverEnv`] with nothing to
    /// find and the policy written into it exercises `python_for_install`
    /// end to end, so this one runs on the Windows CI job as well. The search
    /// order and the interpreter probes stay Unix-only, as the PR text says.
    ///
    /// Each row asserts the whole progress output and the state left behind
    /// together. The control at the end is the install on the same host: it
    /// reaches the managed `uv` (a file that cannot be launched, so nothing is
    /// downloaded or run), which shows the preview's silence about `uv` means it
    /// stayed out of it, not that the stand-in was never wired up.
    #[test]
    fn a_dry_run_decides_from_an_injected_environment_on_every_platform() -> Result<()> {
        use super::super::process::PROGRESS_LINE_SINK;

        let _lock = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, paths) = test_paths("python-dry-run-injected");
        let tools = root.join("tools");
        fs::create_dir_all(&tools)?;
        // Present but not executable, on every platform: `ensure_uv_binary`
        // returns it as the override, and launching it fails.
        let uv = tools.join("uv");
        fs::write(&uv, "not an executable")?;
        let _uv = RestoredEnv::set(&[("ROCM_CLI_UV_BINARY", Some(&uv))]);
        assert!(snapshot(&paths).is_empty());

        let requirement = python_requirement(SourceLayout::Canonical);
        let run = |version: Option<&str>, disabled: bool, dry_run: bool| {
            let _resolver = OverridePythonResolverEnv::new(PythonResolverEnv {
                python_override: None,
                search_dirs: Vec::new(),
                managed_python_version: version.map(str::to_owned),
                managed_bootstrap_disabled: disabled,
            });
            PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
            let result = python_for_install(&paths, requirement, dry_run, None);
            let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
            (result, lines)
        };
        let promise = |version: &str, caveat: &str| {
            format!(
                "No usable Python found; a real install would prepare Python {version}{caveat}. This preview does not."
            )
        };

        // Announced: the default, a patch release of the required minor, a
        // blank override (read as unset) and a version whose tag cannot be read.
        for (version, announced, tag_check) in [
            (None, "3.12", TagCheck::Checked),
            (Some("3.12.4"), "3.12.4", TagCheck::Checked),
            (Some("cpython@3.12"), "cpython@3.12", TagCheck::Checked),
            (Some("  "), "3.12", TagCheck::Checked),
            (Some("pypy@3.10"), "pypy@3.10", TagCheck::Unchecked),
        ] {
            let (result, lines) = run(version, false, true);
            let planned = result.unwrap_or_else(|error| panic!("{version:?}: {error:#}"));
            let PlannedPython::WouldPrepareManaged {
                version: planned_version,
                tag_check: planned_check,
                ..
            } = &planned
            else {
                panic!(
                    "{version:?}: a host with no Python must be planned as needing one, got {planned:?}"
                );
            };
            assert_eq!(planned_version, announced, "{version:?}");
            assert_eq!(*planned_check, tag_check, "{version:?}");
            let caveat = match tag_check {
                TagCheck::Checked => "",
                TagCheck::Unchecked => {
                    " (its wheel tag cannot be checked until it is installed; cp312 is required)"
                }
            };
            assert_eq!(lines, vec![promise(announced, caveat)], "{version:?}");
            assert!(
                planned.note().is_some_and(|note| note.contains(&format!(
                    "install managed Python {announced}{caveat} and save"
                ))),
                "{version:?}: {planned:?}"
            );
        }

        // Refused: a version whose wheel tag is not the layout's, in every
        // spelling that reads as one, and a host where the bootstrap is
        // disabled. Nothing is promised, so no line predicts the install.
        for version in ["3.13", "3.13t", "cpython-3.13", "cpython@3.13"] {
            let (result, lines) = run(Some(version), false, true);
            let error = format!("{:#}", result.unwrap_err());
            assert!(
                error.contains(&format!("would prepare managed Python {version}"))
                    && error.contains("Python wheel tag cp313 is not supported")
                    && error.contains("cp312 is required"),
                "{version}: {error}"
            );
            assert_eq!(lines, Vec::<String>::new(), "{version}");
        }
        let (preview, lines) = run(None, true, true);
        let preview = preview.unwrap_err().to_string();
        let (install, _) = run(None, true, false);
        assert!(
            preview.contains("managed Python bootstrap is disabled"),
            "{preview}"
        );
        assert_eq!(preview, install.unwrap_err().to_string());
        assert_eq!(lines, Vec::<String>::new());

        // State: every preview above left the config, data and cache folders as
        // they were, which for this host is absent.
        assert!(!paths.config_path().exists());
        assert!(!managed_python_manifest_path(&paths).exists());
        assert_eq!(snapshot(&paths), Vec::<PathBuf>::new());

        // Control: the real install on the same host goes on to prepare Python,
        // which reaches the managed `uv`.
        let (install, lines) = run(None, false, false);
        let error = format!("{:#}", install.unwrap_err());
        assert!(
            error.contains("failed to launch uv python install"),
            "the install did not reach uv, so the previews above prove nothing: {error}"
        );
        assert_eq!(
            lines,
            vec![
                "Preparing Python 3.12...".to_owned(),
                "Installing Python 3.12 via uv...".to_owned()
            ]
        );
        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    /// The two lines about a Python that exists but cannot serve the install are
    /// worded for the purpose of the search: an install says what it goes on to
    /// do, a preview states only the finding (what a real install would then do
    /// is decided afterwards, and said by the decision's own line). Pinned here
    /// as they are printed, so that the two wordings stay what they are as the
    /// code that prints them changes; the decisions those lines sit in are
    /// covered by `a_dry_runs_whole_python_output_follows_one_decision_in_every_state`
    /// and `a_dry_run_decides_from_an_injected_environment_on_every_platform`.
    ///
    /// A text file stands for the unusable interpreters (neither the PATH one nor
    /// the saved one can be launched), so this runs on every platform. Bootstrap is
    /// disabled in the injected environment so both searches end in a refusal
    /// before anything could be prepared.
    #[test]
    fn the_findings_about_an_unusable_python_are_worded_for_the_purpose_of_the_search() -> Result<()>
    {
        use super::super::process::PROGRESS_LINE_SINK;

        let (root, paths) = test_paths("python-unusable-findings");
        let search_dir = root.join("search");
        fs::create_dir_all(&search_dir)?;
        fs::write(search_dir.join("python3"), "not an interpreter")?;
        let saved_dir = paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&saved_dir)?;
        let saved = saved_dir.join("python");
        fs::write(&saved, "not an interpreter")?;
        save_managed_python_manifest(
            &paths,
            &ManagedPythonManifest {
                executable: saved,
                version: DEFAULT_MANAGED_PYTHON_VERSION.to_owned(),
                installed_at_unix_ms: 123,
            },
        )?;
        let env = PythonResolverEnv {
            python_override: None,
            search_dirs: vec![search_dir],
            managed_python_version: None,
            managed_bootstrap_disabled: true,
        };
        let requirement = python_requirement(SourceLayout::Canonical);

        for (purpose, expected) in [
            (
                LookupPurpose::Install,
                [
                    "Python from PATH is not usable for this ROCm install (cp312 is required); using ROCm CLI's managed Python.",
                    "Saved managed Python is not usable for this ROCm install (cp312 is required); preparing Python again.",
                ],
            ),
            (
                LookupPurpose::Preview { prefix: None },
                [
                    "Python from PATH is not usable for this ROCm install (cp312 is required).",
                    "Saved managed Python is not usable for this ROCm install (cp312 is required).",
                ],
            ),
        ] {
            PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
            let outcome = find_python_launcher_in(&paths, &env, requirement, purpose);
            let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());
            assert_eq!(lines, expected, "{purpose:?}");
            let error = outcome
                .err()
                .expect("bootstrap is disabled, so the search refuses");
            assert!(
                error
                    .to_string()
                    .contains("managed Python bootstrap is disabled"),
                "{purpose:?}: {error:#}"
            );
        }
        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    /// A Python on the search path is used by the preview exactly as by the
    /// install, and costs nothing: no `uv`, nothing written.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reports_a_usable_path_python_and_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-path")?;
        let python = write_fake_python_with_venv(&host.search_dir, "python3")?;

        let planned = host.python_for_install(true)?;

        let PlannedPython::Ready(launcher) = planned else {
            panic!("a usable PATH Python must be reported as ready, got {planned:?}");
        };
        assert_eq!(launcher.source, PythonSource::Path);
        assert_eq!(launcher.executable, python);
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        Ok(())
    }

    /// A saved managed Python that still works is reported, and left as it was.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reports_a_usable_saved_managed_python_and_changes_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-saved")?;
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let managed = write_fake_python_with_venv(&managed_dir, "python")?;
        let manifest = host.save_managed_python(managed.clone())?;
        let before = snapshot(&host.paths);

        let planned = host.python_for_install(true)?;

        let PlannedPython::Ready(launcher) = planned else {
            panic!("a usable saved Python must be reported as ready, got {planned:?}");
        };
        assert_eq!(launcher.source, PythonSource::Managed);
        assert_eq!(launcher.executable, managed);
        assert!(!host.uv_was_run());
        assert!(
            !host.paths.config_path().exists(),
            "reporting a saved Python must not re-save it in the configuration"
        );
        assert_eq!(
            fs::read(managed_python_manifest_path(&host.paths))?,
            manifest
        );
        assert_eq!(snapshot(&host.paths), before);
        Ok(())
    }

    /// A saved managed Python that no longer works is a host with none: the
    /// preview says one would be prepared, and leaves the stale record alone
    /// rather than clearing or rewriting it.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_with_an_unusable_saved_python_prepares_nothing() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-stale")?;
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        fs::create_dir_all(&managed_dir)?;
        let broken = managed_dir.join("python");
        fs::write(&broken, "not an interpreter")?;
        let manifest = host.save_managed_python(broken)?;
        let before = snapshot(&host.paths);

        let planned = host.python_for_install(true)?;

        assert!(
            matches!(planned, PlannedPython::WouldPrepareManaged { .. }),
            "{planned:?}"
        );
        assert!(!host.uv_was_run());
        assert!(!host.paths.config_path().exists());
        assert_eq!(
            fs::read(managed_python_manifest_path(&host.paths))?,
            manifest
        );
        assert_eq!(snapshot(&host.paths), before);
        Ok(())
    }

    /// Where an install refuses, the preview refuses too, with the same message:
    /// a preview that printed a plan for an install that cannot happen would be
    /// a false promise.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_refuses_where_the_install_does_when_managed_python_is_disabled() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-disabled")?;
        let _disabled = RestoredEnv::set(&[(
            "ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP",
            Some(Path::new("1")),
        )]);

        let preview = host.python_for_install(true).unwrap_err().to_string();
        let install = host.python_for_install(false).unwrap_err().to_string();

        assert!(
            preview.contains("managed Python bootstrap is disabled"),
            "{preview}"
        );
        assert_eq!(preview, install);
        assert!(!host.uv_was_run());
        Ok(())
    }

    /// `ROCM_CLI_MANAGED_PYTHON_VERSION` naming a version whose wheel tag is not
    /// the one the layout needs: a real install prepares it and then refuses it,
    /// so the preview refuses too, by the same rule, and prepares nothing. A
    /// version the layout does accept (a patch release of its own minor) is still
    /// announced as before.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_refuses_a_managed_python_version_the_install_would_refuse() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-version-override")?;
        let overridden =
            RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new("3.13")))]);

        let error = format!("{:#}", host.python_for_install(true).unwrap_err());

        assert!(
            error.contains("would prepare managed Python 3.13")
                && error.contains("Python wheel tag cp313 is not supported")
                && error.contains("cp312 is required"),
            "{error}"
        );
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        drop(overridden);

        // A patch release of the required minor is what the install accepts.
        let _patch =
            RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new("3.12.4")))]);
        let planned = host.python_for_install(true)?;
        assert!(
            matches!(&planned, PlannedPython::WouldPrepareManaged { version, .. } if version == "3.12.4"),
            "{planned:?}"
        );
        assert!(!host.uv_was_run());
        Ok(())
    }

    // ---- the preview's output, row by row -------------------------------------
    //
    // State matrix for what a dry run prints about Python and what it leaves
    // behind. A preview line that says what a real install would do must exist
    // only where the install would do it, so every row below is the whole output
    // of the decision, not a spot check of one line.
    //
    //   PATH Python   usable | unusable | absent
    //   saved Python  usable | unusable | absent
    //   bootstrap     enabled | disabled (ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP)
    //   version       default | accepted patch release | unreadable | refused tag
    //
    // Expected output by row (cp312 required; `P`, `S` are the PATH and saved
    // interpreters):
    //   PATH usable                      -> `Using Python from PATH: P.`; nothing else
    //   PATH unusable, saved usable      -> PATH finding, `Using ROCm CLI's portable Python: S.`
    //   neither usable, bootstrap off    -> findings only, then the refusal; no promise
    //   neither usable, version refused  -> findings only, then the refusal; no promise
    //   neither usable, otherwise        -> findings, then the one promise
    //                                       `No usable Python found; a real install
    //                                       would prepare Python V.` (V checked or not)
    // A finding is `Python from PATH is not usable ... (cp312 is required).` or
    // `Saved managed Python is not usable ... (cp312 is required).`, printed only
    // for a PATH or saved Python that exists and fails. Every row leaves the
    // config, data and cache folders as they were and never runs `uv`.

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum PathPython {
        Usable,
        Unusable,
        Absent,
    }

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum SavedPython {
        Usable,
        Unusable,
        Absent,
    }

    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum VersionOverride {
        Default,
        Accepted,
        Unreadable,
        RefusedTag,
    }

    /// Runs one row of the matrix and checks the preview's complete output and
    /// the state it leaves against what the row's description above says.
    #[cfg(unix)]
    fn check_preview_matrix_row(
        path: PathPython,
        saved: SavedPython,
        bootstrap_disabled: bool,
        version: VersionOverride,
    ) -> Result<()> {
        use super::super::process::PROGRESS_LINE_SINK;

        let row = format!(
            "{path:?} PATH / {saved:?} saved / disabled={bootstrap_disabled} / {version:?}"
        );
        let host = PythonHost::new("python-dry-run-matrix")?;
        let path_python = match path {
            PathPython::Usable => Some(write_fake_python_with_venv(&host.search_dir, "python3")?),
            PathPython::Unusable => {
                write_fake_python_reporting(&host.search_dir, "python3", "cp311", "Python 3.11.0")?;
                None
            }
            PathPython::Absent => None,
        };
        let managed_dir = host.paths.data_dir.join("tools").join("python");
        let saved_python = match saved {
            SavedPython::Usable => {
                fs::create_dir_all(&managed_dir)?;
                let python = write_fake_python_with_venv(&managed_dir, "python")?;
                host.save_managed_python(python.clone())?;
                Some(python)
            }
            SavedPython::Unusable => {
                fs::create_dir_all(&managed_dir)?;
                let broken = managed_dir.join("python");
                fs::write(&broken, "not an interpreter")?;
                host.save_managed_python(broken)?;
                None
            }
            SavedPython::Absent => None,
        };
        let (version_value, version_name) = match version {
            VersionOverride::Default => (None, "3.12"),
            VersionOverride::Accepted => (Some("3.12.4"), "3.12.4"),
            VersionOverride::Unreadable => (Some("pypy@3.10"), "pypy@3.10"),
            VersionOverride::RefusedTag => (Some("3.13"), "3.13"),
        };
        let _env = RestoredEnv::set(&[
            (
                "ROCM_CLI_DISABLE_MANAGED_PYTHON_BOOTSTRAP",
                bootstrap_disabled.then_some(Path::new("1")),
            ),
            (
                "ROCM_CLI_MANAGED_PYTHON_VERSION",
                version_value.map(Path::new),
            ),
        ]);
        let before = snapshot(&host.paths);

        // What the row's description says, written out independently of the code.
        let path_finding =
            "Python from PATH is not usable for this ROCm install (cp312 is required).";
        let saved_finding =
            "Saved managed Python is not usable for this ROCm install (cp312 is required).";
        let mut expected: Vec<String> = Vec::new();
        let mut expected_error: Option<&str> = None;
        let mut expected_promise = false;
        if let Some(python) = &path_python {
            expected.push(format!("Using Python from PATH: {}.", python.display()));
        } else {
            if matches!(path, PathPython::Unusable) {
                expected.push(path_finding.to_owned());
            }
            if let Some(python) = &saved_python {
                expected.push(format!(
                    "Using ROCm CLI's portable Python: {}.",
                    python.display()
                ));
            } else {
                if matches!(saved, SavedPython::Unusable) {
                    expected.push(saved_finding.to_owned());
                }
                if bootstrap_disabled {
                    expected_error = Some("managed Python bootstrap is disabled");
                } else if matches!(version, VersionOverride::RefusedTag) {
                    expected_error = Some("Python wheel tag cp313 is not supported");
                } else {
                    expected_promise = true;
                    expected.push(if matches!(version, VersionOverride::Unreadable) {
                        format!("No usable Python found; a real install would prepare Python {version_name} (its wheel tag cannot be checked until it is installed; cp312 is required). This preview does not.")
                    } else {
                        format!("No usable Python found; a real install would prepare Python {version_name}. This preview does not.")
                    });
                }
            }
        }

        PROGRESS_LINE_SINK.with(|sink| sink.borrow_mut().clear());
        let result = host.python_for_install(true);
        let lines = PROGRESS_LINE_SINK.with(|sink| sink.borrow().clone());

        assert_eq!(lines, expected, "{row}: the preview's progress lines");
        // No line that predicts the install may exist unless the row ends in
        // the promise: a refusal is known before any such line is built.
        let predictions = lines
            .iter()
            .filter(|line| line.contains("a real install would"))
            .count();
        assert_eq!(
            predictions,
            usize::from(expected_promise),
            "{row}: lines that predict the install: {lines:?}"
        );
        match (&result, expected_error) {
            (Err(error), Some(expected)) => {
                assert!(format!("{error:#}").contains(expected), "{row}: {error:#}");
            }
            (Ok(planned), None) => {
                assert_eq!(
                    planned.note().is_some(),
                    expected_promise,
                    "{row}: the note goes with the promise"
                );
                if expected_promise {
                    assert!(
                        planned.note().is_some_and(
                            |note| note.contains(&format!("managed Python {version_name}"))
                        ),
                        "{row}: {planned:?}"
                    );
                }
            }
            (result, expected) => panic!("{row}: got {result:?}, expected error {expected:?}"),
        }
        // State: nothing was prepared and nothing was written.
        assert!(!host.uv_was_run(), "{row}: the preview ran uv");
        assert!(
            !host.paths.config_path().exists(),
            "{row}: a config was written"
        );
        assert_eq!(snapshot(&host.paths), before, "{row}: files changed");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_dry_runs_whole_python_output_follows_one_decision_in_every_state() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        for path in [PathPython::Usable, PathPython::Unusable, PathPython::Absent] {
            for saved in [
                SavedPython::Usable,
                SavedPython::Unusable,
                SavedPython::Absent,
            ] {
                for bootstrap_disabled in [false, true] {
                    for version in [
                        VersionOverride::Default,
                        VersionOverride::Accepted,
                        VersionOverride::Unreadable,
                        VersionOverride::RefusedTag,
                    ] {
                        check_preview_matrix_row(path, saved, bootstrap_disabled, version)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The same refusal for every spelling `uv` accepts for that release: a
    /// free-threaded `3.13t` and the `cpython-` and `cpython@` forms report
    /// `cp313` once installed just as `3.13` does, so a preview that did not read
    /// them would promise a plan the install refuses. A spelling whose tag
    /// cannot be read is announced as unchecked, not as a promise.
    #[cfg(unix)]
    #[test]
    fn a_dry_run_reads_every_spelling_of_a_managed_python_version_the_same_way() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-dry-run-version-spellings")?;
        for spelling in ["3.13t", "cpython-3.13", "cpython@3.13"] {
            let _version =
                RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new(spelling)))]);
            let error = format!("{:#}", host.python_for_install(true).unwrap_err());
            assert!(
                error.contains(&format!("would prepare managed Python {spelling}"))
                    && error.contains("Python wheel tag cp313 is not supported")
                    && error.contains("cp312 is required"),
                "{spelling}: {error}"
            );
        }
        for accepted in ["cpython-3.12", "cpython@3.12.4"] {
            let _version =
                RestoredEnv::set(&[("ROCM_CLI_MANAGED_PYTHON_VERSION", Some(Path::new(accepted)))]);
            let planned = host.python_for_install(true)?;
            assert!(
                matches!(
                    &planned,
                    PlannedPython::WouldPrepareManaged {
                        tag_check: TagCheck::Checked,
                        ..
                    }
                ),
                "{accepted}: {planned:?}"
            );
        }

        let _unreadable = RestoredEnv::set(&[(
            "ROCM_CLI_MANAGED_PYTHON_VERSION",
            Some(Path::new("pypy@3.10")),
        )]);
        let planned = host.python_for_install(true)?;
        assert!(
            matches!(
                &planned,
                PlannedPython::WouldPrepareManaged {
                    tag_check: TagCheck::Unchecked,
                    ..
                }
            ),
            "{planned:?}"
        );
        let note = planned
            .note()
            .expect("a preview that skipped a step says so");
        assert!(
            note.contains("managed Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required)")
                && note.ends_with("which this preview did not do"),
            "{note}"
        );
        assert_eq!(
            planned.progress_text(),
            "No usable Python found; a real install would prepare Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required). This preview does not."
        );
        assert!(!host.uv_was_run());
        assert_eq!(snapshot(&host.paths), Vec::<PathBuf>::new());
        Ok(())
    }

    /// An update check reads the index by the interpreter's wheel tag. On a host
    /// with no interpreter that tag is the one the layout requires, which a real
    /// update would prepare an interpreter to match, so the check needs nothing
    /// prepared. The note it prints says so, and the folders agree; the same host
    /// DOES reach `uv` for the install, so an unrun stand-in means the check
    /// stayed out of it. The Gherkin scenario `update-09` drives the same case
    /// through the built binary.
    #[cfg(unix)]
    #[test]
    fn an_update_check_with_no_usable_python_reads_the_required_tag_and_prepares_nothing()
    -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-update-check-none")?;
        let requirement = python_requirement(SourceLayout::Canonical);

        let planned = host.plan_for_update_check()?;

        let compatibility = planned.wheel_compatibility()?;
        assert_eq!(compatibility.python_tag, requirement.tag);
        assert!(!compatibility.platform_tags.is_empty());
        assert_eq!(
            planned.update_note().as_deref(),
            Some(
                "no usable Python was found; updating would use uv to install managed Python 3.12 and save its path in the configuration, which this check did not do"
            )
        );
        assert!(!host.uv_was_run(), "the update check ran uv");
        assert_eq!(
            snapshot(&host.paths),
            Vec::<PathBuf>::new(),
            "the update check created files or folders under the config, data or cache folder"
        );

        // Control: the install on the same host does reach `uv`.
        assert!(host.python_for_install(false).is_err());
        assert!(
            host.uv_was_run(),
            "the stand-in uv is not reachable, so the checks above prove nothing"
        );
        Ok(())
    }

    /// A usable interpreter needs no note, and a managed Python whose tag cannot
    /// be read off its version says so in the update note as in the preview's.
    #[cfg(unix)]
    #[test]
    fn an_update_note_is_absent_for_a_usable_python_and_flags_an_unreadable_tag() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        let host = PythonHost::new("python-update-note")?;
        {
            let _unreadable = RestoredEnv::set(&[(
                "ROCM_CLI_MANAGED_PYTHON_VERSION",
                Some(Path::new("pypy@3.10")),
            )]);
            let planned = host.plan_for_update_check()?;
            let note = planned.update_note().expect("nothing was prepared");
            assert!(
                note.contains("managed Python pypy@3.10 (its wheel tag cannot be checked until it is installed; cp312 is required)")
                    && note.ends_with("which this check did not do"),
                "{note}"
            );
        }

        write_fake_python_with_venv(&host.search_dir, "python3")?;
        let planned = host.plan_for_update_check()?;
        assert!(matches!(planned, PlannedPython::Ready(_)), "{planned:?}");
        assert_eq!(planned.update_note(), None);
        assert!(!host.uv_was_run());
        Ok(())
    }

    /// The version-to-tag reading the preview's refusal rests on, and that it is
    /// the same rule `python_launcher_install_ready` applies (same message).
    #[test]
    fn a_managed_python_version_is_read_as_the_wheel_tag_it_reports() {
        assert_eq!(python_tag_for_version("3.12").as_deref(), Some("cp312"));
        assert_eq!(python_tag_for_version(" 3.14.1 ").as_deref(), Some("cp314"));
        assert_eq!(python_tag_for_version("3.9").as_deref(), Some("cp39"));
        // Spellings `uv` accepts for the same CPython release read the same.
        assert_eq!(python_tag_for_version("3.13t").as_deref(), Some("cp313"));
        assert_eq!(python_tag_for_version("3.13.1").as_deref(), Some("cp313"));
        assert_eq!(python_tag_for_version("3.13.*").as_deref(), Some("cp313"));
        assert_eq!(
            python_tag_for_version("cpython-3.13").as_deref(),
            Some("cp313")
        );
        assert_eq!(
            python_tag_for_version("cpython@3.13").as_deref(),
            Some("cp313")
        );
        assert_eq!(
            python_tag_for_version("cpython-3.12.4").as_deref(),
            Some("cp312")
        );
        // Anything else: only installing it would say what it reports.
        for unreadable in [
            "",
            "3",
            "3.x",
            "3.*",
            "*",
            "latest",
            "pypy@3.10",
            "3.13rc1",
            ">=3.12",
            "cpython-",
        ] {
            assert_eq!(python_tag_for_version(unreadable), None, "{unreadable:?}");
        }

        let required = python_requirement(SourceLayout::Canonical);
        assert!(ensure_python_tag_matches("cp312", required).is_ok());
        let error = ensure_python_tag_matches("cp313", required)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "Python wheel tag cp313 is not supported by this ROCm install; cp312 is required"
        );
    }

    /// An interpreter that has not been prepared yet is described by the one the
    /// layout requires: that is what a real install insists on (see
    /// `python_launcher_install_ready`), so it is what wheels must be chosen for.
    #[test]
    fn an_unprepared_python_is_described_by_the_interpreter_the_layout_requires() -> Result<()> {
        if current_platform_wheel_tags().is_err() {
            return Ok(());
        }
        for (layout, tag, version) in [
            (SourceLayout::Canonical, "cp312", "3.12"),
            (SourceLayout::Next, "cp314", "3.14"),
        ] {
            let requirement = python_requirement(layout);
            let planned = PlannedPython::WouldPrepareManaged {
                version: requirement.version.to_owned(),
                requirement,
                tag_check: TagCheck::Checked,
            };

            let compatibility = planned.wheel_compatibility()?;
            assert_eq!(compatibility.python_tag, tag);
            assert!(compatibility.platform_tags.contains(&"any".to_owned()));
            assert_eq!(planned.source(), PythonSource::Managed);
            assert_eq!(planned.source().as_str(), "managed");
            assert_eq!(planned.launcher_display(), "not prepared by this preview");
            assert_eq!(planned.venv_python_arg(), PathBuf::from(version));
            assert!(
                planned
                    .note()
                    .is_some_and(|note| note.contains(&format!("managed Python {version}"))),
                "{planned:?}"
            );
            assert!(
                planned.into_launcher().is_err(),
                "an install must never proceed on a Python that was only planned"
            );
        }
        Ok(())
    }

    /// A ready interpreter prints as it always did: its path and its source, no
    /// note.
    #[test]
    fn a_ready_python_prints_its_path_and_source() {
        let planned = PlannedPython::Ready(PythonLauncher {
            executable: PathBuf::from("/usr/bin/python3"),
            source: PythonSource::Path,
        });
        assert_eq!(planned.launcher_display(), "/usr/bin/python3");
        assert_eq!(planned.source().as_str(), "path");
        assert_eq!(planned.venv_python_arg(), PathBuf::from("/usr/bin/python3"));
        assert!(planned.note().is_none());
        assert_eq!(
            planned.into_launcher().unwrap().executable,
            PathBuf::from("/usr/bin/python3")
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
    /// `UV_VENV_CLEAR=1`. Then it creates a working interpreter there, which
    /// reports `$4` as its `sys.prefix`. So a
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
  cat > "$target/bin/python" <<PY
#!/bin/sh
if [ "\$1" = "-c" ]; then
  case "\$2" in *sys.prefix*) echo "$target" ;; *) echo cp312 ;; esac
  exit 0
fi
echo Python 3.12.10
PY
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
            error.contains(&format!(
                "(found a Python at {} with no pyvenv.cfg beside it;",
                venv_python_path(&home).display()
            )),
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
        assert!(error.contains("Choose an empty folder, or one where a ROCm CLI install finished"));
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
        assert_eq!(
            line,
            &format!(
                "Existing Python environment at {} cannot be used as it is (found a Python at {} that does not run); rebuilding only its venv files (kept: .rocm-cli-runtime.json, apps).",
                install_root.display(),
                venv_python_path(&install_root).display()
            )
        );
        for kept in [".rocm-cli-runtime.json", "apps"] {
            assert!(install_root.join(kept).exists(), "{kept} was reported kept");
        }

        fs::remove_dir_all(root).ok();
    }

    /// A rebuild that fails part-way must not leave a folder the next run
    /// treats as fresh. Here a read-only directory inside `lib` stops the
    /// clear after `bin` is gone; the next run, with the directory deletable
    /// again, has no Python to test, and must still go through the
    /// ownership-checked rebuild and `--allow-existing` rather than a plain
    /// `uv venv`, which uv 0.7 answers by deleting the folder.
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
                .any(|line| line.contains(&format!(
                    "cannot be used as it is (found pyvenv.cfg but no Python at {}); rebuilding only its venv files",
                    venv_python_path(&install_root).display()
                ))),
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
            let _ = remove_venv_entry(&install_root.join("bin"));
            fs::write(install_root.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
            fs::create_dir_all(&locked).unwrap();
            fs::write(locked.join("held.py"), b"held").unwrap();
        } else {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    /// The same partial-clear sequence against real uv releases, which differ
    /// in exactly the way that matters here: given a plain `uv venv` over a
    /// leftover venv, 0.7 deletes the folder, 0.9 builds over it, and 0.10
    /// refuses. It then drives every `ClearStop` through the real uv, where a
    /// `bin/python` left without `pyvenv.cfg` really does run as the base
    /// interpreter. Opt-in, since CI has no pinned uv set:
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
            assert_every_clear_stop_converges(&uv, &log, &python, &label.to_string());
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

    /// A clear that cannot look at a venv entry stops with an error. Only an
    /// entry that is not there counts as removed: an entry it could not
    /// inspect (here, a folder that can be listed but not searched) may still
    /// be on disk, and handing that to `uv venv --allow-existing` would build
    /// over a stale venv while reporting a clean rebuild.
    #[test]
    #[cfg(unix)]
    fn a_clear_that_cannot_inspect_a_venv_entry_stops_before_uv() {
        use std::os::unix::fs::PermissionsExt;
        let (root, paths) = venv_rebuild_sandbox("uninspectable");
        let install_root = managed_runtime_root(&paths, "wheel", "release-wheel-x");
        plant_failing_venv_python(&install_root);
        fs::write(install_root.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let launcher = cp312_launcher(&root);
        // Listable (r) but not searchable (no x): names read, entries do not stat.
        fs::set_permissions(&install_root, fs::Permissions::from_mode(0o644)).unwrap();
        let restore = || {
            fs::set_permissions(&install_root, fs::Permissions::from_mode(0o755)).unwrap();
        };
        if fs::symlink_metadata(install_root.join("bin")).is_ok() {
            // Permissions are not enforced (running as root): nothing to show.
            restore();
            fs::remove_dir_all(root).ok();
            return;
        }

        let result = ensure_uv_venv(&paths, &uv, &launcher, &install_root);
        restore();
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(&format!(
                "failed to inspect {}",
                install_root.join("bin").display()
            )),
            "{error}"
        );
        assert!(
            !uv_log.exists(),
            "uv must not run over an entry the clear could not inspect"
        );
        assert!(
            venv_python_path(&install_root).is_file(),
            "the entry is still there"
        );
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
        let error = venv_folder_state(&paths, spelled, "files")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(": /proc/self/cwd is a protected system location"),
            "{error}"
        );

        let link = root.join("looks-like-mine");
        std::os::unix::fs::symlink("/usr/share", &link).unwrap();
        let error = venv_folder_state(&paths, &link, "files")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(": /usr/share is a protected system location"),
            "{error}"
        );
        assert!(error.contains("Nothing was changed"), "{error}");

        fs::remove_dir_all(root).ok();
    }

    /// The `--dry-run` lines `install_wheel_runtime` prints for the venv step,
    /// for every state an install folder can be in, and the guarantee they rest
    /// on: a preview runs nothing from the install folder. Every folder here
    /// holds a Python that records each time it is run, so the last assertion
    /// fails if any branch of the preview executes it.
    ///
    /// Where the files decide alone, one outcome is printed: a fresh venv, a
    /// rebuild with `--allow-existing`, or a refusal with no command. Where only
    /// running the Python could tell (a folder with a Python and `pyvenv.cfg`),
    /// the preview says it could not tell and prints both outcomes.
    #[test]
    #[cfg(unix)]
    fn a_dry_run_reads_the_install_folder_and_runs_nothing_from_it() {
        let (root, paths) = venv_rebuild_sandbox("preview");
        let launcher = cp312_launcher(&root);
        let ran = root.join("prefix-python-ran.log");
        let install_args = vec!["pip".to_owned(), "install".to_owned(), "rocm".to_owned()];
        let venv_command = |folder: &Path, allow: &str| {
            format!(
                "UV_VENV_CLEAR=0 uv venv --python {} {}{allow} && uv pip install rocm",
                launcher.display(),
                folder.display()
            )
        };
        let plan = |folder: &Path| {
            let plan = dry_run_venv_plan(&paths, &launcher, folder, "cp312", &install_args);
            let lines = plan
                .lines
                .iter()
                .map(|(key, value)| format!("{key}: {value}"))
                .collect::<Vec<_>>();
            (plan.python_env, lines)
        };
        let reuse = "reuse the Python environment here if its Python runs, reports wheel tag cp312 and has this folder as its sys.prefix; this preview runs nothing from the folder, so it cannot tell whether it does";
        let rebuild = "rebuild only the venv files ROCm CLI created here: remove bin, Scripts, lib, lib64, Lib, include, Include, CACHEDIR.TAG, pyvenv.cfg first, which the command does not show, then create the venv with --allow-existing; everything else in the folder is kept";

        // Missing folder: a fresh venv.
        let empty = root.join("missing");
        let (env, lines) = plan(&empty);
        assert_eq!(env, "create a new Python environment in an empty folder");
        assert_eq!(lines, [format!("command: {}", venv_command(&empty, ""))]);

        // Someone's folder with a Python but no pyvenv.cfg: certainly refused.
        let home = root.join("home");
        plant_recording_venv_python(&home, &home, &ran);
        let (env, lines) = plan(&home);
        assert_eq!(env, "the install will refuse this folder");
        assert_eq!(
            lines[0],
            "command: none; the install will refuse this folder"
        );
        assert!(
            lines[1].starts_with(&format!(
                "refusal: refusing to create the Python environment at {}: the folder is not empty and ROCm CLI cannot show it holds a ROCm CLI install (found a Python at {} with no pyvenv.cfg beside it; no finished ROCm CLI install recorded here",
                home.display(),
                venv_python_path(&home).display()
            )),
            "{lines:?}"
        );
        assert_eq!(lines.len(), 2, "{lines:?}");

        // Someone's venv (a downloaded or cloned tree): reuse is undecidable
        // without running it, and the install refuses it otherwise.
        let cloned = root.join("cloned");
        plant_recording_venv_python(&cloned, &cloned, &ran);
        fs::write(cloned.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        let (env, lines) = plan(&cloned);
        assert_eq!(
            env,
            format!("{reuse}; if not, the install will refuse this folder")
        );
        assert_eq!(lines[0], "command: uv pip install rocm");
        assert!(
            lines[1].starts_with(&format!(
                "refusal_if_not_reused: refusing to create the Python environment at {}: the folder is not empty and ROCm CLI cannot show it holds a ROCm CLI install (found a Python at {} that this preview did not run;",
                cloned.display(),
                venv_python_path(&cloned).display()
            )),
            "{lines:?}"
        );
        assert_eq!(lines.len(), 2, "{lines:?}");

        // A venv ROCm CLI made, intact on disk: reuse or a rebuild in place.
        let owned = root.join("owned");
        plant_rocm_owned_venv_markers(&owned);
        plant_recording_venv_python(&owned, &owned, &ran);
        let (env, lines) = plan(&owned);
        assert_eq!(env, format!("{reuse}; if not, {rebuild}"));
        assert_eq!(
            lines,
            [
                "command: uv pip install rocm".to_owned(),
                format!(
                    "command_if_rebuilt: {}",
                    venv_command(&owned, " --allow-existing")
                ),
            ]
        );

        // A venv ROCm CLI made whose clear stopped after pyvenv.cfg: certainly
        // rebuilt, and the line names what was found.
        let broken = root.join("broken");
        plant_rocm_owned_venv_markers(&broken);
        plant_recording_venv_python(&broken, &broken, &ran);
        fs::remove_file(broken.join("pyvenv.cfg")).unwrap();
        let (env, lines) = plan(&broken);
        assert_eq!(
            env,
            format!(
                "{rebuild} (found a Python at {} with no pyvenv.cfg beside it)",
                venv_python_path(&broken).display()
            )
        );
        assert_eq!(
            lines,
            [format!(
                "command: {}",
                venv_command(&broken, " --allow-existing")
            )]
        );

        assert!(
            !ran.exists(),
            "a preview ran a Python from the install folder:\n{}",
            fs::read_to_string(&ran).unwrap_or_default()
        );
        for folder in [&home, &cloned, &owned, &broken] {
            assert!(
                venv_python_path(folder).is_file(),
                "a preview changed {}",
                folder.display()
            );
        }
        assert!(cloned.join("pyvenv.cfg").is_file() && owned.join("pyvenv.cfg").is_file());
        fs::remove_dir_all(root).ok();
    }

    /// The interpreter search a preview runs is the other way a preview could
    /// run something from the install folder: the folder's `bin` on `PATH`, or
    /// `ROCM_CLI_PYTHON` naming a Python in it. A preview refuses there rather
    /// than run it, and says so; the same search for an install, or a preview
    /// with no `--prefix`, still runs it, which is what shows the refusal comes
    /// from the guard and not from the Python.
    #[test]
    #[cfg(unix)]
    fn a_dry_run_runs_no_python_from_the_install_folder_while_searching() {
        let _lock = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, paths) = venv_rebuild_sandbox("preview-search");
        let ran = root.join("prefix-python-ran.log");
        let prefix = root.join("prefix");
        plant_recording_venv_python(&prefix, &prefix, &ran);
        let bin = prefix.join("bin");
        fs::rename(bin.join("python"), bin.join("python3")).unwrap();
        let requirement = python_requirement(SourceLayout::Canonical);
        let on_path = PythonResolverEnv {
            python_override: None,
            search_dirs: vec![bin.clone()],
            managed_python_version: None,
            managed_bootstrap_disabled: false,
        };
        let named = |value: &str| PythonResolverEnv {
            python_override: Some(value.to_owned()),
            ..on_path.clone()
        };
        let in_prefix = bin.join("python3").display().to_string();

        for (env, source) in [
            (on_path.clone(), "PATH"),
            (named(&in_prefix), "ROCM_CLI_PYTHON"),
            (named("python3"), "ROCM_CLI_PYTHON"),
        ] {
            let error = plan_python_launcher_in(&paths, &env, requirement, Some(&prefix))
                .unwrap_err()
                .to_string();
            assert!(
                error.starts_with(&format!(
                    "this preview did not check the Python from {source} at "
                )) && error.contains(&format!(
                    "inside the install folder {}, and a preview runs nothing from the install folder",
                    prefix.display()
                )),
                "{error}"
            );
            assert!(!ran.exists(), "the preview ran {in_prefix} ({source})");
        }

        // A folder beside the prefix is not inside it.
        let beside = root.join("prefix-other");
        assert!(
            plan_python_launcher_in(&paths, &on_path, requirement, Some(&beside)).is_ok()
                && ran.exists(),
            "with the Python outside the install folder, the preview checks it as before"
        );
        fs::remove_file(&ran).unwrap();
        assert!(
            plan_python_launcher_in(&paths, &on_path, requirement, None).is_ok() && ran.exists(),
            "without a --prefix, the preview checks the PATH Python as before"
        );
        fs::remove_dir_all(root).ok();
    }

    /// Each condition a venv must meet to be reused is named when it is the
    /// one that failed, in the refusal of a folder ROCm CLI did not make, so a
    /// user is not told a Python "does not run" when it runs fine and only its
    /// `sys.prefix` (a conda env given as `--prefix`) or `pyvenv.cfg` is wrong.
    #[test]
    #[cfg(unix)]
    fn a_refusal_names_the_condition_that_kept_the_venv_from_being_reused() {
        let (root, paths) = venv_rebuild_sandbox("reuse-reasons");
        let launcher = cp312_launcher(&root);
        let (uv, uv_log) = write_logging_fake_uv(&root);
        let ran = root.join("ran.log");
        let elsewhere = root.join("conda-base");
        let cfg =
            |folder: &Path| fs::write(folder.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();

        for label in ["does-not-run", "no-cfg", "foreign-prefix", "wrong-tag"] {
            let folder = root.join(label);
            let python = venv_python_path(&folder).display().to_string();
            let expected = match label {
                "does-not-run" => {
                    plant_failing_venv_python(&folder);
                    cfg(&folder);
                    format!("a Python at {python} that does not run")
                }
                "no-cfg" => {
                    plant_recording_venv_python(&folder, &folder, &ran);
                    format!("a Python at {python} with no pyvenv.cfg beside it")
                }
                "foreign-prefix" => {
                    plant_recording_venv_python(&folder, &elsewhere, &ran);
                    cfg(&folder);
                    format!(
                        "a Python at {python} whose sys.prefix is {}, not this folder",
                        elsewhere.display()
                    )
                }
                _ => {
                    plant_recording_venv_python(&folder, &folder, &ran);
                    // Same Python, but reporting the cp311 tag.
                    let script = fs::read_to_string(&python)
                        .unwrap()
                        .replace("cp312", "cp311");
                    fs::write(&python, script).unwrap();
                    cfg(&folder);
                    format!(
                        "a Python at {python} that reports wheel tag cp311 where cp312 is required"
                    )
                }
            };
            let error = ensure_uv_venv(&paths, &uv, &launcher, &folder)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("(found {expected};")),
                "{label}: {error}"
            );
            assert!(!error.contains("does not match"), "{label}: {error}");
        }
        assert!(!uv_log.exists(), "uv must not run after a refusal");
        fs::remove_dir_all(root).ok();
    }

    /// A venv Python that appends a line to `ran` every time it is run, then
    /// behaves as [`plant_working_venv_python`]'s does: runs, reports `cp312`,
    /// and reports `prefix` as its `sys.prefix`, and creates a venv when asked.
    #[cfg(unix)]
    fn plant_recording_venv_python(install_root: &Path, prefix: &Path, ran: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let python = venv_python_path(install_root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(
            &python,
            format!(
                "#!/bin/sh\necho \"$0 $*\" >> \"{}\"\nif [ \"$1\" = \"-m\" ]; then /bin/mkdir -p \"$3\"; exit 0; fi\nif [ \"$1\" = \"-c\" ]; then\n  case \"$2\" in *sys.prefix*) echo \"{}\" ;; *) echo cp312 ;; esac\n  exit 0\nfi\necho Python 3.12.10\n",
                ran.display(),
                prefix.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// A venv Python that runs, reports the `cp312` tag, and reports `prefix`
    /// as its `sys.prefix`, the way a real one does with or without the
    /// `pyvenv.cfg` that decides what that prefix really is.
    #[cfg(unix)]
    fn plant_working_venv_python(install_root: &Path, prefix: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let python = venv_python_path(install_root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&python);
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then\n  case \"$2\" in *sys.prefix*) echo \"{}\" ;; *) echo cp312 ;; esac\n  exit 0\nfi\necho Python 3.12.10\n",
                prefix.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// Points where a clear of an owned venv can stop, as the states they
    /// leave. Each must converge: the next run ends with a working venv whose
    /// `pyvenv.cfg` is there and whose `sys.prefix` is the folder, keeps the
    /// ComfyUI data, and the run after that reuses it without calling uv.
    #[cfg(unix)]
    #[derive(Debug, Clone, Copy)]
    enum ClearStop {
        /// Only `pyvenv.cfg` removed: on Linux the `bin/python` symlink now
        /// runs as the base interpreter.
        Cfg,
        /// `pyvenv.cfg` and `CACHEDIR.TAG` removed.
        CfgAndCacheTag,
        /// Part of `bin` removed, the interpreter among it.
        PartOfBinWithPython,
        /// Part of `bin` removed, the interpreter still there.
        PartOfBinKeepingPython,
        /// All of `bin` removed.
        WholeBin,
    }

    #[cfg(unix)]
    const CLEAR_STOPS: [ClearStop; 5] = [
        ClearStop::Cfg,
        ClearStop::CfgAndCacheTag,
        ClearStop::PartOfBinWithPython,
        ClearStop::PartOfBinKeepingPython,
        ClearStop::WholeBin,
    ];

    #[cfg(unix)]
    fn apply_clear_stop(install_root: &Path, stop: ClearStop) {
        let remove = |name: &str| {
            let _ = remove_venv_entry(&install_root.join(name));
        };
        match stop {
            ClearStop::Cfg => remove("pyvenv.cfg"),
            ClearStop::CfgAndCacheTag => {
                remove("pyvenv.cfg");
                remove("CACHEDIR.TAG");
            }
            ClearStop::PartOfBinWithPython => {
                for entry in fs::read_dir(install_root.join("bin")).unwrap() {
                    let path = entry.unwrap().path();
                    if path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with("python")
                    {
                        let _ = remove_venv_entry(&path);
                    }
                }
            }
            ClearStop::PartOfBinKeepingPython => {
                let _ = remove_venv_entry(&install_root.join("bin").join("activate"));
            }
            ClearStop::WholeBin => remove("bin"),
        }
    }

    /// Drive every [`ClearStop`] through `ensure_uv_venv` with `uv` (which
    /// appends its arguments to `log`) and assert convergence.
    #[cfg(unix)]
    fn assert_every_clear_stop_converges(uv: &Path, log: &Path, launcher: &Path, label: &str) {
        for stop in CLEAR_STOPS {
            let what = format!("{label} {stop:?}");
            let (root, paths) = venv_rebuild_sandbox("stop");
            let install_root = root.join("prefix");
            ensure_uv_venv(&paths, uv, launcher, &install_root)
                .unwrap_or_else(|e| panic!("{what}: initial venv: {e:#}"));
            write_runtime_marker(&install_root, &install_root, "wheel");
            fs::write(install_root.join("CACHEDIR.TAG"), b"Signature").unwrap();
            fs::write(install_root.join("bin").join("activate"), b"# activate").unwrap();
            let models = install_root.join("apps").join("comfyui").join("models");
            fs::create_dir_all(&models).unwrap();
            fs::write(models.join("x.safetensors"), b"weights").unwrap();
            apply_clear_stop(&install_root, stop);

            fs::remove_file(log).ok();
            ensure_uv_venv(&paths, uv, launcher, &install_root)
                .unwrap_or_else(|e| panic!("{what}: {e:#}"));
            assert!(
                install_root.join("pyvenv.cfg").is_file(),
                "{what}: no pyvenv.cfg"
            );
            let prefix = capture_python_stdout(
                &venv_python_path(&install_root),
                "import sys; print(sys.prefix)",
                "probe",
            )
            .unwrap_or_else(|e| panic!("{what}: {e:#}"));
            assert!(
                paths_name_same_folder(Path::new(prefix.trim()), &install_root),
                "{what}: the venv Python runs as {prefix}, not as the venv"
            );
            assert_eq!(
                fs::read(models.join("x.safetensors")).ok().as_deref(),
                Some(&b"weights"[..]),
                "{what}: ComfyUI data lost"
            );
            if let Ok(calls) = fs::read_to_string(log) {
                assert!(
                    calls
                        .lines()
                        .filter(|line| line.starts_with("venv "))
                        .all(|line| line.ends_with(" --allow-existing")),
                    "{what}: {calls}"
                );
            }

            fs::remove_file(log).ok();
            ensure_uv_venv(&paths, uv, launcher, &install_root)
                .unwrap_or_else(|e| panic!("{what}: second run: {e:#}"));
            assert!(
                !log.exists(),
                "{what}: the converged venv was not reused: {:?}",
                fs::read_to_string(log)
            );
            fs::remove_dir_all(root).ok();
        }
    }

    /// The reuse check alone already refuses a Python without `pyvenv.cfg`;
    /// this order is the second layer `VENV_OWNED_ENTRIES` documents, and no
    /// convergence test can see it because the first layer holds. Pin it here.
    #[test]
    fn a_clear_removes_the_interpreter_first_and_pyvenv_cfg_last() {
        let position = |name: &str| {
            VENV_OWNED_ENTRIES
                .iter()
                .position(|entry| *entry == name)
                .unwrap_or_else(|| panic!("{name} is not a venv-owned entry"))
        };
        assert_eq!(position("bin"), 0, "{VENV_OWNED_ENTRIES:?}");
        assert!(
            position("Scripts") < position("lib"),
            "{VENV_OWNED_ENTRIES:?}"
        );
        assert_eq!(
            VENV_OWNED_ENTRIES.last(),
            Some(&"pyvenv.cfg"),
            "{VENV_OWNED_ENTRIES:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn every_point_a_clear_can_stop_at_converges_on_the_next_run() {
        let (root, _) = venv_rebuild_sandbox("stops-fake");
        let (uv, log) = write_logging_fake_uv(&root);
        let launcher = cp312_launcher(&root);
        assert_every_clear_stop_converges(&uv, &log, &launcher, "fake uv");
        fs::remove_dir_all(root).ok();
    }

    /// Every shape a folder can have when `ensure_uv_venv` meets it with
    /// entries in it: what is left of a venv, what its runtime marker says,
    /// and whether it sits in the managed runtimes folder. The expected
    /// outcome of each is written out, not computed. Every venv-shaped case
    /// also holds user data at paths listed literally here; a rebuild must keep
    /// all of them, and a refusal or a reuse must change nothing.
    #[test]
    #[cfg(unix)]
    fn venv_rebuild_deletes_only_venv_files_and_only_from_an_owned_folder() {
        #[derive(Debug, Clone, Copy)]
        enum Venv {
            /// `pyvenv.cfg`, `lib`, and a Python that fails `--version`.
            BrokenPython,
            /// `lib` and a Python that runs with the right tag, but no
            /// `pyvenv.cfg`: on Linux that is the base interpreter.
            PythonWithoutCfg,
            /// `pyvenv.cfg`, `lib`, and a Python that runs with the right tag
            /// but whose `sys.prefix` is another folder.
            ForeignPrefix,
            /// `pyvenv.cfg` and `lib`, no Python.
            NoPython,
            /// Only `lib`: a clear that removed `bin` and then failed.
            Leftover,
            /// `pyvenv.cfg`, `lib`, and a Python that runs as this venv.
            Working,
            /// Nothing but this one entry.
            Only(&'static str),
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
            Reused,
        }
        use Expect::{Rebuilt, Refused, Reused};
        use Marker::{Garbage, Missing, OtherFolder, TarballHere, WheelHere};
        use Venv::{
            BrokenPython, ForeignPrefix, Leftover, NoPython, Only, PythonWithoutCfg, Working,
        };
        #[rustfmt::skip]
        let table = [
            // venv shape        marker       managed  expected
            (BrokenPython,     Missing,     false,   Refused),
            (BrokenPython,     Garbage,     false,   Refused),
            (BrokenPython,     OtherFolder, false,   Refused),
            (BrokenPython,     TarballHere, false,   Refused),
            (BrokenPython,     WheelHere,   false,   Rebuilt),
            (BrokenPython,     Missing,     true,    Rebuilt),
            (BrokenPython,     Garbage,     true,    Rebuilt),
            (BrokenPython,     OtherFolder, true,    Rebuilt),
            (BrokenPython,     TarballHere, true,    Rebuilt),
            (BrokenPython,     WheelHere,   true,    Rebuilt),
            (PythonWithoutCfg, Missing,     false,   Refused),
            (PythonWithoutCfg, Garbage,     false,   Refused),
            (PythonWithoutCfg, OtherFolder, false,   Refused),
            (PythonWithoutCfg, TarballHere, false,   Refused),
            (PythonWithoutCfg, WheelHere,   false,   Rebuilt),
            (PythonWithoutCfg, Missing,     true,    Rebuilt),
            (PythonWithoutCfg, Garbage,     true,    Rebuilt),
            (PythonWithoutCfg, OtherFolder, true,    Rebuilt),
            (PythonWithoutCfg, TarballHere, true,    Rebuilt),
            (PythonWithoutCfg, WheelHere,   true,    Rebuilt),
            (ForeignPrefix,    Missing,     false,   Refused),
            (ForeignPrefix,    Garbage,     false,   Refused),
            (ForeignPrefix,    OtherFolder, false,   Refused),
            (ForeignPrefix,    TarballHere, false,   Refused),
            (ForeignPrefix,    WheelHere,   false,   Rebuilt),
            (ForeignPrefix,    Missing,     true,    Rebuilt),
            (ForeignPrefix,    Garbage,     true,    Rebuilt),
            (ForeignPrefix,    OtherFolder, true,    Rebuilt),
            (ForeignPrefix,    TarballHere, true,    Rebuilt),
            (ForeignPrefix,    WheelHere,   true,    Rebuilt),
            (NoPython,         Missing,     false,   Refused),
            (NoPython,         Garbage,     false,   Refused),
            (NoPython,         OtherFolder, false,   Refused),
            (NoPython,         TarballHere, false,   Refused),
            (NoPython,         WheelHere,   false,   Rebuilt),
            (NoPython,         Missing,     true,    Rebuilt),
            (NoPython,         Garbage,     true,    Rebuilt),
            (NoPython,         OtherFolder, true,    Rebuilt),
            (NoPython,         TarballHere, true,    Rebuilt),
            (NoPython,         WheelHere,   true,    Rebuilt),
            (Leftover,         Missing,     false,   Refused),
            (Leftover,         Garbage,     false,   Refused),
            (Leftover,         OtherFolder, false,   Refused),
            (Leftover,         TarballHere, false,   Refused),
            (Leftover,         WheelHere,   false,   Rebuilt),
            (Leftover,         Missing,     true,    Rebuilt),
            (Leftover,         Garbage,     true,    Rebuilt),
            (Leftover,         OtherFolder, true,    Rebuilt),
            (Leftover,         TarballHere, true,    Rebuilt),
            (Leftover,         WheelHere,   true,    Rebuilt),
            // A venv that works is reused wherever it is: nothing is deleted.
            (Working,          Missing,     false,   Reused),
            (Working,          WheelHere,   false,   Reused),
            (Working,          Missing,     true,    Reused),
            // A folder that holds anything at all, and no proof ROCm CLI
            // made it, is refused, as `main` refuses it.
            (Only("README.md"), Missing,    false,   Refused),
            (Only(".DS_Store"), Missing,    false,   Refused),
            (Only(".git"),      Missing,    false,   Refused),
        ];
        // Never part of a venv, so a rebuild must leave each exactly as it was.
        let user_paths: [&str; 4] = [
            "Documents/thesis.txt",
            "apps/comfyui/source/models/x",
            "share/notes/todo.txt",
            "work.py",
        ];
        let elsewhere = Path::new("/usr/local");
        for (venv, marker, managed, expected) in table {
            let shape = format!("{venv:?} {marker:?} managed={managed}");
            let (root, paths) = venv_rebuild_sandbox("shapes");
            let install_root = if managed {
                managed_runtime_root(&paths, "wheel", "release-wheel-k")
            } else {
                root.join("prefix")
            };
            fs::create_dir_all(&install_root).unwrap();
            let cfg = || fs::write(install_root.join("pyvenv.cfg"), "home = /x\n").unwrap();
            if let Only(entry) = venv {
                if entry == ".git" {
                    fs::create_dir_all(install_root.join(".git").join("objects")).unwrap();
                } else {
                    fs::write(install_root.join(entry), b"not a venv").unwrap();
                }
            } else {
                fs::create_dir_all(install_root.join("lib").join("python3.12")).unwrap();
                fs::write(
                    install_root.join("lib").join("python3.12").join("stale.py"),
                    b"old",
                )
                .unwrap();
                for user_path in user_paths {
                    let path = install_root.join(user_path);
                    fs::create_dir_all(path.parent().unwrap()).unwrap();
                    fs::write(&path, format!("user data: {user_path}")).unwrap();
                }
            }
            match venv {
                BrokenPython => {
                    plant_failing_venv_python(&install_root);
                    cfg();
                }
                PythonWithoutCfg => plant_working_venv_python(&install_root, &install_root),
                ForeignPrefix => {
                    plant_working_venv_python(&install_root, elsewhere);
                    cfg();
                }
                NoPython => cfg(),
                Working => {
                    plant_working_venv_python(&install_root, &install_root);
                    cfg();
                }
                Leftover | Only(_) => {}
            }
            match marker {
                Missing => {}
                Garbage => fs::write(install_root.join(".rocm-cli-runtime.json"), "{}").unwrap(),
                OtherFolder => write_runtime_marker(&install_root, &root.join("other"), "wheel"),
                TarballHere => write_runtime_marker(&install_root, &install_root, "tarball"),
                WheelHere => write_runtime_marker(&install_root, &install_root, "wheel"),
            }
            let (uv, uv_log) = write_logging_fake_uv(&root);
            let launcher = cp312_launcher(&root);
            let before = tree_snapshot(&install_root);

            let result = ensure_uv_venv(&paths, &uv, &launcher, &install_root);

            assert!(root.join("sentinel").is_file(), "{shape}");
            match expected {
                Rebuilt => {
                    assert!(result.is_ok(), "{shape}: {result:?}");
                    for user_path in user_paths {
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
                    assert!(install_root.join("pyvenv.cfg").is_file(), "{shape}");
                    assert!(venv_python_path(&install_root).is_file(), "{shape}");
                }
                Refused | Reused => {
                    assert_eq!(result.is_ok(), expected == Reused, "{shape}: {result:?}");
                    assert_eq!(
                        tree_snapshot(&install_root),
                        before,
                        "{shape}: the folder changed"
                    );
                    assert!(!uv_log.exists(), "{shape}: uv ran");
                }
            }
            fs::remove_dir_all(root).ok();
        }
    }
}
