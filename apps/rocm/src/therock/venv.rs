// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The runtime's virtual environment in an install folder: reusing a real venv
//! for that folder with the required interpreter tag, creating one in a missing
//! or empty folder, rebuilding only the venv's own entries in a folder ROCm CLI
//! can show it made, and refusing any other non-empty folder
//! (`venv_folder_state`). `dry_run_venv_plan` is the `--dry-run` view of the
//! same decision, read from the folder's files without running anything in it.
//!
//! Split out of `therock/python.rs`, whose interpreter lookup and wheel tags it
//! builds on; nothing there depends on this module.

use super::process::{capture_python_stdout, progress_line, run_command, run_command_with_env};
use super::python::wheel_compatibility_for_python;
use super::{managed_format_root, venv_python_path};
use crate::shell_quote::shell_quote;
use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, normalize_runtime_path_for_host, uv_command_env, uv_venv_args};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

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
            "a Python at {} whose sys.prefix is {}, not this folder",
            env_python.display(),
            shown_from_folder(prefix)
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
/// ([`existing_venv_reuse`]) and sends the folder back through
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
    let managed = rocm_core::resolve_path_through_symlinks(&managed_format_root(paths, "wheel"));
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
            shown_from_folder(&other.display().to_string())
        ),
        MarkerFinding::OtherFormat(format) => {
            format!(
                "a .rocm-cli-runtime.json from a {} install, not a Python environment",
                shown_from_folder(format)
            )
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

/// Text read from the install folder (a marker's recorded path or format, a
/// Python's `sys.prefix`, the names of the entries kept), made safe to print.
/// The folder may be anyone's, and these lines reach a terminal and, through
/// `install_sdk_dry_run`, the assistant's model; a terminal control, format or
/// line-separator character in them is shown as its `\u{..}` escape instead of
/// acting, the same set the tool-argument check refuses.
fn shown_from_folder(text: &str) -> String {
    text.chars()
        .map(|character| {
            if crate::shell_quote::has_terminal_control_character(
                character.encode_utf8(&mut [0; 4]),
            ) {
                character.escape_unicode().to_string()
            } else {
                character.to_string()
            }
        })
        .collect()
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
        format!("kept: {}", shown_from_folder(&kept.join(", ")))
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
    /// Whether reuse of the folder's venv is the one outcome the files cannot
    /// decide: a Python and `pyvenv.cfg` are there, and only running that Python
    /// could tell. Anything that depends on what the venv holds is unknown then,
    /// and known (nothing reused) in every other plan.
    pub(super) reuse_undecided: bool,
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
/// A plan that shows a `uv venv` command also prints an `env:` line naming
/// `UV_VENV_CLEAR=0`, which the install sets for `uv venv`; it is a line of its
/// own rather than a `VAR=value` prefix, which only POSIX shells accept. Every
/// argument is shell-quoted: the install root comes from a
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
            "uv {} && uv {}",
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
            reuse_undecided: false,
            lines: vec![("command", with_venv(false)), ("env", uv_venv_env())],
        },
        (Some(found), Ok(VenvFolder::Owned)) => DryRunVenvPlan {
            python_env: format!("{rebuild} (found {found})"),
            reuse_undecided: false,
            lines: vec![("command", with_venv(true)), ("env", uv_venv_env())],
        },
        (None, Ok(VenvFolder::Owned)) => DryRunVenvPlan {
            python_env: format!("{reuse}; if not, {rebuild}"),
            reuse_undecided: true,
            lines: vec![
                ("command", install),
                ("command_if_rebuilt", with_venv(true)),
                ("env", uv_venv_env()),
            ],
        },
        (Some(_), Err(refusal)) => DryRunVenvPlan {
            python_env: "the install will refuse this folder".to_owned(),
            reuse_undecided: false,
            lines: vec![
                ("command", "none".to_owned()),
                ("refusal", format!("{refusal:#}")),
            ],
        },
        (None, Err(refusal)) => DryRunVenvPlan {
            python_env: format!("{reuse}; if not, the install will refuse this folder"),
            reuse_undecided: true,
            lines: vec![
                ("command", install),
                ("refusal_if_not_reused", format!("{refusal:#}")),
            ],
        },
    }
}

/// The `env:` line beside a previewed `uv venv` command: what the install sets
/// for it, and why, in a form that does not depend on the user's shell.
pub(super) fn uv_venv_env() -> String {
    "UV_VENV_CLEAR=0 for `uv venv`, as the install sets it so that a UV_VENV_CLEAR=1 in the environment cannot make uv clear the folder; set it before running the command".to_owned()
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::super::managed_runtime_root;
    #[cfg(unix)]
    use super::super::process::PROGRESS_LINE_SINK;
    use super::*;
    #[cfg(unix)]
    use crate::therock::test_support::{
        plant_recording_venv_python, venv_rebuild_sandbox, write_fake_python_reporting,
        write_fake_uv_creating_venv,
    };
    #[cfg(unix)]
    use rocm_core::unix_time_millis;

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

    #[cfg(unix)]
    fn plant_failing_venv_python(install_root: &Path) {
        let python = venv_python_path(install_root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&python);
        rocm_core::fake_executable::write_fake_executable(&python, "#!/bin/sh\nexit 1\n").unwrap();
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
  /bin/cat > "$target/bin/python" <<PY
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
        rocm_core::fake_executable::write_fake_executable(&path, script).unwrap();
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
            rocm_core::fake_executable::write_fake_executable(
                &uv,
                format!(
                    "#!/bin/sh\necho \"$@\" >> \"{}\"\nexec \"{}\" \"$@\"\n",
                    log.display(),
                    real_uv.display()
                ),
            )
            .unwrap();
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
                "uv venv --python {} {}{allow} && uv pip install rocm",
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
        // Beside every `uv venv` command, in place of a POSIX-only prefix.
        let env_line = "env: UV_VENV_CLEAR=0 for `uv venv`, as the install sets it so that a UV_VENV_CLEAR=1 in the environment cannot make uv clear the folder; set it before running the command".to_owned();
        let rebuild = "rebuild only the venv files ROCm CLI created here: remove bin, Scripts, lib, lib64, Lib, include, Include, CACHEDIR.TAG, pyvenv.cfg first, which the command does not show, then create the venv with --allow-existing; everything else in the folder is kept";

        // Missing folder: a fresh venv.
        let empty = root.join("missing");
        let (env, lines) = plan(&empty);
        assert_eq!(env, "create a new Python environment in an empty folder");
        assert_eq!(
            lines,
            [
                format!("command: {}", venv_command(&empty, "")),
                env_line.clone()
            ]
        );

        // Someone's folder with a Python but no pyvenv.cfg: certainly refused.
        let home = root.join("home");
        plant_recording_venv_python(&home, &home, &ran);
        let (env, lines) = plan(&home);
        assert_eq!(env, "the install will refuse this folder");
        assert_eq!(lines[0], "command: none");
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
                env_line.clone(),
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
            [
                format!("command: {}", venv_command(&broken, " --allow-existing")),
                env_line,
            ]
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

        // A planted Python with one answer of the recording one changed.
        let rewrite = |python: &str, from: &str, to: &str| {
            let script = fs::read_to_string(python).unwrap();
            assert!(script.contains(from), "{script}");
            rocm_core::fake_executable::write_fake_executable(
                Path::new(python),
                script.replace(from, to),
            )
            .unwrap();
        };

        for label in [
            "does-not-run",
            "no-cfg",
            "foreign-prefix",
            "no-prefix",
            "no-tag",
            "wrong-tag",
        ] {
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
                "no-prefix" => {
                    // Runs, but fails when asked for its sys.prefix.
                    plant_recording_venv_python(&folder, &folder, &ran);
                    rewrite(&python, "*sys.prefix*) echo", "*sys.prefix*) exit 1; echo");
                    cfg(&folder);
                    format!("a Python at {python} that does not report its sys.prefix")
                }
                "no-tag" => {
                    // Reports this folder as its sys.prefix, but fails when
                    // asked for its wheel tag.
                    plant_recording_venv_python(&folder, &folder, &ran);
                    rewrite(&python, "*) echo cp312 ;;", "*) exit 1 ;;");
                    cfg(&folder);
                    format!("a Python at {python} whose wheel tag could not be read")
                }
                _ => {
                    plant_recording_venv_python(&folder, &folder, &ran);
                    // Same Python, but reporting the cp311 tag.
                    rewrite(&python, "cp312", "cp311");
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

    /// A marker in someone else's folder is text that folder chose, and the
    /// refusal carries it to a terminal and, from a preview, to the
    /// assistant's model. A control or bidi character in it is printed as an
    /// escape, so it cannot clear the screen or reorder the line.
    #[test]
    #[cfg(unix)]
    fn a_refusal_prints_marker_text_with_control_characters_escaped() {
        let (root, paths) = venv_rebuild_sandbox("marker-text");
        let folder = root.join("cloned");
        plant_failing_venv_python(&folder);
        write_runtime_marker(
            &folder,
            Path::new("/elsewhere/\u{1b}[2J\u{202e}gpj"),
            "wheel",
        );
        let error = venv_folder_state(&paths, &folder, "files")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("belongs to /elsewhere/\\u{1b}[2J\\u{202e}gpj"),
            "{error}"
        );
        assert!(
            !crate::shell_quote::has_terminal_control_character(&error),
            "{error:?}"
        );
        assert_eq!(shown_from_folder("C:\\envs\\rocm"), "C:\\envs\\rocm");
        fs::remove_dir_all(root).ok();
    }

    /// A venv Python that runs, reports the `cp312` tag, and reports `prefix`
    /// as its `sys.prefix`, the way a real one does with or without the
    /// `pyvenv.cfg` that decides what that prefix really is.
    #[cfg(unix)]
    fn plant_working_venv_python(install_root: &Path, prefix: &Path) {
        let python = venv_python_path(install_root);
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&python);
        rocm_core::fake_executable::write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then\n  case \"$2\" in *sys.prefix*) echo \"{}\" ;; *) echo cp312 ;; esac\n  exit 0\nfi\necho Python 3.12.10\n",
                prefix.display()
            ),
        )
        .unwrap();
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
        /// Part of `bin` removed (`activate`), the interpreter and
        /// `pyvenv.cfg` still there. Nothing the venv needs is gone, so the
        /// next run must reuse it, not rebuild it: this is the stop that pins
        /// that a part-way clear of `bin` alone does not cost a rebuild.
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
            // Which way each stop converges: only the stop that left a whole
            // venv is reused; every other one is rebuilt through uv.
            let rebuilt = fs::read_to_string(log)
                .is_ok_and(|calls| calls.lines().any(|line| line.starts_with("venv ")));
            assert_eq!(
                rebuilt,
                !matches!(stop, ClearStop::PartOfBinKeepingPython),
                "{what}: rebuilt={rebuilt}"
            );
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

    /// The ownership and shape decisions read only files, so they are tested on
    /// every platform, the Windows CI job included, with `venv_python_path`'s
    /// own layout (`Scripts\\python.exe` there, `bin/python` elsewhere). The
    /// interpreter is an empty file nothing runs. Covers what the Unix tests
    /// drive through shell stand-ins: an empty folder, someone else's folder
    /// refused with the condition named and the folder untouched, a venv shape
    /// missing `pyvenv.cfg`, and a folder the wheel marker proves ROCm CLI made.
    #[test]
    fn the_venv_folder_decision_reads_files_on_every_platform() {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-venv-decision-{}-{}-{}",
            std::process::id(),
            unix_time_millis(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let plant_python = |folder: &Path| {
            let python = venv_python_path(folder);
            fs::create_dir_all(python.parent().unwrap()).unwrap();
            fs::write(&python, b"").unwrap();
            python
        };

        let missing = root.join("missing");
        assert_eq!(
            venv_folder_state(&paths, &missing, "files").unwrap(),
            VenvFolder::Empty
        );

        let cloned = root.join("cloned");
        let python = plant_python(&cloned);
        fs::write(cloned.join("notes.txt"), b"the user's").unwrap();
        let finding = venv_shape_finding(&cloned).expect("no pyvenv.cfg is certain");
        assert_eq!(
            finding,
            format!(
                "a Python at {} with no pyvenv.cfg beside it",
                python.display()
            )
        );
        let refusal = venv_folder_state(&paths, &cloned, &finding)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains(&format!("(found {finding};")), "{refusal}");
        assert!(refusal.contains("no .rocm-cli-runtime.json"), "{refusal}");
        assert_eq!(
            fs::read(cloned.join("notes.txt")).unwrap(),
            b"the user's",
            "a refusal changes nothing"
        );

        fs::write(cloned.join("pyvenv.cfg"), b"home = x").unwrap();
        assert_eq!(
            venv_shape_finding(&cloned),
            None,
            "both files: only running it could tell"
        );

        let owned = root.join("owned");
        plant_python(&owned);
        let marker = serde_json::json!({
            "runtime_key": "release-wheel-x",
            "format": "wheel",
            "install_root": owned,
        });
        fs::write(
            owned.join(".rocm-cli-runtime.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        assert_eq!(
            venv_folder_state(&paths, &owned, "files").unwrap(),
            VenvFolder::Owned
        );
        fs::remove_dir_all(root).ok();
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
