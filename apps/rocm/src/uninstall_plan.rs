// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The uninstall plan: what would be removed, how that plan is rendered for
//! review, and the filesystem-removal primitive used to carry it out.
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-212) of the `main.rs` modularization (see `docs/architecture.md`).
//! `dispatch()`'s call sites are unaffected: none of this module's functions
//! are called directly from `dispatch()`.
//!
//! This is deliberately a separate file from `uninstall.rs`, not folded into
//! it: `uninstall.rs` owns only what `rocm uninstall` itself does on top of
//! removing files — undoing the installer's PATH change and deleting the
//! running executable on Windows, which cannot unlink itself. `storage.rs`
//! builds and renders a removal plan of its own (`rocm storage
//! remove-downloads`) and reuses this module's plan type and its
//! file-removal helpers to do it, so keeping them in `uninstall.rs` would
//! make `storage.rs` depend on a module whose name and doc comment both
//! describe the `uninstall` command's own, exclusive logic.
//!
//! This module owns: the plan types (`UninstallOptions`, `UninstallPlan`,
//! `UninstallPlanEntry`), building the plan (`build_uninstall_plan`,
//! `plan_installed_binaries`, and the installed-binary/shared-cache
//! discovery helpers behind them), rendering it (`render_uninstall_dry_run`,
//! `render_uninstall_plan`), the `[y/N]` confirmation prompt
//! (`confirm_uninstall`), and the filesystem removal primitive both
//! `uninstall.rs` and `storage.rs` call to carry a plan out (`remove_path`,
//! and the entry-resolution helpers behind it). `uninstall.rs` calls into it
//! for `UninstallOptions`/`UninstallPlan`/`build_uninstall_plan`/
//! `confirm_uninstall`/`is_rocm_install_entry_name`/`remove_path`/
//! `render_uninstall_plan`; `storage.rs` calls into it for
//! `UninstallPlan`/`UninstallPlanEntry`/`confirm_uninstall`; `runtimes_cmd.rs`
//! calls into it for `confirm_uninstall` (the same `[y/N]` prompt, reused for
//! `rocm runtimes uninstall`); `chat.rs` calls into it for
//! `render_uninstall_dry_run` (the deterministic uninstall-preview tool
//! result); `deletion_properties.rs`'s recursive-deletion property tests call
//! into it for `UninstallOptions`/`build_uninstall_plan`/`remove_path`/
//! `render_uninstall_plan`.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, daemon_binary_path};

use crate::managed_service_status::load_managed_services;
use crate::{remote, uninstall};

#[derive(Debug, Clone, Default)]
pub(crate) struct UninstallOptions {
    pub(crate) yes: bool,
    pub(crate) dry_run: bool,
    pub(crate) keep_binaries: bool,
    pub(crate) keep_config: bool,
    pub(crate) keep_data: bool,
    pub(crate) keep_cache: bool,
    pub(crate) force_dev_binaries: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct UninstallPlanEntry {
    pub(crate) kind: &'static str,
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct UninstallPlan {
    pub(crate) actions: Vec<UninstallPlanEntry>,
    pub(crate) skipped: Vec<String>,
    pub(crate) warnings: Vec<String>,
    /// Installer PATH changes `rocm uninstall` undoes.
    pub(crate) path_edits: Vec<uninstall::PathEdit>,
    /// The installer's record of its PATH change, removed once that change is
    /// undone.
    pub(crate) path_record: Option<PathBuf>,
    /// The running executable, on Windows, which is deleted as the command exits.
    pub(crate) self_delete: Option<PathBuf>,
}

pub(crate) fn build_uninstall_plan(
    paths: &AppPaths,
    options: &UninstallOptions,
) -> Result<UninstallPlan> {
    let mut plan = UninstallPlan::default();

    plan_installed_binaries(
        &mut plan,
        options,
        || daemon_binary_path().context("failed to discover current rocm executable"),
        rocm_core::runtime_home_dir().as_deref(),
    )?;

    for (keep, kind, path) in [
        (options.keep_config, "config", paths.config_dir.clone()),
        (options.keep_data, "data", paths.data_dir.clone()),
        (options.keep_cache, "cache", paths.cache_dir.clone()),
    ] {
        if keep {
            plan.skipped
                .push(format!("{kind} removal disabled by command line flag"));
            continue;
        }
        // Planned under the spelling `remove_path` acts on, so the review names
        // the entry that is actually removed, and `dir/` and `dir` dedup to one
        // line. Existence is the entry's own, not its target's: a dangling link
        // is listed and then really removed, instead of reported as absent.
        // A failed check is not an absence either: it fails the plan, so the
        // uninstall stops before removing anything instead of leaving a folder
        // that may still be on disk out of the review.
        match existing_entry(&path)
            .with_context(|| format!("failed to check the {kind} folder {}", path.display()))?
        {
            Some((entry, _)) => plan.actions.push(UninstallPlanEntry { kind, path: entry }),
            None => plan
                .skipped
                .push(format!("{kind} path not present: {}", path.display())),
        }
    }

    for note in shared_cache_notes(paths) {
        plan.warnings.push(note);
    }

    let managed_services = load_managed_services(paths).unwrap_or_default();
    if !managed_services.is_empty() {
        plan.warnings.push(format!(
            "{} managed service record(s) exist under {}; background processes are not stopped automatically in this pass",
            managed_services.len(),
            paths.services_dir().display()
        ));
    }

    // Remote sessions are worse than local ones to drop silently. The model runs
    // on someone else's machine and its endpoint is published there, so removing
    // the record here does not stop either — it only destroys the last thing that
    // knew they existed. Name them and the command that tears them down properly.
    //
    // A read that fails is reported, not defaulted away. `load_all` already warns
    // past an individual unreadable record and keeps going, so an `Err` here is a
    // directory-level I/O failure, and a missing directory is `Ok`. Taking the
    // default would state "there are none" on the one path where we do not know —
    // and this plan goes on to delete the data directory those records live in,
    // so the warning that something is still published elsewhere would be lost at
    // exactly the moment it was the last copy.
    let remote_sessions = match remote::session::load_all(paths) {
        Ok(sessions) => sessions,
        Err(error) => {
            plan.warnings.push(format!(
                "could not read the remote session records under {}: {error:#}\n\
                 This pass cannot tell whether models are still running on other machines with \
                 their endpoints published. Check with `rocm remote status` before continuing.",
                paths.remote_sessions_dir().display()
            ));
            Vec::new()
        }
    };
    if !remote_sessions.is_empty() {
        plan.warnings.push(format!(
            "{} remote session record(s) exist under {}; their models keep running on the \
             remote machines and their endpoints stay published. Removing these records only \
             loses track of them - run `rocm remote stop <session>` for each first: {}",
            remote_sessions.len(),
            paths.remote_sessions_dir().display(),
            remote_sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    plan.actions
        .sort_by(|left, right| left.path.cmp(&right.path));
    plan.actions.dedup_by(|left, right| left.path == right.path);
    Ok(plan)
}

/// Plan removing the installed binaries beside the running one, which
/// `current_exe` names, and undoing the installer's PATH change for their
/// folder. `--keep-binaries` keeps both, and so does a cargo build's folder.
pub(crate) fn plan_installed_binaries(
    plan: &mut UninstallPlan,
    options: &UninstallOptions,
    current_exe: impl FnOnce() -> Result<PathBuf>,
    home: Option<&Path>,
) -> Result<()> {
    if options.keep_binaries {
        // The PATH change only serves the binaries, so it stays with them.
        plan.skipped.push(
            "binary removal disabled by --keep-binaries, so any PATH change the installer made \
             stays too"
                .to_owned(),
        );
        return Ok(());
    }
    let current_exe = current_exe()?;
    if is_dev_binary_layout(&current_exe) && !options.force_dev_binaries {
        plan.skipped.push(format!(
            "binary removal skipped because {} looks like a cargo target build; pass --force-dev-binaries to remove sibling debug/release binaries",
            current_exe.display()
        ));
        return Ok(());
    }
    let binaries = collect_installed_binary_candidates(&current_exe)?;
    for path in &binaries {
        // Windows will not unlink a running image; it is deleted as the
        // command exits instead.
        if is_running_executable(path, &current_exe, cfg!(windows)) {
            plan.self_delete = Some(path.clone());
            continue;
        }
        // The PATH record goes only once its change is undone; see
        // `uninstall::plan_path_edits`.
        if path.file_name().and_then(|name| name.to_str()) == Some(uninstall::PATH_EDIT_RECORD) {
            continue;
        }
        plan.actions.push(UninstallPlanEntry {
            kind: "binary",
            path: path.clone(),
        });
    }
    if let Some(bin_dir) = current_exe.parent() {
        uninstall::plan_path_edits(plan, bin_dir, &binaries, home);
    }
    Ok(())
}

/// Whether `path`, an entry of the running executable's folder, is the running
/// executable itself, which only Windows has to treat specially. Windows names
/// are case-insensitive, and the running image may report its own name in
/// another case, or in its 8.3 short form (`ROCM.EXE`), than the folder listing
/// does; missing it would plan an ordinary removal that Windows refuses.
fn is_running_executable(path: &Path, current_exe: &Path, windows: bool) -> bool {
    if !windows {
        return false;
    }
    path == current_exe
        || (path.parent() == current_exe.parent()
            && path
                .file_name()
                .zip(current_exe.file_name())
                .is_some_and(|(name, exe)| {
                    name.to_string_lossy()
                        .eq_ignore_ascii_case(&exe.to_string_lossy())
                }))
}

pub(crate) fn render_uninstall_dry_run(paths: &AppPaths) -> Result<String> {
    let options = UninstallOptions {
        dry_run: true,
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan(paths, &options)?;
    Ok(render_uninstall_plan(&plan, &options))
}

pub(crate) fn render_uninstall_plan(plan: &UninstallPlan, options: &UninstallOptions) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Uninstall review");
    let _ = writeln!(output);
    let removals = uninstall::removal_count(plan);
    if uninstall::plan_is_empty(plan) {
        let _ = writeln!(output, "Nothing managed by rocm-cli would be removed.");
    } else if removals > 0 {
        let _ = writeln!(output, "{removals} item(s) would be removed:");
        for entry in &plan.actions {
            let _ = writeln!(output, "  - {}: {}", entry.kind, entry.path.display());
        }
    }
    uninstall::render_extra_plan(&mut output, plan);
    if !plan.warnings.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Please review:");
    }
    for warning in &plan.warnings {
        let _ = writeln!(output, "  - {warning}");
    }
    if !plan.skipped.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "Left alone:");
    }
    for skipped in &plan.skipped {
        let _ = writeln!(output, "  - {skipped}");
    }
    if options.dry_run {
        let _ = writeln!(output);
        let _ = writeln!(output, "Choose Review uninstall to approve removal.");
    }
    output
}

/// Caches that ROCm CLI causes to be filled but does not own, so uninstall
/// leaves them in place.
///
/// These are usually the largest things on disk after an install, so a plan
/// that stays silent about them reads as a clean slate it does not deliver.
/// Only paths that exist and sit outside everything already being removed are
/// reported, so the notes stay truthful as caches move under the data
/// directory.
fn shared_cache_notes(paths: &AppPaths) -> Vec<String> {
    let removed_roots = [
        paths.config_dir.clone(),
        paths.data_dir.clone(),
        paths.cache_dir.clone(),
    ];
    shared_cache_notes_for(&removed_roots, &shared_cache_candidates())
}

/// The pure part of [`shared_cache_notes`], so the filtering is testable without
/// depending on the caller's real environment.
fn shared_cache_notes_for(
    removed_roots: &[PathBuf],
    candidates: &[(PathBuf, &'static str)],
) -> Vec<String> {
    candidates
        .iter()
        .filter(|(path, _)| path.exists())
        .filter(|(path, _)| !removed_roots.iter().any(|root| path.starts_with(root)))
        .map(|(path, description)| format!("{description} not removed: {}", path.display()))
        .collect()
}

/// The shared caches worth reporting, with why each is left alone.
fn shared_cache_candidates() -> Vec<(PathBuf, &'static str)> {
    let mut candidates = Vec::new();

    // `uv` writes to `UV_CACHE_DIR` when set, otherwise its own default. Read it
    // rather than assuming a location, so this stays correct wherever the cache
    // has been pointed.
    let uv_cache = env_path("UV_CACHE_DIR")
        .or_else(|| rocm_core::runtime_home_dir().map(|home| home.join(".cache").join("uv")));
    if let Some(path) = uv_cache {
        candidates.push((
            path,
            "the uv package cache is shared with other uv projects on this computer and is",
        ));
    }

    let hf_cache = env_path("HF_HOME")
        .map(|home| home.join("hub"))
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| {
            rocm_core::runtime_home_dir()
                .map(|home| home.join(".cache").join("huggingface").join("hub"))
        });
    if let Some(path) = hf_cache {
        candidates.push((
            path,
            "downloaded model files are slow to fetch again and may include your own, so they are",
        ));
    }

    candidates
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub(crate) fn confirm_uninstall() -> Result<bool> {
    rocm_core::confirm::print_prompt(&mut io::stdout(), "Proceed with uninstall? [y/N]: ")
        .context("failed to write uninstall prompt")?;
    rocm_core::confirm::read_answer(&mut io::stdin().lock())
        .context("failed to read uninstall confirmation")
}

fn collect_installed_binary_candidates(current_exe: &Path) -> Result<Vec<PathBuf>> {
    let binary_dir = current_exe
        .parent()
        .context("current executable has no parent directory")?;
    let mut binaries = Vec::new();
    for entry in fs::read_dir(binary_dir)
        .with_context(|| format!("failed to read {}", binary_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !path.is_file() {
            continue;
        }
        if is_rocm_install_entry_name(name) {
            binaries.push(path);
        }
    }
    binaries.sort();
    Ok(binaries)
}

pub(crate) fn is_rocm_install_entry_name(name: &str) -> bool {
    if name == ".rocm-cli-manifest" || name == uninstall::PATH_EDIT_RECORD {
        return true;
    }
    let normalized = name.strip_suffix(".exe").unwrap_or(name);
    normalized == "rocm"
        || normalized == "rocmd"
        || normalized == "rocm-codex"
        || normalized.starts_with("rocm-engine-")
}

fn is_dev_binary_layout(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(parent_name) = parent.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    if parent_name != "debug" && parent_name != "release" {
        return false;
    }
    parent
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        == Some("target")
}

/// `path` respelled so it names the final entry itself.
///
/// A trailing separator (`link/`) or a trailing `.` (`link/.`) makes the kernel
/// resolve a final symlink, so `symlink_metadata` reports the link's *target*
/// directory and a recursive delete walks into it. Rebuilding the path from its
/// components drops exactly those spellings (plus repeated separators and
/// interior `.`), none of which otherwise change which entry is named. `..` is
/// kept: dropping it lexically would change the meaning through a link.
fn entry_path(path: &Path) -> PathBuf {
    path.components().collect()
}

/// Does `path`'s text end in a separator or a final `.` (`x/`, `x/.`)? The
/// kernel only lets such a spelling name a directory, so `notes.txt/` names
/// nothing even though `notes.txt` exists.
fn spelled_as_directory(path: &Path) -> bool {
    let text = path.as_os_str().as_encoded_bytes();
    let is_separator = |byte: u8| std::path::is_separator(char::from(byte));
    match text {
        [.., last] if is_separator(*last) => true,
        [.., before, b'.'] => is_separator(*before),
        _ => false,
    }
}

/// The entry `path` names, as [`entry_path`] spells it, with its own
/// (link-not-followed) metadata — or `None` when nothing is there.
///
/// A spelling that can only name a directory (see [`spelled_as_directory`])
/// names nothing when the entry is a regular file, so that counts as absent
/// rather than letting the respelling turn `notes.txt/` into `notes.txt`. A
/// symlink is still the entry: it is what [`remove_path`] unlinks.
fn existing_entry(path: &Path) -> io::Result<Option<(PathBuf, fs::Metadata)>> {
    let entry = entry_path(path);
    // Not `path.exists()`: that follows links, so a dangling link would read as
    // already gone, be skipped, and still be reported as removed by the caller.
    // "Not a directory" means a folder on the way is really a file (as in
    // `notes.txt/../x`), so the path names nothing, the same as "not found".
    // Any other failure is a real error, not an absence.
    let metadata = match fs::symlink_metadata(&entry) {
        Ok(metadata) => metadata,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    if spelled_as_directory(path) && !metadata.is_dir() && !metadata.file_type().is_symlink() {
        return Ok(None);
    }
    Ok(Some((entry, metadata)))
}

/// Remove the entry `path` names: a file or symlink is unlinked (never
/// followed, however `path` is spelled), a real directory is removed
/// recursively, and an entry that is already gone is a no-op.
///
/// Normalizing here rather than only at plan time covers every caller —
/// `rocm uninstall` and `rocm storage remove-downloads` — whatever spelling
/// their plan carries.
pub(crate) fn remove_path(path: &Path) -> Result<()> {
    let Some((path, metadata)) =
        existing_entry(path).with_context(|| format!("failed to stat {}", path.display()))?
    else {
        return Ok(());
    };
    let path = path.as_path();
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else if metadata.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else {
        bail!("unsupported filesystem entry for {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;

    /// A cache that has moved inside a directory uninstall already removes must
    /// not be reported as "not removed" — the note would be false.
    #[test]
    fn shared_cache_notes_skip_paths_already_being_removed() {
        let root = std::env::temp_dir().join(format!("rocm-shared-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let data = root.join("data");
        let inside = data.join("uv-cache");
        std::fs::create_dir_all(&inside).unwrap();

        let notes = shared_cache_notes_for(
            &[root.join("config"), data, root.join("cache")],
            &[(inside, "the uv package cache")],
        );

        assert!(
            notes.is_empty(),
            "a cache inside a removed directory must not be reported: {notes:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A cache that exists outside everything being removed is reported, with
    /// its path, so the plan does not imply a clean slate it will not deliver.
    #[test]
    fn shared_cache_notes_report_paths_left_behind() {
        let root = std::env::temp_dir().join(format!("rocm-shared-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let outside = root.join("elsewhere").join("uv");
        std::fs::create_dir_all(&outside).unwrap();

        let notes = shared_cache_notes_for(
            &[root.join("data")],
            &[(outside.clone(), "the uv package cache")],
        );

        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("not removed"), "{notes:?}");
        assert!(
            notes[0].contains(&outside.display().to_string()),
            "{notes:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A cache that does not exist is not worth mentioning.
    #[test]
    fn shared_cache_notes_ignore_missing_paths() {
        let missing = std::env::temp_dir().join("rocm-definitely-not-here-12345");
        let notes = shared_cache_notes_for(
            &[std::env::temp_dir()],
            &[(missing, "the uv package cache")],
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn uninstall_binary_matcher_includes_packaged_codex_binary() {
        assert!(is_rocm_install_entry_name("rocm-codex"));
        assert!(is_rocm_install_entry_name("rocm-codex.exe"));
    }

    /// Windows names are case-insensitive, and the running image may report its
    /// own name in another case, or in its 8.3 short form, than the folder
    /// listing does; `is_running_executable` must still recognize it.
    #[test]
    fn the_running_executable_is_matched_whatever_case_windows_reports_it_in() {
        let exe = Path::new(r"C:\Users\me\.local\bin\rocm.exe");
        let listed = Path::new(r"C:\Users\me\.local\bin\ROCM.EXE");
        assert!(is_running_executable(listed, exe, true));
        assert!(is_running_executable(exe, exe, true));
        let rocmd = Path::new(r"C:\Users\me\.local\bin\rocmd.exe");
        assert!(!is_running_executable(rocmd, exe, true));
        // Elsewhere the running binary is unlinked like any other.
        assert!(!is_running_executable(exe, exe, false));
    }

    /// `rocm uninstall`'s review warns that a remote session keeps running on
    /// someone else's machine, and names the command that stops it properly.
    ///
    /// Nothing exercised that warning before: no test planted a remote session
    /// record, so the whole sentence -- including the ASCII punctuation this
    /// change put in it (ROCMAI-543) -- shipped uncovered. The remediation is
    /// asserted alongside the encoding because a warning that stops naming
    /// `rocm remote stop` leaves the user with no way to act on it.
    #[test]
    fn uninstall_warns_about_remote_sessions_in_ascii_and_names_the_stop_command() -> Result<()> {
        let (root, paths) = test_paths("uninstall-remote-session-warning");
        paths.ensure()?;
        let session = crate::remote::session::RemoteSessionRecord {
            session_id: crate::remote::session::RemoteSessionRecord::id_for("dev-box", 8000),
            target: "dev-box".to_owned(),
            peer_host: "dev-box".to_owned(),
            ssh_port: None,
            model: "qwen".to_owned(),
            remote_service_id: "svc-remote".to_owned(),
            remote_cli: "rocm".to_owned(),
            remote_port: 8000,
            tailnet_port: 8000,
            base_url: "http://dev-box:8000/v1".to_owned(),
            created_at_unix_ms: 1,
        };
        session.write(&paths)?;

        // `keep_binaries` so the plan never reaches outside the sandbox: the
        // binary paths are the host's real ones and have nothing to do with
        // this warning.
        let options = UninstallOptions {
            dry_run: true,
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let plan = build_uninstall_plan(&paths, &options);
        let _ = fs::remove_dir_all(&root);
        let plan = plan?;

        let warning = plan
            .warnings
            .iter()
            .find(|warning| warning.contains("remote session record(s)"))
            .unwrap_or_else(|| {
                panic!("a planted remote session must be warned about: {plan:#?}");
            });
        assert!(
            warning.contains(&session.session_id),
            "the warning must name the session it found: {warning}"
        );
        assert!(
            warning.contains("rocm remote stop <session>"),
            "the warning must name the command that stops the remote model: {warning}"
        );
        // ROCMAI-543: printed verbatim by `render_uninstall_plan`, so non-ASCII
        // punctuation here renders as mojibake like "ΓÇö" on a Windows console
        // using the OEM code page.
        assert!(
            warning.is_ascii(),
            "the remote-session warning must be ASCII-only (ROCMAI-543): {warning}"
        );
        Ok(())
    }
}
