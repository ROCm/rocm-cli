// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The uninstall plan: what would be removed, how that plan is rendered for
//! review, and the filesystem-removal primitive used to carry it out.
//!
//! First relocated mechanically from `main.rs` — Phase 6f (ROCMAI-212) of
//! the `main.rs` modularization (see `docs/architecture.md`); the
//! protected-root judgment described at the end was added here afterwards.
//! `dispatch()`'s call sites are unaffected: none of this module's functions
//! are called directly from `dispatch()`.
//!
//! This is deliberately a separate file from `uninstall.rs`, not folded into
//! it: `uninstall.rs` owns what `rocm uninstall` itself does — carrying the
//! plan out (`apply_uninstall_plan`, the one step that deletes), undoing the
//! installer's PATH change and deleting the running executable on Windows,
//! which cannot unlink itself. `storage.rs`
//! builds and renders a removal plan of its own (`rocm storage
//! remove-downloads`) and reuses this module's plan type and its
//! file-removal helpers to do it, so keeping them in `uninstall.rs` would
//! make `storage.rs` depend on a module whose name and doc comment both
//! describe the `uninstall` command's own, exclusive logic.
//!
//! This module owns: the plan types (`UninstallOptions`, `UninstallPlan`,
//! `UninstallPlanEntry`, `PlannedAs`, `RefusedRoot`), the `PlanAudience` a
//! rendered plan is written for (a person at a terminal, a captured caller, or
//! the assistant's preview), which decides what the dry-run footer says comes
//! next, building the plan (`build_uninstall_plan`, `build_uninstall_plan_for_home`,
//! `plan_installed_binaries`, and the installed-binary/shared-cache
//! discovery helpers behind them), rendering it (`render_uninstall_dry_run`,
//! `render_uninstall_plan`), the `[y/N]` confirmation prompt
//! (`confirm_uninstall`), and the filesystem removal primitive both
//! `uninstall.rs` and `storage.rs` call to carry a plan out (`remove_path`,
//! and the entry-resolution helpers behind it). `uninstall.rs` calls into it
//! for `PlanAudience`/`UninstallOptions`/`UninstallPlan`/`UninstallPlanEntry`/
//! `PlannedAs`/`build_uninstall_plan`/`confirm_uninstall`/`entry_path`/
//! `is_rocm_install_entry_name`/`remove_path`/`render_uninstall_plan`; `storage.rs` calls into it for
//! `UninstallPlan`/`UninstallPlanEntry`/`confirm_uninstall`; `runtimes_cmd.rs`
//! calls into it for `confirm_uninstall` (the same `[y/N]` prompt, reused for
//! `rocm runtimes uninstall`); `chat/mcp_bridge.rs` calls into it for
//! `render_uninstall_dry_run` (the deterministic uninstall-preview tool
//! result, which renders for `PlanAudience::Assistant` itself and so does not
//! import the type); `deletion_properties.rs`'s recursive-deletion property
//! tests call into it for `PlanAudience` (the `uninstall_case` property renders
//! as `Captured`)/`UninstallOptions`/`build_uninstall_plan`/
//! `build_uninstall_plan_for_home`/`SharedCache`/`shared_cache_notes_for`/
//! `remove_path`/`render_uninstall_plan`.
//!
//! Building the plan also judges each config, data and cache root against the
//! protected locations (`judge_uninstall_root`), so a refused root lands in
//! `UninstallPlan::refused` with the advice that clears it, and records what
//! each planned entry was (`PlannedAs`) so `uninstall.rs` can check it again
//! right before removing it (with `entry_path`).

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
    /// What the entry was when the review was made, checked again right
    /// before it is removed.
    pub(crate) planned: PlannedAs,
}

/// What a planned entry was at review time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlannedAs {
    /// A real directory, and where it really was.
    Dir(PathBuf),
    /// A symbolic link, and where the folder holding it really was.
    Link { parent: Option<PathBuf> },
    /// A file, and where the folder holding it really was.
    File { parent: Option<PathBuf> },
    /// Not re-checked (`rocm storage remove-downloads` entries, which have
    /// their own removal loop).
    Unchecked,
}

impl PlannedAs {
    /// A non-directory entry as it is now, with its folder resolved.
    pub(crate) fn non_dir(
        path: &Path,
        metadata: &fs::Metadata,
        canonicalize: Canonicalize<'_>,
    ) -> Self {
        let parent = path
            .parent()
            .map(|parent| {
                if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                }
            })
            .and_then(|parent| canonicalize(parent).ok());
        if metadata.file_type().is_symlink() {
            Self::Link { parent }
        } else {
            Self::File { parent }
        }
    }
}

/// A config/data/cache root `rocm uninstall` will not remove, and why.
#[derive(Debug, Clone)]
pub(crate) struct RefusedRoot {
    pub(crate) kind: &'static str,
    pub(crate) path: PathBuf,
    /// Where `path` really is, when only that (not its spelling) is protected:
    /// `.` run from home, or home reached through a symlinked parent.
    pub(crate) resolved: Option<PathBuf>,
    /// The resolved location whether or not it differs from `path`; where the
    /// marker would go for [`rocm_core::ProtectedLocation::NotMarked`].
    pub(crate) real: Option<PathBuf>,
    pub(crate) why: rocm_core::ProtectedLocation,
    pub(crate) source: rocm_core::AppPathSource,
}

impl RefusedRoot {
    /// What would clear this refusal, as one sentence. Every way named here is
    /// one the tests follow: `--keep-<kind>` leaves the root out of the plan;
    /// pointing the environment variable elsewhere moves it; creating the
    /// marker in a folder that is ROCm CLI's makes it removable. For a root
    /// that came from `setup.therock_venv` the flag comes first and moving the
    /// setting is advised against: the ROCm SDK is installed in that folder,
    /// and repointing the setting would leave it there, unknown to ROCm CLI.
    pub(crate) fn advice(&self) -> String {
        let keep = format!("re-run with --keep-{} to leave it in place", self.kind);
        let mut ways = Vec::new();
        if let rocm_core::AppPathSource::Env(var) = &self.source {
            ways.push(format!("point {var} at a folder of ROCm CLI's own"));
        }
        if self.why == rocm_core::ProtectedLocation::NotMarked {
            let real = self.real.as_deref().unwrap_or(&self.path);
            ways.push(format!(
                "if this folder belongs to ROCm CLI, create {}",
                real.join(rocm_core::ROCM_CLI_ROOT_MARKER).display()
            ));
        }
        let therock = matches!(self.source, rocm_core::AppPathSource::TheRockVenv(_));
        if therock {
            ways.insert(0, keep);
        } else {
            ways.push(keep);
        }
        let mut sentence = ways.join(", or ");
        if let Some(first) = sentence.get(..1) {
            sentence.replace_range(..1, &first.to_uppercase());
        }
        sentence.push('.');
        if therock {
            sentence.push_str(
                " Moving setup.therock_venv instead would orphan the ROCm SDK installed there.",
            );
        }
        match self.why {
            rocm_core::ProtectedLocation::NotMarked => sentence.push_str(
                " Folders created by ROCm CLI releases before this one carry no marker; a \
                 config.json or runtime registry inside it is not taken as proof that it is \
                 ROCm CLI's.",
            ),
            rocm_core::ProtectedLocation::SharedTopLevel
            | rocm_core::ProtectedLocation::OtherUsersHome
            | rocm_core::ProtectedLocation::System => sentence.push_str(
                " ROCm CLI never removes this folder itself; if you are sure what in it is \
                 ROCm CLI's, remove those files by hand.",
            ),
            _ => {}
        }
        sentence
    }

    /// `<path> is <why> (<source>[; resolves to <resolved>])`.
    pub(crate) fn describe(&self) -> String {
        let resolved = self
            .resolved
            .as_ref()
            .map(|resolved| format!("; resolves to {}", resolved.display()))
            .unwrap_or_default();
        format!(
            "{} is {} ({}{resolved})",
            self.path.display(),
            self.why.describe(),
            self.source.describe()
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct UninstallPlan {
    pub(crate) actions: Vec<UninstallPlanEntry>,
    /// Roots the plan would otherwise remove but that sit in a protected
    /// location. Any entry here refuses the whole uninstall.
    pub(crate) refused: Vec<RefusedRoot>,
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

impl UninstallPlan {
    /// How to get past a refusal, derived from the refused roots themselves:
    /// the setting each one came from, which moves it, and its `--keep-<kind>`
    /// flag, which is exactly what leaves it out of the plan — so the advice
    /// cannot name a flag that does not clear it.
    pub(crate) fn refusal_advice(&self) -> Option<String> {
        if self.refused.is_empty() {
            return None;
        }
        let per_root = self
            .refused
            .iter()
            .map(|root| {
                if self.refused.len() == 1 {
                    root.advice()
                } else {
                    format!("For the {} folder: {}", root.kind, root.advice())
                }
            })
            .collect::<Vec<_>>();
        Some(per_root.join(" "))
    }

    /// The error a real (non-dry-run) uninstall stops with when anything was
    /// refused. It is raised before the confirmation prompt and before the
    /// first removal, which is what makes "nothing was removed" true.
    pub(crate) fn refusal_error(&self) -> Option<String> {
        let advice = self.refusal_advice()?;
        let reasons = self
            .refused
            .iter()
            .map(|root| format!("the {} folder {}", root.kind, root.describe()))
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "uninstall refused, nothing was removed: {reasons}, and ROCm CLI never removes \
             such a folder. {advice}"
        ))
    }
}

/// How a planned root resolves: where it really is, for display and for the
/// check made again right before removal.
pub(crate) type Canonicalize<'a> = &'a dyn Fn(&Path) -> io::Result<PathBuf>;

/// What `rocm uninstall` does with one config, data or cache root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RootJudgment {
    /// Remove it; `planned` is what it is now, to check again before removal.
    Remove { planned: PlannedAs },
    /// Refuse the whole uninstall. `shown` is the resolved location when it
    /// differs from the spelling; `real` is the resolved location regardless.
    Refuse {
        why: rocm_core::ProtectedLocation,
        shown: Option<PathBuf>,
        real: Option<PathBuf>,
    },
}

/// Judge the root `path` by [`rocm_core::uninstall_root_verdict`]: on its
/// spelling first, then — for a real directory, the only kind removed
/// recursively — on where it really is against where home and the system
/// locations really are. `.` run from home, `..` run from `~/proj`, home
/// reached through a symlinked parent (`/home -> var/home`), or `/usr/lib`
/// where `/lib -> usr/lib`, name a protected place without spelling it.
/// A real directory outside the home folder must also carry the marker.
pub(crate) fn judge_uninstall_root(
    path: &Path,
    metadata: &fs::Metadata,
    home: Option<&Path>,
    canonicalize: Canonicalize<'_>,
) -> RootJudgment {
    use rocm_core::{ProtectedLocation, UninstallRootVerdict};
    if let UninstallRootVerdict::Refused(why) = rocm_core::uninstall_root_verdict(path, home, &[])
        && !(why == ProtectedLocation::OtherUsersHome
            && metadata.is_dir()
            && resolves_inside_own_home(path, home, canonicalize))
    {
        return RootJudgment::Refuse {
            why,
            shown: None,
            real: None,
        };
    }
    // `symlink_metadata`: a symlinked root is unlinked, never walked, and a
    // file is just a file.
    if !metadata.is_dir() {
        return RootJudgment::Remove {
            planned: PlannedAs::non_dir(path, metadata, canonicalize),
        };
    }
    // A directory that cannot be resolved cannot be judged by where it is, and
    // is about to be walked recursively: refuse rather than guess.
    let Ok(resolved) = canonicalize(path) else {
        return RootJudgment::Refuse {
            why: ProtectedLocation::Unresolvable,
            shown: None,
            real: None,
        };
    };
    let resolved_home = home.map(|home| {
        canonicalize(home).unwrap_or_else(|_| rocm_core::resolve_path_through_symlinks(home))
    });
    let resolved_system_roots = rocm_core::uninstall_system_roots()
        .iter()
        .filter_map(|root| canonicalize(root).ok())
        .collect::<Vec<_>>();
    let shown = (resolved != path).then(|| resolved.clone());
    let refuse = |why| RootJudgment::Refuse {
        why,
        shown: shown.clone(),
        real: Some(resolved.clone()),
    };
    match rocm_core::uninstall_root_verdict(
        &resolved,
        resolved_home.as_deref(),
        &resolved_system_roots,
    ) {
        UninstallRootVerdict::Refused(why) => refuse(why),
        UninstallRootVerdict::NeedsMarker if !is_root_marker(&resolved) => {
            refuse(ProtectedLocation::NotMarked)
        }
        UninstallRootVerdict::Allowed | UninstallRootVerdict::NeedsMarker => RootJudgment::Remove {
            planned: PlannedAs::Dir(resolved),
        },
    }
}

/// Whether `dir` carries the marker as a regular file of its own. A link named
/// like the marker is not one: anything that can plant a link there could
/// point it at any file, and ROCm CLI only ever writes a plain file.
fn is_root_marker(dir: &Path) -> bool {
    fs::symlink_metadata(dir.join(rocm_core::ROCM_CLI_ROOT_MARKER))
        .is_ok_and(|meta| meta.file_type().is_file())
}

/// Whether the real directory `path` really is the user's home or inside it,
/// though spelled outside it — on Windows the temp folder is often spelled
/// with the home folder's short 8.3 name (`C:\Users\RUNNER~1\...`), which by
/// spelling alone is someone else's folder under `C:\Users`. Such a root is
/// judged by where it really is instead, which allows it inside home and
/// refuses it as the home folder itself.
///
/// A home that resolves to the top of the filesystem decides nothing: every
/// folder is inside it, so it cannot tell another user's folder from one of
/// the user's own.
fn resolves_inside_own_home(
    path: &Path,
    home: Option<&Path>,
    canonicalize: Canonicalize<'_>,
) -> bool {
    let (Some(home), Ok(resolved)) = (home, canonicalize(path)) else {
        return false;
    };
    let Ok(resolved_home) = canonicalize(home) else {
        return false;
    };
    resolved_home.parent().is_some()
        && rocm_core::runtime_path_is_same_or_inside(&resolved, &resolved_home)
}

pub(crate) fn build_uninstall_plan(
    paths: &AppPaths,
    sources: &rocm_core::AppPathSources,
    options: &UninstallOptions,
) -> Result<UninstallPlan> {
    build_uninstall_plan_for_home(
        paths,
        sources,
        options,
        rocm_core::runtime_home_dir().as_deref(),
    )
}

/// [`build_uninstall_plan`] judged against `home` rather than the process's
/// own, so the protected-root check can be tested in a sandbox.
pub(crate) fn build_uninstall_plan_for_home(
    paths: &AppPaths,
    sources: &rocm_core::AppPathSources,
    options: &UninstallOptions,
    home: Option<&Path>,
) -> Result<UninstallPlan> {
    build_uninstall_plan_with(
        paths,
        sources,
        options,
        home,
        &rocm_core::canonicalize_for_compare,
    )
}

/// [`build_uninstall_plan_for_home`] with the resolver passed in, so a test
/// can make a root unresolvable.
pub(crate) fn build_uninstall_plan_with(
    paths: &AppPaths,
    sources: &rocm_core::AppPathSources,
    options: &UninstallOptions,
    home: Option<&Path>,
    canonicalize: Canonicalize<'_>,
) -> Result<UninstallPlan> {
    let mut plan = UninstallPlan::default();

    plan_installed_binaries(
        &mut plan,
        options,
        || daemon_binary_path().context("failed to discover current rocm executable"),
        home,
        canonicalize,
    )?;

    for (keep, kind, path, source) in [
        (
            options.keep_config,
            "config",
            paths.config_dir.clone(),
            &sources.config,
        ),
        (
            options.keep_data,
            "data",
            paths.data_dir.clone(),
            &sources.data,
        ),
        (
            options.keep_cache,
            "cache",
            paths.cache_dir.clone(),
            &sources.cache,
        ),
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
        let Some((path, metadata)) = existing_entry(&path)
            .with_context(|| format!("failed to check the {kind} folder {}", path.display()))?
        else {
            plan.skipped
                .push(format!("{kind} path not present: {}", path.display()));
            continue;
        };
        // These roots come from environment variables or `setup.therock_venv`,
        // so `/` or the home folder is one typo away.
        match judge_uninstall_root(&path, &metadata, home, canonicalize) {
            RootJudgment::Refuse { why, shown, real } => plan.refused.push(RefusedRoot {
                kind,
                path,
                resolved: shown,
                real,
                why,
                source: source.clone(),
            }),
            RootJudgment::Remove { planned } => plan.actions.push(UninstallPlanEntry {
                kind,
                path,
                planned,
            }),
        }
    }

    plan.warnings.extend(shared_cache_notes(&plan.actions));

    let managed_services = load_managed_services(paths).unwrap_or_default();
    if !managed_services.is_empty() {
        plan.warnings.push(format!(
            "{} managed service record(s) exist under {}; background processes are not stopped automatically in this pass",
            managed_services.len(),
            paths.services_dir().display()
        ));
    }

    // Only a helper that is running can be left behind. "Uninstall does not stop
    // it" is said only where that is the whole story: with a watcher enabled the
    // helper stays up; an idle one is already on its way out.
    if let Some(note) = crate::background_helper::background_helper_note(paths)
        && note.running
    {
        plan.warnings.push(if note.outlives {
            format!("{}; uninstall does not stop it", note.text)
        } else {
            note.text
        });
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
/// `current_exe` names, undoing the installer's PATH change for their folder,
/// and removing the fish completion script it recorded writing for them.
/// `--keep-binaries` keeps all three, and so does a cargo build's folder.
pub(crate) fn plan_installed_binaries(
    plan: &mut UninstallPlan,
    options: &UninstallOptions,
    current_exe: impl FnOnce() -> Result<PathBuf>,
    home: Option<&Path>,
    canonicalize: Canonicalize<'_>,
) -> Result<()> {
    if options.keep_binaries {
        // The PATH change and the completions only serve the binaries, so
        // they stay with them.
        plan.skipped.push(
            "binary removal disabled by --keep-binaries, so any PATH change and fish completions \
             the installer made stay too"
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
        let planned = fs::symlink_metadata(path).map_or(PlannedAs::Unchecked, |meta| {
            PlannedAs::non_dir(path, &meta, canonicalize)
        });
        plan.actions.push(UninstallPlanEntry {
            kind: "binary",
            path: path.clone(),
            planned,
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

/// The in-process `uninstall --dry-run` the assistant and dashboard run.
///
/// Honours the same `--keep-*` and `--force-dev-binaries` flags as the real
/// command: a refusal's advice is to re-run with `--keep-<kind>`, and a preview
/// that ignored the flag would keep repeating that advice after it was taken.
///
/// Where each folder came from is derived from `paths` itself
/// ([`rocm_core::AppPathSources::for_paths`]): the source of a folder
/// discovery would pick is discovery's, and a folder it would not pick is
/// reported as pinned — so the source shown can never belong to a different
/// folder than the one judged.
pub(crate) fn render_uninstall_dry_run(paths: &AppPaths, args: &[String]) -> Result<String> {
    let sources = rocm_core::AppPathSources::for_paths(paths);
    render_uninstall_dry_run_with(
        paths,
        &sources,
        args,
        rocm_core::runtime_home_dir().as_deref(),
    )
}

pub(crate) fn render_uninstall_dry_run_with(
    paths: &AppPaths,
    sources: &rocm_core::AppPathSources,
    args: &[String],
    home: Option<&Path>,
) -> Result<String> {
    let options = UninstallOptions {
        dry_run: true,
        keep_binaries: crate::rocm_argv::chat_cli_has_flag(args, "--keep-binaries"),
        keep_config: crate::rocm_argv::chat_cli_has_flag(args, "--keep-config"),
        keep_data: crate::rocm_argv::chat_cli_has_flag(args, "--keep-data"),
        keep_cache: crate::rocm_argv::chat_cli_has_flag(args, "--keep-cache"),
        force_dev_binaries: crate::rocm_argv::chat_cli_has_flag(args, "--force-dev-binaries"),
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan_for_home(paths, sources, &options, home)?;
    Ok(render_uninstall_plan(
        &plan,
        &options,
        PlanAudience::Assistant,
    ))
}

/// Who reads a rendered plan, which decides what its dry-run footer can truthfully
/// say comes next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanAudience {
    /// A person at a terminal ran `rocm uninstall --dry-run` themselves: running
    /// the same command again without `--dry-run` is the next step.
    Terminal,
    /// The output is captured by a caller (a script, the dashboard, the daemon's
    /// tool). Whatever drives that caller has its own way to proceed, so there
    /// is no next step to name; "run it again" would address the wrong reader.
    Captured,
    /// The assistant's read-only preview tool: it offers no command to run next,
    /// since the removal itself goes through an approval step. (A refused
    /// preview still names the `--keep-<kind>` flag that clears the refusal.)
    Assistant,
}

impl PlanAudience {
    /// The audience for `rocm uninstall` itself, from whether `interactive_terminal`
    /// holds: stdin and stdout are both terminals. A dry run with either one
    /// redirected, `--dry-run < /dev/null` at a terminal for one, is `Captured`
    /// and prints no footer.
    pub(crate) const fn for_command(interactive_terminal: bool) -> Self {
        if interactive_terminal {
            Self::Terminal
        } else {
            Self::Captured
        }
    }

    /// What a dry run says to do next, if anything.
    const fn dry_run_footer(self) -> Option<&'static str> {
        match self {
            Self::Terminal => {
                Some("Nothing was removed. To remove these items, run it again without --dry-run.")
            }
            Self::Captured => None,
            // The assistant's own uninstall is an approval action; see
            // `lifecycle_read_mutate_split_is_honest` in `tool_policy.rs`.
            Self::Assistant => Some("Nothing was removed. Removal needs your approval."),
        }
    }
}

pub(crate) fn render_uninstall_plan(
    plan: &UninstallPlan,
    options: &UninstallOptions,
    audience: PlanAudience,
) -> String {
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
    if !plan.refused.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "Refused (nothing is removed while any of these is listed):"
        );
    }
    for root in &plan.refused {
        let _ = writeln!(output, "  - {}: {}", root.kind, root.describe());
    }
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
        if let Some(advice) = plan.refusal_advice() {
            // Whoever reads it, a refused dry run says what clears the refusal
            // instead of a next step: there is none until it is cleared.
            let _ = writeln!(output);
            let _ = writeln!(output, "This uninstall would be refused. {advice}");
        } else if !uninstall::plan_is_empty(plan)
            && let Some(footer) = audience.dry_run_footer()
        {
            // An empty plan has nothing to approve, so a dry run says nothing
            // about a next step.
            let _ = writeln!(output);
            let _ = writeln!(output, "{footer}");
        }
    }
    output
}

/// Caches that ROCm CLI causes to be filled but does not own.
///
/// These are usually the largest things on disk after an install, so a plan
/// that stays silent about them reads as a clean slate it does not deliver.
/// Each one that exists is reported either way: left in place when it sits
/// outside every root being removed, or deleted when it sits inside one — the
/// case where staying silent would be worst, because a root pointed at
/// something like `~/.cache` takes these caches with it.
fn shared_cache_notes(planned: &[UninstallPlanEntry]) -> Vec<String> {
    shared_cache_notes_for(planned, &shared_cache_candidates())
}

/// A shared cache, named the two ways a plan may need to talk about it.
pub(crate) struct SharedCache {
    pub(crate) path: PathBuf,
    /// Leads the "left in place" note, which ends "... not removed: <path>".
    pub(crate) kept: &'static str,
    /// What the cache is, for the "will be deleted" warning.
    pub(crate) name: &'static str,
    /// Why losing it matters, as a sentence.
    pub(crate) loss: &'static str,
    /// `it` or `them`, to agree with `name`.
    pub(crate) pronoun: &'static str,
}

/// The pure part of [`shared_cache_notes`], so the classification is testable
/// without depending on the caller's real environment.
///
/// A planned root only takes a cache with it when it is a real directory: a
/// root that is a symlink is unlinked, never followed, so what sits behind it
/// survives. Containment is judged on resolved paths when both resolve, so a
/// cache reached through a linked ancestor is still recognised as inside.
pub(crate) fn shared_cache_notes_for(
    planned: &[UninstallPlanEntry],
    candidates: &[SharedCache],
) -> Vec<String> {
    let removed_dirs: Vec<&UninstallPlanEntry> = planned
        .iter()
        // `symlink_metadata` reports a symlink as not a directory.
        .filter(|entry| fs::symlink_metadata(&entry.path).is_ok_and(|metadata| metadata.is_dir()))
        .collect();
    let inside = |cache: &Path, root: &Path| match (cache.canonicalize(), root.canonicalize()) {
        (Ok(cache), Ok(root)) => cache.starts_with(root),
        _ => cache.starts_with(root),
    };

    let mut deleted = Vec::new();
    let mut kept = Vec::new();
    for cache in candidates.iter().filter(|cache| cache.path.exists()) {
        if let Some(root) = removed_dirs
            .iter()
            .find(|root| inside(&cache.path, &root.path))
        {
            deleted.push(format!(
                "{} WILL BE DELETED: {} is inside the {} folder being removed ({}). {} \
                 Re-run with --keep-{} to keep {}.",
                cache.name,
                cache.path.display(),
                root.kind,
                root.path.display(),
                cache.loss,
                root.kind,
                cache.pronoun,
            ));
        } else {
            kept.push(format!(
                "{} not removed: {}",
                cache.kept,
                cache.path.display()
            ));
        }
    }
    // Deletions first: they are the notes a reader must not skim past.
    deleted.extend(kept);
    deleted
}

/// The shared caches worth reporting, with why each matters.
fn shared_cache_candidates() -> Vec<SharedCache> {
    let mut candidates = Vec::new();

    // `uv` writes to `UV_CACHE_DIR` when set, otherwise its own default. Read it
    // rather than assuming a location, so this stays correct wherever the cache
    // has been pointed.
    let uv_cache = env_path("UV_CACHE_DIR")
        .or_else(|| rocm_core::runtime_home_dir().map(|home| home.join(".cache").join("uv")));
    if let Some(path) = uv_cache {
        candidates.push(SharedCache {
            path,
            kept: "the uv package cache is shared with other uv projects on this computer and is",
            name: "the uv package cache",
            loss: "It is shared with other uv projects on this computer.",
            pronoun: "it",
        });
    }

    let hf_cache = env_path("HF_HOME")
        .map(|home| home.join("hub"))
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| {
            rocm_core::runtime_home_dir()
                .map(|home| home.join(".cache").join("huggingface").join("hub"))
        });
    if let Some(path) = hf_cache {
        candidates.push(SharedCache {
            path,
            kept: "downloaded model files are slow to fetch again and may include your own, so they are",
            name: "downloaded model files",
            loss: "They are slow to fetch again and may include your own.",
            pronoun: "them",
        });
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
    rocm_core::confirm::read_answer(&mut io::stdin().lock(), &mut io::stdout())
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
pub(crate) fn entry_path(path: &Path) -> PathBuf {
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
    use crate::tests::{ScopedTestEnv, test_paths};
    use serde_json::json;

    fn plan_with_a_binary() -> UninstallPlan {
        UninstallPlan {
            self_delete: Some(PathBuf::from("/opt/bin/rocm")),
            ..UninstallPlan::default()
        }
    }

    fn dry_run() -> UninstallOptions {
        UninstallOptions {
            dry_run: true,
            ..UninstallOptions::default()
        }
    }

    /// A person at a terminal who dry-runs is told to run the same command again
    /// without `--dry-run`: the next step is a rerun, not a control the command
    /// line lacks. The scenario `uninstall-07` runs it again at a terminal and
    /// checks that the listed folder goes.
    #[test]
    fn terminal_dry_run_footer_says_to_run_it_again_without_dry_run() {
        let output =
            render_uninstall_plan(&plan_with_a_binary(), &dry_run(), PlanAudience::Terminal);
        assert!(
            output.ends_with(
                "\nNothing was removed. To remove these items, run it again without --dry-run.\n"
            ),
            "unexpected footer in:\n{output}"
        );
        assert!(!output.contains("Review uninstall"), "{output}");
    }

    /// Output a caller captures is not addressed to someone who can rerun it, so
    /// it carries no next step (the dashboard's `/uninstall` and the daemon's
    /// `rocm_command` tool run the dry run as a subprocess, where the real step
    /// is theirs to offer).
    #[test]
    fn captured_dry_run_names_no_next_step() {
        let output =
            render_uninstall_plan(&plan_with_a_binary(), &dry_run(), PlanAudience::Captured);
        assert!(!output.contains("Nothing was removed."), "{output}");
        assert!(!output.contains("run it again"), "{output}");
        assert!(output.contains("1 item(s) would be removed:"), "{output}");
    }

    /// The command picks its audience from whether it has a terminal.
    #[test]
    fn the_command_is_addressed_to_a_terminal_only_when_it_has_one() {
        assert_eq!(PlanAudience::for_command(true), PlanAudience::Terminal);
        assert_eq!(PlanAudience::for_command(false), PlanAudience::Captured);
    }

    /// An empty plan has nothing to approve, so a dry run adds no next step for
    /// anyone; a real run adds none either, since its confirmation prompt follows.
    #[test]
    fn footer_is_only_for_a_dry_run_that_has_something_to_remove() {
        for audience in [
            PlanAudience::Terminal,
            PlanAudience::Captured,
            PlanAudience::Assistant,
        ] {
            let empty = render_uninstall_plan(&UninstallPlan::default(), &dry_run(), audience);
            assert!(!empty.contains("Nothing was removed."), "{empty}");
            let real_run = render_uninstall_plan(
                &plan_with_a_binary(),
                &UninstallOptions::default(),
                audience,
            );
            assert!(!real_run.contains("Nothing was removed."), "{real_run}");
        }
    }

    /// The assistant's preview has no command line to offer, so it must not point at a
    /// command line it cannot run; what it can truthfully say is that removal
    /// needs approval (`lifecycle_read_mutate_split_is_honest` in `tool_policy.rs` pins
    /// that its own `uninstall` is an approval action). It does not say "these
    /// items": the preview always plans with the default options, while the
    /// approval carries the user's own.
    #[test]
    fn assistant_dry_run_footer_names_no_command() {
        let output =
            render_uninstall_plan(&plan_with_a_binary(), &dry_run(), PlanAudience::Assistant);
        assert!(
            output.ends_with("\nNothing was removed. Removal needs your approval.\n"),
            "unexpected footer in:\n{output}"
        );
        assert!(!output.contains("rocm uninstall"), "{output}");
        assert!(!output.contains("these items"), "{output}");
    }

    fn test_shared_cache(path: PathBuf) -> super::SharedCache {
        super::SharedCache {
            path,
            kept: "the uv package cache is shared with other uv projects on this computer and is",
            name: "the uv package cache",
            loss: "It is shared with other uv projects on this computer.",
            pronoun: "it",
        }
    }

    fn planned_root(kind: &'static str, path: PathBuf) -> super::UninstallPlanEntry {
        super::UninstallPlanEntry {
            kind,
            path,
            planned: super::PlannedAs::Unchecked,
        }
    }

    /// A shared cache inside a root uninstall removes is deleted with it, so the
    /// plan must say so — loudly, with the flag that keeps it — rather than
    /// drop the note it would otherwise print. This is the `~/.cache` case:
    /// the warning used to vanish exactly when the cache was about to go.
    #[test]
    fn shared_cache_inside_a_removed_root_is_reported_as_deleted() {
        let root = std::env::temp_dir().join(format!("rocm-shared-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let cache = root.join("dot-cache");
        let inside = cache.join("uv");
        std::fs::create_dir_all(&inside).unwrap();

        let notes = super::shared_cache_notes_for(
            &[planned_root("cache", cache.clone())],
            &[test_shared_cache(inside.clone())],
        );
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(notes.len(), 1, "{notes:?}");
        let note = &notes[0];
        assert!(note.contains("WILL BE DELETED"), "{note}");
        assert!(note.contains(&inside.display().to_string()), "{note}");
        assert!(note.contains(&cache.display().to_string()), "{note}");
        // The advice is checked against the plan in
        // `uninstall_keep_flag_named_by_the_shared_cache_warning_keeps_it`.
        assert!(note.contains("--keep-cache"), "{note}");
    }

    /// The flag the warning names really keeps the cache: with it, the root
    /// leaves the plan and the same cache is reported as left in place.
    #[test]
    fn uninstall_keep_flag_named_by_the_shared_cache_warning_keeps_it() {
        let root = std::env::temp_dir().join(format!("rocm-shared-keep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("dot-cache"),
        };
        let inside = paths.cache_dir.join("uv");
        std::fs::create_dir_all(&inside).unwrap();
        let options = super::UninstallOptions {
            keep_binaries: true,
            keep_cache: true,
            ..super::UninstallOptions::default()
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &options,
            None,
        )
        .unwrap();
        let notes = super::shared_cache_notes_for(&plan.actions, &[test_shared_cache(inside)]);
        let _ = std::fs::remove_dir_all(&root);

        assert!(
            plan.actions.iter().all(|entry| entry.kind != "cache"),
            "{plan:?}"
        );
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("not removed"), "{notes:?}");
    }

    /// A root that is a symlink is unlinked, never followed, so a cache behind
    /// it survives and must not be reported as deleted.
    #[test]
    #[cfg(unix)]
    fn shared_cache_behind_a_linked_root_is_not_reported_as_deleted() {
        let root = std::env::temp_dir().join(format!("rocm-shared-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let real = root.join("real-cache");
        let inside = real.join("uv");
        std::fs::create_dir_all(&inside).unwrap();
        let link = root.join("cache-link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let notes = super::shared_cache_notes_for(
            &[planned_root("cache", link)],
            &[test_shared_cache(link_join_uv(&root))],
        );
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("not removed"), "{notes:?}");
    }

    #[cfg(unix)]
    fn link_join_uv(root: &Path) -> PathBuf {
        root.join("cache-link").join("uv")
    }

    /// A cache that exists outside everything being removed is reported, with
    /// its path, so the plan does not imply a clean slate it will not deliver.
    #[test]
    fn shared_cache_notes_report_paths_left_behind() {
        let root = std::env::temp_dir().join(format!("rocm-shared-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let outside = root.join("elsewhere").join("uv");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::create_dir_all(root.join("data")).unwrap();

        let notes = super::shared_cache_notes_for(
            &[planned_root("data", root.join("data"))],
            &[test_shared_cache(outside.clone())],
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
        let notes = super::shared_cache_notes_for(
            &[planned_root("cache", std::env::temp_dir())],
            &[test_shared_cache(missing)],
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    /// `ROCM_CLI_DATA_DIR=/` used to plan `- data: /`. The root is refused, the
    /// review names it, why, and the flag that clears it — and that flag does.
    /// The "nothing was removed" half of the message is proven where removal
    /// happens: `uninstall_refuses_and_removes_nothing_when_a_root_is_protected`
    /// below and the `uninstall-17` E2E scenario.
    #[test]
    fn uninstall_plan_refuses_the_filesystem_root() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-root-{}", std::process::id()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: PathBuf::from("/"),
            cache_dir: root.join("cache"),
        };
        let options = super::UninstallOptions {
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &options,
            None,
        )
        .unwrap();
        assert!(
            plan.actions
                .iter()
                .all(|entry| entry.path != Path::new("/")),
            "{plan:?}"
        );
        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(plan.refused[0].kind, "data");
        assert_eq!(
            plan.refused[0].why,
            rocm_core::ProtectedLocation::FilesystemRoot
        );
        let error = plan.refusal_error().expect("a refusal error");
        assert!(
            // The root prints in the host's own spelling (`\\` on Windows).
            error.contains(&format!(
                "data folder {} is the top of the filesystem",
                plan.refused[0].path.display()
            )),
            "{error}"
        );
        assert!(error.contains("--keep-data"), "{error}");
        let rendered = super::render_uninstall_plan(
            &plan,
            &super::UninstallOptions {
                dry_run: true,
                ..options
            },
            super::PlanAudience::Captured,
        );
        assert!(
            rendered.contains(&format!(
                "  - data: {} is the top of the filesystem",
                plan.refused[0].path.display()
            )),
            "{rendered}"
        );
        assert!(!rendered.contains("Choose Review uninstall"), "{rendered}");

        let keep = super::UninstallOptions {
            keep_data: true,
            ..options
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &keep,
            None,
        )
        .unwrap();
        assert!(
            plan.refused.is_empty(),
            "--keep-data must clear it: {plan:?}"
        );
    }

    /// The home folder itself — `setup.therock_venv` set to `$HOME` reaches the
    /// data root this way — is refused, and nothing at all is removed: not the
    /// home folder, and not the other roots either.
    #[test]
    // Unix only: it needs a folder outside the sandbox home that is not another
    // user's, and on Windows the temp folder is itself inside `C:\\Users\\<runner>`,
    // where everything outside the sandbox home counts as another user's home.
    #[cfg(unix)]
    fn uninstall_refuses_and_removes_nothing_when_a_root_is_protected() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let home = root.join("users").join("home");
        std::fs::create_dir_all(home.join("Documents")).unwrap();
        std::fs::write(home.join("Documents").join("thesis.txt"), b"keep").unwrap();
        // A config folder ROCm CLI created (marked), so only the home folder is
        // refused and the test can show that refusal still removes nothing.
        rocm_core::create_marked_root(&root.join("config")).unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: PathBuf::from(format!("{}/", home.display())),
            cache_dir: root.join("cache"),
        };
        let options = super::UninstallOptions {
            yes: true,
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &options,
            Some(&home),
        )
        .unwrap();
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let thesis_left = home.join("Documents").join("thesis.txt").is_file();
        let config_left = root.join("config").is_dir();
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(plan.refused[0].why, rocm_core::ProtectedLocation::Home);
        let error = format!("{:#}", outcome.expect_err("a refused plan must not apply"));
        assert!(error.contains("nothing was removed"), "{error}");
        assert!(error.contains("is your home folder"), "{error}");
        assert!(thesis_left, "the home folder was emptied");
        assert!(config_left, "a refused plan still removed another root");
    }

    fn refused_plan(root: &Path) -> super::UninstallPlan {
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: PathBuf::from("/"),
            cache_dir: root.join("cache"),
        };
        let sources = rocm_core::AppPathSources {
            data: rocm_core::AppPathSource::Env("ROCM_CLI_DATA_DIR".to_owned()),
            ..rocm_core::AppPathSources::default()
        };
        let options = super::UninstallOptions {
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };
        super::build_uninstall_plan_for_home(&paths, &sources, &options, None).unwrap()
    }

    /// The refusal is decided before anything else in the command, and in
    /// particular before the confirmation prompt: a refused uninstall must not
    /// ask "Proceed?" first.
    #[test]
    fn a_refused_uninstall_stops_before_the_prompt() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-prompt-{}", std::process::id()));
        let plan = refused_plan(&root);
        let options = super::UninstallOptions {
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };
        let mut asked = false;
        let outcome = crate::uninstall::run_uninstall_plan(
            &plan,
            &options,
            || true,
            || {
                asked = true;
                Ok(true)
            },
        );
        let error = format!("{:#}", outcome.expect_err("a refused plan must fail"));
        assert!(!asked, "the confirmation prompt ran before the refusal");
        assert!(
            error.contains("uninstall refused, nothing was removed"),
            "{error}"
        );
    }

    /// A plan nothing refused still needs `--yes` or a person to say yes, through
    /// the shared `require_yes_or_interactive` guard: with neither it is
    /// refused before the prompt, and at a terminal the prompt decides.
    /// Either way a "no" removes nothing.
    #[test]
    fn an_accepted_plan_needs_yes_or_a_terminal_answer() {
        let root = std::env::temp_dir().join(format!("rocm-accept-gate-{}", std::process::id()));
        let config = root.join("config");
        std::fs::create_dir_all(&config).unwrap();
        let plan = super::UninstallPlan {
            actions: vec![planned_root("config", config.clone())],
            ..super::UninstallPlan::default()
        };
        let options = super::UninstallOptions {
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };

        let mut asked = false;
        let no_terminal = crate::uninstall::run_uninstall_plan(
            &plan,
            &options,
            || false,
            || {
                asked = true;
                Ok(true)
            },
        );
        let asked_without_terminal = asked;
        let declined = crate::uninstall::run_uninstall_plan(
            &plan,
            &options,
            || true,
            || {
                asked = true;
                Ok(false)
            },
        );
        let config_left = config.is_dir();
        let _ = std::fs::remove_dir_all(&root);

        let error = format!("{:#}", no_terminal.expect_err("no --yes and no terminal"));
        assert_eq!(
            error,
            "uninstall requires --yes outside an interactive terminal"
        );
        assert!(!asked_without_terminal, "asked with no terminal to answer");
        assert!(declined.is_ok(), "{declined:?}");
        assert!(asked, "a terminal without --yes must be asked");
        assert!(
            config_left,
            "a refused or declined uninstall removed the config"
        );
    }

    /// A dry run of a refused plan fails too, so a script that previews before
    /// it removes does not read the refusal as a go-ahead.
    #[test]
    fn a_refused_dry_run_fails() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-dry-{}", std::process::id()));
        let plan = refused_plan(&root);
        let options = super::UninstallOptions {
            dry_run: true,
            keep_binaries: true,
            ..super::UninstallOptions::default()
        };
        let outcome =
            crate::uninstall::run_uninstall_plan(&plan, &options, || true, || panic!("no prompt"));
        let error = format!("{:#}", outcome.expect_err("a refused dry run must fail"));
        assert!(error.contains("would be refused"), "{error}");
    }

    /// The refusal names where the folder came from, and points at that
    /// setting as well as at the `--keep-` flag.
    #[test]
    fn a_refusal_names_the_setting_the_folder_came_from() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-source-{}", std::process::id()));
        let plan = refused_plan(&root);
        let error = plan.refusal_error().expect("refused");
        assert!(
            error.contains(&format!(
                "the data folder {} is the top of the filesystem (set by ROCM_CLI_DATA_DIR)",
                plan.refused[0].path.display()
            )),
            "{error}"
        );
        assert!(
            error.contains(
                "Point ROCM_CLI_DATA_DIR at a folder of ROCm CLI's own, or re-run with \
                 --keep-data to leave it in place."
            ),
            "{error}"
        );
    }

    /// The assistant's and dashboard's in-process `uninstall --dry-run` honours
    /// the `--keep-<kind>` flag the refusal advises, so following the advice
    /// there clears the refusal instead of repeating it.
    #[test]
    fn in_process_uninstall_dry_run_honours_the_advised_keep_flag() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-chat-{}", std::process::id()));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: PathBuf::from("/"),
            cache_dir: root.join("cache"),
        };
        let sources = rocm_core::AppPathSources::default();
        let args = |extra: &[&str]| -> Vec<String> {
            ["--dry-run", "--keep-binaries"]
                .iter()
                .chain(extra)
                .map(|arg| (*arg).to_owned())
                .collect()
        };

        let refused =
            super::render_uninstall_dry_run_with(&paths, &sources, &args(&[]), None).unwrap();
        assert!(refused.contains("would be refused"), "{refused}");
        assert!(refused.contains("--keep-data"), "{refused}");

        let kept =
            super::render_uninstall_dry_run_with(&paths, &sources, &args(&["--keep-data"]), None)
                .unwrap();
        assert!(!kept.contains("Refused"), "{kept}");
        assert!(!kept.contains("would be refused"), "{kept}");
        assert!(
            kept.contains("data removal disabled by command line flag"),
            "{kept}"
        );

        // And the in-process dispatcher hands it the flags.
        let dispatched = crate::chat::run_rocm_read_only_in_process(
            &paths,
            &[
                "uninstall".to_owned(),
                "--dry-run".to_owned(),
                "--keep-binaries".to_owned(),
                "--keep-data".to_owned(),
            ],
        )
        .unwrap();
        assert!(!dispatched.contains("Refused"), "{dispatched}");
    }

    /// Every `--keep-<kind>` the refusal can advise reaches the assistant's
    /// preview: with the refused root set to `/`, the flag for that kind, and
    /// only that one, clears the refusal.
    #[test]
    fn in_process_uninstall_dry_run_honours_every_keep_flag() {
        let root = std::env::temp_dir().join(format!("rocm-keep-each-{}", std::process::id()));
        let sources = rocm_core::AppPathSources::default();
        for kind in ["config", "data", "cache"] {
            let at = |own: &str| {
                if own == kind {
                    PathBuf::from("/")
                } else {
                    root.join(own)
                }
            };
            let paths = AppPaths {
                config_dir: at("config"),
                data_dir: at("data"),
                cache_dir: at("cache"),
            };
            for flag in ["--keep-config", "--keep-data", "--keep-cache"] {
                let args = ["--dry-run", "--keep-binaries", flag].map(str::to_owned);
                let text =
                    super::render_uninstall_dry_run_with(&paths, &sources, &args, None).unwrap();
                let clears = flag == format!("--keep-{kind}");
                assert_eq!(
                    !text.contains("would be refused"),
                    clears,
                    "{kind} root at /, {flag}:\n{text}"
                );
            }
        }
    }

    /// The approval-gated assistant/dashboard uninstall keeps the advised flag
    /// on its way to the real command.
    #[test]
    fn approved_uninstall_keeps_the_keep_flags() {
        let action = crate::chat::chat_rocm_command_action_from_args(vec![
            "uninstall".to_owned(),
            "--keep-data".to_owned(),
        ])
        .unwrap();
        let crate::chat::ChatRocmCommandAction::Approval { args, .. } = action else {
            panic!("a real uninstall must be approval-gated");
        };
        assert!(args.iter().any(|arg| arg == "--keep-data"), "{args:?}");
    }

    /// A relative path from this process's working directory to `target`, built
    /// from `..` segments, so a test can spell a folder relatively without
    /// changing the (process-global) working directory.
    #[cfg(unix)]
    fn relative_from_cwd(target: &Path) -> PathBuf {
        let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        let target = target.canonicalize().unwrap();
        let mut rel = PathBuf::new();
        for _ in cwd.components().skip(1) {
            rel.push("..");
        }
        rel.join(target.strip_prefix("/").unwrap())
    }

    /// The text check alone let these through: `.` run from home and `..` run
    /// from `~/proj` are relative, and home reached through a symlinked parent
    /// (the `/home -> var/home` layout) is spelled differently from `$HOME`.
    /// A real directory is also judged by where it really is.
    #[test]
    #[cfg(unix)]
    fn home_is_refused_however_it_is_reached() {
        let root = std::env::temp_dir().join(format!("rocm-refuse-reached-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let home = root.join("var").join("home").join("alice");
        std::fs::create_dir_all(home.join("proj")).unwrap();
        std::fs::write(home.join("keep.txt"), b"keep").unwrap();
        let _ = std::os::unix::fs::symlink(root.join("var").join("home"), root.join("home"));

        let spellings = [
            relative_from_cwd(&home),
            relative_from_cwd(&home.join("proj")).join(".."),
            root.join("home").join("alice"),
        ];
        let options = super::UninstallOptions {
            yes: true,
            keep_binaries: true,
            keep_config: true,
            keep_cache: true,
            ..super::UninstallOptions::default()
        };
        let mut failures = Vec::new();
        for data_dir in &spellings {
            let paths = AppPaths {
                config_dir: root.join("config"),
                data_dir: data_dir.clone(),
                cache_dir: root.join("cache"),
            };
            let plan = super::build_uninstall_plan_for_home(
                &paths,
                &rocm_core::AppPathSources::default(),
                &options,
                Some(&home),
            )
            .unwrap();
            let refused_as_home = plan.refused.iter().any(|refused| {
                refused.why == rocm_core::ProtectedLocation::Home && refused.resolved.is_some()
            });
            if !refused_as_home {
                failures.push(format!("{}: {plan:?}", data_dir.display()));
            }
            let _ = crate::uninstall::apply_uninstall_plan(&plan);
        }
        let kept = home.join("keep.txt").is_file();
        let _ = std::fs::remove_dir_all(&root);
        assert!(kept, "home was emptied");
        assert!(failures.is_empty(), "not refused as home: {failures:#?}");
    }

    fn keep_binaries() -> super::UninstallOptions {
        super::UninstallOptions {
            yes: true,
            keep_binaries: true,
            ..super::UninstallOptions::default()
        }
    }

    fn data_only(data_dir: PathBuf, root: &Path) -> AppPaths {
        AppPaths {
            config_dir: root.join("no-config"),
            data_dir,
            cache_dir: root.join("no-cache"),
        }
    }

    fn sandbox(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("rocm-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root.canonicalize().unwrap()
    }

    /// The review's dry run planned all of these for deletion. Each is refused
    /// now, judged on the real host folders (plan only: nothing is removed).
    #[test]
    #[cfg(target_os = "linux")]
    fn system_folders_the_review_found_are_refused() {
        let root = std::env::temp_dir();
        for (folder, why) in [
            ("/usr/local", rocm_core::ProtectedLocation::System),
            ("/etc/ssl", rocm_core::ProtectedLocation::System),
            ("/usr/bin", rocm_core::ProtectedLocation::System),
            ("/usr/lib", rocm_core::ProtectedLocation::System),
            ("/var/lib", rocm_core::ProtectedLocation::SharedTopLevel),
        ] {
            if !Path::new(folder).is_dir() {
                continue;
            }
            let plan = super::build_uninstall_plan_for_home(
                &data_only(PathBuf::from(folder), &root),
                &rocm_core::AppPathSources::default(),
                &keep_binaries(),
                Some(Path::new("/home/alice")),
            )
            .unwrap();
            assert!(plan.actions.is_empty(), "{folder}: {plan:?}");
            assert_eq!(plan.refused.len(), 1, "{folder}: {plan:?}");
            assert_eq!(plan.refused[0].why, why, "{folder}");
        }
    }

    /// A real directory outside home without the marker is refused, and the
    /// advice to create the marker is true: once it exists, the same plan goes
    /// ahead and removes the folder.
    #[test]
    // Unix only: it needs a folder outside the sandbox home that is not another
    // user's, and on Windows the temp folder is itself inside `C:\\Users\\<runner>`,
    // where everything outside the sandbox home counts as another user's home.
    #[cfg(unix)]
    fn creating_the_advised_marker_clears_the_refusal() {
        let root = sandbox("marker-advice");
        let home = root.join("users").join("home");
        std::fs::create_dir_all(&home).unwrap();
        let data = root.join("srv").join("rocm");
        std::fs::create_dir_all(&data).unwrap();
        let paths = data_only(data.clone(), &root);
        let sources = rocm_core::AppPathSources::default();

        let plan =
            super::build_uninstall_plan_for_home(&paths, &sources, &keep_binaries(), Some(&home))
                .unwrap();
        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(plan.refused[0].why, rocm_core::ProtectedLocation::NotMarked);
        let error = plan.refusal_error().unwrap();
        let marker = data.join(rocm_core::ROCM_CLI_ROOT_MARKER);
        let advised = format!(
            "f this folder belongs to ROCm CLI, create {}",
            marker.display()
        );
        assert!(error.contains(&advised), "{error}");

        // Follow the advice.
        std::fs::write(&marker, b"").unwrap();
        let plan =
            super::build_uninstall_plan_for_home(&paths, &sources, &keep_binaries(), Some(&home))
                .unwrap();
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let left = data.exists();
        let _ = std::fs::remove_dir_all(&root);
        assert!(plan.refused.is_empty(), "{plan:?}");
        assert!(outcome.is_ok(), "{outcome:?}");
        assert!(!left, "the marked folder was not removed");
    }

    /// "Point ROCM_CLI_DATA_DIR at a folder of ROCm CLI's own" is advice that
    /// works: the same uninstall with the variable repointed is not refused.
    #[test]
    fn pointing_the_named_variable_elsewhere_clears_the_refusal() {
        let root = sandbox("point-advice");
        let home = root.join("home");
        std::fs::create_dir_all(home.join(".rocm").join("data")).unwrap();
        let sources = rocm_core::AppPathSources {
            data: rocm_core::AppPathSource::Env("ROCM_CLI_DATA_DIR".to_owned()),
            ..rocm_core::AppPathSources::default()
        };
        let refused = super::build_uninstall_plan_for_home(
            &data_only(home.clone(), &root),
            &sources,
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let error = refused.refusal_error().unwrap();
        assert!(
            error.contains("Point ROCM_CLI_DATA_DIR at a folder of ROCm CLI's own"),
            "{error}"
        );

        let repointed = super::build_uninstall_plan_for_home(
            &data_only(home.join(".rocm").join("data"), &root),
            &sources,
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(repointed.refused.is_empty(), "{repointed:?}");
        assert_eq!(repointed.actions.len(), 1, "{repointed:?}");
    }

    /// When the folder came from `setup.therock_venv`, the advice leads with
    /// `--keep-data` and warns that moving the setting would orphan the SDK —
    /// and `--keep-data` clears the refusal.
    #[test]
    // Unix only: it needs a folder outside the sandbox home that is not another
    // user's, and on Windows the temp folder is itself inside `C:\\Users\\<runner>`,
    // where everything outside the sandbox home counts as another user's home.
    #[cfg(unix)]
    fn a_therock_venv_refusal_advises_keep_data_first() {
        let root = sandbox("venv-advice");
        let home = root.join("users").join("home");
        std::fs::create_dir_all(&home).unwrap();
        let sources = rocm_core::AppPathSources {
            data: rocm_core::AppPathSource::TheRockVenv(root.join("config.json")),
            ..rocm_core::AppPathSources::default()
        };
        let paths = data_only(home.clone(), &root);
        let plan =
            super::build_uninstall_plan_for_home(&paths, &sources, &keep_binaries(), Some(&home))
                .unwrap();
        let advice = plan.refusal_advice().unwrap();
        let keep = super::UninstallOptions {
            keep_data: true,
            ..keep_binaries()
        };
        let kept =
            super::build_uninstall_plan_for_home(&paths, &sources, &keep, Some(&home)).unwrap();
        // With a second way out (an unmarked prefix outside home), --keep-data
        // still comes first.
        let prefix = root.join("sdk");
        std::fs::create_dir_all(&prefix).unwrap();
        let unmarked = super::build_uninstall_plan_for_home(
            &data_only(prefix, &root),
            &sources,
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let unmarked_advice = unmarked.refusal_advice().unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(
            advice.starts_with("Re-run with --keep-data to leave it in place."),
            "{advice}"
        );
        assert!(
            unmarked_advice.starts_with("Re-run with --keep-data to leave it in place, or if"),
            "{unmarked_advice}"
        );
        assert!(advice.contains("would orphan the ROCm SDK"), "{advice}");
        assert!(!advice.contains("point setup.therock_venv"), "{advice}");
        assert!(kept.refused.is_empty(), "{kept:?}");
    }

    /// A real directory whose location cannot be resolved is refused rather
    /// than judged by its spelling alone.
    #[test]
    fn an_unresolvable_root_is_refused() {
        let root = sandbox("unresolvable");
        let home = root.join("home");
        std::fs::create_dir_all(home.join("data")).unwrap();
        let fail: &dyn Fn(&Path) -> std::io::Result<PathBuf> =
            &|_| Err(std::io::Error::other("cannot resolve"));
        let plan = super::build_uninstall_plan_with(
            &data_only(home.join("data"), &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
            fail,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(
            plan.refused[0].why,
            rocm_core::ProtectedLocation::Unresolvable
        );
    }

    /// HOME is spelled through a symlinked parent (`/home -> var/home`) and the
    /// data folder is the real path: only comparing against the RESOLVED home
    /// shows they are the same folder, and it is refused as home, not merely
    /// as an unmarked folder.
    #[test]
    #[cfg(unix)]
    fn the_real_home_is_refused_when_home_is_spelled_through_a_link() {
        let root = sandbox("resolved-home");
        let real_home = root.join("var").join("home").join("alice");
        std::fs::create_dir_all(&real_home).unwrap();
        std::os::unix::fs::symlink(root.join("var").join("home"), root.join("home")).unwrap();
        let home_as_set = root.join("home").join("alice");
        let plan = super::build_uninstall_plan_for_home(
            &data_only(real_home, &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home_as_set),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(plan.refused[0].why, rocm_core::ProtectedLocation::Home);
    }

    /// A system location that is itself a link (Silverblue: `/media -> run/media`)
    /// protects where it really is: a folder under its target is refused as a
    /// system location although the target is not on the list as written. The
    /// resolver is injected so the layout can be built in a sandbox.
    #[test]
    #[cfg(unix)]
    fn a_system_root_reached_through_its_link_target_is_refused() {
        let root = sandbox("linked-system-root");
        let home = root.join("users").join("home");
        let real_mnt = root.join("var").join("mnt");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(real_mnt.join("data")).unwrap();
        let mnt_target = real_mnt.clone();
        let resolve: &dyn Fn(&Path) -> std::io::Result<PathBuf> = &move |path: &Path| {
            if path == Path::new("/media") {
                Ok(mnt_target.clone())
            } else {
                rocm_core::canonicalize_for_compare(path)
            }
        };
        let plan = super::build_uninstall_plan_with(
            &data_only(real_mnt.join("data"), &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
            resolve,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(plan.refused.len(), 1, "{plan:?}");
        assert_eq!(plan.refused[0].why, rocm_core::ProtectedLocation::System);
    }

    /// A root spelled beside home that really is inside it — Windows spells the
    /// temp folder with the home folder's short 8.3 name (`RUNNER~1`) — is
    /// judged by where it really is and planned; a real folder in another
    /// user's home is still refused. Here a link stands in for the short name.
    #[test]
    #[cfg(unix)]
    fn a_root_spelled_beside_home_but_inside_it_is_planned() {
        let root = sandbox("short-name-home");
        let users = root.join("users");
        let home = users.join("home");
        let data = home.join(".rocm").join("data");
        std::fs::create_dir_all(&data).unwrap();
        let short = users.join("HOME~1");
        std::os::unix::fs::symlink(&home, &short).unwrap();
        let other = users.join("bob").join("data");
        std::fs::create_dir_all(&other).unwrap();

        let own = super::build_uninstall_plan_for_home(
            &data_only(short.join(".rocm").join("data"), &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let bobs = super::build_uninstall_plan_for_home(
            &data_only(other, &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);

        assert!(own.refused.is_empty(), "{own:?}");
        assert_eq!(own.actions.len(), 1, "{own:?}");
        assert_eq!(bobs.refused.len(), 1, "{bobs:?}");
        assert_eq!(
            bobs.refused[0].why,
            rocm_core::ProtectedLocation::OtherUsersHome
        );
    }

    /// The cases that bound the short-name allowance in
    /// [`super::judge_uninstall_root`], each judged directly with the real
    /// resolver unless it says otherwise. `users/` is home's parent, so by
    /// spelling alone everything under it but home is another user's.
    #[test]
    #[cfg(unix)]
    fn the_short_name_allowance_applies_only_to_a_real_folder_in_home() {
        use rocm_core::ProtectedLocation::{Home, OtherUsersHome, System};
        let root = sandbox("short-name-bounds");
        let users = root.join("users");
        let home = users.join("home");
        let data = home.join(".rocm").join("data");
        std::fs::create_dir_all(&data).unwrap();
        let elsewhere = root.join("elsewhere").join("data");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let resolve: Canonicalize<'_> = &rocm_core::canonicalize_for_compare;
        let judge = |path: &Path, home: &Path, resolve: Canonicalize<'_>| {
            let metadata = std::fs::symlink_metadata(path).unwrap();
            super::judge_uninstall_root(path, &metadata, Some(home), resolve)
        };
        let why = |judgment: &super::RootJudgment| match judgment {
            super::RootJudgment::Refuse { why, .. } => Some(*why),
            super::RootJudgment::Remove { .. } => None,
        };

        // A link spelled beside home, pointing into it, is still another
        // user's: only a real folder is judged by where it is.
        let link = users.join("LINK~1");
        std::os::unix::fs::symlink(&data, &link).unwrap();
        let as_link = judge(&link, &home, resolve);

        // A real folder spelled beside home that is home itself is refused
        // as the home folder, by where it really is.
        let alias = users.join("ALIAS~1");
        std::os::unix::fs::symlink(&users, &alias).unwrap();
        let home_itself = judge(&alias.join("home"), &home, resolve);

        // Home given through a link of its own still counts as home.
        let home_link = root.join("home-link");
        std::os::unix::fs::symlink(&home, &home_link).unwrap();
        let through_linked_home = judge(
            &alias.join("home").join(".rocm").join("data"),
            &home_link,
            resolve,
        );

        // A folder spelled under users/ that really is somewhere else
        // entirely keeps the refusal its spelling earns.
        let away = users.join("AWAY~1");
        std::os::unix::fs::symlink(elsewhere.parent().unwrap(), &away).unwrap();
        let spelled_away = judge(&away.join("data"), &home, resolve);

        // A system location is refused by its spelling even when it resolves
        // into home; the resolver is told so, as no real `/usr` does.
        let system = Path::new("/usr/rocm-short-name-test");
        let into_home = data.clone();
        let fake: &dyn Fn(&Path) -> std::io::Result<PathBuf> = &move |path: &Path| {
            if path == system {
                Ok(into_home.clone())
            } else {
                rocm_core::canonicalize_for_compare(path)
            }
        };
        let system_metadata = std::fs::symlink_metadata(&data).unwrap();
        let spelled_system =
            super::judge_uninstall_root(system, &system_metadata, Some(&home), fake);
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(why(&as_link), Some(OtherUsersHome), "{as_link:?}");
        assert_eq!(why(&home_itself), Some(Home), "{home_itself:?}");
        assert_eq!(why(&through_linked_home), None, "{through_linked_home:?}");
        assert_eq!(why(&spelled_away), Some(OtherUsersHome), "{spelled_away:?}");
        assert_eq!(why(&spelled_system), Some(System), "{spelled_system:?}");
    }

    /// With home at the top of the filesystem every folder is inside it, so
    /// the allowance decides nothing and the spelling's refusal stands.
    #[test]
    fn a_home_at_the_filesystem_root_grants_no_short_name_allowance() {
        let root = sandbox("short-name-root-home");
        let top = if cfg!(windows) {
            Path::new(r"C:\")
        } else {
            Path::new("/")
        };
        let resolve: Canonicalize<'_> = &rocm_core::canonicalize_for_compare;
        assert!(!super::resolves_inside_own_home(&root, Some(top), resolve));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A link named like the marker does not mark a folder: only the plain
    /// file ROCm CLI writes does.
    #[test]
    #[cfg(unix)]
    fn a_linked_marker_does_not_mark_a_folder() {
        let root = sandbox("linked-marker");
        let home = root.join("users").join("home");
        std::fs::create_dir_all(&home).unwrap();
        let data = root.join("srv").join("rocm");
        std::fs::create_dir_all(&data).unwrap();
        let elsewhere = root.join("any-file");
        std::fs::write(&elsewhere, b"").unwrap();
        std::os::unix::fs::symlink(&elsewhere, data.join(rocm_core::ROCM_CLI_ROOT_MARKER)).unwrap();
        let linked = super::build_uninstall_plan_for_home(
            &data_only(data.clone(), &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        std::fs::remove_file(data.join(rocm_core::ROCM_CLI_ROOT_MARKER)).unwrap();
        std::fs::write(data.join(rocm_core::ROCM_CLI_ROOT_MARKER), b"").unwrap();
        let marked = super::build_uninstall_plan_for_home(
            &data_only(data, &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(linked.refused.len(), 1, "{linked:?}");
        assert_eq!(
            linked.refused[0].why,
            rocm_core::ProtectedLocation::NotMarked
        );
        assert!(marked.refused.is_empty(), "{marked:?}");
    }

    /// Check, then act, for a root that was a symbolic link at review time:
    /// replaced by a real folder while the prompt was open, it must not be
    /// removed recursively; the same goes for a file that became a folder.
    #[test]
    #[cfg(unix)]
    fn a_link_or_file_root_that_became_a_folder_is_not_removed() {
        let root = sandbox("link-became-dir");
        let home = root.join("users").join("home");
        std::fs::create_dir_all(&home).unwrap();
        let target = root.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let link = home.join("cache");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let file = home.join("config");
        std::fs::write(&file, b"settings").unwrap();
        let paths = AppPaths {
            config_dir: file.clone(),
            data_dir: home.join("no-data"),
            cache_dir: link.clone(),
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        assert_eq!(plan.actions.len(), 2, "{plan:?}");

        // The swaps, as if they happened while the prompt was open.
        std::fs::remove_file(&link).unwrap();
        std::fs::create_dir_all(&link).unwrap();
        std::fs::write(link.join("precious.txt"), b"keep").unwrap();
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let precious = link.join("precious.txt").is_file();

        std::fs::remove_dir_all(&link).unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir_all(&file).unwrap();
        std::fs::write(file.join("precious.txt"), b"keep").unwrap();
        let file_outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let file_precious = file.join("precious.txt").is_file();
        let _ = std::fs::remove_dir_all(&root);

        assert!(
            precious,
            "a link replaced by a folder was removed recursively"
        );
        let error = format!("{:#}", outcome.expect_err("must stop"));
        assert!(
            error.contains("was a symbolic link when reviewed"),
            "{error}"
        );
        assert!(
            file_precious,
            "a file replaced by a folder was removed recursively"
        );
        let error = format!("{:#}", file_outcome.expect_err("must stop"));
        assert!(error.contains("was a file when reviewed"), "{error}");
    }

    /// A symlink root is unlinked only where the review saw it: if the folder
    /// holding it is swapped for a link to somewhere else, a link of the same
    /// name there is not touched.
    #[test]
    #[cfg(unix)]
    fn a_link_root_whose_folder_moved_is_not_unlinked() {
        let root = sandbox("link-parent-moved");
        let home = root.join("users").join("home");
        let holder = home.join("holder");
        std::fs::create_dir_all(&holder).unwrap();
        std::os::unix::fs::symlink(root.join("target"), holder.join("cache")).unwrap();
        let elsewhere = root.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(root.join("theirs"), elsewhere.join("cache")).unwrap();
        let plan = super::build_uninstall_plan_for_home(
            &AppPaths {
                config_dir: home.join("no-config"),
                data_dir: home.join("no-data"),
                cache_dir: holder.join("cache"),
            },
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        assert_eq!(plan.actions.len(), 1, "{plan:?}");
        std::fs::rename(&holder, home.join("holder-old")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &holder).unwrap();
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let theirs_left = std::fs::symlink_metadata(elsewhere.join("cache")).is_ok();
        let _ = std::fs::remove_dir_all(&root);
        assert!(
            theirs_left,
            "the link in the swapped-in folder was unlinked"
        );
        let error = format!("{:#}", outcome.expect_err("must stop"));
        assert!(error.contains("folder holding it now resolves"), "{error}");
    }

    /// A planned folder that is already gone when its turn comes (inside one
    /// removed earlier, or removed by hand during the prompt) is skipped, and
    /// the rest of the plan still runs.
    #[test]
    fn a_planned_folder_already_gone_is_skipped() {
        let root = sandbox("already-gone");
        let home = root.join("users").join("home");
        let config = home.join(".rocm").join("config");
        let data = home.join(".rocm").join("data");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        let paths = AppPaths {
            config_dir: config.clone(),
            data_dir: data.clone(),
            cache_dir: home.join("no-cache"),
        };
        let plan = super::build_uninstall_plan_for_home(
            &paths,
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        assert_eq!(plan.actions.len(), 2, "{plan:?}");
        std::fs::remove_dir_all(&config).unwrap();
        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let data_left = data.exists();
        let _ = std::fs::remove_dir_all(&root);
        assert!(outcome.is_ok(), "{outcome:?}");
        assert!(!data_left, "the rest of the plan did not run");
    }

    /// Check, then act: a parent swapped for a symlink between the review and
    /// the removal must not redirect the removal to another folder.
    #[test]
    #[cfg(unix)]
    fn a_root_that_moved_after_the_review_is_not_removed() {
        let root = sandbox("check-then-act");
        let home = root.join("home");
        let parent = home.join("parent");
        std::fs::create_dir_all(parent.join("data")).unwrap();
        let elsewhere = root.join("elsewhere");
        std::fs::create_dir_all(elsewhere.join("data")).unwrap();
        std::fs::write(elsewhere.join("data").join("precious"), b"keep").unwrap();
        let plan = super::build_uninstall_plan_for_home(
            &data_only(parent.join("data"), &root),
            &rocm_core::AppPathSources::default(),
            &keep_binaries(),
            Some(&home),
        )
        .unwrap();
        assert_eq!(plan.actions.len(), 1, "{plan:?}");

        // The swap, as if it happened while the prompt was open.
        std::fs::rename(&parent, home.join("parent-old")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &parent).unwrap();

        let outcome = crate::uninstall::apply_uninstall_plan(&plan);
        let precious = elsewhere.join("data").join("precious").is_file();
        let _ = std::fs::remove_dir_all(&root);
        assert!(precious, "the removal followed the swapped-in link");
        let error = format!(
            "{:#}",
            outcome.expect_err("a moved root must stop the removal")
        );
        assert!(error.contains("now resolves to"), "{error}");
    }

    /// `discover_with_sources` against real environment and config state:
    /// `setup.therock_venv` moves the data and cache folders and is named as
    /// their source; a parent's forwarded sources win for pinned folders.
    #[test]
    fn discover_with_sources_reads_the_real_environment_and_config() {
        let root = sandbox("discover-sources");
        let config = root.join("config");
        let sdk = root.join("sdk");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(
            config.join("config.json"),
            serde_json::to_vec(&json!({ "setup": { "therock_venv": sdk } })).unwrap(),
        )
        .unwrap();
        // Every lock that guards the `ROCM_CLI_*` variables: `ScopedTestEnv`
        // has its own, other tests that set them hold the shared process lock,
        // and a built-in engine call rewrites them under the engine lock.
        let _process_env = crate::test_support::PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _engine_env = crate::engine_ipc::builtin_engine_env_test_lock();
        let mut env = ScopedTestEnv::new();
        env.set("ROCM_CLI_CONFIG_DIR", &config.display().to_string());
        env.clear("ROCM_CLI_DATA_DIR");
        env.clear("ROCM_CLI_CACHE_DIR");
        env.clear(rocm_core::APP_PATH_SOURCES_ENV);
        let (paths, sources) = AppPaths::discover_with_sources().unwrap();

        // A parent pins the data folder and forwards where it came from.
        let forwarded_json =
            serde_json::to_string(&rocm_core::ForwardedAppPathSources::for_paths(&paths)).unwrap();
        env.set("ROCM_CLI_DATA_DIR", &paths.data_dir.display().to_string());
        env.set(rocm_core::APP_PATH_SOURCES_ENV, &forwarded_json);
        let (_, forwarded) = AppPaths::discover_with_sources().unwrap();
        // Stale or forged: the variable now names a different folder than
        // the forwarded entry describes, so the entry is ignored.
        env.set(
            "ROCM_CLI_DATA_DIR",
            &root.join("elsewhere").display().to_string(),
        );
        let (_, stale) = AppPaths::discover_with_sources().unwrap();
        env.set("ROCM_CLI_DATA_DIR", &paths.data_dir.display().to_string());
        env.clear(rocm_core::APP_PATH_SOURCES_ENV);
        let (_, unforwarded) = AppPaths::discover_with_sources().unwrap();
        drop(env);
        assert_eq!(
            stale.data,
            rocm_core::AppPathSource::Env("ROCM_CLI_DATA_DIR".to_owned())
        );
        let _ = std::fs::remove_dir_all(&root);

        let venv = rocm_core::AppPathSource::TheRockVenv(config.join("config.json"));
        assert_eq!(paths.data_dir, sdk);
        assert_eq!(
            sources.config,
            rocm_core::AppPathSource::Env("ROCM_CLI_CONFIG_DIR".to_owned())
        );
        assert_eq!(sources.data, venv);
        assert_eq!(sources.cache, venv);
        assert_eq!(forwarded.data, venv);
        assert_eq!(forwarded.cache, venv);
        assert_eq!(
            unforwarded.data,
            rocm_core::AppPathSource::Env("ROCM_CLI_DATA_DIR".to_owned())
        );
        // With no cache setting, the cache folder follows the data folder, so
        // the data folder's setting is named for it, not the default location.
        assert_eq!(
            unforwarded.cache,
            rocm_core::AppPathSource::Env("ROCM_CLI_DATA_DIR".to_owned())
        );
    }

    /// The dashboard and assistant path: with `setup.therock_venv` set to the
    /// home folder, `rocm_command uninstall --dry-run` (in-process) names
    /// `setup.therock_venv` and advises `--keep-data`, never a variable the
    /// user did not set; and a `rocm` child started for an approved command is
    /// told the same sources.
    #[test]
    // Unix only: it needs a folder outside the sandbox home that is not another
    // user's, and on Windows the temp folder is itself inside `C:\\Users\\<runner>`,
    // where everything outside the sandbox home counts as another user's home.
    #[cfg(unix)]
    fn the_dashboard_path_names_the_setting_the_user_made() {
        let root = sandbox("dash-sources");
        let home = root.join("home");
        let config = root.join("config");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(
            config.join("config.json"),
            serde_json::to_vec(&json!({ "setup": { "therock_venv": home } })).unwrap(),
        )
        .unwrap();
        // Every lock that guards the `ROCM_CLI_*` variables: `ScopedTestEnv`
        // has its own, other tests that set them hold the shared process lock,
        // and a built-in engine call rewrites them under the engine lock.
        let _process_env = crate::test_support::PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _engine_env = crate::engine_ipc::builtin_engine_env_test_lock();
        let mut env = ScopedTestEnv::new();
        env.set("HOME", &home.display().to_string());
        env.set("ROCM_CLI_CONFIG_DIR", &config.display().to_string());
        env.set(
            "ROCM_CLI_CACHE_DIR",
            &root.join("cache").display().to_string(),
        );
        env.clear("ROCM_CLI_DATA_DIR");
        env.clear(rocm_core::APP_PATH_SOURCES_ENV);
        let paths = AppPaths::discover().unwrap();
        let shown = crate::chat::run_internal_mcp_call(
            &paths,
            "rocm_command",
            json!({ "args": ["uninstall", "--dry-run", "--keep-binaries"] }),
            false,
        )
        .unwrap();
        let child = crate::engine_runtime_select::rocm_child_app_path_env(&paths);
        // An interactive shell or an engine gets the folders, not the sources.
        let mut shell = std::process::Command::new("true");
        paths.apply_env(&mut shell);
        let shell_gets_sources = shell
            .get_envs()
            .any(|(key, _)| key == rocm_core::APP_PATH_SOURCES_ENV);
        assert!(
            !shell_gets_sources,
            "sources were passed to a non-rocm child"
        );
        let forwarded = child
            .iter()
            .find(|(key, _)| *key == rocm_core::APP_PATH_SOURCES_ENV)
            .map(|(_, value)| value.to_string_lossy().into_owned());
        drop(env);
        let _ = std::fs::remove_dir_all(&root);

        let text = shown.to_string();
        assert!(text.contains("set by setup.therock_venv"), "{text}");
        assert!(text.contains("Re-run with --keep-data"), "{text}");
        assert!(!text.contains("ROCM_CLI_DATA_DIR"), "{text}");
        let forwarded: rocm_core::ForwardedAppPathSources =
            serde_json::from_str(&forwarded.expect("sources forwarded")).unwrap();
        assert!(
            matches!(
                forwarded.data.source,
                rocm_core::AppPathSource::TheRockVenv(_)
            ),
            "{forwarded:?}"
        );
        assert_eq!(forwarded.data.path, paths.data_dir);
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
        let plan = build_uninstall_plan(&paths, &rocm_core::AppPathSources::default(), &options);
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

    fn set_helper_facts(paths: &AppPaths, watcher_enabled: bool, helper_running: bool) {
        let mut config = rocm_core::RocmCliConfig::default();
        config.watcher_config_mut("server-recover").enabled = watcher_enabled;
        config.save(paths).expect("save config");
        rocm_core::AutomationRuntimeState {
            running: helper_running,
            automations_enabled: true,
            daemon_pid: std::process::id(),
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: Vec::new(),
        }
        .write(paths)
        .expect("write runtime state");
    }

    /// ROCMAI-827: uninstall does not stop the automation helper, and the plan
    /// says so only where it is true. The exact warning for each state: a helper
    /// that stays up (a watcher is enabled) is named with the command that ends it
    /// and "uninstall does not stop it"; one that is idle and about to exit says
    /// only that, since uninstall leaving it is not the whole story; none running
    /// is not mentioned.
    #[test]
    fn plan_warns_about_a_helper_uninstall_leaves_running_only_when_there_is_one() {
        let (root, paths) = test_paths("uninstall-helper-warning");
        paths.ensure().expect("create app dirs");
        let options = UninstallOptions {
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        let helper_warnings = |plan: &UninstallPlan| -> Vec<String> {
            plan.warnings
                .iter()
                .filter(|w| w.contains("background automation helper"))
                .cloned()
                .collect()
        };

        set_helper_facts(&paths, false, false);
        let none = helper_warnings(
            &build_uninstall_plan(&paths, &rocm_core::AppPathSources::default(), &options)
                .expect("plan"),
        );
        set_helper_facts(&paths, true, false);
        let enabled_dead = helper_warnings(
            &build_uninstall_plan(&paths, &rocm_core::AppPathSources::default(), &options)
                .expect("plan"),
        );
        set_helper_facts(&paths, false, true);
        let idle = helper_warnings(
            &build_uninstall_plan(&paths, &rocm_core::AppPathSources::default(), &options)
                .expect("plan"),
        );
        set_helper_facts(&paths, true, true);
        let outlives = helper_warnings(
            &build_uninstall_plan(&paths, &rocm_core::AppPathSources::default(), &options)
                .expect("plan"),
        );
        let _ = std::fs::remove_dir_all(&root);

        assert!(none.is_empty(), "{none:?}");
        assert!(
            enabled_dead.is_empty(),
            "nothing is running to leave: {enabled_dead:?}"
        );
        assert_eq!(
            idle,
            [
                "the background automation helper (`rocm daemon`) is still running; no automation is enabled, so it exits on its own within a few seconds"
            ]
        );
        assert_eq!(
            outlives,
            [
                "the background automation helper (`rocm daemon`) keeps running while an automation is enabled (server-recover); to stop it, run `rocm automations disable server-recover`, after which it exits on its own within a few seconds; uninstall does not stop it"
            ]
        );
    }
}
