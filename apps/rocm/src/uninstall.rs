// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm uninstall` command handler.
//!
//! The plan itself is built by `build_uninstall_plan` in the crate root, which
//! `rocm storage` shares the plan and removal shape with. This module owns what
//! only uninstall does on top of removing files:
//!
//! - undoing the PATH change the installer recorded making;
//! - deleting the running `rocm.exe` on Windows, which cannot unlink itself.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, interactive_terminal};

use crate::{
    UninstallOptions, UninstallPlan, build_uninstall_plan, confirm_uninstall,
    is_rocm_install_entry_name, remove_path, render_uninstall_plan,
};

/// The file, next to the installed binaries, in which `install.sh` and
/// `install.ps1` record the PATH change they made, one per line:
/// `profile <shell profile>` and `path-expr <how the block spells the folder>`
/// from `install.sh`, `user-path <dir>` from `install.ps1`.
pub(crate) const PATH_EDIT_RECORD: &str = ".rocm-cli-path-edit";

const PROFILE_BLOCK_START: &str = "# >>> rocm-cli path >>>";
const PROFILE_BLOCK_END: &str = "# <<< rocm-cli path <<<";

/// A PATH change the installer made that uninstall will undo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PathEdit {
    /// Remove the marked block `install.sh` appended to a shell profile.
    ShellProfile {
        profile: PathBuf,
        /// The spellings of the install folder the block may use: the ones
        /// the installer recorded, the literal path, and the `$HOME/...` form.
        /// Kept as the OS spells them, since a path need not be UTF-8.
        path_exprs: Vec<OsString>,
    },
    /// Remove the entry `install.ps1` added to the user `Path`.
    UserPath { dir: PathBuf },
}

impl PathEdit {
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::ShellProfile { profile, .. } => {
                format!(
                    "shell profile: the rocm-cli PATH lines in {}",
                    profile.display()
                )
            }
            Self::UserPath { dir } => format!("user PATH: {}", dir.display()),
        }
    }
}

pub(crate) fn uninstall(options: UninstallOptions) -> Result<()> {
    let paths = AppPaths::discover()?;
    let plan = build_uninstall_plan(&paths, &options)?;
    print!("{}", render_uninstall_plan(&plan, &options));

    if plan_is_empty(&plan) || options.dry_run {
        return Ok(());
    }

    if !options.yes {
        if !interactive_terminal() {
            bail!("uninstall requires --yes outside an interactive terminal");
        }
        if !confirm_uninstall()? {
            println!("uninstall cancelled");
            return Ok(());
        }
    }

    execute_plan(&plan, &mut |line| println!("{line}"))?;
    println!("uninstall complete");
    Ok(())
}

/// Carry out `plan`, handing `report` a line for each step as soon as it is
/// done, so a failure partway still leaves the user told what already changed.
///
/// The PATH changes are undone first, so a change that cannot be undone stops
/// the uninstall before anything is removed: `rocm`, the user's settings, and
/// the record of the change are all still there to retry with. Then the
/// config, data, and cache go, while `rocm` is still there to retry a failure
/// with; then the binaries, and the record only once every change is undone.
fn execute_plan(plan: &UninstallPlan, report: &mut dyn FnMut(String)) -> Result<()> {
    for edit in &plan.path_edits {
        match apply_path_edit(edit) {
            Ok(line) => report(line),
            Err(error) => bail!(
                "could not undo the installer's PATH change ({}): {error:#}\n\
                 Nothing else was removed, so you can fix this and run `rocm uninstall` again.",
                edit.describe()
            ),
        }
    }

    let is_binary = |kind: &str| kind == "binary";
    let non_binaries = plan.actions.iter().filter(|entry| !is_binary(entry.kind));
    let binaries = plan.actions.iter().filter(|entry| is_binary(entry.kind));
    for entry in non_binaries.chain(binaries) {
        remove_path(&entry.path)
            .with_context(|| format!("failed to remove {}", entry.path.display()))?;
        report(format!("removed {} {}", entry.kind, entry.path.display()));
    }
    if let Some(record) = &plan.path_record {
        remove_path(record).with_context(|| format!("failed to remove {}", record.display()))?;
        report(format!("removed install record {}", record.display()));
    }

    if let Some(exe) = &plan.self_delete {
        report(delete_running_executable(exe)?);
    }
    Ok(())
}

pub(crate) const fn plan_is_empty(plan: &UninstallPlan) -> bool {
    plan.actions.is_empty()
        && plan.path_edits.is_empty()
        && plan.path_record.is_none()
        && plan.self_delete.is_none()
}

/// How many entries the review lists as removed.
pub(crate) fn removal_count(plan: &UninstallPlan) -> usize {
    plan.actions.len()
        + usize::from(plan.path_record.is_some())
        + usize::from(plan.self_delete.is_some())
}

/// Append the parts of the review only uninstall has: the install record, the
/// running executable, and the PATH changes it undoes.
pub(crate) fn render_extra_plan(output: &mut String, plan: &UninstallPlan) {
    if let Some(record) = &plan.path_record {
        let _ = writeln!(output, "  - install record: {}", record.display());
    }
    if let Some(exe) = &plan.self_delete {
        let _ = writeln!(
            output,
            "  - binary: {} (the running program; deleted as this command exits)",
            exe.display()
        );
    }
    if !plan.path_edits.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "PATH changes from the installer that would be undone:"
        );
        for edit in &plan.path_edits {
            let _ = writeln!(output, "  - {}", edit.describe());
        }
    }
}

// ---------------------------------------------------------------------------
// PATH changes
// ---------------------------------------------------------------------------

/// Plan undoing the PATH change the installer made for `bin_dir`, the folder
/// whose ROCm CLI entries `removed` lists.
///
/// Only a change the installer recorded (or the shell-profile block it marks as
/// its own) is undone, never a PATH line the user wrote. And it is left in place
/// when other programs live in the same folder, since they may depend on it
/// being on PATH; the record then stays too, so a later uninstall still knows.
pub(crate) fn plan_path_edits(
    plan: &mut UninstallPlan,
    bin_dir: &Path,
    removed: &[PathBuf],
    home: Option<&Path>,
) {
    let record_path = bin_dir.join(PATH_EDIT_RECORD);
    // Read as bytes: the installer writes paths as the shell spells them, and
    // a home folder need not be UTF-8.
    let record = match fs::read(&record_path) {
        Ok(record) => Some(record),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            plan.warnings.push(format!(
                "could not read {}, so the installer's PATH change is left in place: {error}",
                record_path.display()
            ));
            return;
        }
    };
    let entries = record
        .as_deref()
        .map(parse_path_edit_record)
        .unwrap_or_default();

    let mut edits = Vec::new();
    // A profile that could not be read may still hold a block, so the record
    // that names it must outlive this uninstall.
    let mut unreadable = false;
    if cfg!(windows) {
        for (kind, target) in &entries {
            if *kind == "user-path" {
                edits.push(PathEdit::UserPath {
                    dir: PathBuf::from(target),
                });
            }
        }
        let on_path = std::env::var("PATH").is_ok_and(|path| {
            rocm_core::user_path::remove_path_list_entry(&path, bin_dir).is_some()
        });
        if edits.is_empty() && on_path {
            plan.skipped.push(format!(
                "{} is on PATH, but this installation did not record adding it, so it is left in place",
                bin_dir.display()
            ));
        }
    } else {
        let mut path_exprs = profile_path_exprs(bin_dir, home);
        for (kind, target) in &entries {
            if *kind == "path-expr" && !path_exprs.contains(target) {
                path_exprs.push(target.clone());
            }
        }
        let recorded: Vec<PathBuf> = entries
            .iter()
            .filter(|(kind, _)| *kind == "profile")
            .map(|(_, target)| PathBuf::from(target))
            .collect();
        // The record names the profile of the latest install, but an earlier
        // install from another shell may have marked a different one, and
        // installs from before the record named none. The marked block is the
        // installer's own wherever it is, so look everywhere it writes one.
        let mut candidates = recorded.clone();
        for profile in home.map(legacy_profile_candidates).unwrap_or_default() {
            if !candidates.contains(&profile) {
                candidates.push(profile);
            }
        }
        for profile in candidates {
            match profile_has_block(&profile, &path_exprs) {
                Ok(true) => edits.push(PathEdit::ShellProfile {
                    profile,
                    path_exprs: path_exprs.clone(),
                }),
                Ok(false) if recorded.contains(&profile) => plan.skipped.push(format!(
                    "{} no longer has the rocm-cli PATH lines, so there is nothing to undo there",
                    profile.display()
                )),
                Ok(false) => {}
                // The installer recorded writing its block here, so it is
                // most likely still there: undoing it is attempted like any
                // other, and if the profile still cannot be read then, the
                // uninstall stops before removing anything, as for any PATH
                // change it cannot undo.
                Err(error) if recorded.contains(&profile) => {
                    plan.warnings.push(format!(
                        "could not read {}, which the installer recorded adding its PATH lines to: {error:#}\n\
                         Unless it can be read when the uninstall runs, nothing will be removed.",
                        profile.display()
                    ));
                    edits.push(PathEdit::ShellProfile {
                        profile,
                        path_exprs: path_exprs.clone(),
                    });
                }
                Err(error) => {
                    unreadable = true;
                    plan.warnings.push(format!(
                        "could not read {}, so any rocm-cli PATH lines in it are left in place: {error:#}",
                        profile.display()
                    ));
                }
            }
        }
    }

    if edits.is_empty() {
        if record.is_some() && !unreadable {
            plan.path_record = Some(record_path);
        }
        return;
    }
    match other_entries(bin_dir, removed) {
        Ok(others) if others.is_empty() => {
            plan.path_edits = edits;
            if record.is_some() && !unreadable {
                plan.path_record = Some(record_path);
            }
        }
        Ok(others) => {
            let kept_record = if record.is_some() {
                format!(
                    "; its record {} stays with it, so a later uninstall can still undo it",
                    record_path.display()
                )
            } else {
                String::new()
            };
            plan.skipped.push(format!(
                "PATH change left in place: {} also holds {}, which may rely on it being on PATH{kept_record}",
                bin_dir.display(),
                others.join(", ")
            ));
        }
        Err(error) => plan.warnings.push(format!(
            "could not list {} to see whether other programs rely on it being on PATH, so the \
             installer's PATH change is left in place: {error}",
            bin_dir.display()
        )),
    }
}

/// The `kind target` lines of a PATH-edit record. A target is kept as the
/// bytes the installer wrote, so a path that is not UTF-8 still names its file.
fn parse_path_edit_record(record: &[u8]) -> Vec<(&str, OsString)> {
    record
        .split(|byte| *byte == b'\n')
        .filter_map(|line| {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let space = line.iter().position(|byte| *byte == b' ')?;
            let (kind, target) = (&line[..space], &line[space + 1..]);
            let kind = std::str::from_utf8(kind).ok()?;
            (!target.trim_ascii().is_empty()).then(|| (kind, os_string_from_bytes(target)))
        })
        .collect()
}

#[cfg(unix)]
fn os_string_from_bytes(bytes: &[u8]) -> OsString {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;
    OsStr::from_bytes(bytes).to_owned()
}

/// `install.ps1` writes the record as UTF-8.
#[cfg(not(unix))]
fn os_string_from_bytes(bytes: &[u8]) -> OsString {
    OsString::from(String::from_utf8_lossy(bytes).into_owned())
}

/// How `install.sh` may have spelled `bin_dir` in the block: `$HOME/...` when it
/// is under the home directory, and always the literal path. Each is tried as
/// given and with links resolved, since the running binary reports its folder
/// resolved while the installer wrote it as the user's `$HOME` named it.
fn profile_path_exprs(bin_dir: &Path, home: Option<&Path>) -> Vec<OsString> {
    let resolve = |path: &Path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut bins = vec![bin_dir.to_path_buf()];
    let resolved_bin = resolve(bin_dir);
    if resolved_bin != bin_dir {
        bins.push(resolved_bin);
    }
    let mut homes: Vec<PathBuf> = home.map(Path::to_path_buf).into_iter().collect();
    if let Some(home) = home {
        let resolved_home = resolve(home);
        if resolved_home != home {
            homes.push(resolved_home);
        }
    }

    let mut exprs = Vec::new();
    let mut push = |expr: OsString| {
        if !exprs.contains(&expr) {
            exprs.push(expr);
        }
    };
    for bin in &bins {
        push(bin.as_os_str().to_owned());
        for home in &homes {
            if bin == home {
                push(OsString::from("$HOME"));
            } else if let Ok(relative) = bin.strip_prefix(home) {
                let mut expr = OsString::from("$HOME/");
                expr.push(relative.as_os_str());
                push(expr);
            }
        }
    }
    exprs
}

/// Every profile `install.sh` can pick, by shell, under `home`.
fn legacy_profile_candidates(home: &Path) -> Vec<PathBuf> {
    vec![
        home.join(".bashrc"),
        home.join(".zshrc"),
        home.join(".config").join("fish").join("config.fish"),
        home.join(".kshrc"),
        home.join(".profile"),
    ]
}

/// Whether `profile` holds the installer's block for one of `path_exprs`. A
/// profile that does not exist has none; one that cannot be read is an error,
/// not an absence.
fn profile_has_block(profile: &Path, path_exprs: &[OsString]) -> Result<bool> {
    match fs::read(profile) {
        Ok(bytes) => Ok(remove_profile_block_bytes(&bytes, path_exprs).is_some()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// `text` without the installer's marked PATH block for one of `path_exprs`, or
/// `None` when it has none.
///
/// A block naming a different folder belongs to another install and stays. The
/// blank line the installer writes in front of the block goes with it, so the
/// profile reads back as it did before the install. Works on bytes: a profile
/// is not required to be UTF-8, and every byte outside the block is kept.
fn remove_profile_block_bytes(text: &[u8], path_exprs: &[OsString]) -> Option<Vec<u8>> {
    let lines: Vec<&[u8]> = text.split_inclusive(|byte| *byte == b'\n').collect();
    let contains = |haystack: &[u8], needle: &[u8]| {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    };
    // The block quotes the folder the way `install.sh`'s
    // `escape_for_double_quotes` does.
    let needles: Vec<[Vec<u8>; 2]> = path_exprs
        .iter()
        .map(|expr| {
            let mut escaped = Vec::new();
            for byte in expr.as_encoded_bytes() {
                if matches!(byte, b'\\' | b'"') {
                    escaped.push(b'\\');
                }
                escaped.push(*byte);
            }
            let quoted = [&b"\""[..], &escaped, b"\""].concat();
            let listed = [&b"\""[..], &escaped, b":"].concat();
            [quoted, listed]
        })
        .collect();
    let names_dir = |body: &[u8]| {
        needles
            .iter()
            .flatten()
            .any(|needle| contains(body, needle))
    };
    let mut kept: Vec<&[u8]> = Vec::with_capacity(lines.len());
    let mut removed = false;
    let mut index = 0;
    while index < lines.len() {
        if lines[index].trim_ascii_end() == PROFILE_BLOCK_START.as_bytes() {
            let end = (index + 1..lines.len()).find(|&candidate| {
                lines[candidate].trim_ascii_end() == PROFILE_BLOCK_END.as_bytes()
            });
            if let Some(end) = end
                && names_dir(&lines[index + 1..end].concat())
            {
                if kept.last().is_some_and(|line| line.trim_ascii().is_empty()) {
                    kept.pop();
                }
                removed = true;
                index = end + 1;
                continue;
            }
        }
        kept.push(lines[index]);
        index += 1;
    }
    removed.then(|| kept.concat())
}

/// [`remove_profile_block_bytes`] for text, so tests read naturally.
#[cfg(test)]
fn remove_profile_block(text: &str, path_exprs: &[OsString]) -> Option<String> {
    remove_profile_block_bytes(text.as_bytes(), path_exprs)
        .map(|bytes| String::from_utf8(bytes).expect("removing whole lines keeps UTF-8"))
}

/// Names of entries in `bin_dir` that are not ROCm CLI's and are not being
/// removed: anything there that PATH may still be serving.
///
/// A folder or entry that cannot be read is an error, not an absence: the
/// caller then keeps the PATH change, since a program it cannot see may still
/// rely on it. A name that is not UTF-8 is not ROCm CLI's, so it counts.
fn other_entries(bin_dir: &Path, removed: &[PathBuf]) -> std::io::Result<Vec<String>> {
    let mut others = Vec::new();
    for entry in fs::read_dir(bin_dir)? {
        let entry = entry?;
        if removed.contains(&entry.path()) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_rocm_install_entry_name(&name) {
            others.push(name);
        }
    }
    others.sort();
    Ok(others)
}

fn apply_path_edit(edit: &PathEdit) -> Result<String> {
    match edit {
        PathEdit::ShellProfile {
            profile,
            path_exprs,
        } => {
            let text = fs::read(profile)
                .with_context(|| format!("failed to read {}", profile.display()))?;
            let Some(updated) = remove_profile_block_bytes(&text, path_exprs) else {
                return Ok(format!(
                    "{} no longer had the rocm-cli PATH lines",
                    profile.display()
                ));
            };
            // A profile kept as a link (a dotfiles manager's, say) is edited
            // where it lives, so the link stays a link.
            let target = fs::canonicalize(profile)
                .with_context(|| format!("failed to resolve {}", profile.display()))?;
            write_replacing(&target, &updated)?;
            Ok(format!(
                "removed the rocm-cli PATH lines from {}; new terminals will not have them",
                profile.display()
            ))
        }
        PathEdit::UserPath { dir } => remove_user_path(dir),
    }
}

#[cfg(windows)]
fn remove_user_path(dir: &Path) -> Result<String> {
    Ok(if rocm_core::user_path::remove_user_path_entry(dir)? {
        format!(
            "removed {} from your user PATH; new terminals will not have it",
            dir.display()
        )
    } else {
        format!("{} was no longer on your user PATH", dir.display())
    })
}

#[cfg(not(windows))]
fn remove_user_path(dir: &Path) -> Result<String> {
    // Only planned on Windows; see `plan_path_edits`.
    bail!("{} is a Windows user PATH entry", dir.display())
}

/// Replace `path`'s contents in one step, keeping its permissions, so an
/// interrupted write never leaves a truncated shell profile behind.
///
/// The replacement is written beside `path` as `<name>.rocm-uninstall.tmp`. It
/// never has wider permissions than `path`, even while it is being written, so
/// a profile only its owner can read never briefly becomes readable to others.
fn write_replacing(path: &Path, contents: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let mut temp_name = file_name.to_owned();
    temp_name.push(".rocm-uninstall.tmp");
    let temp = path.with_file_name(temp_name);
    let permissions = fs::metadata(path)
        .with_context(|| format!("failed to read the permissions of {}", path.display()))?
        .permissions();
    // A file at that name is one an interrupted uninstall left; anything else
    // stays, and fails the write below.
    if fs::symlink_metadata(&temp).is_ok_and(|metadata| !metadata.is_dir()) {
        let _ = fs::remove_file(&temp);
    }
    let mut file = create_replacement(&temp, &permissions)
        .with_context(|| format!("failed to create {}", temp.display()))?;
    let result = file
        .write_all(contents)
        .and_then(|()| file.sync_all())
        .and_then(|()| {
            drop(file);
            // The umask may have taken bits off at creation; put them back.
            fs::set_permissions(&temp, permissions)
        })
        .and_then(|()| fs::rename(&temp, path));
    if let Err(error) = result {
        let _ = fs::remove_file(&temp);
        return Err(error).with_context(|| format!("failed to replace {}", path.display()));
    }
    Ok(())
}

/// Create `temp` as a new file (never through a link left at that name) that
/// is no more readable than `permissions` allow.
fn create_replacement(temp: &Path, permissions: &fs::Permissions) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
        options.mode(permissions.mode() & 0o777);
    }
    #[cfg(not(unix))]
    let _ = permissions;
    options.open(temp)
}

// ---------------------------------------------------------------------------
// The running executable
// ---------------------------------------------------------------------------

/// Delete `exe`, the program running this command. Windows will not unlink a
/// running image, so `self_replace` moves it out of the install folder (into
/// the temp folder, or failing that, beside itself under a hidden name) and
/// deletes it once this process exits.
#[cfg(windows)]
fn delete_running_executable(exe: &Path) -> Result<String> {
    self_replace::self_delete_at(exe)
        .with_context(|| format!("failed to remove the running program {}", exe.display()))?;
    Ok(if exe.exists() {
        format!("removed binary {} as this command exits", exe.display())
    } else {
        format!("removed binary {}", exe.display())
    })
}

/// Only Windows plans one; elsewhere the running binary is unlinked like any
/// other file.
#[cfg(not(windows))]
fn delete_running_executable(exe: &Path) -> Result<String> {
    remove_path(exe)?;
    Ok(format!("removed binary {}", exe.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Result<Self> {
            static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "rocm-uninstall-{label}-{}-{unique}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir)?;
            Ok(Self(dir))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn exprs() -> Vec<OsString> {
        profile_path_exprs(
            Path::new("/home/me/.local/bin"),
            Some(Path::new("/home/me")),
        )
    }

    fn os(texts: &[&str]) -> Vec<OsString> {
        texts.iter().map(OsString::from).collect()
    }

    /// Run `plan`, collecting the lines it reports, including those reported
    /// before a failure.
    #[cfg(unix)]
    fn run(plan: &UninstallPlan) -> (Vec<String>, Result<()>) {
        let mut lines = Vec::new();
        let result = execute_plan(plan, &mut |line| lines.push(line));
        (lines, result)
    }

    const POSIX_BLOCK: &str = "\n# >>> rocm-cli path >>>\ncase \":$PATH:\" in\n  *:\"$HOME/.local/bin\":*) ;;\n  *) export PATH=\"$HOME/.local/bin:$PATH\" ;;\nesac\n# <<< rocm-cli path <<<\n";

    #[test]
    fn profile_exprs_cover_the_literal_and_home_relative_forms() {
        assert_eq!(exprs(), os(&["/home/me/.local/bin", "$HOME/.local/bin"]));
        assert_eq!(
            profile_path_exprs(Path::new("/home/me"), Some(Path::new("/home/me"))),
            os(&["/home/me", "$HOME"])
        );
        assert_eq!(
            profile_path_exprs(Path::new("/opt/rocm-cli"), Some(Path::new("/home/me"))),
            os(&["/opt/rocm-cli"])
        );
    }

    #[test]
    fn removing_the_block_restores_the_profile_as_it_was() {
        let before = "alias ll='ls -l'\nexport EDITOR=vim\n";
        let after_install = format!("{before}{POSIX_BLOCK}");
        assert_eq!(
            remove_profile_block(&after_install, &exprs()).as_deref(),
            Some(before)
        );
    }

    #[test]
    fn a_block_in_a_profile_the_installer_created_leaves_it_empty() {
        assert_eq!(
            remove_profile_block(POSIX_BLOCK, &exprs()).as_deref(),
            Some("")
        );
    }

    #[test]
    fn the_fish_block_and_the_literal_path_form_are_removed() {
        let profile = "set -gx EDITOR vim\n\n# >>> rocm-cli path >>>\nif not contains -- \"/home/me/.local/bin\" $PATH\n    set -gx PATH \"/home/me/.local/bin\" $PATH\nend\n# <<< rocm-cli path <<<\n";
        assert_eq!(
            remove_profile_block(profile, &exprs()).as_deref(),
            Some("set -gx EDITOR vim\n")
        );
    }

    #[test]
    fn crlf_profiles_are_handled() {
        let profile = POSIX_BLOCK.replace('\n', "\r\n");
        let profile = format!("export A=1\r\n{profile}export B=2\r\n");
        assert_eq!(
            remove_profile_block(&profile, &exprs()).as_deref(),
            Some("export A=1\r\nexport B=2\r\n")
        );
    }

    #[test]
    fn a_users_own_path_line_is_never_touched() {
        let profile = "export PATH=\"$HOME/.local/bin:$PATH\"\n";
        assert_eq!(remove_profile_block(profile, &exprs()), None);
    }

    #[test]
    fn a_block_for_another_install_folder_stays() {
        let other = POSIX_BLOCK.replace("$HOME/.local/bin", "/opt/other/bin");
        assert_eq!(remove_profile_block(&other, &exprs()), None);
    }

    #[test]
    fn an_unterminated_block_is_left_alone() {
        let profile = "# >>> rocm-cli path >>>\nexport PATH=\"$HOME/.local/bin:$PATH\"\n";
        assert_eq!(remove_profile_block(profile, &exprs()), None);
    }

    #[test]
    fn the_record_parses_each_kind_and_skips_blank_or_malformed_lines() {
        let record = b"profile /home/me/my profile/.bashrc\r\n\nuser-path C:\\Users\\me\\.local\\bin\ngarbage\nprofile \n";
        assert_eq!(
            parse_path_edit_record(record),
            vec![
                ("profile", OsString::from("/home/me/my profile/.bashrc")),
                ("user-path", OsString::from("C:\\Users\\me\\.local\\bin")),
            ]
        );
    }

    #[test]
    fn other_entries_ignore_rocm_cli_files_and_what_is_being_removed() -> Result<()> {
        let dir = Scratch::new("others")?;
        for name in ["rocm", "rocmd", PATH_EDIT_RECORD, "uv", "notes.txt"] {
            fs::write(dir.path().join(name), "")?;
        }
        let removed = vec![dir.path().join("notes.txt")];
        assert_eq!(other_entries(dir.path(), &removed)?, vec!["uv".to_owned()]);
        Ok(())
    }

    /// A folder that cannot be listed may hold programs that rely on it being
    /// on PATH, so it is an error the caller keeps the change for.
    #[test]
    fn a_folder_that_cannot_be_listed_is_an_error_not_an_empty_one() -> Result<()> {
        let dir = Scratch::new("unlistable")?;
        let not_a_folder = dir.path().join("rocm");
        fs::write(&not_a_folder, "")?;
        assert!(other_entries(&not_a_folder, &[]).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_program_whose_name_is_not_utf8_still_counts_as_another_program() -> Result<()> {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt as _;
        let dir = Scratch::new("non-utf8-other")?;
        fs::write(dir.path().join("rocm"), "")?;
        fs::write(dir.path().join(OsStr::from_bytes(b"tool-\xff")), "")?;
        assert_eq!(
            other_entries(dir.path(), &[dir.path().join("rocm")])?,
            vec!["tool-\u{fffd}".to_owned()]
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_replacement_profile_is_never_more_readable_than_the_original() -> Result<()> {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = Scratch::new("replacement-mode")?;
        let temp = dir.path().join("profile.rocm-uninstall.tmp");
        // Created here at 0600 whatever the umask (which only ever takes bits
        // away), rather than at the umask default and narrowed afterwards.
        let file = create_replacement(&temp, &fs::Permissions::from_mode(0o600))?;
        assert_eq!(file.metadata()?.permissions().mode() & 0o777, 0o600);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_replacement_left_by_an_interrupted_uninstall_does_not_block_the_next() -> Result<()> {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = Scratch::new("stale-replacement")?;
        let profile = dir.path().join("profile");
        fs::write(&profile, "old\n")?;
        fs::set_permissions(&profile, fs::Permissions::from_mode(0o600))?;
        fs::write(dir.path().join("profile.rocm-uninstall.tmp"), "partial")?;
        write_replacing(&profile, b"new\n")?;
        assert_eq!(fs::read_to_string(&profile)?, "new\n");
        assert_eq!(fs::metadata(&profile)?.permissions().mode() & 0o777, 0o600);
        assert!(!dir.path().join("profile.rocm-uninstall.tmp").exists());
        Ok(())
    }

    /// Pins only the review's wording. The behavior each line promises is
    /// proven end to end: the running exe by
    /// `@id:lifecycle-windows-installed-binary-smoke`, the PATH undo by
    /// `@id:lifecycle-linux-uninstall-full-purge` and
    /// `@id:lifecycle-windows-uninstall-restores-user-path`.
    #[test]
    fn the_review_lists_the_record_the_running_exe_and_the_path_changes() {
        let plan = UninstallPlan {
            path_edits: vec![PathEdit::ShellProfile {
                profile: PathBuf::from("/home/me/.bashrc"),
                path_exprs: Vec::new(),
            }],
            path_record: Some(PathBuf::from("/opt/bin/.rocm-cli-path-edit")),
            self_delete: Some(PathBuf::from("/opt/bin/rocm.exe")),
            ..UninstallPlan::default()
        };
        let output = render_uninstall_plan(&plan, &UninstallOptions::default());
        for expected in [
            "2 item(s) would be removed:",
            "  - install record: /opt/bin/.rocm-cli-path-edit",
            "  - binary: /opt/bin/rocm.exe (the running program; deleted as this command exits)",
            "PATH changes from the installer that would be undone:\n  - shell profile: the rocm-cli PATH lines in /home/me/.bashrc",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in:\n{output}"
            );
        }
        assert!(!plan_is_empty(&plan));
        assert!(plan_is_empty(&UninstallPlan::default()));
    }

    #[test]
    fn the_running_executable_is_matched_whatever_case_windows_reports_it_in() {
        let exe = Path::new(r"C:\Users\me\.local\bin\rocm.exe");
        let listed = Path::new(r"C:\Users\me\.local\bin\ROCM.EXE");
        assert!(crate::is_running_executable(listed, exe, true));
        assert!(crate::is_running_executable(exe, exe, true));
        let rocmd = Path::new(r"C:\Users\me\.local\bin\rocmd.exe");
        assert!(!crate::is_running_executable(rocmd, exe, true));
        // Elsewhere the running binary is unlinked like any other.
        assert!(!crate::is_running_executable(exe, exe, false));
    }

    #[cfg(unix)]
    mod plan {
        use super::*;

        struct Layout {
            _root: Scratch,
            home: PathBuf,
            bin: PathBuf,
            removed: Vec<PathBuf>,
        }

        fn layout() -> Result<Layout> {
            let root = Scratch::new("plan")?;
            let home = root.path().join("home");
            let bin = home.join(".local").join("bin");
            fs::create_dir_all(&bin)?;
            let mut removed = Vec::new();
            for name in ["rocm", "rocmd"] {
                let path = bin.join(name);
                fs::write(&path, "")?;
                removed.push(path);
            }
            Ok(Layout {
                _root: root,
                home,
                bin,
                removed,
            })
        }

        fn record(layout: &Layout, lines: &str) -> Result<PathBuf> {
            let path = layout.bin.join(PATH_EDIT_RECORD);
            fs::write(&path, lines)?;
            Ok(path)
        }

        fn plan_for(layout: &Layout) -> UninstallPlan {
            let mut plan = UninstallPlan::default();
            plan_path_edits(&mut plan, &layout.bin, &layout.removed, Some(&layout.home));
            plan
        }

        fn profiles(plan: &UninstallPlan) -> Vec<PathBuf> {
            plan.path_edits
                .iter()
                .filter_map(|edit| match edit {
                    PathEdit::ShellProfile { profile, .. } => Some(profile.clone()),
                    PathEdit::UserPath { .. } => None,
                })
                .collect()
        }

        /// A plan that removes `layout`'s binaries after undoing `plan`'s PATH
        /// changes, as `build_uninstall_plan` lays it out.
        fn with_binaries(layout: &Layout, mut plan: UninstallPlan) -> UninstallPlan {
            plan.actions = layout
                .removed
                .iter()
                .map(|path| crate::UninstallPlanEntry {
                    kind: "binary",
                    path: path.clone(),
                })
                .collect();
            plan
        }

        /// `plan` also removing a config folder holding the user's settings,
        /// as a plain `rocm uninstall` does.
        fn with_config(
            layout: &Layout,
            mut plan: UninstallPlan,
        ) -> Result<(UninstallPlan, PathBuf)> {
            let config = layout.home.join("config");
            fs::create_dir_all(&config)?;
            fs::write(config.join("config.json"), "{}")?;
            plan.actions.insert(
                0,
                crate::UninstallPlanEntry {
                    kind: "config",
                    path: config.clone(),
                },
            );
            Ok((plan, config))
        }

        #[test]
        fn a_recorded_profile_block_is_undone_and_the_record_goes_last() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join("my-profile");
            fs::write(&profile, format!("export A=1\n{POSIX_BLOCK}"))?;
            let record = record(&layout, &format!("profile {}\n", profile.display()))?;
            let plan = plan_for(&layout);
            assert_eq!(profiles(&plan), vec![profile.clone()], "{plan:?}");
            assert_eq!(plan.path_record.as_ref(), Some(&record));

            let (lines, result) = run(&with_binaries(&layout, plan));
            result?;
            assert_eq!(fs::read_to_string(&profile)?, "export A=1\n");
            assert!(!record.exists() && !layout.removed[0].exists());
            let undo = lines
                .iter()
                .position(|line| line.starts_with("removed the rocm-cli PATH lines"));
            let binary = lines
                .iter()
                .position(|line| line.starts_with("removed binary"));
            assert!(undo < binary, "the PATH change is undone first: {lines:?}");
            Ok(())
        }

        #[test]
        fn a_path_change_that_cannot_be_undone_stops_before_the_binaries_go() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join(".bashrc");
            let contents = format!("export A=1\n{POSIX_BLOCK}");
            fs::write(&profile, &contents)?;
            let record = record(&layout, &format!("profile {}\n", profile.display()))?;
            // The edit writes its replacement beside the profile first; a folder
            // in that spot makes the write fail even for root.
            fs::create_dir(layout.home.join(".bashrc.rocm-uninstall.tmp"))?;
            let (plan, config) = with_config(&layout, with_binaries(&layout, plan_for(&layout)))?;

            let (lines, result) = run(&plan);
            let message = format!("{:#}", result.expect_err("the PATH undo fails"));
            // The message claims nothing else went; the state below proves it.
            assert!(
                message.contains("could not undo the installer's PATH change")
                    && message.contains("Nothing else was removed")
                    && message.contains("`rocm uninstall` again"),
                "{message}"
            );
            assert!(lines.is_empty(), "nothing was reported removed: {lines:?}");
            assert_eq!(fs::read_to_string(&profile)?, contents);
            assert!(layout.removed.iter().all(|path| path.exists()));
            assert!(config.join("config.json").exists(), "the settings stay");
            assert!(record.exists(), "the record stays for the retry");

            // And the retry the message names does clear it.
            fs::remove_dir(layout.home.join(".bashrc.rocm-uninstall.tmp"))?;
            let (plan, _) = with_config(&layout, with_binaries(&layout, plan_for(&layout)))?;
            run(&plan).1?;
            assert_eq!(fs::read_to_string(&profile)?, "export A=1\n");
            assert!(!config.exists() && !record.exists());
            Ok(())
        }

        /// A step that fails after the PATH undo must not hide what was
        /// already done: each line is reported as its step finishes.
        #[test]
        fn a_failure_after_the_path_undo_still_reports_what_was_done() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join(".bashrc");
            fs::write(&profile, format!("export A=1\n{POSIX_BLOCK}"))?;
            let (mut plan, config) =
                with_config(&layout, with_binaries(&layout, plan_for(&layout)))?;
            // A name too long for any filesystem fails to remove, even for root.
            plan.actions.push(crate::UninstallPlanEntry {
                kind: "binary",
                path: layout.bin.join("x".repeat(300)),
            });

            let (lines, result) = run(&plan);
            assert!(result.is_err(), "the last removal fails: {lines:?}");
            // Each reported line matches what really happened.
            assert_eq!(fs::read_to_string(&profile)?, "export A=1\n");
            assert!(!config.exists() && !layout.removed[0].exists());
            for expected in [
                format!("removed the rocm-cli PATH lines from {}", profile.display()),
                format!("removed config {}", config.display()),
                format!("removed binary {}", layout.removed[0].display()),
            ] {
                assert!(
                    lines.iter().any(|line| line.starts_with(&expected)),
                    "missing {expected:?} in {lines:?}"
                );
            }
            Ok(())
        }

        #[test]
        fn without_a_record_the_marked_block_in_a_known_profile_is_found() -> Result<()> {
            let layout = layout()?;
            fs::write(layout.home.join(".bashrc"), POSIX_BLOCK)?;
            fs::write(
                layout.home.join(".zshrc"),
                "export PATH=\"$HOME/.local/bin:$PATH\"\n",
            )?;
            let plan = plan_for(&layout);
            assert_eq!(profiles(&plan), vec![layout.home.join(".bashrc")]);
            assert_eq!(plan.path_record, None);
            Ok(())
        }

        #[test]
        fn a_block_from_an_earlier_install_in_another_shell_is_also_undone() -> Result<()> {
            let layout = layout()?;
            fs::write(layout.home.join(".bashrc"), POSIX_BLOCK)?;
            fs::write(layout.home.join(".zshrc"), POSIX_BLOCK)?;
            record(
                &layout,
                &format!("profile {}\n", layout.home.join(".zshrc").display()),
            )?;
            let plan = plan_for(&layout);
            assert_eq!(
                profiles(&plan),
                vec![layout.home.join(".zshrc"), layout.home.join(".bashrc")]
            );
            Ok(())
        }

        #[test]
        fn another_program_in_the_folder_keeps_the_change_and_its_record() -> Result<()> {
            let layout = layout()?;
            fs::write(layout.home.join(".bashrc"), POSIX_BLOCK)?;
            record(
                &layout,
                &format!("profile {}\n", layout.home.join(".bashrc").display()),
            )?;
            fs::write(layout.bin.join("uv"), "")?;
            let plan = plan_for(&layout);
            assert!(plan.path_edits.is_empty(), "{plan:?}");
            assert_eq!(plan.path_record, None, "the record stays with the change");
            let record = layout.bin.join(PATH_EDIT_RECORD);
            assert!(
                plan.skipped
                    .iter()
                    .any(|note| note.starts_with("PATH change left in place")
                        && note.contains("uv")
                        && note.contains(&format!("its record {} stays", record.display()))),
                "{plan:?}"
            );
            Ok(())
        }

        #[test]
        fn a_recorded_profile_without_the_block_is_reported_not_edited() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join(".bashrc");
            fs::write(&profile, "export A=1\n")?;
            let record = record(&layout, &format!("profile {}\n", profile.display()))?;
            let plan = plan_for(&layout);
            assert!(plan.path_edits.is_empty());
            assert!(
                plan.skipped
                    .iter()
                    .any(|note| note.contains("nothing to undo"))
            );
            assert_eq!(plan.path_record, Some(record), "a spent record goes");
            Ok(())
        }

        /// A profile the installer recorded writing to most likely still holds
        /// its block, so one that cannot be read stops the uninstall like any
        /// change that cannot be undone: before anything is removed.
        #[test]
        fn a_recorded_profile_that_cannot_be_read_stops_before_anything_goes() -> Result<()> {
            let layout = layout()?;
            // A folder where the profile should be cannot be read as one, even
            // by root.
            let profile = layout.home.join("my-profile");
            fs::create_dir(&profile)?;
            let record = record(&layout, &format!("profile {}\n", profile.display()))?;
            let plan = plan_for(&layout);
            assert_eq!(profiles(&plan), vec![profile], "{plan:?}");
            assert!(
                plan.warnings
                    .iter()
                    .any(|note| note.starts_with("could not read")
                        && note.contains("my-profile")
                        && note.contains("nothing will be removed")),
                "{plan:?}"
            );

            let (plan, config) = with_config(&layout, with_binaries(&layout, plan))?;
            let message = format!("{:#}", run(&plan).1.expect_err("the profile is unreadable"));
            assert!(message.contains("could not undo"), "{message}");
            assert!(layout.removed.iter().all(|path| path.exists()));
            assert!(config.exists() && record.exists());
            Ok(())
        }

        /// A profile only scanned for, not recorded, may not hold a block at
        /// all, so one that cannot be read is reported and does not stop the
        /// uninstall.
        #[test]
        fn an_unrecorded_profile_that_cannot_be_read_is_a_warning_not_an_absence() -> Result<()> {
            let layout = layout()?;
            fs::create_dir(layout.home.join(".bashrc"))?;
            let plan = plan_for(&layout);
            assert!(plan.path_edits.is_empty(), "{plan:?}");
            assert!(
                plan.warnings
                    .iter()
                    .any(|note| note.starts_with("could not read")
                        && note.contains(".bashrc")
                        && note.contains("left in place")),
                "{plan:?}"
            );
            Ok(())
        }

        #[test]
        fn a_record_naming_a_profile_that_is_not_utf8_is_still_followed() -> Result<()> {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt as _;
            let layout = layout()?;
            let profile = layout.home.join(OsStr::from_bytes(b"caf\xe9-profile"));
            fs::write(&profile, format!("export A=1\n{POSIX_BLOCK}"))?;
            let mut record_bytes = b"profile ".to_vec();
            record_bytes.extend_from_slice(profile.as_os_str().as_bytes());
            record_bytes.push(b'\n');
            fs::write(layout.bin.join(PATH_EDIT_RECORD), record_bytes)?;
            let plan = plan_for(&layout);
            assert!(plan.warnings.is_empty(), "{plan:?}");
            assert_eq!(profiles(&plan), vec![profile.clone()]);
            apply_path_edit(&plan.path_edits[0])?;
            assert_eq!(fs::read(&profile)?, b"export A=1\n");
            Ok(())
        }

        #[test]
        fn an_install_folder_whose_path_is_not_utf8_is_still_found() -> Result<()> {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt as _;
            let layout = layout()?;
            // Outside the home folder, so the block spells it literally.
            let bin = layout.home.with_file_name(OsStr::from_bytes(b"bin-\xff"));
            fs::create_dir_all(&bin)?;
            let mut block = b"\n# >>> rocm-cli path >>>\ncase \":$PATH:\" in\n  *:\"".to_vec();
            block.extend_from_slice(bin.as_os_str().as_bytes());
            block.extend_from_slice(b"\":*) ;;\nesac\n# <<< rocm-cli path <<<\n");
            fs::write(layout.home.join(".bashrc"), &block)?;
            let mut plan = UninstallPlan::default();
            plan_path_edits(&mut plan, &bin, &[], Some(&layout.home));
            assert_eq!(
                profiles(&plan),
                vec![layout.home.join(".bashrc")],
                "{plan:?}"
            );
            Ok(())
        }

        #[test]
        fn a_profile_that_is_not_utf8_is_still_edited_byte_for_byte() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join(".bashrc");
            let before = b"# caf\xe9 latin-1\nexport A=1\n".to_vec();
            let mut installed = before.clone();
            installed.extend_from_slice(POSIX_BLOCK.as_bytes());
            fs::write(&profile, &installed)?;
            fs::write(layout.home.join(".zshrc"), b"# \xff not ours\n")?;
            let plan = plan_for(&layout);
            assert!(plan.warnings.is_empty(), "{plan:?}");
            assert_eq!(profiles(&plan), vec![profile.clone()]);
            apply_path_edit(&plan.path_edits[0])?;
            assert_eq!(fs::read(&profile)?, before);
            Ok(())
        }

        #[test]
        fn a_block_written_through_a_linked_home_is_still_found() -> Result<()> {
            let layout = layout()?;
            // The installer saw HOME as a link; the binary reports its folder
            // resolved, under the real home.
            let link = layout.home.with_file_name("home-link");
            std::os::unix::fs::symlink(&layout.home, &link)?;
            fs::write(layout.home.join(".bashrc"), POSIX_BLOCK)?;
            let mut plan = UninstallPlan::default();
            plan_path_edits(&mut plan, &layout.bin, &layout.removed, Some(&link));
            assert_eq!(profiles(&plan), vec![link.join(".bashrc")], "{plan:?}");
            Ok(())
        }

        #[test]
        fn a_recorded_spelling_is_matched_even_when_no_derived_one_is() -> Result<()> {
            let layout = layout()?;
            let block = POSIX_BLOCK.replace("$HOME/.local/bin", "/elsewhere/bin/");
            fs::write(layout.home.join(".bashrc"), &block)?;
            record(&layout, "path-expr /elsewhere/bin/\n")?;
            assert_eq!(
                profiles(&plan_for(&layout)),
                vec![layout.home.join(".bashrc")]
            );
            Ok(())
        }

        /// `--keep-binaries` keeps the installer's PATH change and its record
        /// with the binaries, and says so. Run against an installed folder that
        /// holds both, so the same plan without the flag would undo them.
        #[test]
        fn keeping_the_binaries_keeps_the_path_change_and_says_so() -> Result<()> {
            let layout = layout()?;
            let profile = layout.home.join(".bashrc");
            fs::write(&profile, POSIX_BLOCK)?;
            let record = record(&layout, &format!("profile {}\n", profile.display()))?;
            let plan_with = |keep_binaries| -> Result<UninstallPlan> {
                let mut plan = UninstallPlan::default();
                let options = UninstallOptions {
                    keep_binaries,
                    ..UninstallOptions::default()
                };
                crate::plan_installed_binaries(
                    &mut plan,
                    &options,
                    || Ok(layout.removed[0].clone()),
                    Some(&layout.home),
                )?;
                Ok(plan)
            };

            let plan = plan_with(false)?;
            assert_eq!(profiles(&plan), vec![profile.clone()], "{plan:?}");
            assert_eq!(plan.path_record.as_ref(), Some(&record));

            let plan = plan_with(true)?;
            assert!(
                plan.actions.is_empty() && plan.path_edits.is_empty(),
                "{plan:?}"
            );
            assert_eq!(plan.path_record, None, "{plan:?}");
            assert!(
                plan.skipped
                    .iter()
                    .any(|note| note.contains("--keep-binaries")
                        && note.contains("PATH change the installer made stays too")),
                "{plan:?}"
            );
            // Carrying the plan out leaves the change, its record, and the
            // binaries where they are.
            run(&plan).1?;
            assert_eq!(fs::read_to_string(&profile)?, POSIX_BLOCK);
            assert!(record.exists() && layout.removed.iter().all(|path| path.exists()));
            Ok(())
        }

        #[test]
        fn a_linked_profile_is_edited_where_it_lives_and_stays_a_link() -> Result<()> {
            use std::os::unix::fs::PermissionsExt as _;
            let layout = layout()?;
            let dotfiles = layout.home.join("dotfiles");
            fs::create_dir(&dotfiles)?;
            let real = dotfiles.join("bashrc");
            fs::write(&real, format!("export A=1\n{POSIX_BLOCK}"))?;
            fs::set_permissions(&real, fs::Permissions::from_mode(0o640))?;
            let profile = layout.home.join(".bashrc");
            std::os::unix::fs::symlink(&real, &profile)?;
            let plan = plan_for(&layout);
            let line = apply_path_edit(&plan.path_edits[0])?;
            assert!(
                line.starts_with("removed the rocm-cli PATH lines"),
                "{line}"
            );
            assert!(fs::symlink_metadata(&profile)?.is_symlink());
            assert_eq!(fs::read_to_string(&real)?, "export A=1\n");
            assert_eq!(fs::metadata(&real)?.permissions().mode() & 0o777, 0o640);
            Ok(())
        }
    }
}
