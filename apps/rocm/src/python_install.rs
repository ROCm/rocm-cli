// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether the running `rocm` was put there by a Python package manager.
//!
//! `rocm-cli` is also published as a wheel (see `scripts/build_wheel.py`) that
//! installs the real `rocm` and `rocmd` executables into a Python environment's
//! `bin` (Linux) or `Scripts` (Windows) folder. `rocm uninstall` removes every
//! `rocm*` file beside the running executable, which inside such an
//! environment deletes files `pip` still lists as installed: the wheel's
//! `RECORD` is left dangling, `pip list` keeps reporting the package, and `pip`
//! can no longer remove its own install. [`is_python_managed_layout`] is the
//! guard `uninstall_plan::plan_installed_binaries` consults before it plans any
//! binary for removal; when it answers yes the binaries are left alone and the
//! user is pointed at the package manager instead ([`skip_notice`]).
//!
//! Only paths derived from the running executable are inspected, never an
//! environment variable or a process global, so the answer is reproducible
//! over a temporary tree in a unit test.
//!
//! The probe is deliberately conservative, because the two mistakes do not cost
//! the same. A false positive leaves the binaries in place and says how to
//! remove them; the user at worst deletes the files by hand. A false negative
//! deletes files another package manager owns. So an *absent* marker means "not
//! Python-managed" (no `pyvenv.cfg`, no site-packages and no matching `RECORD`
//! row is the ordinary shape of an installer-managed `~/.local/bin`), but a
//! probe that cannot read something that exists answers "Python-managed"
//! rather than guess: [`ProbeFailed`].
//!
//! Layouts recognised, with the folder holding the running executable's
//! parent (`bin` or `Scripts`) as the environment root:
//!
//! - any environment with a `pyvenv.cfg` at its root: `python -m venv`,
//!   `pipx` and `uv tool` on Linux, and the same on Windows when the executable
//!   is run from inside the environment;
//! - a `RECORD` that lists the executable, under `lib/python*/site-packages`
//!   and `lib64/python*/site-packages` (Fedora-family wheels with
//!   `Root-Is-Purelib: false` land in `lib64`), `lib/python*/dist-packages`
//!   (Debian's system and `/usr/local` trees, including `lib/python3`), the
//!   Windows `Lib/site-packages`, and the Windows `pip install --user` layout
//!   whose `site-packages` sits beside `Scripts`.
//!
//! Known limits. A Windows `pipx` or `uv tool` install copies the launcher
//! into `%USERPROFILE%\.local\bin`, a folder with no Python environment above
//! it, so a `rocm.exe` run from there looks exactly like an installer-managed
//! one and is not detected; use `pipx uninstall rocm-cli` or `uv tool uninstall
//! rocm-cli` there. And the `RECORD` scan is not scoped to the `rocm-cli`
//! distribution: any distribution whose `RECORD` lists a script of this name
//! beside the executable claims it, which only keeps binaries that could have
//! been deleted, the safe side of the asymmetry above.

use std::borrow::Cow;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A probe that could not complete: something that exists could not be listed
/// or read, so who owns the executable is unknown. Callers resolve it to
/// "Python-managed" (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProbeFailed;

/// The line `rocm uninstall` prints, in place of removing binaries, when
/// [`is_python_managed_layout`] says a package manager owns `current_exe`.
///
/// It says what the layout is evidence of ("may own"), not what happened: a
/// `pyvenv.cfg` marks the folder as an environment, not the files as a wheel's.
pub(crate) fn skip_notice(current_exe: &Path) -> String {
    format!(
        "binary removal skipped because {} sits in a Python environment whose package manager may own it; remove it with the matching command instead: `pip uninstall rocm-cli`, `pipx uninstall rocm-cli`, or `uv tool uninstall rocm-cli`",
        current_exe.display()
    )
}

/// Whether `current_exe` is an executable a Python package manager installed
/// and may still track. See the module docs for what is recognised and why an
/// unreadable tree counts as managed.
pub(crate) fn is_python_managed_layout(current_exe: &Path) -> bool {
    probe(current_exe).unwrap_or(true)
}

fn probe(current_exe: &Path) -> Result<bool, ProbeFailed> {
    let Some(script_dir) = current_exe.parent() else {
        return Ok(false);
    };
    let Some(script_dir_name) = script_dir.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    // Python environments place installed scripts directly under the
    // environment root, in `bin` on unix and `Scripts` on Windows.
    if !script_dir_name.eq_ignore_ascii_case("bin")
        && !script_dir_name.eq_ignore_ascii_case("Scripts")
    {
        return Ok(false);
    }
    let Some(env_root) = script_dir.parent() else {
        return Ok(false);
    };
    // `python -m venv`, `pipx` and `uv tool` all write a real `pyvenv.cfg` at
    // the environment root.
    if metadata_if_present(&env_root.join("pyvenv.cfg"))?.is_some_and(|meta| meta.is_file()) {
        return Ok(true);
    }
    // No `pyvenv.cfg`: a `--user` or system-site install still records the
    // script in its distribution's `RECORD`.
    for site_packages in site_packages_dirs(env_root)? {
        if dist_info_record_owns_script(&site_packages, current_exe)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `fs::metadata`, where a path that is not there is `Ok(None)` and one that is
/// there but cannot be inspected (a permission error, or a parent that is a file)
/// is [`ProbeFailed`]. `Path::is_dir` and `is_file` fold that second case into
/// "no", which would read an unreadable marker as an absent one.
fn metadata_if_present(path: &Path) -> Result<Option<fs::Metadata>, ProbeFailed> {
    match fs::metadata(path) {
        Ok(meta) => Ok(Some(meta)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(ProbeFailed),
    }
}

/// [`metadata_if_present`] for a folder.
fn dir_if_present(path: &Path) -> Result<bool, ProbeFailed> {
    Ok(metadata_if_present(path)?.is_some_and(|meta| meta.is_dir()))
}

/// `fs::read_dir`, where a folder that is not there is `Ok(None)` and one that
/// is there but cannot be listed is [`ProbeFailed`].
fn read_dir_if_present(dir: &Path) -> Result<Option<fs::ReadDir>, ProbeFailed> {
    match fs::read_dir(dir) {
        Ok(entries) => Ok(Some(entries)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(ProbeFailed),
    }
}

/// The folders to search for the `RECORD` that owns a script under `env_root`.
///
/// Every layout is probed on every host: this is pure path inspection, and a
/// tree must be classified the same way whichever platform inspects it. A
/// folder that is missing contributes nothing (the ordinary shape of a plain
/// `~/.local/bin` install); one that exists but cannot be listed is
/// [`ProbeFailed`], since its contents could name the owner.
fn site_packages_dirs(env_root: &Path) -> Result<Vec<PathBuf>, ProbeFailed> {
    let mut dirs = Vec::new();
    // Windows: the interpreter's own `Lib\site-packages`, and `pip install
    // --user`'s `Python3xx\site-packages` beside `Python3xx\Scripts`.
    for fixed in [
        env_root.join("Lib").join("site-packages"),
        env_root.join("site-packages"),
    ] {
        if dir_if_present(&fixed)? {
            dirs.push(fixed);
        }
    }
    // Unix: `<root>/lib*/python3.x/{site-packages,dist-packages}`. `lib/python3`
    // is Debian's version-independent `dist-packages` parent.
    for lib in ["lib", "lib64"] {
        let Some(entries) = read_dir_if_present(&env_root.join(lib))? else {
            continue;
        };
        for entry in entries {
            let version_dir = entry.map_err(|_| ProbeFailed)?.path();
            let is_python = version_dir
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("python"));
            if !is_python {
                continue;
            }
            for leaf in ["site-packages", "dist-packages"] {
                let candidate = version_dir.join(leaf);
                if dir_if_present(&candidate)? {
                    dirs.push(candidate);
                }
            }
        }
    }
    Ok(dirs)
}

/// Whether any installed distribution under `site_packages` lists `current_exe`
/// in its `RECORD`.
///
/// A missing folder, a `*.dist-info` without a `RECORD`, and a `RECORD` whose
/// rows never name the script all mean "not owned". A folder or `RECORD` that
/// exists but cannot be read is [`ProbeFailed`]: the row that would have
/// claimed the script may be exactly the one that could not be read. A `RECORD`
/// that is not valid UTF-8 is still read, lossily, since only a path's file
/// name and folder name are compared.
fn dist_info_record_owns_script(
    site_packages: &Path,
    current_exe: &Path,
) -> Result<bool, ProbeFailed> {
    let Some(script_name) = current_exe.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    let Some(script_dir_name) = current_exe
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
    else {
        return Ok(false);
    };
    let Some(entries) = read_dir_if_present(site_packages)? else {
        return Ok(false);
    };
    for entry in entries {
        let dist_info = entry.map_err(|_| ProbeFailed)?.path();
        let is_dist_info = dist_info
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".dist-info"));
        if !is_dist_info {
            continue;
        }
        let record = match fs::read(dist_info.join("RECORD")) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(_) => return Err(ProbeFailed),
        };
        let record = String::from_utf8_lossy(&record);
        if record
            .lines()
            .any(|row| record_row_targets_script(row, script_name, script_dir_name))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether a wheel `RECORD` row (`<path>,<hash>,<size>`, with `/` separators on
/// every platform) refers to the installed script.
///
/// Wheel records name installed scripts by a path relative to the folder holding
/// the `.dist-info`, such as `../../../bin/rocm`. Matching the file name plus a
/// `bin`/`Scripts` path component is intentional: resolving those paths exactly
/// would mean canonicalizing through the environment's links and junctions,
/// which is fragile and can fail outright mid-uninstall. Names compare
/// ASCII-case-insensitively, as Windows file names do.
fn record_row_targets_script(row: &str, script_name: &str, script_dir_name: &str) -> bool {
    let Some(target) = record_row_target_path(row) else {
        return false;
    };
    let target = Path::new(&*target);
    let named = target
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(script_name));
    named
        && target.components().any(|component| {
            component
                .as_os_str()
                .to_str()
                .is_some_and(|part| part.eq_ignore_ascii_case(script_dir_name))
        })
}

/// The first (path) field of a wheel `RECORD` row, unquoted and unescaped.
///
/// `RECORD` is RFC 4180 CSV, so a path holding a comma or a double quote is
/// written quoted with every inner quote doubled. Splitting such a row on its
/// first comma would truncate the path and miss the script it names, so the
/// quoted form is parsed here. The common unquoted row stays borrowed. A row
/// that opens a quote and never closes it is malformed and names nothing.
fn record_row_target_path(row: &str) -> Option<Cow<'_, str>> {
    let row = row.trim_start();
    let Some(quoted) = row.strip_prefix('"') else {
        let target = row.split(',').next().unwrap_or_default().trim_end();
        return (!target.is_empty()).then_some(Cow::Borrowed(target));
    };
    let mut target = String::new();
    let mut characters = quoted.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '"' {
            target.push(character);
        } else if characters.peek() == Some(&'"') {
            // A doubled quote inside the field is one literal quote.
            characters.next();
            target.push('"');
        } else {
            // A lone quote closes the field; the rest is the hash and size.
            return (!target.is_empty()).then_some(Cow::Owned(target));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uninstall_plan::{UninstallOptions, UninstallPlan, plan_installed_binaries};
    use anyhow::Result;
    use tempfile::TempDir;

    /// A tree under a temp folder, with the running executable `exe` at the
    /// given path inside it and a sibling `rocmd` beside it, as an install has.
    struct Tree {
        root: TempDir,
        exe: PathBuf,
    }

    impl Tree {
        fn new(exe_rel: &str) -> Self {
            let root = TempDir::new().expect("temp dir");
            let exe = root.path().join(exe_rel);
            fs::create_dir_all(exe.parent().expect("exe has a parent")).expect("bin dir");
            fs::write(&exe, b"binary").expect("write exe");
            let sibling = exe.with_file_name(if exe.extension().is_some() {
                "rocmd.exe"
            } else {
                "rocmd"
            });
            fs::write(sibling, b"binary").expect("write sibling");
            Self { root, exe }
        }

        fn write(&self, rel: &str, contents: &str) {
            let path = self.root.path().join(rel);
            fs::create_dir_all(path.parent().expect("file has a parent")).expect("parent");
            fs::write(path, contents).expect("write file");
        }

        fn plan(&self) -> Result<UninstallPlan> {
            let mut plan = UninstallPlan::default();
            plan_installed_binaries(
                &mut plan,
                &UninstallOptions::default(),
                || Ok(self.exe.clone()),
                Some(self.root.path()),
                &rocm_core::canonicalize_for_compare,
            )?;
            Ok(plan)
        }

        /// The decision, its message and the resulting plan, together (AGENTS
        /// section 3): a managed tree names the skip and plans nothing to
        /// delete, not even the sibling `rocmd`; an unmanaged one plans both
        /// binaries and prints no skip.
        fn assert_decision(&self, managed: bool) {
            assert_eq!(is_python_managed_layout(&self.exe), managed, "decision");
            let plan = self.plan().expect("plan");
            let binaries: Vec<_> = plan
                .actions
                .iter()
                .filter(|entry| entry.kind == "binary")
                .collect();
            if managed {
                assert_eq!(plan.skipped, [skip_notice(&self.exe)]);
                assert!(binaries.is_empty(), "{binaries:?}");
                assert!(plan.self_delete.is_none(), "{:?}", plan.self_delete);
                assert!(plan.path_edits.is_empty(), "{:?}", plan.path_edits);
            } else {
                assert!(plan.skipped.is_empty(), "{:?}", plan.skipped);
                // The running image is deleted on exit on Windows instead.
                let planned = binaries.len() + usize::from(plan.self_delete.is_some());
                assert_eq!(planned, 2, "{binaries:?} {:?}", plan.self_delete);
            }
        }
    }

    const UNIX_RECORD: &str = "rocm_cli-0.3.0.dist-info/METADATA,sha256=aaa,512\n\
         ../../../bin/rocm,sha256=bbb,17000000\n\
         ../../../bin/rocmd,sha256=ccc,16000000\n";

    #[test]
    fn a_pyvenv_cfg_marks_a_unix_environment_as_python_managed() {
        let tree = Tree::new("venv/bin/rocm");
        tree.write("venv/pyvenv.cfg", "home = /usr/bin\n");
        tree.assert_decision(true);
    }

    #[test]
    fn a_pyvenv_cfg_marks_a_windows_scripts_folder_as_python_managed() {
        let tree = Tree::new("venv/Scripts/rocm.exe");
        tree.write("venv/pyvenv.cfg", "home = C:/Python313\n");
        tree.assert_decision(true);
    }

    #[test]
    fn a_plain_local_bin_install_is_removed_as_before() {
        // No `pyvenv.cfg`, no site-packages: the installer's own layout. The
        // pair below flips one file, so the negative is not a constant.
        let tree = Tree::new("home/.local/bin/rocm");
        tree.assert_decision(false);
        tree.write("home/.local/pyvenv.cfg", "home = /usr/bin\n");
        tree.assert_decision(true);
    }

    #[test]
    fn a_bin_folder_beside_unrelated_python_trees_is_removed() {
        // site-packages and a RECORD exist, but nothing in them names the
        // executable.
        let tree = Tree::new("usr/bin/rocm");
        tree.write(
            "usr/lib/python3.12/site-packages/other-1.dist-info/RECORD",
            "other-1.dist-info/METADATA,sha256=aaa,5\n../../../bin/other,sha256=bbb,9\n",
        );
        tree.assert_decision(false);
        tree.write(
            "usr/lib/python3.12/site-packages/other-1.dist-info/RECORD",
            "../../../bin/other,sha256=bbb,9\n../../../bin/rocm,sha256=ccc,9\n",
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_in_lib_site_packages_owns_a_script_without_a_pyvenv_cfg() {
        let tree = Tree::new("local/bin/rocm");
        tree.write(
            "local/lib/python3.13/site-packages/rocm_cli-0.3.0.dist-info/RECORD",
            UNIX_RECORD,
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_in_lib64_owns_a_script() {
        // Fedora-family `platlib`, where a wheel with `Root-Is-Purelib: false`
        // is unpacked.
        let tree = Tree::new("usr/bin/rocm");
        tree.write(
            "usr/lib64/python3.13/site-packages/rocm_cli-0.3.0.dist-info/RECORD",
            UNIX_RECORD,
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_in_debians_python3_dist_packages_owns_a_script() {
        let tree = Tree::new("usr/bin/rocm");
        tree.write(
            "usr/lib/python3/dist-packages/rocm_cli-0.3.0.dist-info/RECORD",
            UNIX_RECORD,
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_in_usr_local_dist_packages_owns_a_script() {
        let tree = Tree::new("usr/local/bin/rocm");
        tree.write(
            "usr/local/lib/python3.12/dist-packages/rocm_cli-0.3.0.dist-info/RECORD",
            UNIX_RECORD,
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_in_a_windows_lib_site_packages_owns_a_script() {
        let tree = Tree::new("Python313/Scripts/rocm.exe");
        tree.write(
            "Python313/Lib/site-packages/rocm_cli-0.3.0.dist-info/RECORD",
            "../../Scripts/rocm.exe,sha256=bbb,17000000\n",
        );
        tree.assert_decision(true);
    }

    #[test]
    fn the_windows_pip_install_user_layout_owns_a_script() {
        // `%APPDATA%\Python\Python313\Scripts\rocm.exe`, with `site-packages`
        // beside `Scripts` and no `Lib` folder and no `pyvenv.cfg`.
        let tree = Tree::new("Python/Python313/Scripts/rocm.exe");
        tree.write(
            "Python/Python313/site-packages/rocm_cli-0.3.0.dist-info/RECORD",
            "../Scripts/rocm.exe,sha256=bbb,17000000\n../Scripts/rocmd.exe,sha256=ccc,16000000\n",
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_record_row_matches_whatever_the_case_of_the_name() {
        let tree = Tree::new("Python313/Scripts/rocm.exe");
        tree.write(
            "Python313/Lib/site-packages/rocm_cli-0.3.0.dist-info/RECORD",
            "../../scripts/ROCM.EXE,sha256=bbb,17000000\n",
        );
        tree.assert_decision(true);
    }

    #[test]
    fn a_folder_that_is_not_bin_or_scripts_is_never_python_managed() {
        let tree = Tree::new("pyenv/sbin/rocm");
        tree.write("pyenv/pyvenv.cfg", "home = /usr/bin\n");
        tree.assert_decision(false);
    }

    #[test]
    fn an_unlistable_lib_folder_is_treated_as_managed() {
        // `lib` exists but is a file, so it cannot be listed: its contents
        // could name the owner, so the answer is "managed", not "free to
        // delete". Replacing the file by nothing flips it back.
        let tree = Tree::new("home/.local/bin/rocm");
        tree.write("home/.local/lib", "not a folder");
        tree.assert_decision(true);
        fs::remove_file(tree.root.path().join("home/.local/lib")).expect("remove");
        tree.assert_decision(false);
    }

    #[test]
    fn a_lib_folder_that_is_a_file_hides_nothing_it_can_be_asked_about() {
        // `Lib` is where Windows keeps `site-packages`. Here it is a file, so
        // `Lib/site-packages` cannot be inspected (it is not merely absent) and
        // the answer is "managed". `Path::is_dir` would have said "absent".
        let tree = Tree::new("Python313/Scripts/rocm.exe");
        tree.write("Python313/Lib", "not a folder");
        tree.assert_decision(true);
        fs::remove_file(tree.root.path().join("Python313/Lib")).expect("remove");
        tree.assert_decision(false);
    }

    #[test]
    fn a_record_row_outside_a_script_folder_does_not_claim_the_script() {
        // A row whose file name is `rocm` but which sits in no `bin`/`Scripts`
        // folder (a docs page, say) is not the installed script.
        let tree = Tree::new("usr/bin/rocm");
        tree.write(
            "usr/lib/python3.12/site-packages/other-1.dist-info/RECORD",
            "other/docs/rocm,sha256=aaa,5\n",
        );
        tree.assert_decision(false);
        tree.write(
            "usr/lib/python3.12/site-packages/other-1.dist-info/RECORD",
            "../../../bin/rocm,sha256=aaa,5\n",
        );
        tree.assert_decision(true);
    }

    #[test]
    fn an_unlistable_lib64_folder_is_treated_as_managed() {
        let tree = Tree::new("usr/bin/rocm");
        tree.write("usr/lib64", "not a folder");
        tree.assert_decision(true);
    }

    #[test]
    fn an_unreadable_record_is_treated_as_managed() {
        // A `RECORD` that exists but cannot be read (here a folder of that
        // name) may hold the very row that claims the script.
        let tree = Tree::new("usr/bin/rocm");
        let record = tree
            .root
            .path()
            .join("usr/lib/python3.12/site-packages/other-1.dist-info/RECORD");
        fs::create_dir_all(&record).expect("RECORD folder");
        tree.assert_decision(true);
        fs::remove_dir(&record).expect("remove");
        tree.assert_decision(false);
    }

    #[test]
    fn a_dist_info_without_a_record_claims_nothing() {
        let tree = Tree::new("usr/bin/rocm");
        fs::create_dir_all(
            tree.root
                .path()
                .join("usr/lib/python3.12/site-packages/other-1.dist-info"),
        )
        .expect("dist-info");
        tree.assert_decision(false);
    }

    #[test]
    fn a_record_that_is_not_utf8_is_still_read() {
        let tree = Tree::new("usr/bin/rocm");
        let record = tree
            .root
            .path()
            .join("usr/lib/python3.12/site-packages/other-1.dist-info/RECORD");
        fs::create_dir_all(record.parent().expect("parent")).expect("dist-info");
        let mut bytes = b"../../../bin/caf\xe9,sha256=aaa,5\n".to_vec();
        bytes.extend_from_slice(b"../../../bin/rocm,sha256=bbb,9\n");
        fs::write(&record, bytes).expect("write RECORD");
        tree.assert_decision(true);
    }

    #[test]
    fn quoted_record_rows_are_parsed_as_csv() {
        // A path with a comma is quoted; splitting on the first comma would
        // truncate it to a fragment and miss the script.
        assert!(record_row_targets_script(
            "\"../../../bin/nested,dir/rocm\",sha256=aaa,17000000",
            "rocm",
            "bin"
        ));
        assert!(record_row_targets_script(
            "\"../../../bin/ro\"\"cm\",sha256=bbb,4096",
            "ro\"cm",
            "bin"
        ));
        assert!(!record_row_targets_script(
            "\"../../../bin/rocm,sha256=ccc,4096",
            "rocm",
            "bin"
        ));
    }

    #[test]
    fn keep_binaries_still_wins_over_the_python_notice() {
        let tree = Tree::new("venv/bin/rocm");
        tree.write("venv/pyvenv.cfg", "home = /usr/bin\n");
        let mut plan = UninstallPlan::default();
        let options = UninstallOptions {
            keep_binaries: true,
            ..UninstallOptions::default()
        };
        plan_installed_binaries(
            &mut plan,
            &options,
            || Ok(tree.exe.clone()),
            Some(tree.root.path()),
            &rocm_core::canonicalize_for_compare,
        )
        .expect("plan");
        assert_eq!(plan.skipped.len(), 1, "{:?}", plan.skipped);
        assert!(plan.skipped[0].contains("--keep-binaries"));
    }

    #[test]
    fn force_dev_binaries_does_not_reach_a_python_environment() {
        // `plan_installed_binaries` reads `--force-dev-binaries` only in its
        // cargo-build-folder check, which runs before this guard; a Python
        // environment stays skipped with it set.
        let tree = Tree::new("venv/bin/rocm");
        tree.write("venv/pyvenv.cfg", "home = /usr/bin\n");
        let mut plan = UninstallPlan::default();
        let options = UninstallOptions {
            force_dev_binaries: true,
            ..UninstallOptions::default()
        };
        plan_installed_binaries(
            &mut plan,
            &options,
            || Ok(tree.exe.clone()),
            Some(tree.root.path()),
            &rocm_core::canonicalize_for_compare,
        )
        .expect("plan");
        assert_eq!(plan.skipped, [skip_notice(&tree.exe)]);
        assert!(plan.actions.iter().all(|entry| entry.kind != "binary"));
    }
}
