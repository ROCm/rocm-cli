// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which `librocdxg` a managed wheel runtime's HSA runtime actually loads on WSL.
//!
//! The ROCm wheels ship their own `librocdxg.so.1` in `_rocm_sdk_core/lib`, next
//! to `libhsa-runtime64.so.1`. The HSA runtime does not ask for that name: it
//! opens the unversioned `librocdxg.so`, and the wheel ships no such file. The
//! lookup therefore skips the runtime's own copy and lands on whatever the host
//! provides (`/opt/rocm/lib/librocdxg.so` from the `rocdxg-roct` package), which
//! can be older than the runtime expects. A 10.2 or newer nightly runtime asks
//! that copy for `hsaKmtSetPersistingCacheSize`, which `rocdxg-roct` 1.2.2 does
//! not export, and crashes on first GPU use.
//!
//! The mechanism (the unversioned open skipping the runtime's directory and
//! landing on `/opt/rocm/lib`) was observed with `LD_DEBUG=libs` on the 10.2
//! nightly. Which component requests `hsaKmtSetPersistingCacheSize` is not
//! identified.
//!
//! Linking `librocdxg.so` to the runtime's own `librocdxg.so.1` makes the runtime
//! load the copy it was built with. The link sits in the runtime's own library
//! directory, which `libhsa-runtime64` searches through `$ORIGIN` (a `DT_RPATH`
//! in the 10.1, 10.2 and 10.3 wheels per `readelf -d`, which the loader searches
//! before `LD_LIBRARY_PATH`), so it works for a bare `<runtime>/bin/python` as
//! well as for a process the CLI launches.
//!
//! The name is judged three ways ([`NameState`]): absent, present and resolving
//! (a regular file or a link that resolves: left alone), and
//! dangling (a link whose target is missing). A dangling `librocdxg.so` is as bad
//! as an absent one: the loader's `open` fails with `ENOENT` on a broken link,
//! tries the next directory and loads the host's copy (`LD_DEBUG=libs` on the
//! 10.2 nightly with a dangling link first on `LD_LIBRARY_PATH` showed the link
//! tried and skipped, then exit 139 as with no link). So detection lists a
//! dangling name like an absent one, and the install replaces it atomically.
//!
//! Limits. The versioned name is the hard-coded `librocdxg.so.1`: a wheel whose
//! soname is no longer that is neither detected nor linked, and no globbing is
//! done (which versioned file would win is a design decision). A dangling
//! `librocdxg.so` beside a runtime that ships no `librocdxg.so.1` is not
//! detected either, because there is nothing to link it to. A versioned copy that
//! is itself dangling (or a link loop) is likewise left out of both `rocm examine`
//! and the install, and deliberately not reported: there is nothing valid to link
//! to, and a runtime whose own `librocdxg.so.1` is broken is a different, wider
//! fault than the one this module covers.
//!
//! "Never replaced" holds for a name that resolves when it is checked. The
//! replacement of a dangling link re-checks the name immediately before the
//! rename, but a name that becomes a working link between that re-check and the
//! rename is still replaced: `rename` has no "only if still dangling" form. The
//! window is short in a single-process, one-shot install (not measured). A name
//! that vanishes between the first check and the re-check is simply linked.
//!
//! Only wheel runtime installs and updates (`install_wheel_runtime`) link.
//! Tarballs ship both names, and an adopted or imported runtime is never linked
//! by the CLI. The link is not tracked by the installer, so an uninstall that
//! removes the runtime's files by package leaves it behind.

use std::io;
use std::path::{Path, PathBuf};

/// The versioned file the wheels ship.
const SHIPPED_NAME: &str = "librocdxg.so.1";
/// The name the HSA runtime opens.
const OPENED_NAME: &str = "librocdxg.so";

/// What is at a path, as the loader sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameState {
    /// Nothing is there.
    Absent,
    /// A regular file, or a link whose target exists.
    Resolves,
    /// A link whose target does not exist, or that loops (`ELOOP`): `open` fails
    /// on it, so the loader skips it as it would an absent name.
    Dangling,
}

/// The [`NameState`] of `path` on this machine.
#[must_use]
pub fn name_state(path: &Path) -> NameState {
    if path.metadata().is_ok() {
        NameState::Resolves
    } else if path.symlink_metadata().is_ok() {
        NameState::Dangling
    } else {
        NameState::Absent
    }
}

/// The runtime's own `librocdxg.so.1` for each directory in `dirs` that holds
/// one next to a `libhsa-runtime64` but whose `librocdxg.so` is absent or
/// dangling.
///
/// In such a directory the runtime's lookup for `librocdxg.so` does not find the
/// runtime's own copy, so it loads the host's. A directory with the library but
/// no runtime beside it is left out: nothing there opens it by name. One whose
/// `librocdxg.so` resolves (a file or a working link) is left out as well; there
/// is nothing for this module to add. A directory whose `librocdxg.so.1` is not
/// there or does not resolve (absent, dangling, or a link loop) is left out:
/// there is nothing valid to link to, and that broken copy is not reported.
///
/// Takes the state check rather than reading the machine, so the layouts can be
/// tested without one; the real check is [`name_state`].
#[must_use]
pub fn unlinked_runtime_copies(
    dirs: &[PathBuf],
    state: impl Fn(&Path) -> NameState,
) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        let copy = dir.join(SHIPPED_NAME);
        let has_runtime = ["libhsa-runtime64.so.1", "libhsa-runtime64.so"]
            .iter()
            .any(|name| state(&dir.join(name)) == NameState::Resolves);
        if state(&copy) == NameState::Resolves
            && state(&dir.join(OPENED_NAME)) != NameState::Resolves
            && has_runtime
            && !found.contains(&copy)
        {
            found.push(copy);
        }
    }
    found
}

/// The `librocdxg.so.1` copies of every managed runtime whose own HSA runtime
/// would not load them, for `rocm examine`.
///
/// Empty off WSL (`is_wsl` false): the advice is about the WSL loader path and
/// would mislead on a native host.
#[must_use]
pub fn unlinked_managed_runtime_copies(is_wsl: bool, paths: &crate::AppPaths) -> Vec<PathBuf> {
    if !is_wsl {
        return Vec::new();
    }
    unlinked_runtime_copies(
        &crate::managed_runtime::managed_runtime_library_dirs(paths),
        name_state,
    )
}

/// The whole post-install step: on WSL, link every runtime copy in `dirs` that
/// its HSA runtime would not load, and return one line per copy saying what
/// happened, for the install report.
///
/// Reads the WSL gate itself ([`crate::is_wsl_host`], which follows a simulated
/// host root in `e2e-test-hooks` builds), so a caller cannot pass the wrong
/// answer and a test can drive both hosts.
#[must_use]
pub fn link_installed_runtime_copies(dirs: &[PathBuf]) -> Vec<String> {
    link_installed_runtime_copies_when(crate::is_wsl_host(), dirs)
}

/// [`link_installed_runtime_copies`] with the gate supplied.
///
/// Each line is built from the outcome of that copy's own link attempt, so a
/// link that failed cannot be reported as made. A failure is not an error for
/// the install: the line gives the error and the command to run once its cause
/// is fixed (the same link call failed, so the command alone may fail the same
/// way). Nothing is touched off WSL, where the HSA runtime does not open
/// `librocdxg`.
pub fn link_installed_runtime_copies_when(is_wsl: bool, dirs: &[PathBuf]) -> Vec<String> {
    if !is_wsl {
        return Vec::new();
    }
    link_copies_with(dirs, name_state, link_runtime_copy)
}

fn link_copies_with(
    dirs: &[PathBuf],
    state: impl Fn(&Path) -> NameState,
    link: impl Fn(&Path) -> io::Result<Linked>,
) -> Vec<String> {
    unlinked_runtime_copies(dirs, state)
        .iter()
        .map(|copy| match link(copy) {
            Ok(linked) if linked.replaced_dangling => format!(
                "librocdxg: replaced a dangling {} with a link to the runtime's own copy; the runtime searches its own directory first",
                linked.path.display()
            ),
            Ok(linked) => format!(
                "librocdxg: linked {} to the runtime's own copy; the runtime searches its own directory first",
                linked.path.display()
            ),
            Err(error) => format!(
                "librocdxg: could not link {}: {error}; the runtime may load the host's librocdxg, which fails at first GPU use if it is older than the runtime expects; once the cause above is fixed, run: {}",
                copy.with_file_name(OPENED_NAME).display(),
                link_command(copy)
            ),
        })
        .collect()
}

/// The command that gives `copy`'s directory the unversioned name, for advice
/// that has to name something the user can run.
///
/// `-f` and `-n` make it clear a dangling `librocdxg.so` as well as an absent
/// one (a plain `ln -s` fails with "File exists" on a dangling link). The advice
/// is only printed for an absent or dangling name, never for one that resolves.
///
/// A path containing a newline or invalid UTF-8 is not handled: the advice is one
/// line printed through `Path::display`.
#[must_use]
pub fn link_command(copy: &Path) -> String {
    let dir = copy.parent().unwrap_or_else(|| Path::new("."));
    // A single quote in the path is closed, escaped and reopened.
    let target = dir
        .join(OPENED_NAME)
        .display()
        .to_string()
        .replace('\'', "'\\''");
    format!("ln -sfn {SHIPPED_NAME} '{target}'")
}

/// What [`link_runtime_copy`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linked {
    /// The `librocdxg.so` link that now exists.
    pub path: PathBuf,
    /// A dangling `librocdxg.so` was there and was replaced.
    pub replaced_dangling: bool,
}

/// Give `copy`'s directory a `librocdxg.so` that points at `copy`.
///
/// The link is relative (`librocdxg.so -> librocdxg.so.1`), so it survives the
/// runtime directory being moved or restored from a cache.
///
/// What is already at `librocdxg.so` decides (see [`NameState`]): nothing, and
/// the link is created; a dangling link, and it is replaced (a temporary link in
/// the same directory is renamed over it, so the name is never absent in
/// between; see `replace_with_link` for the one window that remains); a file
/// or a working link, and it is left alone and reported as an
/// [`io::ErrorKind::AlreadyExists`] error. `copy` itself is not checked: the
/// caller links only a copy that exists.
///
/// # Errors
/// Returns the underlying error when the link cannot be created, or
/// [`io::ErrorKind::Unsupported`] off Unix (the WSL path is Linux only).
pub fn link_runtime_copy(copy: &Path) -> io::Result<Linked> {
    let dir = copy.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "librocdxg copy has no directory",
        )
    })?;
    let link = dir.join(OPENED_NAME);
    #[cfg(unix)]
    {
        match name_state(&link) {
            NameState::Absent => {
                std::os::unix::fs::symlink(SHIPPED_NAME, &link)?;
                Ok(Linked {
                    path: link,
                    replaced_dangling: false,
                })
            }
            NameState::Dangling => {
                // Whether a dangling link was really replaced is what the final
                // check found: a name that vanished since is linked, not replaced.
                let replaced_dangling = replace_with_link(dir, &link)?;
                Ok(Linked {
                    path: link,
                    replaced_dangling,
                })
            }
            NameState::Resolves => Err(io::Error::new(io::ErrorKind::AlreadyExists, "File exists")),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = link;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "linking librocdxg is only done on Linux",
        ))
    }
}

/// Replace the dangling `link` with a link to [`SHIPPED_NAME`] without a window in
/// which the name is absent: make a temporary link beside it, then rename over it.
///
/// The name is re-checked just before the rename. Still dangling, or gone since
/// the first check: the temporary is renamed into place (nothing is clobbered).
/// Resolving by now (a file or working link appeared): left alone, with an
/// [`io::ErrorKind::AlreadyExists`] error. That narrows the check-then-act window
/// but cannot close it: a name that becomes a working link between the re-check
/// and the rename is replaced. If the rename fails, or the name is refused, the
/// temporary is removed, and a failure to remove it is part of the returned error.
///
/// The temporary's name carries the process id and a process-wide counter, so two
/// replacements in one process never share one. A leftover from a crashed process
/// could only collide after the pid is reused, and then `symlink` fails with
/// `File exists`, which is reported, not acted on.
#[cfg(unix)]
fn replace_with_link(dir: &Path, link: &Path) -> io::Result<bool> {
    replace_with_link_using(
        dir,
        link,
        name_state,
        |from, to| std::fs::rename(from, to),
        |path| std::fs::remove_file(path),
    )
}

#[cfg(unix)]
fn replace_with_link_using(
    dir: &Path,
    link: &Path,
    state: impl Fn(&Path) -> NameState,
    rename: impl Fn(&Path, &Path) -> io::Result<()>,
    remove: impl Fn(&Path) -> io::Result<()>,
) -> io::Result<bool> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let temporary = dir.join(format!(
        ".{OPENED_NAME}.tmp-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::os::unix::fs::symlink(SHIPPED_NAME, &temporary)?;
    // Whether the name dangled at the final check: only then was a link replaced.
    let mut replaced_dangling = false;
    let outcome = match state(link) {
        NameState::Dangling => {
            replaced_dangling = true;
            rename(&temporary, link)
        }
        NameState::Absent => rename(&temporary, link),
        NameState::Resolves => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "File exists (it resolves now)",
        )),
    };
    if let Err(error) = outcome {
        return Err(match remove(&temporary) {
            Ok(()) => error,
            Err(cleanup) => io::Error::new(
                error.kind(),
                format!(
                    "{error}; and the temporary link {} could not be removed: {cleanup}",
                    temporary.display()
                ),
            ),
        });
    }
    Ok(replaced_dangling)
}

#[cfg(test)]
mod tests {
    // The tests pin Unix paths and the shell command, and the code under test is
    // WSL-only, so they run on Unix only. The module is a plain `cfg(test)` one
    // so the registry-writer guard in `apps/rocm/src/runtime_registry.rs` cuts it
    // off as test code.
    #[cfg(unix)]
    mod unix {
        use super::super::*;
        use std::collections::HashSet;

        fn files(paths: &[&str]) -> impl Fn(&Path) -> NameState {
            let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
            move |path| {
                if set.contains(path) {
                    NameState::Resolves
                } else {
                    NameState::Absent
                }
            }
        }

        fn write(path: &Path, contents: &[u8]) {
            std::fs::create_dir_all(path.parent().expect("parent")).expect("dir");
            std::fs::write(path, contents).expect("write");
        }

        #[test]
        fn a_runtime_that_ships_only_the_versioned_copy_is_reported() {
            let dirs = vec![PathBuf::from("/rt/_rocm_sdk_core/lib")];
            let exists = files(&[
                "/rt/_rocm_sdk_core/lib/librocdxg.so.1",
                "/rt/_rocm_sdk_core/lib/libhsa-runtime64.so.1",
            ]);
            assert_eq!(
                unlinked_runtime_copies(&dirs, exists),
                vec![PathBuf::from("/rt/_rocm_sdk_core/lib/librocdxg.so.1")]
            );
        }

        #[test]
        fn a_directory_with_the_unversioned_name_is_not_reported() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = files(&[
                "/rt/lib/librocdxg.so.1",
                "/rt/lib/librocdxg.so",
                "/rt/lib/libhsa-runtime64.so.1",
            ]);
            assert!(unlinked_runtime_copies(&dirs, exists).is_empty());
        }

        #[test]
        fn the_library_without_a_runtime_beside_it_is_not_reported() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = files(&["/rt/lib/librocdxg.so.1"]);
            assert!(unlinked_runtime_copies(&dirs, exists).is_empty());
        }

        #[test]
        fn a_runtime_directory_without_the_versioned_copy_is_not_reported() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = files(&["/rt/lib/libhsa-runtime64.so.1"]);
            assert!(unlinked_runtime_copies(&dirs, exists).is_empty());
        }

        /// `link_runtime_copy` itself links whatever it is given; the install step
        /// is what decides to call it, and it must not make a dangling link in a
        /// directory that holds a runtime but no `librocdxg.so.1`.
        #[test]
        fn the_install_step_makes_no_link_where_the_versioned_copy_is_missing() {
            let dir = tempfile::tempdir().expect("temp dir");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            let lines = link_installed_runtime_copies_when(true, &[dir.path().to_path_buf()]);

            assert!(lines.is_empty(), "{lines:?}");
            assert!(dir.path().join("librocdxg.so").symlink_metadata().is_err());
        }

        #[test]
        fn a_directory_listed_twice_is_reported_once() {
            let dirs = vec![PathBuf::from("/rt/lib"), PathBuf::from("/rt/lib")];
            let exists = files(&["/rt/lib/librocdxg.so.1", "/rt/lib/libhsa-runtime64.so"]);
            assert_eq!(unlinked_runtime_copies(&dirs, exists).len(), 1);
        }

        #[test]
        fn link_command_names_the_unversioned_file_beside_the_copy() {
            let copy = PathBuf::from("/rt/lib/librocdxg.so.1");
            assert_eq!(
                link_command(&copy),
                "ln -sfn librocdxg.so.1 '/rt/lib/librocdxg.so'"
            );
        }

        #[test]
        fn link_command_escapes_a_single_quote_in_the_path() {
            let copy = PathBuf::from("/rt/it's/librocdxg.so.1");
            assert_eq!(
                link_command(&copy),
                "ln -sfn librocdxg.so.1 '/rt/it'\\''s/librocdxg.so'"
            );
        }

        #[test]
        fn the_command_and_the_install_make_the_same_relative_link() {
            let by_install = tempfile::tempdir().expect("temp dir");
            let by_command = tempfile::tempdir().expect("temp dir");
            // The command's directory has a quote in it, so the quoting is exercised.
            let quoted = by_command.path().join("it's");
            write(&by_install.path().join("librocdxg.so.1"), b"own");
            write(&quoted.join("librocdxg.so.1"), b"own");

            let link = link_runtime_copy(&by_install.path().join("librocdxg.so.1"))
                .expect("link")
                .path;
            let status = std::process::Command::new("sh")
                .arg("-c")
                .arg(link_command(&quoted.join("librocdxg.so.1")))
                .status()
                .expect("run the advised command");

            assert!(status.success());
            let installed = std::fs::read_link(&link).expect("read_link");
            assert_eq!(
                installed,
                PathBuf::from("librocdxg.so.1"),
                "relative target"
            );
            assert_eq!(
                std::fs::read_link(quoted.join("librocdxg.so")).expect("read_link"),
                installed,
                "the advised command makes the link the install makes"
            );
        }

        #[test]
        fn a_failed_link_is_reported_as_failed_with_the_manual_command() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = files(&["/rt/lib/librocdxg.so.1", "/rt/lib/libhsa-runtime64.so.1"]);
            let lines = link_copies_with(&dirs, exists, |_| {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
            });
            assert_eq!(lines.len(), 1);
            assert!(lines[0].contains("could not link"), "{lines:?}");
            assert!(!lines[0].contains("linked /"), "{lines:?}");
            assert!(
                lines[0].contains("ln -sfn librocdxg.so.1 '/rt/lib/librocdxg.so'"),
                "{lines:?}"
            );
        }

        /// A runtime directory whose `librocdxg.so` is a link to nowhere.
        fn runtime_dir_with_a_dangling_link() -> tempfile::TempDir {
            let dir = tempfile::tempdir().expect("temp dir");
            write(&dir.path().join("librocdxg.so.1"), b"own");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            std::os::unix::fs::symlink("gone.so.9", dir.path().join("librocdxg.so"))
                .expect("dangling link");
            dir
        }

        /// The runtime directory with the versioned name in a given state; the
        /// unversioned name is absent.
        fn runtime_dir_with_versioned_name(make: impl Fn(&Path)) -> tempfile::TempDir {
            let dir = tempfile::tempdir().expect("temp dir");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            make(&dir.path().join("librocdxg.so.1"));
            dir
        }

        fn temporaries_in(dir: &Path) -> usize {
            std::fs::read_dir(dir)
                .expect("read_dir")
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
                .count()
        }

        #[test]
        fn the_versioned_copy_decides_whether_a_directory_is_listed() {
            type Make = fn(&Path);
            let cases: [(&str, Make, bool); 5] = [
                ("a regular file", |p| write(p, b"own"), true),
                (
                    "a link that resolves",
                    |p| {
                        write(&p.with_file_name("real.so"), b"own");
                        std::os::unix::fs::symlink("real.so", p).expect("link");
                    },
                    true,
                ),
                ("absent", |_| {}, false),
                (
                    "a dangling link",
                    |p| std::os::unix::fs::symlink("gone.so", p).expect("link"),
                    false,
                ),
                (
                    "a link loop",
                    |p| std::os::unix::fs::symlink("librocdxg.so.1", p).expect("link"),
                    false,
                ),
            ];
            for (what, make, listed) in cases {
                let dir = runtime_dir_with_versioned_name(make);
                let dirs = vec![dir.path().to_path_buf()];
                let found = unlinked_runtime_copies(&dirs, name_state);
                assert_eq!(found.is_empty(), !listed, "{what}: {found:?}");

                let lines = link_installed_runtime_copies_when(true, &dirs);
                assert_eq!(lines.is_empty(), !listed, "{what}: {lines:?}");
                assert_eq!(
                    dir.path().join("librocdxg.so").symlink_metadata().is_ok(),
                    listed,
                    "{what}: a link is made only for a copy that can be linked to"
                );
                assert_eq!(temporaries_in(dir.path()), 0, "{what}");
            }
        }

        #[test]
        fn examine_does_not_list_a_runtime_whose_versioned_copy_dangles() {
            let (root, paths) = crate::test_support::temp_app_paths("runtime-rocdxg-dangling-copy");
            let lib = root.join("wheel/lib");
            write(&lib.join("libhsa-runtime64.so.1"), b"");
            std::os::unix::fs::symlink("gone.so", lib.join("librocdxg.so.1")).expect("link");
            register_runtime(
                &paths.data_dir.join("runtimes").join("registry"),
                "k",
                &[&lib],
            );

            assert!(unlinked_managed_runtime_copies(true, &paths).is_empty());
        }

        #[test]
        fn examine_scans_every_directory_the_install_would_link() {
            // The install links the probe's `library_paths`; the registry record
            // holds the same list, and examine scans what the environment builder
            // makes of it. Both are computed separately, so pin that the second
            // contains the first for a wheel-shaped runtime.
            let (root, paths) = crate::test_support::temp_app_paths("runtime-rocdxg-dir-sets");
            let wheel = root.join("rt/lib/python3.14/site-packages");
            let probe_library_paths = [
                wheel.join("_rocm_sdk_core/lib"),
                wheel.join("_rocm_sdk_core/lib/rocm_sysdeps/lib"),
                wheel.join("_rocm_sdk_libraries/lib"),
            ];
            for dir in &probe_library_paths {
                std::fs::create_dir_all(dir).expect("dir");
            }
            let refs: Vec<&Path> = probe_library_paths.iter().map(PathBuf::as_path).collect();
            register_runtime(
                &paths.data_dir.join("runtimes").join("registry"),
                "k",
                &refs,
            );

            let scanned = crate::managed_runtime::managed_runtime_library_dirs(&paths);
            for dir in &probe_library_paths {
                assert!(
                    scanned.contains(dir),
                    "examine does not scan {dir:?}: {scanned:?}"
                );
            }
        }

        #[test]
        fn a_dangling_unversioned_link_is_listed_like_an_absent_one() {
            let dir = runtime_dir_with_a_dangling_link();
            let dirs = vec![dir.path().to_path_buf()];

            assert_eq!(
                name_state(&dir.path().join("librocdxg.so")),
                NameState::Dangling
            );
            assert_eq!(
                unlinked_runtime_copies(&dirs, name_state),
                vec![dir.path().join("librocdxg.so.1")],
                "the loader skips a broken link, so the runtime loads the host's copy"
            );
        }

        #[test]
        fn the_install_step_replaces_a_dangling_link_and_leaves_no_temporary() {
            let dir = runtime_dir_with_a_dangling_link();
            let lines = link_installed_runtime_copies_when(true, &[dir.path().to_path_buf()]);

            let link = dir.path().join("librocdxg.so");
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert!(
                lines[0].contains(&format!("replaced a dangling {}", link.display())),
                "{lines:?}"
            );
            assert_eq!(std::fs::read(&link).expect("resolves now"), b"own");
            assert_eq!(
                std::fs::read_link(&link).expect("read_link"),
                PathBuf::from("librocdxg.so.1")
            );
            let leftovers: Vec<_> = std::fs::read_dir(dir.path())
                .expect("read_dir")
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .filter(|name| name.contains(".tmp-"))
                .collect();
            assert!(
                leftovers.is_empty(),
                "no temporary link is left: {leftovers:?}"
            );
        }

        #[test]
        fn a_resolving_link_or_a_file_is_never_replaced() {
            for make in [
                |dir: &Path| {
                    write(&dir.join("other.so"), b"theirs");
                    std::os::unix::fs::symlink("other.so", dir.join("librocdxg.so")).expect("link");
                },
                |dir: &Path| write(&dir.join("librocdxg.so"), b"theirs"),
            ] {
                let dir = runtime_dir_shipping_only_the_versioned_copy();
                make(dir.path());
                let lines = link_installed_runtime_copies_when(true, &[dir.path().to_path_buf()]);

                assert!(lines.is_empty(), "nothing to do: {lines:?}");
                assert_eq!(
                    std::fs::read(dir.path().join("librocdxg.so")).expect("untouched"),
                    b"theirs"
                );
                assert!(link_runtime_copy(&dir.path().join("librocdxg.so.1")).is_err());
                assert_eq!(
                    std::fs::read(dir.path().join("librocdxg.so")).expect("still untouched"),
                    b"theirs"
                );
            }
        }

        #[test]
        fn a_name_that_stopped_dangling_before_the_rename_is_left_alone() {
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            write(&dir.path().join("librocdxg.so"), b"theirs");
            let link = dir.path().join("librocdxg.so");
            // The caller saw a dangling name; by the re-check it resolves.
            let outcome = replace_with_link(dir.path(), &link);

            assert_eq!(
                outcome.expect_err("refused").kind(),
                io::ErrorKind::AlreadyExists
            );
            assert_eq!(std::fs::read(&link).expect("untouched"), b"theirs");
            let leftovers = std::fs::read_dir(dir.path())
                .expect("read_dir")
                .flatten()
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
                .count();
            assert_eq!(leftovers, 0, "the temporary link is removed");
        }

        #[test]
        fn a_failed_cleanup_is_part_of_the_error() {
            let dir = runtime_dir_with_a_dangling_link();
            let link = dir.path().join("librocdxg.so");
            let outcome = replace_with_link_using(
                dir.path(),
                &link,
                name_state,
                |_, _| {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "rename denied",
                    ))
                },
                |_| {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "remove denied",
                    ))
                },
            );

            let message = outcome.expect_err("failed").to_string();
            assert!(message.contains("rename denied"), "{message}");
            assert!(
                message.contains("and the temporary link")
                    && message.contains("could not be removed: remove denied"),
                "{message}"
            );
        }

        #[test]
        fn a_failed_rename_removes_the_temporary_link_from_disk() {
            let dir = runtime_dir_with_a_dangling_link();
            let link = dir.path().join("librocdxg.so");
            // The rename is injected to fail (the name still dangles at the
            // re-check, so the rename is reached); removal is the real one, so the
            // temporary link that was made on disk must be gone afterwards.
            let outcome = replace_with_link_using(
                dir.path(),
                &link,
                name_state,
                |_, _| {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "rename denied",
                    ))
                },
                |path| std::fs::remove_file(path),
            );

            assert!(outcome.is_err());
            assert_eq!(
                temporaries_in(dir.path()),
                0,
                "the temporary link is removed"
            );
            assert_eq!(
                name_state(&link),
                NameState::Dangling,
                "the dangling name is as it was"
            );
        }

        #[test]
        fn a_name_that_vanished_before_the_recheck_is_linked() {
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            let link = dir.path().join("librocdxg.so");
            // The caller saw a dangling name; by the re-check nothing is there.
            let outcome = replace_with_link_using(
                dir.path(),
                &link,
                |_| NameState::Absent,
                |from, to| std::fs::rename(from, to),
                |path| std::fs::remove_file(path),
            );

            // Nothing was replaced: the name was gone, so `false`.
            assert!(!outcome.expect("linked"));
            assert_eq!(std::fs::read(&link).expect("resolves"), b"own");
            assert_eq!(temporaries_in(dir.path()), 0);
        }

        #[test]
        fn the_three_cases_each_print_their_own_line() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = || files(&["/rt/lib/librocdxg.so.1", "/rt/lib/libhsa-runtime64.so.1"]);
            let line = |replaced_dangling| {
                link_copies_with(&dirs, exists(), |copy| {
                    Ok(Linked {
                        path: copy.with_file_name("librocdxg.so"),
                        replaced_dangling,
                    })
                })
                .remove(0)
            };

            // Absent from the start, and vanished at the re-check, both `false`.
            assert!(line(false).starts_with("librocdxg: linked /rt/lib/librocdxg.so to"));
            assert!(!line(false).contains("replaced"));
            assert!(
                line(true).starts_with("librocdxg: replaced a dangling /rt/lib/librocdxg.so with")
            );
        }

        #[test]
        fn a_dangling_name_replaced_reports_true_and_an_absent_one_linked_reports_false() {
            let dir = runtime_dir_with_a_dangling_link();
            let copy = dir.path().join("librocdxg.so.1");
            let replaced = link_runtime_copy(&copy).expect("replaced");
            assert!(replaced.replaced_dangling);

            let absent = runtime_dir_shipping_only_the_versioned_copy();
            let linked = link_runtime_copy(&absent.path().join("librocdxg.so.1")).expect("linked");
            assert!(!linked.replaced_dangling);
        }

        /// Two threads replacing dangling links in one directory at once (two
        /// runtime copies in one directory is the shape the counter exists for):
        /// every temporary name must differ, or one rename fails or a link is lost.
        #[test]
        fn concurrent_replacements_in_one_directory_use_distinct_temporary_names() {
            const ROUNDS: usize = 200;
            let dir = tempfile::tempdir().expect("temp dir");
            let barrier = std::sync::Barrier::new(2);
            let results: Vec<Vec<io::Result<bool>>> = std::thread::scope(|scope| {
                let run = |name: &'static str| {
                    let (dir, barrier) = (dir.path(), &barrier);
                    scope.spawn(move || {
                        (0..ROUNDS)
                            .map(|round| {
                                let link = dir.join(format!("{name}-{round}"));
                                std::os::unix::fs::symlink("gone", &link).expect("dangling");
                                barrier.wait();
                                replace_with_link(dir, &link)
                            })
                            .collect::<Vec<_>>()
                    })
                };
                let (a, b) = (run("a"), run("b"));
                vec![a.join().expect("thread a"), b.join().expect("thread b")]
            });

            for outcome in results.iter().flatten() {
                assert!(outcome.is_ok(), "{outcome:?}");
            }
            for name in ["a", "b"] {
                for round in 0..ROUNDS {
                    let link = dir.path().join(format!("{name}-{round}"));
                    assert_eq!(
                        std::fs::read_link(&link).expect("still a link"),
                        PathBuf::from("librocdxg.so.1"),
                        "{link:?}"
                    );
                }
            }
        }

        #[test]
        fn the_advised_command_clears_a_dangling_link() {
            let dir = runtime_dir_with_a_dangling_link();
            let status = std::process::Command::new("sh")
                .arg("-c")
                .arg(link_command(&dir.path().join("librocdxg.so.1")))
                .status()
                .expect("run the advised command");

            assert!(
                status.success(),
                "a plain `ln -s` would fail with File exists"
            );
            assert_eq!(
                std::fs::read(dir.path().join("librocdxg.so")).expect("resolves"),
                b"own"
            );
            assert!(unlinked_runtime_copies(&[dir.path().to_path_buf()], name_state).is_empty());
        }

        fn runtime_dir_shipping_only_the_versioned_copy() -> tempfile::TempDir {
            let dir = tempfile::tempdir().expect("temp dir");
            write(&dir.path().join("librocdxg.so.1"), b"own");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            dir
        }

        #[test]
        fn the_install_step_on_wsl_links_and_reports_it() {
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            let lines = link_installed_runtime_copies_when(true, &[dir.path().to_path_buf()]);

            let link = dir.path().join("librocdxg.so");
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert!(
                lines[0].contains(&format!(
                    "linked {} to the runtime's own copy",
                    link.display()
                )),
                "{lines:?}"
            );
            assert_eq!(std::fs::read(&link).expect("through link"), b"own");
        }

        #[test]
        fn the_install_step_off_wsl_touches_nothing() {
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            let lines = link_installed_runtime_copies_when(false, &[dir.path().to_path_buf()]);

            assert!(lines.is_empty(), "{lines:?}");
            assert!(
                dir.path().join("librocdxg.so").symlink_metadata().is_err(),
                "no link may be made off WSL"
            );
        }

        /// Drives the real gate, [`crate::is_wsl_host`], through the simulated host
        /// root, so the entry point the install calls is tested on a WSL host and on a
        /// native one. Needs the `e2e-test-hooks` feature (see docs/testing.md).
        #[cfg(feature = "e2e-test-hooks")]
        #[test]
        fn the_install_entry_point_reads_the_wsl_gate_from_the_host() {
            use crate::hardware_root::TEST_HOST_ROOT_ENV;
            use crate::test_env::RestoredEnvVar;

            let wsl_host = tempfile::tempdir().expect("temp dir");
            write(&wsl_host.path().join("dev/dxg"), b"");
            let native_host = tempfile::tempdir().expect("temp dir");
            write(
                &native_host.path().join("proc/version"),
                b"Linux version 6.8.0-generic",
            );

            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            let native = runtime_dir_shipping_only_the_versioned_copy();
            let root_guard = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, native_host.path());
            let lines = link_installed_runtime_copies(&[native.path().to_path_buf()]);
            assert!(lines.is_empty(), "{lines:?}");
            assert!(
                native
                    .path()
                    .join("librocdxg.so")
                    .symlink_metadata()
                    .is_err()
            );
            drop(root_guard);

            let wsl = runtime_dir_shipping_only_the_versioned_copy();
            let _wsl_root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, wsl_host.path());
            let lines = link_installed_runtime_copies(&[wsl.path().to_path_buf()]);
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert_eq!(
                std::fs::read(wsl.path().join("librocdxg.so")).expect("through link"),
                b"own"
            );
        }

        #[test]
        fn the_install_step_reports_a_failed_link_with_the_manual_command() {
            // The failure does not rely on permissions (root ignores them): the state
            // check handed to `link_copies_with` is told `librocdxg.so` is absent
            // while a file really is there, so `link_runtime_copy`'s own check of
            // the name refuses it with `AlreadyExists` ("File exists") for any
            // user. It is that check, not the OS `symlink` call, that fails here.
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            write(&dir.path().join("librocdxg.so"), b"someone else's file");
            let absent_unversioned = |path: &Path| {
                if path.file_name().is_some_and(|name| name == "librocdxg.so") {
                    NameState::Absent
                } else {
                    name_state(path)
                }
            };
            let lines = link_copies_with(
                &[dir.path().to_path_buf()],
                absent_unversioned,
                link_runtime_copy,
            );

            assert_eq!(lines.len(), 1, "{lines:?}");
            let link = dir.path().join("librocdxg.so");
            // The error text stays in the line, and the command is offered only
            // once its cause is fixed: the same link call failed here.
            assert!(
                lines[0].contains(&format!("could not link {}: ", link.display()))
                    && lines[0].contains("File exists"),
                "{lines:?}"
            );
            assert!(
                lines[0].contains("once the cause above is fixed, run: ln -sfn librocdxg.so.1"),
                "{lines:?}"
            );
            assert_eq!(
                std::fs::read(dir.path().join("librocdxg.so")).expect("untouched"),
                b"someone else's file"
            );
        }

        fn register_runtime(registry: &Path, key: &str, libs: &[&Path]) {
            let record = serde_json::json!({
                "runtime_key": key, "runtime_id": "therock-nightly:gfx1151",
                "rocm_sdk": { "import_ok": true, "library_paths": libs },
            });
            write(
                &registry.join(format!("{key}.json")),
                record.to_string().as_bytes(),
            );
        }

        #[test]
        fn examine_lists_the_copies_of_every_registered_runtime_only_on_wsl() {
            let (root, paths) = crate::test_support::temp_app_paths("runtime-rocdxg-registry");
            let (first, shared, second) = (
                root.join("first/lib"),
                root.join("shared/lib"),
                root.join("second/lib"),
            );
            for lib in [&first, &shared, &second] {
                write(&lib.join("librocdxg.so.1"), b"");
                write(&lib.join("libhsa-runtime64.so.1"), b"");
            }
            let registry = paths.data_dir.join("runtimes").join("registry");
            // Two records, one directory in both: every runtime is scanned, and the
            // shared directory is listed once.
            register_runtime(&registry, "a", &[&first, &shared]);
            register_runtime(&registry, "b", &[&shared, &second]);

            let mut listed = unlinked_managed_runtime_copies(true, &paths);
            listed.sort();
            let mut expected = vec![
                first.join("librocdxg.so.1"),
                shared.join("librocdxg.so.1"),
                second.join("librocdxg.so.1"),
            ];
            expected.sort();
            assert_eq!(listed, expected);
            assert!(
                unlinked_managed_runtime_copies(false, &paths).is_empty(),
                "a native host gets no WSL advice"
            );
        }

        #[test]
        fn a_made_link_is_reported_as_made() {
            let dirs = vec![PathBuf::from("/rt/lib")];
            let exists = files(&["/rt/lib/librocdxg.so.1", "/rt/lib/libhsa-runtime64.so.1"]);
            let lines = link_copies_with(&dirs, exists, |copy| {
                Ok(Linked {
                    path: copy.with_file_name("librocdxg.so"),
                    replaced_dangling: false,
                })
            });
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("linked /rt/lib/librocdxg.so to the runtime's own copy"),
                "{lines:?}"
            );
        }

        #[test]
        fn linking_makes_the_unversioned_name_resolve_to_the_runtimes_copy() {
            let dir = tempfile::tempdir().expect("temp dir");
            let copy = dir.path().join("librocdxg.so.1");
            write(&copy, b"runtime copy");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            let dirs = vec![dir.path().to_path_buf()];
            assert_eq!(unlinked_runtime_copies(&dirs, name_state).len(), 1);

            let link = link_runtime_copy(&copy).expect("link").path;

            assert_eq!(link, dir.path().join("librocdxg.so"));
            assert_eq!(
                std::fs::read_link(&link).expect("read_link"),
                PathBuf::from("librocdxg.so.1"),
                "the target is relative, so the link survives the directory moving"
            );
            assert_eq!(
                std::fs::read(&link).expect("read through link"),
                b"runtime copy"
            );
            assert!(
                unlinked_runtime_copies(&dirs, name_state).is_empty(),
                "once linked, the runtime is no longer reported"
            );
            assert!(
                link_runtime_copy(&copy).is_err(),
                "a librocdxg.so that resolves is left alone"
            );
        }
    }
}
