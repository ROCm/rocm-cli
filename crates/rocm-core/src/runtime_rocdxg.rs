// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which `librocdxg` a managed wheel runtime's HSA runtime actually loads on WSL2.
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

/// The runtime's own `librocdxg.so.1` for each directory in `dirs` that holds
/// one next to a `libhsa-runtime64` but has no `librocdxg.so` beside it.
///
/// In such a directory the runtime's lookup for `librocdxg.so` does not find the
/// runtime's own copy, so it loads the host's. A directory with the library but
/// no runtime beside it is left out: nothing there opens it by name. One that
/// already has `librocdxg.so` (a file or a link, even a broken one that the
/// loader will refuse) is left out as well; there is nothing for this module to
/// add.
///
/// Takes a presence check rather than reading the machine, so the layouts can
/// be tested without one. The real check is [`path_present`], which does not
/// follow links, so a dangling `librocdxg.so` counts as present.
#[must_use]
pub fn unlinked_runtime_copies(dirs: &[PathBuf], exists: impl Fn(&Path) -> bool) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        let copy = dir.join(SHIPPED_NAME);
        let has_runtime = ["libhsa-runtime64.so.1", "libhsa-runtime64.so"]
            .iter()
            .any(|name| exists(&dir.join(name)));
        // A dangling `librocdxg.so` is the loader's to report, and `ln -s` over
        // it would fail with "File exists", so the check must not follow links.
        if exists(&copy) && !exists(&dir.join(OPENED_NAME)) && has_runtime && !found.contains(&copy)
        {
            found.push(copy);
        }
    }
    found
}

/// Whether something, a dangling link included, is at `path`.
fn path_present(path: &Path) -> bool {
    path.symlink_metadata().is_ok()
}

/// The `librocdxg.so.1` copies of every managed runtime whose own HSA runtime
/// would not load them, for `rocm examine`.
///
/// Empty off WSL2 (`is_wsl` false): the advice is about the WSL2 loader path and
/// would mislead on a native host.
#[must_use]
pub fn unlinked_managed_runtime_copies(is_wsl: bool, paths: &crate::AppPaths) -> Vec<PathBuf> {
    if !is_wsl {
        return Vec::new();
    }
    unlinked_runtime_copies(
        &crate::managed_runtime::managed_runtime_library_dirs(paths),
        path_present,
    )
}

/// The whole post-install step: on WSL2, link every runtime copy in `dirs` that
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
/// is fixed (the same link call failed, so the command alone may fail the same way). Nothing is
/// touched off WSL2, where the HSA runtime does not open `librocdxg`.
pub fn link_installed_runtime_copies_when(is_wsl: bool, dirs: &[PathBuf]) -> Vec<String> {
    if !is_wsl {
        return Vec::new();
    }
    link_copies_with(dirs, path_present, link_runtime_copy)
}

fn link_copies_with(
    dirs: &[PathBuf],
    exists: impl Fn(&Path) -> bool,
    link: impl Fn(&Path) -> io::Result<PathBuf>,
) -> Vec<String> {
    unlinked_runtime_copies(dirs, exists)
        .iter()
        .map(|copy| match link(copy) {
            Ok(link) => format!(
                "librocdxg: linked {} to the runtime's own copy; the runtime searches its own directory first",
                link.display()
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
    format!("ln -s {SHIPPED_NAME} '{target}'")
}

/// Give `copy`'s directory a `librocdxg.so` that points at `copy`, and return
/// the link's path.
///
/// The link is relative (`librocdxg.so -> librocdxg.so.1`), so it survives the
/// runtime directory being moved or restored from a cache. An existing
/// `librocdxg.so` is left alone and reported as an error rather than replaced.
///
/// # Errors
/// Returns the underlying error when the link cannot be created, or
/// [`io::ErrorKind::Unsupported`] off Unix (the WSL2 path is Linux only).
pub fn link_runtime_copy(copy: &Path) -> io::Result<PathBuf> {
    let dir = copy.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "librocdxg copy has no directory",
        )
    })?;
    let link = dir.join(OPENED_NAME);
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(SHIPPED_NAME, &link)?;
        Ok(link)
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

#[cfg(test)]
mod tests {
    // The tests pin Unix paths and the shell command, and the code under test is
    // WSL2-only, so they run on Unix only. The module is a plain `cfg(test)` one
    // so the registry-writer guard in `apps/rocm/src/runtime_registry.rs` cuts it
    // off as test code.
    #[cfg(unix)]
    mod unix {
        use super::super::*;
        use std::collections::HashSet;

        fn files(paths: &[&str]) -> impl Fn(&Path) -> bool {
            let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
            move |path| set.contains(path)
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
                "ln -s librocdxg.so.1 '/rt/lib/librocdxg.so'"
            );
        }

        #[test]
        fn link_command_escapes_a_single_quote_in_the_path() {
            let copy = PathBuf::from("/rt/it's/librocdxg.so.1");
            assert_eq!(
                link_command(&copy),
                "ln -s librocdxg.so.1 '/rt/it'\\''s/librocdxg.so'"
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

            let link = link_runtime_copy(&by_install.path().join("librocdxg.so.1")).expect("link");
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
                lines[0].contains("ln -s librocdxg.so.1 '/rt/lib/librocdxg.so'"),
                "{lines:?}"
            );
        }

        #[test]
        fn a_dangling_unversioned_link_is_left_alone_by_detection_and_linking() {
            let dir = tempfile::tempdir().expect("temp dir");
            let copy = dir.path().join("librocdxg.so.1");
            write(&copy, b"x");
            write(&dir.path().join("libhsa-runtime64.so.1"), b"");
            std::os::unix::fs::symlink("gone.so", dir.path().join("librocdxg.so")).expect("link");
            let dirs = vec![dir.path().to_path_buf()];

            assert!(
                unlinked_runtime_copies(&dirs, path_present).is_empty(),
                "a dangling link is present, and `ln -s` over it would fail"
            );
            assert!(link_runtime_copy(&copy).is_err());
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
            // The failure does not rely on permissions (root ignores them): the
            // presence check is told `librocdxg.so` is absent while a file really is
            // there, so the link attempt fails with "File exists" for any user.
            let dir = runtime_dir_shipping_only_the_versioned_copy();
            write(&dir.path().join("librocdxg.so"), b"someone else's file");
            let absent_unversioned = |path: &Path| {
                path.file_name().is_none_or(|name| name != "librocdxg.so") && path_present(path)
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
                lines[0].contains("once the cause above is fixed, run: ln -s librocdxg.so.1"),
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
            let lines =
                link_copies_with(
                    &dirs,
                    exists,
                    |copy| Ok(copy.with_file_name("librocdxg.so")),
                );
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
            assert_eq!(unlinked_runtime_copies(&dirs, Path::exists).len(), 1);

            let link = link_runtime_copy(&copy).expect("link");

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
                unlinked_runtime_copies(&dirs, Path::exists).is_empty(),
                "once linked, the runtime is no longer reported"
            );
            assert!(
                link_runtime_copy(&copy).is_err(),
                "an existing librocdxg.so is never replaced"
            );
        }
    }
}
