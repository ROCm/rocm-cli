// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which binary a daemon spawn runs.
//!
//! The daemon runs in one of two hosts: the `rocmd` binary (`rocmd run`), or
//! embedded in `rocm` (`rocm daemon run`, which enters through
//! [`crate::run_from_args`]). The two binaries do not answer the same commands:
//! `supervise` belongs to `rocmd` alone, and the hidden engine commands
//! (`__engine-serve-http`, `__engine-stdio`) belong to `rocm` alone. A spawn
//! that re-executes the current executable is right only in the host that has
//! the command, so a spawn asks this module for the binary instead of reading
//! `current_exe` itself.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const ROCMD_BINARY: &str = "rocmd";
const ROCM_BINARY: &str = "rocm";

/// Set once by the embedded entry point. An explicit signal rather than a guess
/// from the executable's name, which a renamed or upper-cased binary defeats.
static EMBEDDED_IN_ROCM: AtomicBool = AtomicBool::new(false);

/// Record that this process is `rocm` running the daemon, for the rest of its life.
pub(crate) fn mark_embedded_in_rocm() {
    EMBEDDED_IN_ROCM.store(true, Ordering::Relaxed);
}

fn is_embedded_in_rocm() -> bool {
    EMBEDDED_IN_ROCM.load(Ordering::Relaxed)
}

/// The host the daemon is running in, with the one executable path that every
/// decision and every sibling lookup is made from, so they cannot disagree.
pub(crate) struct DaemonHost {
    current_exe: PathBuf,
    embedded_in_rocm: bool,
}

impl DaemonHost {
    /// The host this process is running in.
    ///
    /// A unit test never gets this one: it passes the code under test a host of
    /// its own, so that nothing outside the test's stand-in binaries can run,
    /// not even a `rocm` that happens to sit in the build directory.
    pub(crate) fn detect() -> Result<Self> {
        if cfg!(test) {
            anyhow::bail!(
                "a unit test reached `DaemonHost::detect`; pass it a host so that no binary \
                 outside its stand-ins can run"
            );
        }
        let current_exe = rocm_core::current_executable_path()
            .context("failed to resolve the current executable path")?;
        Ok(Self::new(current_exe, is_embedded_in_rocm()))
    }

    pub(crate) const fn new(current_exe: PathBuf, embedded_in_rocm: bool) -> Self {
        Self {
            current_exe,
            embedded_in_rocm,
        }
    }

    /// The binary that has the `supervise` command: this one when the daemon is
    /// `rocmd`, the `rocmd` installed beside `rocm` when it is embedded there.
    pub(crate) fn rocmd_binary(&self) -> Result<PathBuf> {
        self.rocmd_binary_via(rocm_core::sibling_binary_path_from)
    }

    /// [`Self::rocmd_binary`] with the sibling lookup passed in, given the
    /// executable and the binary's name.
    fn rocmd_binary_via(
        &self,
        sibling: impl FnOnce(&Path, &str) -> Result<PathBuf>,
    ) -> Result<PathBuf> {
        if !self.embedded_in_rocm {
            return Ok(self.current_exe.clone());
        }
        sibling(&self.current_exe, ROCMD_BINARY).context(
            "the daemon runs inside `rocm`, which has no `supervise` command, \
             and recovery restarts a service through the `rocmd` binary installed next to it",
        )
    }

    /// The binary that has the hidden engine commands: this one when the daemon
    /// is embedded in `rocm`, the `rocm` installed beside `rocmd` when it is not.
    pub(crate) fn rocm_binary(&self) -> Result<PathBuf> {
        self.rocm_binary_via(rocm_core::sibling_binary_path_from)
    }

    /// [`Self::rocm_binary`] with the sibling lookup passed in.
    fn rocm_binary_via(
        &self,
        sibling: impl FnOnce(&Path, &str) -> Result<PathBuf>,
    ) -> Result<PathBuf> {
        if self.embedded_in_rocm {
            return Ok(self.current_exe.clone());
        }
        sibling(&self.current_exe, ROCM_BINARY).context(
            "the `rocmd` binary has no engine commands, and a managed service's engine \
             runs through the `rocm` binary installed next to it",
        )
    }

    /// A host whose own executable is taken to have every command, so a test
    /// that reaches a spawn runs the test binary, as each one did before the
    /// binary was chosen here.
    #[cfg(test)]
    pub(crate) fn test_binary_standing_in_for_rocm() -> Self {
        Self::new(
            std::env::current_exe().expect("the test binary has a path"),
            true,
        )
    }

    /// The same for a test that restarts a service: the test binary is the
    /// `rocmd` that is re-executed.
    #[cfg(test)]
    pub(crate) fn test_binary_standing_in_for_rocmd() -> Self {
        Self::new(
            std::env::current_exe().expect("the test binary has a path"),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// The names a binary can carry on disk. The choice is made from the
    /// embedded signal, so none of them may change it.
    const ROCM_NAMES: [&str; 4] = ["/opt/rocm-cli/bin/rocm", "rocm.exe", "ROCM.EXE", "rocm-0.5"];
    const ROCMD_NAMES: [&str; 4] = [
        "/opt/rocm-cli/bin/rocmd",
        "rocmd.exe",
        "ROCMD.EXE",
        "/repo/target/debug/deps/rocmd-3f9a41c2d0e7b655",
    ];

    #[test]
    fn embedded_in_rocm_runs_supervise_on_the_sibling_rocmd_whatever_rocm_is_called() {
        for exe in ROCM_NAMES {
            let asked = RefCell::new(None);
            let chosen = DaemonHost::new(PathBuf::from(exe), true)
                .rocmd_binary_via(|from, name| {
                    *asked.borrow_mut() = Some((from.to_path_buf(), name.to_owned()));
                    Ok(PathBuf::from("/opt/rocm-cli/bin/rocmd"))
                })
                .unwrap();
            assert_eq!(chosen, PathBuf::from("/opt/rocm-cli/bin/rocmd"), "{exe}");
            // The lookup starts from the same executable the choice was made
            // from, and asks for `rocmd`.
            assert_eq!(
                asked.into_inner(),
                Some((PathBuf::from(exe), "rocmd".to_owned())),
                "{exe}"
            );
        }
    }

    #[test]
    fn the_rocmd_daemon_keeps_its_own_executable_whatever_it_is_called() {
        for exe in ROCMD_NAMES.into_iter().chain(["rocm"]) {
            let chosen = DaemonHost::new(PathBuf::from(exe), false)
                .rocmd_binary_via(|_, _| {
                    panic!("{exe}: not embedded in rocm, so it has `supervise` itself")
                })
                .unwrap();
            assert_eq!(chosen, PathBuf::from(exe));
        }
    }

    #[test]
    fn a_missing_rocmd_beside_rocm_is_named() {
        let error = DaemonHost::new(PathBuf::from("/opt/rocm-cli/bin/rocm"), true)
            .rocmd_binary_via(|_, _| {
                anyhow::bail!("unable to locate sibling binary rocmd; checked /opt/rocm-cli/bin")
            })
            .unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains("no `supervise` command"), "{text}");
        assert!(
            text.contains("unable to locate sibling binary rocmd"),
            "{text}"
        );
    }

    /// The hidden engine commands are `rocm`'s: a daemon that is `rocmd`, under
    /// any name, runs them on the `rocm` beside it, and says what it looked for.
    #[test]
    fn the_rocmd_daemon_runs_engine_commands_on_the_sibling_rocm() {
        for exe in ROCMD_NAMES.into_iter().chain(["rocm"]) {
            let asked = RefCell::new(None);
            let chosen = DaemonHost::new(PathBuf::from(exe), false)
                .rocm_binary_via(|from, name| {
                    *asked.borrow_mut() = Some((from.to_path_buf(), name.to_owned()));
                    Ok(PathBuf::from("/opt/rocm-cli/bin/rocm"))
                })
                .unwrap();
            assert_eq!(chosen, PathBuf::from("/opt/rocm-cli/bin/rocm"), "{exe}");
            assert_eq!(
                asked.into_inner(),
                Some((PathBuf::from(exe), "rocm".to_owned())),
                "{exe}"
            );
        }
    }

    #[test]
    fn embedded_in_rocm_keeps_its_own_executable_for_engine_commands() {
        for exe in ROCM_NAMES {
            let chosen = DaemonHost::new(PathBuf::from(exe), true)
                .rocm_binary_via(|_, _| {
                    panic!("{exe}: embedded in rocm, so it has the engine commands itself")
                })
                .unwrap();
            assert_eq!(chosen, PathBuf::from(exe));
        }
    }

    #[test]
    fn a_missing_rocm_beside_rocmd_is_named() {
        let error = DaemonHost::new(PathBuf::from("/opt/rocm-cli/bin/rocmd"), false)
            .rocm_binary_via(|_, _| {
                anyhow::bail!("unable to locate sibling binary rocm; checked /opt/rocm-cli/bin")
            })
            .unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains("no engine commands"), "{text}");
        assert!(
            text.contains("unable to locate sibling binary rocm"),
            "{text}"
        );
    }

    /// The real lookup, from the host's own executable: the `rocmd` next to a
    /// `rocm` that is not the test binary.
    #[test]
    fn the_real_sibling_lookup_starts_from_the_hosts_executable() {
        let dir = crate::test_support::unique_test_root("daemon-host-sibling-lookup");
        let rocm = dir.join(rocm_core::platform_binary_name("rocm"));
        let rocmd = dir.join(rocm_core::platform_binary_name("rocmd"));
        std::fs::write(&rocmd, b"").expect("write rocmd");

        let chosen = DaemonHost::new(rocm, true).rocmd_binary();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(chosen.unwrap(), rocmd);
    }

    /// A process that did not enter through `run_from_args` is not embedded.
    #[test]
    fn a_plain_process_is_not_embedded_in_rocm() {
        assert!(!is_embedded_in_rocm());
    }

    /// The real host is not reachable from a unit test, whatever sits beside it.
    #[test]
    fn a_unit_test_cannot_reach_the_real_host() {
        let error = DaemonHost::detect()
            .err()
            .expect("detect is refused in tests");
        assert!(format!("{error:#}").contains("pass it a host"), "{error:#}");
    }

    /// `rocm daemon run` enters through `run_from_args`, and that is what marks
    /// the process. The flag is process-wide, so it is checked in a child run of
    /// this test binary instead of being set here, where it would change the
    /// answer for every other test.
    #[test]
    fn run_from_args_marks_the_process_as_embedded_in_rocm() {
        const CHILD: &str = "ROCMD_TEST_RUN_FROM_ARGS_CHILD";
        let test_name = "daemon_host::tests::run_from_args_marks_the_process_as_embedded_in_rocm";
        if std::env::var_os(CHILD).is_some() {
            // `--help` stops clap before anything runs; the marking comes first.
            let _ = crate::run_from_args(vec!["rocmd".into(), "--help".into()]);
            assert!(is_embedded_in_rocm());
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .expect("run the child test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains(&format!("test {test_name} ... ok")),
            "child run failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
