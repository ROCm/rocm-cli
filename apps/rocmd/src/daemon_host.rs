// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which binary a daemon spawn runs.
//!
//! The daemon runs in one of two hosts: the `rocmd` binary (`rocmd run`), or
//! embedded in `rocm` (`rocm daemon`, which enters through
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
    #[cfg(test)]
    {
        if let Some(flag) = EMBEDDED_OVERRIDE.with(std::cell::Cell::get) {
            return flag;
        }
    }
    EMBEDDED_IN_ROCM.load(Ordering::Relaxed)
}

// Unit tests only: the answer `is_embedded_in_rocm` gives on this thread. The
// real flag is process-wide, so a test that set it would change the answer for
// every other test; a thread-local lets a test see both answers and no other
// test see either.
#[cfg(test)]
thread_local! {
    static EMBEDDED_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The host the daemon is running in, with the one executable path that every
/// decision and every sibling lookup is made from, so they cannot disagree.
pub(crate) struct DaemonHost {
    current_exe: PathBuf,
    embedded_in_rocm: bool,
    /// Unit tests only: the test binary has every command, so a spawn runs it
    /// and no sibling is ever looked for.
    #[cfg(test)]
    test_binary_has_every_command: bool,
    /// Unit tests only: a host that stands in for one binary looks for no
    /// sibling, so asking it for the other is an error and not a lookup that
    /// could find, and run, a build of the other binary beside the test binary.
    #[cfg(test)]
    test_looks_for_no_sibling: bool,
}

impl DaemonHost {
    /// The host this process is running in.
    ///
    /// Under test, the process is the test binary, which stands in for both
    /// `rocm` and `rocmd` and looks for neither: code that reaches this runs
    /// the test binary and nothing else, not even a `rocm` that happens to sit
    /// in the build directory. A test that needs another binary passes the code
    /// it exercises a host of its own.
    pub(crate) fn detect() -> Result<Self> {
        #[cfg(test)]
        let current_exe = std::env::current_exe().context("the test binary has a path")?;
        #[cfg(not(test))]
        let current_exe = rocm_core::current_executable_path()
            .context("failed to resolve the current executable path")?;
        // The same composition in both builds, so a test reaches the line that
        // takes the host's embedded flag from the process.
        let host = Self::new(current_exe, is_embedded_in_rocm());
        #[cfg(test)]
        let host = Self {
            test_binary_has_every_command: true,
            ..host
        };
        Ok(host)
    }

    pub(crate) const fn new(current_exe: PathBuf, embedded_in_rocm: bool) -> Self {
        Self {
            current_exe,
            embedded_in_rocm,
            #[cfg(test)]
            test_binary_has_every_command: false,
            #[cfg(test)]
            test_looks_for_no_sibling: false,
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
        #[cfg(test)]
        {
            if self.test_binary_has_every_command {
                return Ok(self.current_exe.clone());
            }
        }
        if !self.embedded_in_rocm {
            return Ok(self.current_exe.clone());
        }
        #[cfg(test)]
        {
            self.refuse_sibling_lookup(ROCMD_BINARY)?;
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
        #[cfg(test)]
        {
            if self.test_binary_has_every_command {
                return Ok(self.current_exe.clone());
            }
        }
        if self.embedded_in_rocm {
            return Ok(self.current_exe.clone());
        }
        #[cfg(test)]
        {
            self.refuse_sibling_lookup(ROCM_BINARY)?;
        }
        sibling(&self.current_exe, ROCM_BINARY).context(
            "the `rocmd` binary has no engine commands, and a managed service's engine \
             runs through the `rocm` binary installed next to it",
        )
    }

    /// The error a test stand-in answers with when asked for the binary it does
    /// not stand in for.
    #[cfg(test)]
    fn refuse_sibling_lookup(&self, binary: &str) -> Result<()> {
        if self.test_looks_for_no_sibling {
            anyhow::bail!(
                "this test host stands in for one binary and looks for no sibling, \
                 so it cannot name `{binary}`"
            );
        }
        Ok(())
    }

    /// A host whose own executable is taken to have every command, so a test
    /// that reaches a spawn runs the test binary, as each one did before the
    /// binary was chosen here.
    #[cfg(test)]
    pub(crate) fn test_binary_standing_in_for_rocm() -> Self {
        let mut host = Self::new(
            std::env::current_exe().expect("the test binary has a path"),
            true,
        );
        host.test_looks_for_no_sibling = true;
        host
    }

    /// The same for a test that restarts a service: the test binary is the
    /// `rocmd` that is re-executed.
    #[cfg(test)]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn test_binary_standing_in_for_rocmd() -> Self {
        let mut host = Self::new(
            std::env::current_exe().expect("the test binary has a path"),
            false,
        );
        host.test_looks_for_no_sibling = true;
        host
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

    /// A stand-in host names only the binary it stands in for. Asked for the
    /// other it fails, and it never starts the real sibling lookup, which would
    /// find the other binary in the build directory and let a test run it.
    #[test]
    fn a_stand_in_host_looks_for_no_sibling() {
        let this = std::env::current_exe().unwrap();
        let rocm = DaemonHost::test_binary_standing_in_for_rocm();
        assert_eq!(rocm.rocm_binary().unwrap(), this);
        let error = rocm
            .rocmd_binary_via(|_, _| panic!("a stand-in host must not look for a sibling"))
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("looks for no sibling"),
            "{error:#}"
        );
        assert!(rocm.rocmd_binary().is_err());

        let rocmd = DaemonHost::test_binary_standing_in_for_rocmd();
        assert_eq!(rocmd.rocmd_binary().unwrap(), this);
        let error = rocmd
            .rocm_binary_via(|_, _| panic!("a stand-in host must not look for a sibling"))
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("looks for no sibling"),
            "{error:#}"
        );
        assert!(rocmd.rocm_binary().is_err());
    }

    /// `detect` takes the embedded flag from the process, and not its negation
    /// or a constant: with the flag set on this thread the host it returns is
    /// the embedded one, and with it clear it is not. The two answers are the
    /// whole premise of the fix, since they decide which binary each spawn runs.
    /// What a unit test still cannot reach is the production arm that resolves
    /// the executable (`rocm_core::current_executable_path`).
    #[test]
    fn detect_takes_the_embedded_flag_from_the_process() {
        let this = std::env::current_exe().unwrap();
        for flag in [false, true] {
            EMBEDDED_OVERRIDE.with(|cell| cell.set(Some(flag)));
            let host = DaemonHost::detect();
            EMBEDDED_OVERRIDE.with(|cell| cell.set(None));
            let host = host.unwrap();
            assert_eq!(host.embedded_in_rocm, flag, "embedded flag {flag}");
            // The stand-in still answers with the test binary for both
            // commands whatever the flag says.
            assert_eq!(host.rocmd_binary().unwrap(), this, "embedded flag {flag}");
            assert_eq!(host.rocm_binary().unwrap(), this, "embedded flag {flag}");
        }
        assert!(!is_embedded_in_rocm(), "the override is cleared again");
    }

    /// A process that did not enter through `run_from_args` is not embedded.
    #[test]
    fn a_plain_process_is_not_embedded_in_rocm() {
        assert!(!is_embedded_in_rocm());
    }

    /// The real host is not reachable from a unit test: under test, `detect` is
    /// the test binary for both commands, and nothing beside it is looked for.
    #[test]
    fn under_test_detect_is_the_test_binary_and_looks_for_no_sibling() {
        let host = DaemonHost::detect().unwrap();
        let this = std::env::current_exe().unwrap();
        assert_eq!(host.rocmd_binary().unwrap(), this);
        assert_eq!(host.rocm_binary().unwrap(), this);
    }

    /// `rocm daemon` enters through `run_from_args`, and that is what marks
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
