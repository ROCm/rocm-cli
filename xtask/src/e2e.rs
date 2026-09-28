// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Run the cucumber-rs E2E suite.
//!
//! Builds the release `rocm` and `rocmd` binaries, points the test harness at
//! `rocm` via `ROCM_CLI_BINARY`, then runs `cargo test -p e2e-cucumber --test e2e`,
//! forwarding any extra arguments to the cucumber CLI (e.g. `-t`, `-n`,
//! `--fail-fast`). Used by both CI and local dev so the build+run recipe lives
//! in one cross-platform place instead of a bash wrapper.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths::{binary_name, target_dir, workspace_root};

#[derive(Debug, PartialEq, Eq)]
struct E2eBinaries {
    rocm: PathBuf,
    rocmd: Option<PathBuf>,
    build_release: bool,
}

fn resolve_caller_path(invocation_dir: &Path, path: OsString) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        invocation_dir.join(path)
    }
}

fn e2e_binaries_from(
    release_dir: &Path,
    invocation_dir: &Path,
    rocm_override: Option<OsString>,
    rocmd_override: Option<OsString>,
) -> E2eBinaries {
    if let Some(rocm) = rocm_override {
        E2eBinaries {
            rocm: resolve_caller_path(invocation_dir, rocm),
            rocmd: rocmd_override.map(|path| resolve_caller_path(invocation_dir, path)),
            build_release: false,
        }
    } else {
        E2eBinaries {
            rocm: release_dir.join(binary_name("rocm")),
            rocmd: Some(release_dir.join(binary_name("rocmd"))),
            build_release: true,
        }
    }
}

fn configure_harness_env(command: &mut Command, binaries: &E2eBinaries) {
    command.env("ROCM_CLI_BINARY", &binaries.rocm);
    if let Some(rocmd) = &binaries.rocmd {
        command.env("ROCM_CLI_ROCMD_BINARY", rocmd);
    } else {
        command.env_remove("ROCM_CLI_ROCMD_BINARY");
    }
}

/// Build the release binaries and run the E2E suite, forwarding `args` to the
/// cucumber CLI. If `ROCM_CLI_BINARY` is already set in the environment, the
/// build step is skipped and that binary is used as-is. A caller selecting a
/// rocmd-backed scenario must also set `ROCM_CLI_ROCMD_BINARY` explicitly.
///
/// The harness resolves each scenario to pass / xfail / skip per host (see the
/// e2e-cucumber `expectation` module), so there is no tier flag: one invocation
/// runs everything applicable on this platform and self-reports the outcome.
pub fn run(args: &[String]) -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let root = workspace_root()?;
    let invocation_dir = std::env::current_dir().context("failed to read the current directory")?;
    let binaries = e2e_binaries_from(
        &target_dir(&root).join("release"),
        &invocation_dir,
        std::env::var_os("ROCM_CLI_BINARY"),
        std::env::var_os("ROCM_CLI_ROCMD_BINARY"),
    );

    if binaries.build_release {
        let status = Command::new(&cargo)
            .args(["build", "--release", "-p", "rocm", "-p", "rocmd"])
            .current_dir(&root)
            .status()
            .context("failed to run `cargo build --release -p rocm -p rocmd`")?;
        if !status.success() {
            bail!("building the rocm and rocmd binaries failed");
        }
    }

    let mut cmd = Command::new(&cargo);
    cmd.args(["test", "-p", "e2e-cucumber", "--test", "e2e"])
        .current_dir(&root);
    configure_harness_env(&mut cmd, &binaries);
    // Hand the lifecycle E2E steps the path to this already-built `xtask`
    // executable (the running process). Those steps drive `xtask package` and
    // `xtask keygen` as subprocesses; they must run this prebuilt binary
    // directly instead of re-entering cargo (`cargo xtask …`). On Windows cargo
    // cannot rebuild `xtask.exe` while it is still running as the harness — it
    // fails to replace the locked file with "Access is denied", which broke
    // every lifecycle scenario. Executing the existing binary needs no rebuild
    // and takes no such lock.
    if let Ok(xtask_bin) = std::env::current_exe() {
        cmd.env("ROCM_XTASK_BINARY", xtask_bin);
    }
    if !args.is_empty() {
        cmd.arg("--").args(args);
    }

    let status = cmd.status().context("failed to run the E2E test binary")?;
    if !status.success() {
        bail!("E2E suite failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsStr, OsString};
    use std::path::Path;

    fn command_env(command: &Command, key: &str) -> Option<OsString> {
        command
            .get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(OsStr::to_os_string))
    }

    #[test]
    fn release_build_resolves_and_exports_both_cli_binaries() {
        let binaries = e2e_binaries_from(
            Path::new("/ws/out/release"),
            Path::new("/invoked"),
            None,
            None,
        );
        assert!(binaries.build_release);
        assert_eq!(
            binaries.rocm,
            PathBuf::from("/ws/out/release").join(crate::paths::binary_name("rocm"))
        );
        assert_eq!(
            binaries.rocmd,
            Some(PathBuf::from("/ws/out/release").join(crate::paths::binary_name("rocmd")))
        );

        let mut command = Command::new("cargo");
        configure_harness_env(&mut command, &binaries);
        assert_eq!(
            command_env(&command, "ROCM_CLI_BINARY"),
            Some(binaries.rocm.into_os_string())
        );
        assert_eq!(
            command_env(&command, "ROCM_CLI_ROCMD_BINARY"),
            binaries.rocmd.map(PathBuf::into_os_string)
        );
    }

    #[test]
    fn prebuilt_cli_paths_resolve_from_the_invocation_directory_and_propagate() {
        let binaries = e2e_binaries_from(
            Path::new("/ws/out/release"),
            Path::new("/invoked"),
            Some(OsString::from("bin/rocm")),
            Some(OsString::from("daemons/rocmd")),
        );
        assert!(!binaries.build_release);
        assert_eq!(binaries.rocm, PathBuf::from("/invoked/bin/rocm"));
        assert_eq!(
            binaries.rocmd,
            Some(PathBuf::from("/invoked/daemons/rocmd"))
        );

        let mut command = Command::new("cargo");
        configure_harness_env(&mut command, &binaries);
        assert_eq!(
            command_env(&command, "ROCM_CLI_ROCMD_BINARY"),
            binaries.rocmd.map(PathBuf::into_os_string)
        );
    }

    #[test]
    fn prebuilt_rocm_without_explicit_rocmd_does_not_invent_or_export_one() {
        let binaries = e2e_binaries_from(
            Path::new("/ws/out/release"),
            Path::new("/invoked"),
            Some(OsString::from("/prebuilt/rocm")),
            None,
        );
        assert_eq!(binaries.rocmd, None);

        let mut command = Command::new("cargo");
        configure_harness_env(&mut command, &binaries);
        assert_eq!(command_env(&command, "ROCM_CLI_ROCMD_BINARY"), None);
    }
}
