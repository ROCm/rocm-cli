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

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths::{target_dir, workspace_root};

/// Build the release binaries and run the E2E suite, forwarding `args` to the
/// cucumber CLI. If `ROCM_CLI_BINARY` is already set in the environment, the
/// build step is skipped and that binary is used as-is; callers opting into
/// lifecycle scenarios must then provide a sibling `rocmd` for packaging.
///
/// The harness resolves each scenario to pass / xfail / skip per host (see the
/// e2e-cucumber `expectation` module), so there is no tier flag: one invocation
/// runs everything applicable on this platform and self-reports the outcome.
pub fn run(args: &[String]) -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let root = workspace_root()?;

    let binary = if let Some(path) = std::env::var_os("ROCM_CLI_BINARY") {
        // The cucumber steps spawn the binary from a working directory that may
        // differ from where `cargo xtask` was invoked, so a relative path would
        // fail to resolve. Make any caller-supplied path absolute.
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            std::env::current_dir()
                .context("failed to read the current directory")?
                .join(path)
        }
    } else {
        let status = Command::new(&cargo)
            .args(["build", "--release", "-p", "rocm", "-p", "rocmd"])
            .current_dir(&root)
            .status()
            .context("failed to run `cargo build --release -p rocm`")?;
        if !status.success() {
            bail!("building the rocm and rocmd binaries failed");
        }
        // Honor CARGO_TARGET_DIR — the built binary is under the active target
        // dir, which is not always `<root>/target` (e.g. a cross-platform
        // container build points it elsewhere). Using a hardcoded `target/`
        // here can pick up a stale binary built for a different OS/arch and
        // fail with an exec-format error.
        target_dir(&root).join("release/rocm")
    };

    let mut cmd = Command::new(&cargo);
    cmd.args(["test", "-p", "e2e-cucumber", "--test", "e2e"])
        .current_dir(&root)
        .env("ROCM_CLI_BINARY", &binary);
    if !args.is_empty() {
        cmd.arg("--").args(args);
    }

    let status = cmd.status().context("failed to run the E2E test binary")?;
    if !status.success() {
        bail!("E2E suite failed");
    }
    Ok(())
}
