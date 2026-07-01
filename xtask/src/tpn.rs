// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Generate (or verify) THIRD_PARTY_NOTICES.txt — the attribution file listing
//! every third-party crate distributed with rocm-cli and the full text of its
//! license.
//!
//! The notices are produced by [`cargo-about`] from the committed `about.toml`
//! config and `about.hbs` template; this module is a thin, repo-aware wrapper
//! that fixes the output path and adds a `--check` staleness gate (mirroring
//! `cargo xtask manifest --check`). The pure decision logic ([`decide`]) is
//! separated from the cargo-about/filesystem I/O so it can be unit-tested.
//!
//! [`cargo-about`]: https://github.com/EmbarkStudios/cargo-about

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Generated notices file, written at the workspace root.
const TPN_FILE: &str = "THIRD_PARTY_NOTICES.txt";
/// cargo-about template, committed at the workspace root.
const ABOUT_TEMPLATE: &str = "about.hbs";
/// cargo-about config, committed at the workspace root.
const ABOUT_CONFIG: &str = "about.toml";

/// What to do with freshly generated notices relative to what is on disk.
#[derive(Debug, PartialEq, Eq)]
enum Decision {
    /// On-disk file already matches the generated output; nothing to do.
    UpToDate,
    /// `--check` mode and the on-disk file is missing or differs.
    Stale,
    /// Write mode and the on-disk file is missing or differs.
    Write,
}

/// Pure policy: given whether we are in check mode and the current on-disk
/// contents (`None` if the file is absent), decide what to do with the freshly
/// generated notices.
fn decide(check: bool, current: Option<&str>, generated: &str) -> Decision {
    if current == Some(generated) {
        return Decision::UpToDate;
    }
    if check {
        Decision::Stale
    } else {
        Decision::Write
    }
}

/// Locate the workspace root (the directory containing the virtual-manifest
/// `Cargo.toml`) so the command works regardless of the directory it is invoked
/// from.
fn workspace_root() -> Result<PathBuf> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()
        .context("failed to run `cargo locate-project`")?;
    if !output.status.success() {
        bail!(
            "`cargo locate-project` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let manifest = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let manifest = Path::new(&manifest);
    manifest.parent().map(Path::to_path_buf).with_context(|| {
        format!(
            "could not derive workspace root from {}",
            manifest.display()
        )
    })
}

/// Fail early with an actionable message if cargo-about is not installed, rather
/// than surfacing cargo's opaque "no such subcommand" error.
fn ensure_cargo_about(cargo: &std::ffi::OsStr) -> Result<()> {
    let probe = Command::new(cargo).args(["about", "--version"]).output();
    let ok = matches!(probe, Ok(output) if output.status.success());
    if !ok {
        bail!(
            "cargo-about is required to generate {TPN_FILE}.\n\
             Install it with: cargo install cargo-about --locked --features cli"
        );
    }
    Ok(())
}

/// Run `cargo about generate` against the committed template/config and return
/// the rendered notices.
fn generate(root: &Path) -> Result<String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    ensure_cargo_about(&cargo)?;

    let output = Command::new(&cargo)
        .current_dir(root)
        .args([
            "about",
            "generate",
            "--workspace",
            "--config",
            ABOUT_CONFIG,
            ABOUT_TEMPLATE,
        ])
        .output()
        .context("failed to run `cargo about generate`")?;
    if !output.status.success() {
        bail!(
            "`cargo about generate` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("`cargo about generate` produced non-UTF-8 output")
}

/// Entry point for the `tpn` subcommand.
pub fn run(check: bool) -> Result<()> {
    let root = workspace_root()?;
    let generated = generate(&root)?;
    let path = root.join(TPN_FILE);
    let current = fs::read_to_string(&path).ok();

    match decide(check, current.as_deref(), &generated) {
        Decision::UpToDate => Ok(()),
        Decision::Stale => {
            bail!("{TPN_FILE} is out of date; run `cargo xtask tpn` to regenerate it")
        }
        Decision::Write => {
            fs::write(&path, &generated)
                .with_context(|| format!("failed to write {}", path.display()))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn up_to_date_when_contents_match() {
        assert_eq!(decide(false, Some("same"), "same"), Decision::UpToDate);
        assert_eq!(decide(true, Some("same"), "same"), Decision::UpToDate);
    }

    #[test]
    fn check_mode_reports_stale_on_diff_or_missing() {
        assert_eq!(decide(true, Some("old"), "new"), Decision::Stale);
        assert_eq!(decide(true, None, "new"), Decision::Stale);
    }

    #[test]
    fn write_mode_writes_on_diff_or_missing() {
        assert_eq!(decide(false, Some("old"), "new"), Decision::Write);
        assert_eq!(decide(false, None, "new"), Decision::Write);
    }
}
