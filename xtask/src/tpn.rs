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
/// cargo-about version the committed notices are reproducible with. Different
/// versions format the output differently, so regenerating with anything else
/// produces a file that fails the byte-for-byte `--check` gate in CI. Kept in
/// sync with the workflows by `pinned_version_matches_workflows` below.
const ABOUT_VERSION: &str = "0.9.1";

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

/// Usability of the local cargo-about install.
#[derive(Debug, PartialEq, Eq)]
enum Generator {
    /// Installed at [`ABOUT_VERSION`]; its output will match CI byte-for-byte.
    Ready,
    /// Not installed at all.
    Absent,
    /// Installed, but at a version whose output would not match CI.
    WrongVersion,
}

/// Pure policy: classify the generator from the detected version (`None` when
/// the subcommand is absent or unrecognisable).
fn classify_generator(detected: Option<&str>) -> Generator {
    match detected {
        Some(ABOUT_VERSION) => Generator::Ready,
        Some(_) => Generator::WrongVersion,
        None => Generator::Absent,
    }
}

/// Pure parser for `cargo about --version` output, which prints
/// `cargo-about <semver>`. `None` if the output is not in that form, so an
/// unexpected format is treated as "unusable" rather than silently accepted.
fn parse_about_version(stdout: &str) -> Option<&str> {
    let mut parts = stdout.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some("cargo-about"), Some(version)) => Some(version),
        _ => None,
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

/// Probe the installed cargo-about version, so a missing or mismatched
/// generator produces an actionable message rather than cargo's opaque "no such
/// subcommand" error or notices that silently fail CI.
fn detect_about_version(cargo: &std::ffi::OsStr) -> Option<String> {
    let output = Command::new(cargo)
        .args(["about", "--version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_about_version(&stdout).map(str::to_owned)
}

/// The `cargo install` line that produces the exact generator CI uses.
fn install_hint() -> String {
    format!("cargo install cargo-about@{ABOUT_VERSION} --locked --features cli")
}

/// Run `cargo about generate` against the committed template/config and return
/// the rendered notices.
fn generate(root: &Path, cargo: &std::ffi::OsStr) -> Result<String> {
    let output = Command::new(cargo)
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
///
/// With `if_available`, an unusable generator is a clean no-op instead of an
/// error, so the local git hook does not block commits for contributors who
/// have not installed cargo-about (or have a different version). CI never
/// passes it, and clap forbids combining it with `--check`, so the staleness
/// gate can never be silently skipped.
pub fn run(check: bool, if_available: bool) -> Result<()> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let generator = classify_generator(detect_about_version(&cargo).as_deref());

    match (&generator, if_available) {
        (Generator::Ready, _) => {}
        (Generator::Absent, true) => {
            eprintln!(
                "tpn: cargo-about not installed; skipping {TPN_FILE} regeneration. \
                 Install it with: {}",
                install_hint()
            );
            return Ok(());
        }
        (Generator::WrongVersion, true) => {
            eprintln!(
                "tpn: cargo-about is not version {ABOUT_VERSION}; skipping {TPN_FILE} \
                 regeneration, because other versions format the notices differently and \
                 would fail CI. Pin it with: {}",
                install_hint()
            );
            return Ok(());
        }
        (Generator::Absent, false) => bail!(
            "cargo-about is required to generate {TPN_FILE}.\n\
             Install it with: {}",
            install_hint()
        ),
        (Generator::WrongVersion, false) => bail!(
            "cargo-about must be version {ABOUT_VERSION} to generate {TPN_FILE} \
             reproducibly; other versions format the notices differently.\n\
             Pin it with: {}",
            install_hint()
        ),
    }

    let root = workspace_root()?;
    let generated = generate(&root, &cargo)?;
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

    #[test]
    fn generator_is_ready_only_at_the_pinned_version() {
        assert_eq!(classify_generator(Some(ABOUT_VERSION)), Generator::Ready);
        assert_eq!(classify_generator(Some("0.9.0")), Generator::WrongVersion);
        assert_eq!(classify_generator(Some("0.10.0")), Generator::WrongVersion);
        assert_eq!(classify_generator(None), Generator::Absent);
    }

    #[test]
    fn version_parser_accepts_cargo_about_output_only() {
        assert_eq!(parse_about_version("cargo-about 0.9.1\n"), Some("0.9.1"));
        // Unexpected shapes must read as unusable, never as a usable version.
        assert_eq!(parse_about_version("cargo-about"), None);
        assert_eq!(parse_about_version("cargo-deny 0.9.1"), None);
        assert_eq!(parse_about_version(""), None);
    }

    /// The pinned constant and the version CI installs must never drift: if they
    /// do, the hook regenerates notices that fail the byte-for-byte gate.
    #[test]
    fn pinned_version_matches_workflows() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf();
        let expected = format!("cargo-about@{ABOUT_VERSION}");
        for workflow in ["ci.yml", "dependabot-manifests.yml"] {
            let path = root.join(".github/workflows").join(workflow);
            let yaml = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            assert!(
                yaml.contains(&expected),
                "{workflow} must install {expected} to match ABOUT_VERSION in tpn.rs"
            );
        }
    }
}
