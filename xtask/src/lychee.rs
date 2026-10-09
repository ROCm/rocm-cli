// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Check that every local markdown link resolves, with [lychee] and the
//! committed `lychee.toml` — the local counterpart of the `docs-links` CI job,
//! run by the `lychee` prek hook.
//!
//! Walks `.` rather than using CI's `./**/*.md` glob: lychee's glob ignores
//! `.gitignore`, so gitignored scratch markdown (`plans/`, `target/`, ...)
//! would fail every commit here while CI never sees it. Both skip hidden
//! directories (`.github`, `.claude`), so for tracked files the two check the
//! same set. This still checks the working tree, not the commit: an untracked,
//! non-ignored `.md` is checked here but not in CI, and a link to a file that
//! hasn't been `git add`ed passes here but fails in CI.
//!
//! lychee's output is shown only on failure, so the hook (which prek runs
//! `verbose` to surface the warnings below) stays quiet on a passing commit.
//!
//! [lychee]: https://github.com/lycheeverse/lychee

use std::ffi::OsStr;
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths;

/// lychee config, committed at the workspace root.
const LYCHEE_CONFIG: &str = "lychee.toml";
/// lychee version the `docs-links` CI job runs. Kept in sync with ci.yml's
/// `lycheeVersion` and CONTRIBUTING.md's install line by
/// `pinned_version_matches_ci_and_contributing` below.
const LYCHEE_VERSION: &str = "0.24.2";

/// Usability of the local lychee install.
#[derive(Debug, PartialEq, Eq)]
enum Checker {
    /// Installed at [`LYCHEE_VERSION`], the version CI runs.
    Ready,
    /// Not on `PATH`.
    Missing,
    /// Runs, but `--version` failed or printed something other than
    /// `lychee <version>`. Still run, with a warning: it is installed, so
    /// skipping it as "not installed" would be untrue.
    Unrecognised,
    /// Installed at another version. Still run — a link that resolves for one
    /// version resolves for another — but warned about, since CI is the gate.
    OtherVersion(String),
}

/// What probing `lychee --version` found.
#[derive(Debug, PartialEq, Eq)]
enum Probe {
    /// The program could not be found.
    NotFound,
    /// The program ran; its stdout if it exited successfully.
    Ran(Option<String>),
}

/// Pure policy: classify the checker from what the probe found.
fn classify_checker(probe: &Probe) -> Checker {
    match probe {
        Probe::NotFound => Checker::Missing,
        Probe::Ran(stdout) => match stdout.as_deref().and_then(parse_lychee_version) {
            Some(LYCHEE_VERSION) => Checker::Ready,
            Some(other) => Checker::OtherVersion(other.to_owned()),
            None => Checker::Unrecognised,
        },
    }
}

/// Pure parser for `lychee --version` output, which prints `lychee <semver>`.
/// `None` if the output is not in that form.
fn parse_lychee_version(stdout: &str) -> Option<&str> {
    let mut parts = stdout.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some("lychee"), Some(version)) => Some(version),
        _ => None,
    }
}

/// Program name of the checker, resolved on `PATH`.
const LYCHEE: &str = "lychee";

/// Run `program --version`.
fn probe(program: &OsStr) -> Result<Probe> {
    match Command::new(program).arg("--version").output() {
        Ok(output) => {
            Ok(Probe::Ran(output.status.success().then(|| {
                String::from_utf8_lossy(&output.stdout).into_owned()
            })))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(Probe::NotFound),
        Err(e) => Err(e).context("failed to run `lychee --version`"),
    }
}

/// The `cargo install` line that produces the version CI runs.
fn install_hint() -> String {
    format!("cargo install lychee@{LYCHEE_VERSION} --locked")
}

/// Entry point for the `lychee` subcommand.
///
/// With `if_available`, a missing lychee is a warning and a pass instead of an
/// error, so the local git hook never blocks a commit on a missing tool; the
/// `docs-links` CI job stays the gate.
pub fn run(if_available: bool) -> Result<()> {
    check(OsStr::new(LYCHEE), &paths::workspace_root()?, if_available)
}

/// [`run`] with the checker program and the directory it runs in injectable,
/// so tests can drive every path with a stub on a host that has lychee
/// installed.
fn check(program: &OsStr, root: &Path, if_available: bool) -> Result<()> {
    match (classify_checker(&probe(program)?), if_available) {
        (Checker::Ready, _) => {}
        (Checker::OtherVersion(found), _) => eprintln!(
            "lychee: found version {found}, but CI runs {LYCHEE_VERSION}; results may differ. \
             Pin it with: {}",
            install_hint()
        ),
        (Checker::Unrecognised, _) => eprintln!(
            "lychee: found, but could not read its version; running it anyway. CI runs \
             {LYCHEE_VERSION}; pin it with: {}",
            install_hint()
        ),
        (Checker::Missing, true) => {
            eprintln!(
                "lychee: not found on PATH; skipping the markdown-link check (the docs-links CI \
                 job still runs it). Install it with: {}",
                install_hint()
            );
            return Ok(());
        }
        (Checker::Missing, false) => bail!(
            "lychee is required to check markdown links.\nInstall it with: {}",
            install_hint()
        ),
    }

    let output = Command::new(program)
        .current_dir(root)
        .args(["--config", LYCHEE_CONFIG, "--extensions", "md", "."])
        .output()
        .context("failed to run lychee")?;
    if output.status.success() {
        return Ok(());
    }
    // Raw bytes: lychee's report contains emoji, which a text write to a
    // non-UTF-8 console could mangle. Its report is on stdout, so it comes
    // first.
    std::io::stdout().write_all(&output.stdout)?;
    std::io::stderr().write_all(&output.stderr)?;
    bail!(
        "lychee exited with {}: broken markdown links, or an error in lychee itself \
         (see its output above)",
        output.status
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_classification() {
        let ran = |stdout: &str| Probe::Ran(Some(stdout.to_owned()));
        assert_eq!(
            classify_checker(&ran(&format!("lychee {LYCHEE_VERSION}\n"))),
            Checker::Ready
        );
        assert_eq!(
            classify_checker(&ran("lychee 0.24.1\n")),
            Checker::OtherVersion("0.24.1".to_owned())
        );
        assert_eq!(
            classify_checker(&ran("lychee-bin 0.24.2")),
            Checker::Unrecognised
        );
        assert_eq!(classify_checker(&Probe::Ran(None)), Checker::Unrecognised);
        assert_eq!(classify_checker(&Probe::NotFound), Checker::Missing);
    }

    #[test]
    fn version_parser_accepts_lychee_output_only() {
        assert_eq!(parse_lychee_version("lychee 0.24.2\r\n"), Some("0.24.2"));
        assert_eq!(parse_lychee_version("lychee"), None);
        assert_eq!(parse_lychee_version("hawkeye 0.24.2"), None);
        assert_eq!(parse_lychee_version(""), None);
    }

    /// File a stub writes into the directory it was run in, proving the check
    /// actually ran lychee (and where) rather than skipping it.
    const RAN_MARKER: &str = "stub-lychee-ran";

    /// Write a stub lychee into `dir` that prints `version_line` for
    /// `--version`, and otherwise drops [`RAN_MARKER`] in its working
    /// directory and exits with `exit_code`.
    fn stub_lychee(dir: &Path, version_line: &str, exit_code: u8) -> std::path::PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = dir.join("lychee");
            std::fs::write(
                &path,
                format!(
                    "#!/bin/sh\nif [ \"$1\" = --version ]; then echo '{version_line}'; exit 0; fi\n\
                     echo ran > {RAN_MARKER}\nexit {exit_code}\n"
                ),
            )
            .expect("write stub");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod stub");
            path
        }
        #[cfg(windows)]
        {
            let path = dir.join("lychee.cmd");
            std::fs::write(
                &path,
                format!(
                    "@echo off\r\nif \"%1\"==\"--version\" (echo {version_line}& exit /b 0)\r\n\
                     echo ran> {RAN_MARKER}\r\nexit /b {exit_code}\r\n"
                ),
            )
            .expect("write stub");
            path
        }
    }

    /// Run [`check`] against a stub; returns the result and whether it ran.
    fn check_stub(version_line: &str, exit_code: u8, if_available: bool) -> (Result<()>, bool) {
        let bin = tempfile::tempdir().expect("tempdir");
        let root = tempfile::tempdir().expect("tempdir");
        let stub = stub_lychee(bin.path(), version_line, exit_code);
        let result = check(stub.as_os_str(), root.path(), if_available);
        (result, root.path().join(RAN_MARKER).exists())
    }

    #[test]
    fn pinned_version_runs_and_passes_or_fails_with_lychee() {
        let pinned = format!("lychee {LYCHEE_VERSION}");
        let (result, ran) = check_stub(&pinned, 0, true);
        assert!(result.is_ok() && ran, "a passing lychee run passes");
        let (result, ran) = check_stub(&pinned, 2, true);
        let err = result.expect_err("a failing lychee run must fail the check");
        assert!(ran, "the failure must come from running lychee");
        assert!(err.to_string().contains("exited with"), "{err}");
    }

    #[test]
    fn other_or_unreadable_version_still_runs() {
        // The warnings for these say lychee runs anyway; prove it does.
        assert!(check_stub("lychee 0.23.0", 0, true).1);
        assert!(check_stub("lychee-bin 0.24.2", 0, true).1);
    }

    /// Name of a program that cannot exist on `PATH`.
    const MISSING: &str = "lychee-missing-for-xtask-test";

    #[test]
    fn missing_checker_passes_only_with_if_available() {
        let root = tempfile::tempdir().expect("tempdir");
        let missing = OsStr::new(MISSING);
        // The hook's contract: a missing lychee never blocks a commit...
        check(missing, root.path(), true).expect("--if-available must pass without lychee");
        // ...but a direct run fails, naming the install command.
        let err =
            check(missing, root.path(), false).expect_err("a direct run must fail without lychee");
        assert!(
            err.to_string().contains(&install_hint()),
            "error must say how to install lychee: {err}"
        );
    }

    /// The pinned constant, the version CI runs, and the version contributors
    /// are told to install must never drift apart.
    #[test]
    fn pinned_version_matches_ci_and_contributing() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf();
        for (file, expected) in [
            (
                ".github/workflows/ci.yml",
                format!("lycheeVersion: v{LYCHEE_VERSION}"),
            ),
            ("CONTRIBUTING.md", install_hint()),
        ] {
            let path = root.join(file);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            assert!(
                text.contains(&expected),
                "{file} must contain `{expected}` to match LYCHEE_VERSION in lychee.rs"
            );
        }
    }
}
