// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The vendored mise bootstrap launchers, `runtimes/bin/mise` and
//! `runtimes/bin/mise.cmd`: `cargo xtask deps launchers` moves them to the
//! newest mise past the cooldown, and `cargo xtask deps verify-launchers`
//! checks, from code that was not part of the update, that they are exactly
//! what mise generates for the version they pin.
//!
//! Each launcher embeds the version it starts and that release's SHA-256s, so
//! a bump is a change to both files. They are regenerated, never edited: the
//! command is `mise generate install-script --version <v> --write
//! runtimes/bin/mise --windows`, which writes both (MANIFEST.md).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

/// The POSIX launcher; `mise generate install-script --write` takes this path
/// and writes the Windows one beside it, with `.cmd` added.
pub(crate) const POSIX: &str = "runtimes/bin/mise";
/// The Windows launcher.
pub(crate) const WINDOWS: &str = "runtimes/bin/mise.cmd";
/// mise as mise names it for `latest`: its own GitHub releases.
const MISE_TOOL: &str = "aqua:jdx/mise";

/// What these commands need from a mise binary.
pub(crate) trait Mise {
    /// The newest mise release at least `cooldown_days` old, or with
    /// `version`, that release if it is that old; empty when there is none.
    fn latest(&self, version: Option<&str>, cooldown_days: u32) -> Result<String>;
    /// Write both launchers for `version` into `dir` (as `mise` and
    /// `mise.cmd`).
    fn generate(&self, version: &str, dir: &Path) -> Result<()>;
}

/// A mise binary, or a launcher script that starts one. It runs in an empty
/// directory, so no `mise.toml` (the update's included) configures it.
pub(crate) struct RealMise {
    pub(crate) bin: PathBuf,
    pub(crate) work: PathBuf,
}

impl RealMise {
    fn run(&self, args: &[&str]) -> Result<String> {
        std::fs::create_dir_all(&self.work)?;
        let output = Command::new(&self.bin)
            .current_dir(&self.work)
            .args(args)
            .output()
            .with_context(|| format!("failed to run {}", self.bin.display()))?;
        ensure!(
            output.status.success(),
            "{} {} failed: {}",
            self.bin.display(),
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
}

impl Mise for RealMise {
    fn latest(&self, prefix: Option<&str>, cooldown_days: u32) -> Result<String> {
        let tool = prefix.map_or_else(|| MISE_TOOL.to_owned(), |p| format!("{MISE_TOOL}@{p}"));
        self.run(&[
            "latest",
            "--minimum-release-age",
            &format!("{cooldown_days}d"),
            &tool,
        ])
    }

    fn generate(&self, version: &str, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let out = dir.join("mise");
        self.run(&[
            "generate",
            "install-script",
            "--version",
            version,
            "--write",
            &out.to_string_lossy(),
            "--windows",
        ])
        .map(drop)
    }
}

/// A mise version as its releases name them (`2026.10.25`): numbers and dots.
fn checked_version(version: &str) -> Result<&str> {
    ensure!(
        !version.is_empty()
            && version
                .split('.')
                .all(|part| { !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) }),
        "{version:?} is not a mise version"
    );
    Ok(version)
}

/// Whether `a` is a later mise version than `b`.
fn newer(a: &str, b: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> { v.split('.').map(|p| p.parse().unwrap_or(0)).collect() };
    parts(a) > parts(b)
}

/// The mise version one launcher starts: `file` is [`POSIX`] or [`WINDOWS`].
pub(crate) fn version_in(file: &str, text: &str) -> Result<String> {
    let (start, end, missing) = if file == WINDOWS {
        (
            "set \"pinned_version=",
            '"',
            "the Windows launcher names no pinned_version",
        )
    } else {
        (
            "${MISE_VERSION:-",
            '}',
            "the POSIX launcher names no default MISE_VERSION",
        )
    };
    let from = text.find(start).context(missing)? + start.len();
    let len = text[from..].find(end).context(missing)?;
    Ok(checked_version(&text[from..from + len])?.to_owned())
}

/// The mise version both launchers start; they must agree.
pub(crate) fn pinned_in(posix: &str, windows: &str) -> Result<String> {
    let posix = version_in(POSIX, posix)?;
    let windows = version_in(WINDOWS, windows)?;
    ensure!(
        posix == windows,
        "the launchers pin different mise versions: {posix} and {windows}"
    );
    Ok(posix)
}

fn read_pair(posix: &Path, windows: &Path) -> Result<(String, String)> {
    let read = |path: &Path| {
        std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
    };
    Ok((read(posix)?, read(windows)?))
}

/// The mise version `root`'s launchers start.
pub(crate) fn pinned(root: &Path) -> Result<String> {
    let (posix, windows) = read_pair(&root.join(POSIX), &root.join(WINDOWS))?;
    pinned_in(&posix, &windows)
}

/// `cargo xtask deps launchers`: move the launchers to the newest mise at
/// least `cooldown_days` old, if that is newer than what they pin. Returns
/// the version moved to, if any.
pub(crate) fn upgrade(
    root: &Path,
    work: &Path,
    mise: &dyn Mise,
    cooldown_days: u32,
) -> Result<Option<String>> {
    let current = pinned(root)?;
    let latest = mise.latest(None, cooldown_days)?;
    let latest = checked_version(&latest)?.to_owned();
    if !newer(&latest, &current) {
        eprintln!("mise {current} is the newest release past the cooldown");
        return Ok(None);
    }
    let dir = work.join("launchers");
    let _ = std::fs::remove_dir_all(&dir);
    mise.generate(&latest, &dir)?;
    let (posix, windows) = read_pair(&dir.join("mise"), &dir.join("mise.cmd"))?;
    ensure!(
        pinned_in(&posix, &windows)? == latest,
        "mise generated launchers for a version other than {latest}"
    );
    // Written over the committed files, never renamed into place, so the
    // POSIX launcher keeps its executable bit.
    std::fs::write(root.join(POSIX), posix)?;
    std::fs::write(root.join(WINDOWS), windows)?;
    eprintln!("moved the mise launchers from {current} to {latest}");
    Ok(Some(latest))
}

/// `cargo xtask deps verify-launchers`: `root`'s launchers must be exactly
/// what `base` (a mise from before the update) generates for the version they
/// pin, and a version other than `base_version` must be newer than it and at
/// least `cooldown_days` old. Returns one message per problem.
pub(crate) fn verify(
    root: &Path,
    work: &Path,
    base: &dyn Mise,
    base_version: &str,
    cooldown_days: u32,
) -> Result<Vec<String>> {
    let (posix, windows) = read_pair(&root.join(POSIX), &root.join(WINDOWS))?;
    let version = match pinned_in(&posix, &windows) {
        Ok(version) => version,
        Err(err) => return Ok(vec![format!("{err:#}")]),
    };
    let mut problems = Vec::new();
    if version != base_version {
        if !newer(&version, base_version) {
            problems.push(format!(
                "the launchers move mise from {base_version} back to {version}"
            ));
        }
        let aged = base.latest(Some(&version), cooldown_days)?;
        if aged != version {
            problems.push(format!(
                "mise {version} is not a release at least {cooldown_days} days old"
            ));
        }
    }
    let dir = work.join("launchers-expected");
    let _ = std::fs::remove_dir_all(&dir);
    base.generate(&version, &dir)?;
    for (committed, file, text) in [(POSIX, "mise", &posix), (WINDOWS, "mise.cmd", &windows)] {
        let expected = std::fs::read_to_string(dir.join(file))
            .with_context(|| format!("mise generated no {file}"))?;
        // `.gitattributes` checks `mise.cmd` out with CRLF (cmd needs it) and
        // stores it with LF, as mise writes it; compare in the stored form.
        if text.replace("\r\n", "\n") != expected.replace("\r\n", "\n") {
            problems.push(format!(
                "{committed} is not what mise generates for {version}; regenerate it (MANIFEST.md)"
            ));
        }
    }
    Ok(problems)
}

pub(crate) fn run_upgrade(mise: PathBuf, cooldown_days: u32) -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let work = crate::paths::target_dir(&root).join("deps-launchers");
    let mise = RealMise {
        bin: mise,
        work: work.join("run"),
    };
    upgrade(&root, &work, &mise, cooldown_days).map(drop)
}

pub(crate) fn run_verify(base: &Path, cooldown_days: u32) -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let work = crate::paths::target_dir(&root).join("deps-verify-launchers");
    let (posix, windows) = read_pair(&base.join("mise"), &base.join("mise.cmd"))?;
    let base_version = pinned_in(&posix, &windows).context("the base launchers")?;
    let mise = RealMise {
        bin: base.join("mise"),
        work: work.join("run"),
    };
    let problems = verify(&root, &work, &mise, &base_version, cooldown_days)?;
    if problems.is_empty() {
        return Ok(());
    }
    for problem in &problems {
        eprintln!("  - {problem}");
    }
    bail!(
        "{} problem(s) with the mise launchers; do not commit them",
        problems.len()
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn launchers(version: &str, body: &str) -> (String, String) {
        (
            format!("#!/usr/bin/env bash\nmise_version=\"${{MISE_VERSION:-{version}}}\"\n{body}\n"),
            // LF, as mise writes it and git stores it.
            format!("@echo off\nset \"pinned_version={version}\"\n{body}\n"),
        )
    }

    /// Generates launchers whose body is `body`, and serves `latest` from a
    /// list of `(version, days old)` releases.
    struct FakeMise {
        releases: Vec<(&'static str, u32)>,
        body: &'static str,
        generated: RefCell<Vec<String>>,
    }

    impl Mise for FakeMise {
        fn latest(&self, prefix: Option<&str>, cooldown_days: u32) -> Result<String> {
            // As real mise: a prefix that is a whole version matches only that
            // version, and no match prints nothing (exit 0).
            Ok(self
                .releases
                .iter()
                .filter(|(v, age)| *age >= cooldown_days && prefix.is_none_or(|p| *v == p))
                .map(|(v, _)| (*v).to_owned())
                .reduce(|a, b| if newer(&a, &b) { a } else { b })
                .unwrap_or_default())
        }

        fn generate(&self, version: &str, dir: &Path) -> Result<()> {
            self.generated.borrow_mut().push(version.to_owned());
            std::fs::create_dir_all(dir)?;
            let (posix, windows) = launchers(version, self.body);
            std::fs::write(dir.join("mise"), posix)?;
            std::fs::write(dir.join("mise.cmd"), windows)?;
            Ok(())
        }
    }

    fn fake(releases: Vec<(&'static str, u32)>) -> FakeMise {
        FakeMise {
            releases,
            body: "generated",
            generated: RefCell::default(),
        }
    }

    fn workspace(version: &str, body: &str) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("runtimes/bin")).unwrap();
        let (posix, windows) = launchers(version, body);
        std::fs::write(root.path().join(POSIX), posix).unwrap();
        std::fs::write(root.path().join(WINDOWS), windows).unwrap();
        root
    }

    #[test]
    fn upgrade_moves_to_the_newest_release_past_the_cooldown() {
        let root = workspace("2026.10.25", "generated");
        let mise = fake(vec![("2026.10.25", 16), ("2026.11.1", 9), ("2026.11.5", 5)]);
        let moved = upgrade(root.path(), &root.path().join("work"), &mise, 7).unwrap();
        assert_eq!(moved.as_deref(), Some("2026.11.1"));
        assert_eq!(pinned(root.path()).unwrap(), "2026.11.1");
        let (posix, windows) = launchers("2026.11.1", "generated");
        assert_eq!(
            std::fs::read_to_string(root.path().join(POSIX)).unwrap(),
            posix
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join(WINDOWS)).unwrap(),
            windows
        );
    }

    #[test]
    fn upgrade_changes_nothing_when_nothing_newer_is_past_the_cooldown() {
        let root = workspace("2026.10.25", "committed");
        let mise = fake(vec![
            ("2026.10.8", 30),
            ("2026.10.25", 16),
            ("2026.11.5", 5),
        ]);
        assert_eq!(
            upgrade(root.path(), &root.path().join("work"), &mise, 7).unwrap(),
            None
        );
        assert!(mise.generated.borrow().is_empty());
        let (posix, _) = launchers("2026.10.25", "committed");
        assert_eq!(
            std::fs::read_to_string(root.path().join(POSIX)).unwrap(),
            posix
        );
    }

    #[cfg(unix)]
    #[test]
    fn upgrade_keeps_the_posix_launcher_executable() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = workspace("2026.10.25", "generated");
        let path = root.path().join(POSIX);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        upgrade(
            root.path(),
            &root.path().join("work"),
            &fake(vec![("2026.11.1", 9)]),
            7,
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }

    #[test]
    fn launchers_that_disagree_or_name_no_version_are_refused() {
        let (posix, _) = launchers("2026.10.25", "x");
        let (_, windows) = launchers("2026.11.1", "x");
        assert_eq!(
            format!("{:#}", pinned_in(&posix, &windows).unwrap_err()),
            "the launchers pin different mise versions: 2026.10.25 and 2026.11.1"
        );
        let (posix, windows) = launchers("2026.10.25|x", "x");
        assert!(pinned_in(&posix, &windows).is_err());
        assert!(pinned_in("no version", &windows).is_err());
    }

    #[test]
    fn verify_passes_launchers_the_base_regenerates_byte_for_byte() {
        let base = fake(vec![("2026.10.25", 16), ("2026.11.1", 9)]);
        for version in ["2026.10.25", "2026.11.1"] {
            let root = workspace(version, "generated");
            let found = verify(
                root.path(),
                &root.path().join("work"),
                &base,
                "2026.10.25",
                7,
            )
            .unwrap();
            assert_eq!(found, Vec::<String>::new(), "{version}");
        }
    }

    #[test]
    fn verify_refuses_launchers_that_differ_from_what_the_base_generates() {
        let root = workspace("2026.11.1", "tampered");
        let base = fake(vec![("2026.11.1", 9)]);
        let found = verify(
            root.path(),
            &root.path().join("work"),
            &base,
            "2026.10.25",
            7,
        )
        .unwrap();
        assert_eq!(
            found,
            [
                "runtimes/bin/mise is not what mise generates for 2026.11.1; regenerate it (MANIFEST.md)",
                "runtimes/bin/mise.cmd is not what mise generates for 2026.11.1; regenerate it (MANIFEST.md)",
            ]
        );
    }

    #[test]
    fn verify_refuses_a_release_inside_the_cooldown_or_an_older_one() {
        let base = fake(vec![
            ("2026.10.8", 30),
            ("2026.10.25", 16),
            ("2026.11.5", 5),
        ]);
        let root = workspace("2026.11.5", "generated");
        let found = verify(
            root.path(),
            &root.path().join("work"),
            &base,
            "2026.10.25",
            7,
        )
        .unwrap();
        assert_eq!(
            found,
            ["mise 2026.11.5 is not a release at least 7 days old"]
        );
        let root = workspace("2026.10.8", "generated");
        let found = verify(
            root.path(),
            &root.path().join("work"),
            &base,
            "2026.10.25",
            7,
        )
        .unwrap();
        assert_eq!(
            found,
            ["the launchers move mise from 2026.10.25 back to 2026.10.8"]
        );
    }

    /// A checked-out `mise.cmd` has CRLF line endings (`.gitattributes`);
    /// it is still mise's own.
    #[test]
    fn verify_reads_a_checked_out_windows_launcher_in_its_stored_form() {
        let root = workspace("2026.10.25", "generated");
        let (_, windows) = launchers("2026.10.25", "generated");
        std::fs::write(root.path().join(WINDOWS), windows.replace('\n', "\r\n")).unwrap();
        let found = verify(
            root.path(),
            &root.path().join("work"),
            &fake(vec![]),
            "2026.10.25",
            7,
        )
        .unwrap();
        assert_eq!(found, Vec::<String>::new());
    }

    #[test]
    fn verify_reports_launchers_that_disagree() {
        let root = workspace("2026.11.1", "generated");
        let (_, windows) = launchers("2026.10.25", "generated");
        std::fs::write(root.path().join(WINDOWS), windows).unwrap();
        let found = verify(
            root.path(),
            &root.path().join("work"),
            &fake(vec![]),
            "2026.10.25",
            7,
        )
        .unwrap();
        assert_eq!(
            found,
            ["the launchers pin different mise versions: 2026.11.1 and 2026.10.25"]
        );
    }

    #[test]
    fn versions_compare_by_number_not_text() {
        assert!(newer("2026.10.25", "2026.9.30"));
        assert!(newer("2026.11.1", "2026.10.25"));
        assert!(!newer("2026.10.25", "2026.10.25"));
    }
}
