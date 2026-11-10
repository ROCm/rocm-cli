// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps`: lock and check the third-party runtimes rocm-cli
//! installs. Binaries are pinned in `runtimes/mise.toml` and locked by mise;
//! the Python stacks are pinned in `runtimes/profiles.toml` and locked here,
//! per OS, with uv. See docs/runtime-dependencies.md.

mod check;
mod diff;
mod launchers;
mod lock;
mod mise_lock;
/// The generator `crates/rocm-deps/build.rs` runs, compiled in so `deps check`
/// applies the build's own rules to the lock instead of a second copy of them.
#[path = "../../../crates/rocm-deps/src/pins_gen.rs"]
mod pins_gen;
mod profiles;
mod pylock;
mod verify;

use anyhow::Result;
use clap::Subcommand;

/// The command that re-locks every runtime. `deps check`'s message uses this
/// constant; the design doc, `runtimes/mise.toml` and the workflows are
/// checked to spell it the same way (`docs/testing.md` and
/// `docs/engine-plugins.md` spell it too, unchecked). It is run from the
/// repository root; mise only finds the tasks with `--cd runtimes`. The
/// `runtime-locks` CI job runs exactly this, which is what shows it clears
/// a missing or stale lock.
pub(crate) const LOCK_COMMAND: &str = "runtimes/bin/mise run --cd runtimes lock";

#[derive(Subcommand)]
pub(crate) enum DepsCommand {
    /// Compile each Python runtime profile into its per-OS lock with the uv
    /// pinned in `runtimes/mise.toml`. Each lock records the `exclude-newer`
    /// cutoff it was resolved at; without `--upgrade` a re-lock resolves at
    /// that same cutoff, so it reproduces the lock unless an input changed or
    /// AMD published a new build of an AMD-only package, which no cutoff
    /// holds back (AMD's indexes publish no upload times).
    Lock {
        /// Lock only this profile.
        #[arg(long)]
        profile: Option<String>,
        /// Resolve at a new cutoff, the cooldown before now, so packages move
        /// to the newest releases their requirements allow.
        #[arg(long)]
        upgrade: bool,
        /// How many days before now a new cutoff is set (with `--upgrade`, for
        /// a lock that does not exist yet, or for one whose recorded cutoff is
        /// missing or invalid). Defaults to 0. AMD's indexes publish no upload
        /// times, so the AMD-only packages named in the profile policy are
        /// exempt from every cutoff; everything else, including what those
        /// indexes mirror from PyPI, is held back.
        #[arg(long)]
        cooldown_days: Option<u32>,
    },
    /// Verify, offline, that every runtime lock is complete and consistent.
    Check,
    /// Download every binary `runtimes/mise.lock` records without a build
    /// attestation (Lemonade) and check its SHA-256 against the lock (network).
    VerifyDownloads,
    /// Regenerate the mise bootstrap launchers in `runtimes/bin/` for the
    /// newest mise release at least `--cooldown-days` old, if that is newer
    /// than the one they pin (network).
    Launchers {
        /// The mise binary that looks up releases and generates the launchers.
        #[arg(long)]
        mise: std::path::PathBuf,
        /// How many days old a release must be.
        #[arg(long)]
        cooldown_days: u32,
    },
    /// Check that `runtimes/bin/`'s launchers are exactly what the launchers
    /// in `--base` (from before an update) generate for the version they pin,
    /// and that a changed version is newer and at least `--cooldown-days` old
    /// (network).
    VerifyLaunchers {
        /// A directory holding the launchers from before the update, as
        /// `mise` and `mise.cmd`; its `mise` is run.
        #[arg(long)]
        base: std::path::PathBuf,
        /// How many days old a release must be.
        #[arg(long)]
        cooldown_days: u32,
    },
    /// Print a Markdown table of versions changed since a git revision.
    Diff {
        /// Revision to compare against.
        #[arg(long, default_value = "HEAD")]
        base: String,
    },
}

pub(crate) fn run(command: DepsCommand) -> Result<()> {
    match command {
        DepsCommand::Lock {
            profile,
            upgrade,
            cooldown_days,
        } => lock::run(&lock::LockOptions {
            profile,
            upgrade,
            cooldown_days,
        }),
        DepsCommand::Check => check::run(),
        DepsCommand::Diff { base } => diff::run(&base),
        DepsCommand::VerifyDownloads => verify::run(),
        DepsCommand::Launchers {
            mise,
            cooldown_days,
        } => launchers::run_upgrade(mise, cooldown_days),
        DepsCommand::VerifyLaunchers {
            base,
            cooldown_days,
        } => launchers::run_verify(&base, cooldown_days),
    }
}

#[cfg(test)]
mod tests {
    use super::LOCK_COMMAND;

    /// Every place that tells someone how to clear a stale or missing lock
    /// spells the command the same way. (That the `runtime-locks` job's own
    /// `run:` line is this command is pinned by `workflow_contract`.)
    #[test]
    fn the_remediation_command_is_spelled_the_same_in_its_checked_files() {
        let root = crate::paths::workspace_root().unwrap();
        for file in [
            // `docs/testing.md` and `docs/engine-plugins.md` also spell the
            // command but are not read here: a doc an xtask test reads forces
            // every crate's tests to run on each edit (`XTASK_READ_DOCS`), and
            // those two are edited by most pull requests.
            "docs/runtime-dependencies.md",
            "runtimes/mise.toml",
            ".github/workflows/ci.yml",
            ".github/workflows/runtime-deps-update.yml",
        ] {
            let text = std::fs::read_to_string(root.join(file)).unwrap();
            // The update workflow names the `update` task, not `lock`.
            if !file.ends_with("runtime-deps-update.yml") {
                assert!(
                    text.contains(LOCK_COMMAND),
                    "{file} does not name `{LOCK_COMMAND}`"
                );
            }
            assert!(
                !text.contains("runtimes/bin/mise run lock")
                    && !text.contains("runtimes/bin/mise run update"),
                "{file} names a task without `--cd runtimes`, which mise cannot find from the root"
            );
        }
    }

    /// Both bootstrap launchers start the same mise, with the checksums
    /// embedded for that version. Nothing else spells the version: the weekly
    /// update moves the launchers alone (`deps launchers`), and the Windows
    /// CI step reads the version from `mise.cmd`.
    #[test]
    fn the_launchers_pin_the_same_mise() {
        let root = crate::paths::workspace_root().unwrap();
        let read = |path: &str| std::fs::read_to_string(root.join(path)).unwrap();
        let version = super::launchers::pinned(&root).unwrap();
        // The POSIX launcher's embedded checksums apply only while its
        // `current_version` is the default; otherwise it falls back to fetching
        // SHASUMS256.txt.
        let posix_text = read(super::launchers::POSIX);
        let current: Vec<&str> = posix_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("current_version=\"v"))
            .collect();
        assert!(
            !current.is_empty(),
            "the POSIX launcher embeds a current_version"
        );
        for embedded in current {
            assert_eq!(
                embedded,
                format!("{version}\""),
                "current_version drifted from the default"
            );
        }
        // MANIFEST.md records the command, not a version that would go stale
        // on the next weekly update.
        let manifest = read("MANIFEST.md");
        assert!(
            manifest
                .contains("install-script --version <version> --write runtimes/bin/mise --windows"),
            "MANIFEST.md must record the command that generates the launchers"
        );
        assert!(
            !manifest.contains(&version)
                && !read("docs/runtime-dependencies.md").contains(&format!("mise {version}")),
            "the mise version is spelled only in the launchers"
        );
    }

    /// One cooldown for every dependency: the binaries' `minimum_release_age`
    /// and both mise tasks' `--cooldown-days` equal what Dependabot waits
    /// before proposing a Cargo, Actions or pre-commit update.
    #[test]
    fn the_runtime_cooldown_matches_dependabots() {
        let root = crate::paths::workspace_root().unwrap();
        let read = |path: &str| std::fs::read_to_string(root.join(path)).unwrap();
        let dependabot: Vec<String> = read(".github/dependabot.yml")
            .lines()
            .filter_map(|l| {
                l.trim()
                    .strip_prefix("default-days:")
                    .map(|d| d.trim().to_owned())
            })
            .collect();
        assert!(!dependabot.is_empty(), "dependabot.yml sets a cooldown");
        let days = &dependabot[0];
        assert!(dependabot.iter().all(|d| d == days), "{dependabot:?}");
        let mise = read("runtimes/mise.toml");
        assert!(
            mise.lines()
                .any(|l| l.trim() == format!("minimum_release_age = \"{days}d\"")),
            "runtimes/mise.toml must wait {days} days, as Dependabot does"
        );
        let flag = format!("--cooldown-days {days}\"");
        assert_eq!(
            mise.matches(&flag).count(),
            3,
            "`lock`, and `update`'s re-lock and launcher steps, must pass `--cooldown-days {days}`"
        );
        // The weekly update's `check` job holds the launchers to the same wait.
        assert!(
            read(".github/workflows/runtime-deps-update.yml").lines().any(|l| l.trim()
                == format!(
                    "cargo xtask deps verify-launchers --base \"$RUNNER_TEMP/base-launchers\" --cooldown-days {days}"
                )),
            "the update's check job must verify the launchers with --cooldown-days {days}"
        );
    }
}
