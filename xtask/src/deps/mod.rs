// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps`: lock and check the third-party runtimes rocm-cli
//! installs. Binaries are pinned in `runtimes/mise.toml` and locked by mise;
//! the Python stacks are pinned in `runtimes/profiles.toml` and locked here,
//! per OS, with uv. See docs/runtime-dependencies.md.

mod check;
mod diff;
mod lock;
mod mise_lock;
/// The generator `crates/rocm-deps/build.rs` runs, compiled in so `deps check`
/// applies the build's own rules to the lock instead of a second copy of them.
#[path = "../../../crates/rocm-deps/src/pins_gen.rs"]
mod pins_gen;
mod profiles;
mod pylock;

use anyhow::Result;
use clap::Subcommand;

/// The command that re-locks every runtime, as every message, doc and
/// workflow that tells someone to run it spells it. It is run from the
/// repository root; mise only finds the tasks with `--cd runtimes`. The
/// `runtime-locks` CI job runs exactly this, which is what shows it clears
/// a missing or stale lock.
pub(crate) const LOCK_COMMAND: &str = "runtimes/bin/mise run --cd runtimes lock";

#[derive(Subcommand)]
pub(crate) enum DepsCommand {
    /// Compile each Python runtime profile into its per-OS lock with the uv
    /// pinned in `runtimes/mise.toml`. Each lock records the `exclude-newer`
    /// cutoff it was resolved at; without `--upgrade` a re-lock resolves at
    /// that same cutoff, so it reproduces the lock unless an input changed.
    Lock {
        /// Lock only this profile.
        #[arg(long)]
        profile: Option<String>,
        /// Resolve at a new cutoff, the cooldown before now, so packages move
        /// to the newest releases their requirements allow.
        #[arg(long)]
        upgrade: bool,
        /// How many days before now a new cutoff is set (with `--upgrade`, or
        /// for a lock that does not exist yet). AMD's indexes publish no upload
        /// times, so the AMD-only packages named in the profile policy are
        /// exempt from every cutoff; everything else, including what those
        /// indexes mirror from PyPI, is held back.
        #[arg(long)]
        cooldown_days: Option<u32>,
    },
    /// Verify, offline, that every runtime lock is complete and consistent.
    Check,
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
    }
}

#[cfg(test)]
mod tests {
    use super::LOCK_COMMAND;

    /// Every place that tells someone how to clear a stale or missing lock
    /// spells the command the same way. (That the `runtime-locks` job's own
    /// `run:` line is this command is pinned by `workflow_contract`.)
    #[test]
    fn the_remediation_command_is_spelled_the_same_everywhere() {
        let root = crate::paths::workspace_root().unwrap();
        for file in [
            // The design doc is where the command is documented; other docs
            // link to it rather than being read here, so an edit to them does
            // not force every crate's tests to run (`XTASK_READ_DOCS`).
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

    /// Both bootstrap launchers start the same mise, and the Windows CI step
    /// that runs `mise.cmd` expects that version. Bumping mise means
    /// regenerating both launchers (MANIFEST.md) and this step together.
    #[test]
    fn the_launchers_pin_the_same_mise() {
        let root = crate::paths::workspace_root().unwrap();
        let read = |path: &str| std::fs::read_to_string(root.join(path)).unwrap();
        let between = |text: &str, start: &str, end: &str| -> String {
            let from = text.find(start).unwrap_or_else(|| panic!("no {start:?}")) + start.len();
            text[from..from + text[from..].find(end).unwrap()].to_owned()
        };
        let windows = between(
            &read("runtimes/bin/mise.cmd"),
            "set \"pinned_version=",
            "\"",
        );
        let posix = between(&read("runtimes/bin/mise"), "${MISE_VERSION:-", "}");
        assert_eq!(
            windows, posix,
            "the two launchers pin different mise versions"
        );
        // The POSIX launcher's embedded checksums apply only while its
        // `current_version` is the default; otherwise it falls back to fetching
        // SHASUMS256.txt.
        let posix_text = read("runtimes/bin/mise");
        let current: Vec<&str> = posix_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("current_version=\"v"))
            .collect();
        assert!(
            !current.is_empty(),
            "the POSIX launcher embeds a current_version"
        );
        for version in current {
            assert_eq!(
                version,
                format!("{posix}\""),
                "current_version drifted from the default"
            );
        }
        // The regeneration command and the doc that cite the version.
        assert!(
            read("MANIFEST.md").contains(&format!("install-script --version {windows} ")),
            "MANIFEST.md must record the command for mise {windows}"
        );
        assert!(
            read("docs/runtime-dependencies.md").contains(&format!("(mise {windows})")),
            "docs/runtime-dependencies.md must name mise {windows}"
        );
        assert!(
            read(".github/workflows/ci.yml").contains(&format!(
                "findstr /b /c:\"{windows} \" \"%RUNNER_TEMP%\\mise-version.txt\""
            )),
            "the Windows CI step must expect mise {windows}"
        );
    }
}
