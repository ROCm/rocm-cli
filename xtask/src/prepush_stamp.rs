// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pass stamps for `cargo xtask pre-push-check`: remember that a cargo step
//! already passed on exactly this content, and skip it the next time.
//!
//! The pre-push hook has no memory, so a push retried after a dropped
//! connection, a manual run before the push, an amend that only rewords the
//! message, or the same tree pushed to a second remote re-ran every step on a
//! tree it had just checked. A stamp is written after a step's cargo command
//! exits 0 and is looked up before the next run of that step.
//!
//! **What a stamp stands for.** One step (`clippy`, `clippy-e2e` or `test`), the
//! exact cargo arguments it ran, the git *tree* hash of `HEAD` (not the commit
//! hash, so an amend or a re-commit with identical content still hits), the
//! compiler and cargo identity (`rustc -vV`, `cargo -vV`), and a digest of the
//! `CARGO_*`/`RUST*` environment. Because the arguments are part of the key, a
//! pass recorded for a narrow selection can never let a later, wider selection
//! skip: `-p xtask` and `--workspace` are different stamps.
//!
//! **The working tree must be clean.** Cargo checks the working tree, not the
//! pushed commit, so a stamp keyed on `HEAD`'s tree is only true when the
//! working tree *is* that tree. With uncommitted changes or untracked files
//! (an untracked `mod` target can hide a missing `git add`) a stamp is neither
//! honored nor written, and the step says why. After the run the tree is checked
//! again: if cargo changed it (a rewritten `Cargo.lock`, say), nothing is
//! recorded.
//!
//! **Fail safe.** Every error while locating the store, reading, parsing or
//! verifying a stamp means the step runs; a stamp that does not match its key
//! byte for byte, is unreadable, belongs to a different format, is dated in the
//! future or is older than [`MAX_AGE_SECS`] is ignored; a failed write is
//! reported and never fails the hook. A stamp is only written after cargo
//! succeeded.
//!
//! **Where.** Under the git *common* directory, so every linked worktree of a
//! clone shares the stamps, nothing lands in the work tree (no ignore rule, no
//! chance of committing one), and `cargo clean` does not erase them. Each write
//! prunes stamps older than [`MAX_AGE_SECS`] and keeps at most [`MAX_STAMPS`].
//!
//! **Not covered**, and said in the skip message: user-level cargo
//! configuration (`~/.cargo/config.toml`), installed tools, files git ignores,
//! network-dependent or flaky tests. `--rerun` or [`RERUN_VAR`] ignores recorded
//! passes (and records a fresh one), and `SKIP=<hook id>` still skips a hook
//! outright. The `verify-commits` hook is a separate hook and runs every time.
//!
//! As elsewhere in this crate the decisions are pure and the world is injected:
//! git and the toolchain sit behind [`Probe`], the filesystem behind [`Store`],
//! and the clock is a parameter, so the rules above are unit-tested without
//! cargo, git or a disk.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::affected;

/// Setting this to anything but empty or `0` ignores recorded passes.
pub const RERUN_VAR: &str = "ROCM_PRE_PUSH_RERUN";

/// Directory name under the git common directory.
const STAMP_DIR: &str = "rocm-cli-prepush-passes";
const STAMP_EXT: &str = ".stamp";
/// Scratch files a write goes through before it is renamed into place.
const TMP_EXT: &str = ".tmp";
const FORMAT_LINE: &str = "rocm-cli pre-push pass stamp v1";

/// The revision whose hash identifies the checked content.
const HEAD_TREE: &str = "HEAD^{tree}";

/// A pass older than this is ignored and pruned: toolchains, dependencies and
/// the registry move on, and a week is longer than any realistic retry loop.
pub const MAX_AGE_SECS: u64 = 7 * 24 * 60 * 60;

/// At most this many stamps are kept; the oldest go first.
pub const MAX_STAMPS: usize = 64;

/// A scratch file older than this belongs to a write that died; a younger one
/// may be a concurrent writer's, so it is left alone.
pub const TMP_MAX_AGE_SECS: u64 = 60 * 60;

/// A stamp dated this far ahead of the clock is treated as corrupt.
const FUTURE_SLACK_SECS: u64 = 5 * 60;

/// Environment variable prefixes that can change what a cargo step does.
/// Over-including only costs a re-run; leaving one out could skip a step that
/// would now fail.
const ENV_PREFIXES: &[&str] = &["CARGO_", "RUST"];

/// Variables under those prefixes that differ between runs without changing the
/// outcome: where artifacts go, terminal styling, and what `cargo run` sets for
/// the `xtask` binary itself.
const ENV_IGNORED: &[&str] = &[
    "CARGO_TARGET_DIR",
    "CARGO_MAKEFLAGS",
    "CARGO_MANIFEST_DIR",
    "CARGO_MANIFEST_PATH",
    "CARGO_BIN_NAME",
    "CARGO_CRATE_NAME",
    "CARGO_PRIMARY_PACKAGE",
    "RUST_BACKTRACE",
    "RUST_LIB_BACKTRACE",
];
const ENV_IGNORED_PREFIXES: &[&str] = &["CARGO_TERM_", "CARGO_PKG_", "CARGO_BIN_EXE_"];

fn hex_digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// Whether `--rerun` or [`RERUN_VAR`] asks for recorded passes to be ignored.
pub fn rerun_requested(flag: bool, lookup: impl Fn(&str) -> Option<String>) -> bool {
    flag || lookup(RERUN_VAR).is_some_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0"
    })
}

/// Digest of the environment variables that can change a cargo step, in a
/// stable order. Only the digest is kept, never a value.
pub fn env_digest(vars: impl IntoIterator<Item = (String, String)>) -> String {
    let mut relevant: Vec<(String, String)> = vars
        .into_iter()
        .filter(|(name, _)| {
            ENV_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
                && !ENV_IGNORED.contains(&name.as_str())
                && !ENV_IGNORED_PREFIXES
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
        })
        .collect();
    relevant.sort();
    let mut text = String::new();
    for (name, value) in relevant {
        text.push_str(&name);
        text.push('=');
        text.push_str(&value);
        text.push('\n');
    }
    hex_digest(&text)
}

/// Everything one pass depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    step: String,
    args: String,
    tree: String,
    toolchain: String,
    env: String,
}

impl Key {
    fn canonical(&self) -> String {
        format!(
            "step: {}\ncargo: {}\ntree: {}\ntoolchain: {}\nenv: {}\n",
            self.step, self.args, self.tree, self.toolchain, self.env
        )
    }

    fn file_name(&self) -> String {
        let digest = hex_digest(&self.canonical());
        format!("{}-{}{STAMP_EXT}", self.step, &digest[..32])
    }

    fn stamp_text(&self, passed_at: u64) -> String {
        format!(
            "{FORMAT_LINE}\n{}passed-at: {passed_at}\n",
            self.canonical()
        )
    }
}

/// What git and the toolchain say about the state being checked.
pub trait Probe {
    /// The tree hash of `HEAD`, only when the working tree is exactly that tree.
    /// `Err` carries the reason stamps cannot be trusted right now.
    fn clean_tree(&self) -> std::result::Result<String, String>;
    /// A description of the compiler and cargo that will run the step.
    fn toolchain(&self) -> std::result::Result<String, String>;
    /// [`env_digest`] of the current environment.
    fn env_digest(&self) -> String;
}

/// Where stamps are kept. Names are file names inside the store.
pub trait Store {
    fn read(&self, name: &str) -> Result<Option<String>>;
    fn write(&self, name: &str, contents: &str) -> Result<()>;
    /// Every stamp as `(name, modified time in seconds since the epoch)`.
    fn list(&self) -> Result<Vec<(String, u64)>>;
    fn remove(&self, name: &str) -> Result<()>;
    /// Where the stamps are, for messages.
    fn location(&self) -> String;
}

/// The outcome of looking for a recorded pass.
#[derive(Debug, PartialEq, Eq)]
pub enum Lookup {
    /// A matching pass is on record: skip the step.
    Hit { name: String, age_secs: u64 },
    /// Run the step. `key` is `Some` when a pass can be recorded afterwards, and
    /// `note` explains anything the user should know (a dirty tree, a rejected
    /// stamp, `--rerun`).
    Miss {
        key: Option<Key>,
        note: Option<String>,
    },
}

/// Stamp handling for one invocation, with the world injected.
pub struct Stamps<'a> {
    pub probe: &'a dyn Probe,
    pub store: &'a dyn Store,
    /// Seconds since the epoch.
    pub now: u64,
    /// Ignore recorded passes (but still record a new one).
    pub rerun: bool,
}

impl<'a> Stamps<'a> {
    /// The stamps for one invocation. `rerun_flag` is `--rerun`; `lookup` reads
    /// the environment for [`RERUN_VAR`], so tests need no process environment.
    pub fn new(
        probe: &'a dyn Probe,
        store: &'a dyn Store,
        now: u64,
        rerun_flag: bool,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Self {
        Self {
            probe,
            store,
            now,
            rerun: rerun_requested(rerun_flag, lookup),
        }
    }
}

impl Stamps<'_> {
    fn key(&self, step: &str, args: &[String]) -> std::result::Result<Key, String> {
        let tree = self.probe.clean_tree()?;
        let toolchain = hex_digest(&self.probe.toolchain()?);
        Ok(Key {
            step: step.to_string(),
            args: args.join(" "),
            tree,
            toolchain,
            env: self.probe.env_digest(),
        })
    }

    /// Look for a pass of `step` run with `args` on the current tree.
    pub fn lookup(&self, step: &str, args: &[String]) -> Lookup {
        let key = match self.key(step, args) {
            Ok(key) => key,
            Err(why) => {
                return Lookup::Miss {
                    key: None,
                    note: Some(format!("pass stamps are not used: {why}")),
                };
            }
        };
        if self.rerun {
            return Lookup::Miss {
                key: Some(key),
                note: Some(format!(
                    "recorded passes are ignored because --rerun or {RERUN_VAR} is set"
                )),
            };
        }
        let name = key.file_name();
        let text = match self.store.read(&name) {
            Ok(Some(text)) => text,
            Ok(None) => {
                return Lookup::Miss {
                    key: Some(key),
                    note: None,
                };
            }
            Err(error) => {
                return Lookup::Miss {
                    key: Some(key),
                    note: Some(format!(
                        "could not read the pass stamp ({error:#}); running the step"
                    )),
                };
            }
        };
        match self.verify(&key, &text) {
            Ok(Some(age_secs)) => Lookup::Hit { name, age_secs },
            Ok(None) => Lookup::Miss {
                key: Some(key),
                note: None,
            },
            Err(why) => Lookup::Miss {
                key: Some(key),
                note: Some(format!("ignoring the pass stamp {name}: {why}")),
            },
        }
    }

    /// `Ok(Some(age))` for a valid, fresh stamp, `Ok(None)` for an expired one,
    /// `Err(reason)` for one that is not what its name says.
    fn verify(&self, key: &Key, text: &str) -> std::result::Result<Option<u64>, String> {
        let expected = format!("{FORMAT_LINE}\n{}", key.canonical());
        let rest = text
            .strip_prefix(&expected)
            .ok_or("it does not match this step, tree, toolchain and environment")?;
        let passed_at: u64 = rest
            .strip_prefix("passed-at: ")
            .and_then(|value| value.trim().parse().ok())
            .ok_or("its time stamp is unreadable")?;
        if passed_at > self.now.saturating_add(FUTURE_SLACK_SECS) {
            return Err("it is dated in the future".to_string());
        }
        let age = self.now.saturating_sub(passed_at);
        Ok((age <= MAX_AGE_SECS).then_some(age))
    }

    /// Record that the step passed. Returns a note when nothing was recorded.
    ///
    /// The tree is probed again first: a stamp is only true when the working
    /// tree is still the one the key was computed for, and a step that rewrote
    /// it (`Cargo.lock`, a generated file) did not check `HEAD`'s tree.
    pub fn record(&self, key: &Key) -> Option<String> {
        match self.probe.clean_tree() {
            Ok(tree) if tree == key.tree => {}
            Ok(_) => return Some("pass not recorded: HEAD moved while the step ran".to_string()),
            Err(why) => return Some(format!("pass not recorded: {why}")),
        }
        if let Err(error) = self
            .store
            .write(&key.file_name(), &key.stamp_text(self.now))
        {
            return Some(format!(
                "pass not recorded: could not write it under {} ({error:#})",
                self.store.location()
            ));
        }
        self.prune();
        None
    }

    /// Drop stamps past [`MAX_AGE_SECS`], then the oldest beyond [`MAX_STAMPS`],
    /// and scratch files left by a write that died ([`TMP_MAX_AGE_SECS`]).
    /// Best effort: a file that cannot be removed is only wasted space.
    fn prune(&self) {
        let Ok(mut entries) = self.store.list() else {
            return;
        };
        entries.retain(|(name, modified)| {
            let age = self.now.saturating_sub(*modified);
            let expired = if name.ends_with(TMP_EXT) {
                age > TMP_MAX_AGE_SECS
            } else {
                age > MAX_AGE_SECS
            };
            if expired {
                let _ = self.store.remove(name);
            }
            !expired
        });
        // A scratch file is not a pass: it does not count toward the cap.
        entries.retain(|(name, _)| !name.ends_with(TMP_EXT));
        entries.sort_by_key(|(_, modified)| std::cmp::Reverse(*modified));
        for (name, _) in entries.iter().skip(MAX_STAMPS) {
            let _ = self.store.remove(name);
        }
    }

    /// Where the stamps live, for the skip message.
    pub fn location(&self) -> String {
        self.store.location()
    }
}

/// `3 min`, `2 h`, `1 d`: how long ago a pass was recorded.
pub fn age_text(secs: u64) -> String {
    match secs {
        0..=59 => "under a minute".to_string(),
        60..=3599 => format!("{} min", secs / 60),
        3600..=86_399 => format!("{} h", secs / 3600),
        _ => format!("{} d", secs / 86_400),
    }
}

/// The lines printed when a step is skipped because its pass is on record.
pub fn skip_lines(
    step: &str,
    args: &[String],
    age_secs: u64,
    name: &str,
    location: &str,
) -> Vec<String> {
    vec![
        format!(
            "pre-push-check {step}: skipped, this tree already passed `cargo {}` {} ago \
             (stamp {name})",
            args.join(" "),
            age_text(age_secs)
        ),
        format!(
            "pre-push-check {step}: that does not cover your user-level cargo configuration, \
             installed tools, files git ignores or flaky tests; run it again with --rerun \
             or {RERUN_VAR}=1, or delete {location}"
        ),
    ]
}

// ---- the real world: git, the toolchain, the filesystem ----

/// A `git` command for the repository in `dir`. With an explicit directory the
/// repository-selecting variables are removed ([`affected::scrub_git_env`]): git
/// and prek export an absolute `GIT_DIR` to a pre-push hook run from a linked
/// worktree, and `git -C <dir>` would otherwise still act on the repository
/// being pushed. With `None` the environment is kept, because in the hook that
/// inherited `GIT_DIR` correctly names the repository being pushed.
fn git_command(dir: Option<&Path>) -> Command {
    let mut git = Command::new("git");
    if let Some(dir) = dir {
        affected::scrub_git_env(&mut git).arg("-C").arg(dir);
    }
    git
}

/// Run `git` with `args` (in `dir`, or the current directory) and return its
/// trimmed stdout.
fn git_stdout(dir: Option<&Path>, args: &[&str]) -> std::result::Result<String, String> {
    let output = git_command(dir)
        .args(args)
        .output()
        .map_err(|error| format!("could not run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// `<tool> -vV`, or the reason it could not run.
fn version_text(tool: &std::ffi::OsStr) -> std::result::Result<String, String> {
    let output = Command::new(tool)
        .arg("-vV")
        .output()
        .map_err(|error| format!("could not run {}: {error}", tool.to_string_lossy()))?;
    if !output.status.success() {
        return Err(format!("`{} -vV` failed", tool.to_string_lossy()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// [`Probe`] backed by the `git` and toolchain on `PATH`, for the repository in
/// `dir` (the current directory when `None`).
#[derive(Default)]
pub struct SystemProbe {
    pub dir: Option<PathBuf>,
}

impl Probe for SystemProbe {
    fn clean_tree(&self) -> std::result::Result<String, String> {
        let dir = self.dir.as_deref();
        let status = git_stdout(dir, &["status", "--porcelain", "--untracked-files=normal"])?;
        let dirty = status
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        if dirty > 0 {
            return Err(format!(
                "the working tree has {dirty} uncommitted or untracked path(s), and cargo checks \
                 the working tree rather than HEAD; commit or remove them (`git status` lists them)"
            ));
        }
        git_stdout(dir, &["rev-parse", HEAD_TREE])
    }

    fn toolchain(&self) -> std::result::Result<String, String> {
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        Ok(format!(
            "{}\n{}",
            version_text(std::ffi::OsStr::new("rustc"))?,
            version_text(&cargo)?
        ))
    }

    fn env_digest(&self) -> String {
        env_digest(std::env::vars())
    }
}

/// Stamps in a directory on disk.
pub struct FsStore {
    dir: PathBuf,
}

impl FsStore {
    /// The stamp directory under the git common directory of the repository in
    /// `dir` (the current directory when `None`), shared by every worktree of
    /// that clone.
    pub fn in_git_common_dir(dir: Option<&Path>) -> Result<Self> {
        let common = git_stdout(dir, &["rev-parse", "--git-common-dir"])
            .map_err(|why| anyhow::anyhow!(why))?;
        let common = PathBuf::from(common);
        // git prints the path relative to where it ran when it is inside there.
        let common = if common.is_absolute() {
            common
        } else {
            let base = match dir {
                Some(dir) => dir.to_path_buf(),
                None => std::env::current_dir().context("could not read the current directory")?,
            };
            base.join(common)
        };
        Ok(Self {
            dir: common.join(STAMP_DIR),
        })
    }

    fn path(&self, name: &str) -> Result<PathBuf> {
        if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
            bail!("refusing the stamp name {name:?}");
        }
        Ok(self.dir.join(name))
    }
}

impl Store for FsStore {
    fn read(&self, name: &str) -> Result<Option<String>> {
        match std::fs::read_to_string(self.path(name)?) {
            Ok(text) => Ok(Some(text)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).context("failed to read the stamp"),
        }
    }

    fn write(&self, name: &str, contents: &str) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("failed to create {}", self.dir.display()))?;
        let target = self.path(name)?;
        // Write beside the target and rename, so a reader never sees half a stamp.
        let scratch = self
            .dir
            .join(format!(".{name}.{}{TMP_EXT}", std::process::id()));
        let result = std::fs::write(&scratch, contents)
            .context("failed to write the stamp")
            .and_then(|()| {
                std::fs::rename(&scratch, &target).context("failed to move the stamp into place")
            });
        if result.is_err() {
            // Do not leave the scratch file behind; `prune` only sees what a
            // later successful write lists.
            let _ = std::fs::remove_file(&scratch);
        }
        result
    }

    fn list(&self) -> Result<Vec<(String, u64)>> {
        let mut entries = Vec::new();
        let read_dir = match std::fs::read_dir(&self.dir) {
            Ok(read_dir) => read_dir,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
            Err(error) => return Err(error).context("failed to list the stamps"),
        };
        for entry in read_dir.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with(STAMP_EXT) && !name.ends_with(TMP_EXT) {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |since| since.as_secs());
            entries.push((name, modified));
        }
        Ok(entries)
    }

    fn remove(&self, name: &str) -> Result<()> {
        match std::fs::remove_file(self.path(name)?) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).context("failed to remove the stamp"),
        }
    }

    fn location(&self) -> String {
        self.dir.display().to_string()
    }
}

/// A store that could not be set up: every read and write reports why, so the
/// step runs and says so.
pub struct UnavailableStore(pub String);

impl Store for UnavailableStore {
    fn read(&self, _name: &str) -> Result<Option<String>> {
        bail!("{}", self.0)
    }
    fn write(&self, _name: &str, _contents: &str) -> Result<()> {
        bail!("{}", self.0)
    }
    fn list(&self) -> Result<Vec<(String, u64)>> {
        bail!("{}", self.0)
    }
    fn remove(&self, _name: &str) -> Result<()> {
        bail!("{}", self.0)
    }
    fn location(&self) -> String {
        "the stamp directory".to_string()
    }
}

/// Seconds since the epoch, or 0 when the clock is before it (which makes every
/// stamp look dated in the future, so none is honored).
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

#[cfg(test)]
pub(crate) mod fakes {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use anyhow::{Result, bail};

    use super::{Probe, Store};

    /// A probe whose answers a test sets, and which counts its calls.
    pub struct FakeProbe {
        pub tree: RefCell<std::result::Result<String, String>>,
        pub toolchain: RefCell<std::result::Result<String, String>>,
        pub env: RefCell<String>,
    }

    impl FakeProbe {
        pub fn clean(tree: &str) -> Self {
            Self {
                tree: RefCell::new(Ok(tree.to_string())),
                toolchain: RefCell::new(Ok("rustc 1.96.0\ncargo 1.96.0\n".to_string())),
                env: RefCell::new("env-a".to_string()),
            }
        }
    }

    impl Probe for FakeProbe {
        fn clean_tree(&self) -> std::result::Result<String, String> {
            self.tree.borrow().clone()
        }
        fn toolchain(&self) -> std::result::Result<String, String> {
            self.toolchain.borrow().clone()
        }
        fn env_digest(&self) -> String {
            self.env.borrow().clone()
        }
    }

    /// An in-memory store; `fail_writes` makes every write error.
    #[derive(Default)]
    pub struct MemStore {
        pub files: RefCell<BTreeMap<String, (String, u64)>>,
        pub fail_writes: RefCell<bool>,
        pub writes: RefCell<usize>,
    }

    impl Store for MemStore {
        fn read(&self, name: &str) -> Result<Option<String>> {
            Ok(self.files.borrow().get(name).map(|(text, _)| text.clone()))
        }
        fn write(&self, name: &str, contents: &str) -> Result<()> {
            if *self.fail_writes.borrow() {
                bail!("read-only file system");
            }
            *self.writes.borrow_mut() += 1;
            // The mtime is the stamp's own `passed-at`, as on a real disk.
            let modified = contents
                .lines()
                .find_map(|line| line.strip_prefix("passed-at: "))
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            self.files
                .borrow_mut()
                .insert(name.to_string(), (contents.to_string(), modified));
            Ok(())
        }
        fn list(&self) -> Result<Vec<(String, u64)>> {
            Ok(self
                .files
                .borrow()
                .iter()
                .map(|(name, (_, modified))| (name.clone(), *modified))
                .collect())
        }
        fn remove(&self, name: &str) -> Result<()> {
            self.files.borrow_mut().remove(name);
            Ok(())
        }
        fn location(&self) -> String {
            "<memory>".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::{FakeProbe, MemStore};
    use super::*;

    const NOW: u64 = 1_800_000_000;

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    fn stamps<'a>(probe: &'a FakeProbe, store: &'a MemStore, now: u64) -> Stamps<'a> {
        Stamps {
            probe,
            store,
            now,
            rerun: false,
        }
    }

    /// Record a pass of `test -p xtask` and return the stamp's file name.
    fn record_pass(probe: &FakeProbe, store: &MemStore, now: u64) -> String {
        let stamps = stamps(probe, store, now);
        let Lookup::Miss { key: Some(key), .. } = stamps.lookup("test", &args("test -p xtask"))
        else {
            panic!("expected a miss with a key");
        };
        assert_eq!(stamps.record(&key), None);
        key.file_name()
    }

    // ---- the key ----

    fn base_key() -> Key {
        Key {
            step: "test".to_string(),
            args: "test -p xtask".to_string(),
            tree: "tree".to_string(),
            toolchain: "toolchain".to_string(),
            env: "env".to_string(),
        }
    }

    #[test]
    fn every_part_of_the_key_changes_the_stamp_name() {
        let base = base_key();
        let variants = [
            Key {
                step: "clippy".into(),
                ..base.clone()
            },
            Key {
                args: "test --workspace".into(),
                ..base.clone()
            },
            Key {
                tree: "other".into(),
                ..base.clone()
            },
            Key {
                toolchain: "other".into(),
                ..base.clone()
            },
            Key {
                env: "other".into(),
                ..base.clone()
            },
        ];
        for variant in &variants {
            assert_ne!(variant.file_name(), base.file_name(), "{variant:?}");
        }
        assert_eq!(
            base.file_name(),
            base_key().file_name(),
            "the name is stable"
        );
        let name = base.file_name();
        assert!(
            name.starts_with("test-") && name.strip_suffix(".stamp").is_some(),
            "{name}"
        );
    }

    // ---- age: expiry and a clock that disagrees ----

    #[test]
    fn a_pass_is_honored_up_to_the_maximum_age_and_not_a_second_longer() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        record_pass(&probe, &store, NOW);
        let at = |now| stamps(&probe, &store, now).lookup("test", &args("test -p xtask"));
        assert!(matches!(at(NOW + MAX_AGE_SECS), Lookup::Hit { .. }));
        // Expired: a quiet miss that is allowed to record a fresh pass.
        match at(NOW + MAX_AGE_SECS + 1) {
            Lookup::Miss { key, note } => {
                assert!(key.is_some());
                assert_eq!(note, None);
            }
            Lookup::Hit { .. } => panic!("an expired pass must not skip the step"),
        }
    }

    #[test]
    fn a_stamp_dated_in_the_future_is_ignored_and_says_so() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        let name = record_pass(&probe, &store, NOW + 10 * 60);
        match stamps(&probe, &store, NOW).lookup("test", &args("test -p xtask")) {
            Lookup::Miss { note, .. } => {
                let note = note.expect("a note");
                assert!(
                    note.contains(&name) && note.contains("dated in the future"),
                    "{note}"
                );
            }
            Lookup::Hit { .. } => panic!("a future-dated stamp must not skip the step"),
        }
        // A little clock skew is tolerated.
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        record_pass(&probe, &store, NOW + 60);
        assert!(matches!(
            stamps(&probe, &store, NOW).lookup("test", &args("test -p xtask")),
            Lookup::Hit { .. }
        ));
    }

    #[test]
    fn a_stamp_from_another_format_or_with_a_bad_time_is_ignored() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        let name = record_pass(&probe, &store, NOW);
        let good = store.files.borrow()[&name].0.clone();
        for (broken, why) in [
            (
                good.replace("pass stamp v1", "pass stamp v2"),
                "does not match",
            ),
            (good.replace("tree: t", "tree: u"), "does not match"),
            (
                good.replace(&format!("passed-at: {NOW}"), "passed-at: soon"),
                "time stamp is unreadable",
            ),
            (
                good.replace(&format!("passed-at: {NOW}\n"), ""),
                "time stamp is unreadable",
            ),
        ] {
            store.files.borrow_mut().insert(name.clone(), (broken, NOW));
            match stamps(&probe, &store, NOW + 1).lookup("test", &args("test -p xtask")) {
                Lookup::Miss { note, .. } => {
                    let note = note.expect("a note");
                    assert!(note.contains(why), "{note}");
                }
                Lookup::Hit { .. } => panic!("a broken stamp must not skip the step: {why}"),
            }
        }
    }

    // ---- expiry and the size cap ----

    #[test]
    fn recording_prunes_expired_stamps_and_keeps_the_newest_up_to_the_cap() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        {
            let mut files = store.files.borrow_mut();
            files.insert(
                "old-a.stamp".into(),
                (String::new(), NOW - MAX_AGE_SECS - 10),
            );
            files.insert(
                "old-b.stamp".into(),
                (String::new(), NOW - MAX_AGE_SECS - 20),
            );
            for index in 0..(MAX_STAMPS + 5) {
                files.insert(
                    format!("fresh-{index:03}.stamp"),
                    (String::new(), NOW - 1000 + index as u64),
                );
            }
        }
        record_pass(&probe, &store, NOW);
        let files = store.files.borrow();
        assert_eq!(
            files.len(),
            MAX_STAMPS,
            "{:?}",
            files.keys().collect::<Vec<_>>()
        );
        assert!(!files.contains_key("old-a.stamp") && !files.contains_key("old-b.stamp"));
        assert!(
            !files.contains_key("fresh-000.stamp"),
            "the oldest beyond the cap go first"
        );
        assert!(
            files.contains_key(&format!("fresh-{:03}.stamp", MAX_STAMPS + 4)),
            "the newest stay"
        );
        assert!(
            files.keys().any(|name| name.starts_with("test-")),
            "and so does the new pass"
        );
    }

    #[test]
    fn expired_stamps_are_pruned_even_when_the_store_is_far_below_the_cap() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        {
            let mut files = store.files.borrow_mut();
            files.insert("old.stamp".into(), (String::new(), NOW - MAX_AGE_SECS - 1));
            files.insert("edge.stamp".into(), (String::new(), NOW - MAX_AGE_SECS));
            files.insert("fresh.stamp".into(), (String::new(), NOW - 60));
        }
        record_pass(&probe, &store, NOW);
        let files = store.files.borrow();
        assert!(
            !files.contains_key("old.stamp"),
            "{:?}",
            files.keys().collect::<Vec<_>>()
        );
        assert!(
            files.contains_key("edge.stamp"),
            "a stamp exactly at the limit is still honored"
        );
        assert!(files.contains_key("fresh.stamp"));
    }

    // ---- the environment ----

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn the_environment_digest_follows_cargo_and_rust_settings_only() {
        let base = env_digest(vars(&[("PATH", "/bin"), ("RUSTFLAGS", "-D warnings")]));
        // Settings that can change a step's outcome change the digest ...
        assert_ne!(
            base,
            env_digest(vars(&[("RUSTFLAGS", "-C target-cpu=native")]))
        );
        assert_ne!(
            base,
            env_digest(vars(&[
                ("RUSTFLAGS", "-D warnings"),
                ("CARGO_BUILD_TARGET", "x")
            ]))
        );
        // ... unrelated or per-run variables do not, whatever their order.
        assert_eq!(
            base,
            env_digest(vars(&[
                ("HOME", "/home/a"),
                ("CARGO_TARGET_DIR", "/elsewhere"),
                ("CARGO_TERM_COLOR", "always"),
                ("CARGO_PKG_NAME", "xtask"),
                ("CARGO_MANIFEST_DIR", "/repo/xtask"),
                ("RUST_BACKTRACE", "1"),
                ("RUSTFLAGS", "-D warnings"),
            ]))
        );
        assert_eq!(
            env_digest(vars(&[("RUSTFLAGS", "a"), ("CARGO_INCREMENTAL", "0")])),
            env_digest(vars(&[("CARGO_INCREMENTAL", "0"), ("RUSTFLAGS", "a")])),
        );
    }

    #[test]
    fn no_environment_value_ends_up_in_a_stamp() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        *probe.env.borrow_mut() = env_digest(vars(&[("CARGO_REGISTRY_TOKEN", "hunter2-secret")]));
        let name = record_pass(&probe, &store, NOW);
        let text = store.files.borrow()[&name].0.clone();
        assert!(
            !text.contains("hunter2") && !text.contains("CARGO_REGISTRY_TOKEN"),
            "{text}"
        );
    }

    #[test]
    fn rerun_comes_from_the_flag_or_a_non_empty_non_zero_variable() {
        let var = |value: Option<&'static str>| {
            move |name: &str| {
                (name == RERUN_VAR)
                    .then(|| value.map(String::from))
                    .flatten()
            }
        };
        assert!(rerun_requested(true, var(None)));
        assert!(rerun_requested(false, var(Some("1"))));
        assert!(rerun_requested(false, var(Some("yes"))));
        assert!(!rerun_requested(false, var(Some("0"))));
        assert!(!rerun_requested(false, var(Some(""))));
        assert!(!rerun_requested(false, var(Some("  "))));
        assert!(!rerun_requested(false, var(None)));
    }

    #[test]
    fn ages_read_as_minutes_hours_or_days() {
        assert_eq!(age_text(5), "under a minute");
        assert_eq!(age_text(90), "1 min");
        assert_eq!(age_text(2 * 3600 + 5), "2 h");
        assert_eq!(age_text(3 * 86_400), "3 d");
    }

    // ---- the real store on a real disk ----

    #[test]
    fn the_file_store_round_trips_lists_only_stamps_and_removes_idempotently() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FsStore {
            dir: dir.path().join("passes"),
        };
        assert_eq!(
            store.read("a.stamp").expect("read"),
            None,
            "a missing stamp is not an error"
        );
        assert!(
            store.list().expect("list").is_empty(),
            "nor is a missing directory"
        );

        store.write("a.stamp", "one").expect("write");
        store
            .write("a.stamp", "two")
            .expect("an overwrite replaces it");
        assert_eq!(store.read("a.stamp").expect("read").as_deref(), Some("two"));
        std::fs::write(dir.path().join("passes/notes.txt"), "x").expect("a foreign file");

        let listed = store.list().expect("list");
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].0, "a.stamp");
        assert!(
            listed[0].1 > 1_000_000_000,
            "the modified time is read: {listed:?}"
        );
        let leftovers = std::fs::read_dir(dir.path().join("passes"))
            .expect("dir")
            .count();
        assert_eq!(
            leftovers, 2,
            "no scratch file is left behind (the stamp and the foreign file)"
        );

        store.remove("a.stamp").expect("remove");
        store.remove("a.stamp").expect("removing it again is fine");
        assert_eq!(store.read("a.stamp").expect("read"), None);
    }

    #[test]
    fn the_file_store_refuses_names_that_leave_its_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FsStore {
            dir: dir.path().join("passes"),
        };
        for name in ["../escape.stamp", "a/b.stamp", "a\\b.stamp", "..", ".", ""] {
            // The guard itself refuses, not whatever the filesystem then does.
            let refusals = [
                store.write(name, "x").unwrap_err(),
                store.read(name).unwrap_err(),
                store.remove(name).unwrap_err(),
            ];
            for error in refusals {
                assert!(
                    format!("{error:#}").contains("refusing the stamp name"),
                    "{name}: {error:#}"
                );
            }
        }
        assert!(!dir.path().join("escape.stamp").exists());
    }

    // ---- git: the tree hash, the dirty check and the shared location ----

    fn git_in(dir: &Path, args: &[&str]) -> String {
        let output = git_command(Some(dir))
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed in {}",
            dir.display()
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    /// A repository with one commit, in a fresh directory.
    fn repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        git_in(dir.path(), &["init", "-q"]);
        std::fs::write(dir.path().join("a.txt"), "one\n").expect("write");
        git_in(dir.path(), &["add", "a.txt"]);
        git_in(dir.path(), &["commit", "-q", "-m", "first"]);
        dir
    }

    fn probe_for(dir: &Path) -> SystemProbe {
        SystemProbe {
            dir: Some(dir.to_path_buf()),
        }
    }

    #[test]
    fn a_clean_tree_is_keyed_by_its_tree_hash_which_an_amend_does_not_change() {
        let repo = repo();
        let probe = probe_for(repo.path());
        let tree = probe.clean_tree().expect("a clean tree");
        assert_eq!(tree, git_in(repo.path(), &["rev-parse", HEAD_TREE]));

        let commit = git_in(repo.path(), &["rev-parse", "HEAD"]);
        git_in(repo.path(), &["commit", "-q", "--amend", "-m", "reworded"]);
        assert_ne!(
            git_in(repo.path(), &["rev-parse", "HEAD"]),
            commit,
            "the commit hash changed"
        );
        assert_eq!(
            probe.clean_tree().expect("still clean"),
            tree,
            "the tree hash did not"
        );

        std::fs::write(repo.path().join("a.txt"), "two\n").expect("write");
        git_in(repo.path(), &["commit", "-q", "-am", "second"]);
        assert_ne!(
            probe.clean_tree().expect("clean"),
            tree,
            "new content is a new tree"
        );
    }

    #[test]
    fn uncommitted_or_untracked_files_make_the_tree_unusable_for_stamps() {
        let repo = repo();
        let probe = probe_for(repo.path());
        probe.clean_tree().expect("clean to start with");

        std::fs::write(repo.path().join("a.txt"), "changed\n").expect("write");
        let why = probe.clean_tree().expect_err("a modified tracked file");
        assert!(why.contains("1 uncommitted or untracked path"), "{why}");
        assert!(
            why.contains("git status"),
            "the reason says how to fix it: {why}"
        );
        git_in(repo.path(), &["checkout", "--", "a.txt"]);
        probe.clean_tree().expect("clean again");

        // An untracked file can hide a missing `git add`, so it counts too.
        std::fs::write(repo.path().join("new.rs"), "mod x;\n").expect("write");
        assert!(probe.clean_tree().is_err());
    }

    #[test]
    fn linked_worktrees_share_one_stamp_directory_outside_every_work_tree() {
        let repo = repo();
        let linked = tempfile::tempdir().expect("tempdir");
        let linked_path = linked.path().join("wt");
        git_in(
            repo.path(),
            &[
                "worktree",
                "add",
                "-q",
                linked_path.to_str().expect("utf-8"),
                "-b",
                "side",
            ],
        );

        let main = FsStore::in_git_common_dir(Some(repo.path())).expect("store");
        let side = FsStore::in_git_common_dir(Some(&linked_path)).expect("store");
        let canonical = |store: &FsStore| {
            std::fs::canonicalize(store.dir.parent().expect("parent")).expect("exists")
        };
        assert_eq!(
            canonical(&main),
            canonical(&side),
            "one directory for the whole clone"
        );
        assert!(main.dir.ends_with(STAMP_DIR));
        assert!(canonical(&main).ends_with(".git"), "{:?}", canonical(&main));

        // A stamp written from one worktree is read from the other, and writing
        // it does not dirty either work tree.
        main.write("t.stamp", "shared").expect("write");
        assert_eq!(
            side.read("t.stamp").expect("read").as_deref(),
            Some("shared")
        );
        probe_for(repo.path())
            .clean_tree()
            .expect("the main work tree is still clean");
        probe_for(&linked_path)
            .clean_tree()
            .expect("so is the linked one");
    }

    #[test]
    fn a_directory_git_does_not_know_has_no_store_and_no_usable_tree() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The temp directory is not normally inside a repository; if it were,
        // git would answer and there would be nothing to assert.
        if git_stdout(Some(dir.path()), &["rev-parse", "--git-dir"]).is_err() {
            assert!(probe_for(dir.path()).clean_tree().is_err());
            assert!(FsStore::in_git_common_dir(Some(dir.path())).is_err());
        }
    }

    #[test]
    fn the_toolchain_is_identified_from_rustc_and_cargo() {
        let text = SystemProbe::default()
            .toolchain()
            .expect("rustc and cargo run in the test environment");
        assert!(text.contains("rustc") && text.contains("cargo"), "{text}");
    }

    // ---- hermetic git: a hostile ambient GIT_DIR must not redirect any of it ----

    const HOSTILE_CHILD_VAR: &str = "ROCM_STAMP_TEST_HOSTILE_CHILD";

    #[test]
    fn an_explicit_directory_drops_the_repository_selecting_environment() {
        let with_dir = git_command(Some(Path::new("scratch")));
        let envs: Vec<_> = with_dir.get_envs().collect();
        for var in [
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
        ] {
            assert!(
                envs.iter()
                    .any(|(name, value)| name.to_str() == Some(var) && value.is_none()),
                "{var} must be removed for a command pointed at a directory"
            );
        }
        // Without a directory the hook's own environment is kept: there GIT_DIR
        // correctly names the repository being pushed.
        assert_eq!(git_command(None).get_envs().count(), 0);
    }

    /// Does nothing on its own. The test below re-runs this binary with a hostile
    /// git environment and this variable set, and this body then drives the real
    /// probe, store and test helpers against a scratch repository.
    #[test]
    fn hostile_environment_child() {
        let Some(dir) = std::env::var_os(HOSTILE_CHILD_VAR) else {
            return;
        };
        let dir = PathBuf::from(dir);
        assert!(
            std::env::var_os("GIT_DIR").is_some(),
            "the child is supposed to run under a hostile GIT_DIR"
        );
        let probe = probe_for(&dir);
        let tree = git_in(&dir, &["rev-parse", HEAD_TREE]);
        assert_eq!(
            probe.clean_tree().expect("the scratch repository's tree"),
            tree
        );

        // Committing through the helper lands in the scratch repository.
        std::fs::write(dir.join("b.txt"), "two\n").expect("write");
        git_in(&dir, &["add", "b.txt"]);
        git_in(&dir, &["commit", "-q", "-m", "second"]);
        assert_ne!(probe.clean_tree().expect("clean"), tree);

        let store = FsStore::in_git_common_dir(Some(&dir)).expect("store");
        let scratch = std::fs::canonicalize(&dir).expect("canonical");
        let stamp_parent =
            std::fs::canonicalize(store.dir.parent().expect("parent")).expect("exists");
        assert!(
            stamp_parent.starts_with(&scratch),
            "the store must be in the scratch repository, not the outer one: {stamp_parent:?}"
        );
        println!("hostile-child-ran");
    }

    #[test]
    fn git_driven_behaviour_ignores_a_hostile_git_environment() {
        let outer = repo();
        let inner = repo();
        let outer_git = outer.path().join(".git");
        let outer_head = git_in(outer.path(), &["rev-parse", "HEAD"]);

        let output = Command::new(std::env::current_exe().expect("this test binary"))
            .args([
                "--exact",
                "prepush_stamp::tests::hostile_environment_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(HOSTILE_CHILD_VAR, inner.path())
            .env("GIT_DIR", &outer_git)
            .env("GIT_WORK_TREE", outer.path())
            .env("GIT_INDEX_FILE", outer_git.join("index"))
            .env("GIT_COMMON_DIR", &outer_git)
            .output()
            .expect("re-run the test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}\n{stderr}");
        assert!(
            stdout.contains("hostile-child-ran"),
            "the child did not run its body: {stdout}"
        );

        // The outer repository is exactly as it was: no stray commit, nothing staged.
        assert_eq!(git_in(outer.path(), &["rev-parse", "HEAD"]), outer_head);
        assert_eq!(git_in(outer.path(), &["rev-list", "--count", "HEAD"]), "1");
        assert_eq!(git_in(outer.path(), &["status", "--porcelain"]), "");
    }

    // ---- scratch files left by a write that died ----

    #[test]
    fn a_write_that_cannot_be_moved_into_place_leaves_no_scratch_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FsStore {
            dir: dir.path().join("passes"),
        };
        // A non-empty directory where the stamp should go makes the rename fail.
        std::fs::create_dir_all(store.dir.join("a.stamp")).expect("mkdir");
        std::fs::write(store.dir.join("a.stamp").join("inner"), "x").expect("write");

        let error = store
            .write("a.stamp", "x")
            .expect_err("a directory is in the way");
        assert!(
            format!("{error:#}").contains("failed to move the stamp into place"),
            "{error:#}"
        );
        let names: Vec<String> = std::fs::read_dir(&store.dir)
            .expect("dir")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            ["a.stamp"],
            "no scratch file is left behind: {names:?}"
        );
    }

    #[test]
    fn stale_scratch_files_are_pruned_and_fresh_ones_kept_and_none_count_toward_the_cap() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        {
            let mut files = store.files.borrow_mut();
            files.insert(
                ".a.stamp.1.tmp".into(),
                (String::new(), NOW - TMP_MAX_AGE_SECS - 1),
            );
            files.insert(
                ".b.stamp.2.tmp".into(),
                (String::new(), NOW - TMP_MAX_AGE_SECS),
            );
            files.insert(".c.stamp.3.tmp".into(), (String::new(), NOW - 60));
            for index in 0..MAX_STAMPS {
                files.insert(
                    format!("fresh-{index:03}.stamp"),
                    (String::new(), NOW - 1000 + index as u64),
                );
            }
        }
        record_pass(&probe, &store, NOW);
        let files = store.files.borrow();
        let names: Vec<&String> = files.keys().collect();
        assert!(
            !files.contains_key(".a.stamp.1.tmp"),
            "a dead writer's scratch file goes: {names:?}"
        );
        assert!(
            files.contains_key(".b.stamp.2.tmp"),
            "one exactly at the bound stays"
        );
        assert!(
            files.contains_key(".c.stamp.3.tmp"),
            "a young one may be a live writer's"
        );
        let stamps = files
            .keys()
            .filter(|name| name.ends_with(STAMP_EXT))
            .count();
        assert_eq!(
            stamps, MAX_STAMPS,
            "scratch files do not push stamps out: {names:?}"
        );
    }

    #[test]
    fn a_real_store_prunes_a_dead_writers_scratch_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FsStore {
            dir: dir.path().join("passes"),
        };
        std::fs::create_dir_all(&store.dir).expect("mkdir");
        let now = now_secs();
        let age_of = |name: &str, secs: u64| {
            let path = store.dir.join(name);
            std::fs::write(&path, "x").expect("write");
            std::fs::File::options()
                .write(true)
                .open(&path)
                .expect("open")
                .set_modified(SystemTime::now() - std::time::Duration::from_secs(secs))
                .expect("set the modified time");
            path
        };
        let dead = age_of(".a.stamp.1.tmp", 2 * 60 * 60);
        let live = age_of(".b.stamp.2.tmp", 60);

        let probe = FakeProbe::clean("t");
        let stamps = Stamps::new(&probe, &store, now, false, |_| None);
        let Lookup::Miss { key: Some(key), .. } = stamps.lookup("test", &args("test -p xtask"))
        else {
            panic!("expected a miss with a key");
        };
        assert_eq!(stamps.record(&key), None);
        assert!(
            !dead.exists(),
            "the dead writer's scratch file is removed from disk"
        );
        assert!(live.exists(), "the young one is left alone");
    }

    // ---- Stamps::new: the --rerun flag and the variable reach the lookup ----

    #[test]
    fn stamps_new_reads_the_flag_and_the_variable() {
        let (probe, store) = (FakeProbe::clean("t"), MemStore::default());
        let var =
            |value: &'static str| move |name: &str| (name == RERUN_VAR).then(|| value.to_string());
        assert!(Stamps::new(&probe, &store, NOW, true, |_| None).rerun);
        assert!(Stamps::new(&probe, &store, NOW, false, var("1")).rerun);
        assert!(!Stamps::new(&probe, &store, NOW, false, var("0")).rerun);
        assert!(!Stamps::new(&probe, &store, NOW, false, |_| None).rerun);
    }
}
