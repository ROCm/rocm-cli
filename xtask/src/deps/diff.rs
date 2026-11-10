// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps diff`: a Markdown table of every version that changed in
//! the runtime locks since a git revision. The update workflow uses it as the
//! body of the pull request it opens.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

use super::mise_lock::{self, MISE_LOCK};
use super::profiles::{LOCK_DIR, Profiles, normalize_name, slash_path};
use super::pylock::PyLock;

/// The whole summary when nothing moved. `scripts/validate_runtime_update.py`
/// accepts exactly this, and `the_summary_passes_the_update_validator` ties
/// the two together.
pub(crate) const NO_CHANGES: &str = "No runtime versions changed.";

/// `(lock, package) -> version`.
type Versions = BTreeMap<(String, String), String>;

pub(crate) fn run(base: &str) -> Result<()> {
    let root = crate::paths::workspace_root()?;
    print!("{}", summary(&root, base)?);
    Ok(())
}

/// The table of versions changed in `root`'s runtime locks since `base`.
fn summary(root: &Path, base: &str) -> Result<String> {
    let profiles = Profiles::load(root)?;
    let base = resolve_commit(root, base)?;
    // The locks the profiles produce now, plus any the base had: a profile
    // or OS the change removes must show up as removed, not drop out.
    let mut lock_files = std::collections::BTreeSet::from([MISE_LOCK.to_owned()]);
    lock_files.extend(
        profiles
            .targets()
            .map(|(name, _, os, _)| slash_path(&Profiles::lock_path(name, os))),
    );
    lock_files.extend(files_at(root, &base, LOCK_DIR)?);

    let mut before = Versions::new();
    let mut after = Versions::new();
    for file in &lock_files {
        if let Some(text) = file_at(root, &base, file)? {
            collect(file, &text, &mut before)?;
        }
        if let Some(text) = file_now(root, file)? {
            collect(file, &text, &mut after)?;
        }
    }
    Ok(render(&before, &after))
}

/// The files directly under `dir` at `commit` (none if it has no such dir).
/// The lock directory holds nothing but locks (`deps check` refuses anything
/// else), so every file there is read as one.
fn files_at(root: &Path, commit: &str, dir: &str) -> Result<Vec<String>> {
    let output = git(
        root,
        &["ls-tree", "--name-only", commit, "--", &format!("{dir}/")],
    )?;
    ensure!(
        output.status.success(),
        "git ls-tree {commit} {dir} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

fn git(root: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .context("failed to run git")
}

/// The commit `base` names. A revision that does not resolve is an error,
/// never "nothing was there": that would list every locked version as new.
fn resolve_commit(root: &Path, base: &str) -> Result<String> {
    let output = git(
        root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{base}^{{commit}}"),
        ],
    )?;
    ensure!(output.status.success(), "{base:?} does not name a commit");
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// `file` as it was at `commit`, or `None` when `commit` has no such path.
/// Any other git failure is an error. `commit` must already be verified with
/// [`resolve_commit`]: git reports a well-formed id that names no commit with
/// the same "does not exist in" message as a missing path.
fn file_at(root: &Path, commit: &str, file: &str) -> Result<Option<String>> {
    let output = Command::new("git")
        .current_dir(root)
        .env("LC_ALL", "C")
        .args(["show", &format!("{commit}:{file}")])
        .output()
        .context("failed to run git")?;
    if output.status.success() {
        return Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned()));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if is_missing_path(&stderr) {
        return Ok(None);
    }
    bail!("git show {commit}:{file} failed: {}", stderr.trim())
}

/// Whether `git show <rev>:<path>` failed only because `<rev>` has no `<path>`.
fn is_missing_path(stderr: &str) -> bool {
    stderr.contains("does not exist in") || stderr.contains("exists on disk, but not in")
}

/// `file` in the working tree, or `None` when it does not exist there (a lock
/// the change removes). Any other read failure is an error.
fn file_now(root: &Path, file: &str) -> Result<Option<String>> {
    match std::fs::read_to_string(root.join(file)) {
        Ok(text) => Ok(Some(text)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).with_context(|| format!("failed to read {file}")),
    }
}

/// Read one lock's versions. A lock that does not parse is an error: reading
/// it as empty would report every package in it as added or removed.
fn collect(file: &str, text: &str, into: &mut Versions) -> Result<()> {
    if file == MISE_LOCK {
        // One row per platform, with the digest's first characters after the
        // version, so a digest swapped at an unchanged version still shows up
        // in the table the update pull request is opened with. A URL changed
        // at the same digest does not; `deps check` binds the URL instead.
        for (name, tool) in
            mise_lock::parse_lock(text).with_context(|| format!("invalid {file}"))?
        {
            for (platform, asset) in &tool.platforms {
                let digest = asset.sha256.get(..12).unwrap_or(&asset.sha256);
                into.insert(
                    (file.to_owned(), format!("{name}/{platform}")),
                    format!("{}-{digest}", tool.version),
                );
            }
        }
    } else {
        for package in PyLock::parse(text)
            .with_context(|| format!("invalid {file}"))?
            .packages
        {
            // `deps check` refuses both of these; the table must not hide
            // them by dropping the row or keeping only one of two.
            let Some(version) = package.version else {
                bail!("{file}: {} is locked without a version", package.name);
            };
            // As for binaries: a re-spun artifact at an unchanged version
            // changes the cell, so the update's table shows it.
            let cell = match artifacts_digest(&package.wheels, package.sdist.as_ref()) {
                Some(digest) => format!("{version}-{digest}"),
                None => version,
            };
            let key = (file.to_owned(), normalize_name(&package.name));
            if into.insert(key, cell).is_some() {
                bail!("{file}: {} is locked more than once", package.name);
            }
        }
    }
    Ok(())
}

/// The first 12 hex digits of a SHA-256 over every artifact's sha256, in a
/// fixed order; `None` for a package with no hashed artifact.
fn artifacts_digest(
    wheels: &[crate::deps::pylock::Artifact],
    sdist: Option<&crate::deps::pylock::Artifact>,
) -> Option<String> {
    use sha2::Digest as _;
    let mut hashes: Vec<&str> = wheels
        .iter()
        .chain(sdist)
        .filter_map(|a| a.hashes.get("sha256").map(String::as_str))
        .collect();
    if hashes.is_empty() {
        return None;
    }
    hashes.sort_unstable();
    let digest = sha2::Sha256::digest(hashes.join("\n").as_bytes());
    Some(digest.iter().take(6).fold(String::new(), |mut hex, byte| {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
        hex
    }))
}

fn render(before: &Versions, after: &Versions) -> String {
    let mut rows = Vec::new();
    let keys: std::collections::BTreeSet<_> = before.keys().chain(after.keys()).collect();
    for key in keys {
        let old = before.get(key).map_or("—", String::as_str);
        let new = after.get(key).map_or("—", String::as_str);
        if old != new {
            let lock = key.0.rsplit('/').next().unwrap_or(&key.0);
            rows.push(format!("| `{lock}` | `{}` | {old} | {new} |", key.1));
        }
    }
    if rows.is_empty() {
        return format!("{NO_CHANGES}\n");
    }
    format!(
        "| Lock | Package | Before | After |\n|---|---|---|---|\n{}\n",
        rows.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions(rows: &[(&str, &str, &str)]) -> Versions {
        rows.iter()
            .map(|(lock, name, version)| {
                (
                    ((*lock).to_owned(), (*name).to_owned()),
                    (*version).to_owned(),
                )
            })
            .collect()
    }

    #[test]
    fn only_changed_added_and_removed_versions_are_listed() {
        let before = versions(&[
            ("runtimes/mise.lock", "uv", "0.13.6"),
            (
                "runtimes/locks/pylock.p-linux.toml",
                "torch",
                "2.13.0+rocm10.1.0",
            ),
            ("runtimes/locks/pylock.p-linux.toml", "gone", "1.0"),
        ]);
        let after = versions(&[
            ("runtimes/mise.lock", "uv", "0.13.7"),
            (
                "runtimes/locks/pylock.p-linux.toml",
                "torch",
                "2.13.0+rocm10.1.0",
            ),
            ("runtimes/locks/pylock.p-linux.toml", "new", "2.0"),
        ]);
        let table = render(&before, &after);
        assert!(
            table.contains("| `mise.lock` | `uv` | 0.13.6 | 0.13.7 |"),
            "{table}"
        );
        assert!(
            table.contains("| `pylock.p-linux.toml` | `gone` | 1.0 | — |"),
            "{table}"
        );
        assert!(
            table.contains("| `pylock.p-linux.toml` | `new` | — | 2.0 |"),
            "{table}"
        );
        assert!(!table.contains("torch"), "{table}");
    }

    #[test]
    fn no_change_says_so() {
        let same = versions(&[("runtimes/mise.lock", "uv", "0.13.6")]);
        assert_eq!(render(&same, &same), format!("{NO_CHANGES}\n"));
    }

    fn git_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .current_dir(dir.path())
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?}");
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "t@example.invalid"]);
        run(&["config", "user.name", "t"]);
        run(&["config", "commit.gpgsign", "false"]);
        std::fs::write(dir.path().join("a.toml"), "one\n").unwrap();
        run(&["add", "a.toml"]);
        run(&["commit", "-q", "-m", "a"]);
        dir
    }

    #[test]
    fn a_revision_that_does_not_resolve_is_an_error() {
        let repo = git_repo();
        assert!(resolve_commit(repo.path(), "no-such-rev").is_err());
        assert!(resolve_commit(repo.path(), "HEAD").is_ok());
    }

    #[test]
    fn a_path_missing_at_the_revision_is_none_and_a_present_one_is_read() {
        let repo = git_repo();
        let head = resolve_commit(repo.path(), "HEAD").unwrap();
        assert_eq!(
            file_at(repo.path(), &head, "a.toml").unwrap().as_deref(),
            Some("one\n")
        );
        assert_eq!(file_at(repo.path(), &head, "missing.toml").unwrap(), None);
    }

    #[test]
    fn mise_lock_versions_are_collected_per_tool() {
        let sum = "0123456789abcdef".repeat(4);
        let text = format!(
            "lockfile_version = 3\n\n[[tools.\"aqua:astral-sh/uv\"]]\nversion = \"0.13.6\"\nbackend = \"aqua:astral-sh/uv\"\n\n\
             [tools.\"aqua:astral-sh/uv\".\"platforms.linux-x64\"]\nchecksum = \"sha256:{sum}\"\nurl = \"https://x/uv.tar.gz\"\n"
        );
        let mut versions = Versions::new();
        collect(MISE_LOCK, &text, &mut versions).unwrap();
        assert_eq!(
            versions[&(
                MISE_LOCK.to_owned(),
                "aqua:astral-sh/uv/linux-x64".to_owned()
            )],
            "0.13.6-0123456789ab"
        );
    }

    #[test]
    fn a_git_failure_other_than_a_missing_path_is_an_error() {
        let repo = git_repo();
        let err = file_at(repo.path(), "no-such-rev", "a.toml").unwrap_err();
        assert!(format!("{err:#}").contains("git show"), "{err:#}");
        assert!(is_missing_path("fatal: path 'x' does not exist in 'HEAD'"));
        assert!(is_missing_path(
            "fatal: path 'x' exists on disk, but not in 'HEAD'"
        ));
        assert!(!is_missing_path("fatal: invalid object name 'nope'."));
    }

    #[test]
    fn a_lock_that_does_not_parse_is_an_error() {
        let mut versions = Versions::new();
        assert!(collect(MISE_LOCK, "not = [toml", &mut versions).is_err());
        assert!(
            collect(
                "runtimes/locks/pylock.p-linux.toml",
                "lock-version = \"9\"",
                &mut versions
            )
            .is_err()
        );
    }

    #[test]
    fn a_lock_missing_from_the_working_tree_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(file_now(dir.path(), "gone.toml").unwrap(), None);
        std::fs::write(dir.path().join("here.toml"), "x").unwrap();
        assert_eq!(
            file_now(dir.path(), "here.toml").unwrap().as_deref(),
            Some("x")
        );
    }

    /// The table `deps diff` prints is the pull request body the update
    /// workflow opens, and `scripts/validate_runtime_update.py` refuses any
    /// other shape; this runs real output through the real validator.
    #[test]
    #[cfg_attr(
        windows,
        ignore = "needs python3 >= 3.11 on PATH; covered on the Linux CI jobs"
    )]
    fn the_summary_passes_the_update_validator() {
        let root = crate::paths::workspace_root().unwrap();
        let before = BTreeMap::from([
            (
                (
                    "runtimes/mise.lock".to_owned(),
                    "github:lemonade-sdk/lemonade/linux-x64".to_owned(),
                ),
                "11.5.1-bdfd3c3e5d6e".to_owned(),
            ),
            (
                (
                    "runtimes/locks/pylock.sdk-rocm10-1-linux.toml".to_owned(),
                    "torch".to_owned(),
                ),
                "2.13.0+rocm10.1.0".to_owned(),
            ),
            (
                (
                    "runtimes/locks/pylock.sdk-rocm10-1-linux.toml".to_owned(),
                    "gone".to_owned(),
                ),
                "1.0".to_owned(),
            ),
        ]);
        let after = BTreeMap::from([
            (
                (
                    "runtimes/mise.lock".to_owned(),
                    "github:lemonade-sdk/lemonade/linux-x64".to_owned(),
                ),
                "2026.43.3-0123456789ab".to_owned(),
            ),
            (
                (
                    "runtimes/locks/pylock.sdk-rocm10-1-linux.toml".to_owned(),
                    "torch".to_owned(),
                ),
                "2.13.1+rocm10.1.0".to_owned(),
            ),
            (
                (
                    "runtimes/locks/pylock.sdk-rocm10-1-linux.toml".to_owned(),
                    "new".to_owned(),
                ),
                "2.0.post1".to_owned(),
            ),
        ]);
        for summary in [render(&before, &after), render(&before, &before)] {
            let dir = tempfile::tempdir().unwrap();
            let artifact = dir.path().join("update");
            std::fs::create_dir_all(artifact.join("locks")).unwrap();
            std::fs::copy(root.join("runtimes/mise.toml"), artifact.join("mise.toml")).unwrap();
            std::fs::write(artifact.join("mise.lock"), "lockfile_version = 3\n").unwrap();
            std::fs::write(artifact.join("locks/pylock.sdk-rocm10-1-linux.toml"), "").unwrap();
            std::fs::write(artifact.join("summary.md"), &summary).unwrap();
            let output = Command::new("python3")
                .arg("-I")
                .arg(root.join("scripts/validate_runtime_update.py"))
                .arg(&artifact)
                .arg(root.join("runtimes/mise.toml"))
                .output()
                .expect("python3 must be on PATH");
            assert!(
                output.status.success(),
                "validator refused deps diff output:\n{summary}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn a_well_formed_id_that_names_no_commit_is_an_error() {
        let repo = git_repo();
        assert!(resolve_commit(repo.path(), "0123456789abcdef0123456789abcdef01234567").is_err());
    }

    /// A repository holding the committed runtime files, with the Lemonade pin
    /// then bumped in the working tree only.
    fn runtimes_repo() -> tempfile::TempDir {
        let source = crate::paths::workspace_root().unwrap();
        let repo = git_repo();
        let runtimes = repo.path().join("runtimes");
        std::fs::create_dir_all(&runtimes).unwrap();
        for file in ["profiles.toml", "mise.lock"] {
            std::fs::copy(source.join("runtimes").join(file), runtimes.join(file)).unwrap();
        }
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .current_dir(repo.path())
                    .args(args)
                    .status()
                    .unwrap()
                    .success()
            );
        };
        git(&["add", "runtimes"]);
        git(&["commit", "-q", "-m", "runtimes"]);
        let lock = std::fs::read_to_string(runtimes.join("mise.lock")).unwrap();
        let version = format!("version = \"{}\"", rocm_deps_version(&lock));
        std::fs::write(
            runtimes.join("mise.lock"),
            lock.replace(&version, "version = \"9.9.9\""),
        )
        .unwrap();
        repo
    }

    fn rocm_deps_version(lock: &str) -> String {
        mise_lock::parse_lock(lock).unwrap()["github:lemonade-sdk/lemonade"]
            .version
            .clone()
    }

    #[test]
    fn the_summary_lists_the_binary_bump_made_since_the_base() {
        let repo = runtimes_repo();
        let table = summary(repo.path(), "HEAD").unwrap();
        assert!(
            table.contains("| `mise.lock` | `github:lemonade-sdk/lemonade/linux-x64` |"),
            "{table}"
        );
        assert!(table.contains("| 9.9.9-"), "{table}");
    }

    /// A lock the change adds (the base has no lock directory at all) is read
    /// from the profiles, so its packages show as added.
    #[test]
    fn a_lock_added_since_the_base_is_listed_as_added() {
        let repo = runtimes_repo();
        let locks = repo.path().join("runtimes/locks");
        std::fs::create_dir_all(&locks).unwrap();
        std::fs::write(
            locks.join("pylock.sdk-rocm10-1-linux.toml"),
            "lock-version = \"1.0\"\n\n[[packages]]\nname = \"Torch_Audio\"\nversion = \"2.11.0.2\"\n",
        )
        .unwrap();
        let table = summary(repo.path(), "HEAD").unwrap();
        assert!(
            table.contains("| `pylock.sdk-rocm10-1-linux.toml` | `torch-audio` | — | 2.11.0.2 |"),
            "{table}"
        );
    }

    /// A wheel swapped at an unchanged version is a changed row.
    #[test]
    fn an_artifact_swapped_at_the_same_version_is_listed() {
        let lock = |sum: &str| {
            format!(
                "lock-version = \"1.0\"\n\n[[packages]]\nname = \"numpy\"\nversion = \"2.3.0\"\n\
                 wheels = [{{ url = \"https://files.pythonhosted.org/n.whl\", hashes = {{ sha256 = \"{sum}\" }} }}]\n"
            )
        };
        let mut before = Versions::new();
        let mut after = Versions::new();
        let file = "runtimes/locks/pylock.p-linux.toml";
        collect(file, &lock(&"a".repeat(64)), &mut before).unwrap();
        collect(file, &lock(&"b".repeat(64)), &mut after).unwrap();
        let table = render(&before, &after);
        let old = &before[&(file.to_owned(), "numpy".to_owned())];
        let new = &after[&(file.to_owned(), "numpy".to_owned())];
        assert!(old.starts_with("2.3.0-") && new.starts_with("2.3.0-") && old != new);
        // A 12-hex-digit digest after the version.
        let digest = old.strip_prefix("2.3.0-").unwrap();
        assert!(
            digest.len() == 12 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "{old}"
        );
        assert!(
            table.contains(&format!(
                "| `pylock.p-linux.toml` | `numpy` | {old} | {new} |"
            )),
            "{table}"
        );
        // The same artifact again is not a change.
        let mut again = Versions::new();
        collect(file, &lock(&"a".repeat(64)), &mut again).unwrap();
        assert_eq!(render(&before, &again), format!("{NO_CHANGES}\n"));
    }

    #[test]
    fn an_sdist_only_package_carries_its_digest_and_wheel_order_does_not_matter() {
        let sdist = |sum: &str| {
            format!(
                "lock-version = \"1.0\"\n\n[[packages]]\nname = \"rocm\"\nversion = \"10.1.0\"\n\
                 sdist = {{ url = \"https://stable.repo.amd.com/rocm.tar.gz\", hashes = {{ sha256 = \"{sum}\" }} }}\n"
            )
        };
        let file = "runtimes/locks/pylock.p-linux.toml";
        let cell = |text: &str, package: &str| {
            let mut into = Versions::new();
            collect(file, text, &mut into).unwrap();
            into[&(file.to_owned(), package.to_owned())].clone()
        };
        let a = cell(&sdist(&"a".repeat(64)), "rocm");
        assert!(a.starts_with("10.1.0-"), "{a}");
        assert_ne!(a, cell(&sdist(&"b".repeat(64)), "rocm"));

        let wheels = |first: &str, second: &str| {
            format!(
                "lock-version = \"1.0\"\n\n[[packages]]\nname = \"numpy\"\nversion = \"2.3.0\"\n\
                 wheels = [{{ url = \"https://files.pythonhosted.org/1.whl\", hashes = {{ sha256 = \"{first}\" }} }}, \
                 {{ url = \"https://files.pythonhosted.org/2.whl\", hashes = {{ sha256 = \"{second}\" }} }}]\n"
            )
        };
        let (x, y) = ("c".repeat(64), "d".repeat(64));
        assert_eq!(
            cell(&wheels(&x, &y), "numpy"),
            cell(&wheels(&y, &x), "numpy")
        );
    }

    /// What `deps check` refuses, the table must not paper over: a package
    /// with no version, or one locked twice, is an error rather than a row
    /// that is missing or shows only one of the two.
    #[test]
    fn a_package_without_a_version_or_locked_twice_is_an_error() {
        let file = "runtimes/locks/pylock.p-linux.toml";
        let entry = |version: &str| {
            format!(
                "[[packages]]\nname = \"numpy\"\n{version}wheels = [{{ url = \"https://files.pythonhosted.org/n.whl\", hashes = {{ sha256 = \"{}\" }} }}]\n",
                "a".repeat(64)
            )
        };
        let versionless = format!("lock-version = \"1.0\"\n\n{}", entry(""));
        let twice = format!(
            "lock-version = \"1.0\"\n\n{}\n{}",
            entry("version = \"2.3.0\"\n"),
            entry("version = \"2.4.0\"\n")
        );
        for (expected, text) in [
            ("numpy is locked without a version", versionless),
            ("numpy is locked more than once", twice),
        ] {
            let err = collect(file, &text, &mut Versions::new()).unwrap_err();
            assert_eq!(format!("{err:#}"), format!("{file}: {expected}"));
        }
    }

    #[test]
    fn a_failed_listing_at_the_base_is_an_error() {
        let repo = git_repo();
        let err = files_at(repo.path(), "no-such-rev", "runtimes/locks").unwrap_err();
        assert!(
            format!("{err:#}").contains("git ls-tree no-such-rev"),
            "{err:#}"
        );
    }

    /// Only a missing lock reads as removed; one that cannot be read is an
    /// error, not a removal.
    #[test]
    fn an_unreadable_lock_in_the_working_tree_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("lock.toml")).unwrap();
        let err = file_now(dir.path(), "lock.toml").unwrap_err();
        assert!(
            format!("{err:#}").contains("failed to read lock.toml"),
            "{err:#}"
        );
    }

    #[test]
    fn the_summary_refuses_a_base_that_names_no_commit() {
        let repo = runtimes_repo();
        assert!(summary(repo.path(), "0123456789abcdef0123456789abcdef01234567").is_err());
    }

    #[test]
    fn windows_paths_are_spelled_the_way_git_names_them() {
        assert_eq!(
            slash_path(Path::new("runtimes\\locks\\pylock.p-windows.toml")),
            "runtimes/locks/pylock.p-windows.toml"
        );
        assert_eq!(
            slash_path(Path::new("runtimes/locks/x.toml")),
            "runtimes/locks/x.toml"
        );
    }

    #[test]
    fn python_package_names_are_normalised_so_a_rename_in_case_is_no_change() {
        let lock = |name: &str| {
            format!(
                "lock-version = \"1.0\"\n\n[[packages]]\nname = \"{name}\"\nversion = \"1.0\"\n"
            )
        };
        let mut before = Versions::new();
        let mut after = Versions::new();
        let file = "runtimes/locks/pylock.p-linux.toml";
        collect(file, &lock("Typing_Extensions"), &mut before).unwrap();
        collect(file, &lock("typing-extensions"), &mut after).unwrap();
        assert_eq!(before, after);
        assert_eq!(render(&before, &after), format!("{NO_CHANGES}\n"));
    }

    #[test]
    fn a_lock_the_change_removes_is_listed_as_removed() {
        let repo = runtimes_repo();
        let locks = repo.path().join("runtimes/locks");
        std::fs::create_dir_all(&locks).unwrap();
        std::fs::write(
            locks.join("pylock.gone-linux.toml"),
            "lock-version = \"1.0\"\n\n[[packages]]\nname = \"old\"\nversion = \"1.0\"\n",
        )
        .unwrap();
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .current_dir(repo.path())
                    .args(args)
                    .status()
                    .unwrap()
                    .success()
            );
        };
        git(&["add", "runtimes/locks"]);
        git(&["commit", "-q", "-m", "gone"]);
        std::fs::remove_file(locks.join("pylock.gone-linux.toml")).unwrap();
        let table = summary(repo.path(), "HEAD").unwrap();
        assert!(
            table.contains("| `pylock.gone-linux.toml` | `old` | 1.0 | — |"),
            "{table}"
        );
    }

    #[test]
    fn a_non_lock_file_at_the_base_is_an_error_not_skipped() {
        let repo = runtimes_repo();
        let locks = repo.path().join("runtimes/locks");
        std::fs::create_dir_all(&locks).unwrap();
        std::fs::write(locks.join("README.md"), "not a lock\n").unwrap();
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .current_dir(repo.path())
                    .args(args)
                    .status()
                    .unwrap()
                    .success()
            );
        };
        git(&["add", "runtimes/locks"]);
        git(&["commit", "-q", "-m", "readme"]);
        let err = summary(repo.path(), "HEAD").unwrap_err();
        assert!(format!("{err:#}").contains("README.md"), "{err:#}");
    }
}
