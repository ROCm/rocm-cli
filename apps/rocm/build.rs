// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Besides the Windows stack-size workaround below, embeds the string
//! `rocm version` and `-V`/`--version` print. `format_cli_version` turns the
//! ref this file already resolves (the release tag for a tag build, otherwise
//! the branch name, otherwise `unknown`) and the commit hash into one value,
//! `ROCM_CLI_VERSION`:
//!
//! - exact tag `vX.Y.Z` (optional pre-release or build suffix) →
//!   `X.Y.Z (vX.Y.Z, <hash>)`, with a single leading `v` removed
//! - any other resolved ref, including `main` and feature branches →
//!   `0.0.0-dev (<ref>, <hash>)`, which is not a release
//! - unresolved ref (`unknown`: no tag, no branch, no git) →
//!   `<CARGO_PKG_VERSION> (unknown, <hash>)`
//!
//! `ROCM_CLI_VERSION_REF` and `ROCM_CLI_GIT_HASH` stay set to those inputs.
//! The workspace package version is shared by every crate and unpublished, so
//! it is the leading version only when the ref did not resolve.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // The generated clap parser for the unified CLI has a large command graph.
    // Windows reserves a much smaller main-thread stack than our Unix targets,
    // and parsing can otherwise overflow before dispatch reaches the selected
    // command. This affects the executable only; test-harness worker threads
    // already receive Rust's independently configured stack size.
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some("windows".as_ref())
        && std::env::var_os("CARGO_CFG_TARGET_ENV").as_deref() == Some("msvc".as_ref())
    {
        println!("cargo:rustc-link-arg-bin=rocm=/STACK:8388608");
    }

    println!("cargo:rerun-if-changed=build.rs");
    // Watch both HEAD and its resolved ref because HEAD itself does not change
    // when a commit advances the current branch, plus the tags directory so a
    // freshly created tag invalidates a build that ran before it existed.
    // Resolves the per-worktree and common Git directories instead of
    // assuming `.git` is a directory.
    for path in git_watch_paths(Path::new(".")) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=ROCM_CLI_RELEASE_REF");
    println!("cargo:rerun-if-env-changed=GITHUB_REF_TYPE");
    println!("cargo:rerun-if-env-changed=GITHUB_REF_NAME");
    println!("cargo:rerun-if-env-changed=GITHUB_HEAD_REF");

    let version_ref = ref_descriptor();
    let git_hash =
        run_git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    // Cargo always sets this for the package whose build script is running.
    // An empty value is not a version; `format_cli_version` substitutes one.
    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap_or_default();

    println!("cargo:rustc-env=ROCM_CLI_VERSION_REF={version_ref}");
    println!("cargo:rustc-env=ROCM_CLI_GIT_HASH={git_hash}");
    // The only string `-V`/`--version` and `rocm version` print. Formatting
    // it here, once, is what keeps those surfaces from disagreeing.
    println!(
        "cargo:rustc-env=ROCM_CLI_VERSION={}",
        format_cli_version(&version_ref, &git_hash, &pkg_version)
    );
}

/// Leading version for a build whose ref resolved and is not a release tag.
/// Not the workspace package version: a `main` or feature-branch binary must
/// not be presented as a release.
const NON_RELEASE_VERSION: &str = "0.0.0-dev";

/// `rocm version` / `-V` text after the `rocm-cli ` prefix.
///
/// See the module docs for the tag, branch, and unknown shapes. `version_ref`
/// and `git_hash` are the values this build script already resolves; an empty
/// input is treated as `unknown` so the result is never an empty version.
fn format_cli_version(version_ref: &str, git_hash: &str, cargo_pkg_version: &str) -> String {
    let resolved_ref = if version_ref.is_empty() {
        "unknown"
    } else {
        version_ref
    };
    let resolved_hash = if git_hash.is_empty() {
        "unknown"
    } else {
        git_hash
    };
    let version = displayed_version(resolved_ref, cargo_pkg_version);
    format!("{version} ({resolved_ref}, {resolved_hash})")
}

fn displayed_version(version_ref: &str, cargo_pkg_version: &str) -> String {
    if let Some(release) = release_version(version_ref) {
        return release.to_owned();
    }
    if version_ref == "unknown" {
        let package = if cargo_pkg_version.is_empty() {
            "0.0.0"
        } else {
            cargo_pkg_version
        };
        return package.to_owned();
    }
    NON_RELEASE_VERSION.to_owned()
}

/// `vX.Y.Z` with an optional pre-release (`-rc.1`) or build (`+meta`) suffix.
/// Returns the version with that one leading `v` removed. A ref that merely
/// starts with `v` (`vllm`) is not a release tag.
fn release_version(version_ref: &str) -> Option<&str> {
    let rest = version_ref.strip_prefix('v')?;
    let core = match rest.split_once(['-', '+']) {
        Some((core, suffix)) if !suffix.is_empty() => core,
        Some(_) => return None,
        None => rest,
    };
    let mut components = core.split('.');
    let has_major_minor_patch = (0..3).all(|_| components.next().is_some_and(is_ascii_digits));
    // Exactly three components: `v1.2` and `v1.2.3.4` are not `vX.Y.Z`.
    if has_major_minor_patch && components.next().is_none() {
        Some(rest)
    } else {
        None
    }
}

fn is_ascii_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// The release tag for a tag build, else the branch name, else "unknown".
///
/// `ROCM_CLI_RELEASE_REF` takes priority when set: the release workflow
/// resolves the ref once, in its own "Determine version" step, and passes it
/// in explicitly. That is the only reliable source for the Linux release
/// build, which runs inside a `docker run` that does not forward GitHub's own
/// `GITHUB_REF_*` context into the container — those ambient vars, and the
/// local `git` fallback below, are read only when no explicit ref was given
/// (every other build: CI, nightly, local dev).
///
/// GitHub Actions checks out a detached HEAD even for branch builds, so
/// `git rev-parse --abbrev-ref HEAD` can't recover the branch name in CI —
/// its own ref env vars are the only source there. Local dev builds have no
/// such env vars but do have an attached HEAD, so git is the fallback.
fn ref_descriptor() -> String {
    let env_var = |name| env::var(name).ok().filter(|value| !value.is_empty());

    if let Some(release_ref) = env_var("ROCM_CLI_RELEASE_REF") {
        return release_ref;
    }
    if env_var("GITHUB_REF_TYPE").as_deref() == Some("tag")
        && let Some(tag) = env_var("GITHUB_REF_NAME")
    {
        return tag;
    }
    if let Some(branch) = env_var("GITHUB_HEAD_REF").or_else(|| env_var("GITHUB_REF_NAME")) {
        return branch;
    }

    if let Some(tag) = run_git(&["describe", "--tags", "--exact-match", "--match", "v*"]) {
        return tag;
    }
    match run_git(&["rev-parse", "--abbrev-ref", "HEAD"]) {
        Some(branch) if branch != "HEAD" => branch,
        _ => "unknown".to_owned(),
    }
}

fn git_watch_paths(cwd: &Path) -> Vec<PathBuf> {
    // Production calls this with a relative `cwd` ("."); resolving to an
    // absolute base first is what makes every path below actually absolute,
    // rather than merely relative-looking-absolute (`./../../.git/...`) —
    // joining a relative `--git-common-dir` onto a relative `cwd` would
    // otherwise stay relative, which Cargo still resolves against the package
    // root, but only by accident: a `rerun-if-changed` path that merely looks
    // absolute-ish is not actually one, and nothing about `cwd.join(path)`
    // guarantees it becomes one unless `cwd` itself already is.
    let cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());

    let Some(git_dir) = run_git_at(&cwd, &["rev-parse", "--absolute-git-dir"]).map(PathBuf::from)
    else {
        return Vec::new();
    };
    let common_dir = run_git_at(&cwd, &["rev-parse", "--git-common-dir"])
        .map(PathBuf::from)
        .map_or_else(
            || git_dir.clone(),
            |path| {
                if path.is_absolute() {
                    path
                } else {
                    cwd.join(path)
                }
            },
        );

    let mut paths = vec![git_dir.join("HEAD")];
    // Cargo treats a missing `rerun-if-changed` path as permanently dirty (it
    // has nothing to compare a missing file's mtime against), so a path that
    // doesn't exist in *this* checkout must be skipped rather than watched.
    // packed-refs in particular does not exist in every checkout shape, and
    // the branch ref is not always a loose file either: `git pack-refs`/
    // `git gc` (including its automatic `gc --auto`) deletes the loose
    // branch-ref file while HEAD still points at it, packing the ref into
    // packed-refs instead — a repo that has ever been gc'd or packed would
    // otherwise permanently rebuild on every single build. The trade-off: if
    // one of these is created later from nothing (this specific checkout's
    // first-ever tag, or its first `git gc`), that one transition can be
    // missed until some other rebuild trigger fires: a far smaller gap than
    // recompiling on every single build, forever.
    let mut watched_refs = Vec::new();
    if let Some(reference) = run_git_at(&cwd, &["symbolic-ref", "-q", "HEAD"]) {
        watched_refs.push(common_dir.join(reference));
    }
    watched_refs.push(common_dir.join("packed-refs"));
    watched_refs.push(common_dir.join("refs").join("tags"));
    for watched in watched_refs {
        if watched.exists() {
            paths.push(watched);
        }
    }
    paths
}

fn run_git(args: &[&str]) -> Option<String> {
    run_git_at(Path::new("."), args)
}

fn run_git_at(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .current_dir(cwd)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{format_cli_version, git_watch_paths};
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A fresh, empty git repository under `CARGO_MANIFEST_DIR/target`, with
    /// `HEAD` pointed at `refs/heads/main`. Under `CARGO_MANIFEST_DIR`, not
    /// `std::env::temp_dir()`: `cargo test` runs this integration test with
    /// the crate root as its process cwd, so a path relative to
    /// `CARGO_MANIFEST_DIR` is also relative to the real cwd — which is what
    /// lets tests call `git_watch_paths` with a *relative* path, mirroring
    /// the real `Path::new(".")` call site in `main()`. A test that only
    /// ever passes an absolute cwd cannot catch a relative one leaking
    /// through into a real build's `rerun-if-changed`.
    fn init_test_repo(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before Unix epoch")
            .as_nanos();
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo = manifest_dir
            .join("target")
            .join(format!("build-metadata-test-{tag}-{nonce}"));
        fs::create_dir_all(&repo).expect("create temporary repository directory");
        let status = Command::new("git")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .arg("init")
            .arg(&repo)
            .status()
            .expect("run git init");
        assert!(status.success(), "git init failed");
        let status = Command::new("git")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .current_dir(&repo)
            .args(["symbolic-ref", "HEAD", "refs/heads/main"])
            .status()
            .expect("set initial branch");
        assert!(status.success(), "setting initial branch failed");
        repo
    }

    /// Commit in `repo` with a throwaway identity, so tests don't depend on
    /// the environment having `user.name`/`user.email` configured.
    fn commit_something(repo: &Path) {
        let status = Command::new("git")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .current_dir(repo)
            .args([
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=test",
                "commit",
                "--allow-empty",
                "-m",
                "test commit",
            ])
            .status()
            .expect("run git commit");
        assert!(status.success(), "git commit failed");
    }

    #[test]
    fn git_watch_paths_skips_packed_refs_and_uncommitted_branch_ref() {
        // Regression test: Cargo treats a missing `rerun-if-changed` path as
        // permanently dirty, and a fresh `git init` never creates
        // packed-refs (only a later `git gc`/`git repack`/some clone shapes
        // do) or a loose ref for a branch with no commits yet (`symbolic-ref`
        // still resolves HEAD to it, but nothing has written the file), so
        // watching either unconditionally would recompile on every build,
        // forever, in exactly this common a checkout shape.
        // `refs/tags` is a different case: `git init` always creates it
        // (empty), so it is not exercised by this test.
        let repo = init_test_repo("no-packed-refs");
        assert!(!repo.join(".git").join("packed-refs").exists());
        assert!(
            !repo
                .join(".git")
                .join("refs")
                .join("heads")
                .join("main")
                .exists()
        );
        assert!(repo.join(".git").join("refs").join("tags").exists());

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let relative_repo = repo
            .strip_prefix(manifest_dir)
            .expect("repo is under the manifest dir");
        let paths = git_watch_paths(relative_repo);
        fs::remove_dir_all(&repo).expect("remove temporary repository");

        assert_eq!(
            paths.len(),
            2,
            "HEAD and refs/tags, but not the absent branch ref or packed-refs: {paths:?}"
        );
        assert!(paths[0].ends_with(Path::new("HEAD")));
        assert!(paths[1].ends_with(Path::new("refs").join("tags")));
    }

    #[test]
    fn git_watch_paths_include_head_branch_tags_and_packed_refs() {
        let repo = init_test_repo("with-packed-refs");
        commit_something(&repo);
        // An empty file is a valid (trivial) packed-refs; anything else needs
        // real header/ref-line syntax or git rejects *every* command in this
        // repo with "unexpected line in .git/packed-refs" — including the
        // `symbolic-ref` this same function relies on to find the branch ref.
        fs::write(repo.join(".git").join("packed-refs"), "").expect("create packed-refs");

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let relative_repo = repo
            .strip_prefix(manifest_dir)
            .expect("repo is under the manifest dir");
        let paths = git_watch_paths(relative_repo);
        fs::remove_dir_all(&repo).expect("remove temporary repository");
        assert_eq!(paths.len(), 4);
        assert!(
            paths.iter().all(|path| path.is_absolute()),
            "every watched path must be absolute even when cwd is relative: {paths:?}"
        );
        assert!(paths[0].ends_with(Path::new("HEAD")));
        assert!(paths[1].ends_with(Path::new("refs").join("heads").join("main")));
        assert!(paths[2].ends_with(Path::new("packed-refs")));
        assert!(paths[3].ends_with(Path::new("refs").join("tags")));
    }

    #[test]
    fn git_watch_paths_treats_a_packed_branch_ref_as_gone_not_missing() {
        // `git pack-refs --all` (which `git gc`, including the automatic
        // `gc --auto`, runs) deletes the loose `refs/heads/<branch>` file
        // while HEAD still resolves to it, folding the ref into packed-refs
        // instead. A watch list built after that must not include the now-gone
        // loose file (Cargo would treat it as permanently dirty) and must
        // still include packed-refs, which now carries the branch's value.
        let repo = init_test_repo("packed-branch-ref");
        commit_something(&repo);
        let branch_ref = repo.join(".git").join("refs").join("heads").join("main");
        assert!(
            branch_ref.exists(),
            "commit should create the loose branch ref"
        );

        let status = Command::new("git")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .current_dir(&repo)
            .args(["pack-refs", "--all"])
            .status()
            .expect("run git pack-refs");
        assert!(status.success(), "git pack-refs failed");
        assert!(
            !branch_ref.exists(),
            "git pack-refs --all should remove the loose branch ref"
        );
        assert!(repo.join(".git").join("packed-refs").exists());

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let relative_repo = repo
            .strip_prefix(manifest_dir)
            .expect("repo is under the manifest dir");
        let paths = git_watch_paths(relative_repo);
        fs::remove_dir_all(&repo).expect("remove temporary repository");

        assert_eq!(
            paths.len(),
            3,
            "HEAD, packed-refs, and refs/tags, but not the now-packed branch ref: {paths:?}"
        );
        assert!(paths[0].ends_with(Path::new("HEAD")));
        assert!(paths[1].ends_with(Path::new("packed-refs")));
        assert!(paths[2].ends_with(Path::new("refs").join("tags")));
        assert!(
            !paths
                .iter()
                .any(|path| path.ends_with(Path::new("refs").join("heads").join("main"))),
            "the packed branch ref has no loose file anymore and must not be watched: {paths:?}"
        );
    }

    #[test]
    fn format_cli_version_reports_a_release_tag_without_its_leading_v() {
        assert_eq!(
            format_cli_version("v0.2.1", "1cde824", "0.1.0"),
            "0.2.1 (v0.2.1, 1cde824)"
        );
        // A pre-release or build suffix is still that release's version.
        assert_eq!(
            format_cli_version("v0.2.1-rc.1", "1cde824", "0.1.0"),
            "0.2.1-rc.1 (v0.2.1-rc.1, 1cde824)"
        );
        assert_eq!(
            format_cli_version("v0.2.1+build.1", "1cde824", "0.1.0"),
            "0.2.1+build.1 (v0.2.1+build.1, 1cde824)"
        );
    }

    #[test]
    fn format_cli_version_reports_main_as_a_dev_build() {
        assert_eq!(
            format_cli_version("main", "abcdef1", "0.1.0"),
            "0.0.0-dev (main, abcdef1)"
        );
        // Same class as `main`: a feature branch, and a name that starts with
        // `v` but is not `vX.Y.Z`, are not releases either.
        assert_eq!(
            format_cli_version("fix/cli-version-string", "abcdef1", "0.1.0"),
            "0.0.0-dev (fix/cli-version-string, abcdef1)"
        );
        assert_eq!(
            format_cli_version("vllm", "abcdef1", "0.1.0"),
            "0.0.0-dev (vllm, abcdef1)"
        );
        // Not an exact `vX.Y.Z`: too few components, or a separator with no
        // pre-release or build suffix.
        assert_eq!(
            format_cli_version("v0.2", "abcdef1", "0.1.0"),
            "0.0.0-dev (v0.2, abcdef1)"
        );
        assert_eq!(
            format_cli_version("v1.2.3-", "abcdef1", "0.1.0"),
            "0.0.0-dev (v1.2.3-, abcdef1)"
        );
    }

    #[test]
    fn format_cli_version_falls_back_to_the_package_version_only_when_the_ref_is_unknown() {
        assert_eq!(
            format_cli_version("unknown", "unknown", "0.1.0"),
            "0.1.0 (unknown, unknown)"
        );
        // The hash can still resolve when the ref cannot (detached HEAD, no
        // exact tag). The package version is the fallback for that case only,
        // and an empty input does not produce an empty version.
        assert_eq!(
            format_cli_version("unknown", "abcdef1", "0.1.0"),
            "0.1.0 (unknown, abcdef1)"
        );
        assert_eq!(format_cli_version("", "", ""), "0.0.0 (unknown, unknown)");
    }

    #[test]
    fn embedded_cli_version_uses_format_cli_version() {
        // The build script exports one string. This ties that export to the
        // formatter, using the ref and hash this checkout actually resolved.
        assert_eq!(
            env!("ROCM_CLI_VERSION"),
            format_cli_version(
                env!("ROCM_CLI_VERSION_REF"),
                env!("ROCM_CLI_GIT_HASH"),
                env!("CARGO_PKG_VERSION"),
            )
        );
    }
}
