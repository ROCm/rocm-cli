// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Embeds what `rocm version` prints, so the binary reports exactly what it
//! was built from instead of the `Cargo.toml` version alone: the release tag
//! for a tag build, or the branch name otherwise (feature branches and `main`
//! CI builds aren't tagged, and a bare commit hash isn't enough to trace a
//! build back to its branch), plus the commit hash either way.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Watch both HEAD and its resolved ref because HEAD itself does not change
    // when a commit advances the current branch. Resolve the per-worktree and
    // common Git directories instead of assuming `.git` is a directory.
    for path in git_watch_paths(Path::new(".")) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=GITHUB_REF_TYPE");
    println!("cargo:rerun-if-env-changed=GITHUB_REF_NAME");
    println!("cargo:rerun-if-env-changed=GITHUB_HEAD_REF");

    let git_hash =
        run_git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());

    println!("cargo:rustc-env=ROCM_CLI_VERSION_REF={}", ref_descriptor());
    println!("cargo:rustc-env=ROCM_CLI_GIT_HASH={git_hash}");
}

/// The release tag for a tag build, else the branch name, else "unknown".
///
/// GitHub Actions checks out a detached HEAD even for branch builds, so
/// `git rev-parse --abbrev-ref HEAD` can't recover the branch name in CI —
/// its own ref env vars are the only source there. Local dev builds have no
/// such env vars but do have an attached HEAD, so git is the fallback.
fn ref_descriptor() -> String {
    let env_var = |name| env::var(name).ok().filter(|value| !value.is_empty());

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
    let Some(git_dir) = run_git_at(cwd, &["rev-parse", "--absolute-git-dir"]).map(PathBuf::from)
    else {
        return Vec::new();
    };
    let common_dir = run_git_at(cwd, &["rev-parse", "--git-common-dir"])
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
    if let Some(reference) = run_git_at(cwd, &["symbolic-ref", "-q", "HEAD"]) {
        paths.push(common_dir.join(reference));
    }
    paths.push(common_dir.join("packed-refs"));
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
    use super::git_watch_paths;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn git_watch_paths_include_head_branch_and_packed_refs() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before Unix epoch")
            .as_nanos();
        let repo = std::env::temp_dir().join(format!("rocm-build-metadata-{nonce}"));
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

        let paths = git_watch_paths(&repo);
        fs::remove_dir_all(&repo).expect("remove temporary repository");
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().all(|path| path.is_absolute()));
        assert!(paths[0].ends_with(Path::new("HEAD")));
        assert!(paths[1].ends_with(Path::new("refs").join("heads").join("main")));
        assert!(paths[2].ends_with(Path::new("packed-refs")));
    }
}
