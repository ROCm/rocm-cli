// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Embeds what `rocm version` prints, so the binary reports exactly what it
//! was built from instead of the `Cargo.toml` version alone: the release tag
//! for a tag build, or the branch name otherwise (feature branches and `main`
//! CI builds aren't tagged, and a bare commit hash isn't enough to trace a
//! build back to its branch), plus the commit hash either way.

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Rebuild when HEAD moves (checkout/commit), not on every build.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
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

    if env_var("GITHUB_REF_TYPE").as_deref() == Some("tag") {
        if let Some(tag) = env_var("GITHUB_REF_NAME") {
            return tag;
        }
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

fn run_git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
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
