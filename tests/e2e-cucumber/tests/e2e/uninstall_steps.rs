// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm uninstall` and the folders its plan removes.
//!
//! Black-box against the real binary. Every run is confined to the scenario's
//! isolated root: `--keep-binaries` so the binary under test survives, and HOME,
//! `UV_CACHE_DIR` and `HF_HOME` pointed inside the root so no uninstall can reach
//! a shared cache the runner keeps between scenarios.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

const USER_FILE: &str = "keep.txt";

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
}

/// Where the scenario's cache setting points: a link in the isolated root.
fn cache_link(world: &E2eWorld) -> PathBuf {
    root(world).join("rocm-cache")
}

/// The folder the link points at, holding a file ROCm CLI never created.
fn link_target(world: &E2eWorld) -> PathBuf {
    root(world).join("relocated-cache")
}

#[given("the cache folder is a link to another folder holding the user's files")]
async fn cache_is_live_link(world: &mut E2eWorld) {
    let target = link_target(world);
    std::fs::create_dir_all(&target).expect("failed to create link target");
    std::fs::write(target.join(USER_FILE), "user data").expect("failed to write user file");
    E2eWorld::link_directory(&target, &cache_link(world)).expect("failed to plant cache link");
}

#[given("the cache folder is a link to a folder that no longer exists")]
async fn cache_is_dangling_link(world: &mut E2eWorld) {
    let gone = root(world).join("gone");
    assert!(
        std::fs::symlink_metadata(&gone).is_err(),
        "the link target must not exist"
    );
    E2eWorld::link_directory(&gone, &cache_link(world)).expect("failed to plant cache link");
}

fn run_cache_only_uninstall(world: &mut E2eWorld, cache_setting: &str) {
    let root = root(world).to_path_buf();
    let home = root.join("home");
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let home = home.display().to_string();
    let uv = root.join("uv-cache").display().to_string();
    let hf = root.join("hf-home").display().to_string();
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &[
            "uninstall",
            "--yes",
            "--keep-binaries",
            "--keep-config",
            "--keep-data",
        ],
        &[
            ("ROCM_CLI_CACHE_DIR", cache_setting),
            ("HOME", &home),
            ("UV_CACHE_DIR", &uv),
            ("HF_HOME", &hf),
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user uninstalls only the cache, writing its folder with a trailing slash")]
async fn uninstall_cache_trailing_slash(world: &mut E2eWorld) {
    let setting = format!("{}/", cache_link(world).display());
    run_cache_only_uninstall(world, &setting);
}

#[when("the user uninstalls only the cache")]
async fn uninstall_cache(world: &mut E2eWorld) {
    let setting = cache_link(world).display().to_string();
    run_cache_only_uninstall(world, &setting);
}

fn output(world: &E2eWorld) -> &str {
    world
        .cli_output
        .as_deref()
        .expect("no CLI output captured - did the When step run?")
}

#[then("the uninstall succeeds")]
async fn uninstall_succeeds(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    assert!(
        rc == 0,
        "uninstall must exit 0, got {rc}\nstdout:\n{}\nstderr:\n{}",
        output(world),
        world.cli_stderr.as_deref().unwrap_or("")
    );
}

#[then("the uninstall reports the cache link as removed")]
async fn reports_link_removed(world: &mut E2eWorld) {
    // The line names the link itself, without the trailing slash it was
    // configured with: that is the entry that is removed, so it is what the
    // plan and the confirmation show.
    let line = format!("removed cache {}", cache_link(world).display());
    let stdout = output(world);
    assert!(
        stdout.lines().any(|l| l == line),
        "expected a `{line}` line:\n{stdout}"
    );
}

#[then("the cache link is gone")]
async fn cache_link_gone(world: &mut E2eWorld) {
    let link = cache_link(world);
    assert!(
        std::fs::symlink_metadata(&link).is_err(),
        "the cache link {} must be removed, not left behind:\n{}",
        link.display(),
        output(world)
    );
}

#[then("the folder the link pointed to still holds the user's files")]
async fn target_intact(world: &mut E2eWorld) {
    let file = link_target(world).join(USER_FILE);
    assert!(
        file.is_file(),
        "uninstall followed the link and deleted {}:\n{}",
        file.display(),
        output(world)
    );
}
