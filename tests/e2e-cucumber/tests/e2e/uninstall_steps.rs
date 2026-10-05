// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm uninstall` and the folders its plan removes.
//!
//! Black-box against the real binary. Every run is confined to the scenario's
//! isolated root: `--keep-binaries` so the binary under test survives, and
//! HOME, the cache folder, `UV_CACHE_DIR` and `HF_HOME` pointed inside the root
//! so no uninstall can reach a shared cache the runner keeps between
//! scenarios. The one setting that names a folder outside the root, a data
//! folder of `/`, is only ever used with `--dry-run`.

use std::ffi::OsString;
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

/// The scenario's own HOME.
fn home(world: &E2eWorld) -> PathBuf {
    root(world).join("home")
}

fn user_file_in_home(world: &E2eWorld) -> PathBuf {
    home(world).join("Documents").join(USER_FILE)
}

fn set_env(world: &mut E2eWorld, key: &'static str, value: impl Into<OsString>) {
    world.command_env.retain(|(existing, _)| *existing != key);
    world.command_env.push((key, value.into()));
}

/// Run `rocm uninstall --keep-binaries <flags>` confined to the scenario root.
///
/// The defaults keep every folder the CLI might touch inside the root; a Given
/// step overrides one through `world.command_env`, which is kept (not
/// consumed) so a scenario can re-run the command with different flags.
fn run_uninstall(world: &mut E2eWorld, flags: &[&str]) {
    let root = root(world).to_path_buf();
    let home = home(world);
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let mut env: Vec<(&'static str, OsString)> = vec![
        ("HOME", home.clone().into_os_string()),
        // Windows reads the home folder from USERPROFILE first.
        ("USERPROFILE", home.into_os_string()),
        // Not the world's cache folder: on CI that can be a cache shared across
        // scenarios, and an uninstall removes the cache folder.
        (
            "ROCM_CLI_CACHE_DIR",
            root.join("cache-local").into_os_string(),
        ),
        ("UV_CACHE_DIR", root.join("uv-cache").into_os_string()),
        ("HF_HOME", root.join("hf-home").into_os_string()),
    ];
    for (key, value) in &world.command_env {
        env.retain(|(existing, _)| existing != key);
        env.push((key, value.clone()));
    }

    let mut args = vec!["uninstall", "--keep-binaries"];
    args.extend_from_slice(flags);
    let binary = crate::rocm_binary();
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(&args);
    world.isolate_cmd(&mut cmd);
    for (key, value) in env {
        cmd.env(key, value);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));
    let rc = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    crate::record_command(world.current_scenario.as_deref(), &args, rc, &stdout);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(String::from_utf8_lossy(&output.stderr).to_string());
    world.cli_rc = Some(rc);
}

// ── Given ──────────────────────────────────────────────────────────

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

#[given("the data folder is set to the top of the filesystem")]
async fn data_is_filesystem_root(world: &mut E2eWorld) {
    set_env(world, "ROCM_CLI_DATA_DIR", "/");
}

#[given("the data folder is set to the user's home folder, which holds the user's files")]
async fn data_is_home(world: &mut E2eWorld) {
    let file = user_file_in_home(world);
    std::fs::create_dir_all(file.parent().expect("parent")).expect("failed to create home");
    std::fs::write(&file, "user data").expect("failed to write user file");
    // The config folder exists too, so a refusal that still removed the other
    // roots would show.
    std::fs::create_dir_all(root(world).join("config")).expect("failed to create config");
    let home = home(world);
    set_env(world, "ROCM_CLI_DATA_DIR", home);
}

#[given(
    "the cache folder is set to the user's own cache folder, which holds the uv and model caches"
)]
async fn cache_holds_shared_caches(world: &mut E2eWorld) {
    let dot_cache = home(world).join(".cache");
    let uv = dot_cache.join("uv");
    let hf = dot_cache.join("huggingface");
    std::fs::create_dir_all(&uv).expect("failed to create uv cache");
    std::fs::create_dir_all(hf.join("hub")).expect("failed to create model cache");
    std::fs::write(uv.join("blob"), "wheel").expect("failed to write uv blob");
    set_env(world, "ROCM_CLI_CACHE_DIR", dot_cache);
    set_env(world, "UV_CACHE_DIR", uv);
    set_env(world, "HF_HOME", hf);
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user uninstalls only the cache, writing its folder with a trailing slash")]
async fn uninstall_cache_trailing_slash(world: &mut E2eWorld) {
    let setting = format!("{}/", cache_link(world).display());
    set_env(world, "ROCM_CLI_CACHE_DIR", setting);
    run_uninstall(world, &["--yes", "--keep-config", "--keep-data"]);
}

#[when("the user uninstalls only the cache")]
async fn uninstall_cache(world: &mut E2eWorld) {
    let setting = cache_link(world);
    set_env(world, "ROCM_CLI_CACHE_DIR", setting);
    run_uninstall(world, &["--yes", "--keep-config", "--keep-data"]);
}

#[when("the user previews an uninstall")]
async fn preview_uninstall(world: &mut E2eWorld) {
    run_uninstall(world, &["--dry-run"]);
}

#[when("the user uninstalls")]
async fn uninstall_everything(world: &mut E2eWorld) {
    run_uninstall(world, &["--yes"]);
}

#[when("the user uninstalls again with the flag the refusal advised")]
async fn uninstall_with_advised_flag(world: &mut E2eWorld) {
    // Take the flag from the message rather than restating it, so this proves
    // the advice the user actually reads.
    let message = format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    let flag = message
        .split("Re-run with ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .filter(|word| word.starts_with("--keep-"))
        .unwrap_or_else(|| panic!("the refusal named no `Re-run with --keep-...`:\n{message}"))
        .to_owned();
    run_uninstall(world, &["--yes", &flag]);
}

// ── Then ───────────────────────────────────────────────────────────

fn output(world: &E2eWorld) -> &str {
    world
        .cli_output
        .as_deref()
        .expect("no CLI output captured - did the When step run?")
}

fn stderr(world: &E2eWorld) -> &str {
    world.cli_stderr.as_deref().unwrap_or("")
}

const fn rc(world: &E2eWorld) -> i32 {
    world.cli_rc.expect("no exit code captured")
}

#[then("the uninstall succeeds")]
async fn uninstall_succeeds(world: &mut E2eWorld) {
    assert!(
        rc(world) == 0,
        "uninstall must exit 0, got {}\nstdout:\n{}\nstderr:\n{}",
        rc(world),
        output(world),
        stderr(world)
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

#[then(expr = "the uninstall is refused because the data folder {string} is {string}")]
async fn refused_because(world: &mut E2eWorld, folder: String, why: String) {
    let folder = if folder == "<home>" {
        home(world).display().to_string()
    } else {
        folder
    };
    let stdout = output(world);
    assert!(
        rc(world) != 0,
        "a refused uninstall must fail, got exit 0:\n{stdout}"
    );
    // The review lists the root under "Refused", with the reason, and never in
    // the would-be-removed list.
    let refused_line = format!("  - data: {folder} is {why}");
    assert!(
        stdout.lines().any(|line| line == refused_line),
        "expected `{refused_line}` in the review:\n{stdout}"
    );
    let removed_line = format!("  - data: {folder}");
    assert!(
        !stdout.lines().any(|line| line == removed_line),
        "the refused root must not be listed for removal:\n{stdout}"
    );
}

#[then("the refusal advises re-running with --keep-data")]
async fn refusal_advises_keep_data(world: &mut E2eWorld) {
    let message = format!("{}{}", output(world), stderr(world));
    assert!(
        message
            .contains("Re-run with --keep-data to remove everything else and leave it in place."),
        "the refusal must say how to proceed:\n{message}"
    );
}

#[then("the refusal says nothing was removed")]
async fn refusal_says_nothing_removed(world: &mut E2eWorld) {
    // Asserted together with the state in the steps that follow it in the
    // scenario: the home file and the config folder are both still there.
    assert!(
        stderr(world).contains("uninstall refused, nothing was removed"),
        "stderr:\n{}",
        stderr(world)
    );
}

#[then("the user's files in the home folder are still there")]
async fn home_files_intact(world: &mut E2eWorld) {
    let file = user_file_in_home(world);
    assert!(
        file.is_file(),
        "uninstall deleted {}:\n{}\n{}",
        file.display(),
        output(world),
        stderr(world)
    );
}

#[then("the config folder is still there")]
async fn config_intact(world: &mut E2eWorld) {
    let config = root(world).join("config");
    assert!(
        config.is_dir(),
        "a refused uninstall removed {}:\n{}",
        config.display(),
        output(world)
    );
}

#[then("the config folder is gone")]
async fn config_gone(world: &mut E2eWorld) {
    let config = root(world).join("config");
    assert!(
        !config.exists(),
        "the uninstall did not remove {}:\n{}",
        config.display(),
        output(world)
    );
}

#[then("the review warns that the uv and model caches will be deleted with the cache folder")]
async fn warns_shared_caches_deleted(world: &mut E2eWorld) {
    let stdout = output(world);
    let dot_cache = home(world).join(".cache");
    let uv = dot_cache.join("uv");
    let hub = dot_cache.join("huggingface").join("hub");
    for (name, path) in [
        ("the uv package cache", uv),
        ("downloaded model files", hub),
    ] {
        let warning = format!(
            "{name} WILL BE DELETED: {} is inside the cache folder being removed ({})",
            path.display(),
            dot_cache.display()
        );
        assert!(
            stdout.contains(&warning),
            "expected `{warning}` in the review:\n{stdout}"
        );
        assert!(
            !stdout.contains(&format!("not removed: {}", path.display())),
            "the review must not also call {} kept:\n{stdout}",
            path.display()
        );
    }
    // Shown in the review, which precedes the confirmation prompt.
    let review_ends = stdout
        .find("Left alone:")
        .or_else(|| stdout.find("Choose Review uninstall"))
        .unwrap_or(stdout.len());
    assert!(
        stdout
            .find("WILL BE DELETED")
            .is_some_and(|at| at < review_ends),
        "the warning must be part of the review:\n{stdout}"
    );
    assert!(
        stdout.contains("Re-run with --keep-cache to keep"),
        "the warning must say how to keep the caches:\n{stdout}"
    );
}

#[then("the uv cache is still there after the preview")]
async fn uv_cache_intact(world: &mut E2eWorld) {
    let blob = home(world).join(".cache").join("uv").join("blob");
    assert!(blob.is_file(), "a dry run deleted {}", blob.display());
    assert!(
        rc(world) == 0,
        "the preview must exit 0:\n{}",
        stderr(world)
    );
}
