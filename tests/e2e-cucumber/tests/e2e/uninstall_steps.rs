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
use crate::e2e::tui_driver::{TuiSession, default_timeout};

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

/// A real config folder in the isolated root, holding a file of the user's.
fn config_dir(world: &E2eWorld) -> PathBuf {
    root(world).join("rocm-config")
}

/// A cache setting the kernel cannot check: its parent is a link to itself.
fn looping_cache(world: &E2eWorld) -> PathBuf {
    root(world).join("loop").join("cache")
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

#[given("the config folder holds the user's files")]
async fn config_holds_user_files(world: &mut E2eWorld) {
    let config = config_dir(world);
    std::fs::create_dir_all(&config).expect("failed to create config folder");
    std::fs::write(config.join(USER_FILE), "user data").expect("failed to write user file");
}

#[given("the cache folder is reached through a link that points at itself")]
async fn cache_behind_looping_link(world: &mut E2eWorld) {
    let looping = root(world).join("loop");
    E2eWorld::link_directory(&looping, &looping).expect("failed to plant looping link");
}

fn run_uninstall(world: &mut E2eWorld, keep: &[&str], settings: &[(&str, &str)]) {
    run_uninstall_with(world, true, keep, settings);
}

/// `confirmed: false` leaves `--yes` off, so the run reaches the confirmation
/// gate; the harness gives the binary no terminal, which is the case the gate
/// has to refuse.
fn run_uninstall_with(
    world: &mut E2eWorld,
    confirmed: bool,
    keep: &[&str],
    settings: &[(&str, &str)],
) {
    let mut args = vec!["uninstall", "--keep-binaries"];
    if confirmed {
        args.push("--yes");
    }
    args.extend_from_slice(keep);
    run_uninstall_args(world, &args, settings);
}

/// Run `rocm` with exactly `args`, confined to the scenario's own folders like
/// every other uninstall here.
fn run_uninstall_args(world: &mut E2eWorld, args: &[&str], settings: &[(&str, &str)]) {
    assert_keeps_binaries(args);
    let root = root(world).to_path_buf();
    let home = root.join("home");
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let home = home.display().to_string();
    let uv = root.join("uv-cache").display().to_string();
    let hf = root.join("hf-home").display().to_string();
    let mut envs = vec![
        ("HOME", home.as_str()),
        ("UV_CACHE_DIR", &uv),
        ("HF_HOME", &hf),
    ];
    envs.extend_from_slice(settings);
    let (stdout, stderr, rc) = crate::run_rocm_with_env(world, args, &envs);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn run_cache_only_uninstall(world: &mut E2eWorld, cache_setting: &str) {
    run_uninstall(
        world,
        &["--keep-config", "--keep-data"],
        &[("ROCM_CLI_CACHE_DIR", cache_setting)],
    );
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

#[when("the user uninstalls the config and the cache")]
async fn uninstall_config_and_cache(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    let cache = looping_cache(world).display().to_string();
    run_uninstall(
        world,
        &["--keep-data"],
        &[
            ("ROCM_CLI_CONFIG_DIR", &config),
            ("ROCM_CLI_CACHE_DIR", &cache),
        ],
    );
}

#[when("the user uninstalls the config without confirming")]
async fn uninstall_config_unconfirmed(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    run_uninstall_with(
        world,
        false,
        &["--keep-data", "--keep-cache"],
        &[("ROCM_CLI_CONFIG_DIR", &config)],
    );
}

/// The scenario's own home folder, standing in for the user's.
fn home_dir(world: &E2eWorld) -> PathBuf {
    root(world).join("home")
}

/// The cache folder a default install in [`home_dir`] uses (`~/.rocm/cache`).
fn default_cache(world: &E2eWorld) -> PathBuf {
    home_dir(world).join(".rocm").join("cache")
}

/// Where the cache folder belongs when only the config and data folders are
/// set: inside the scenario's data folder, the one `isolate_env` hands the CLI.
fn data_cache(world: &E2eWorld) -> PathBuf {
    world.app_paths().data_dir.join("cache")
}

/// Run `rocm` with the config and data folders set but no cache folder, as an
/// install isolated by `ROCM_CLI_CONFIG_DIR` and `ROCM_CLI_DATA_DIR` alone is.
/// HOME (and USERPROFILE, which is where Windows finds it) is the scenario's
/// own, so a cache resolved to the default lands in [`default_cache`], not the
/// runner's.
fn run_without_cache_setting(world: &mut E2eWorld, args: &[&str]) {
    let root = root(world).to_path_buf();
    let home = home_dir(world);
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let home = home.display().to_string();
    let uv = root.join("uv-cache").display().to_string();
    let hf = root.join("hf-home").display().to_string();
    let envs = [
        ("HOME", home.as_str()),
        ("USERPROFILE", home.as_str()),
        ("UV_CACHE_DIR", uv.as_str()),
        ("HF_HOME", hf.as_str()),
    ];
    let (stdout, stderr, rc) =
        crate::run_rocm_with_env_and_unset(world, args, &envs, &["ROCM_CLI_CACHE_DIR"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[given("a default install's cache folder in the user's home holds the user's files")]
async fn default_cache_holds_user_files(world: &mut E2eWorld) {
    let cache = default_cache(world);
    std::fs::create_dir_all(&cache).expect("failed to create the default cache folder");
    std::fs::write(cache.join(USER_FILE), "user data").expect("failed to write user file");
}

#[given("the data folder's own cache folder holds a cache file")]
async fn data_cache_holds_file(world: &mut E2eWorld) {
    let cache = data_cache(world);
    std::fs::create_dir_all(&cache).expect("failed to create the data folder's cache folder");
    std::fs::write(cache.join("cached.bin"), "cached").expect("failed to write cache file");
}

#[when("the user asks what ROCm CLI keeps on disk with only the config and data folders set")]
async fn storage_without_cache_setting(world: &mut E2eWorld) {
    run_without_cache_setting(world, &["storage", "report"]);
}

#[when("the user uninstalls with only the config and data folders set")]
async fn uninstall_without_cache_setting(world: &mut E2eWorld) {
    run_without_cache_setting(world, &["uninstall", "--yes", "--keep-binaries"]);
}

#[then("the storage report names the cache folder inside the data folder")]
async fn storage_names_data_cache(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    let stdout = output(world);
    assert_eq!(rc, 0, "storage report must exit 0:\n{stdout}");
    let row = stdout
        .lines()
        .find(|line| line.trim_start().starts_with("- ROCm CLI cache folder:"))
        .unwrap_or_else(|| panic!("the report must name the cache folder:\n{stdout}"));
    let expected = format!("({})", data_cache(world).display());
    assert!(
        row.ends_with(&expected),
        "the cache folder must be the one inside the data folder {expected}, got `{row}`:\n{stdout}"
    );
}

#[then("the uninstall does not name the default install's cache folder")]
async fn uninstall_skips_default_cache(world: &mut E2eWorld) {
    let stdout = output(world);
    let default = default_cache(world).display().to_string();
    assert!(
        !stdout.contains(&default),
        "an uninstall of an isolated install must not touch {default}:\n{stdout}"
    );
    // It removed exactly one cache folder, and that one is the isolated
    // install's own; zero removals must not pass.
    let expected = format!("removed cache {}", data_cache(world).display());
    let removals: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("removed cache "))
        .collect();
    assert_eq!(
        removals,
        vec![expected.as_str()],
        "the uninstall must report removing exactly the data folder's cache:\n{stdout}"
    );
}

#[then("the data folder's own cache folder is gone")]
async fn data_cache_gone(world: &mut E2eWorld) {
    let cache = data_cache(world);
    assert!(
        !cache.exists(),
        "the uninstall left {} behind:
{}",
        cache.display(),
        output(world)
    );
}

#[then("the default install's cache folder still holds the user's files")]
async fn default_cache_intact(world: &mut E2eWorld) {
    let file = default_cache(world).join(USER_FILE);
    assert!(
        file.is_file(),
        "uninstalling the isolated install deleted {}:\n{}",
        file.display(),
        output(world)
    );
}

/// Every place the CLI keeps state when only the config and data folders are
/// set: those two, the cache folder inside the data folder, and the default
/// `~/.rocm` that a stray write would land in. The harness's own cache folder
/// is left out because a runner may share it between scenarios.
fn state_folders(world: &E2eWorld) -> Vec<PathBuf> {
    let paths = world.app_paths();
    vec![
        paths.config_dir,
        paths.data_dir.clone(),
        paths.data_dir.join("cache"),
        home_dir(world).join(".rocm"),
    ]
}

#[given("ROCm CLI has never created its folders in the user's home")]
async fn no_folders_yet(world: &mut E2eWorld) {
    // The harness creates the config, data and cache folders empty before each
    // scenario; a machine ROCm CLI never touched has none of them.
    for sub in ["config", "data", "cache"] {
        let folder = root(world).join(sub);
        std::fs::remove_dir(&folder)
            .unwrap_or_else(|e| panic!("failed to remove the empty {}: {e}", folder.display()));
    }
    for folder in state_folders(world) {
        assert!(
            std::fs::symlink_metadata(&folder).is_err(),
            "the scenario must start without {}",
            folder.display()
        );
    }
}

#[when("the user dry-runs an uninstall")]
async fn dry_run_uninstall(world: &mut E2eWorld) {
    run_without_cache_setting(world, &["uninstall", "--keep-binaries", "--dry-run"]);
}

#[then("the uninstall reports that nothing would be removed")]
async fn reports_nothing_to_remove(world: &mut E2eWorld) {
    let stdout = output(world);
    assert!(
        stdout.contains("Nothing managed by rocm-cli would be removed."),
        "a machine with no folders has nothing to remove:\n{stdout}"
    );
}

#[then("ROCm CLI has still not created its folders")]
async fn still_no_folders(world: &mut E2eWorld) {
    for folder in state_folders(world) {
        assert!(
            std::fs::symlink_metadata(&folder).is_err(),
            "the dry run created {}:\n{}",
            folder.display(),
            output(world)
        );
    }
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

#[then("the uninstall fails")]
async fn uninstall_fails(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    assert!(
        rc != 0,
        "uninstall must fail, got exit 0\nstdout:\n{}",
        output(world)
    );
}

#[then("the uninstall is refused and the user is told to confirm explicitly")]
async fn uninstall_refused_without_yes(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert_eq!(
        rc,
        1,
        "a refused uninstall must exit 1\nstdout:\n{}\nstderr:\n{stderr}",
        output(world)
    );
    assert!(
        stderr.contains("uninstall requires --yes outside an interactive terminal"),
        "expected the --yes refusal on stderr:\n{stderr}"
    );
    // The help promises the plan is printed before the refusal, so the person
    // who is refused can see what `--yes` would have removed.
    let config = config_dir(world).display().to_string();
    assert!(
        output(world).contains(&config),
        "the plan must name the config folder ({config}) before refusing:\n{}",
        output(world)
    );
}

#[then("the error names the cache folder it could not check")]
async fn error_names_cache(world: &mut E2eWorld) {
    let expected = format!(
        "failed to check the cache folder {}",
        looping_cache(world).display()
    );
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains(&expected),
        "expected `{expected}` on stderr:\n{stderr}"
    );
}

#[then("the config folder still holds the user's files")]
async fn config_intact(world: &mut E2eWorld) {
    let file = config_dir(world).join(USER_FILE);
    assert!(
        file.is_file(),
        "uninstall removed {} although the plan failed:\n{}",
        file.display(),
        output(world)
    );
}

/// Every run here keeps the binaries: the binary under test must survive, and an
/// uninstall that reached it would delete the program the scenario is running.
/// Checked at the point a run starts, so no step can reach it by accident.
fn assert_keeps_binaries(args: &[&str]) {
    assert!(
        args.contains(&"--keep-binaries"),
        "an uninstall in a scenario must pass --keep-binaries: {args:?}"
    );
}

/// The arguments for uninstalling only the config folder, as a dry run or for real.
fn config_only_args(dry_run: bool) -> Vec<&'static str> {
    let mut args = vec![
        "uninstall",
        "--keep-binaries",
        "--keep-data",
        "--keep-cache",
    ];
    if dry_run {
        args.push("--dry-run");
    }
    args
}

/// Run `rocm` under a pseudo-terminal, confined to the scenario's folders like
/// every other uninstall here, so the command sees a terminal on both streams as
/// a person typing it would. Answers its `[y/N]` prompt when `answer` is given.
async fn run_uninstall_at_terminal(world: &mut E2eWorld, args: &[&str], answer: Option<&str>) {
    assert_keeps_binaries(args);
    let root = root(world).to_path_buf();
    let home = root.join("home");
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let home = home.display().to_string();
    let uv = root.join("uv-cache").display().to_string();
    let hf = root.join("hf-home").display().to_string();
    let config = config_dir(world).display().to_string();
    let mut session = TuiSession::spawn_with_env(
        world,
        args,
        &[
            ("HOME", home.as_str()),
            ("UV_CACHE_DIR", uv.as_str()),
            ("HF_HOME", hf.as_str()),
            ("ROCM_CLI_CONFIG_DIR", config.as_str()),
        ],
    )
    .unwrap_or_else(|e| panic!("failed to run {args:?} at a terminal: {e}"));
    if let Some(answer) = answer {
        session
            .wait_for_screen("[y/N]:", default_timeout())
            .await
            .unwrap_or_else(|e| panic!("the confirmation prompt never appeared: {e}"));
        session
            .send(&format!("{answer}\r"))
            .unwrap_or_else(|e| panic!("failed to type the answer: {e}"));
    }
    let rc = session
        .wait_for_exit_code(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the uninstall never exited: {e}"));
    world.cli_output = Some(session.screen_text());
    world.cli_stderr = None;
    world.cli_rc = Some(rc);
    world.tui = Some(session);
}

#[when("the user dry-runs an uninstall of only the config")]
async fn dry_run_config_only(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    run_uninstall_args(
        world,
        &config_only_args(true),
        &[("ROCM_CLI_CONFIG_DIR", &config)],
    );
}

#[when("the user dry-runs an uninstall of only the config at a terminal")]
async fn dry_run_config_only_at_terminal(world: &mut E2eWorld) {
    run_uninstall_at_terminal(world, &config_only_args(true), None).await;
}

#[when("the user runs it again without --dry-run at a terminal and answers yes")]
async fn run_it_again_and_confirm(world: &mut E2eWorld) {
    // The same arguments as the dry run, without `--dry-run`: the step the
    // dry run's last line names.
    run_uninstall_at_terminal(world, &config_only_args(false), Some("y")).await;
}

#[when("the user dry-runs an uninstall that keeps everything at a terminal")]
async fn dry_run_keeping_everything_at_terminal(world: &mut E2eWorld) {
    run_uninstall_at_terminal(
        world,
        &[
            "uninstall",
            "--keep-binaries",
            "--keep-config",
            "--keep-data",
            "--keep-cache",
            "--dry-run",
        ],
        None,
    )
    .await;
}

#[then("the dry run says to run it again without --dry-run")]
async fn dry_run_says_to_run_it_again(world: &mut E2eWorld) {
    let stdout = output(world);
    assert!(
        stdout.contains(
            "Nothing was removed. To remove these items, run it again without --dry-run."
        ),
        "the dry run must say how to remove what it listed:\n{stdout}"
    );
    assert!(
        !stdout.contains("Review uninstall"),
        "the command line has no such control:\n{stdout}"
    );
}

#[then("the dry run lists the config folder")]
async fn dry_run_lists_the_config(world: &mut E2eWorld) {
    let stdout = output(world);
    let config = config_dir(world).display().to_string();
    assert!(
        stdout.contains(&format!("config: {config}")),
        "the plan must list the config folder ({config}):\n{stdout}"
    );
}

#[then("the config folder is gone")]
async fn config_gone(world: &mut E2eWorld) {
    let config = config_dir(world);
    assert!(
        std::fs::symlink_metadata(&config).is_err(),
        "running it again as the dry run said did not remove {}:\n{}",
        config.display(),
        output(world)
    );
}

#[then("the dry run offers no step to remove anything")]
async fn dry_run_offers_no_step(world: &mut E2eWorld) {
    let stdout = output(world);
    for phrase in [
        "Nothing was removed.",
        "run it again",
        "Review uninstall",
        "approval",
    ] {
        assert!(
            !stdout.contains(phrase),
            "this dry run must not offer {phrase:?}:\n{stdout}"
        );
    }
}

#[when("the user uninstalls the config with --force")]
async fn uninstall_with_force(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    let mut args = config_only_args(false);
    args.push("--force");
    run_uninstall_args(world, &args, &[("ROCM_CLI_CONFIG_DIR", &config)]);
}

#[when("the user uninstalls the config with a near miss of --force")]
async fn uninstall_with_near_miss_of_force(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    let mut args = config_only_args(false);
    args.push("--forc");
    run_uninstall_args(world, &args, &[("ROCM_CLI_CONFIG_DIR", &config)]);
}

#[then("the uninstall is refused and told to use --yes instead")]
async fn refused_and_told_to_use_yes(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert_eq!(
        world.cli_rc,
        Some(1),
        "a refused --force exits 1\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("`--force` is not an option of `rocm uninstall`")
            && stderr.contains("use `--yes` to skip the confirmation"),
        "expected the refusal to name --yes:\n{stderr}"
    );
    assert!(
        !stderr.contains("force-dev-binaries"),
        "the refusal must not name the destructive flag:\n{stderr}"
    );
}

#[then("the error does not point at the flag that removes development binaries")]
async fn error_does_not_suggest_dev_binaries(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert_eq!(
        world.cli_rc,
        Some(2),
        "an unknown flag exits 2\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("unexpected argument '--forc' found"),
        "expected the unknown-argument error on stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("force-dev-binaries"),
        "the error must not name the destructive flag:\n{stderr}"
    );
}

#[when("the user asks for the uninstall help")]
async fn ask_for_uninstall_help(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(world, &["uninstall", "--help"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the help shows a dry run before the real run")]
async fn help_shows_dry_run_first(world: &mut E2eWorld) {
    let stdout = output(world);
    let examples = stdout.split_once("EXAMPLES:").map_or_else(
        || panic!("the help has no examples:\n{stdout}"),
        |(_, rest)| rest,
    );
    let rows: Vec<&str> = examples
        .lines()
        .map(str::trim)
        .filter(|row| !row.is_empty())
        .collect();
    assert_eq!(
        rows.first().copied(),
        Some("rocm uninstall --dry-run"),
        "the first example must be the dry run:\n{examples}"
    );
    assert!(
        rows.contains(&"rocm uninstall"),
        "the examples must also show the real run:\n{examples}"
    );
}
