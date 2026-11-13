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

/// A real data folder in the isolated root, holding a file of the user's.
fn data_dir(world: &E2eWorld) -> PathBuf {
    root(world).join("rocm-data")
}

/// The folder settings of a run that uninstalls only the config folder: the
/// config and data folders in the isolated root, so a run that wrongly removed
/// the data it was told to keep would show.
fn config_only_settings(world: &E2eWorld) -> Vec<(&'static str, String)> {
    vec![
        (
            "ROCM_CLI_CONFIG_DIR",
            config_dir(world).display().to_string(),
        ),
        ("ROCM_CLI_DATA_DIR", data_dir(world).display().to_string()),
    ]
}

/// [`run_uninstall_args`] with [`config_only_settings`].
fn run_config_only(world: &mut E2eWorld, args: &[&str]) {
    let settings = config_only_settings(world);
    let settings = settings
        .iter()
        .map(|(name, value)| (*name, value.as_str()))
        .collect::<Vec<_>>();
    run_uninstall_args(world, args, &settings);
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

#[given("the data folder holds the user's files")]
async fn data_holds_user_files(world: &mut E2eWorld) {
    let data = data_dir(world);
    std::fs::create_dir_all(&data).expect("failed to create data folder");
    std::fs::write(data.join(USER_FILE), "user data").expect("failed to write user file");
}

#[given("the cache folder is reached through a link that points at itself")]
async fn cache_behind_looping_link(world: &mut E2eWorld) {
    let looping = root(world).join("loop");
    E2eWorld::link_directory(&looping, &looping).expect("failed to plant looping link");
}

/// HOME, the uv cache and the Hugging Face cache inside the scenario's own
/// folder, so nothing a run resolves can reach the runner's shared caches.
fn isolated_env(world: &E2eWorld) -> Vec<(&'static str, String)> {
    let root = root(world);
    let home = root.join("home");
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    vec![
        ("HOME", home.display().to_string()),
        ("UV_CACHE_DIR", root.join("uv-cache").display().to_string()),
        ("HF_HOME", root.join("hf-home").display().to_string()),
    ]
}

/// A confirmed uninstall that keeps the binaries and whatever `keep` names.
fn run_uninstall(world: &mut E2eWorld, keep: &[&str], settings: &[(&str, &str)]) {
    let mut args = vec!["uninstall", "--keep-binaries", "--yes"];
    args.extend_from_slice(keep);
    run_uninstall_args(world, &args, settings);
}

/// Run `rocm` with exactly `args`, confined to the scenario's own folders like
/// every other uninstall here.
fn run_uninstall_args(world: &mut E2eWorld, args: &[&str], settings: &[(&str, &str)]) {
    assert_keeps_binaries(args);
    let isolated = isolated_env(world);
    let mut envs = isolated
        .iter()
        .map(|(name, value)| (*name, value.as_str()))
        .collect::<Vec<_>>();
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

/// Without `--yes` the run reaches the confirmation gate; the harness gives the
/// binary no terminal, which is the case the gate has to refuse.
#[when("the user uninstalls the config without confirming")]
async fn uninstall_config_unconfirmed(world: &mut E2eWorld) {
    run_config_only(world, &config_only_args(false));
}

/// Run `rocm uninstall` of the config folder only, without `--yes`, under a real
/// terminal, wait for the confirmation prompt and send `keys` as the answer.
///
/// The piped harness can never reach the prompt (no terminal, so the command is
/// refused first); a pseudo-terminal is the only way to answer it. The screen
/// the user would be looking at is kept as the scenario's output.
async fn answer_uninstall_prompt(world: &mut E2eWorld, keys: &str) {
    let root = root(world).to_path_buf();
    let home = root.join("home");
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let home = home.display().to_string();
    let uv = root.join("uv-cache").display().to_string();
    let hf = root.join("hf-home").display().to_string();
    let config = config_dir(world).display().to_string();
    let mut session = TuiSession::spawn_with_env(
        world,
        &[
            "uninstall",
            "--keep-binaries",
            "--keep-data",
            "--keep-cache",
        ],
        &[
            ("HOME", home.as_str()),
            ("UV_CACHE_DIR", uv.as_str()),
            ("HF_HOME", hf.as_str()),
            ("ROCM_CLI_CONFIG_DIR", config.as_str()),
        ],
    )
    .unwrap_or_else(|e| panic!("failed to open the uninstall prompt: {e}"));
    session
        .wait_for_screen("Proceed with uninstall? [y/N]:", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the confirmation prompt never appeared: {e}"));
    session
        .send(keys)
        .unwrap_or_else(|e| panic!("failed to answer the prompt: {e}"));
    let rc = session
        .wait_for_exit_code(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the CLI never exited after the answer: {e}"));
    world.cli_output = Some(session.screen_text());
    world.cli_rc = Some(rc);
    world.tui = Some(session);
}

#[when(
    "the user is asked to confirm an uninstall on a real terminal and types an answer that is neither yes nor no"
)]
async fn uninstall_prompt_unrecognized(world: &mut E2eWorld) {
    answer_uninstall_prompt(world, "maybe\r").await;
}

#[when(
    "the user is asked to confirm an uninstall on a real terminal and ends the input instead of answering"
)]
async fn uninstall_prompt_end_of_input(world: &mut E2eWorld) {
    // Ctrl-D on an empty line: the terminal reports end of input and echoes
    // nothing, which is what left the cursor on the prompt line.
    answer_uninstall_prompt(world, "\x04").await;
}

#[when("the user is asked to confirm an uninstall on a real terminal and types no")]
async fn uninstall_prompt_no(world: &mut E2eWorld) {
    answer_uninstall_prompt(world, "no\r").await;
}

/// The screen's lines with trailing padding removed.
fn screen_lines(world: &E2eWorld) -> Vec<String> {
    output(world)
        .lines()
        .map(|l| l.trim_end().to_owned())
        .collect()
}

#[then("the user is told the answer was not understood and is treated as no")]
async fn told_answer_not_understood(world: &mut E2eWorld) {
    let lines = screen_lines(world);
    let expected = "\"maybe\" is not yes or no; treating it as no.";
    assert!(
        lines.iter().any(|l| l == expected),
        "expected a `{expected}` line on the terminal:\n{}",
        output(world)
    );
}

#[then("the user is not told the answer was misunderstood")]
async fn not_told_answer_misunderstood(world: &mut E2eWorld) {
    assert!(
        !output(world).contains("is not yes or no"),
        "a plain no must not be commented on:\n{}",
        output(world)
    );
}

#[then("the uninstall is reported as cancelled")]
async fn uninstall_reported_cancelled(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    assert_eq!(
        rc,
        0,
        "declining is not a failure, got exit {rc}:\n{}",
        output(world)
    );
    assert!(
        screen_lines(world)
            .iter()
            .any(|l| l == "uninstall cancelled"),
        "expected an `uninstall cancelled` line on its own:\n{}",
        output(world)
    );
}

#[then("the cancellation is reported on its own line, not on the prompt line")]
async fn cancellation_on_its_own_line(world: &mut E2eWorld) {
    let lines = screen_lines(world);
    let prompt = lines
        .iter()
        .find(|l| l.starts_with("Proceed with uninstall? [y/N]:"))
        .unwrap_or_else(|| panic!("no prompt line on the terminal:\n{}", output(world)));
    assert_eq!(
        prompt,
        "Proceed with uninstall? [y/N]:",
        "nothing may follow the prompt after the input ended:\n{}",
        output(world)
    );
    assert!(
        lines.iter().any(|l| l == "uninstall cancelled"),
        "expected an `uninstall cancelled` line on its own:\n{}",
        output(world)
    );
    assert_eq!(
        world.cli_rc,
        Some(0),
        "ending the input declines, it does not fail:\n{}",
        output(world)
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
        "running the command the previous run named did not remove {}:\n{}",
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

/// The assistant's uninstall preview, called without a model through the hidden
/// `rocm mcp-call` entry point the dashboard and chat use for their tools.
///
/// The one run here that cannot pass `--keep-binaries`: the preview takes no
/// flags, plans every folder and the binary under test, and renders the plan. It
/// removes nothing, and it is made without `--allow-mutation`, so the tool
/// refuses a call that would remove something instead of running it.
#[when("the assistant previews an uninstall")]
async fn assistant_previews_uninstall(world: &mut E2eWorld) {
    let config = config_dir(world).display().to_string();
    let isolated = isolated_env(world);
    let mut envs = isolated
        .iter()
        .map(|(name, value)| (*name, value.as_str()))
        .collect::<Vec<_>>();
    envs.push(("ROCM_CLI_CONFIG_DIR", &config));
    let arguments = serde_json::json!({ "args": ["uninstall", "--dry-run"] }).to_string();
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["mcp-call", "rocm_command", "--arguments-json", &arguments],
        &envs,
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// What the preview printed: the command's own text inside the tool result.
fn preview_text(world: &E2eWorld) -> String {
    let result: serde_json::Value = serde_json::from_str(output(world)).unwrap_or_else(|error| {
        panic!("the tool result is not JSON ({error}):\n{}", output(world))
    });
    result["structuredContent"]["stdout"]
        .as_str()
        .unwrap_or_else(|| panic!("the tool result has no command output:\n{}", output(world)))
        .to_owned()
}

#[then("the preview succeeds")]
async fn preview_succeeds(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    assert!(
        rc == 0,
        "the preview must exit 0, got {rc}\nstdout:\n{}\nstderr:\n{}",
        output(world),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    let result: serde_json::Value = serde_json::from_str(output(world)).expect("tool result JSON");
    assert_eq!(
        result["isError"],
        serde_json::Value::Bool(false),
        "the tool reported an error:\n{}",
        output(world)
    );
}

#[then("the preview lists the config folder")]
async fn preview_lists_the_config(world: &mut E2eWorld) {
    let text = preview_text(world);
    let config = config_dir(world).display().to_string();
    assert!(
        text.contains(&format!("config: {config}")),
        "the preview must list the config folder ({config}):\n{text}"
    );
}

#[then("the preview says removal needs approval")]
async fn preview_says_removal_needs_approval(world: &mut E2eWorld) {
    let text = preview_text(world);
    assert!(
        text.contains("Nothing was removed. Removal needs your approval."),
        "the preview must say removal needs approval:\n{text}"
    );
    // The preview has no flag to offer, so it must not point at one.
    for phrase in ["run it again", "--dry-run.", "Review uninstall"] {
        assert!(
            !text.contains(phrase),
            "the assistant's preview must not offer {phrase:?}:\n{text}"
        );
    }
}

#[when("the user uninstalls the config with --force")]
async fn uninstall_with_force(world: &mut E2eWorld) {
    let mut args = config_only_args(false);
    args.push("--force");
    run_config_only(world, &args);
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

/// The refusals in `uninstall-16` and `uninstall-17` both come from the
/// config-only uninstall (`--force` added for the second), and end with the
/// command the user meant: that uninstall with `--yes`, and without `--force`.
#[then("the refusal ends by suggesting the same uninstall, confirmed")]
async fn refusal_suggests_confirmed_uninstall(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let mut expected = vec!["rocm"];
    expected.extend(config_only_args(false));
    expected.push("--yes");
    assert_eq!(
        rocm_core::recovery_hint::try_command(stderr),
        Some(expected.join(" ").as_str()),
        "the refusal must end with `Try: {}`:\n{stderr}",
        expected.join(" ")
    );
}

#[when("the user runs the command the refusal suggests")]
async fn run_suggested_command(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let advised = rocm_core::recovery_hint::try_command(&stderr)
        .unwrap_or_else(|| panic!("the refusal suggests no command:\n{stderr}"))
        .to_owned();
    // Run as the user would copy it, in the same shell: the folder settings are
    // the environment's, and the advice only repeats the command line.
    let argv: Vec<&str> = advised.split_whitespace().collect();
    assert_eq!(argv.first(), Some(&"rocm"), "{advised}");
    run_config_only(world, &argv[1..]);
}

#[then("the data folder still holds the user's files")]
async fn data_intact(world: &mut E2eWorld) {
    let file = data_dir(world).join(USER_FILE);
    assert!(
        file.is_file(),
        "uninstall removed {}, which it was told to keep:\n{}",
        file.display(),
        output(world)
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
        stderr.contains("unexpected argument"),
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
