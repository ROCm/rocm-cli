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

/// The scenario's config folder (absent until a step or the CLI creates it).
fn config_dir(world: &E2eWorld) -> PathBuf {
    config_dir_at(root(world))
}

/// Inside the scenario's home, where ROCm CLI's folders normally live: on
/// Windows the scenario root is itself under `C:\\Users\\<runner>`, so a folder
/// outside the scenario home would count as another user's home there.
fn config_dir_at(root: &Path) -> PathBuf {
    home_at(root).join(".rocm").join("config")
}

fn home_at(root: &Path) -> PathBuf {
    root.join("users").join("home")
}

/// The scenario's own HOME.
fn home(world: &E2eWorld) -> PathBuf {
    // One level down, so the scenario's config/data/cache folders are not
    // siblings of HOME (which `rocm uninstall` treats as other users' homes).
    root(world).join("users").join("home")
}

fn user_file_in_home(world: &E2eWorld) -> PathBuf {
    home(world).join("Documents").join(USER_FILE)
}

/// The scenario's config folder, as ROCm CLI would have left it: created by
/// the CLI and so carrying the marker that lets `rocm uninstall` remove a
/// folder outside the home folder.
fn plant_config_rocm_cli_created(world: &E2eWorld) {
    let config = config_dir(world);
    std::fs::create_dir_all(&config).expect("failed to create config");
    std::fs::write(config.join(".rocm-cli-root"), "created by ROCm CLI\n")
        .expect("failed to mark config");
}

fn set_env(world: &mut E2eWorld, key: &'static str, value: impl Into<OsString>) {
    world.command_env.retain(|(existing, _)| *existing != key);
    world.command_env.push((key, value.into()));
}

/// The environment every run here gets: HOME (and USERPROFILE, which is where
/// Windows finds it), the config, data and cache folders, the uv cache and the
/// Hugging Face cache, all inside the scenario's own folder, so nothing a run
/// resolves can reach the runner's shared caches. A Given step overrides one
/// through `world.command_env`, which is kept (not consumed) so a scenario can
/// re-run the command with different flags.
fn isolated_env(world: &E2eWorld) -> Vec<(&'static str, OsString)> {
    let root = root(world).to_path_buf();
    let home = home(world);
    std::fs::create_dir_all(&home).expect("failed to create isolated HOME");
    let mut env: Vec<(&'static str, OsString)> = vec![
        ("HOME", home.clone().into_os_string()),
        // Windows reads the home folder from USERPROFILE first.
        ("USERPROFILE", home.into_os_string()),
        // Not the world's folders: the harness pre-creates those (so they are
        // not ones ROCm CLI created, and carry no marker), and on CI its cache
        // can be shared across scenarios, while an uninstall removes the cache
        // folder. These start absent, as on a machine before ROCm CLI ran.
        ("ROCM_CLI_CONFIG_DIR", config_dir_at(&root).into_os_string()),
        (
            "ROCM_CLI_DATA_DIR",
            home_at(&root).join(".rocm").join("data").into_os_string(),
        ),
        (
            "ROCM_CLI_CACHE_DIR",
            home_at(&root).join(".rocm").join("cache").into_os_string(),
        ),
        ("UV_CACHE_DIR", root.join("uv-cache").into_os_string()),
        ("HF_HOME", root.join("hf-home").into_os_string()),
    ];
    for (key, value) in &world.command_env {
        env.retain(|(existing, _)| existing != key);
        env.push((key, value.clone()));
    }
    env
}

/// [`isolated_env`] as text, for the pseudo-terminal runner.
fn isolated_env_text(world: &E2eWorld) -> Vec<(&'static str, String)> {
    isolated_env(world)
        .into_iter()
        .map(|(key, value)| (key, value.to_string_lossy().into_owned()))
        .collect()
}

/// Run `rocm uninstall --keep-binaries <flags>` confined to the scenario root.
fn run_uninstall(world: &mut E2eWorld, flags: &[&str]) {
    run_uninstall_in(world, flags, None);
}

/// [`run_uninstall`] with the working directory set, for a relative setting.
fn run_uninstall_in(world: &mut E2eWorld, flags: &[&str], cwd: Option<&Path>) {
    let mut args = vec!["uninstall", "--keep-binaries"];
    args.extend_from_slice(flags);
    assert_keeps_binaries(&args);
    run_command_in(world, &args, cwd);
}

/// Run `rocm <args>` confined to the scenario root (see [`isolated_env`]).
fn run_command_in(world: &mut E2eWorld, args: &[&str], cwd: Option<&Path>) {
    let env = isolated_env(world);
    let binary = crate::rocm_binary();
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    world.isolate_cmd(&mut cmd);
    for (key, value) in env {
        cmd.env(key, value);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));
    let rc = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    crate::record_command(world.current_scenario.as_deref(), args, rc, &stdout);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(String::from_utf8_lossy(&output.stderr).to_string());
    world.cli_rc = Some(rc);
}

/// Run `rocm` under a pseudo-terminal, confined to the scenario's folders like
/// every other uninstall here, so the command sees a terminal on both streams as
/// a person typing it would.
fn spawn_at_terminal(world: &E2eWorld, args: &[&str]) -> TuiSession {
    assert_keeps_binaries(args);
    let env = isolated_env_text(world);
    let env = env
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect::<Vec<_>>();
    TuiSession::spawn_with_env(world, args, &env)
        .unwrap_or_else(|e| panic!("failed to run {args:?} at a terminal: {e}"))
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

/// A cache setting the kernel cannot check: its parent is a link to itself.
fn looping_cache(world: &E2eWorld) -> PathBuf {
    root(world).join("loop").join("cache")
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
    plant_config_rocm_cli_created(world);
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

#[when("the user uninstalls the config and the cache")]
async fn uninstall_config_and_cache(world: &mut E2eWorld) {
    let cache = looping_cache(world);
    set_env(world, "ROCM_CLI_CACHE_DIR", cache);
    run_uninstall(world, &["--yes", "--keep-data"]);
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
        .split("e-run with ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .filter(|word| word.starts_with("--keep-"))
        .unwrap_or_else(|| panic!("the refusal named no `re-run with --keep-...`:\n{message}"))
        .to_owned();
    run_uninstall(world, &["--yes", &flag]);
}

#[when("the user uninstalls the config without confirming")]
async fn uninstall_config_unconfirmed(world: &mut E2eWorld) {
    // No `--yes`, so the run reaches the confirmation gate; the harness gives
    // the binary no terminal, which is the case the gate has to refuse.
    run_uninstall(world, &["--keep-data", "--keep-cache"]);
}

/// Run `rocm uninstall` of the config folder only, without `--yes`, under a real
/// terminal, wait for the confirmation prompt and send `keys` as the answer.
///
/// The piped harness can never reach the prompt (no terminal, so the command is
/// refused first); a pseudo-terminal is the only way to answer it. The screen
/// the user would be looking at is kept as the scenario's output.
async fn answer_uninstall_prompt(world: &mut E2eWorld, keys: &str) {
    let mut session = spawn_at_terminal(world, &config_only_args(false));
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

/// The cache folder a default install in [`home`] uses (`~/.rocm/cache`).
fn default_cache(world: &E2eWorld) -> PathBuf {
    home(world).join(".rocm").join("cache")
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
    let home = home(world);
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
    // The harness's config and data folders sit outside the scenario's home,
    // where `rocm uninstall` removes only a folder ROCm CLI marked as its own.
    // They stand for an install ROCm CLI created, so they carry its marker,
    // and so does the cache folder it would have created inside the data one.
    let paths = world.app_paths();
    for folder in [paths.config_dir, paths.data_dir, cache] {
        std::fs::write(folder.join(".rocm-cli-root"), "created by ROCm CLI\n")
            .expect("failed to mark the folder as ROCm CLI's own");
    }
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
        home(world).join(".rocm"),
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
    // Followed by where the folder came from, in parentheses.
    let refused_line = format!("  - data: {folder} is {why} (");
    assert!(
        stdout.lines().any(|line| line.starts_with(&refused_line)),
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
        message.contains("e-run with --keep-data to leave it in place."),
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
    let config = config_dir(world);
    assert!(
        config.is_dir(),
        "a refused uninstall removed {}:\n{}",
        config.display(),
        output(world)
    );
}

#[then("the config folder is gone")]
async fn config_gone(world: &mut E2eWorld) {
    let config = config_dir(world);
    // `symlink_metadata`, not `exists`: a folder left behind as a dangling
    // link is not gone.
    assert!(
        std::fs::symlink_metadata(&config).is_err(),
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
    let review_ends = stdout.find("Left alone:").unwrap_or(stdout.len());
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

#[when("the user uninstalls keeping the cache")]
async fn uninstall_keeping_cache(world: &mut E2eWorld) {
    run_uninstall(world, &["--yes", "--keep-cache"]);
}

#[then("the uv cache is still there")]
async fn uv_cache_kept(world: &mut E2eWorld) {
    let blob = home(world).join(".cache").join("uv").join("blob");
    assert!(
        blob.is_file(),
        "--keep-cache did not keep {}:\n{}",
        blob.display(),
        output(world)
    );
}

#[then("the uv cache is gone")]
async fn uv_cache_gone(world: &mut E2eWorld) {
    let cache = home(world).join(".cache");
    assert!(
        std::fs::symlink_metadata(&cache).is_err(),
        "the review said the uv cache WILL BE DELETED, but {} is still there:\n{}",
        cache.display(),
        output(world)
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

#[given(
    "the data folder is set to `.`, and the user is in their home folder, which holds their files"
)]
async fn data_is_dot_in_home(world: &mut E2eWorld) {
    let file = user_file_in_home(world);
    std::fs::create_dir_all(file.parent().expect("parent")).expect("failed to create home");
    std::fs::write(&file, "user data").expect("failed to write user file");
    plant_config_rocm_cli_created(world);
    set_env(world, "ROCM_CLI_DATA_DIR", ".");
}

#[when("the user uninstalls from their home folder")]
async fn uninstall_from_home(world: &mut E2eWorld) {
    let home = home(world);
    run_uninstall_in(world, &["--yes"], Some(&home));
}

#[then(expr = "the refusal names {word} as where the data folder came from")]
async fn refusal_names_source(world: &mut E2eWorld, setting: String) {
    let message = format!("{}{}", output(world), stderr(world));
    assert!(
        message.contains(&format!("(set by {setting}"))
            && message.contains(&format!("Point {setting} at a folder of ROCm CLI's own")),
        "the refusal must name {setting} and say to change it:\n{message}"
    );
}

#[then(expr = "the refusal says the data folder {string} resolves to the home folder")]
async fn refusal_says_resolves_to_home(world: &mut E2eWorld, folder: String) {
    let stdout = output(world);
    let home = crate::e2e::runtime_steps::canonical_without_verbatim(&home(world));
    let line = format!(
        "  - data: {folder} is your home folder (set by ROCM_CLI_DATA_DIR; resolves to {})",
        home.display()
    );
    assert!(
        stdout.lines().any(|l| l == line),
        "expected `{line}` in the review:\n{stdout}"
    );
    assert!(rc(world) != 0, "a refused uninstall must fail:\n{stdout}");
}

// ── The marker, and the dashboard's path ───────────────────────────

/// A data folder outside the home folder that ROCm CLI did not mark.
fn unmarked_data(world: &E2eWorld) -> PathBuf {
    root(world).join("srv").join("rocm")
}

#[given("the data folder is a folder outside the home folder without ROCm CLI's marker")]
async fn data_is_unmarked(world: &mut E2eWorld) {
    let data = unmarked_data(world);
    std::fs::create_dir_all(&data).expect("failed to create data folder");
    std::fs::write(data.join("keep.txt"), "user data").expect("failed to write");
    set_env(world, "ROCM_CLI_DATA_DIR", data);
}

#[then("the refusal advises creating the marker if the folder is ROCm CLI's")]
async fn advises_marker(world: &mut E2eWorld) {
    let stdout = output(world);
    let marker = crate::e2e::runtime_steps::canonical_without_verbatim(&unmarked_data(world))
        .join(".rocm-cli-root");
    assert!(rc(world) != 0, "a refused preview must fail:\n{stdout}");
    assert!(
        stdout.contains(&format!(
            "if this folder belongs to ROCm CLI, create {}",
            marker.display()
        )),
        "the refusal must name the marker to create:\n{stdout}"
    );
}

#[when("the user creates the marker the refusal named")]
async fn create_advised_marker(world: &mut E2eWorld) {
    // The marker the refusal names for the data folder, taken from the
    // message itself, so this follows the advice the user reads.
    let stdout = output(world).to_owned();
    let data = crate::e2e::runtime_steps::canonical_without_verbatim(&unmarked_data(world));
    let marker = stdout
        .split("create ")
        .skip(1)
        .filter_map(|rest| rest.split(", or").next())
        .map(str::trim)
        .find(|path| Path::new(path).parent() == Some(data.as_path()))
        .unwrap_or_else(|| panic!("no marker path for {} in:\n{stdout}", data.display()))
        .to_owned();
    std::fs::write(&marker, "").unwrap_or_else(|e| panic!("cannot create {marker}: {e}"));
}

#[then("the preview plans to remove the data folder")]
async fn preview_plans_data(world: &mut E2eWorld) {
    let stdout = output(world);
    assert!(
        rc(world) == 0,
        "the preview must succeed now:\n{stdout}\n{}",
        stderr(world)
    );
    let line = format!("  - data: {}", unmarked_data(world).display());
    assert!(
        stdout.lines().any(|l| l == line),
        "expected `{line}` in the review:\n{stdout}"
    );
    assert!(!stdout.contains("Refused"), "{stdout}");
}

#[given("setup.therock_venv points at the user's home folder, which holds their files")]
async fn venv_is_home(world: &mut E2eWorld) {
    let file = user_file_in_home(world);
    std::fs::create_dir_all(file.parent().expect("parent")).expect("failed to create home");
    std::fs::write(&file, "user data").expect("failed to write user file");
    let config = config_dir(world);
    plant_config_rocm_cli_created(world);
    std::fs::write(
        config.join("config.json"),
        serde_json::to_vec(&serde_json::json!({ "setup": { "therock_venv": home(world) } }))
            .expect("json"),
    )
    .expect("failed to write config.json");
    // An empty value is "unset" to ROCm CLI, so discovery falls back to
    // `setup.therock_venv` as it would for a user who never set the variable.
    set_env(world, "ROCM_CLI_DATA_DIR", "");
}

#[when("the dashboard runs an approved uninstall")]
async fn dashboard_uninstall(world: &mut E2eWorld) {
    // The same entry point the dashboard and the assistant use for an approved
    // `rocm_command`: it starts a `rocm` child with the folders pinned.
    let arguments = r#"{"args":["uninstall","--keep-binaries","--keep-cache"]}"#;
    run_command_in(
        world,
        &[
            "mcp-call",
            "rocm_command",
            "--allow-mutation",
            "--arguments-json",
            arguments,
        ],
        None,
    );
}

#[then(
    "the refusal names setup.therock_venv and advises --keep-data, not a variable the user never set"
)]
async fn dashboard_refusal_names_venv(world: &mut E2eWorld) {
    let stdout = output(world);
    // `mcp-call` itself succeeds and reports the child's result as JSON; the
    // approved `rocm uninstall` it ran must have failed.
    let reply: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("mcp-call did not print JSON ({e}):\n{stdout}"));
    let child_exit = reply["structuredContent"]["exit_status"].as_i64();
    assert!(
        child_exit.is_some_and(|code| code != 0)
            && reply["isError"] == serde_json::Value::Bool(true),
        "the approved uninstall must exit non-zero, got {child_exit:?}:\n{stdout}"
    );
    assert!(
        stdout.contains("set by setup.therock_venv"),
        "the refusal must name setup.therock_venv:\n{stdout}"
    );
    assert!(
        stdout.contains("Re-run with --keep-data to leave it in place."),
        "the refusal must advise --keep-data:\n{stdout}"
    );
    assert!(
        !stdout.contains("ROCM_CLI_DATA_DIR"),
        "the refusal named a variable the user never set:\n{stdout}"
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

#[then("the uninstall fails")]
async fn uninstall_fails(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code captured");
    assert!(
        rc != 0,
        "uninstall must fail, got exit 0\nstdout:\n{}",
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
async fn config_still_holds_user_files(world: &mut E2eWorld) {
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

/// Run `rocm` at a terminal (see [`spawn_at_terminal`]), answering its `[y/N]`
/// prompt when `answer` is given.
async fn run_uninstall_at_terminal(world: &mut E2eWorld, args: &[&str], answer: Option<&str>) {
    let mut session = spawn_at_terminal(world, args);
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
    run_uninstall(world, &["--keep-data", "--keep-cache", "--dry-run"]);
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
/// The one run here made without `--keep-binaries`: called with no flags, the
/// preview plans every folder and the binary under test, and renders the plan.
/// It removes nothing, and it is made without `--allow-mutation`, so the tool
/// refuses a call that would remove something instead of running it.
#[when("the assistant previews an uninstall")]
async fn assistant_previews_uninstall(world: &mut E2eWorld) {
    let arguments = serde_json::json!({ "args": ["uninstall", "--dry-run"] }).to_string();
    run_command_in(
        world,
        &["mcp-call", "rocm_command", "--arguments-json", &arguments],
        None,
    );
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
    // The preview has no command line to offer, so it must not point at one.
    for phrase in ["run it again", "--dry-run.", "Review uninstall"] {
        assert!(
            !text.contains(phrase),
            "the assistant's preview must not offer {phrase:?}:\n{text}"
        );
    }
}

#[when("the user uninstalls the config with --force")]
async fn uninstall_with_force(world: &mut E2eWorld) {
    run_uninstall(world, &["--keep-data", "--keep-cache", "--force"]);
}

#[when("the user uninstalls the config with a near miss of --force")]
async fn uninstall_with_near_miss_of_force(world: &mut E2eWorld) {
    run_uninstall(world, &["--keep-data", "--keep-cache", "--forc"]);
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
