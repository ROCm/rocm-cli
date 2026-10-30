// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm config <...>` mutations. Black-box: each runs the real binary
//! against the scenario's isolated config dir and asserts on the printed
//! confirmation / exit code. Contracts verified against the running Linux binary
//! (see EAI-8072). No GPU or network — mock lane.

use cucumber::{given, then, when};

use crate::E2eWorld;

/// A distinctive fake key we pipe into `set-provider-key`; the negative scenario
/// asserts it is NEVER echoed back in stdout or stderr.
const FAKE_PROVIDER_KEY: &str = "e2e-secret-key-must-not-be-echoed";

#[given("a fresh CLI configuration")]
async fn fresh_config(world: &mut E2eWorld) {
    // The World already gives each scenario an isolated, empty config dir. Also
    // plant the automation daemon-suppression marker: this Given backs the
    // automations scenarios too, and `automations enable` would otherwise spawn a
    // detached daemon and leak it (see automations_steps::suppress_daemon_spawn).
    // Harmless for pure-config scenarios, which never touch automation state.
    crate::e2e::automations_steps::suppress_daemon_spawn(world);
}

#[given("a machine with no secure secret storage")]
async fn no_secret_storage(_world: &mut E2eWorld) {
    // The premise is made deterministic in the When step: the CLI is spawned with
    // DBUS_SESSION_BUS_ADDRESS pointed at an unreachable socket, so the Linux
    // Secret Service backend cannot connect and the save fails regardless of
    // whether the runner happens to have a session bus. The scenario is
    // `@requires-os:linux` because the Windows/macOS credential stores are not
    // reachable through D-Bus and cannot be disabled the same way.
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user sets the default engine")]
async fn set_default_engine(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "set-default-engine", "vllm"]);
    record(world, stdout, stderr, rc);
}

#[when("the user clears the default engine")]
async fn clear_default_engine(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "clear-default-engine"]);
    record(world, stdout, stderr, rc);
}

#[when("the user sets the default runtime")]
async fn set_default_runtime(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["config", "set-default-runtime", "therock-release:gfx942"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user clears the default runtime")]
async fn clear_default_runtime(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "clear-default-runtime"]);
    record(world, stdout, stderr, rc);
}

#[when("the user turns telemetry off")]
async fn set_telemetry_off(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "set-telemetry", "off"]);
    record(world, stdout, stderr, rc);
}

#[when("the user selects a permissions mode")]
async fn set_permissions(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "set-permissions", "ask"]);
    record(world, stdout, stderr, rc);
}

#[when("the user configures an engine without saying what to change")]
async fn set_engine_no_target(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "set-engine", "vllm"]);
    record(world, stdout, stderr, rc);
}

#[when("the user configures an engine with a runtime to use")]
async fn set_engine_with_runtime(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "config",
            "set-engine",
            "vllm",
            "--runtime-id",
            "therock-release:gfx942",
        ],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user enables a cloud provider")]
async fn enable_provider(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "enable-provider", "openai"]);
    record(world, stdout, stderr, rc);
}

#[when("the user disables that provider")]
async fn disable_provider(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "disable-provider", "openai"]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to enable the local provider")]
async fn enable_local_provider(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["config", "enable-provider", "local"]);
    record(world, stdout, stderr, rc);
}

#[when("the user saves a provider API key")]
async fn save_provider_key(world: &mut E2eWorld) {
    // `set-provider-key` reads the key from stdin (non-interactive). Pipe a
    // distinctive fake key so the "no echo" assertion can look for it. Force the
    // Linux Secret Service unreachable by pointing D-Bus at a nonexistent socket,
    // so the save deterministically fails on any Linux runner (see the Given).
    let (stdout, stderr, rc) = crate::run_rocm_with_stdin(
        world,
        &["config", "set-provider-key", "openai"],
        FAKE_PROVIDER_KEY,
        &[(
            "DBUS_SESSION_BUS_ADDRESS",
            "unix:path=/nonexistent/e2e-no-bus",
        )],
    );
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI confirms the default engine was set")]
async fn confirm_engine_set(world: &mut E2eWorld) {
    assert_ok_contains(world, "default engine set to vllm");
}

#[then("the CLI confirms the default engine was cleared")]
async fn confirm_engine_cleared(world: &mut E2eWorld) {
    assert_ok_contains(world, "default engine cleared");
}

#[then("the CLI confirms the default runtime was set")]
async fn confirm_runtime_set(world: &mut E2eWorld) {
    assert_ok_contains(world, "default runtime set to therock-release:gfx942");
}

#[then("the CLI confirms the default runtime was cleared")]
async fn confirm_runtime_cleared(world: &mut E2eWorld) {
    assert_ok_contains(world, "default runtime cleared");
}

#[then("the CLI confirms the telemetry mode and states the policy")]
async fn confirm_telemetry(world: &mut E2eWorld) {
    assert_ok_contains(world, "telemetry mode set to off");
    assert_output_contains(world, "policy:");
}

#[then("the CLI confirms the permissions mode")]
async fn confirm_permissions(world: &mut E2eWorld) {
    assert_ok_contains(world, "permissions mode set to ask");
}

#[then("the CLI refuses and explains a target is required")]
async fn refuse_engine_no_target(world: &mut E2eWorld) {
    assert_failed(world);
    assert_output_contains(
        world,
        "set-engine requires --runtime-id, --env-id, or --clear",
    );
}

#[then("the CLI confirms the engine configuration was updated")]
async fn confirm_engine_config_updated(world: &mut E2eWorld) {
    assert_ok_contains(world, "updated engine config for vllm");
}

#[then("the CLI confirms the provider is enabled for prompt sending")]
async fn confirm_provider_enabled(world: &mut E2eWorld) {
    assert_ok_contains(world, "provider openai enabled for prompt sending");
}

#[then("the CLI confirms the provider is disabled")]
async fn confirm_provider_disabled(world: &mut E2eWorld) {
    assert_ok_contains(world, "provider openai disabled for prompt sending");
}

#[then("the CLI refuses and explains the local provider is always enabled")]
async fn refuse_local_provider(world: &mut E2eWorld) {
    assert_failed(world);
    assert_output_contains(world, "local provider is always enabled");
}

#[then("the CLI reports it could not save the key securely")]
async fn confirm_key_save_failed(world: &mut E2eWorld) {
    assert_failed(world);
    assert_output_contains(world, "failed to save openai API key in secure storage");
}

#[then("the key value never appears in the output")]
async fn confirm_key_not_echoed(world: &mut E2eWorld) {
    let combined = combined_output(world);
    assert!(
        !combined.contains(FAKE_PROVIDER_KEY),
        "the provider key leaked into the CLI output:\n{combined}"
    );
}

// ── Unreadable settings file ──────────────────────────────────────

/// The custom ROCm folder the damaged settings file records, inside the
/// scenario's sandbox so the suggested `ROCM_CLI_DATA_DIR` can be used as-is.
fn custom_rocm_folder(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("custom-rocm")
}

/// A `config.json` cut off just after it recorded `setup.therock_venv`, the
/// field that tells the CLI where a custom-folder install keeps its runtimes.
/// Black-box: plain bytes, as an interrupted write or a bad hand edit leaves
/// them. Derived from the sandbox, so every Then can rebuild it and compare
/// byte for byte.
fn cut_short_settings(world: &E2eWorld) -> String {
    let folder = serde_json::to_string(&custom_rocm_folder(world))
        .expect("failed to encode the custom folder");
    format!(
        "{{\n  \"active_runtime_key\": \"therock-release-gfx1151-7.13.0\",\n  \"setup\": {{\n    \
         \"therock_venv\": {folder},\n    \"comp"
    )
}

fn sandbox(world: &E2eWorld) -> std::path::PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has an isolated root")
        .path()
        .to_path_buf()
}

fn settings_path(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("config").join("config.json")
}

fn moved_aside_settings_path(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("config").join("config.json.broken")
}

#[given("the CLI's settings file was cut short just after it recorded a custom ROCm folder")]
async fn settings_cut_short(world: &mut E2eWorld) {
    let path = settings_path(world);
    std::fs::create_dir_all(path.parent().expect("settings file has a parent"))
        .expect("failed to create config dir");
    std::fs::write(&path, cut_short_settings(world)).expect("failed to plant settings file");
}

#[when("the user checks the CLI version without naming a data folder")]
async fn version_without_data_dir(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_without_data_dir_override(world, &["version"]);
    record(world, stdout, stderr, rc);
}

#[when("the user opens the CLI with the data folder the refusal suggested")]
async fn open_with_suggested_data_dir(world: &mut E2eWorld) {
    // Take the value from the refusal itself, so this runs the advice as
    // printed rather than a value the test already knows.
    let folder = suggested_data_dir(world);
    let (stdout, stderr, rc) =
        crate::run_rocm_with_env(world, &[], &[("ROCM_CLI_DATA_DIR", &folder)]);
    record(world, stdout, stderr, rc);
}

#[when("the user changes a setting with the data folder the refusal suggested")]
async fn change_setting_with_suggested_data_dir(world: &mut E2eWorld) {
    let folder = suggested_data_dir(world);
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["config", "set-telemetry", "off"],
        &[("ROCM_CLI_DATA_DIR", &folder)],
    );
    record(world, stdout, stderr, rc);
}

/// The `ROCM_CLI_DATA_DIR` value the refusal told the user to set, read from
/// its stderr — "setting ROCM_CLI_DATA_DIR=<folder> points commands ..." — the
/// first time it is needed, and remembered for the steps after that.
fn suggested_data_dir(world: &mut E2eWorld) -> String {
    if let Some(folder) = &world.suggested_data_dir {
        return folder.clone();
    }
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let (_, rest) = stderr
        .split_once("setting ROCM_CLI_DATA_DIR=")
        .unwrap_or_else(|| panic!("the refusal suggested no ROCM_CLI_DATA_DIR value:\n{stderr}"));
    let (folder, _) = rest
        .split_once(" points commands")
        .unwrap_or_else(|| panic!("cannot find the end of the suggested value:\n{stderr}"));
    world.suggested_data_dir = Some(folder.to_owned());
    folder.to_owned()
}

#[when("the user moves the settings file aside and opens the CLI without naming a data folder")]
async fn open_after_moving_settings_aside(world: &mut E2eWorld) {
    std::fs::rename(settings_path(world), moved_aside_settings_path(world))
        .expect("failed to move the settings file aside");
    let (stdout, stderr, rc) = crate::run_rocm_without_data_dir_override(world, &[]);
    record(world, stdout, stderr, rc);
}

#[then("the CLI refuses, naming the settings file, the recorded folder and how to recover")]
async fn refuses_unreadable_settings(world: &mut E2eWorld) {
    assert_failed(world);
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let suggestion = format!("ROCM_CLI_DATA_DIR={}", custom_rocm_folder(world).display());
    for needle in [
        "cannot tell where rocm-cli keeps its data",
        "config.json",
        "cannot be read",
        "Repair that file",
        suggestion.as_str(),
        "any command that changes settings still stops on it",
        "run on default settings",
        "Moving the file aside resets every setting",
    ] {
        assert!(
            stderr.contains(needle),
            "expected stderr to contain {needle:?}, got:\n{}",
            combined_output(world)
        );
    }
}

#[then("the settings file is left exactly as it was")]
async fn settings_unchanged(world: &mut E2eWorld) {
    let after = std::fs::read_to_string(settings_path(world)).expect("settings file is gone");
    assert_eq!(
        after,
        cut_short_settings(world),
        "the CLI rewrote the settings file"
    );
}

/// Where config-11 keeps the file its settings link points at: outside the
/// config directory, as a dotfiles checkout would be.
fn linked_settings_target(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("dotfiles").join("rocm-config.json")
}

fn telemetry_mode_in(path: &std::path::Path) -> String {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{} is not JSON ({e}):\n{text}", path.display()));
    value["telemetry"]["mode"]
        .as_str()
        .unwrap_or_else(|| panic!("{} records no telemetry mode:\n{text}", path.display()))
        .to_owned()
}

#[given("a settings file that only its owner can read")]
async fn private_settings_file(world: &mut E2eWorld) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = settings_path(world);
        std::fs::create_dir_all(path.parent().expect("settings file has a parent"))
            .expect("failed to create config dir");
        std::fs::write(&path, "{}\n").expect("failed to plant settings file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("failed to make the settings file private");
    }
    #[cfg(not(unix))]
    {
        let _ = world;
        panic!("tagged @requires-os:linux: needs Unix permissions");
    }
}

#[then("the settings file is still readable only by its owner")]
async fn settings_still_private(world: &mut E2eWorld) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(settings_path(world))
            .expect("settings file is gone")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "saving changed the settings file's permissions to {:o}",
            mode & 0o777
        );
    }
    #[cfg(not(unix))]
    {
        let _ = world;
        panic!("tagged @requires-os:linux: needs Unix permissions");
    }
}

#[then("the settings file records telemetry as off")]
async fn settings_record_telemetry_off(world: &mut E2eWorld) {
    assert_eq!(telemetry_mode_in(&settings_path(world)), "off");
}

#[given("a settings file that is a link to a file kept elsewhere")]
async fn linked_settings_file(world: &mut E2eWorld) {
    #[cfg(unix)]
    {
        let target = linked_settings_target(world);
        std::fs::create_dir_all(target.parent().expect("link target has a parent"))
            .expect("failed to create the dotfiles folder");
        std::fs::write(&target, "{}\n").expect("failed to plant the linked-to file");
        let path = settings_path(world);
        std::fs::create_dir_all(path.parent().expect("settings file has a parent"))
            .expect("failed to create config dir");
        std::os::unix::fs::symlink(&target, &path).expect("failed to link the settings file");
    }
    #[cfg(not(unix))]
    {
        let _ = world;
        panic!("tagged @requires-os:linux: needs Unix links");
    }
}

#[then("the settings file is still a link to that file")]
async fn settings_still_linked(world: &mut E2eWorld) {
    let path = settings_path(world);
    let metadata = std::fs::symlink_metadata(&path).expect("settings file is gone");
    assert!(
        metadata.file_type().is_symlink(),
        "saving replaced the linked settings file with a regular file"
    );
    assert_eq!(
        std::fs::read_link(&path).expect("cannot read the settings link"),
        linked_settings_target(world),
        "the settings link now points somewhere else"
    );
}

#[then("the linked-to file records telemetry as off")]
async fn linked_target_records_telemetry_off(world: &mut E2eWorld) {
    assert_eq!(telemetry_mode_in(&linked_settings_target(world)), "off");
}

/// The launcher summary's `data dir:` line, from a run that succeeded.
fn launcher_data_dir(world: &E2eWorld) -> String {
    assert_ok_contains(world, "active runtime key: <unset>");
    let stdout = world.cli_output.as_deref().unwrap_or("");
    stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("data dir: "))
        .unwrap_or_else(|| panic!("the CLI did not say where its data is:\n{stdout}"))
        .to_owned()
}

// "On default settings" is what `active runtime key: <unset>` shows: the broken
// file records an active runtime key, and the CLI is not reading it.
#[then("the CLI keeps its data in the recorded folder, on default settings")]
async fn data_in_recorded_folder(world: &mut E2eWorld) {
    let data_dir = launcher_data_dir(world);
    assert_eq!(
        std::path::Path::new(&data_dir),
        custom_rocm_folder(world),
        "the suggested ROCM_CLI_DATA_DIR did not point the CLI at the recorded folder"
    );
}

#[then("the CLI keeps its data in its default folder, on default settings")]
async fn data_in_default_folder(world: &mut E2eWorld) {
    let data_dir = launcher_data_dir(world);
    assert_ne!(
        std::path::Path::new(&data_dir),
        custom_rocm_folder(world),
        "with the settings file moved aside, the CLI still used the folder it recorded"
    );
}

#[then("the CLI refuses to change the setting")]
async fn refuses_to_change_setting(world: &mut E2eWorld) {
    assert_failed(world);
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("config.json"),
        "expected the refusal to name config.json, got:\n{}",
        combined_output(world)
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn assert_output_contains(world: &E2eWorld, needle: &str) {
    let combined = combined_output(world);
    assert!(
        combined.contains(needle),
        "expected output to contain {needle:?}, got:\n{combined}"
    );
}

/// Assert the last command succeeded (rc 0) and its output contains `needle`.
fn assert_ok_contains(world: &E2eWorld, needle: &str) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(
        rc,
        0,
        "expected success, got rc={rc}:\n{}",
        combined_output(world)
    );
    assert_output_contains(world, needle);
}

/// Assert the last command failed (non-zero rc).
fn assert_failed(world: &E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(
        rc != 0,
        "expected the command to fail, but it exited 0:\n{}",
        combined_output(world)
    );
}
