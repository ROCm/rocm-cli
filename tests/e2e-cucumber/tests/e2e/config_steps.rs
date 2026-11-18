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
    assert_refused_before_locking(world);
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
    assert_refused_before_locking(world);
}

/// A refused settings change is refused before it takes the settings lock, so
/// it leaves nothing behind in the config folder, not even the lock file (and
/// on a read-only config folder the user sees the refusal, not a failure to
/// create that file).
///
/// Skipped on a host with a legacy rocm-dash `config.toml`: every `rocm`
/// command first migrates it into a config folder that has no `config.json`
/// yet, under the lock, so the lock file is there whatever the refusal did.
/// That file sits in the user's real config folder, which the scenario's
/// environment does not move (Windows finds it through the known-folder API).
fn assert_refused_before_locking(world: &E2eWorld) {
    if let Some(legacy) = legacy_dashboard_config_on_this_host() {
        eprintln!(
            "skipped the lock-file check: {} exists, and its migration takes the lock",
            legacy.display()
        );
        return;
    }
    let lock = settings_path(world).with_file_name("config.json.lock");
    assert!(
        !lock.exists(),
        "the refused change took the settings lock: {} exists",
        lock.display()
    );
}

/// The legacy rocm-dash `config.toml` where `rocm` looks for it on this host,
/// when one is there: under `%APPDATA%` on Windows, under `$XDG_CONFIG_HOME`
/// (or `~/.config`) elsewhere.
fn legacy_dashboard_config_on_this_host() -> Option<std::path::PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(std::path::PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".config"))
            })
    }?;
    Some(base.join("rocm-dash").join("config.toml")).filter(|path| path.exists())
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
/// Its name has a space, so the refusal's advice only works if it quotes it.
fn custom_rocm_folder(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("custom rocm")
}

/// The path in double quotes right after the first `label` in `stderr`: what
/// a user copies when the refusal tells them which path to use.
fn quoted_after(stderr: &str, label: &str) -> String {
    let (_, rest) = stderr
        .split_once(label)
        .unwrap_or_else(|| panic!("the refusal does not say {label:?}:\n{stderr}"));
    let rest = rest
        .strip_prefix('"')
        .unwrap_or_else(|| panic!("no quoted path after {label:?}:\n{stderr}"));
    let (path, _) = rest
        .split_once('"')
        .unwrap_or_else(|| panic!("an unclosed quote after {label:?}:\n{stderr}"));
    path.to_owned()
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

fn renamed_settings_path(world: &E2eWorld) -> std::path::PathBuf {
    sandbox(world).join("config").join("config.json.broken")
}

/// The runtime planted in the custom folder; the damaged settings file names
/// it as the active one.
const CUSTOM_FOLDER_RUNTIME_KEY: &str = "therock-release-gfx1151-7.13.0";

#[given(
    "the CLI's settings file was cut short just after it recorded a custom ROCm folder holding \
     an installed runtime"
)]
async fn settings_cut_short(world: &mut E2eWorld) {
    let path = settings_path(world);
    std::fs::create_dir_all(path.parent().expect("settings file has a parent"))
        .expect("failed to create config dir");
    std::fs::write(&path, cut_short_settings(world)).expect("failed to plant settings file");

    // A read-only tarball runtime registered in the custom folder's own
    // registry, as `rocm install sdk --prefix` leaves it. Plain JSON matching
    // the on-disk manifest schema, black-box like the settings file above.
    let folder = custom_rocm_folder(world);
    let install_root = folder.join("runtime-gfx1151");
    std::fs::create_dir_all(&install_root).expect("failed to create install root");
    std::fs::write(install_root.join("payload.txt"), "payload")
        .expect("failed to write runtime payload");
    let registry = folder.join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create registry dir");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": CUSTOM_FOLDER_RUNTIME_KEY,
        "runtime_id": "therock-release:gfx1151",
        "channel": "release",
        "format": "tarball",
        "family": "gfx1151",
        "family_source": "manual",
        "version": "7.13.0",
        "install_root": install_root,
        "selected_artifact_url": format!("https://example.invalid/{CUSTOM_FOLDER_RUNTIME_KEY}.tar.gz"),
        "read_only": true,
        "installed_at_unix_ms": 1_700_000_000_000u64,
    }))
    .expect("failed to serialize runtime manifest");
    std::fs::write(
        registry.join(format!("{CUSTOM_FOLDER_RUNTIME_KEY}.json")),
        manifest,
    )
    .expect("failed to write runtime manifest");
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
/// its stderr — "setting the environment variable ROCM_CLI_DATA_DIR to
/// \"<folder>\" ..." — the first time it is needed, and remembered for the
/// steps after that.
fn suggested_data_dir(world: &mut E2eWorld) -> String {
    if let Some(folder) = &world.suggested_data_dir {
        return folder.clone();
    }
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let folder = quoted_after(
        stderr,
        "setting the environment variable ROCM_CLI_DATA_DIR to ",
    );
    world.suggested_data_dir = Some(folder.clone());
    folder
}

#[when(
    "the user renames the settings file as the refusal said and opens the CLI without naming a \
     data folder"
)]
async fn open_after_renaming_settings(world: &mut E2eWorld) {
    // The paths come from the refusal itself, read by the step that checked
    // it, so this follows the advice as printed.
    let (from, to) = world
        .suggested_settings_rename
        .clone()
        .expect("the refusal was checked before its advice is followed");
    std::fs::rename(&from, &to).expect("failed to rename the settings file");
    assert_eq!(
        std::fs::read_to_string(&to).expect("the renamed settings file is gone"),
        cut_short_settings(world),
        "renaming the settings file changed its contents"
    );
    let (stdout, stderr, rc) = crate::run_rocm_without_data_dir_override(world, &[]);
    record(world, stdout, stderr, rc);
}

#[then("the CLI refuses, naming the settings file, the recorded folder and how to recover")]
async fn refuses_unreadable_settings(world: &mut E2eWorld) {
    assert_failed(world);
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let suggestion = format!(
        "setting the environment variable ROCM_CLI_DATA_DIR to \"{}\"",
        custom_rocm_folder(world).display()
    );
    for needle in [
        "cannot tell where rocm-cli keeps its data",
        "config.json",
        "cannot be read",
        "Repair that file",
        suggestion.as_str(),
        "any command that changes settings still stops on it",
        "run on default settings",
    ] {
        assert!(
            stderr.contains(needle),
            "expected stderr to contain {needle:?}, got:\n{}",
            combined_output(world)
        );
    }
    // The file to rename is the exact one the CLI read — here under the
    // scenario's own config folder — and the name to give it sits beside it.
    // Both are read from between their quotes, as a user would copy them.
    let from = quoted_after(stderr, "keeps its data: ");
    let to = quoted_after(stderr, "Renaming it to ");
    assert_eq!(std::path::Path::new(&from), settings_path(world));
    assert_eq!(std::path::Path::new(&to), renamed_settings_path(world));
    world.suggested_settings_rename = Some((from, to));
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

/// The default data folder: `.rocm` under the home directory
/// `run_rocm_without_data_dir_override` gives the CLI.
#[then("the CLI keeps its data in its default folder")]
async fn data_in_default_folder(world: &mut E2eWorld) {
    let data_dir = launcher_data_dir(world);
    assert_eq!(
        std::path::Path::new(&data_dir),
        sandbox(world).join("home").join(".rocm"),
        "with the settings file renamed, the CLI did not go back to its default folder"
    );
}

#[when("the user lists runtimes with the data folder the refusal suggested")]
async fn list_runtimes_with_suggested_data_dir(world: &mut E2eWorld) {
    let folder = suggested_data_dir(world);
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["runtimes", "list"],
        &[("ROCM_CLI_DATA_DIR", &folder)],
    );
    record(world, stdout, stderr, rc);
}

#[then("the runtime installed in the recorded folder is listed")]
async fn custom_folder_runtime_listed(world: &mut E2eWorld) {
    assert_ok_contains(world, CUSTOM_FOLDER_RUNTIME_KEY);
}

/// The data folder is already the one the refusal suggested, so the advice
/// is the rename alone, naming the exact file and the name to give it; that
/// following it works is the rename step later in the same scenario.
#[then("the CLI refuses to change the setting, naming the settings file to rename")]
async fn refuses_to_change_setting(world: &mut E2eWorld) {
    assert_failed(world);
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let rename = format!(
        "To start from default settings instead, rename \"{}\" to \"{}\"",
        settings_path(world).display(),
        renamed_settings_path(world).display()
    );
    assert!(
        stderr.contains("Repair that file") && stderr.contains(&rename),
        "expected the refusal to say {rename:?}, got:\n{}",
        combined_output(world)
    );
    // The user already set the folder the first refusal suggested, so being
    // told to set it again would be wrong.
    assert!(
        !stderr.contains("then also need the environment variable ROCM_CLI_DATA_DIR"),
        "the refusal repeated the ROCM_CLI_DATA_DIR advice the user already followed:\n{}",
        combined_output(world)
    );
}

// ── Settings changed at the same time ─────────────────────────────

/// One of `config-12`'s writers: the two `rocm` commands it alternates
/// between, each with what the settings file then records at `pointer`
/// (`None`: nothing there).
struct SettingsWriter {
    name: &'static str,
    changes: [(&'static [&'static str], Option<&'static str>); 2],
    pointer: &'static str,
}

/// Every kind of `rocm config` change that loads, changes and saves the
/// settings: a setter, a `clear-*`, and `set-engine` with each of its flags.
/// `enable-provider` is left out: it also reads the secret store, which this
/// lane cannot rely on, and goes through the same locked update as the rest.
const SETTINGS_WRITERS: [SettingsWriter; 7] = [
    SettingsWriter {
        name: "default engine",
        changes: [
            (
                &["config", "set-default-engine", "lemonade"],
                Some("lemonade"),
            ),
            (&["config", "clear-default-engine"], None),
        ],
        pointer: "/default_engine",
    },
    SettingsWriter {
        name: "telemetry",
        changes: [
            (&["config", "set-telemetry", "local"], Some("local")),
            (&["config", "set-telemetry", "off"], Some("off")),
        ],
        pointer: "/telemetry/mode",
    },
    SettingsWriter {
        name: "permissions",
        changes: [
            (&["config", "set-permissions", "ask"], Some("ask")),
            (
                &["config", "set-permissions", "full-access"],
                Some("full_access"),
            ),
        ],
        pointer: "/permissions/mode",
    },
    SettingsWriter {
        name: "planner provider",
        changes: [
            (
                &["config", "set-planner-provider", "anthropic"],
                Some("anthropic"),
            ),
            (&["config", "clear-planner-provider"], None),
        ],
        pointer: "/planner_provider",
    },
    SettingsWriter {
        name: "default runtime",
        changes: [
            (
                &["config", "set-default-runtime", "therock-release:gfx942"],
                Some("therock-release:gfx942"),
            ),
            (&["config", "clear-default-runtime"], None),
        ],
        pointer: "/default_runtime_id",
    },
    SettingsWriter {
        name: "vllm runtime",
        changes: [
            (
                &["config", "set-engine", "vllm", "--runtime-id", "runtime-a"],
                Some("runtime-a"),
            ),
            (
                &["config", "set-engine", "vllm", "--runtime-id", "runtime-b"],
                Some("runtime-b"),
            ),
        ],
        pointer: "/engines/vllm/preferred_runtime_id",
    },
    SettingsWriter {
        name: "lemonade env",
        changes: [
            (
                &["config", "set-engine", "lemonade", "--env-id", "env-a"],
                Some("env-a"),
            ),
            (&["config", "set-engine", "lemonade", "--clear"], None),
        ],
        pointer: "/engines/lemonade/preferred_env_id",
    },
];

/// How many changes each writer makes, as many as in the run that found the
/// lost updates. A race can only be caught, never ruled out, by running it:
/// against binaries without the lock this scenario failed 3 of 3 runs in WSL
/// and 2 of 6 on native Windows. The deterministic check is `rocm-core`'s
/// `an_update_waits_for_the_one_in_progress_and_keeps_its_change`. Under half a
/// minute on a desktop.
const SETTINGS_WRITER_ROUNDS: usize = 50;

/// What `config-12`'s writers found.
#[derive(Debug, Default)]
pub struct ConcurrentSettingsFindings {
    /// Commands that exited non-zero, and settings-file reads that failed.
    failures: Vec<String>,
    /// Changes a writer found undone when it came to make its next one.
    undone: Vec<String>,
}

/// The value at `pointer` in the settings file at `path`.
fn setting_in(path: &std::path::Path, pointer: &str) -> Result<Option<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not JSON ({e}):\n{text}", path.display()))?;
    Ok(value
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned))
}

/// How long one settings command may take. It waits at most 30 seconds for
/// the settings lock, so anything longer is a hang, not a slow turn.
const SETTINGS_COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

/// Run one writer: before each change, check that its previous change is
/// still in the file, then make the next one.
fn run_settings_writer(
    writer: &SettingsWriter,
    env: &[(&'static str, std::ffi::OsString)],
    scenario: Option<&str>,
    settings: &std::path::Path,
) -> ConcurrentSettingsFindings {
    let mut findings = ConcurrentSettingsFindings::default();
    for round in 0..SETTINGS_WRITER_ROUNDS {
        if round > 0 {
            let expected = writer.changes[(round - 1) % 2].1;
            match setting_in(settings, writer.pointer) {
                Ok(found) if found.as_deref() == expected => {}
                Ok(found) => findings.undone.push(format!(
                    "{}: before change {round} the file held {found:?}, not the {expected:?} \
                     this command had just set",
                    writer.name
                )),
                Err(error) => findings
                    .failures
                    .push(format!("{} before change {round}: {error}", writer.name)),
            }
        }
        let args = writer.changes[round % 2].0;
        let (_, stderr, rc) =
            crate::run_rocm_bounded_in(env, scenario, args, &[], SETTINGS_COMMAND_TIMEOUT);
        if rc != 0 {
            findings.failures.push(format!(
                "`rocm {}` (change {round}) exited {rc}: {}",
                args.join(" "),
                stderr.trim()
            ));
        }
    }
    findings
}

#[when("seven commands change different settings over and over at the same time")]
async fn change_settings_concurrently(world: &mut E2eWorld) {
    let env = world.isolate_env();
    let scenario = world.current_scenario.clone();
    let settings = settings_path(world);
    let findings = std::thread::scope(|scope| {
        // Every writer is started before any is joined, so they overlap.
        let mut handles = Vec::new();
        for writer in &SETTINGS_WRITERS {
            handles.push(
                scope.spawn(|| run_settings_writer(writer, &env, scenario.as_deref(), &settings)),
            );
        }
        let mut all = ConcurrentSettingsFindings::default();
        for handle in handles {
            let one = handle.join().expect("a settings writer panicked");
            all.failures.extend(one.failures);
            all.undone.extend(one.undone);
        }
        all
    });
    world.concurrent_settings_findings = Some(findings);
}

const fn concurrent_settings_findings(world: &E2eWorld) -> &ConcurrentSettingsFindings {
    world
        .concurrent_settings_findings
        .as_ref()
        .expect("the concurrent settings writers were run")
}

#[then("every one of those commands succeeded")]
async fn concurrent_commands_succeeded(world: &mut E2eWorld) {
    let failures = &concurrent_settings_findings(world).failures;
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[then("no command found a change it had made undone")]
async fn no_concurrent_change_undone(world: &mut E2eWorld) {
    let undone = &concurrent_settings_findings(world).undone;
    assert!(
        undone.is_empty(),
        "{} of {} changes were undone by another command:\n{}",
        undone.len(),
        SETTINGS_WRITERS.len() * (SETTINGS_WRITER_ROUNDS - 1),
        undone.join("\n")
    );
}

#[then("the settings file keeps the last value each command set")]
async fn settings_keep_last_values(world: &mut E2eWorld) {
    let settings = settings_path(world);
    for writer in &SETTINGS_WRITERS {
        let last = writer.changes[(SETTINGS_WRITER_ROUNDS - 1) % 2].1;
        let found = setting_in(&settings, writer.pointer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            found.as_deref(),
            last,
            "the {} setting lost the last change made to it",
            writer.name
        );
    }
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
