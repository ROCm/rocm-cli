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
