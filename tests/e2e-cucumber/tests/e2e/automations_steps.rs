// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm automations enable/disable`. Black-box against the isolated
//! config dir. `automations enable` would otherwise spawn a detached background
//! daemon (`rocm daemon`) on first enable, which both adds a nondeterministic
//! `helper:` line and leaks a process past the scenario. To keep the mock lane
//! hermetic, every scenario first plants an automation runtime-state marking the
//! daemon already running under THIS test process's (live) pid, so the CLI's
//! double-spawn guard skips the spawn. Contracts verified against the running
//! Linux binary (EAI-8072). Scoped to the enable/disable/mode slice; the broader
//! automations feature is covered separately.

use cucumber::{given, then, when};

use crate::E2eWorld;

/// A built-in watcher id (`BUILTIN_WATCHERS` in rocm-core); stable and always
/// present, so scenarios can enable/disable it without depending on host state.
const WATCHER: &str = "therock-update";

/// Plant an automation runtime-state that marks the background daemon already
/// running under the test harness's own (live) pid. `automations enable` guards
/// against a second spawn when the recorded daemon pid is a live process, so this
/// suppresses the detached-daemon spawn — no `helper:` line, no leaked process.
///
/// Shared with the `a fresh CLI configuration` Given (in `config_steps`), which
/// calls this so every automations scenario starting from a fresh config is also
/// spawn-suppressed. Exposed `pub(crate)` for that single caller.
pub(crate) fn suppress_daemon_spawn(world: &E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let dir = root.path().join("data").join("automations");
    std::fs::create_dir_all(&dir).expect("failed to create automations dir");
    let state = serde_json::json!({
        "running": true,
        "automations_enabled": true,
        "daemon_pid": std::process::id(),
        "started_at_unix_ms": 1_700_000_000_000u64,
        "last_tick_unix_ms": 1_700_000_000_000u64,
        "active_watchers": [],
    });
    std::fs::write(
        dir.join("runtime-state.json"),
        serde_json::to_vec_pretty(&state).expect("failed to serialize automation state"),
    )
    .expect("failed to write automation runtime state");
}

#[given("an enabled automation watcher")]
async fn enabled_watcher(world: &mut E2eWorld) {
    suppress_daemon_spawn(world);
    crate::run_rocm_ok(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user enables an automation watcher in observe mode")]
async fn enable_observe(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user re-enables the same watcher in propose mode")]
async fn enable_propose(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["automations", "enable", WATCHER, "--mode", "propose"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user disables the watcher")]
async fn disable_watcher(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["automations", "disable", WATCHER]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to enable a watcher that does not exist")]
async fn enable_unknown(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "automations",
            "enable",
            "e2e-no-such-watcher",
            "--mode",
            "observe",
        ],
    );
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then(regex = r"^the CLI confirms the watcher is enabled in (observe|propose) mode$")]
async fn confirm_enabled(world: &mut E2eWorld, mode: String) {
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher enabled"),
        "expected an enable confirmation, got:\n{out}"
    );
    assert!(
        out.contains(&format!("watcher: {WATCHER}")),
        "expected the watcher id, got:\n{out}"
    );
    assert!(
        out.contains(&format!("mode: {mode}")),
        "expected mode {mode}, got:\n{out}"
    );
}

#[then("the CLI confirms the watcher is disabled")]
async fn confirm_disabled(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher disabled") && out.contains(&format!("watcher: {WATCHER}")),
        "expected a disable confirmation for {WATCHER}, got:\n{out}"
    );
}

#[then("the CLI refuses and names it as unknown")]
async fn refuse_unknown(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("unknown watcher: e2e-no-such-watcher"),
        "expected an unknown-watcher error, got:\n{}",
        combined(world)
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{}", combined(world));
    world.cli_output.clone().unwrap_or_default()
}
