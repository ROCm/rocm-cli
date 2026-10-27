// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{then, when};

use crate::E2eWorld;

#[when("the user previews driver installation on this WSL host")]
async fn preview_wsl_driver_install(world: &mut E2eWorld) {
    // No env signal: `is_wsl_host()` reads `/dev/dxg` and `/proc/version` and
    // does not trust `$WSL_DISTRO_NAME`, so the scenario relies on the
    // `@requires-wsl` gate having put it on a real WSL host.
    let (stdout, stderr, rc) =
        crate::run_rocm_with_env(world, &["install", "driver", "--dry-run"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user reviews driver installation on this WSL host without approval")]
async fn review_wsl_driver_install_without_approval(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(world, &["install", "driver"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the driver plan is supported and mutating")]
async fn assert_supported_mutating_driver_plan(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "driver dry-run should succeed");
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("supported: true"), "{output}");
    assert!(output.contains("mutating: true"), "{output}");
    assert!(output.contains("dry_run: true"), "{output}");
}

#[then("the dry-run driver plan requires no approval and previews no execution")]
async fn assert_wsl_driver_dry_run_needs_no_approval(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("policy: wsl_rocdxg"), "{output}");
    assert!(output.contains("approval: not required"), "{output}");
    assert!(output.contains("dry_run: true"), "{output}");
    assert!(
        output.contains("action: dry run only; no driver commands executed"),
        "{output}"
    );
}

#[then("the unapproved WSL driver plan is actionable but not executed")]
async fn assert_unapproved_wsl_driver_plan_not_executed(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "driver plan review should succeed");
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("policy: wsl_rocdxg"), "{output}");
    assert!(output.contains("supported: true"), "{output}");
    assert!(output.contains("mutating: true"), "{output}");
    assert!(output.contains("approval: required"), "{output}");
    assert!(output.contains("dry_run: false"), "{output}");
    assert!(output.contains("execution_commands:"), "{output}");
    assert!(
        output.contains(
            "action: rerun with --yes after reviewing this plan, or approve from the TUI"
        ),
        "{output}"
    );
    assert!(
        !output.lines().any(|line| line.trim() == "execution:"),
        "unapproved plan unexpectedly reported execution:\n{output}"
    );
    let state_path = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
        .join("driver")
        .join("state.json");
    assert!(
        !state_path.exists(),
        "unapproved plan wrote execution state at {}",
        state_path.display()
    );
}

#[then("the driver plan does not direct the user to the removed WSL setup script")]
async fn assert_no_removed_wsl_script_guidance(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(!output.contains("scripts/wsl_setup_rocdxg.sh"), "{output}");
}

#[then("the driver plan verifies the download before installing it")]
async fn assert_driver_plan_verifies_download(world: &mut E2eWorld) {
    // The package is handed to `apt-get install`, which runs its maintainer
    // scripts as root, so the plan the user approves has to show the check.
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("sha256sum -c -"), "{output}");
    assert!(
        !output.contains("skipping checksum verification"),
        "verification must not be conditional on an unset variable:\n{output}"
    );
    let commands: Vec<&str> = output.lines().map(str::trim).collect();
    let check = commands
        .iter()
        .position(|line| line.contains("sha256sum -c -"))
        .expect("plan verifies the download");
    let install = commands
        .iter()
        .position(|line| line.contains("apt-get install -y '/tmp/"))
        .expect("plan installs the package");
    assert!(
        check < install,
        "digest must be checked before the root install:\n{output}"
    );
}

#[when("the user previews driver installation for a ROCDXG release with no known digest")]
async fn preview_wsl_driver_install_unpinned(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["install", "driver", "--dry-run"],
        &[("ROCM_CLI_ROCDXG_VERSION", "99.99.99")],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the driver plan refuses rather than installing an unverified package")]
async fn assert_driver_plan_refuses_unverified(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("supported: false"), "{output}");
    assert!(output.contains("mutating: false"), "{output}");
    // The refusal has to name the way out, or it is just a dead end.
    assert!(output.contains("ROCM_CLI_ROCDXG_SHA256"), "{output}");
    assert!(
        output.contains("ROCM_CLI_ROCDXG_ALLOW_UNVERIFIED"),
        "{output}"
    );
    assert!(
        !output.contains("apt-get install"),
        "a refusal must not offer install commands:\n{output}"
    );
}

// ── Windows: the driver install only reports ───────────────────────

/// Refuse any other host before the step runs the CLI. Explicit cucumber
/// filters bypass the platform tags, and on Linux or WSL `install driver --yes`
/// approves a real package-manager plan. The OS name is `linux` on WSL.
fn require_native_windows() {
    super::host_guard::require_os_is_windows(std::env::consts::OS);
}

#[when("the user asks to install the driver on this Windows host")]
async fn ask_windows_driver_install(world: &mut E2eWorld) {
    // Explicit cucumber filters bypass platform tags. Reject a Linux/WSL host
    // before invoking its real driver plan, not after it has run.
    require_native_windows();
    let (stdout, stderr, rc) = crate::run_rocm_with_env(world, &["install", "driver"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user approves the driver install on this Windows host")]
async fn approve_windows_driver_install(world: &mut E2eWorld) {
    // On Linux/WSL, --yes approves a real package-manager plan. The host check
    // must precede every command, even when a name filter bypasses the tags.
    require_native_windows();
    let (stdout, stderr, rc) =
        crate::run_rocm_with_env(world, &["install", "driver", "--yes"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI says that no driver is installed or changed")]
async fn assert_windows_driver_install_says_nothing_changes(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "driver install should report, not fail"
    );
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("policy: windows_validate_only"), "{output}");
    assert!(
        output.contains("no driver is installed or changed"),
        "{output}"
    );
}

#[then("no driver command is shown or run, and no installer state is written")]
async fn assert_windows_driver_install_runs_nothing(world: &mut E2eWorld) {
    // The promise made in words, asserted together with the state it describes.
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(output.contains("supported: false"), "{output}");
    assert!(output.contains("mutating: false"), "{output}");
    assert!(output.contains("execution_commands: <none>"), "{output}");
    assert!(
        output.contains("action: no driver commands will be executed"),
        "{output}"
    );
    assert!(
        !output.contains("driver install completed"),
        "an installer run was reported:\n{output}"
    );
    let state_path = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
        .join("driver")
        .join("state.json");
    assert!(
        !state_path.exists(),
        "the driver install wrote execution state at {}",
        state_path.display()
    );
}

#[then("it describes the display driver the inspection reports")]
async fn assert_windows_driver_install_matches_the_inspection(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no driver plan output");
    let (examine_json, _, _) = crate::run_rocm(world, &["examine", "--json"]);
    let json: serde_json::Value = serde_json::from_str(&examine_json).unwrap_or_else(|e| {
        panic!("`examine --json` did not emit valid JSON ({e}):\n{examine_json}")
    });
    let Some(record) = super::examine_steps::windows_driver_record(&json) else {
        // No AMD display driver was read: the plan says so and installs nothing.
        assert!(output.contains("could not be read"), "{output}");
        return;
    };
    let version = record
        .get("version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    assert!(!version.is_empty(), "{record:#}");
    assert!(output.contains(&format!("version {version},")), "{output}");
    let date = record
        .get("date")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    assert!(output.contains(&format!("dated {date},")), "{output}");
    let verdict = match record.get("age").and_then(serde_json::Value::as_str) {
        Some("predates_first_hip_sdk") => "outdated: yes",
        Some("not_older") => "outdated: no",
        _ => "outdated: unknown",
    };
    assert!(
        output.contains(verdict),
        "expected {verdict:?} in:\n{output}"
    );
    // The fix id is named only for an old driver, and `rocm diagnose` is the
    // command that raises it.
    assert_eq!(
        output.contains("fix-22-display-driver-old"),
        verdict == "outdated: yes",
        "{output}"
    );
}
