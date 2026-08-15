// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{then, when};

use crate::E2eWorld;

#[when("the user previews driver installation with a WSL detection signal")]
async fn preview_wsl_driver_install(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["install", "driver", "--dry-run"],
        &[("WSL_DISTRO_NAME", "Ubuntu")],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user reviews driver installation with a WSL detection signal without approval")]
async fn review_wsl_driver_install_without_approval(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["install", "driver"],
        &[("WSL_DISTRO_NAME", "Ubuntu")],
    );
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
