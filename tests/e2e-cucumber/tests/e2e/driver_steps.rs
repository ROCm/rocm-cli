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
        &[
            ("WSL_DISTRO_NAME", "Ubuntu"),
            ("ROCM_CLI_ROCDXG_VERSION", "99.99.99"),
        ],
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
