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

#[when("the user previews driver installation on this host")]
async fn preview_driver_install_on_this_host(world: &mut E2eWorld) {
    // `--dkms` is required here: every driver-plan builder gates its entire
    // command list (repo setup, signing key, package install) behind
    // `dkms`, so a plan built without it is empty and the apt-repo-file/key
    // assertion below would have nothing to find.
    let (stdout, stderr, rc) =
        crate::run_rocm_with_env(world, &["install", "driver", "--dkms", "--dry-run"], &[]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the driver plan matches this host's actual Ubuntu release")]
async fn assert_driver_plan_matches_host_release(world: &mut E2eWorld) {
    assert_eq!(world.cli_rc, Some(0), "driver dry-run should succeed");
    let output = world.cli_output.as_ref().expect("no driver plan output");

    // Derived from the host's own `/etc/os-release`, not from `output` --
    // asserting against the very thing being checked would let a change that
    // breaks the arm-selection logic pass by construction. The hosted
    // `ubuntu-26.04` E2E image reports 26.04 ("resolute"), which has no
    // published repo.radeon.com/amdgpu suite yet and takes
    // `ubuntu_native_archive`; the self-hosted GPU lane reports 24.04 and
    // takes the apt arm. See the `("ubuntu", "26.04")` arm in
    // `driver_install.rs`.
    let os_release_text =
        rocm_core::hardware_root::read_os_release(rocm_core::hardware_root::identity_path_for_str)
            .text()
            .to_owned();
    let Ok(os_release) = rocm_core::os_release::parse(&os_release_text) else {
        panic!("host's /etc/os-release is unreadable:\n{output}");
    };
    if os_release.get("ID").map(String::as_str) != Some("ubuntu") {
        // Every bare-metal Linux lane in this repo runs Ubuntu today, but
        // this scenario's premise is Ubuntu specifically (every assertion
        // below hard-codes Ubuntu's apt/native-archive command text), so a
        // future non-Ubuntu bare-metal lane has no premise here rather than
        // failing against text it was never meant to produce.
        return;
    }

    if os_release.get("VERSION_ID").map(String::as_str) == Some("26.04") {
        assert!(output.contains("policy: ubuntu_native_archive"), "{output}");
        assert!(output.contains("supported: false"), "{output}");
        assert!(
            output.contains("ships ROCm natively via the standard archive"),
            "{output}"
        );
        return;
    }

    // The runner may already be root (the self-hosted WSL runner always is),
    // in which case the plan omits the `sudo ` prefix entirely.
    let sudo = if rocm_core::openmpi::running_as_root() {
        ""
    } else {
        "sudo "
    };
    assert!(
        output.contains("policy: linux_official_amd_dkms_wrapper"),
        "{output}"
    );
    assert!(
        output.contains(&format!(
            "curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | {sudo}gpg --batch --yes --dearmor -o /etc/apt/keyrings/rocm.gpg"
        )),
        "{output}"
    );
    assert!(
        output.contains("signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/"),
        "{output}"
    );
    assert!(
        output.contains("tee /etc/apt/sources.list.d/amdgpu.list >/dev/null"),
        "{output}"
    );
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

#[when("the user previews driver installation with an invalid version override")]
async fn preview_driver_install_invalid_override(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["install", "driver", "--dkms", "--dry-run"],
        &[(
            "ROCM_CLI_AMDGPU_DRIVER_VERSION",
            "'; touch /tmp/pwned; echo '",
        )],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the driver plan refuses the invalid version override and plans no commands")]
async fn assert_driver_plan_refuses_invalid_override(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no driver plan output");
    assert!(
        output.contains("policy: invalid_driver_version_override"),
        "{output}"
    );
    assert!(output.contains("supported: false"), "{output}");
    assert!(output.contains("mutating: false"), "{output}");
    assert!(
        output.contains("ROCM_CLI_AMDGPU_DRIVER_VERSION"),
        "{output}"
    );
    assert!(output.contains("execution_commands: <none>"), "{output}");
}
