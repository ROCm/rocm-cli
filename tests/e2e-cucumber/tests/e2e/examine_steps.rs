// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

fn run_rocm(world: &E2eWorld, args: &[&str]) -> (String, String, i32) {
    let binary = crate::rocm_binary();
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(args);
    world.isolate_cmd(&mut cmd);
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.code().unwrap_or(-1),
    )
}

#[given("a machine with an AMD GPU")]
async fn setup_gpu_machine(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["examine"]);
    assert!(
        stdout.contains("AMD GPU detected") || stdout.contains("detected_gfx_target"),
        "no AMD GPU detected on this machine:\n{stdout}"
    );
}

#[given("a machine with a ROCm install that was not set up by the CLI")]
async fn setup_unmanaged_rocm(_world: &mut E2eWorld) {}

#[when("the user asks for the version")]
async fn user_asks_version(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["version"]);
    world.cli_output = Some(stdout);
}

#[when("the user lists available engines")]
async fn user_lists_engines(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["engines", "list"]);
    world.cli_output = Some(stdout);
}

#[when("the user inspects the system")]
async fn user_inspects_system(world: &mut E2eWorld) {
    let (stdout, _, _) = run_rocm(world, &["examine"]);
    world.cli_output = Some(stdout);
}

#[then("a version string is returned")]
async fn assert_version_returned(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.trim().starts_with("rocm "),
        "expected version string starting with 'rocm ': {output}"
    );
}

#[then("all supported engines are listed")]
async fn assert_all_engines_listed(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    for engine in ["lemonade", "vllm"] {
        assert!(
            output.contains(engine),
            "engine '{engine}' not found in:\n{output}"
        );
    }
}

#[then("the inspection reports which GPU is installed")]
async fn assert_gpu_detected(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("detected_gfx_target:"),
        "no GPU target in examine output:\n{output}"
    );
    let gfx = output
        .lines()
        .find(|l| l.contains("detected_gfx_target:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("", str::trim);
    assert!(
        gfx.starts_with("gfx"),
        "GPU target does not start with 'gfx': {gfx}"
    );
}

#[then("the inspection reports that the driver is available")]
async fn assert_driver_available(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("amdgpu") || output.contains("driver_status"),
        "driver status not found in examine output:\n{output}"
    );
}

#[then("the inspection reports the install as pre-existing")]
async fn assert_rocm_unmanaged(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("detected_unmanaged") || output.contains("legacy"),
        "expected unmanaged ROCm status:\n{output}"
    );
}

#[then("the inspection suggests setting up a CLI-managed install")]
async fn assert_suggests_managed_runtime(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("rocm install sdk"),
        "expected guidance to install sdk:\n{output}"
    );
}
