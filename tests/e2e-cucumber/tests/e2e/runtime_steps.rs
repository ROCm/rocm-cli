// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

#[given("a machine with no CLI-managed runtimes")]
async fn setup_no_runtimes(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        stdout.contains("installed: none") || stdout.contains("managed_runtimes: 0"),
        "expected no managed runtimes:\n{stdout}"
    );
}

#[given("a machine with a standard ROCm install")]
async fn setup_standard_rocm(_world: &mut E2eWorld) {}

#[given("a managed runtime is active")]
async fn setup_active_runtime(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        let (install_out, _, rc) = crate::run_rocm(world, &["install", "sdk"]);
        assert!(rc == 0, "rocm install sdk failed (rc={rc}):\n{install_out}");
    }
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no managed runtime is active:\n{stdout}"
    );
}

#[when("the user installs the SDK")]
async fn user_installs_sdk(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm(world, &["install", "sdk"]);
    assert!(rc == 0, "rocm install sdk failed (rc={rc}):\n{stdout}");
    world.cli_output = Some(stdout);
}

#[when("the user tries to adopt the existing install")]
async fn user_tries_adopt(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "runtimes",
            "adopt",
            "--python",
            "/usr/bin/python3",
            "--root",
            "/opt/rocm",
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("a runtime is registered")]
async fn assert_runtime_registered(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no runtime registered after install:\n{stdout}"
    );
}

#[then("the runtime is set as active")]
async fn assert_runtime_active(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    let active = stdout
        .lines()
        .find(|l| l.contains("active_runtime_key:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("", str::trim);
    assert!(
        !active.is_empty() && active != "<unset>",
        "runtime not set as active:\n{stdout}"
    );
}

#[then("the runtime includes an inference engine")]
async fn assert_runtime_has_stack(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["examine"]);
    assert!(
        stdout.contains("torch") || stdout.contains("vllm"),
        "no inference stack found in runtime:\n{stdout}"
    );
}

#[then("the adoption is refused")]
async fn assert_adoption_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command was run");
    assert!(rc != 0, "adopt unexpectedly succeeded");
}

#[then("the error explains which install types can be adopted")]
async fn assert_adopt_error_explains(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let combined = format!("{stdout}{stderr}").to_lowercase();
    assert!(
        combined.contains("therock")
            || combined.contains("rocm_sdk")
            || combined.contains("not supported"),
        "error does not explain TheRock requirement:\n{stdout}\n{stderr}"
    );
}
