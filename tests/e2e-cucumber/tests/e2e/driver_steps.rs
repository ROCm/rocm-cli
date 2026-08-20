// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The tools the driver plan legitimately shells out to on the DKMS path, minus
/// `sudo`. The scenario runs `rocm` with a `PATH` exposing only these, so the
/// plan can still execute its commands via `sh -c` while `sudo` is genuinely
/// absent — the exact state EAI-8053 reproduces. `sudo` is deliberately excluded;
/// `apt-get`/`curl` etc. are omitted too, so a fixed (uid-aware) CLI fails at a
/// real package step rather than at `sudo`, which is precisely the distinction the
/// contract asserts.
const PLAN_TOOLS_WITHOUT_SUDO: &[&str] = &["sh", "env"];

#[given("a root machine with no sudo command available")]
async fn setup_root_no_sudo(_world: &mut E2eWorld) {
    // Applicability is enforced by the scenario's @requires-root tag (the runner
    // must be root) and by the sanitized PATH the When step builds (no `sudo`);
    // nothing to arrange on the World here.
}

#[when("the user installs the native driver with dkms")]
async fn user_installs_driver(world: &mut E2eWorld) {
    let (stdout, stderr, rc, _path) = crate::run_rocm_with_only_tools(
        world,
        &["install", "driver", "--dkms", "--yes"],
        PLAN_TOOLS_WITHOUT_SUDO,
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the install does not fail merely because sudo is missing")]
async fn assert_not_broken_by_missing_sudo(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    let combined = format!("{stdout}\n{stderr}");
    // The bug's signature: the plan prefixed `sudo` even though the process is
    // root, so `sh -c "sudo …"` dies with `sudo: not found` and the CLI reports
    // the sudo-prefixed command as the failure. A root-aware plan would run the
    // command without `sudo` and get to the actual package work. We assert the
    // ABSENCE of the sudo-not-found failure, not a successful install (impossible
    // in a container), so the row goes stale the day the prefix becomes uid-aware.
    assert!(
        !combined.contains("sudo: not found")
            && !combined.contains("sudo: command not found")
            && !combined.contains("driver command failed: sudo "),
        "driver install as root failed because it invoked a missing `sudo`:\n{combined}"
    );
}
