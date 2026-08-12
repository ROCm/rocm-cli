// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the ROCm-preserving guard on automatic system-package installs.
//!
//! `rocm engines install vllm` installs the OpenMPI runtime vLLM needs through
//! the host package manager. `apt-get install -y` assumes yes for *removals* as
//! well as installs, so a dependency solution that evicts the ROCm stack used to
//! be applied without asking. These steps stand in for the package manager so a
//! scenario can present that exact solution and assert the CLI refuses it.

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Runtime key for the planted managed runtime. Any key works; the CLI matches
/// the registry entry by name.
const RUNTIME_KEY: &str = "therock-release-gfx94X-dcgpu-e2e-guard";

/// The ROCm packages the planted `apt-get` simulation reports as removals,
/// mirroring the set seen in the field report.
const REMOVED_ROCM: &[&str] = &[
    "rocm",
    "rocm-hip",
    "rocm-hip-runtime-dev",
    "mivisionx-dev",
    "rpp-dev",
];

fn shim_dir(world: &E2eWorld) -> PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    root.path().join("guard-bin")
}

/// Write `body` to `path` and mark it executable.
fn write_shim(path: &PathBuf, body: &str) {
    std::fs::write(path, body)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| panic!("failed to mark {} executable: {e}", path.display()));
    }
}

#[given("a machine with a registered ROCm runtime")]
async fn plant_runtime(world: &mut E2eWorld) {
    // Plant the registry entry `rocm install sdk` would have written, so the
    // engine install reaches the dependency-setup step without a GPU or a
    // multi-GiB SDK pull. Black-box: plain JSON matching the CLI's on-disk
    // schema, not a typed import from the product crates.
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let install_root = root.path().join("guard-runtime");
    std::fs::create_dir_all(&install_root).expect("failed to create the planted install root");

    let registry = root.path().join("data").join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create the runtime registry dir");

    let manifest = serde_json::json!({
        "runtime_key": RUNTIME_KEY,
        "runtime_id": RUNTIME_KEY,
        "channel": "release",
        "format": "wheel",
        "family": "gfx94X-dcgpu",
        "family_source": "e2e",
        "version": "7.14.0",
        "install_root": install_root.display().to_string(),
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "installed_at_unix_ms": 1,
    });
    std::fs::write(
        registry.join(format!("{RUNTIME_KEY}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .expect("failed to write the planted runtime manifest");
}

#[given("installing OpenMPI would remove the ROCm packages")]
async fn plant_destructive_apt(world: &mut E2eWorld) {
    // Stand in for the host package manager so the scenario owns the dependency
    // solution apt reports. `apt-get -s` (the simulation the CLI runs before
    // installing) prints the operation records apt itself would print; any other
    // apt invocation succeeds silently. A companion `ldconfig` shim reports no
    // libmpi, so OpenMPI reads as missing whatever the host has installed.
    let shim = shim_dir(world);
    std::fs::create_dir_all(&shim).expect("failed to create the package-manager shim dir");

    let removals = REMOVED_ROCM
        .iter()
        .map(|package| format!("echo 'Remv {package} [7.14.0]'"))
        .collect::<Vec<_>>()
        .join("\n");
    let apt_body = format!(
        "#!/bin/sh\nfor arg in \"$@\"; do\n  if [ \"$arg\" = -s ]; then\n{removals}\n    echo 'Inst openmpi-bin (4.1.6 [amd64])'\n    exit 0\n  fi\ndone\nexit 0\n"
    );
    write_shim(&shim.join("apt-get"), &apt_body);
    write_shim(&shim.join("ldconfig"), "#!/bin/sh\nexit 0\n");
}

#[when("the user installs the vLLM engine and approves system changes")]
async fn install_vllm_approving_changes(world: &mut E2eWorld) {
    // A PATH of only the shim dir keeps the run hermetic: the CLI finds the
    // planted `apt-get` / `ldconfig`, and finds no `mpirun`, so OpenMPI reads as
    // missing on any host. `--yes` is the approval the guard has to override —
    // that is the whole point, since the reported break happened unattended.
    let shim = shim_dir(world);
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &[
            "engines",
            "install",
            "vllm",
            "--runtime-id",
            RUNTIME_KEY,
            "--yes",
        ],
        &[("PATH", shim.display().to_string().as_str())],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Everything the CLI wrote to either stream, which is what a user sees.
fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[then("the CLI refuses instead of removing them")]
async fn assert_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no exit code recorded");
    let output = combined_output(world);
    assert!(
        rc != 0,
        "expected a non-zero exit for the refusal, got {rc}:\n{output}"
    );
    assert!(
        output.contains("would remove ROCm packages"),
        "expected the CLI to say it refused because ROCm would be removed:\n{output}"
    );
}

#[then("it does so before changing anything on the system")]
async fn assert_refused_before_install(world: &mut E2eWorld) {
    let output = combined_output(world);
    // The apt shim records nothing, so assert on the CLI's own account of what
    // it did: the refusal must precede the engine install it was guarding, or
    // the damage is already done by the time the user reads it.
    assert!(
        !output.contains("engine install"),
        "the engine install proceeded despite the refusal:\n{output}"
    );
    assert!(
        !output.contains("status: installed"),
        "a system package install was reported despite the refusal:\n{output}"
    );
}

#[then("it lists every ROCm package that would have been removed")]
async fn assert_lists_removals(world: &mut E2eWorld) {
    let output = combined_output(world);
    for package in REMOVED_ROCM {
        assert!(
            output.contains(package),
            "the refusal did not name `{package}`:\n{output}"
        );
    }
}
