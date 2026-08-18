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
    // On a no-GPU host this is a no-op: a managed TheRock SDK runtime can only be
    // installed where there's a GPU family to select wheels for (see
    // `runtime-install-active`, @requires-gpu). The only scenarios that reach this
    // step without `@requires-gpu` are the mock-lane chat scenarios, whose serve
    // is backed by MockServer and needs no runtime at all — so skip the install
    // rather than attempt a multi-GiB SDK pull that can't succeed here.
    if !e2e_cucumber::capability::host_capability().has_amd_gpu {
        return;
    }
    // This precondition only needs *a* runtime present — it does not assert a
    // clean slate — so opt into the shared runtimes tree: the first scenario to
    // hit an empty shared tree installs once, and every later scenario finds the
    // runtime already there instead of re-installing a multi-GiB TheRock SDK
    // (the per-scenario install count is what blew the GPU time cap). No-op unless
    // E2E_SHARED_RUNTIMES_DIR is set (CI on a persistent runner).
    world.use_shared_runtimes();
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        crate::run_rocm_ok(world, &["install", "sdk"]);
    }
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no managed runtime is active:\n{stdout}"
    );
}

#[when("the user installs the SDK")]
async fn user_installs_sdk(world: &mut E2eWorld) {
    let stdout = crate::run_rocm_ok(world, &["install", "sdk"]);
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

#[then("the runtime excludes the compiler toolchain")]
async fn assert_runtime_excludes_devel(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated state root")
        .path();
    let registry = root.join("data/runtimes/registry");
    let entries = std::fs::read_dir(&registry).unwrap_or_else(|error| {
        panic!(
            "failed to read runtime registry {}: {error}",
            registry.display()
        )
    });
    let manifests = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    assert_eq!(
        manifests.len(),
        1,
        "expected one freshly installed runtime manifest in {}: {manifests:?}",
        registry.display()
    );

    let manifest_path = &manifests[0];
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(manifest_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display())),
    )
    .unwrap_or_else(|error| panic!("failed to parse {}: {error}", manifest_path.display()));
    assert_eq!(
        manifest.get("devel").and_then(serde_json::Value::as_bool),
        Some(false),
        "default SDK install recorded the compiler toolchain as present: {manifest}"
    );

    let python = manifest
        .get("python_executable")
        .and_then(serde_json::Value::as_str)
        .expect("wheel runtime manifest has no python_executable");
    let output = std::process::Command::new(python)
        .args([
            "-c",
            "import importlib.metadata as m; print('present' if any(d.metadata['Name'].lower() == 'rocm-sdk-devel' for d in m.distributions()) else 'absent')",
        ])
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to inspect the installed runtime with {python}: {error}")
        });
    assert!(
        output.status.success(),
        "failed to enumerate installed runtime packages:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "absent",
        "default SDK install pulled rocm-sdk-devel back transitively"
    );
}

#[then("the managed runtime folder path is not recursively nested")]
async fn assert_runtime_path_not_nested(world: &mut E2eWorld) {
    // `rocm examine` prints `Folder: <install_root>` for the active runtime.
    let output = world.cli_output.as_ref().expect("no examine output");
    let Some(folder) = output
        .lines()
        .find_map(|l| l.trim().strip_prefix("Folder:"))
        .map(str::trim)
    else {
        panic!("no 'Folder:' line in examine output:\n{output}");
    };
    // A healthy path contains `runtimes/wheel` at most once. Re-provisioning
    // inside an existing runtime produces `runtimes/wheel/.../runtimes/wheel/`
    // (dogfooding #17). Count occurrences of the marker segment.
    let nested = folder.matches("runtimes/wheel").count() > 1
        || folder.matches("runtimes\\wheel").count() > 1;
    assert!(
        !nested,
        "managed runtime folder path is recursively nested (dogfooding #17):\n{folder}"
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
