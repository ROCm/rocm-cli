// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `comfyui.feature`.
//!
//! The runtime this install targets is entirely planted: a `wheel` manifest
//! with a satisfying `rocm_sdk` probe, a pre-existing source checkout (so the
//! CLI never attempts the real network download of ComfyUI's source archive)
//! and a fake Python interpreter that answers the torch-stack version probe
//! with valid, empty JSON. The only thing under test is what happens when the
//! `uv` dependency install that follows all of that fails.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

const RUNTIME_KEY: &str = "e2e-comfyui-runtime";

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
}

fn install_root(world: &E2eWorld) -> PathBuf {
    root(world).join("comfyui-fixture").join("install-root")
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create fixture directory");
    }
    std::fs::write(path, contents).expect("failed to write fixture file");
}

/// Writes an executable POSIX shell script, standing in for a real binary the
/// CLI shells out to (`uv`, the runtime's Python).
fn write_shim(path: &Path, body: &str) {
    write_fixture(path, body);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|e| panic!("failed to chmod fake {}: {e}", path.display()));
    }
}

#[given("a ready ROCm install with a ComfyUI checkout pending dependencies")]
async fn ready_install_pending_dependencies(world: &mut E2eWorld) {
    let install_root = install_root(world);

    // `validate_runtime_manifest_for_activation` requires this marker to exist
    // directly inside `install_root` for a non-read-only runtime.
    write_fixture(&install_root.join(".rocm-cli-runtime.json"), "{}");

    // A pre-existing source checkout with a non-torch-stack requirement makes
    // `install()` take the "use existing checkout" branch (no network) and
    // reach the `uv` install step (a requirements file with only
    // torch/torchvision/torchaudio would be filtered down to nothing and skip
    // it entirely).
    let source_dir = install_root.join("apps").join("comfyui").join("source");
    write_fixture(&source_dir.join("requirements.txt"), "numpy==1.26.0\n");

    // Fake Python interpreter: succeeds and prints valid (empty) JSON, so the
    // torch-stack constraint probe that runs before `uv` passes cleanly.
    let python = root(world)
        .join("comfyui-fixture")
        .join("python")
        .join("rocm-python");
    write_shim(&python, "#!/bin/sh\nprintf '{}'\n");

    // A `rocm_sdk` probe that satisfies `validate_rocm_sdk_runtime_probe`:
    // importable, with an existing root/bin dir and resolved amdhip64/hipblas
    // libraries.
    let sdk_root = install_root.join("rocm_sdk").join("root");
    let sdk_bin = install_root.join("rocm_sdk").join("bin");
    std::fs::create_dir_all(&sdk_root).expect("failed to create fake rocm_sdk root dir");
    std::fs::create_dir_all(&sdk_bin).expect("failed to create fake rocm_sdk bin dir");
    let amdhip64 = sdk_root.join("libamdhip64.so");
    let hipblas = sdk_root.join("libhipblas.so");
    write_fixture(&amdhip64, "");
    write_fixture(&hipblas, "");

    let registry = root(world).join("data").join("runtimes").join("registry");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": RUNTIME_KEY,
        "runtime_id": RUNTIME_KEY,
        "channel": "release",
        "format": "wheel",
        "family": "gfx94X-dcgpu",
        "family_source": "e2e",
        "version": "7.14.0",
        "install_root": install_root,
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "installed_at_unix_ms": 1u64,
        "python_executable": python,
        "rocm_sdk": {
            "import_ok": true,
            "root_path": sdk_root,
            "bin_path": sdk_bin,
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip64]},
                {"shortname": "hipblas", "paths": [hipblas]},
            ],
        },
    }))
    .expect("failed to serialize runtime manifest");
    write_fixture(&registry.join(format!("{RUNTIME_KEY}.json")), &manifest);
}

#[given("the ComfyUI dependency install with uv fails")]
async fn uv_install_fails(world: &mut E2eWorld) {
    let uv = root(world)
        .join("comfyui-fixture")
        .join("uv-bin")
        .join("uv");
    write_shim(&uv, "#!/bin/sh\nexit 7\n");
    world
        .command_env
        .push(("ROCM_CLI_UV_BINARY", uv.into_os_string()));
}

#[when("the user installs ComfyUI")]
async fn install_comfyui(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(
        world,
        &["comfyui", "install", "--runtime-id", RUNTIME_KEY],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI fails and names the install log it wrote")]
async fn cli_names_install_log(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    let rc = world.cli_rc.unwrap_or(0);
    assert!(
        rc != 0,
        "expected `rocm comfyui install` to fail, got rc={rc}\nstderr:\n{stderr}"
    );

    let logs_dir = install_root(world)
        .join("apps")
        .join("comfyui")
        .join("logs");
    let entries: Vec<_> = std::fs::read_dir(&logs_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", logs_dir.display()))
        .map(|entry| entry.expect("failed to read log dir entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("install-") && name.ends_with(".log"))
        })
        .collect();
    assert!(
        entries.len() == 1,
        "expected exactly one install log under {}, found {entries:?}",
        logs_dir.display()
    );
    let log_path = entries[0].display().to_string();

    assert!(
        stderr.contains("install ComfyUI dependencies: uv exited with"),
        "expected the uv-failure message in stderr, got:\n{stderr}"
    );
    assert!(
        stderr.contains(&log_path),
        "expected stderr to name the install log {log_path}, got:\n{stderr}"
    );
}
