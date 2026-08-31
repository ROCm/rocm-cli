// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the ComfyUI runtime-selection refusal.
//!
//! `rocm comfyui install` refuses when more than one managed ROCm runtime is
//! ready and none is activated, rather than guessing which one to install into.
//! These steps plant two ready wheel runtimes on disk (readiness is filesystem
//! and manifest state, so no GPU is needed) and assert the refusal is actionable
//! on both the CLI (`--runtime-id`, `rocm runtimes activate`) and the TUI
//! (`/runtimes`) surfaces it reaches. Black-box: the planted registry manifests
//! are plain JSON matching the CLI's on-disk schema, not typed imports from the
//! product crates.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The two runtime keys planted for the scenario. Distinct so the assertion that
/// the refusal lists both is meaningful.
const RUNTIME_KEYS: [&str; 2] = [
    "release-wheel-gfx94x-dcgpu-7-13-0",
    "nightly-wheel-gfx94x-dcgpu-7-14-0",
];

/// The scenario's isolated `data` dir — where the CLI reads its runtime registry
/// (`ROCM_CLI_DATA_DIR`, set by `isolate_env`).
fn data_dir(world: &E2eWorld) -> PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    root.path().join("data")
}

/// Write `body` to `path`, creating parent dirs first.
fn write_stub(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
    }
    std::fs::write(path, body)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

/// Plant one ready wheel runtime: the on-disk stubs the CLI's readiness check
/// requires (an install root holding its local manifest, a Python executable,
/// and a rocm_sdk bin exposing amdhip64 + hipblas) plus the registry manifest
/// that points at them. The readiness gate validates recorded manifest state and
/// that these paths exist — it never executes anything — so a GPU-less host can
/// present a runtime the CLI accepts as "ready".
fn plant_ready_runtime(data: &Path, key: &str) {
    let install_root = data.join("runtimes").join("roots").join(key);
    let sdk_root = install_root.join("sdk");
    let sdk_bin = sdk_root.join("bin");
    let python = install_root.join("bin").join("python3");
    let amdhip = sdk_bin.join("libamdhip64.so");
    let hipblas = sdk_bin.join("libhipblas.so");

    write_stub(&install_root.join(".rocm-cli-runtime.json"), "{}");
    write_stub(&python, "#!/bin/sh\nexit 0\n");
    write_stub(&amdhip, "stub");
    write_stub(&hipblas, "stub");

    let manifest = serde_json::json!({
        "runtime_key": key,
        "runtime_id": key,
        "channel": "release",
        "format": "wheel",
        "family": "gfx94X-dcgpu",
        "family_source": "e2e",
        "version": "7.13.0",
        "install_root": install_root.display().to_string(),
        "selected_artifact_url": "https://example.invalid/e2e.whl",
        "python_executable": python.display().to_string(),
        "rocm_sdk": {
            "import_ok": true,
            "root_path": sdk_root.display().to_string(),
            "bin_path": sdk_bin.display().to_string(),
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip.display().to_string()]},
                {"shortname": "hipblas", "paths": [hipblas.display().to_string()]},
            ],
        },
        "installed_at_unix_ms": 1,
    });

    let registry = data.join("runtimes").join("registry");
    std::fs::create_dir_all(&registry)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", registry.display()));
    std::fs::write(
        registry.join(format!("{key}.json")),
        serde_json::to_vec_pretty(&manifest).expect("manifest serialises"),
    )
    .unwrap_or_else(|e| panic!("failed to write the planted runtime manifest: {e}"));
}

/// Combined stdout+stderr of the recorded `rocm` invocation. The refusal is an
/// `anyhow` error printed to stderr, so both streams are searched.
fn refusal_text(world: &E2eWorld) -> String {
    let stdout = world.cli_output.clone().unwrap_or_default();
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    format!("{stdout}\n{stderr}")
}

#[given("two ready ROCm runtimes and no active default")]
async fn plant_two_ready_runtimes(world: &mut E2eWorld) {
    // No `active.json` and no `activate` step: with two ready runtimes and no
    // configured default, the CLI must refuse to guess rather than auto-select.
    let data = data_dir(world);
    for key in RUNTIME_KEYS {
        plant_ready_runtime(&data, key);
    }
}

#[when("the user installs ComfyUI without choosing a runtime")]
async fn install_comfyui_without_runtime(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "install"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("ComfyUI install is refused as ambiguous")]
async fn comfyui_install_refused(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "expected a non-zero refusal, got rc={:?}\n{text}",
        world.cli_rc
    );
    assert!(
        text.contains("Multiple ROCm runtimes are ready"),
        "expected the ambiguity refusal, got:\n{text}"
    );
}

#[then("the refusal offers the /runtimes picker")]
async fn refusal_offers_runtimes_picker(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("/runtimes"),
        "refusal should point to the TUI `/runtimes` picker, got:\n{text}"
    );
}

#[then("the refusal names the --runtime-id flag")]
async fn refusal_names_runtime_id(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("--runtime-id"),
        "refusal should name the `--runtime-id` flag, got:\n{text}"
    );
}

#[then("the refusal names rocm runtimes activate")]
async fn refusal_names_activate(world: &mut E2eWorld) {
    let text = refusal_text(world);
    assert!(
        text.contains("rocm runtimes activate"),
        "refusal should name the durable `rocm runtimes activate` remedy, got:\n{text}"
    );
}

#[then("the refusal lists both runtime keys")]
async fn refusal_lists_both_keys(world: &mut E2eWorld) {
    let text = refusal_text(world);
    for key in RUNTIME_KEYS {
        assert!(
            text.contains(key),
            "refusal should list runtime key `{key}`, got:\n{text}"
        );
    }
}
