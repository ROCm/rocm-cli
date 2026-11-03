// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! vLLM is refused up front on native Windows, by the real binary.
//!
//! `rocm engines list` already reports vLLM as `unsupported_native_windows`,
//! and `serve`/`engines install` take the same platform decision
//! (`engine_ruled_out_by_platform`). These tests prove the observable half:
//! a non-zero exit, the platform message instead of an unrelated runtime
//! error, and no managed-service record left behind. Native Windows only —
//! on Linux and WSL vLLM is a supported engine and nothing here applies.
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh scratch root for one test, so the CLI never reads or writes the
/// user's real config, data or cache.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "rocm-vllm-windows-gate-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create scratch root");
        Self { root }
    }

    fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    fn uv_marker(&self) -> PathBuf {
        self.root.join("uv-was-run.txt")
    }

    /// Write (idempotently) the recording `uv` stand-in and return its path.
    fn fake_uv(&self) -> PathBuf {
        let uv = self.root.join("fake-uv.cmd");
        std::fs::write(
            &uv,
            format!(
                "@echo off
echo %* > \"{}\"
exit /b 1
",
                self.uv_marker().display()
            ),
        )
        .expect("write fake uv");
        uv
    }

    fn rocm(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_rocm"))
            .args(args)
            .env("ROCM_CLI_CONFIG_DIR", self.root.join("config"))
            .env("ROCM_CLI_DATA_DIR", self.data_dir())
            .env("ROCM_CLI_CACHE_DIR", self.root.join("cache"))
            // A recording stand-in for `uv`, the tool a real vLLM install runs.
            // It never touches a network; it only leaves a marker, so a test can
            // tell whether an install got as far as running it.
            .env("ROCM_CLI_UV_BINARY", self.fake_uv())
            .stdin(std::process::Stdio::null())
            .output()
            .expect("run rocm")
    }

    /// Register a read-only runtime and make it the active one, so `serve`
    /// has a runtime to resolve. Without the gate, this is the shape that
    /// reported "managed service launched" and exited 0.
    ///
    /// Returns the runtime's `install_root`. A real vLLM install writes its
    /// environment under `<install_root>/engines` (`env_root_for_runtime`), so
    /// that is the folder the "nothing was created" checks look in.
    fn activate_runtime(&self) -> PathBuf {
        let key = "therock-release-gfx1151-1-0-0";
        let install_root = self.root.join("runtime-gfx1151");
        std::fs::create_dir_all(&install_root).expect("create install root");
        std::fs::write(install_root.join("payload.txt"), "payload").expect("write payload");
        // A runtime that resolves as a managed Python environment is what lets
        // an ungated vLLM install get as far as invoking `uv`.
        let python = install_root.join("python.exe");
        std::fs::write(&python, "not a real interpreter").expect("write python stand-in");
        let registry = self.data_dir().join("runtimes").join("registry");
        std::fs::create_dir_all(&registry).expect("create registry");
        let manifest = serde_json::json!({
            "runtime_key": key,
            "runtime_id": "therock-release:gfx1151",
            "channel": "release",
            "format": "tarball",
            "family": "gfx1151",
            "family_source": "manual",
            "version": "1.0.0",
            "install_root": install_root,
            "python_executable": python,
            "selected_artifact_url": format!("https://example.invalid/{key}.tar.gz"),
            "read_only": true,
            "installed_at_unix_ms": 1_700_000_000_000u64,
        });
        std::fs::write(
            registry.join(format!("{key}.json")),
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        let output = self.rocm(&["runtimes", "activate", key]);
        assert!(
            output.status.success(),
            "activating the planted runtime failed:\n{}",
            combined(&output)
        );
        install_root
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn combined(output: &Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Every file in the managed-service record folder. The refusal claims
/// nothing was launched, and a launch always writes a record here.
fn service_records(data_dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(data_dir.join("services"))
        .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default()
}

fn assert_refused_with_platform_message(output: &Output, remedy: &str) {
    let text = combined(output);
    assert!(
        !output.status.success(),
        "expected a refusal, but it exited 0:\n{text}"
    );
    assert!(
        text.contains("vLLM is not supported on native Windows"),
        "expected the native-Windows platform message:\n{text}"
    );
    assert!(
        text.contains("WSL or Linux") && text.contains(remedy),
        "expected the message to name WSL/Linux and `{remedy}`:\n{text}"
    );
    assert!(
        !text.contains("no active ROCm runtime"),
        "the platform refusal must come before runtime resolution:\n{text}"
    );
    assert!(
        !text.contains("managed service launched"),
        "nothing may be launched:\n{text}"
    );
}

#[test]
fn serve_vllm_is_refused_without_an_active_runtime() {
    let scratch = Scratch::new("serve-no-runtime");
    let output = scratch.rocm(&["serve", "nonexistent-model-xyz", "--engine", "vllm"]);
    assert_refused_with_platform_message(&output, "--engine lemonade");
    assert_eq!(service_records(&scratch.data_dir()), Vec::<PathBuf>::new());
}

/// A refused vLLM request must not have started an install. With a runtime
/// planted, an install that is not refused resolves it to a managed Python
/// environment and runs `uv pip install vllm` (the recording stand-in leaves
/// `uv-was-run.txt`), so the marker is the proof it was never reached. The
/// runtime's `<install_root>/engines` folder (`env_root_for_runtime`, where the
/// install is rooted) must also be untouched.
fn assert_no_install_started(scratch: &Scratch, install_root: &Path) {
    let marker = scratch.uv_marker();
    assert!(
        !marker.exists(),
        "a refused vLLM request must not run the install, but uv was invoked: {}",
        std::fs::read_to_string(&marker).unwrap_or_default()
    );
    let engines = install_root.join("engines");
    assert!(
        !engines.exists(),
        "a refused vLLM request must not start an install, but {} exists",
        engines.display()
    );
}

#[test]
fn serve_vllm_is_refused_with_an_active_runtime() {
    let scratch = Scratch::new("serve-runtime");
    let install_root = scratch.activate_runtime();
    let output = scratch.rocm(&["serve", "nonexistent-model-xyz", "--engine", "vllm"]);
    assert_no_install_started(&scratch, &install_root);
    assert_refused_with_platform_message(&output, "--engine lemonade");
    assert_eq!(service_records(&scratch.data_dir()), Vec::<PathBuf>::new());
}

#[test]
fn engines_install_vllm_is_refused() {
    let scratch = Scratch::new("install");
    let output = scratch.rocm(&["engines", "install", "vllm", "--yes"]);
    assert_refused_with_platform_message(&output, "rocm engines install lemonade");
}

#[test]
fn engines_install_vllm_is_refused_with_an_active_runtime() {
    let scratch = Scratch::new("install-runtime");
    let install_root = scratch.activate_runtime();
    let output = scratch.rocm(&["engines", "install", "vllm", "--yes"]);
    assert_no_install_started(&scratch, &install_root);
    assert_refused_with_platform_message(&output, "rocm engines install lemonade");
}

/// `engines shell` installs the engine itself when an active runtime has no
/// environment for it (`resolve_engine_env`), so it is an install route too.
#[test]
fn engines_shell_vllm_is_refused_with_an_active_runtime() {
    let scratch = Scratch::new("shell-runtime");
    let install_root = scratch.activate_runtime();
    let output = scratch.rocm(&["engines", "shell", "vllm"]);
    assert_no_install_started(&scratch, &install_root);
    assert_refused_with_platform_message(&output, "rocm engines install lemonade");
}
