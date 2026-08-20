// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

/// A one-shot Python probe run against the managed runtime's OWN interpreter (not
/// the ambient one `rocm examine` uses). Emits JSON `{hip, version}` when torch
/// imports — `hip` is non-null for a ROCm/HIP build and null for a CUDA build — or
/// `{error}` when torch cannot be imported at all.
const TORCH_HIP_PROBE: &str = "import json,sys\n\
     out={}\n\
     try:\n\
     \x20 import torch\n\
     \x20 out['hip']=getattr(torch.version,'hip',None)\n\
     \x20 out['version']=torch.__version__\n\
     except Exception as ex:\n\
     \x20 out['error']=type(ex).__name__+': '+str(ex)\n\
     sys.stdout.write(json.dumps(out))\n";

/// Locate the active managed runtime's venv interpreter. `rocm examine` prints
/// `Folder: <install_root>` for the active runtime; the interpreter lives under a
/// `bin/python` (Unix) / `Scripts/python.exe` (Windows) inside that tree. The
/// exact env sub-layout is an internal detail, so search for the interpreter
/// rather than reconstruct the path — black-box, and tolerant of layout changes.
fn active_runtime_python(world: &E2eWorld) -> PathBuf {
    let (examine, _, _) = crate::run_rocm(world, &["examine"]);
    let folder = examine
        .lines()
        .find_map(|l| l.trim().strip_prefix("Folder:"))
        .map(str::trim)
        .unwrap_or_else(|| panic!("no active-runtime 'Folder:' line in examine:\n{examine}"));
    find_venv_python(Path::new(folder)).unwrap_or_else(|| {
        panic!("could not locate a venv python under the runtime folder {folder}")
    })
}

/// Depth-limited search for a `bin/python` (Unix) or `Scripts/python.exe`
/// (Windows) under `root`. The managed runtime keeps its interpreter a few levels
/// down; cap the walk so a pathological tree can't hang the scenario.
fn find_venv_python(root: &Path) -> Option<PathBuf> {
    #[cfg(windows)]
    let (bin, exe) = ("Scripts", "python.exe");
    #[cfg(not(windows))]
    let (bin, exe) = ("bin", "python");
    let mut frontier = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = frontier.pop() {
        let candidate = dir.join(bin).join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
        if depth >= 6 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                frontier.push((entry.path(), depth + 1));
            }
        }
    }
    None
}

/// Whether the interpreter's torch is a ROCm/HIP build.
fn torch_is_rocm(python: &Path) -> bool {
    let output = std::process::Command::new(python)
        .args(["-c", TORCH_HIP_PROBE])
        .output()
        .unwrap_or_else(|e| panic!("failed to run runtime python {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let data: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|_| panic!("torch probe returned non-JSON:\n{stdout}"));
    data.get("hip")
        .is_some_and(|hip| !hip.is_null() && hip.as_str().is_some_and(|s| !s.is_empty()))
}

/// The `nvidia-*` CUDA distributions installed in the interpreter's environment,
/// as reported by `pip list`. A ROCm runtime should have none; ComfyUI's install
/// dragging any in is the EAI-8051 defect.
fn nvidia_distributions(python: &Path) -> Vec<String> {
    let output = std::process::Command::new(python)
        .args(["-m", "pip", "list", "--format=freeze"])
        .output()
        .unwrap_or_else(|e| panic!("failed to run pip list on {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| line.split("==").next())
        .map(str::trim)
        .filter(|name| name.to_ascii_lowercase().starts_with("nvidia-"))
        .map(str::to_owned)
        .collect()
}

#[given("an isolated machine with a managed ROCm runtime")]
async fn setup_isolated_runtime(world: &mut E2eWorld) {
    // DELIBERATELY do NOT call `world.use_shared_runtimes()`: this scenario may
    // corrupt the runtime (that is the bug it pins), so it must own a private,
    // throwaway runtime prefix. Each World already has isolated ROCM_CLI_* dirs,
    // so a plain `install sdk` here lands in this scenario's own tree.
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    if stdout.contains("installed: none") {
        crate::run_rocm_ok(world, &["install", "sdk"]);
    }
    let (stdout, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    assert!(
        !stdout.contains("installed: none"),
        "no managed runtime is active after install:\n{stdout}"
    );
}

#[given("the runtime's torch is a ROCm build")]
async fn assert_baseline_rocm_torch(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    assert!(
        torch_is_rocm(&python),
        "baseline runtime torch is not a ROCm build; scenario premise absent ({})",
        python.display()
    );
    assert!(
        nvidia_distributions(&python).is_empty(),
        "runtime already has nvidia-* distributions before ComfyUI install; premise absent"
    );
}

#[when("the user installs ComfyUI")]
async fn user_installs_comfyui(world: &mut E2eWorld) {
    // Do NOT use run_rocm_ok: the real install exits non-zero while still leaving
    // the runtime damaged, so the exit code is not the contract (see the feature
    // comment). Capture the outcome for diagnostics only.
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "install"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the runtime's torch is still a ROCm build")]
async fn assert_torch_still_rocm(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    assert!(
        torch_is_rocm(&python),
        "ComfyUI install replaced the runtime's ROCm torch with a non-ROCm build ({})",
        python.display()
    );
}

#[then("no CUDA nvidia packages were added to the runtime")]
async fn assert_no_nvidia_packages(world: &mut E2eWorld) {
    let python = active_runtime_python(world);
    let nvidia = nvidia_distributions(&python);
    assert!(
        nvidia.is_empty(),
        "ComfyUI install added CUDA nvidia-* distributions to the ROCm runtime: {}",
        nvidia.join(", ")
    );
}
