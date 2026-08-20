// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Report the installed torch distribution's version string via `importlib.metadata`
/// — WITHOUT importing torch. Emits JSON `{version}` (e.g. `2.7.0+rocm6.4` for a
/// ROCm build, `2.7.0+cu128` for a CUDA build) or `{error}` when no torch
/// distribution is installed.
///
/// Deliberately does not `import torch`: importing it loads the ROCm/CUDA shared
/// libraries, which need the runtime's `LD_LIBRARY_PATH`/`ROCM_PATH` set up (the
/// product runs its own torch probe *with* that env, ours runs the interpreter
/// bare). The `+rocm` / `+cu` local-version label in the dist metadata is the
/// definitive ROCm-vs-CUDA discriminator and is readable with no native load — so
/// a bare interpreter suffices, and a runtime whose torch is present but whose
/// native libs aren't on our env no longer reads as "not a ROCm build".
const TORCH_DIST_PROBE: &str = "import json,sys\n\
     from importlib import metadata\n\
     out={}\n\
     try:\n\
     \x20 out['version']=metadata.version('torch')\n\
     except Exception as ex:\n\
     \x20 out['error']=type(ex).__name__+': '+str(ex)\n\
     sys.stdout.write(json.dumps(out))\n";

/// Locate the managed runtime's venv interpreter. `rocm runtimes list` prints an
/// `install_root: <path>` line for each installed runtime; the interpreter lives
/// under a `bin/python` (Unix) / `Scripts/python.exe` (Windows) inside that tree.
/// The exact env sub-layout is an internal detail, so search for the interpreter
/// rather than reconstruct the path — black-box, and tolerant of layout changes.
///
/// Reads `runtimes list` rather than `examine`: examine only prints a `Folder:`
/// line for the *active* runtime and takes a different branch when none is marked
/// active, so it is not a reliable source for the install root (this cost a GPU
/// dispatch — the scenario panicked on a missing `Folder:` there). `runtimes list`
/// prints `install_root:` for every installed runtime unconditionally.
fn active_runtime_python(world: &E2eWorld) -> PathBuf {
    let (listing, _, _) = crate::run_rocm(world, &["runtimes", "list"]);
    let root = listing
        .lines()
        .find_map(|l| l.trim().strip_prefix("install_root:"))
        .map_or_else(
            || panic!("no 'install_root:' line in `runtimes list`:\n{listing}"),
            str::trim,
        );
    find_venv_python(Path::new(root)).unwrap_or_else(|| {
        panic!("could not locate a venv python under the runtime install_root {root}")
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

/// The installed torch distribution's version string, or `None` if no torch
/// distribution is installed. Reads dist metadata without importing torch (see
/// [`TORCH_DIST_PROBE`]), so it works against a bare interpreter. A ROCm wheel
/// carries a `+rocm...` local version, a CUDA wheel `+cu...`; callers judge the
/// build from that label.
fn torch_version(python: &Path) -> Option<String> {
    let output = std::process::Command::new(python)
        .args(["-c", TORCH_DIST_PROBE])
        .output()
        .unwrap_or_else(|e| panic!("failed to run runtime python {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let data: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|_| {
        panic!("torch version probe returned non-JSON:\nstdout: {stdout}\nstderr: {stderr}")
    });
    data.get("version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Enumerate installed distributions via `importlib.metadata` and emit their names
/// as a JSON array. Used instead of `pip list` because uv-created managed runtimes
/// have no `pip` module — `python -m pip` there exits non-zero with empty stdout,
/// which a naive reader would misread as "no packages installed" and pass the
/// nvidia check while the runtime is actually corrupted. `importlib.metadata` is in
/// the stdlib, so it is always present; the probe emits `{names}` on success or
/// `{error}` on failure so the caller can fail loudly rather than treat a broken
/// probe as a clean result.
const DISTRIBUTIONS_PROBE: &str = "import json,sys\n\
     out={}\n\
     try:\n\
     \x20 from importlib import metadata\n\
     \x20 out['names']=sorted({(d.metadata['Name'] or '') for d in metadata.distributions()})\n\
     except Exception as ex:\n\
     \x20 out['error']=type(ex).__name__+': '+str(ex)\n\
     sys.stdout.write(json.dumps(out))\n";

/// The `nvidia-*` CUDA distributions installed in the interpreter's environment.
/// A ROCm runtime should have none; ComfyUI's install dragging any in is the
/// EAI-8051 defect. Panics if the interpreter cannot be run or the probe reports
/// an error — a probe that cannot enumerate packages must NOT read as "no nvidia
/// packages", which would pass the contract on a runtime it never actually checked.
fn nvidia_distributions(python: &Path) -> Vec<String> {
    let output = std::process::Command::new(python)
        .args(["-c", DISTRIBUTIONS_PROBE])
        .output()
        .unwrap_or_else(|e| panic!("failed to run runtime python {}: {e}", python.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let data: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|_| {
        panic!("distributions probe returned non-JSON:\nstdout: {stdout}\nstderr: {stderr}")
    });
    if let Some(error) = data.get("error").and_then(serde_json::Value::as_str) {
        panic!(
            "could not enumerate installed distributions on {}: {error}",
            python.display()
        );
    }
    let names = data
        .get("names")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("distributions probe returned no 'names' array:\n{stdout}"));
    names
        .iter()
        .filter_map(serde_json::Value::as_str)
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
    let version = torch_version(&python);
    assert!(
        version
            .as_deref()
            .is_some_and(|v| v.to_ascii_lowercase().contains("rocm")),
        "baseline runtime torch is not a ROCm build; scenario premise absent \
         (torch version: {version:?}, python: {})",
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
    let version = torch_version(&python);
    assert!(
        version
            .as_deref()
            .is_some_and(|v| v.to_ascii_lowercase().contains("rocm")),
        "ComfyUI install replaced the runtime's ROCm torch with a non-ROCm build \
         (torch version now: {version:?}, python: {})",
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
