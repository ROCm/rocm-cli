// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Probing an installed runtime through its own interpreter: what `rocm_sdk`
//! reports (`probe_rocm_sdk_runtime`), what its torch can open
//! (`probe_runtime_devices`), and which torch is installed against which an
//! engine pins (`probe_torch_alignment`), with the Python scripts each one runs
//! and the torch-pin helpers (`split_local_version`, `requirement_pinned_version`,
//! `install_pinned_package`) the engine install uses to put the SDK's torch back.
//!
//! Builds on `process` and `python`; nothing in those depends on this module.

use super::process::{capture_python_stdout_with_env, run_command_with_env};
use super::python::split_runtime_path;
use crate::runtime_registry::RocmSdkPythonProbe;
use anyhow::{Context, Result};
use rocm_core::{AppPaths, RUNTIME_LIBRARY_PATH_ENV};
use std::path::{Path, PathBuf};

/// What `rocm_sdk` reports about an installed runtime.
///
/// A newly composed aggregate runtime is probed with the exact device payload
/// selected from the canonical source. Without that input `rocm_sdk` may choose
/// an unrelated default target even though only one device package was installed,
/// producing library paths and a kernel check for the wrong GPU. Adopted legacy
/// runtimes remain unforced so the probe reports their existing composition.
pub(crate) fn probe_rocm_sdk_runtime(python_executable: &Path) -> Result<RocmSdkPythonProbe> {
    probe_rocm_sdk_runtime_for_target(python_executable, None)
}

pub(super) fn probe_rocm_sdk_runtime_for_target(
    python_executable: &Path,
    device_target: Option<&str>,
) -> Result<RocmSdkPythonProbe> {
    let env = device_target
        .map(|target| vec![("ROCM_SDK_TARGET_FAMILY".to_owned(), target.to_owned())])
        .unwrap_or_default();
    let text = capture_python_stdout_with_env(
        python_executable,
        ROCM_SDK_PROBE_SCRIPT,
        &env,
        "launch rocm_sdk probe",
    )
    .with_context(|| {
        format!(
            "failed to launch rocm_sdk probe via {}",
            python_executable.display()
        )
    })?;
    parse_rocm_sdk_probe(&text)
}

fn parse_rocm_sdk_probe(output: &str) -> Result<RocmSdkPythonProbe> {
    serde_json::from_str(output.trim()).context("failed to parse rocm_sdk probe output")
}

/// What the runtime's torch reports about the GPUs it can actually open, and
/// whether those GPUs can actually run a kernel.
///
/// [`validate_rocm_sdk_runtime_probe`] establishes that the SDK's libraries are
/// present and resolvable. That is not the same question as whether the torch
/// sharing the venv can enumerate a device: a torch built against a different
/// HIP version loads happily against those libraries and then reports no
/// devices at all.
///
/// [`validate_rocm_sdk_runtime_probe`]: crate::runtime_registry::validate_rocm_sdk_runtime_probe
///
/// Enumeration succeeding is in turn not the same question as the device being
/// usable. A torch built against a different HIP version can enumerate the
/// GPUs and then fault on the first kernel it launches, so the two failures are
/// reported separately and must not be conflated.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct RuntimeDeviceProbe {
    pub import_ok: bool,
    pub torch_version: Option<String>,
    pub hip_version: Option<String>,
    /// `None` when torch never imported, so "unknown" stays distinct from "zero".
    pub device_count: Option<u32>,
    /// An import or enumeration failure. Never a kernel failure.
    pub error: Option<String>,
    /// A GPU kernel failure observed *after* devices enumerated successfully.
    /// `None` when no kernel was attempted (no devices, or enumeration failed).
    #[serde(default)]
    pub kernel_error: Option<String>,
}

/// Ask the runtime's own interpreter how many devices its torch can open.
///
/// `library_paths` must be the runtime's recorded ROCm library directories (see
/// [`RocmSdkPythonProbe::library_paths`]). They are prepended to
/// `LD_LIBRARY_PATH` for the child, which is how a served process resolves them.
///
/// This cannot be replaced with an in-process `rocm_sdk.initialize_process()`
/// call in the probe script. Measured on MI300X against a runtime that serves
/// correctly: with the library directories on `LD_LIBRARY_PATH` torch reports 8
/// devices, and with `initialize_process()` alone it reports 0. A probe built on
/// the latter would fail healthy runtimes and send people to reinstall them,
/// which is the operation that breaks them.
pub(crate) fn probe_runtime_devices(
    python_executable: &Path,
    library_paths: &[PathBuf],
) -> Result<RuntimeDeviceProbe> {
    let mut env = Vec::new();
    if !library_paths.is_empty() {
        let mut entries = library_paths.to_vec();
        if let Some(existing) = std::env::var_os(RUNTIME_LIBRARY_PATH_ENV) {
            entries.extend(split_runtime_path(&existing));
        }
        let joined = std::env::join_paths(entries)
            .context("failed to compose the runtime library path for the device probe")?;
        env.push((
            RUNTIME_LIBRARY_PATH_ENV.to_owned(),
            joined.to_string_lossy().into_owned(),
        ));
    }
    let text = capture_python_stdout_with_env(
        python_executable,
        RUNTIME_DEVICE_PROBE_SCRIPT,
        &env,
        "launch runtime device probe",
    )
    .with_context(|| {
        format!(
            "failed to launch runtime device probe via {}",
            python_executable.display()
        )
    })?;
    parse_runtime_device_probe(&text)
}

fn parse_runtime_device_probe(output: &str) -> Result<RuntimeDeviceProbe> {
    serde_json::from_str(output.trim()).context("failed to parse runtime device probe output")
}

/// Reports what torch sees, never raising: an unusable runtime must be
/// described, not turned into a probe crash.
const RUNTIME_DEVICE_PROBE_SCRIPT: &str = r#"
import json

out = {
    "import_ok": False,
    "torch_version": None,
    "hip_version": None,
    "device_count": None,
    "error": None,
    "kernel_error": None,
}

try:
    import torch

    out["import_ok"] = True
    out["torch_version"] = getattr(torch, "__version__", None)
    out["hip_version"] = getattr(getattr(torch, "version", None), "hip", None)
    out["device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)

# Enumeration is not execution. A runtime whose torch and HIP disagree can
# report devices and then fault on the first kernel, so the kernel is launched
# under its own guard and its failure is recorded in its own field. Only run it
# once enumeration actually produced a device: with no devices there is nothing
# to execute on, and a failed import has already been described.
if out["error"] is None and (out["device_count"] or 0) > 0:
    try:
        probe = torch.ones(32, device="cuda")
        probe.add_(1.0)
        torch.cuda.synchronize()
    except Exception as exc:
        out["kernel_error"] = type(exc).__name__ + ": " + str(exc)

print(json.dumps(out))
"#;

/// What torch is installed, and what torch the engine's metadata demands.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct TorchAlignmentProbe {
    /// The installed torch, e.g. `2.11.0+rocm7.13.0`. `None` if torch is absent.
    pub installed_torch: Option<String>,
    /// The engine's pinned requirement, e.g. `torch==2.11.0+gitd0c8b1f`. `None`
    /// when the engine is not installed yet or does not pin torch.
    pub engine_requires_torch: Option<String>,
}

/// Read installed/required torch from distribution metadata, without importing
/// torch. Metadata questions must not depend on the runtime being usable — this
/// is called both before and after an engine install, and in the broken state
/// importing torch is exactly what fails.
pub(crate) fn probe_torch_alignment(
    python_executable: &Path,
    engine_distribution: &str,
) -> Result<TorchAlignmentProbe> {
    let env = vec![(
        "ROCM_CLI_PROBE_DIST".to_owned(),
        engine_distribution.to_owned(),
    )];
    let text = capture_python_stdout_with_env(
        python_executable,
        TORCH_ALIGNMENT_PROBE_SCRIPT,
        &env,
        "launch torch alignment probe",
    )?;
    serde_json::from_str(text.trim()).context("failed to parse torch alignment probe output")
}

const TORCH_ALIGNMENT_PROBE_SCRIPT: &str = r#"
import json
import os
import re
import importlib.metadata as md

out = {"installed_torch": None, "engine_requires_torch": None}

try:
    out["installed_torch"] = md.version("torch")
except Exception:
    pass

try:
    for raw in md.requires(os.environ.get("ROCM_CLI_PROBE_DIST", "vllm")) or []:
        # `Requires-Dist` entries carry environment markers after ';'. Only the
        # requirement itself matters here.
        requirement = raw.split(";")[0].strip()
        # The distribution name runs up to the first specifier, extra or space -
        # and there is usually no space at all, as in `torch==2.11.0+gitd0c8b1f`.
        matched = re.match(r"[A-Za-z0-9._-]+", requirement)
        if matched is None:
            continue
        if matched.group(0).lower().replace("_", "-") == "torch":
            out["engine_requires_torch"] = requirement
            break
except Exception:
    pass

print(json.dumps(out))
"#;

/// Split a version into its public part and its local segment.
///
/// `2.11.0+rocm7.13.0` -> `("2.11.0", Some("rocm7.13.0"))`. The local segment is
/// the build identifier: for TheRock wheels it names the ROCm build, and for the
/// engine's own index it is an opaque commit tag.
///
/// Defined in `rocm-core` because the vLLM engine has to make the same split to
/// recognise a runtime the CLI deliberately realigned; two copies of this would be
/// two places for the two sides to drift apart.
pub(crate) fn split_local_version(version: &str) -> (&str, Option<&str>) {
    rocm_core::uv::split_local_version(version)
}

/// The version pinned by a `==` requirement, e.g. `torch==2.11.0+git…` -> the
/// version. Returns `None` for any looser requirement, since only an exact pin
/// tells us which release the engine was built against.
pub(crate) fn requirement_pinned_version(requirement: &str) -> Option<&str> {
    let (_, version) = requirement.split_once("==")?;
    let version = version.trim();
    if version.is_empty() || version.contains(',') {
        return None;
    }
    Some(version)
}

/// Install one exact package version from `index_url` into `python_executable`.
///
/// Used to put the SDK's build of a package back after another installer has
/// replaced it. `--reinstall-package` is required: without it uv treats the
/// already-present distribution as satisfying the request and does nothing.
///
/// `--index-url`, not `--extra-index-url`, so the SDK index is the only place a
/// candidate can come from. This matches how `install_therock_runtime` installs
/// from the same index, and it is load-bearing rather than cosmetic: with PyPI
/// left in the candidate set, a `+rocm` build the SDK index does not publish can
/// resolve against PyPI instead, and the caller's `Unavailable` classification —
/// the whole point of which is to say "the SDK index has no such build" — never
/// gets the resolver error it keys on.
///
/// `--no-deps` because this is a surgical swap of one build for another build of
/// the *same release*: the environment already carries a resolved dependency tree
/// and re-resolving it here is free to move `torchvision`/`torchaudio` as a side
/// effect, which is the mixed stack this change is trying not to create. A build
/// that genuinely needs a different dependency is not silently ignored — the
/// `uv pip check` that runs immediately after reports it as a violation.
pub(crate) fn install_pinned_package(
    paths: &AppPaths,
    python_executable: &Path,
    index_url: &str,
    package: &str,
    requirement: &str,
) -> Result<()> {
    let uv = rocm_core::uv::ensure_uv_binary(paths)
        .context("failed to acquire uv binary for the torch alignment install")?;
    let mut args = rocm_core::uv::uv_pip_install_base(python_executable);
    args.push("--index-url".to_owned());
    args.push(index_url.to_owned());
    args.push("--no-deps".to_owned());
    args.push("--reinstall-package".to_owned());
    args.push(package.to_owned());
    args.push(requirement.to_owned());
    let borrowed = args.iter().map(String::as_str).collect::<Vec<_>>();
    // `run_command_with_env`, not `run_command`, for two reasons. The uv environment
    // carries `UV_HTTP_TIMEOUT` (uv has no `--timeout` flag) and `UV_CACHE_DIR`;
    // without the latter uv falls back to `$HOME/.cache/uv`, loses hardlinking when
    // that is on another filesystem, and silently copies the whole torch stack per
    // environment — and the e2e lanes' shared cache is threaded through the same
    // helper. It also captures stderr on every platform, where `run_command`'s
    // Windows branch reports only an exit status; the caller classifies this
    // install's outcome by matching the resolver's message, so on Windows an
    // unpublished build would otherwise be reported as a generic failure.
    run_command_with_env(
        &uv,
        &borrowed,
        &rocm_core::uv::uv_command_env(paths),
        "install the SDK build of the engine's torch",
    )
}

const ROCM_SDK_PROBE_SCRIPT: &str = r#"
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sysconfig

out = {
    "import_ok": False,
    "rocm_sdk_version": None,
    "site_packages": sysconfig.get_paths().get("purelib"),
    "root_path": None,
    "bin_path": None,
    "cmake_path": None,
    "runtime_roots": [],
    "bin_paths": [],
    "library_paths": [],
    "default_target_family": None,
    "available_target_families": [],
    "resolved_target_family": None,
    "packages": [],
    "library_shortnames": [],
    "resolved_libraries": [],
    "error": None,
}

def add_path(key, path):
    if path is None:
        return
    value = str(path)
    if value not in out[key]:
        out[key].append(value)

def package_root(package, target_family=None):
    module_name = package.get_py_package_name(target_family)
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).parent

def add_runtime_root(root):
    if root is None:
        return
    add_path("runtime_roots", root)
    for child in [root / "bin", root / "lib", root / "lib64", root / "lib" / "rocm_sysdeps" / "lib"]:
        if child.is_dir():
            if child.name == "bin":
                add_path("bin_paths", child)
            add_path("library_paths", child)

try:
    import rocm_sdk
    from rocm_sdk import _dist_info as di

    out["import_ok"] = True
    out["rocm_sdk_version"] = getattr(rocm_sdk, "__version__", None)
    out["default_target_family"] = getattr(di, "DEFAULT_TARGET_FAMILY", None)
    out["available_target_families"] = list(getattr(di, "AVAILABLE_TARGET_FAMILIES", []))
    try:
        from rocm_sdk import _devel
        root_path = _devel.get_devel_root()
        out["root_path"] = str(root_path)
        out["bin_path"] = str(root_path / "bin")
        out["cmake_path"] = str(root_path / "lib" / "cmake")
        add_runtime_root(root_path)
    except Exception as exc:
        out["root_path_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        out["resolved_target_family"] = di.determine_target_family()
    except Exception as exc:
        out["resolved_target_family_error"] = type(exc).__name__ + ": " + str(exc)

    target_family = out["resolved_target_family"] or out["default_target_family"]
    for logical_name, target in [
        ("core", None),
        ("libraries", target_family),
        ("device", target_family),
        ("profiler", None),
    ]:
        try:
            package = di.ALL_PACKAGES[logical_name]
            if package.has_py_package(target):
                add_runtime_root(package_root(package, target))
        except Exception as exc:
            out.setdefault("package_root_errors", {})[logical_name] = type(exc).__name__ + ": " + str(exc)

    scripts_path = sysconfig.get_path("scripts")
    if scripts_path:
        scripts_path = Path(scripts_path)
        if scripts_path.is_dir():
            add_path("bin_paths", scripts_path)

    if out["root_path"] is None and out["runtime_roots"]:
        out["root_path"] = out["runtime_roots"][0]
    if out["bin_path"] is None and out["bin_paths"]:
        out["bin_path"] = out["bin_paths"][0]
    if out["cmake_path"] is None and out["root_path"] is not None:
        cmake_path = Path(out["root_path"]) / "lib" / "cmake"
        if cmake_path.is_dir():
            out["cmake_path"] = str(cmake_path)

    out["library_shortnames"] = sorted(getattr(di, "ALL_LIBRARIES", {}).keys())
    resolved_libraries = []
    for shortname in out["library_shortnames"]:
        try:
            paths = [str(path) for path in rocm_sdk.find_libraries(shortname)]
        except Exception:
            paths = []
        if paths:
            resolved_libraries.append({"shortname": shortname, "paths": paths})
    out["resolved_libraries"] = resolved_libraries

    packages = []
    for dist in md.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower().startswith("rocm"):
            packages.append({"name": name, "version": dist.version})
    out["packages"] = sorted(packages, key=lambda item: item["name"].lower())
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)

print(json.dumps(out))
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_registry::validate_rocm_sdk_runtime_probe;
    use crate::therock::test_support::test_paths;
    use std::fs;

    #[test]
    fn parses_rocm_sdk_probe_contract() -> Result<()> {
        let root_path = if cfg!(windows) {
            PathBuf::from(r"C:\venv\Lib\site-packages\_rocm_sdk_devel")
        } else {
            PathBuf::from("/tmp/venv/lib/python3.12/site-packages/_rocm_sdk_devel")
        };
        let bin_path = root_path.join("bin");
        let cmake_path = root_path.join("lib").join("cmake");
        let site_packages = root_path
            .parent()
            .expect("test root has a parent")
            .display()
            .to_string();
        let payload = serde_json::json!({
            "import_ok": true,
            "rocm_sdk_version": "7.13.0a20260423",
            "site_packages": site_packages,
            "root_path": root_path,
            "bin_path": bin_path,
            "cmake_path": cmake_path,
            "runtime_roots": [root_path],
            "bin_paths": [bin_path],
            "library_paths": [root_path.join("lib")],
            "default_target_family": "gfx1151",
            "available_target_families": ["gfx1151"],
            "resolved_target_family": "gfx1151",
            "packages": [{"name": "rocm", "version": "7.13.0a20260423"}],
            "library_shortnames": ["amdhip64", "hipblas"],
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [root_path.join("bin").join("amdhip64_7.dll")]},
                {"shortname": "hipblas", "paths": [root_path.join("bin").join("hipblas.dll")]}
            ],
            "error": null
        })
        .to_string();
        let probe = parse_rocm_sdk_probe(&payload)?;

        assert!(probe.import_ok);
        assert_eq!(probe.rocm_sdk_version.as_deref(), Some("7.13.0a20260423"));
        assert_eq!(probe.resolved_target_family.as_deref(), Some("gfx1151"));
        assert_eq!(
            probe
                .root_path
                .as_deref()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str()),
            Some("_rocm_sdk_devel")
        );
        assert_eq!(
            probe
                .bin_path
                .as_deref()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str()),
            Some("bin")
        );
        assert_eq!(probe.available_target_families, vec!["gfx1151"]);
        assert_eq!(probe.packages[0].name, "rocm");
        assert!(probe.library_shortnames.contains(&"amdhip64".to_owned()));
        assert_eq!(probe.resolved_libraries.len(), 2);
        Ok(())
    }

    #[test]
    fn runtime_only_rocm_sdk_probe_validates_without_devel_root() -> Result<()> {
        let (root, _paths) = test_paths("runtime-only-probe");
        let site_packages = root.join("venv").join("Lib").join("site-packages");
        let core_root = site_packages.join("_rocm_sdk_core");
        let core_bin = core_root.join("bin");
        let libraries_root = site_packages.join("_rocm_sdk_libraries_gfx120X_all");
        let libraries_bin = libraries_root.join("bin");
        fs::create_dir_all(&core_bin)?;
        fs::create_dir_all(&libraries_bin)?;
        let amdhip = core_bin.join("amdhip64_7.dll");
        let hipblas = libraries_bin.join("hipblas.dll");
        fs::write(&amdhip, b"test")?;
        fs::write(&hipblas, b"test")?;
        let payload = serde_json::json!({
            "import_ok": true,
            "rocm_sdk_version": "7.13.0a20260416",
            "site_packages": site_packages,
            "root_path": core_root,
            "bin_path": core_bin,
            "cmake_path": null,
            "runtime_roots": [core_root, libraries_root],
            "bin_paths": [core_bin, libraries_bin],
            "library_paths": [core_bin, libraries_bin],
            "default_target_family": "gfx120X-all",
            "available_target_families": ["gfx120X-all"],
            "resolved_target_family": "gfx120X-all",
            "root_path_error": "ModuleNotFoundError: rocm_sdk_devel is not installed",
            "packages": [
                {"name": "rocm", "version": "7.13.0a20260416"},
                {"name": "rocm-sdk-core", "version": "7.13.0a20260416"},
                {"name": "rocm-sdk-libraries-gfx120X-all", "version": "7.13.0a20260416"}
            ],
            "library_shortnames": ["amdhip64", "hipblas"],
            "resolved_libraries": [
                {"shortname": "amdhip64", "paths": [amdhip]},
                {"shortname": "hipblas", "paths": [hipblas]}
            ],
            "error": null
        })
        .to_string();

        let probe = parse_rocm_sdk_probe(&payload)?;
        validate_rocm_sdk_runtime_probe(&probe)?;
        let _ = fs::remove_dir_all(root);

        assert!(probe.import_ok);
        assert_eq!(probe.runtime_roots.len(), 2);
        assert_eq!(probe.bin_paths.len(), 2);
        assert_eq!(probe.resolved_target_family.as_deref(), Some("gfx120X-all"));
        Ok(())
    }

    #[test]
    fn runtime_device_probe_without_kernel_error_field_still_parses() {
        // Output produced before the kernel probe existed must not become a parse
        // failure: an older runtime's probe is still a valid "no kernel attempted".
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":true,"torch_version":"2.11.0","hip_version":"7.13",
                "device_count":8,"error":null}"#,
        )
        .expect("probe without kernel_error should parse");

        assert_eq!(probe.device_count, Some(8));
        assert_eq!(probe.error, None);
        assert_eq!(probe.kernel_error, None);
    }

    #[test]
    fn runtime_device_probe_keeps_kernel_failures_out_of_the_enumeration_error() {
        // The distinction the caller acts on: devices were found, so this is not a
        // "no devices" runtime, but the GPU cannot run work.
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":true,"torch_version":"2.11.0","hip_version":"7.13",
                "device_count":8,"error":null,
                "kernel_error":"RuntimeError: HIP error: invalid device function"}"#,
        )
        .expect("probe with kernel_error should parse");

        assert_eq!(probe.device_count, Some(8));
        assert_eq!(probe.error, None);
        assert_eq!(
            probe.kernel_error.as_deref(),
            Some("RuntimeError: HIP error: invalid device function")
        );
    }

    #[test]
    fn runtime_device_probe_reports_import_failures_only_as_enumeration_errors() {
        let probe = parse_runtime_device_probe(
            r#"{"import_ok":false,"torch_version":null,"hip_version":null,
                "device_count":null,"error":"ImportError: no module named torch",
                "kernel_error":null}"#,
        )
        .expect("failed-import probe should parse");

        assert!(!probe.import_ok);
        assert_eq!(probe.device_count, None);
        assert_eq!(probe.kernel_error, None);
        assert_eq!(
            probe.error.as_deref(),
            Some("ImportError: no module named torch")
        );
    }

    #[test]
    fn runtime_device_probe_script_guards_the_kernel_behind_successful_enumeration() {
        // The script is the contract: a kernel must never be launched when the
        // import or enumeration already failed, or when there is no device to launch
        // it on. Getting this wrong turns a "no devices" runtime into a crash.
        assert!(
            RUNTIME_DEVICE_PROBE_SCRIPT
                .contains(r#"if out["error"] is None and (out["device_count"] or 0) > 0:"#),
            "the kernel attempt must be gated on a clean enumeration with devices"
        );

        let guard = RUNTIME_DEVICE_PROBE_SCRIPT
            .split_once(r#"if out["error"] is None"#)
            .expect("script should contain the kernel guard")
            .0;
        assert!(
            !guard.contains("device=\"cuda\"") && !guard.contains("synchronize"),
            "no kernel work may run before the guard"
        );
    }

    #[test]
    fn runtime_device_probe_script_records_kernel_failures_in_their_own_field() {
        let kernel_section = RUNTIME_DEVICE_PROBE_SCRIPT
            .split_once(r#"if out["error"] is None"#)
            .expect("script should contain the kernel guard")
            .1;

        // Allocate, mutate, and synchronize: an unusable GPU commonly survives the
        // allocation and only faults once work is actually launched and awaited.
        assert!(kernel_section.contains(r#"torch.ones(32, device="cuda")"#));
        assert!(kernel_section.contains("probe.add_(1.0)"));
        assert!(kernel_section.contains("torch.cuda.synchronize()"));

        assert!(
            kernel_section.contains(r#"out["kernel_error"] = type(exc).__name__"#),
            "a kernel failure must be recorded in kernel_error"
        );
        assert!(
            !kernel_section.contains(r#"out["error"] ="#),
            "the kernel attempt must never overwrite the enumeration error"
        );
        assert!(
            !kernel_section.contains(r#"out["device_count"] ="#),
            "a kernel failure must preserve the enumerated device count"
        );
    }
}
