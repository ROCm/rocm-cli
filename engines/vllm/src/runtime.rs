// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, split_local_version};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct VllmRuntime {
    pub runtime_id: String,
    pub env_id: String,
    pub command: PathBuf,
    pub python_executable: Option<PathBuf>,
    pub version: Option<String>,
    pub source: String,
    pub sdk_root: Option<PathBuf>,
    pub sdk_bin: Option<PathBuf>,
    pub sdk_bin_paths: Vec<PathBuf>,
    pub sdk_library_paths: Vec<PathBuf>,
    /// ROCm SDK version recorded in the runtime manifest, if known. Drives
    /// `vllm_install_route` and `apply_therock_env`'s ROCm 10.x discovery
    /// dispatch.
    pub rocm_sdk_version: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TheRockRuntimeManifest {
    #[serde(default)]
    pub runtime_key: Option<String>,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub python_executable: Option<PathBuf>,
    #[serde(default)]
    pub rocm_sdk: Option<RocmSdkRuntimeProbe>,
    /// The SDK's own version, used only to reconstruct a build identifier for
    /// manifests written before `sdk_torch` was recorded.
    #[serde(default)]
    pub version: Option<String>,
    /// The torch the SDK install wrote, e.g. `2.11.0+rocm7.13.0`.
    ///
    /// The CLI records it so a later engine install can be told apart from the SDK's
    /// own work. Read here for the same reason in reverse: it is the only way this
    /// engine can recognise a torch the CLI deliberately put back, as opposed to one
    /// some other installer left behind.
    #[serde(default)]
    pub sdk_torch: Option<String>,
    #[serde(default)]
    pub installed_at_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RocmSdkRuntimeProbe {
    #[serde(default)]
    pub import_ok: bool,
    #[serde(default)]
    pub root_path: Option<PathBuf>,
    #[serde(default)]
    pub bin_path: Option<PathBuf>,
    #[serde(default)]
    pub bin_paths: Vec<PathBuf>,
    #[serde(default)]
    pub library_paths: Vec<PathBuf>,
    #[serde(default)]
    pub rocm_sdk_version: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ManagedRuntimePython {
    pub runtime_id: String,
    pub python_executable: PathBuf,
    /// ROCm SDK version recorded for the specific manifest this Python came
    /// from. `runtime_id` is the GPU-family identifier shared by every
    /// installed version, so it cannot be used to look this back up: re-deriving
    /// it from `runtime_id` picks whichever version was installed most recently,
    /// not the one this Python executable actually belongs to.
    pub rocm_sdk_version: Option<String>,
}
/// The environment a repair must target: the one that was assessed, named by its own
/// runtime id rather than by the (possibly prefix-matching) id the caller requested.
pub(crate) fn assessed_python_for_repair(runtime: &VllmRuntime) -> Option<ManagedRuntimePython> {
    Some(ManagedRuntimePython {
        runtime_id: runtime.runtime_id.clone(),
        python_executable: runtime.python_executable.clone()?,
        rocm_sdk_version: runtime.rocm_sdk_version.clone(),
    })
}

pub(crate) fn runtime_is_managed(runtime: &VllmRuntime) -> bool {
    runtime.source.starts_with("managed_runtime_manifest")
}

pub(crate) fn vllm_runtime_warnings(runtime: &VllmRuntime) -> Vec<String> {
    let runtime_scope = if runtime_is_managed(runtime) {
        "rocm-cli records this vLLM command from a managed TheRock runtime; `rocm engines install vllm` can install vLLM into that runtime"
    } else {
        "rocm-cli records this as an external vLLM runtime; install/upgrade vLLM in that environment manually"
    };
    let mut warnings = vec![
        runtime_scope.to_owned(),
        "vLLM serving remains ROCm GPU required; no CPU fallback is used".to_owned(),
    ];
    if !cfg!(windows) && !rocm_core::openmpi::detect_openmpi().present {
        warnings.push(format!(
            "OpenMPI runtime (libmpi.so / libmpi_cxx.so / mpirun) was not found; vLLM requires it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::install_hint()
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libatomic_present() {
        warnings.push(format!(
            "libatomic runtime (libatomic.so.1) was not found; vLLM's torch wheel requires it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::libatomic_install_hint()
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libnuma_present() {
        warnings.push(format!(
            "system numactl runtime (libnuma.so.1 with libnuma_1.2) was not found; vLLM's torch wheel requires it and the ROCm SDK's bundled numa cannot satisfy it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::libnuma_install_hint()
        ));
    }
    warnings
}

pub(crate) fn resolve_vllm_runtime(runtime_id: Option<&str>) -> Result<VllmRuntime> {
    if cfg!(windows) {
        bail!("{}", crate::windows_unsupported_message());
    }

    if let Some(command) = std::env::var_os("ROCM_CLI_VLLM_COMMAND")
        .or_else(|| std::env::var_os("VLLM_COMMAND"))
        .map(PathBuf::from)
    {
        let command = crate::process::resolve_command_path(&command)?;
        return Ok(VllmRuntime {
            runtime_id: runtime_id.unwrap_or("external-vllm").to_owned(),
            env_id: "external-vllm-command".to_owned(),
            command,
            python_executable: None,
            version: None,
            source: "environment command".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        });
    }

    if let Some(python) = std::env::var_os("ROCM_CLI_VLLM_PYTHON")
        .or_else(|| std::env::var_os("VLLM_PYTHON"))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return runtime_from_python(ManagedRuntimeCandidate {
            runtime_id: runtime_id.unwrap_or("external-vllm-python").to_owned(),
            source: "environment python".to_owned(),
            python_executable: python,
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        });
    }

    if let Some(runtime) = resolve_managed_runtime(runtime_id)? {
        return Ok(runtime);
    }

    if let Some(command) = crate::process::find_command_on_path("vllm") {
        return Ok(VllmRuntime {
            runtime_id: runtime_id.unwrap_or("external-vllm-path").to_owned(),
            env_id: "external-vllm-path".to_owned(),
            command,
            python_executable: None,
            version: None,
            source: "PATH".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        });
    }

    let base = "vLLM is not installed in a Linux/WSL ROCm Python environment. Install/build vLLM against a ROCm-capable Python environment, then set ROCM_CLI_VLLM_COMMAND, set ROCM_CLI_VLLM_PYTHON, or install it into the active rocm-cli TheRock runtime. Native Windows is skipped; no CPU fallback is used.";
    match describe_skipped_managed_runtimes(runtime_id) {
        Some(skipped) => bail!("{base}\n\n{skipped}"),
        None => bail!("{base}"),
    }
}

fn runtime_from_python(candidate: ManagedRuntimeCandidate) -> Result<VllmRuntime> {
    let python = candidate.python_executable;
    let command = crate::process::vllm_command_from_python(&python)
        .with_context(|| format!("vLLM command not found beside {}", python.display()))?;
    let version = crate::process::probe_vllm_version(&python).ok().flatten();
    Ok(VllmRuntime {
        env_id: format!(
            "external-vllm-{}",
            stable_id_component(&candidate.runtime_id)
        ),
        runtime_id: candidate.runtime_id,
        command,
        python_executable: Some(python),
        version,
        source: candidate.source,
        sdk_root: candidate.sdk_root,
        sdk_bin: candidate.sdk_bin,
        sdk_bin_paths: candidate.sdk_bin_paths,
        sdk_library_paths: candidate.sdk_library_paths,
        rocm_sdk_version: candidate.rocm_sdk_version,
    })
}

fn resolve_managed_runtime(runtime_id: Option<&str>) -> Result<Option<VllmRuntime>> {
    let candidates = collect_managed_runtime_candidates(runtime_id)?;
    for candidate in candidates {
        if let Ok(runtime) = runtime_from_python(candidate) {
            return Ok(Some(runtime));
        }
    }
    Ok(None)
}

pub(crate) fn resolve_managed_runtime_python(
    runtime_id: Option<&str>,
) -> Result<Option<ManagedRuntimePython>> {
    let candidates = collect_managed_runtime_candidates(runtime_id)?;
    let Some(candidate) = candidates.into_iter().next() else {
        return Ok(None);
    };
    Ok(Some(ManagedRuntimePython {
        runtime_id: candidate.runtime_id,
        python_executable: candidate.python_executable,
        rocm_sdk_version: candidate.rocm_sdk_version,
    }))
}

#[derive(Debug, Clone)]
struct ManagedRuntimeCandidate {
    runtime_id: String,
    source: String,
    python_executable: PathBuf,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
    rocm_sdk_version: Option<String>,
}
/// The runtime manifests matching `runtime_id`, most recently installed first.
fn load_runtime_manifests(runtime_id: Option<&str>) -> Result<Vec<TheRockRuntimeManifest>> {
    let paths = AppPaths::discover()?;
    let registry = paths.data_dir.join("runtimes").join("registry");
    if !registry.is_dir() {
        return Ok(Vec::new());
    }
    let mut manifests = Vec::new();
    for entry in
        fs::read_dir(&registry).with_context(|| format!("failed to read {}", registry.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !runtime_matches(&manifest, runtime_id) {
            continue;
        }
        manifests.push((manifest.installed_at_unix_ms.unwrap_or(0), manifest));
    }
    manifests.sort_by_key(|(installed_at, _)| std::cmp::Reverse(*installed_at));
    Ok(manifests
        .into_iter()
        .map(|(_, manifest)| manifest)
        .collect())
}
/// The torch build the SDK installed into this runtime, as its manifest records it.
///
/// Read from the manifest, never from the environment. By the time this runs the
/// environment may already hold some other installer's build, and taking that for
/// the SDK's would conclude the runtime is correct and leave it wrong for good.
pub(crate) fn recorded_sdk_torch_build(runtime_id: &str) -> Option<String> {
    let manifest = load_runtime_manifests(Some(runtime_id))
        .ok()?
        .into_iter()
        .next()?;
    sdk_torch_build_from_manifest(&manifest)
}
/// The SDK's torch build identifier, with the fallback for older manifests.
///
/// Manifests written before `sdk_torch` was recorded still name the SDK version, and
/// TheRock builds that into the local segment as `rocm<version>`. Mirrors the CLI's
/// `sdk_torch_build_for_key` so both sides agree on what "the SDK's build" means.
pub(crate) fn sdk_torch_build_from_manifest(manifest: &TheRockRuntimeManifest) -> Option<String> {
    if let Some(recorded) = manifest.sdk_torch.as_deref()
        && let Some(build) = split_local_version(recorded).1
    {
        return Some(build.to_owned());
    }
    let version = manifest
        .rocm_sdk
        .as_ref()
        .and_then(|probe| probe.rocm_sdk_version.clone())
        .or_else(|| manifest.version.clone())?;
    (!version.trim().is_empty()).then(|| format!("rocm{version}"))
}
/// The bare ROCm SDK version a manifest records (e.g. `7.2.3`), unlike
/// [`sdk_torch_build_from_manifest`] which wraps it as a `rocm<version>` build tag.
fn rocm_sdk_version_from_manifest(manifest: &TheRockRuntimeManifest) -> Option<String> {
    let version = manifest
        .rocm_sdk
        .as_ref()
        .and_then(|probe| probe.rocm_sdk_version.clone())
        .or_else(|| manifest.version.clone())?;
    (!version.trim().is_empty()).then_some(version)
}
/// Registered runtimes that matched the request but were passed over because the
/// interpreter they record is not there, phrased for the end of an error message.
///
/// [`collect_managed_runtime_candidates`] drops those silently, which is right for
/// resolution — a runtime that cannot run is not a candidate — but leaves the
/// failure describing a registry that looks empty while `rocm runtimes list`
/// happily prints the entry. Naming the manifest and the interpreter turns
/// "nothing resolved" into something actionable.
///
/// Best-effort: this only ever decorates an error that is already being returned,
/// so any problem reading the registry yields no note rather than replacing the
/// original failure.
pub(crate) fn describe_skipped_managed_runtimes(runtime_id: Option<&str>) -> Option<String> {
    let paths = AppPaths::discover().ok()?;
    let registry = paths.data_dir.join("runtimes").join("registry");
    let mut skipped = Vec::new();
    for entry in fs::read_dir(&registry).ok()? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !runtime_matches(&manifest, runtime_id) {
            continue;
        }
        let Some(python) = manifest.python_executable.as_ref() else {
            continue;
        };
        if python.is_file() {
            continue;
        }
        let key = manifest.runtime_key.as_deref().unwrap_or("<unnamed>");
        skipped.push(format!("  {key}: {} is missing", python.display()));
    }
    if skipped.is_empty() {
        return None;
    }
    skipped.sort();
    Some(format!(
        "These registered runtimes were skipped because the Python interpreter they \
         record is not there:\n{}\nReinstall one with `rocm install sdk`, or drop it with \
         `rocm runtimes uninstall <runtime_key>`.",
        skipped.join("\n")
    ))
}

fn collect_managed_runtime_candidates(
    runtime_id: Option<&str>,
) -> Result<Vec<ManagedRuntimeCandidate>> {
    let mut candidates = Vec::new();
    for manifest in load_runtime_manifests(runtime_id)? {
        let Some(python) = manifest
            .python_executable
            .clone()
            .filter(|path| path.is_file())
        else {
            continue;
        };
        let runtime_id = manifest
            .runtime_id
            .as_deref()
            .unwrap_or("therock-vllm-runtime")
            .to_owned();
        let source = manifest.runtime_key.as_deref().map_or_else(
            || "managed_runtime_manifest".to_owned(),
            |key| format!("managed_runtime_manifest:{key}"),
        );
        let rocm_sdk_version = rocm_sdk_version_from_manifest(&manifest);
        let (sdk_root, sdk_bin, sdk_bin_paths, sdk_library_paths) = manifest
            .rocm_sdk
            .as_ref()
            .filter(|probe| probe.import_ok)
            .map_or((None, None, Vec::new(), Vec::new()), |probe| {
                (
                    probe.root_path.clone(),
                    probe.bin_path.clone(),
                    probe.bin_paths.clone(),
                    probe.library_paths.clone(),
                )
            });
        candidates.push(ManagedRuntimeCandidate {
            runtime_id,
            source,
            python_executable: python,
            sdk_root,
            sdk_bin,
            sdk_bin_paths,
            sdk_library_paths,
            rocm_sdk_version,
        });
    }
    Ok(candidates)
}

fn runtime_matches(manifest: &TheRockRuntimeManifest, requested: Option<&str>) -> bool {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let requested = requested.to_ascii_lowercase();
    if requested == "external" || requested == "external-vllm" {
        return false;
    }
    for candidate in [
        manifest.runtime_id.as_deref(),
        manifest.runtime_key.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        let candidate = candidate.to_ascii_lowercase();
        if candidate == requested || candidate.starts_with(&requested) {
            return true;
        }
    }
    false
}

fn stable_id_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_env_reflects_managed_runtime_manifest_source() {
        let mut runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\vllm.exe"
            } else {
                "/home/user/.venv/bin/vllm"
            }),
            python_executable: None,
            version: Some("test".to_owned()),
            source: "managed_runtime_manifest:vllm-source-pip-gfx120x-all".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };

        assert!(runtime_is_managed(&runtime));
        assert!(
            vllm_runtime_warnings(&runtime)
                .iter()
                .any(|warning| warning.contains("managed TheRock runtime"))
        );

        runtime.source = "environment command".to_owned();
        assert!(!runtime_is_managed(&runtime));
        assert!(
            vllm_runtime_warnings(&runtime)
                .iter()
                .any(|warning| warning.contains("external vLLM runtime"))
        );
    }
    #[test]
    fn a_repair_targets_the_environment_that_was_assessed() {
        // `resolve_managed_runtime_python` returns the first candidate unconditionally,
        // while the assessment walked on to the candidate that actually has vLLM. Both
        // match when `runtime_id` matching is by prefix, so the repair must follow the
        // assessed environment or it fixes an interpreter nobody found broken.
        let assessed = VllmRuntime {
            runtime_id: "nightly-wheel-gfx94x-dcgpu-7-14-0a20260611".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("/rocm/runtimes/wheel/nightly-gfx94x/bin/vllm"),
            python_executable: Some(PathBuf::from(
                "/rocm/runtimes/wheel/nightly-gfx94x/bin/python",
            )),
            version: Some("0.26.0".to_owned()),
            source: "managed_runtime_manifest:nightly-wheel-gfx94x-dcgpu".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };

        let target = assessed_python_for_repair(&assessed).expect("a managed runtime has a python");

        assert_eq!(
            target.python_executable,
            PathBuf::from("/rocm/runtimes/wheel/nightly-gfx94x/bin/python")
        );
        assert_eq!(
            target.runtime_id, "nightly-wheel-gfx94x-dcgpu-7-14-0a20260611",
            "the assessed runtime's own id, not the prefix the caller asked for"
        );
    }
    #[test]
    fn a_recorded_sdk_torch_names_the_build() {
        let manifest = TheRockRuntimeManifest {
            sdk_torch: Some("2.11.0+rocm7.13.0".to_owned()),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0")
        );
    }
    #[test]
    fn a_manifest_without_sdk_torch_reconstructs_the_build_from_the_sdk_version() {
        // Written before `sdk_torch` was recorded. These are the runtimes already on
        // real machines, so the fallback is what repairs them rather than a nicety.
        let manifest = TheRockRuntimeManifest {
            rocm_sdk: Some(RocmSdkRuntimeProbe {
                rocm_sdk_version: Some("7.13.0".to_owned()),
                ..RocmSdkRuntimeProbe::default()
            }),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0")
        );
    }
    #[test]
    fn a_manifest_that_identifies_no_sdk_build_says_so() {
        assert_eq!(
            sdk_torch_build_from_manifest(&TheRockRuntimeManifest::default()),
            None
        );
    }
    #[test]
    fn a_recorded_rocm_sdk_probe_names_the_bare_version() {
        let manifest = TheRockRuntimeManifest {
            rocm_sdk: Some(RocmSdkRuntimeProbe {
                rocm_sdk_version: Some("7.2.3".to_owned()),
                ..RocmSdkRuntimeProbe::default()
            }),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            rocm_sdk_version_from_manifest(&manifest).as_deref(),
            Some("7.2.3")
        );
    }
    #[test]
    fn a_manifest_without_an_sdk_probe_falls_back_to_its_own_version() {
        let manifest = TheRockRuntimeManifest {
            version: Some("7.13.0".to_owned()),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            rocm_sdk_version_from_manifest(&manifest).as_deref(),
            Some("7.13.0")
        );
    }
    #[test]
    fn a_manifest_that_identifies_no_rocm_sdk_version_says_so() {
        assert_eq!(
            rocm_sdk_version_from_manifest(&TheRockRuntimeManifest::default()),
            None
        );
    }
}
