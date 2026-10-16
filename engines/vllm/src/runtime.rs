// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, split_local_version};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub(crate) struct VllmRuntime {
    pub(crate) runtime_id: String,
    pub(crate) env_id: String,
    pub(crate) command: PathBuf,
    pub(crate) python_executable: Option<PathBuf>,
    pub(crate) version: Option<String>,
    pub(crate) source: String,
    pub(crate) sdk_root: Option<PathBuf>,
    pub(crate) sdk_bin: Option<PathBuf>,
    pub(crate) sdk_bin_paths: Vec<PathBuf>,
    pub(crate) sdk_library_paths: Vec<PathBuf>,
    /// ROCm SDK version recorded in the runtime manifest, if known. Drives
    /// [`crate::install::vllm_install_route`] and
    /// [`crate::process::apply_therock_env`]'s ROCm 10.x discovery dispatch.
    pub(crate) rocm_sdk_version: Option<String>,
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

/// The ROCm SDK 10.0 release line, as it appears in both
/// [`ROCM_DISCOVER_BUILD_VERSIONS`] and `install.rs`'s discover-build table.
pub(crate) const ROCM_SDK_10_0: &str = "10.0.0";
/// The ROCm SDK 10.1 release line, as it appears in both
/// [`ROCM_DISCOVER_BUILD_VERSIONS`] and `install.rs`'s discover-build table.
pub(crate) const ROCM_SDK_10_1: &str = "10.1.0";

/// ROCm SDK versions keyed by a row in `install.rs`'s discover-build table
/// (see [`crate::install::vllm_rocm_discover_build`] for the recipe each one
/// resolves to: wheel indexes, version prefixes, interpreter tag). Defined
/// here, in the lower layer, as the single source for which versions count —
/// `install.rs`'s table literally uses these constants for its
/// `rocm_sdk_version` fields, and [`runtime_has_discover_build`] below
/// matches against the same two strings. This list and that table are still
/// two separate pieces of data, kept in agreement only by
/// `install.rs`'s `vllm_rocm_discover_build_table_matches_the_discover_version_list`
/// test, not by construction — a new table row needs a matching entry here,
/// and that test is what catches it if one is missed.
pub(crate) const ROCM_DISCOVER_BUILD_VERSIONS: &[&str] = &[ROCM_SDK_10_0, ROCM_SDK_10_1];

/// Whether `recorded` and `table_key` name the same ROCm `major.minor` line,
/// ignoring patch and any dev/pre-release suffix. See
/// [`crate::install::vllm_rocm_discover_build`] for why the line, not the
/// exact release, is what a discover row covers. Lives here, not in
/// `install.rs`, alongside [`ROCM_DISCOVER_BUILD_VERSIONS`] so
/// [`runtime_has_discover_build`] can match without calling into `install`.
pub(crate) fn rocm_sdk_series_matches(recorded: &str, table_key: &str) -> bool {
    fn series(version: &str) -> Option<(u64, u64)> {
        let mut parts = version.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor: String = parts
            .next()?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        Some((major, minor.parse().ok()?))
    }
    series(recorded).is_some_and(|version| Some(version) == series(table_key))
}

/// Whether `runtime`'s recorded ROCm SDK version has a live discover-build
/// recipe. Exposed as runtime metadata, matched directly against
/// [`ROCM_DISCOVER_BUILD_VERSIONS`], so callers outside `install` — the vLLM
/// launch path in `process.rs`, specifically — don't need to reach into
/// `install`'s build-selection table for a plain yes/no check.
pub(crate) fn runtime_has_discover_build(runtime: &VllmRuntime) -> bool {
    runtime.rocm_sdk_version.as_deref().is_some_and(|version| {
        ROCM_DISCOVER_BUILD_VERSIONS
            .iter()
            .any(|key| rocm_sdk_series_matches(version, key))
    })
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
            "OpenMPI runtime (libmpi.so / libmpi_cxx.so / mpirun) was not found; vLLM requires it. {}.",
            rocm_core::openmpi::install_hint()
                .or("rerun `rocm engines install vllm --yes`")
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libatomic_present() {
        warnings.push(format!(
            "libatomic runtime (libatomic.so.1) was not found; vLLM's torch wheel requires it. {}.",
            rocm_core::openmpi::libatomic_install_hint()
                .or("rerun `rocm engines install vllm --yes`")
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libnuma_present() {
        warnings.push(format!(
            "system numactl runtime (libnuma.so.1 with libnuma_1.2) was not found; vLLM's torch wheel requires it and the ROCm SDK's bundled numa cannot satisfy it. {}.",
            rocm_core::openmpi::libnuma_install_hint()
                .or("rerun `rocm engines install vllm --yes`")
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
        let command = resolve_command_path(&command)?;
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

    if let Some(command) = find_command_on_path("vllm") {
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
    let command = vllm_command_from_python(&python)
        .with_context(|| format!("vLLM command not found beside {}", python.display()))?;
    let version = probe_vllm_version(&python).ok().flatten();
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
pub(crate) fn rocm_sdk_version_from_manifest(manifest: &TheRockRuntimeManifest) -> Option<String> {
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

fn vllm_command_from_python(python: &Path) -> Option<PathBuf> {
    let dir = python.parent()?;
    candidate_command_names("vllm")
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn find_command_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in candidate_command_names(name) {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn resolve_command_path(command: &Path) -> Result<PathBuf> {
    if command.components().count() > 1 || command.is_absolute() {
        if command.is_file() {
            return Ok(command.to_path_buf());
        }
        bail!(
            "configured vLLM command is not a file: {}",
            command.display()
        );
    }
    find_command_on_path(&command.display().to_string()).with_context(|| {
        format!(
            "configured vLLM command `{}` was not found on PATH",
            command.display()
        )
    })
}

fn candidate_command_names(name: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![
            format!("{name}.exe"),
            format!("{name}.cmd"),
            name.to_owned(),
        ]
    } else {
        vec![name.to_owned()]
    }
}

fn probe_vllm_version(python: &Path) -> Result<Option<String>> {
    let script = r#"import importlib.metadata, importlib.util, json
spec = importlib.util.find_spec("vllm")
version = None
if spec is not None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
print(json.dumps({"present": spec is not None, "version": version}))
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .with_context(|| format!("failed to probe vLLM with {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "vLLM probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout).context("invalid vLLM probe JSON")?;
    if value
        .get("present")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        Ok(value
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_owned))
    } else {
        bail!("Python environment does not contain the vLLM package")
    }
}

/// Describes a resolved runtime's TheRock SDK layout for the running-state
/// payload, or `None` when the runtime is not backed by a TheRock SDK root.
/// Performs no filesystem writes: `therock_library_path_entries` only derives
/// paths here, it never writes the compat shims a launch needs those paths to
/// contain — see [`ensure_therock_compat_shims`].
pub(crate) fn therock_runtime_env_payload(runtime: &VllmRuntime) -> Option<Value> {
    let root = runtime.sdk_root.as_ref()?;
    Some(json!({
        "runtime_id": runtime.runtime_id,
        "env_id": runtime.env_id,
        "root": root.display().to_string(),
        "bin": runtime.sdk_bin.as_ref().map(|path| path.display().to_string()),
        "bin_paths": runtime_bin_paths(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "library_paths": therock_library_path_entries(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "source": runtime.source,
    }))
}

pub(crate) fn runtime_bin_paths(runtime: &VllmRuntime) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if let Some(bin) = runtime.sdk_bin.as_ref() {
        entries.push(bin.clone());
    }
    entries.extend(runtime.sdk_bin_paths.iter().cloned());
    dedupe_paths(entries)
}

/// How [`therock_library_path_entries_inner`] decides whether the compat dir
/// belongs on the loader path.
enum CompatDirSignal {
    /// `ensure_therock_compat_shims`'s answer is not passed in to this
    /// derivation (the running-state payload path, which never calls it).
    /// Falls back to a read-only directory listing of whatever is already on
    /// disk: not authoritative, but the best available signal without
    /// performing a write.
    Unchecked,
    /// This launch's authoritative answer from `ensure_therock_compat_shims`:
    /// on the launch path, `Some(dir)` when a shim was needed and written,
    /// `None` when it was not needed or could not be written — even if one is
    /// still sitting on disk
    /// from an earlier run.
    Checked(Option<PathBuf>),
}

pub(crate) fn therock_library_path_entries(runtime: &VllmRuntime) -> Vec<PathBuf> {
    therock_library_path_entries_inner(runtime, CompatDirSignal::Unchecked)
}

/// Same path derivation as [`therock_library_path_entries`], for the launch
/// path: `compat_shim` is the `Option<PathBuf>` [`ensure_therock_compat_shims`]
/// returned for *this launch*'s `ensure_mpi_cxx_compat` call, and is used as
/// the authoritative answer for whether the compat dir belongs on the loader
/// path. Unlike a directory listing, it is not fooled by a stub some earlier
/// run left behind after the host later gained a real `libmpi_cxx` — that
/// case resolves to `None` here, exactly as `ensure_mpi_cxx_compat` reports
/// it for this launch.
pub(crate) fn therock_library_path_entries_for_launch(
    runtime: &VllmRuntime,
    compat_shim: Option<PathBuf>,
) -> Vec<PathBuf> {
    therock_library_path_entries_inner(runtime, CompatDirSignal::Checked(compat_shim))
}

fn therock_library_path_entries_inner(
    runtime: &VllmRuntime,
    compat_dir_signal: CompatDirSignal,
) -> Vec<PathBuf> {
    let Some(root) = runtime.sdk_root.as_ref() else {
        return dedupe_paths(runtime.sdk_library_paths.clone());
    };
    let mut entries = runtime.sdk_library_paths.clone();
    entries.extend([
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ]);
    if cfg!(target_os = "linux") {
        let wsl_dxcore_lib = PathBuf::from("/usr/lib/wsl/lib");
        if wsl_dxcore_lib.is_dir() {
            entries.push(wsl_dxcore_lib);
        }
        // OpenMPI is installed outside the default loader path on some distros
        // (notably RHEL-family under /usr/lib64/openmpi/lib); make sure vLLM can
        // load libmpi.so at launch when it lives there.
        entries.extend(rocm_core::openmpi::openmpi_library_dirs());

        // The compat shims this directory may hold (see
        // `ensure_therock_compat_shims`) are not ensured here — this function
        // only derives the path, never writes. On the launch path
        // (`Checked`), include the candidate exactly when
        // `ensure_therock_compat_shims` says a shim is needed this launch —
        // this is what keeps a leftover stub from an earlier run (the host
        // has since gained a real `libmpi_cxx.so*`) from shadowing the loader
        // path. On the running-state payload path (`Unchecked`), which never
        // calls that function, fall back to a read-only directory listing of
        // whatever is already there: not authoritative, but the best
        // available signal without performing a write.
        match compat_dir_signal {
            CompatDirSignal::Checked(shim) => entries.extend(shim),
            CompatDirSignal::Unchecked => {
                if let Some(compat_dir) = runtime_compat_dir(runtime)
                    && compat_dir_has_shims(&compat_dir)
                {
                    entries.push(compat_dir);
                }
            }
        }
    }
    dedupe_paths(entries)
}

/// Whether `compat_dir` currently has any entry at all — a plain directory
/// listing, not a check for a specific shim [`ensure_therock_compat_shims`]
/// would recognize. This is the fallback [`therock_library_path_entries`]
/// uses on the running-state payload path, which never calls that function
/// and so has no authoritative answer to consult; see
/// [`therock_library_path_entries_for_launch`] for the path that does.
fn compat_dir_has_shims(compat_dir: &Path) -> bool {
    compat_dir
        .read_dir()
        .is_ok_and(|mut entries| entries.next().is_some())
}
/// Materializes the runtime-owned OpenMPI compatibility shims that
/// `therock_library_path_entries`' `compat_dir` entry depends on. Call once
/// from the launch path, before building the environment a server is spawned
/// with — never from running-state payload assembly, which only needs the
/// path list `therock_library_path_entries` derives, not the shims
/// themselves. Returns the `libmpi_cxx` compat dir when a shim is needed for
/// *this* launch and `ensure_mpi_cxx_compat` wrote it — pass this on to
/// [`therock_library_path_entries_for_launch`] rather than having that
/// derivation re-check the directory itself. Returns `None` without touching
/// the filesystem when `runtime.sdk_root` is absent, so this function stays
/// safe to call independent of a caller's own gating.
///
/// `ensure_mpi_cxx_compat` is the real probe in production
/// ([`rocm_core::openmpi::ensure_mpi_cxx_compat`]); tests inject a fixed
/// answer here to exercise the rest of this hand-off without depending on
/// this host's own OpenMPI/compiler state.
pub(crate) fn ensure_therock_compat_shims(
    runtime: &VllmRuntime,
    ensure_mpi_cxx_compat: impl FnOnce(&Path) -> Option<PathBuf>,
) -> Option<PathBuf> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    // No-op without `sdk_root`, matching `therock_library_path_entries_inner`'s
    // own gating.
    runtime.sdk_root.as_ref()?;
    let compat_dir = runtime_compat_dir(runtime)?;
    // PyTorch's `libtorch_global_deps.so` lists `libmpi_cxx.so.40` as a
    // NEEDED dependency, but OpenMPI 5.x removed the legacy C++ bindings, so
    // `import torch` aborts with `libmpi_cxx.so.40: cannot open shared object
    // file`. When no real `libmpi_cxx.so*` exists, materialize an embedded
    // `libmpi_cxx.so.40` stub (built at compile time, see rocm-core's
    // build.rs) into this directory. The stub only *defines* the legacy C++
    // binding symbols torch needs; they are never called in single-node
    // serving, so this is safe.
    let mpi_cxx_shim = ensure_mpi_cxx_compat(&compat_dir);
    // PyTorch's `libc10.so` NEEDS the standard `libnuma.so.1` soname with the
    // upstream `libnuma_1.2` symbol version. TheRock bundles numa only under
    // the renamed soname `librocm_sysdeps_numa.so.1` whose versions are
    // rewritten to `AMDROCM_SYSDEPS_1.0_libnuma_*`, which cannot satisfy that
    // binding. An older rocm-cli release symlinked `libnuma.so.1` to that
    // bundled library; on the loader path it shadowed any real system libnuma
    // and broke `import torch` with `version 'libnuma_1.2' not found`. Remove
    // that stale shim here so the real numactl runtime (installed via the
    // package manager) wins; `libnuma_present()`/`spawn_vllm_server` handle
    // the install/preflight.
    remove_stale_numa_shim(&compat_dir);
    mpi_cxx_shim
}
/// Remove a stale `libnuma.so.1` compatibility symlink left by older rocm-cli
/// versions in `compat_dir`. That shim pointed at the ROCm SDK's bundled
/// `librocm_sysdeps_numa.so.1`, whose renamed symbol versions cannot satisfy the
/// `libnuma_1.2` symbol PyTorch's `libc10.so` binds; leaving it on the loader
/// path would shadow a correctly installed system libnuma. No-op when absent or
/// when the entry is not a symlink.
fn remove_stale_numa_shim(compat_dir: &Path) {
    let link = compat_dir.join("libnuma.so.1");
    if let Ok(meta) = link.symlink_metadata()
        && meta.file_type().is_symlink()
    {
        let _ = fs::remove_file(&link);
    }
}
/// A writable, runtime-owned directory for managed-runtime library compatibility
/// shims (see [`therock_library_path_entries`]). Prefers the managed Python
/// environment root (`<env>/bin/python` -> `<env>`); falls back to the SDK root
/// when no Python launcher is recorded.
fn runtime_compat_dir(runtime: &VllmRuntime) -> Option<PathBuf> {
    const COMPAT_DIR_NAME: &str = "rocm-cli-lib-compat";
    if let Some(env_root) = runtime
        .python_executable
        .as_ref()
        .and_then(|python| python.parent())
        .and_then(|bin| bin.parent())
    {
        return Some(env_root.join(COMPAT_DIR_NAME));
    }
    runtime
        .sdk_root
        .as_ref()
        .map(|root| root.join(COMPAT_DIR_NAME))
}

fn dedupe_paths(entries: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for entry in entries {
        if !entry.as_os_str().is_empty() && !deduped.iter().any(|seen| seen == &entry) {
            deduped.push(entry);
        }
    }
    deduped
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
    fn runtime_with_sdk_version(version: Option<&str>) -> VllmRuntime {
        VllmRuntime {
            runtime_id: "nightly-wheel-gfx94x-dcgpu".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("/rocm/runtimes/wheel/nightly-gfx94x/bin/vllm"),
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:nightly-wheel-gfx94x-dcgpu".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: version.map(ToOwned::to_owned),
        }
    }
    #[test]
    fn runtime_has_discover_build_matches_both_discover_lines() {
        assert!(runtime_has_discover_build(&runtime_with_sdk_version(Some(
            "10.0.0"
        ))));
        assert!(runtime_has_discover_build(&runtime_with_sdk_version(Some(
            "10.1.0a20260611"
        ))));
    }
    #[test]
    fn runtime_has_discover_build_rejects_a_line_with_no_discover_row() {
        assert!(!runtime_has_discover_build(&runtime_with_sdk_version(
            Some("10.2.0")
        )));
        assert!(!runtime_has_discover_build(&runtime_with_sdk_version(
            Some("7.2.3")
        )));
    }
    #[test]
    fn runtime_has_discover_build_rejects_no_recorded_version() {
        assert!(!runtime_has_discover_build(&runtime_with_sdk_version(None)));
    }
    #[test]
    fn therock_library_path_entries_include_sysdeps_for_hip_apps() {
        // A fixed path is safe here: this function only derives candidate
        // paths and never writes to disk (see `ensure_therock_compat_shims`
        // for the step that does).
        let root = PathBuf::from("/tmp/rocm-sdk");
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(root.clone()),
            sdk_bin: Some(root.join("bin")),
            sdk_bin_paths: vec![root.join("runtime").join("bin")],
            sdk_library_paths: vec![root.join("runtime").join("lib")],
            rocm_sdk_version: None,
        };
        let entries = therock_library_path_entries(&runtime);
        assert!(entries.contains(&root.join("runtime").join("lib")));
        assert!(entries.contains(&root.join("lib")));
        assert!(
            entries
                .iter()
                .any(|entry| entry.ends_with(Path::new("lib").join("rocm_sysdeps").join("lib")))
        );
    }

    #[test]
    fn compat_dir_has_shims_reflects_directory_contents() {
        let empty = tempfile::tempdir().unwrap();
        assert!(!compat_dir_has_shims(empty.path()));

        let populated = tempfile::tempdir().unwrap();
        fs::write(populated.path().join("libmpi_cxx.so.40"), b"stub").unwrap();
        assert!(compat_dir_has_shims(populated.path()));

        let missing = empty.path().join("never-created");
        assert!(!compat_dir_has_shims(&missing));
    }

    #[cfg(target_os = "linux")]
    fn runtime_with_compat_dir(env_root: &Path) -> VllmRuntime {
        VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: Some(env_root.join("bin").join("python")),
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(env_root.join("sdk")),
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn therock_library_path_entries_omits_compat_dir_with_nothing_in_it() {
        let env = tempfile::tempdir().unwrap();
        let runtime = runtime_with_compat_dir(env.path());

        let entries = therock_library_path_entries(&runtime);

        assert!(
            !entries.contains(&env.path().join("rocm-cli-lib-compat")),
            "an absent/empty compat dir must not shadow the loader path: {entries:?}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn therock_library_path_entries_includes_compat_dir_once_a_shim_exists() {
        let env = tempfile::tempdir().unwrap();
        let runtime = runtime_with_compat_dir(env.path());
        let compat_dir = env.path().join("rocm-cli-lib-compat");
        fs::create_dir_all(&compat_dir).unwrap();
        fs::write(compat_dir.join("libmpi_cxx.so.40"), b"stub").unwrap();

        let entries = therock_library_path_entries(&runtime);

        assert!(
            entries.contains(&compat_dir),
            "a populated compat dir must be on the loader path: {entries:?}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn therock_library_path_entries_for_launch_ignores_a_leftover_stub_this_launch_does_not_need() {
        // A leftover `libmpi_cxx.so.40` from an earlier run, now that the host
        // has gained a real `libmpi_cxx` (`ensure_mpi_cxx_compat` reports
        // `None` for this launch), must not shadow the loader path — unlike
        // `therock_library_path_entries`'s directory-listing fallback, which
        // cannot tell a leftover stub from one still needed.
        let env = tempfile::tempdir().unwrap();
        let runtime = runtime_with_compat_dir(env.path());
        let compat_dir = env.path().join("rocm-cli-lib-compat");
        fs::create_dir_all(&compat_dir).unwrap();
        fs::write(compat_dir.join("libmpi_cxx.so.40"), b"stale stub").unwrap();

        let entries = therock_library_path_entries_for_launch(&runtime, None);

        assert!(
            !entries.contains(&compat_dir),
            "a stub this launch's ensure call says is not needed must not be \
             included just because one is still on disk: {entries:?}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn therock_library_path_entries_for_launch_includes_the_shim_ensure_reported() {
        let env = tempfile::tempdir().unwrap();
        let runtime = runtime_with_compat_dir(env.path());
        let compat_dir = env.path().join("rocm-cli-lib-compat");

        let entries = therock_library_path_entries_for_launch(&runtime, Some(compat_dir.clone()));

        assert!(
            entries.contains(&compat_dir),
            "the dir `ensure_therock_compat_shims` reports needed must be on \
             the loader path, even before anything is listed on disk: {entries:?}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn ensure_therock_compat_shims_writes_nothing_without_an_sdk_root() {
        // `python_executable` alone (no `sdk_root`) is the real shape of the
        // `ROCM_CLI_VLLM_PYTHON`/`VLLM_PYTHON` external-python runtime that
        // `resolve_vllm_runtime` builds. Exercises this `pub(crate)`
        // function's own `sdk_root` guard directly, independent of whether
        // any caller reaches it with that runtime shape.
        //
        // Whether a write would otherwise happen on this host (a stub
        // embedded at compile time, and no real `libmpi_cxx.so*` already
        // present) varies by host, so probe `ensure_mpi_cxx_compat` against a
        // throwaway directory first: when it would do nothing anyway, this
        // test cannot distinguish "guarded correctly" from "nothing to guard
        // against", so skip the directory-emptiness assertion in that case —
        // the `shim.is_none()` assertion below still holds either way.
        let probe = tempfile::tempdir().unwrap();
        let would_write_without_the_guard =
            rocm_core::openmpi::ensure_mpi_cxx_compat(probe.path()).is_some();

        let env = tempfile::tempdir().unwrap();
        let runtime = VllmRuntime {
            runtime_id: "external-vllm-python".to_owned(),
            env_id: "external-vllm-python".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: Some(env.path().join("bin").join("python")),
            version: None,
            source: "environment python".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };

        let shim = ensure_therock_compat_shims(&runtime, rocm_core::openmpi::ensure_mpi_cxx_compat);

        assert!(
            shim.is_none(),
            "no sdk_root means the compat dir is never on the loader path, \
             so there is nothing to report needing a shim for"
        );
        if would_write_without_the_guard {
            assert!(
                !env.path().join("rocm-cli-lib-compat").exists(),
                "must not write into the venv root's compat dir when sdk_root is absent"
            );
        }
    }
}
