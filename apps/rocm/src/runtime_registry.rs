// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The installed-runtime registry: the manifest model every runtime install
//! writes, where those manifests live on disk, the loaders that read them back,
//! and the registry selection and activation validation built on them (picking
//! the active or a named runtime, and checking that a manifest is usable).
//!
//! This is a crate-level module that depends on neither `therock` (which writes
//! manifests when it installs a runtime) nor `runtimes_cmd` (which lists and
//! activates them); both build on it. `runtimes_cmd` re-exports the selection
//! and path-helper names that other modules import from it, but it still
//! reaches the manifest model (the manifest and probe types and the loaders)
//! through `therock`'s re-exports, along with a few `therock` helpers. So
//! `runtimes_cmd` depends on `therock`, while `therock` no longer depends on
//! `runtimes_cmd` in production code. The serde shape of
//! [`InstalledRuntimeManifest`] and its probe structs is the on-disk contract of
//! the registry files and of each install's `.rocm-cli-runtime.json`.

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, normalize_runtime_path_for_host, normalize_runtime_path_for_storage,
    normalize_runtime_path_text_for_host, normalize_runtime_path_text_for_storage,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Exact canonical wheel install intent last applied successfully to a runtime.
///
/// Version alone cannot identify a reusable environment: adding a required ROCm
/// extra at the same release version must make an older cache repairable.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct WheelRuntimeComposition {
    pub source_layout_generation: String,
    pub package_specs: Vec<String>,
    /// Exact target supplied to `rocm_sdk` when resolving runtime libraries.
    #[serde(default)]
    pub rocm_sdk_target: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InstalledRuntimeManifest {
    pub runtime_key: String,
    pub runtime_id: String,
    pub channel: String,
    pub format: String,
    pub family: String,
    pub family_source: String,
    pub version: String,
    pub install_root: PathBuf,
    pub selected_artifact_url: String,
    /// Which published layout this runtime's artifact came from.
    ///
    /// Provenance, not identity: an update reads it so a runtime resolves the
    /// stream it was installed from instead of whichever stream the channel
    /// happens to default to. Absent on every manifest written before the
    /// layout existed, which reads back as the canonical stream those runtimes
    /// came from.
    #[serde(default)]
    pub source_layout_generation: Option<String>,
    #[serde(default)]
    pub index_url: Option<String>,
    #[serde(default)]
    pub tarball_file_name: Option<String>,
    #[serde(default)]
    pub python_launcher: Option<String>,
    #[serde(default)]
    pub python_executable: Option<String>,
    #[serde(default)]
    pub pip_cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub rocm_sdk: Option<RocmSdkPythonProbe>,
    /// The torch this SDK install wrote, e.g. `2.11.0+rocm7.13.0`.
    ///
    /// Recorded because an engine install later overwrites torch in the same
    /// environment. Reading the environment afterwards tells you what is there
    /// now, not which build belongs to these libraries; only the manifest still
    /// knows that, and it has to survive repeat installs to be worth anything.
    #[serde(default)]
    pub sdk_torch: Option<String>,
    /// Missing on manifests written before composition-aware freshness. Such a
    /// managed wheel runtime is repaired once and rewritten with this field.
    #[serde(default)]
    pub wheel_composition: Option<WheelRuntimeComposition>,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub imported_from: Option<PathBuf>,
    /// Whether the compiler toolchain (`devel`) is part of this install, so an
    /// update reinstalls the same thing the user originally chose.
    #[serde(default = "devel_default_for_legacy_manifest")]
    pub devel: bool,
    pub installed_at_unix_ms: u128,
}

/// Manifests written before `devel` became opt-in always included the
/// toolchain, so a missing field means it is present. Defaulting to `false`
/// here would silently strip it on the next update.
const fn devel_default_for_legacy_manifest() -> bool {
    true
}

impl InstalledRuntimeManifest {
    /// Whether this runtime has the compiler toolchain.
    ///
    /// Prefers the recorded `wheel_composition`, because those specs are what
    /// `uv` was actually given — so the answer cannot drift from the install
    /// the way a separately-written flag can. `devel` is the fallback for
    /// tarball installs and for manifests written before compositions were
    /// recorded; on a manifest older still it defaults to `true`, since every
    /// install predating the flag shipped the toolchain.
    pub(crate) fn includes_devel(&self) -> bool {
        wheel_composition_includes_devel(self.wheel_composition.as_ref()).unwrap_or(self.devel)
    }

    fn normalize_host_paths(mut self) -> Self {
        self.install_root = normalize_manifest_path(self.install_root);
        self.python_launcher = self
            .python_launcher
            .map(|value| normalize_runtime_path_text_for_host(&value));
        self.python_executable = self
            .python_executable
            .map(|value| normalize_runtime_path_text_for_host(&value));
        self.pip_cache_dir = self.pip_cache_dir.map(normalize_manifest_path);
        self.imported_from = self.imported_from.map(normalize_manifest_path);
        if let Some(probe) = self.rocm_sdk.as_mut() {
            probe.normalize_host_paths();
        }
        self
    }

    pub(crate) fn normalize_storage_paths(mut self) -> Self {
        self.install_root = normalize_storage_manifest_path(&self.install_root);
        self.python_launcher = self
            .python_launcher
            .map(|value| normalize_runtime_path_text_for_storage(&value));
        self.python_executable = self
            .python_executable
            .map(|value| normalize_runtime_path_text_for_storage(&value));
        self.pip_cache_dir = self
            .pip_cache_dir
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.imported_from = self
            .imported_from
            .as_deref()
            .map(normalize_storage_manifest_path);
        if let Some(probe) = self.rocm_sdk.as_mut() {
            probe.normalize_storage_paths();
        }
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RocmSdkPythonProbe {
    #[serde(default)]
    pub import_ok: bool,
    #[serde(default)]
    pub rocm_sdk_version: Option<String>,
    #[serde(default)]
    pub site_packages: Option<PathBuf>,
    #[serde(default)]
    pub root_path: Option<PathBuf>,
    #[serde(default)]
    pub bin_path: Option<PathBuf>,
    #[serde(default)]
    pub cmake_path: Option<PathBuf>,
    #[serde(default)]
    pub runtime_roots: Vec<PathBuf>,
    #[serde(default)]
    pub bin_paths: Vec<PathBuf>,
    #[serde(default)]
    pub library_paths: Vec<PathBuf>,
    #[serde(default)]
    pub default_target_family: Option<String>,
    #[serde(default)]
    pub available_target_families: Vec<String>,
    #[serde(default)]
    pub resolved_target_family: Option<String>,
    #[serde(default)]
    pub packages: Vec<RocmSdkPackageProbe>,
    #[serde(default)]
    pub library_shortnames: Vec<String>,
    #[serde(default)]
    pub resolved_libraries: Vec<RocmSdkLibraryProbe>,
    #[serde(default)]
    pub error: Option<String>,
}

impl RocmSdkPythonProbe {
    fn normalize_host_paths(&mut self) {
        self.site_packages = self.site_packages.take().map(normalize_manifest_path);
        self.root_path = self.root_path.take().map(normalize_manifest_path);
        self.bin_path = self.bin_path.take().map(normalize_manifest_path);
        self.cmake_path = self.cmake_path.take().map(normalize_manifest_path);
        self.runtime_roots = std::mem::take(&mut self.runtime_roots)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        self.bin_paths = std::mem::take(&mut self.bin_paths)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        self.library_paths = std::mem::take(&mut self.library_paths)
            .into_iter()
            .map(normalize_manifest_path)
            .collect();
        for library in &mut self.resolved_libraries {
            library.paths = std::mem::take(&mut library.paths)
                .into_iter()
                .map(normalize_manifest_path)
                .collect();
        }
    }

    fn normalize_storage_paths(&mut self) {
        self.site_packages = self
            .site_packages
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.root_path = self
            .root_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.bin_path = self
            .bin_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.cmake_path = self
            .cmake_path
            .as_deref()
            .map(normalize_storage_manifest_path);
        self.runtime_roots = self
            .runtime_roots
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        self.bin_paths = self
            .bin_paths
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        self.library_paths = self
            .library_paths
            .iter()
            .map(|path| normalize_storage_manifest_path(path))
            .collect();
        for library in &mut self.resolved_libraries {
            library.paths = library
                .paths
                .iter()
                .map(|path| normalize_storage_manifest_path(path))
                .collect();
        }
    }
}

fn normalize_manifest_path(path: PathBuf) -> PathBuf {
    normalize_runtime_path_for_host(&path)
}

fn normalize_storage_manifest_path(path: &Path) -> PathBuf {
    normalize_runtime_path_for_storage(path)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RocmSdkPackageProbe {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RocmSdkLibraryProbe {
    pub shortname: String,
    pub paths: Vec<PathBuf>,
}

/// Whether a recorded composition installed the compiler toolchain, read back
/// out of its `rocm[...]` extras.
///
/// Same reasoning as [`wheel_composition_device_target`]: the specs are stored
/// verbatim, so the answer is derivable from what was actually installed and
/// cannot drift from a second field. `None` for a manifest with no recorded
/// composition — see [`InstalledRuntimeManifest::includes_devel`] for how that
/// legacy case is resolved.
pub(crate) fn wheel_composition_includes_devel(
    composition: Option<&WheelRuntimeComposition>,
) -> Option<bool> {
    let composition = composition?;
    composition.package_specs.iter().find_map(|spec| {
        let extras = spec.strip_prefix("rocm[")?.split_once(']')?.0;
        Some(
            extras
                .split(',')
                .map(str::trim)
                .any(|extra| extra == "devel"),
        )
    })
}

/// The GFX target a recorded composition installed, read back out of its
/// `rocm[...,device-<target>]` requirement.
///
/// Update planning needs the target the runtime was built with, not the one this
/// host happens to report now; storing the specs verbatim means that answer
/// survives without a second manifest field to keep in sync.
pub(crate) fn wheel_composition_device_target(
    composition: Option<&WheelRuntimeComposition>,
) -> Option<&str> {
    let composition = composition?;
    composition.rocm_sdk_target.as_deref().or_else(|| {
        composition.package_specs.iter().find_map(|spec| {
            let extras = spec.strip_prefix("rocm[")?.split_once(']')?.0;
            extras
                .split(',')
                .map(str::trim)
                .find_map(|extra| extra.strip_prefix("device-"))
        })
    })
}

pub(crate) fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

pub(crate) fn runtime_manifest_path(paths: &AppPaths, runtime_key: &str) -> PathBuf {
    runtime_registry_dir(paths).join(format!("{runtime_key}.json"))
}

pub(crate) fn save_runtime_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime manifest registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&manifest).context("failed to serialize runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;

    let local_manifest_path = manifest.install_root.join(".rocm-cli-runtime.json");
    fs::write(
        &local_manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .context("failed to serialize local runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", local_manifest_path.display()))?;
    Ok(())
}

pub(crate) fn load_runtime_manifests(paths: &AppPaths) -> Result<Vec<InstalledRuntimeManifest>> {
    Ok(load_runtime_manifests_reporting_unparsed(paths)?.0)
}

/// [`load_runtime_manifests`] plus the registry entries that read fine but did
/// not deserialize.
///
/// A manifest written by an older binary is the ordinary way to land here:
/// `family_source`, `selected_artifact_url` and `installed_at_unix_ms` carry no
/// `#[serde(default)]`, so an older file fails `from_slice` against a newer
/// binary. Dropping those silently is right for the listing and lookup callers
/// — one stale file must not brick `rocm runtimes list` — but it is wrong for
/// the install consent gate, which has to know that its view of "what is
/// active" is incomplete. Hence two entry points rather than one hard error.
pub(crate) fn load_runtime_manifests_reporting_unparsed(
    paths: &AppPaths,
) -> Result<(Vec<InstalledRuntimeManifest>, Vec<PathBuf>)> {
    let registry_dir = runtime_registry_dir(paths);
    if !registry_dir.is_dir() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut manifests = Vec::new();
    let mut unparsed = Vec::new();
    for entry in fs::read_dir(&registry_dir)
        .with_context(|| format!("failed to read {}", registry_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        match serde_json::from_slice::<InstalledRuntimeManifest>(&bytes) {
            Ok(manifest) => manifests.push(manifest.normalize_host_paths()),
            Err(_) => unparsed.push(path),
        }
    }
    sort_manifests_newest_install_first(&mut manifests);
    unparsed.sort();
    Ok((manifests, unparsed))
}

/// Order installed runtimes newest install first, breaking ties on the runtime
/// key.
///
/// The tiebreak is what makes this a function of the registry contents rather
/// than of the order `read_dir` happened to return them in. Two runtimes can
/// share an `installed_at_unix_ms` — a `--devel` install and its plain sibling
/// land in the same millisecond, and the field is only millisecond-resolution —
/// and `select_startup_update_manifest` falls back to the first entry when no
/// active runtime key is configured, so without it the filesystem decided which
/// runtime the startup update check looked at. `select_runtimes_to_remove`
/// already applies the same rule to its retention groups.
pub(crate) fn sort_manifests_newest_install_first(manifests: &mut [InstalledRuntimeManifest]) {
    manifests.sort_by(|left, right| {
        right
            .installed_at_unix_ms
            .cmp(&left.installed_at_unix_ms)
            .then_with(|| left.runtime_key.cmp(&right.runtime_key))
    });
}

pub(crate) fn validate_rocm_sdk_runtime_probe(probe: &RocmSdkPythonProbe) -> Result<()> {
    if !probe.import_ok {
        bail!(
            "TheRock packages did not expose a usable rocm_sdk runtime: {}",
            probe.error.as_deref().unwrap_or("<unknown error>")
        );
    }
    let Some(rocm_sdk_root) = probe.root_path.as_ref() else {
        bail!("TheRock packages exposed rocm_sdk but did not report a runtime root path");
    };
    if !rocm_sdk_root.is_dir() {
        bail!(
            "TheRock rocm_sdk runtime root path does not exist: {}",
            rocm_sdk_root.display()
        );
    }
    let Some(rocm_sdk_bin) = probe.bin_path.as_ref() else {
        bail!("TheRock packages exposed rocm_sdk but did not report a runtime bin path");
    };
    if !rocm_sdk_bin.is_dir() {
        bail!(
            "TheRock rocm_sdk runtime bin path does not exist: {}",
            rocm_sdk_bin.display()
        );
    }
    if !probe_has_resolved_library(probe, "amdhip64") {
        bail!("TheRock rocm_sdk runtime did not expose amdhip64 through rocm_sdk.find_libraries");
    }
    if !probe_has_resolved_library(probe, "hipblas") {
        bail!("TheRock rocm_sdk runtime did not expose hipblas through rocm_sdk.find_libraries");
    }
    Ok(())
}

fn probe_has_resolved_library(probe: &RocmSdkPythonProbe, shortname: &str) -> bool {
    probe.resolved_libraries.iter().any(|library| {
        library.shortname == shortname && library.paths.iter().any(|path| path.is_file())
    })
}

pub(crate) fn current_runtime_manifest<'a>(
    config: &RocmCliConfig,
    manifests: &'a [InstalledRuntimeManifest],
) -> Option<&'a InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref()
        && let Some(manifest) = manifests
            .iter()
            .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(active_key))
    {
        return Some(manifest);
    }

    let matches = default_runtime_id_matches(config, manifests);
    match matches.as_slice() {
        [manifest] => Some(*manifest),
        _ => None,
    }
}

pub(crate) fn default_runtime_id_matches<'a>(
    config: &RocmCliConfig,
    manifests: &'a [InstalledRuntimeManifest],
) -> Vec<&'a InstalledRuntimeManifest> {
    let Some(default_runtime_id) = config.default_runtime_id.as_deref() else {
        return Vec::new();
    };
    manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id))
        .collect()
}

pub(crate) fn runtime_keys_text(manifests: &[&InstalledRuntimeManifest]) -> String {
    manifests
        .iter()
        .map(|manifest| manifest.runtime_key.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolves a user-typed selector to the one installed runtime it names.
///
/// Matching a `runtime_key` regardless of letter case is a deliberate
/// convenience, but it only names one runtime while no two installed keys
/// differ just in case — and they can. Generated keys are slugified to lower
/// case, while `runtimes adopt --runtime-key` takes the key the user typed and
/// `runtimes import` takes it from the supplied manifest; neither slugifies,
/// so the key is stored verbatim. The registry keeps one file per key, so
/// `ABC.json` and `abc.json` are two entries on a case-sensitive filesystem
/// and nothing rejects the pair.
///
/// So an exact key match is taken first and wins outright: it is the one
/// reading of the selector that cannot be mistaken. The case-insensitive
/// fallback applies only when it lands on a single record; two case twins are
/// reported as the ambiguity they are, the same way a `runtime_id` naming
/// several installs already is.
///
/// Callers rely on that being a function of the key alone, because several of
/// them resolve the same key more than once and must land on the same record
/// every time: `storage remove-old-installs` decides about a manifest and then
/// names it to the deleter by key, and the uninstall revalidation re-derives
/// its plan from disk after the confirmation prompt. Resolving by "newest
/// case-insensitive match" made that decide about one install and act on
/// another.
pub(crate) fn select_runtime_manifest<'a>(
    manifests: &'a [InstalledRuntimeManifest],
    selector: &str,
) -> Result<&'a InstalledRuntimeManifest> {
    let selector = selector.trim();
    if selector.is_empty() {
        bail!("runtime selector must not be empty");
    }

    if let Some(manifest) = manifests
        .iter()
        .find(|manifest| manifest.runtime_key == selector)
    {
        return Ok(manifest);
    }

    let key_matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match key_matches.as_slice() {
        [manifest] => return Ok(manifest),
        [] => {}
        _ => bail!(
            "runtime selector `{selector}` matches several installed runtime keys that differ only in letter case; name one of them exactly: {}",
            runtime_keys_text(&key_matches)
        ),
    }

    let id_matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match id_matches.as_slice() {
        [manifest] => Ok(manifest),
        [] => bail!("installed runtime not found: {selector}"),
        _ => bail!(
            "runtime selector `{selector}` matches multiple installed runtimes; activate one by runtime_key: {}",
            runtime_keys_text(&id_matches)
        ),
    }
}

pub(crate) fn runtime_usability_status(manifest: &InstalledRuntimeManifest) -> String {
    match validate_runtime_manifest_for_activation(manifest) {
        Ok(()) => "ready".to_owned(),
        Err(error) => format!("unusable ({error})"),
    }
}

pub(crate) fn validate_runtime_manifest_for_activation(
    manifest: &InstalledRuntimeManifest,
) -> Result<()> {
    if manifest.runtime_key.trim().is_empty() {
        bail!("manifest runtime_key is empty");
    }
    if manifest.runtime_id.trim().is_empty() {
        bail!("manifest runtime_id is empty");
    }
    if !manifest.install_root.is_dir() {
        bail!(
            "install root is missing: {}",
            manifest.install_root.display()
        );
    }
    let local_manifest = manifest.install_root.join(".rocm-cli-runtime.json");
    if !manifest.read_only && !local_manifest.is_file() {
        bail!(
            "local runtime manifest is missing: {}",
            local_manifest.display()
        );
    }

    match manifest.format.as_str() {
        "wheel" => validate_wheel_runtime_manifest(manifest),
        "tarball" => validate_tarball_runtime_manifest(manifest),
        other => bail!("unsupported runtime format in manifest: {other}"),
    }
}

fn validate_wheel_runtime_manifest(manifest: &InstalledRuntimeManifest) -> Result<()> {
    let python_executable = manifest
        .python_executable
        .as_deref()
        .context("pip runtime manifest is missing python_executable")?;
    if !Path::new(python_executable).is_file() {
        bail!("runtime Python executable is missing: {python_executable}");
    }
    let probe = manifest
        .rocm_sdk
        .as_ref()
        .context("pip runtime manifest is missing rocm_sdk probe data")?;
    validate_rocm_sdk_runtime_probe(probe)?;
    Ok(())
}

fn validate_tarball_runtime_manifest(manifest: &InstalledRuntimeManifest) -> Result<()> {
    if runtime_install_root_has_payload(&manifest.install_root)? {
        Ok(())
    } else {
        bail!(
            "tarball runtime install root has no payload files: {}",
            manifest.install_root.display()
        )
    }
}

fn runtime_install_root_has_payload(path: &Path) -> Result<bool> {
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_test_pip_runtime;

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("runtime_registry")
            .join(format!(
                "rocm-cli-registry-test-{name}-{}",
                crate::test_support::unique_suffix()
            ));
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    fn test_runtime_manifest(
        runtime_key: &str,
        runtime_id: &str,
        installed_at_unix_ms: u128,
    ) -> InstalledRuntimeManifest {
        InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: runtime_id.to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: runtime_id
                .split_once(':')
                .map_or_else(|| "gfx120X-all".to_owned(), |(_, family)| family.to_owned()),
            family_source: "test".to_owned(),
            version: "7.13.0a20260416".to_owned(),
            install_root: PathBuf::from("runtime-root"),
            selected_artifact_url: "https://example.invalid/rocm".to_owned(),
            source_layout_generation: None,
            index_url: Some("https://example.invalid/simple".to_owned()),
            tarball_file_name: None,
            python_launcher: Some("python".to_owned()),
            python_executable: Some("python".to_owned()),
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms,
        }
    }

    /// A manifest from before compositions were recorded has only the field,
    /// and one older still has neither — those installs all had the toolchain.
    #[test]
    fn manifest_without_a_composition_falls_back_to_the_devel_field() {
        let without_composition = InstalledRuntimeManifest {
            wheel_composition: None,
            devel: false,
            ..test_runtime_manifest(
                "release-wheel-gfx94X-dcgpu-7.13.0",
                "therock-release:gfx94X-dcgpu",
                1,
            )
        };
        assert!(!without_composition.includes_devel());

        let legacy = InstalledRuntimeManifest {
            wheel_composition: None,
            devel: devel_default_for_legacy_manifest(),
            ..test_runtime_manifest(
                "release-wheel-gfx94X-dcgpu-7.13.0",
                "therock-release:gfx94X-dcgpu",
                1,
            )
        };
        assert!(
            legacy.includes_devel(),
            "a manifest predating the flag must be treated as a toolchain install"
        );
    }

    /// Writing a manifest over an existing one for the same `runtime_key`
    /// REPLACES it — `devel` included — rather than merging with what was on
    /// disk. That is the storage half of the reinstall-drift case: a user who
    /// installed with `--devel` and later reinstalls without it keeps
    /// `hipcc`/headers on disk while the manifest records `devel: false`, and
    /// `apply_runtime_update` reinstalls from that field on the next `rocm
    /// update`. Left as-is rather than fixed — the flag reflects what the most
    /// recent install asked for, which is arguably correct; the drift risk is
    /// that neither `rocm runtimes list` nor `rocm examine` surfaces `devel`,
    /// so the state is invisible.
    ///
    /// LIMIT: this covers `save_runtime_manifest` only. It does NOT reach
    /// `install_sdk`, which is what actually threads `include_devel` into the
    /// manifest — that call needs `uv`, a live index, and a real `rocm_sdk`
    /// probe, so it has no unit coverage here. Hardcoding `devel: true` at that
    /// write site passes this test and the rest of the suite.
    #[test]
    fn save_runtime_manifest_replaces_devel_rather_than_merging() -> Result<()> {
        let (root, paths) = test_paths("manifest-replaces-devel");
        let install_root = root.join("install-root");
        fs::create_dir_all(&install_root)?;
        let with_devel = InstalledRuntimeManifest {
            install_root,
            ..test_runtime_manifest(
                "release-wheel-gfx110X-all-7.13.0",
                "therock-release:gfx110X-all",
                1,
            )
        };
        assert!(
            with_devel.devel,
            "test fixture should start with devel: true"
        );
        save_runtime_manifest(&paths, &with_devel)?;

        let without_devel = InstalledRuntimeManifest {
            devel: false,
            installed_at_unix_ms: 2,
            ..with_devel.clone()
        };
        save_runtime_manifest(&paths, &without_devel)?;

        let manifests = load_runtime_manifests(&paths)?;
        let reloaded = manifests
            .iter()
            .find(|manifest| manifest.runtime_key == with_devel.runtime_key)
            .expect("manifest should still be registered under the same runtime_key");
        assert!(
            !reloaded.devel,
            "saving a manifest must replace devel, not merge with the prior one"
        );

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// A manifest written before `devel` became opt-in has no such field, and
    /// those installs all had the toolchain. Reading one back must not claim
    /// otherwise, or the next update would silently strip it.
    #[test]
    fn legacy_manifest_without_devel_field_is_treated_as_having_it() {
        let json = r#"{
            "runtime_key": "release-wheel-gfx110X-all-7.13.0",
            "runtime_id": "therock-release:gfx110X-all",
            "channel": "release",
            "format": "wheel",
            "family": "gfx110X-all",
            "family_source": "detected",
            "version": "7.13.0",
            "install_root": "/tmp/rocm-runtime",
            "selected_artifact_url": "https://example.invalid/simple",
            "installed_at_unix_ms": 1
        }"#;

        let manifest: InstalledRuntimeManifest =
            serde_json::from_str(json).expect("legacy manifest should still parse");

        assert!(
            manifest.devel,
            "a manifest predating the flag must be treated as a toolchain install"
        );
    }

    #[test]
    fn recorded_composition_names_the_device_payload_it_installed() {
        let composition = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec![
                "rocm[libraries,devel,device-gfx1103]==7.14.1".to_owned(),
                "torch==2.11.0+rocm7.14.1".to_owned(),
            ],
            rocm_sdk_target: Some("gfx1103".to_owned()),
        };

        assert_eq!(
            wheel_composition_device_target(Some(&composition)),
            Some("gfx1103"),
            "update planning must recover the installed target without probing the current host"
        );
        assert_eq!(wheel_composition_device_target(None), None);
    }

    /// Matching a key regardless of case is a convenience for a user who typed
    /// the wrong one; it must not quietly pick between two installs whose keys
    /// differ only in case. `runtimes adopt --runtime-key` stores the key
    /// verbatim while generated keys are lower-cased, so such a pair is
    /// reachable, and a caller that resolves the same key twice — prune decides
    /// about a manifest and then names it to the deleter by key — has to get
    /// the same install both times.
    ///
    /// No registry is involved here: the manifests are handed straight to the
    /// resolver, so this holds on every platform even though only a
    /// case-sensitive filesystem can store such a pair.
    #[test]
    fn runtime_selector_prefers_the_exact_key_over_a_case_twin() -> Result<()> {
        let (root, paths) = test_paths("runtime-case-twin");
        let base = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        // Newest first, as `load_runtime_manifests` hands them over. The
        // adopted twin is stamped with the moment it was adopted, so it leads.
        let manifests = vec![
            InstalledRuntimeManifest {
                runtime_key: "RELEASE-PIP-GFX120X-ALL-7-13-0".to_owned(),
                installed_at_unix_ms: 99,
                read_only: true,
                ..base.clone()
            },
            base.clone(),
        ];

        let exact = select_runtime_manifest(&manifests, &base.runtime_key)?;
        assert_eq!(
            exact.runtime_key, base.runtime_key,
            "the key as stored resolves to its own record, not to the newer twin"
        );
        let twin = select_runtime_manifest(&manifests, "RELEASE-PIP-GFX120X-ALL-7-13-0")?;
        assert_eq!(twin.runtime_key, "RELEASE-PIP-GFX120X-ALL-7-13-0");

        // Neither key matches exactly, so the selector genuinely names both.
        // Reported as the ambiguity it is, like a runtime_id naming several.
        let error = select_runtime_manifest(&manifests, "Release-Pip-Gfx120X-All-7-13-0")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("differ only in letter case"),
            "an ambiguous selector must be refused, not silently resolved: {error}"
        );
        assert!(error.contains("release-pip-gfx120x-all-7-13-0"));
        assert!(error.contains("RELEASE-PIP-GFX120X-ALL-7-13-0"));

        // The convenience itself is untouched: with one candidate, any casing
        // still resolves.
        let only = vec![base.clone()];
        assert_eq!(
            select_runtime_manifest(&only, "Release-Pip-Gfx120X-All-7-13-0")?.runtime_key,
            base.runtime_key
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
