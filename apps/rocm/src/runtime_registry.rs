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
//! and path-helper names that other modules import from it, but it,
//! `runtime_adopt` and `runtime_uninstall` still reach the manifest model (the
//! manifest and probe types and the loaders) through `therock`'s re-exports,
//! along with a few `therock` helpers. So `runtimes_cmd` depends on `therock`,
//! while `therock` no longer depends on `runtimes_cmd` in production code. It
//! also holds `paths_equivalent` and `normalize_path_for_compare`, which adopt
//! and uninstall both need, and the on-disk active-runtime marker
//! (`ActiveRuntimeMarker`, `active_runtime_marker_path`,
//! `active_runtime_marker_matches`) that activation writes and uninstall reads
//! and clears. The serde shape of [`InstalledRuntimeManifest`] and its probe
//! structs is the on-disk contract of the registry files and of each install's
//! `.rocm-cli-runtime.json`; the marker is user state too, and its serde shape
//! is as fixed.

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

    /// Whether ROCm CLI owns the folder this runtime lives in: it installed it,
    /// rather than importing or adopting somebody else's. Ownership is what
    /// decides whether the install root carries a `.rocm-cli-runtime.json` (see
    /// [`write_runtime_manifest`]) and whether uninstall and prune may delete
    /// the folder, so those cannot disagree about which runtimes are whose.
    pub(crate) const fn owns_install_root(&self) -> bool {
        !self.read_only && self.imported_from.is_none()
    }

    pub(crate) fn normalize_host_paths(mut self) -> Self {
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

/// What [`write_runtime_manifest`] does when the registry already holds an entry
/// under the manifest's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManifestOverwrite {
    /// Write over it. An install of a runtime ROCm CLI already installed is the
    /// same runtime again.
    Replace,
    /// Refuse, and name `--replace`. Importing or adopting a runtime must not
    /// displace a registration the user made earlier unless they ask to.
    Refuse,
}

impl ManifestOverwrite {
    /// The policy for a command with a `--replace` flag.
    pub(crate) const fn for_replace_flag(replace: bool) -> Self {
        if replace { Self::Replace } else { Self::Refuse }
    }
}

/// Write the manifest files of a runtime: the registry entry, and, for a
/// runtime ROCm CLI owns ([`InstalledRuntimeManifest::owns_install_root`]), the
/// `.rocm-cli-runtime.json` in its install root.
///
/// Every way a runtime gets registered goes through this one routine: an SDK
/// install, `rocm runtimes import`, `rocm runtimes adopt`, and the recovery of
/// the runtime `rocm setup` created. `overwrite` is the only thing a caller
/// chooses.
///
/// The file in the install root is ROCm CLI's proof that it owns the folder:
/// uninstall and prune delete a folder only when it carries one that matches
/// the registry, and activation requires one of a runtime that is not
/// read-only. An imported or adopted runtime is somebody else's folder, which
/// uninstall and prune never delete and activation accepts without one, so it
/// gets none: a root the user may not be able to write to must not make
/// registering it fail, and an import's source manifest may be that very file
/// in another install.
///
/// The registry entry is written last, after the install-root file. A failed
/// install-root write therefore leaves the registry exactly as it was, and a
/// failed registry write leaves no new entry, so an error that says a runtime
/// could not be registered or restored never describes one whose entry exists.
/// The install-root file may be left behind by a failed registry write; it holds
/// what a retry writes.
///
/// The file is not rewritten when it already holds what would be written, so
/// recovering a runtime from its own manifest leaves it alone. "Holds" is judged
/// on the decoded manifest, not the bytes ([`install_root_manifest_is_current`]):
/// a file an older binary wrote omits keys that now read back as defaults and
/// are written out explicitly, and a root the user cannot write to must not turn
/// that into a failure.
///
/// Neither file is written atomically; that is a separate piece of work shared
/// with the other writers in the workspace.
pub(crate) fn write_runtime_manifest(
    paths: &AppPaths,
    manifest: &InstalledRuntimeManifest,
    overwrite: ManifestOverwrite,
) -> Result<()> {
    let manifest = manifest.clone().normalize_storage_paths();
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    if overwrite == ManifestOverwrite::Refuse && registry_path.exists() {
        bail!(
            "runtime registry entry already exists: {}; pass --replace to overwrite it",
            registry_path.display()
        );
    }
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime manifest registry path has no parent directory")?,
    )?;
    let encoded =
        serde_json::to_vec_pretty(&manifest).context("failed to serialize runtime manifest")?;

    if manifest.owns_install_root() {
        let local_manifest_path = manifest.install_root.join(".rocm-cli-runtime.json");
        let already_current = fs::read(&local_manifest_path)
            .is_ok_and(|existing| install_root_manifest_is_current(&existing, &encoded));
        if !already_current {
            fs::write(&local_manifest_path, &encoded)
                .with_context(|| format!("failed to write {}", local_manifest_path.display()))?;
        }
    }
    // Last, because the registry entry is what registering a runtime means; see
    // the doc comment.
    fs::write(&registry_path, &encoded)
        .with_context(|| format!("failed to write {}", registry_path.display()))?;
    Ok(())
}

/// Whether an install root manifest already holds the manifest that `encoded`
/// is the storage form of. Equal bytes qualify, and so does a file that decodes
/// to the same manifest once normalised for storage: a manifest written before
/// a field existed has no key for it, reads back as the field's default, and
/// serializes with an explicit value, so a byte comparison reads it as changed.
/// A file that does not decode is not current.
fn install_root_manifest_is_current(existing: &[u8], encoded: &[u8]) -> bool {
    if existing == encoded {
        return true;
    }
    serde_json::from_slice::<InstalledRuntimeManifest>(existing)
        .ok()
        .and_then(|decoded| serde_json::to_vec_pretty(&decoded.normalize_storage_paths()).ok())
        .is_some_and(|normalised| normalised == encoded)
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
    if rocm_core::install_root_has_payload(&manifest.install_root)? {
        Ok(())
    } else {
        bail!(
            "tarball runtime install root has no payload files: {}",
            manifest.install_root.display()
        )
    }
}

pub(crate) fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = normalize_path_for_compare(left);
    let right = normalize_path_for_compare(right);
    rocm_core::runtime_paths_equivalent(&left, &right)
}

pub(crate) fn normalize_path_for_compare(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ActiveRuntimeMarker {
    pub(crate) runtime_id: String,
    pub(crate) runtime_key: String,
    pub(crate) manifest_path: PathBuf,
    pub(crate) install_root: PathBuf,
    pub(crate) previous_runtime_id: Option<String>,
    pub(crate) previous_runtime_key: Option<String>,
    pub(crate) activated_at_unix_ms: u128,
}

pub(crate) fn active_runtime_marker_matches(paths: &AppPaths, runtime_key: &str) -> Result<bool> {
    let marker_path = active_runtime_marker_path(paths);
    if !marker_path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&marker_path)
        .with_context(|| format!("failed to read {}", marker_path.display()))?;
    let marker: ActiveRuntimeMarker = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", marker_path.display()))?;
    Ok(marker.runtime_key.eq_ignore_ascii_case(runtime_key))
}

pub(crate) fn active_runtime_marker_path(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("active.json")
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
    /// LIMIT: this covers `write_runtime_manifest` only. It does NOT reach
    /// `install_sdk`, which is what actually threads `include_devel` into the
    /// manifest — that call needs `uv`, a live index, and a real `rocm_sdk`
    /// probe, so it has no unit coverage here. Hardcoding `devel: true` at that
    /// write site passes this test and the rest of the suite.
    #[test]
    fn write_runtime_manifest_replaces_devel_rather_than_merging() -> Result<()> {
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
        write_runtime_manifest(&paths, &with_devel, ManifestOverwrite::Replace)?;

        let without_devel = InstalledRuntimeManifest {
            devel: false,
            installed_at_unix_ms: 2,
            ..with_devel.clone()
        };
        write_runtime_manifest(&paths, &without_devel, ManifestOverwrite::Replace)?;

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

    /// A manifest for `runtime_key` whose install root is a folder under `root`.
    fn manifest_in(
        root: &Path,
        runtime_key: &str,
        read_only: bool,
        imported_from: Option<PathBuf>,
    ) -> Result<InstalledRuntimeManifest> {
        let install_root = root.join(runtime_key);
        fs::create_dir_all(&install_root)?;
        Ok(InstalledRuntimeManifest {
            install_root,
            read_only,
            imported_from,
            ..test_runtime_manifest(runtime_key, "therock-release:gfx120X-all", 1)
        })
    }

    /// The policy is the only thing a caller chooses, and it decides one thing:
    /// whether an existing registry entry is replaced or left alone. The refusal
    /// is asserted as message and state together, and with the entry still the
    /// old bytes, because a refusal that wrote first would print the same text.
    #[test]
    fn write_runtime_manifest_refuses_an_existing_entry_only_under_the_refuse_policy() -> Result<()>
    {
        let (root, paths) = test_paths("manifest-overwrite-policy");
        let manifest = manifest_in(&root, "release-wheel-gfx120X-all-7.13.0", false, None)?;
        let entry = runtime_manifest_path(&paths, &manifest.runtime_key);

        write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Refuse)?;
        assert!(
            entry.is_file(),
            "nothing registered yet, so nothing to refuse"
        );

        fs::write(&entry, "stale entry")?;
        let error = write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Refuse)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("already exists") && error.contains("--replace"),
            "{error}"
        );
        assert_eq!(fs::read_to_string(&entry)?, "stale entry");

        write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace)?;
        let replaced: InstalledRuntimeManifest = serde_json::from_slice(&fs::read(&entry)?)?;
        assert_eq!(replaced.runtime_key, manifest.runtime_key);

        assert_eq!(
            ManifestOverwrite::for_replace_flag(true),
            ManifestOverwrite::Replace
        );
        assert_eq!(
            ManifestOverwrite::for_replace_flag(false),
            ManifestOverwrite::Refuse
        );
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// Which manifests get a `.rocm-cli-runtime.json` in the install root:
    /// exactly the ones ROCm CLI owns. A read-only runtime and an imported one
    /// (the two ways a runtime is not ROCm CLI's) get a registry entry and leave
    /// their folder alone, whichever policy registered them.
    #[test]
    fn write_runtime_manifest_writes_the_install_root_manifest_only_for_an_owned_runtime()
    -> Result<()> {
        let (root, paths) = test_paths("manifest-install-root-file");
        let cases = [
            ("owned", false, None, true),
            ("read-only", true, None, false),
            (
                "imported",
                false,
                Some(PathBuf::from("elsewhere/manifest.json")),
                false,
            ),
        ];
        for overwrite in [ManifestOverwrite::Replace, ManifestOverwrite::Refuse] {
            for (label, read_only, imported_from, gets_the_file) in &cases {
                let key = format!("{label}-{overwrite:?}");
                let manifest = manifest_in(&root, &key, *read_only, imported_from.clone())?;

                write_runtime_manifest(&paths, &manifest, overwrite)?;

                assert_eq!(
                    manifest.owns_install_root(),
                    *gets_the_file,
                    "{label} / {overwrite:?}"
                );
                assert!(
                    runtime_manifest_path(&paths, &key).is_file(),
                    "{label} / {overwrite:?}: registry entry"
                );
                assert_eq!(
                    manifest
                        .install_root
                        .join(".rocm-cli-runtime.json")
                        .is_file(),
                    *gets_the_file,
                    "{label} / {overwrite:?}: install root manifest"
                );
            }
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The registry entry is what registers a runtime, so it is written last and
    /// each failing write is asserted as the error and the state together. When
    /// the install-root file cannot be written (a directory stands where the file
    /// should be, which stops an administrator as well) the error names that file
    /// and the registry is exactly as it was: no entry, so a caller's "could not
    /// register" or "could not restore" is true, under either policy. When the
    /// registry entry cannot be written (a directory stands where it should be)
    /// the error names the entry and there is still no entry that reads as a
    /// runtime.
    #[test]
    fn a_failed_manifest_write_never_leaves_a_registered_runtime_behind() -> Result<()> {
        for overwrite in [ManifestOverwrite::Replace, ManifestOverwrite::Refuse] {
            let (root, paths) = test_paths("manifest-write-fails");
            let manifest = manifest_in(&root, "release-wheel-gfx120X-all-7.13.0", false, None)?;
            let entry = runtime_manifest_path(&paths, &manifest.runtime_key);
            let local = manifest.install_root.join(".rocm-cli-runtime.json");

            fs::create_dir_all(&local)?;
            let error = format!(
                "{:#}",
                write_runtime_manifest(&paths, &manifest, overwrite)
                    .expect_err("the install-root manifest cannot be written")
            );
            assert!(
                error.contains("failed to write") && error.contains(".rocm-cli-runtime.json"),
                "{overwrite:?}: {error}"
            );
            assert!(
                !entry.exists() && load_runtime_manifests(&paths)?.is_empty(),
                "{overwrite:?}: a failed install-root write leaves no registry entry"
            );

            fs::remove_dir(&local)?;
            fs::create_dir_all(&entry)?;
            let error = format!(
                "{:#}",
                write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace)
                    .expect_err("the registry entry cannot be written")
            );
            assert!(
                error.contains("failed to write") && error.contains(&entry.display().to_string()),
                "{overwrite:?}: {error}"
            );
            assert!(
                !entry.is_file(),
                "{overwrite:?}: a failed registry write leaves no entry"
            );
            let _ = fs::remove_dir_all(&root);
        }
        Ok(())
    }

    /// Recovering a runtime from its own install-root manifest must not touch
    /// that file when it already holds what would be written, so a setup folder
    /// the user cannot write to does not turn a recovery into an error. The file
    /// is made read-only: a write would fail where a skip succeeds. (Not
    /// asserted when the file stays writable regardless, as for an
    /// administrator.)
    #[test]
    #[allow(clippy::permissions_set_readonly_false)]
    fn write_runtime_manifest_leaves_a_current_install_root_manifest_alone() -> Result<()> {
        let (root, paths) = test_paths("manifest-current-install-root-file");
        let manifest = manifest_in(&root, "release-wheel-gfx120X-all-7.13.0", false, None)?;
        write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace)?;
        let local = manifest.install_root.join(".rocm-cli-runtime.json");
        let mut permissions = fs::metadata(&local)?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&local, permissions.clone())?;
        let still_writable = fs::OpenOptions::new().append(true).open(&local).is_ok();

        let unchanged = write_runtime_manifest(&paths, &manifest, ManifestOverwrite::Replace);
        let changed = InstalledRuntimeManifest {
            version: "7.14.0".to_owned(),
            ..manifest
        };
        let rewritten = write_runtime_manifest(&paths, &changed, ManifestOverwrite::Replace);

        permissions.set_readonly(false);
        fs::set_permissions(&local, permissions)?;
        unchanged.context("an install root manifest that is already current is not rewritten")?;
        if !still_writable {
            assert!(
                rewritten.is_err(),
                "a different manifest is written, which the read-only file refuses"
            );
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The state matrix for "is the install root manifest already current":
    /// the same bytes, the same manifest in an older shape (keys that hold no
    /// value omitted, so a byte comparison reads it as changed), a different
    /// manifest in that shape, and a file that is not a manifest at all. Pure, so
    /// it runs on every platform, unlike the read-only-file tests that show the
    /// effect on the filesystem.
    #[test]
    fn install_root_manifest_is_current_compares_the_decoded_manifest() -> Result<()> {
        let (root, _paths) = test_paths("manifest-current-decoded");
        let manifest = manifest_in(&root, "release-wheel-gfx120X-all-7.13.0", false, None)?;
        let stored = manifest.clone().normalize_storage_paths();
        let encoded = serde_json::to_vec_pretty(&stored)?;
        let legacy_shaped = |manifest: &InstalledRuntimeManifest| -> Result<Vec<u8>> {
            let mut document = serde_json::to_value(manifest.clone().normalize_storage_paths())?;
            let object = document.as_object_mut().context("a JSON object")?;
            let empty: Vec<String> = object
                .iter()
                .filter(|(_, value)| value.is_null())
                .map(|(key, _)| key.clone())
                .collect();
            assert!(!empty.is_empty(), "the fixture needs a null-valued key");
            for key in empty {
                object.remove(&key);
            }
            Ok(serde_json::to_vec_pretty(&document)?)
        };

        assert!(install_root_manifest_is_current(&encoded, &encoded));
        let legacy = legacy_shaped(&manifest)?;
        assert_ne!(legacy, encoded, "the legacy shape must differ in bytes");
        assert!(
            install_root_manifest_is_current(&legacy, &encoded),
            "a legacy-shaped file that decodes to the same manifest is current"
        );
        let newer = InstalledRuntimeManifest {
            version: "7.14.0".to_owned(),
            ..manifest
        };
        assert!(
            !install_root_manifest_is_current(&legacy_shaped(&newer)?, &encoded),
            "a legacy-shaped file that decodes to a different manifest is not current"
        );
        assert!(!install_root_manifest_is_current(b"not json", &encoded));
        assert!(!install_root_manifest_is_current(b"", &encoded));
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// Calls that put bytes in a file, which is what a second writer of a registry
    /// entry would use. Matched as text, so `tokio::fs::write(` counts too. The
    /// file builders (`OpenOptions`, `File::options`) are not here: they write
    /// only when a write option is chained on them, see [`open_options_write`].
    const WRITE_PRIMITIVES: [&str; 7] = [
        "fs::write(",
        "File::create(",
        "File::create_new(",
        "fs::copy(",
        "fs::rename(",
        "fs::hard_link(",
        "write_all(",
    ];

    /// The file builders, and the options that make one write. `OpenOptions::new()`
    /// followed by `.read(true).open(..)` is a read-only probe and not a writer.
    const FILE_BUILDERS: [&str; 2] = ["OpenOptions::new(", "File::options("];
    const WRITE_OPTIONS: [&str; 5] = ["write", "append", "create", "create_new", "truncate"];

    /// The index just past the `)` that closes the `(` at `open`, skipping string
    /// literals, or the end of `text` when it never closes.
    fn skip_parens(text: &[u8], open: usize) -> usize {
        let mut depth = 0usize;
        let mut at = open;
        let mut in_string = false;
        while at < text.len() {
            let byte = text[at];
            if in_string {
                match byte {
                    b'\\' => at += 1,
                    b'"' => in_string = false,
                    _ => {}
                }
            } else {
                match byte {
                    b'"' => in_string = true,
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            return at + 1;
                        }
                    }
                    _ => {}
                }
            }
            at += 1;
        }
        text.len()
    }

    /// Whether `production` builds a file with a builder that can write. The
    /// method chain after the builder is read across lines and whitespace: it is
    /// a read-only probe only when the chain reaches `.open(` and none of
    /// [`WRITE_OPTIONS`] is called on it. A builder that is stored (`let mut o =
    /// OpenOptions::new();`) or whose chain does not reach `.open(` cannot be
    /// judged and counts as a writer.
    fn open_options_write(production: &str) -> bool {
        let text = production.as_bytes();
        for builder in FILE_BUILDERS {
            for (at, _) in production.match_indices(builder) {
                let mut next = skip_parens(text, at + builder.len() - 1);
                let mut methods = Vec::new();
                loop {
                    while text.get(next).is_some_and(u8::is_ascii_whitespace) {
                        next += 1;
                    }
                    if text.get(next) != Some(&b'.') {
                        break;
                    }
                    next += 1;
                    while text.get(next).is_some_and(u8::is_ascii_whitespace) {
                        next += 1;
                    }
                    let start = next;
                    while text
                        .get(next)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                    {
                        next += 1;
                    }
                    methods.push(&production[start..next]);
                    while text.get(next).is_some_and(u8::is_ascii_whitespace) {
                        next += 1;
                    }
                    if text.get(next) == Some(&b'(') {
                        next = skip_parens(text, next);
                    }
                }
                let writes = methods.iter().any(|method| WRITE_OPTIONS.contains(method));
                if writes || !methods.contains(&"open") {
                    return true;
                }
            }
        }
        false
    }

    /// Every way `production` writes a file, by name.
    fn write_calls(production: &str) -> Vec<&'static str> {
        let mut calls: Vec<&'static str> = WRITE_PRIMITIVES
            .iter()
            .copied()
            .filter(|primitive| production.contains(primitive))
            .collect();
        if open_options_write(production) {
            calls.push("OpenOptions with a write option");
        }
        calls
    }

    /// `source` as the compiler sees production code, for a text scan: everything
    /// from the file's `#[cfg(test)]` test module on is cut, and comments are
    /// removed, so neither a fixture nor a sentence in a comment can be mistaken
    /// for a writer. String literals stay, because the path segments of the
    /// registry directory are string literals. The scan is not a lexer: a raw
    /// string that holds a double quote would confuse it.
    fn production_code(source: &str) -> String {
        let end = [
            "#[cfg(test)]\nmod tests",
            "#[cfg(test)]\npub(crate) mod tests",
        ]
        .iter()
        .filter_map(|marker| source.find(marker))
        .min()
        .unwrap_or(source.len());
        let chars: Vec<char> = source[..end].chars().collect();
        let mut code = String::with_capacity(end);
        let mut at = 0;
        let mut in_string = false;
        while at < chars.len() {
            let c = chars[at];
            if in_string {
                code.push(c);
                if c == '\\' && at + 1 < chars.len() {
                    code.push(chars[at + 1]);
                    at += 2;
                    continue;
                }
                in_string = c != '"';
                at += 1;
                continue;
            }
            match (c, chars.get(at + 1)) {
                ('/', Some('/')) => {
                    while at < chars.len() && chars[at] != '\n' {
                        at += 1;
                    }
                }
                ('/', Some('*')) => {
                    at += 2;
                    while at + 1 < chars.len() && !(chars[at] == '*' && chars[at + 1] == '/') {
                        at += 1;
                    }
                    at += 2;
                }
                // The character literal `'"'` must not open a string.
                ('\'', Some('"')) if chars.get(at + 2) == Some(&'\'') => {
                    code.push_str("'\"'");
                    at += 3;
                }
                ('"', _) => {
                    in_string = true;
                    code.push(c);
                    at += 1;
                }
                _ => {
                    code.push(c);
                    at += 1;
                }
            }
        }
        code
    }

    /// The ways production code can name the registry directory: through the
    /// two helpers in this file, or by spelling the path itself (the `runtimes`
    /// and `registry` segments, or one literal holding both).
    fn registry_location_spellings(production: &str) -> Vec<&'static str> {
        let mut spellings = Vec::new();
        for helper in ["runtime_manifest_path(", "runtime_registry_dir("] {
            if production.contains(helper) {
                spellings.push(helper);
            }
        }
        if production.contains("\"runtimes\"") && production.contains("\"registry\"") {
            spellings.push("the \"runtimes\" and \"registry\" path segments");
        }
        if production.contains("runtimes/registry") || production.contains("runtimes\\\\registry") {
            spellings.push("a literal path through runtimes/registry");
        }
        spellings
    }

    /// Why `production` writes a registry entry on its own, one line per way it
    /// can reach the registry, or nothing. A file that both writes (any of
    /// [`WRITE_PRIMITIVES`], or a file builder with a write option, see
    /// [`open_options_write`]) and names the registry directory (by any of
    /// [`registry_location_spellings`]) is a writer, however far apart the two
    /// are and whatever the path variable is called. A file that only reads or
    /// removes entries has no write call and is not one.
    fn registry_writer_findings(production: &str) -> Vec<String> {
        let writes = write_calls(production);
        if writes.is_empty() {
            return Vec::new();
        }
        registry_location_spellings(production)
            .into_iter()
            .map(|spelling| format!("{spelling} in a file that calls {}", writes.join(", ")))
            .collect()
    }

    /// The detector sees a writer however it builds its path, and nothing that
    /// only reads. The cases are the ways the previous scan (a 200-character
    /// window after `fs::write(` looking for three names) went blind: a path built
    /// from literals, a path held in a variable away from the write, one literal,
    /// a `//` inside a string literal before the write. A file builder
    /// (`OpenOptions`, `File::options`) counts as a write only when a write option
    /// is chained on it or the chain cannot be read; a read-only probe in a file
    /// that builds the registry path, as `amd_smi.rs` does, is not a writer.
    #[test]
    fn registry_writer_detector_sees_every_way_of_building_the_path() {
        let filler = "// padding\n".repeat(40);
        let far_apart = format!(
            "fn dir(p: &AppPaths) -> PathBuf {{ p.data_dir.join(\"runtimes\").join(\"registry\") }}\n{filler}fn save(d: &Path) {{ fs::write(d.join(\"x.json\"), b\"\").ok(); }}"
        );
        let registry = "fn dir(p: &AppPaths) -> PathBuf { p.data_dir.join(\"runtimes\").join(\"registry\") }\n";
        let cases: [(&str, String, bool); 17] = [
            (
                "amd_smi shape: the registry path is built, and only read-only probes open files",
                format!(
                    "{registry}fn probe(path: &Path) -> bool {{ fs::OpenOptions::new().read(true).open(path).is_ok() }}\nfn device(path: &Path) -> bool {{ std::fs::OpenOptions::new().read(true).open(path).is_ok() }}"
                ),
                false,
            ),
            (
                "a read-only probe whose chain spans lines",
                format!(
                    "{registry}fn probe(path: &Path) -> bool {{\n    OpenOptions::new()\n        .read(true)\n        .open(path)\n        .is_ok()\n}}"
                ),
                false,
            ),
            (
                "a read-only probe through File::options",
                format!("{registry}fn probe(p: &Path) -> bool {{ File::options().read(true).open(p).is_ok() }}"),
                false,
            ),
            (
                "OpenOptions with .write(true)",
                format!(
                    "{registry}fn save(d: &Path) {{ OpenOptions::new().write(true).create(true).open(d.join(\"k.json\")).ok(); }}"
                ),
                true,
            ),
            (
                "OpenOptions with a write option on a later line",
                format!(
                    "{registry}fn save(d: &Path) {{\n    fs::OpenOptions::new()\n        .read(true)\n        .truncate(true)\n        .open(d.join(\"k.json\"))\n        .ok();\n}}"
                ),
                true,
            ),
            (
                "File::options with .append(true)",
                format!("{registry}fn save(d: &Path) {{ File::options().append(true).open(d).ok(); }}"),
                true,
            ),
            (
                "a builder kept in a variable cannot be judged",
                format!(
                    "{registry}fn save(d: &Path) {{ let mut options = OpenOptions::new(); options.write(true); options.open(d).ok(); }}"
                ),
                true,
            ),
            (
                "a chain that never reaches .open( cannot be judged",
                format!("{registry}fn save() {{ let options = OpenOptions::new().read(true).clone(); drop(options); }}"),
                true,
            ),
            (
                "through the helper",
                "fn f(p: &AppPaths) { fs::write(runtime_manifest_path(p, \"k\"), b\"\").ok(); }"
                    .to_owned(),
                true,
            ),
            ("segments, far from the write", far_apart, true),
            (
                "one literal",
                "fn f(base: &Path) { let d = base.join(\"runtimes/registry\"); File::create(d.join(\"k.json\")).ok(); }"
                    .to_owned(),
                true,
            ),
            (
                "a // inside a string before the write",
                "fn f(p: &AppPaths) { let note = \"see http://x\"; fs::write(runtime_registry_dir(p).join(note), b\"\").ok(); }"
                    .to_owned(),
                true,
            ),
            (
                "reads only",
                "fn f(p: &AppPaths) { let _ = fs::read_dir(p.data_dir.join(\"runtimes\").join(\"registry\")); }"
                    .to_owned(),
                false,
            ),
            (
                "writes something else",
                "fn f(p: &Path) { fs::write(p.join(\"config.json\"), b\"\").ok(); }".to_owned(),
                false,
            ),
            (
                "only a comment names the writer",
                "// fs::write(runtime_manifest_path(p, k), b\"\")\nfn f() {}".to_owned(),
                false,
            ),
            (
                "the writer is in the test module",
                "fn f() {}\n#[cfg(test)]\nmod tests {\n fn t(p: &AppPaths) { fs::write(runtime_manifest_path(p, \"k\"), b\"\").ok(); }\n}"
                    .to_owned(),
                false,
            ),
            (
                "a block comment holds the path",
                "fn f(p: &Path) { /* runtime_registry_dir( */ fs::write(p, b\"\").ok(); }"
                    .to_owned(),
                false,
            ),
        ];
        for (label, source, is_a_writer) in cases {
            assert_eq!(
                !registry_writer_findings(&production_code(&source)).is_empty(),
                is_a_writer,
                "{label}"
            );
        }
    }

    /// Every way of registering a runtime goes through
    /// [`write_runtime_manifest`]. A second writer would be free to drift again
    /// (a different overwrite rule, a missing install-root manifest), so no
    /// production code outside this file may both name the registry directory and
    /// write a file. This reads the sources of the `rocm` app, `rocmd`,
    /// `rocm-core` and the engines: the writers differ only in how they are
    /// called, which no behavioural test of one caller can see in another.
    ///
    /// Two further rules keep that from being walked around. The files that call
    /// the two path helpers are an explicit list, each with the reason it needs
    /// the path and none of them writing, so a new caller is reviewed rather than
    /// passed, and a stale entry fails too. The two fixture modules are exempt
    /// because the modules are `#[cfg(test)]` where they are declared, which is
    /// checked, not because of what they are called.
    ///
    /// What this does not catch, by design: a writer that is handed the registry
    /// path by a function in another file and never spells it, a write primitive
    /// that is not in [`WRITE_PRIMITIVES`] (a new helper with its own name), a
    /// file builder whose write option is set through something other than the
    /// methods in [`WRITE_OPTIONS`] (the chain is read as text; a chain that
    /// reaches `.open(` with only read options is a read), and code outside the
    /// source roots it reads. Review catches those; this test
    /// catches the ordinary ways of growing a second writer.
    #[test]
    fn only_the_shared_routine_writes_a_registry_entry() {
        fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(dir).expect("source folder readable").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    rust_files(&path, found);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    found.push(path);
                }
            }
        }
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("the app sits two levels below the workspace root");
        let mut roots = vec![
            workspace.join("apps/rocm/src"),
            workspace.join("apps/rocmd/src"),
            workspace.join("crates/rocm-core/src"),
        ];
        for engine in fs::read_dir(workspace.join("engines"))
            .expect("engines folder readable")
            .flatten()
        {
            roots.push(engine.path().join("src"));
        }
        let mut sources = Vec::new();
        for root in &roots {
            rust_files(root, &mut sources);
        }
        assert!(sources.len() > 100, "the scan found the source tree");

        // This file holds the routine.
        let routine = "apps/rocm/src/runtime_registry.rs";
        // Fixture modules, each with the file that declares it.
        let fixtures = [
            ("apps/rocm/src/test_support.rs", "apps/rocm/src/main.rs"),
            (
                "apps/rocm/src/therock/test_support.rs",
                "apps/rocm/src/therock.rs",
            ),
        ];
        for (fixture, declared_in) in fixtures {
            let declaration =
                fs::read_to_string(workspace.join(declared_in)).expect("declaring file readable");
            assert!(
                declaration.contains("#[cfg(test)]\nmod test_support;"),
                "{fixture} is exempt only while {declared_in} declares it under #[cfg(test)]"
            );
        }
        // Production files that name an entry's path without writing it, and why
        // (in sorted order, which is how they are compared).
        let path_helper_callers = [
            // Whether setup's runtime is registered, and the error that names the registry.
            "apps/rocm/src/runtime_adopt.rs",
            // The entry uninstall removes.
            "apps/rocm/src/runtime_uninstall.rs",
            // The path an activation, import or adopt report prints.
            "apps/rocm/src/runtimes_cmd.rs",
            // The path an install plan prints.
            "apps/rocm/src/therock.rs",
        ];

        let mut helper_callers = Vec::new();
        for path in sources {
            let relative = path
                .strip_prefix(workspace)
                .expect("source under the workspace")
                .to_string_lossy()
                .replace('\\', "/");
            if relative == routine || fixtures.iter().any(|(fixture, _)| *fixture == relative) {
                continue;
            }
            let production = production_code(&fs::read_to_string(&path).expect("source readable"));
            let findings = registry_writer_findings(&production);
            assert!(
                findings.is_empty(),
                "{relative} writes a registry entry itself instead of calling write_runtime_manifest: {findings:?}"
            );
            if production.contains("runtime_manifest_path(")
                || production.contains("runtime_registry_dir(")
            {
                helper_callers.push(relative);
            }
        }
        helper_callers.sort();
        assert_eq!(
            helper_callers, path_helper_callers,
            "the files that call runtime_manifest_path or runtime_registry_dir changed: a new caller must be reviewed (and listed with its reason) and a file that stopped calling them removed"
        );
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

    /// Activation validation of a tarball runtime follows
    /// `rocm_core::install_root_has_payload`: the root's own `.rocm-cli-runtime.json`
    /// marker is a dot-entry and does not make a payload, so a root holding only
    /// that is reported unusable (by the message the runtime listing prints), and
    /// one more entry makes it ready. A root that is gone is reported by the
    /// earlier "install root is missing" check, never reaching the payload rule.
    #[test]
    fn tarball_validation_follows_the_shared_payload_rule() {
        let (root, _) = test_paths("tarball-shared-payload-rule");
        let install_root = root.join("tarball-runtime");
        fs::create_dir_all(&install_root).unwrap();
        fs::write(install_root.join(".rocm-cli-runtime.json"), b"{}").unwrap();
        let manifest = InstalledRuntimeManifest {
            format: "tarball".to_owned(),
            install_root: install_root.clone(),
            ..test_runtime_manifest(
                "release-tarball-gfx120X-all-7-13-0",
                "therock-release:gfx120X-all",
                1,
            )
        };

        let error = validate_runtime_manifest_for_activation(&manifest)
            .expect_err("a root with only the dot-marker has no payload")
            .to_string();
        assert_eq!(
            error,
            format!(
                "tarball runtime install root has no payload files: {}",
                install_root.display()
            )
        );
        assert_eq!(
            runtime_usability_status(&manifest),
            format!("unusable ({error})")
        );

        fs::write(install_root.join("rocm-install-file"), b"x").unwrap();
        validate_runtime_manifest_for_activation(&manifest)
            .expect("one entry beside the marker is a payload");
        assert_eq!(runtime_usability_status(&manifest), "ready");

        fs::remove_dir_all(&install_root).unwrap();
        let error = validate_runtime_manifest_for_activation(&manifest)
            .expect_err("a missing root is unusable")
            .to_string();
        assert_eq!(
            error,
            format!("install root is missing: {}", install_root.display())
        );
        fs::remove_dir_all(root).ok();
    }
}
