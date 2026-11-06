// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, download_file_to_path, normalize_runtime_path_for_host, runtime_is_windows,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::archive_cache::ensure_cached_archive;
use crate::archive_extract::{copy_tree, extract_archive};
use crate::{ENGINE_NAME, LLAMACPP_RECIPE, ROCM_BACKEND_NAME, current_unix_millis};

/// Records the extracted embeddable version inside the runtime tree itself
/// (not the CLI's own manifest) so the version check survives even when the
/// manifest lives under a data dir that is not shared with the runtime tree
/// (e.g. each e2e scenario's isolated data dir vs. a shared runtime tree).
/// Living inside `runtime_dir` rather than beside it also means a wipe of the
/// tree removes the marker in the same operation — "marker present implies
/// extraction completed" holds by construction. Without this marker,
/// [`prepare_embeddable`] can never prove the shared tree is already current,
/// wipes it, and re-extracts on every call — destroying any llama.cpp backend
/// installed into it in the process.
const RUNTIME_VERSION_MARKER: &str = "runtime-version.txt";

/// How deep the extracted tree is searched for the embeddable root before the
/// search gives up. Official archives place the root at the top level, so this
/// only bounds the work spent looking; exceeding it is not an error.
const MAX_EMBEDDABLE_SEARCH_DEPTH: usize = 4;

/// Digests of the pinned embeddable archives. Unlike the archive names and
/// URLs, these cannot be derived from the version in `runtime-deps.toml`: each
/// release has its own digest, so bumping the pin means recording the new pair
/// here as well.
const EMBEDDABLE_WINDOWS_SHA256: &str =
    "50a133bbc35c4f3f8971eafef2c9fe56c4bbbfb0f1032bf8728324d8d8c8a0e1";
const EMBEDDABLE_LINUX_SHA256: &str =
    "bdfd3c3e5d6eda5101c8a32f36e6dd9236ceb9ab2eb66734c32628e6e86e18ac";

/// Embeddable asset token and archive extension for the host this adapter runs
/// on. Only `windows-x64` and `ubuntu-x64` are wired up here; the release also
/// publishes `macos-arm64` and `ubuntu-arm64`, which this adapter never selects.
const fn embeddable_os_arch() -> (&'static str, &'static str) {
    if runtime_is_windows() {
        ("windows-x64", "zip")
    } else {
        ("ubuntu-x64", "tar.gz")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LemonadeInstallManifest {
    pub(crate) env_id: String,
    pub(crate) version: String,
    pub(crate) runtime_dir: PathBuf,
    pub(crate) lemond: PathBuf,
    pub(crate) lemonade: PathBuf,
    pub(crate) backend_recipe: String,
    pub(crate) backend_name: String,
    pub(crate) installed_at_unix_ms: u128,
}

/// Whether the embeddable archive must be extracted over the runtime tree.
///
/// Pure so the upgrade case is testable without a real archive. `installed`
/// is the version recorded for the tree already on disk, if any.
fn needs_extraction(
    reinstall: bool,
    installed: Option<&str>,
    wanted: &str,
    server_present: bool,
) -> bool {
    reinstall || !server_present || installed != Some(wanted)
}

/// The Lemonade runtime version last extracted into `runtime_dir`, if any.
fn installed_runtime_version(runtime_dir: &Path) -> Option<String> {
    fs::read_to_string(runtime_dir.join(RUNTIME_VERSION_MARKER))
        .ok()
        .map(|s| s.trim().to_owned())
}

/// Record `version` as installed into `runtime_dir`. Marker lives inside
/// `runtime_dir` (not beside it) so `remove_dir_all(runtime_dir)` clears it as
/// part of the same wipe that removes the tree it describes — "marker present
/// implies extraction completed" holds by construction, with no separate
/// bookkeeping to keep in sync.
fn record_runtime_version(runtime_dir: &Path, version: &str) -> Result<()> {
    fs::write(runtime_dir.join(RUNTIME_VERSION_MARKER), version).with_context(|| {
        format!(
            "failed to record installed Lemonade runtime version under {}",
            runtime_dir.display()
        )
    })
}

pub(crate) fn prepare_embeddable(
    paths: &AppPaths,
    env_root: Option<&Path>,
    reinstall: bool,
) -> Result<LemonadeInstallManifest> {
    prepare_embeddable_with(
        paths,
        env_root,
        reinstall,
        rocm_deps::LEMONADE_VERSION,
        embeddable_archive_sha256(),
        download_file,
    )
}

/// Same as [`prepare_embeddable`], but with the archive version, its expected
/// sha256, and the download step all injectable — the seam a test uses to
/// exercise the real decision of which source `installed_version` trusts,
/// rather than reimplementing that decision inline.
fn prepare_embeddable_with<F>(
    paths: &AppPaths,
    env_root: Option<&Path>,
    reinstall: bool,
    version: &str,
    expected_sha256: &str,
    download: F,
) -> Result<LemonadeInstallManifest>
where
    F: FnMut(&str, &Path) -> Result<()>,
{
    let root = lemonade_root(paths, env_root);
    let version = version.to_owned();
    let archive_name = embeddable_archive_name(&version);
    let archive_url = embeddable_url(&version);
    let downloads = root.join("downloads");
    let archive = downloads.join(&archive_name);
    fs::create_dir_all(&downloads)?;
    // Held until the archive bytes have been extracted and copied out: the
    // verified bytes and the extracted bytes must be the same read.
    let archive_guard = ensure_cached_archive(&archive_url, &archive, expected_sha256, download)?;
    let runtime_dir = runtime_dir_in(&root);
    // The runtime directory is not version-scoped, so a bump downloads a new
    // archive into a tree that already holds `lemond` from the previous
    // version. Without comparing versions the extraction would be skipped and
    // the old binaries reported as the new version.
    //
    // The version is read from a marker inside `runtime_dir` (the same shared
    // tree it describes), not from the CLI's own manifest: the manifest lives
    // under `paths` (the per-invocation data dir), which is not guaranteed to
    // be the same data dir that last extracted into this `runtime_dir` (e.g.
    // each e2e scenario gets an isolated data dir but shares one runtime
    // tree). Sourcing the check from the isolated manifest would report
    // "unknown version" on every call, wipe the shared tree, and re-extract
    // over it — destroying any llama.cpp backend already installed there.
    let installed_version = installed_runtime_version(&runtime_dir);
    if needs_extraction(
        reinstall,
        installed_version.as_deref(),
        &version,
        lemond_path_in(&runtime_dir).is_file(),
    ) {
        if runtime_dir.exists() {
            fs::remove_dir_all(&runtime_dir)
                .with_context(|| format!("failed to clear {}", runtime_dir.display()))?;
        }
        let extract_root = root
            .join("extract")
            .join(format!("{}", current_unix_millis()));
        fs::create_dir_all(&extract_root)?;
        extract_archive(&archive, &extract_root)?;
        let embeddable_root = find_embeddable_root(&extract_root)?;
        copy_tree(&embeddable_root, &runtime_dir)?;
        fs::remove_dir_all(&extract_root).ok();
        record_runtime_version(&runtime_dir, &version)?;
    }
    drop(archive_guard);
    let lemond = lemond_path_in(&runtime_dir);
    let lemonade = lemonade_path_in(&runtime_dir);
    if !lemond.is_file() || !lemonade.is_file() {
        bail!(
            "Lemonade embeddable extraction did not produce lemonade/lemond binaries in {}",
            runtime_dir.display()
        );
    }
    Ok(LemonadeInstallManifest {
        env_id: format!("lemonade-embeddable-{version}"),
        version,
        runtime_dir,
        lemond,
        lemonade,
        backend_recipe: LLAMACPP_RECIPE.to_owned(),
        // Resolved during install once Lemonade reports per-GPU backend support.
        backend_name: ROCM_BACKEND_NAME.to_owned(),
        installed_at_unix_ms: current_unix_millis(),
    })
}

fn manifest_path(paths: &AppPaths) -> PathBuf {
    paths.engine_manifests_dir(ENGINE_NAME).join("runtime.json")
}

fn lemonade_root(paths: &AppPaths, env_root: Option<&Path>) -> PathBuf {
    env_root.map_or_else(
        || paths.engine_dir(ENGINE_NAME),
        |root| normalize_runtime_path_for_host(root).join(ENGINE_NAME),
    )
}

fn runtime_dir_in(root: &Path) -> PathBuf {
    root.join("runtime")
}

pub(crate) fn read_manifest(paths: &AppPaths) -> Result<LemonadeInstallManifest> {
    let path = manifest_path(paths);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

pub(crate) fn write_manifest(paths: &AppPaths, manifest: &LemonadeInstallManifest) -> Result<()> {
    let path = manifest_path(paths);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct LemonadeRuntime {
    pub(crate) manifest: LemonadeInstallManifest,
}

pub(crate) fn resolve_runtime() -> Result<LemonadeRuntime> {
    let paths = AppPaths::discover()?;
    let manifest = read_manifest(&paths)?;
    if !manifest.lemond.is_file() {
        bail!(
            "Lemonade runtime is missing {}; run `rocm engines install lemonade`",
            manifest.lemond.display()
        );
    }
    Ok(LemonadeRuntime { manifest })
}

/// Put `resources/backend_versions.json` back to what the cached embeddable
/// archive ships. Returns `None` when the file already held the packaged
/// values, and otherwise a [`PinFileBackup`] that puts the previous bytes back.
///
/// That file is the one piece of the extracted tree rocm-cli itself edits (ROCm
/// backend alignment pins a ROCm version and a llama.cpp build into it), and
/// the edit outlives the process: the runtime tree is shared and is only
/// re-extracted on a version change or `--reinstall`. A pin kept from an
/// earlier successful alignment can therefore name assets that have since
/// been withdrawn, and nothing but this restore (or a full `--reinstall`)
/// clears it. Only that one file is rewritten, so backends already installed
/// in the tree, and any server running from it, are left alone.
///
/// This runs only after an install has already failed, so it never downloads:
/// the archive is read from the cache `prepare_embeddable_with` filled earlier
/// in the same install, and its digest is verified before any byte of it is
/// trusted. When the cached archive is gone or does not verify, this fails and
/// the caller reports the original install error.
///
/// The pin file is read only after the archive work is done, immediately
/// before the rewrite, so a pin another process wrote in the meantime is
/// neither overwritten blindly nor missing from the backup.
pub(crate) fn restore_packaged_backend_versions(
    manifest: &LemonadeInstallManifest,
    resource: &str,
) -> Result<Option<PinFileBackup>> {
    restore_packaged_backend_versions_with(manifest, resource, embeddable_archive_sha256(), || {})
}

/// [`restore_packaged_backend_versions`] with the archive digest injectable.
/// `after_archive` runs once the archive work is done and before the pin file
/// is read; tests use it to stand in for a writer that gets in between.
fn restore_packaged_backend_versions_with(
    manifest: &LemonadeInstallManifest,
    resource: &str,
    expected_sha256: &str,
    after_archive: impl FnOnce(),
) -> Result<Option<PinFileBackup>> {
    let root = manifest
        .runtime_dir
        .parent()
        .context("Lemonade runtime directory has no parent to hold its archive cache")?;
    let archive = root
        .join("downloads")
        .join(embeddable_archive_name(&manifest.version));
    let target = manifest.runtime_dir.join(resource);

    // Checked first so that a missing cache creates nothing at all.
    if !archive.is_file() {
        bail!(
            "the embeddable archive is not cached at {}; not downloading it while recovering from a failed install",
            archive.display()
        );
    }
    // The download step can only fail: a cached archive that does not verify is
    // dropped by `ensure_cached_archive` and is not replaced from the network.
    let guard = ensure_cached_archive("", &archive, expected_sha256, |_, _| {
        bail!(
            "the cached embeddable archive {} does not match its digest; not downloading it while recovering from a failed install",
            archive.display()
        )
    })?;
    let extract_root = root
        .join("extract")
        .join(format!("{}-packaged-pins", current_unix_millis()));
    fs::create_dir_all(&extract_root)?;
    let packaged = (|| -> Result<Vec<u8>> {
        extract_archive(&archive, &extract_root)?;
        let embeddable_root = find_embeddable_root(&extract_root)?;
        let packaged_path = embeddable_root.join(resource);
        fs::read(&packaged_path).with_context(|| format!("embeddable archive has no {resource}"))
    })();
    fs::remove_dir_all(&extract_root).ok();
    drop(guard);
    let packaged = packaged?;
    after_archive();

    // Read only now, after the archive work above, and used for both the
    // computation and the backup: a pin file written by another process
    // while the archive was being verified and extracted is what gets
    // restored over and backed up, not a stale earlier read.
    // Only "no such file" means there was nothing to keep. Any other failure
    // (a sharing violation from a concurrent writer, a permission error) says
    // nothing about what the file holds, and treating it as absent would make
    // the backup delete a real pin file if the retry fails. Like the other
    // cache-only skips above, it fails the restore, so nothing is rewritten and
    // the caller reports the original install error.
    let current = match fs::read(&target) {
        Ok(bytes) => Some(bytes),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "could not read {} to back it up; leaving it as it is",
                    target.display()
                )
            });
        }
    };

    // Only the two pins backend alignment writes are restored, so an edit to any
    // other entry of the file survives and a file that already holds the packaged
    // values for those two is not drift. Compared as JSON, not bytes: an edit
    // re-serializes the file, so the same values can sit in different bytes.
    let packaged_value: serde_json::Value = serde_json::from_slice(&packaged)
        .with_context(|| format!("{resource} in the embeddable archive is not valid JSON"))?;
    let restored = match current
        .as_deref()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok())
    {
        Some(mut value) => {
            if alignment_pins(&value) == alignment_pins(&packaged_value) {
                return Ok(None);
            }
            for (section, key) in ALIGNMENT_PIN_KEYS {
                set_or_remove_pin(&mut value, section, key, &packaged_value);
            }
            serde_json::to_vec_pretty(&value)?
        }
        // Missing (or not JSON): there is nothing of the user's to keep.
        None => packaged,
    };
    rocm_core::write_atomic(&target, &restored, "pins")
        .with_context(|| format!("failed to restore {}", target.display()))?;
    Ok(Some(PinFileBackup {
        target,
        previous: current,
    }))
}

/// The two entries of `resources/backend_versions.json` that backend alignment
/// writes: the ROCm version a backend is built for and the llama.cpp build tag.
const ALIGNMENT_PIN_KEYS: [(&str, &str); 2] = [("therock", "version"), ("llamacpp", "rocm-stable")];

fn alignment_pins(value: &serde_json::Value) -> [Option<&serde_json::Value>; 2] {
    ALIGNMENT_PIN_KEYS.map(|(section, key)| value.get(section).and_then(|s| s.get(key)))
}

/// Gives `section.key` of `value` the value it has in `packaged`, or removes it
/// when `packaged` has none, leaving every other entry as it was.
fn set_or_remove_pin(
    value: &mut serde_json::Value,
    section: &str,
    key: &str,
    packaged: &serde_json::Value,
) {
    let packaged_pin = packaged.get(section).and_then(|s| s.get(key));
    match packaged_pin {
        Some(pin) => {
            if !value.is_object() {
                *value = serde_json::json!({});
            }
            if let Some(object) = value.as_object_mut() {
                let section_value = object
                    .entry(section.to_owned())
                    .or_insert_with(|| serde_json::json!({}));
                if !section_value.is_object() {
                    *section_value = serde_json::json!({});
                }
                if let Some(section_object) = section_value.as_object_mut() {
                    section_object.insert(key.to_owned(), pin.clone());
                }
            }
        }
        None => {
            if let Some(section_object) = value.get_mut(section).and_then(|s| s.as_object_mut()) {
                section_object.remove(key);
            }
        }
    }
}

/// What `resources/backend_versions.json` held before
/// [`restore_packaged_backend_versions`] rewrote it. Written back through
/// `rocm_core::write_atomic`, like the restore itself.
///
/// A restore is attempted after any failed install, including a transient
/// network or disk error, so a restore whose retry fails too must not leave a
/// pin that was fine discarded.
#[derive(Debug)]
pub(crate) struct PinFileBackup {
    target: PathBuf,
    /// `None` when the file did not exist, so putting it back removes it.
    previous: Option<Vec<u8>>,
}

impl PinFileBackup {
    /// For tests that stand in for the restore step; the real restore builds
    /// the struct itself.
    #[cfg(test)]
    pub(crate) const fn new(target: PathBuf, previous: Option<Vec<u8>>) -> Self {
        Self { target, previous }
    }

    /// Writes the previous bytes back, or removes the file if there were none.
    pub(crate) fn put_back(self) -> Result<()> {
        match self.previous {
            Some(bytes) => rocm_core::write_atomic(&self.target, &bytes, "pins")
                .with_context(|| format!("failed to put back {}", self.target.display())),
            None => match fs::remove_file(&self.target) {
                Err(err) if err.kind() != std::io::ErrorKind::NotFound => {
                    Err(err).with_context(|| format!("failed to remove {}", self.target.display()))
                }
                _ => Ok(()),
            },
        }
    }
}

fn lemond_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemond"))
}

fn lemonade_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemonade"))
}

pub(crate) fn platform_binary_name(name: &str) -> String {
    if runtime_is_windows() {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

fn embeddable_archive_name(version: &str) -> String {
    let (os_arch, extension) = embeddable_os_arch();
    rocm_deps::lemonade_archive_name(version, os_arch, extension)
}

fn embeddable_url(version: &str) -> String {
    rocm_deps::lemonade_download_url(version, &embeddable_archive_name(version))
}

const fn embeddable_archive_sha256() -> &'static str {
    if runtime_is_windows() {
        EMBEDDABLE_WINDOWS_SHA256
    } else {
        EMBEDDABLE_LINUX_SHA256
    }
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    download_file_to_path(url, destination, Duration::from_mins(15))
}

fn find_embeddable_root(extract_root: &Path) -> Result<PathBuf> {
    let extract_root = fs::canonicalize(extract_root)
        .with_context(|| format!("failed to canonicalize {}", extract_root.display()))?;
    let mut candidates = Vec::new();
    collect_embeddable_roots(&extract_root, &extract_root, &mut candidates, 0)?;
    candidates.into_iter().next().with_context(|| {
        format!(
            "no Lemonade embeddable root found in {}",
            extract_root.display()
        )
    })
}

fn collect_embeddable_roots(
    extract_root: &Path,
    path: &Path,
    candidates: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<()> {
    if depth > MAX_EMBEDDABLE_SEARCH_DEPTH {
        return Ok(());
    }
    let metadata = fs::symlink_metadata(path)?;
    // See the link policy on `archive_extract::extract_archive`.
    if metadata.file_type().is_symlink() {
        bail!("archive contains symlink at {}", path.display());
    }
    if !metadata.file_type().is_dir() {
        bail!("archive contains non-directory entry at {}", path.display());
    }
    let canonical_path = fs::canonicalize(path)?;
    if !canonical_path.starts_with(extract_root) {
        bail!("archive entry escapes extraction root: {}", path.display());
    }

    let lemond = lemond_path_in(path);
    let lemonade = lemonade_path_in(path);
    let lemond_type = fs::symlink_metadata(&lemond)
        .ok()
        .map(|metadata| metadata.file_type());
    let lemonade_type = fs::symlink_metadata(&lemonade)
        .ok()
        .map(|metadata| metadata.file_type());
    if lemond_type.as_ref().is_some_and(fs::FileType::is_symlink)
        || lemonade_type.as_ref().is_some_and(fs::FileType::is_symlink)
    {
        bail!(
            "archive contains symlinked Lemonade binary in {}",
            path.display()
        );
    }
    if lemond_type.as_ref().is_some_and(fs::FileType::is_file)
        && lemonade_type.as_ref().is_some_and(fs::FileType::is_file)
    {
        candidates.push(canonical_path);
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("archive contains symlink at {}", entry.path().display());
        }
        if file_type.is_dir() {
            collect_embeddable_roots(extract_root, &entry.path(), candidates, depth + 1)?;
        } else if !file_type.is_file() {
            bail!(
                "archive contains non-regular entry at {}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    #[cfg(windows)]
    use std::process::{Command as ProcessCommand, Stdio};

    #[test]
    fn embeddable_package_matches_runtime_os() {
        let version = rocm_deps::LEMONADE_VERSION;
        let (os_arch, extension) = embeddable_os_arch();
        let suffix = format!("{os_arch}.{extension}");
        assert!(embeddable_archive_name(version).ends_with(&suffix));
        assert!(embeddable_url(version).ends_with(&suffix));
        if runtime_is_windows() {
            assert_eq!(suffix, "windows-x64.zip");
            assert_eq!(platform_binary_name("lemond"), "lemond.exe");
        } else {
            assert_eq!(suffix, "ubuntu-x64.tar.gz");
            assert_eq!(platform_binary_name("lemond"), "lemond");
        }
    }

    #[test]
    fn valid_embeddable_tree_is_found_and_copied() {
        let dir = scratch_dir("valid-embeddable-tree");
        let extract = dir.join("extract");
        let package = extract.join("package");
        fs::create_dir_all(package.join("lib")).unwrap();
        fs::write(lemond_path_in(&package), b"lemond").unwrap();
        fs::write(lemonade_path_in(&package), b"lemonade").unwrap();
        fs::write(package.join("lib").join("backend"), b"backend").unwrap();

        let found = find_embeddable_root(&extract).unwrap();
        assert_eq!(found, fs::canonicalize(&package).unwrap());
        let destination = dir.join("runtime");
        copy_tree(&found, &destination).unwrap();
        assert_eq!(
            fs::read(destination.join("lib").join("backend")).unwrap(),
            b"backend"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deeply_nested_embeddable_tree_is_copied() {
        let dir = scratch_dir("deep-embeddable-tree");
        let extract = dir.join("extract");
        let package = extract.join("package");
        let nested = package
            .join("lib")
            .join("site-packages")
            .join("backend")
            .join("kernels")
            .join("gfx");
        fs::create_dir_all(&nested).unwrap();
        fs::write(lemond_path_in(&package), b"lemond").unwrap();
        fs::write(lemonade_path_in(&package), b"lemonade").unwrap();
        fs::write(nested.join("kernel.so"), b"kernel").unwrap();

        let found = find_embeddable_root(&extract).unwrap();
        let destination = dir.join("runtime");
        copy_tree(&found, &destination).unwrap();
        assert_eq!(
            fs::read(
                destination
                    .join("lib")
                    .join("site-packages")
                    .join("backend")
                    .join("kernels")
                    .join("gfx")
                    .join("kernel.so")
            )
            .unwrap(),
            b"kernel"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn embeddable_tree_rejects_external_file_symlink() {
        use std::os::unix::fs::symlink;

        let dir = scratch_dir("external-file-symlink");
        let extract = dir.join("extract");
        let package = extract.join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(lemond_path_in(&package), b"lemond").unwrap();
        fs::write(lemonade_path_in(&package), b"lemonade").unwrap();
        let outside = dir.join("outside-file");
        fs::write(&outside, b"secret").unwrap();
        symlink(&outside, package.join("linked-file")).unwrap();

        let found = find_embeddable_root(&extract).unwrap();
        let error = copy_tree(&found, &dir.join("runtime")).unwrap_err();
        assert!(error.to_string().contains("symlink"));
        fs::remove_dir_all(&dir).ok();
    }

    /// Windows counterpart to the Unix symlink rejection tests.
    ///
    /// Creating a symlink on Windows needs privilege, but `mklink /J` creates a
    /// directory junction unelevated, and Rust reports junctions as symlinks
    /// (the reparse tag carries the name-surrogate bit) — so this exercises the
    /// same rejection branch on the platform where the zip artifact is
    /// first-class.
    #[cfg(windows)]
    #[test]
    fn embeddable_tree_rejects_external_directory_junction() {
        let dir = scratch_dir("external-directory-junction");
        let extract = dir.join("extract");
        let package = extract.join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(lemond_path_in(&package), b"lemond").unwrap();
        fs::write(lemonade_path_in(&package), b"lemonade").unwrap();
        let outside = dir.join("outside-directory");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), b"secret").unwrap();

        let junction = package.join("linked-directory");
        let status = ProcessCommand::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(&outside)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "mklink /J failed");
        assert!(
            fs::symlink_metadata(&junction)
                .unwrap()
                .file_type()
                .is_symlink(),
            "junction was not reported as a symlink"
        );

        let found = find_embeddable_root(&extract).unwrap();
        let error = copy_tree(&found, &dir.join("runtime")).unwrap_err();
        assert!(error.to_string().contains("symlink"), "{error}");
        assert!(!dir.join("runtime").join("linked-directory").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn embeddable_tree_rejects_external_directory_symlink() {
        use std::os::unix::fs::symlink;

        let dir = scratch_dir("external-directory-symlink");
        let extract = dir.join("extract");
        fs::create_dir_all(&extract).unwrap();
        let outside = dir.join("outside-package");
        fs::create_dir_all(&outside).unwrap();
        fs::write(lemond_path_in(&outside), b"lemond").unwrap();
        fs::write(lemonade_path_in(&outside), b"lemonade").unwrap();
        symlink(&outside, extract.join("linked-package")).unwrap();

        let error = find_embeddable_root(&extract).unwrap_err();
        assert!(error.to_string().contains("symlink"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_version_change_forces_re_extraction_over_the_existing_runtime() {
        let pin = rocm_deps::LEMONADE_VERSION;
        // Same version already unpacked: nothing to do.
        assert!(!needs_extraction(false, Some(pin), pin, true));
        // A bump must not be silently skipped just because `lemond` exists.
        assert!(needs_extraction(false, Some("10.10.0"), pin, true));
        // Nothing recorded, nothing unpacked, or an explicit reinstall.
        assert!(needs_extraction(false, None, pin, true));
        assert!(needs_extraction(false, Some(pin), pin, false));
        assert!(needs_extraction(true, Some(pin), pin, true));
    }

    #[test]
    fn runtime_version_marker_round_trips_through_needs_extraction() {
        // Unit coverage for the helpers themselves and their wiring into
        // `needs_extraction` — NOT a regression test for the bug this marker
        // fixes (see `prepare_embeddable_trusts_the_runtime_tree_over_an_absent_manifest`
        // below for that): this never calls `prepare_embeddable`, so it can't
        // constrain which source it reads `installed_version` from.
        let dir = tempfile::tempdir().unwrap();
        let runtime_dir = dir.path();

        assert_eq!(installed_runtime_version(runtime_dir), None);

        record_runtime_version(runtime_dir, "7.13.0").unwrap();
        let installed = installed_runtime_version(runtime_dir);
        assert_eq!(installed.as_deref(), Some("7.13.0"));
        assert!(!needs_extraction(
            false,
            installed.as_deref(),
            "7.13.0",
            true
        ));
        assert!(needs_extraction(
            false,
            installed.as_deref(),
            "7.14.0",
            true
        ));
    }

    #[test]
    fn prepare_embeddable_trusts_the_runtime_tree_over_an_absent_manifest() {
        // The actual regression: `prepare_embeddable` must source
        // `installed_version` from the marker inside `runtime_dir`, not from
        // `paths`' manifest. An isolated caller (fresh `AppPaths`, no
        // manifest on disk) sharing a runtime tree that a *different* caller
        // already extracted must recognize it as current and skip
        // re-extraction. Reverting to a manifest read (the original bug)
        // makes every isolated caller miss, wipe, and re-extract — destroying
        // any llama.cpp backend already installed there. Calling
        // `prepare_embeddable_with` (the real function, seamed for an
        // injectable download step) rather than reimplementing its decision
        // inline is what makes this test fail under that reversion.
        let env_root = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: env_root.path().join("config"),
            data_dir: env_root.path().join("data"), // no manifest ever written here
            cache_dir: env_root.path().join("cache"),
        };
        let version = "7.13.0";
        let root = lemonade_root(&paths, Some(env_root.path()));
        let runtime_dir = runtime_dir_in(&root);
        fs::create_dir_all(&runtime_dir).unwrap();
        fs::write(lemond_path_in(&runtime_dir), b"fake lemond").unwrap();
        fs::write(lemonade_path_in(&runtime_dir), b"fake lemonade").unwrap();
        record_runtime_version(&runtime_dir, version).unwrap();
        let sentinel = runtime_dir.join("llamacpp-rocm-backend.sentinel");
        fs::write(&sentinel, b"a backend a prior scenario installed").unwrap();

        let archive_bytes = b"never parsed unless extraction wrongly triggers";
        let expected_sha256 = format!("{:x}", Sha256::digest(archive_bytes));

        let manifest = prepare_embeddable_with(
            &paths,
            Some(env_root.path()),
            false,
            version,
            &expected_sha256,
            |_, destination| {
                fs::write(destination, archive_bytes)?;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(manifest.version, version);
        assert!(
            sentinel.is_file(),
            "prepare_embeddable wiped the shared runtime tree instead of recognizing it as current"
        );
    }

    #[test]
    fn prepare_embeddable_records_the_version_after_a_real_extraction() {
        // The write-side sibling of the test above: an isolated caller
        // extracting into an empty `runtime_dir` for the first time must
        // leave a marker behind, or the *next* isolated caller sharing this
        // tree sees no marker, wipes, and re-extracts on every call — the
        // same defect the read-side regression guards against, from the
        // write side instead. Dropping `record_runtime_version`'s call site
        // in `prepare_embeddable` fails this test.
        let env_root = tempfile::tempdir().unwrap();
        let paths = AppPaths {
            config_dir: env_root.path().join("config"),
            data_dir: env_root.path().join("data"),
            cache_dir: env_root.path().join("cache"),
        };
        let version = "7.13.0";

        // A real, well-formed archive (unlike the stub above): this test
        // exercises the extraction branch, so `extract_archive` must succeed.
        // `extract_archive` dispatches on the destination's extension, which
        // `prepare_embeddable_with` derives from the current platform (see
        // `embeddable_os_arch`) — so the bytes built here must be in that
        // same format, not always a tar.gz.
        let (_, extension) = embeddable_os_arch();
        let archive_dir = tempfile::tempdir().unwrap();
        let archive = archive_dir.path().join(format!("embeddable.{extension}"));
        let entry_names = [
            platform_binary_name("lemond"),
            platform_binary_name("lemonade"),
        ];
        if runtime_is_windows() {
            let file = fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            for name in &entry_names {
                writer
                    .start_file(format!("embeddable/bin/{name}"), options)
                    .unwrap();
                writer.write_all(b"hello").unwrap();
            }
            writer.finish().unwrap();
        } else {
            let file = fs::File::create(&archive).unwrap();
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
            let mut builder = tar::Builder::new(encoder);
            for name in &entry_names {
                let mut header = tar::Header::new_ustar();
                header.set_size(5);
                header.set_mode(0o755);
                header.set_entry_type(tar::EntryType::Regular);
                header.set_path(format!("embeddable/bin/{name}")).unwrap();
                header.set_cksum();
                builder.append(&header, &b"hello"[..]).unwrap();
            }
            builder.into_inner().unwrap().finish().unwrap();
        }
        let expected_sha256 = format!("{:x}", Sha256::digest(fs::read(&archive).unwrap()));

        let root = lemonade_root(&paths, Some(env_root.path()));
        let runtime_dir = runtime_dir_in(&root);
        assert_eq!(installed_runtime_version(&runtime_dir), None);

        prepare_embeddable_with(
            &paths,
            Some(env_root.path()),
            false,
            version,
            &expected_sha256,
            |_, destination| {
                fs::copy(&archive, destination)?;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            installed_runtime_version(&runtime_dir).as_deref(),
            Some(version),
            "prepare_embeddable extracted the archive but never recorded the version"
        );
    }

    /// Writes an embeddable archive in the host's format holding the two
    /// binaries and a packaged `resources/backend_versions.json`.
    fn write_embeddable_with_pins(archive: &Path, pins: &str) {
        let files = [
            (platform_binary_name("lemond"), b"lemond".to_vec()),
            (platform_binary_name("lemonade"), b"lemonade".to_vec()),
            (
                "resources/backend_versions.json".to_owned(),
                pins.as_bytes().to_vec(),
            ),
        ];
        let file = fs::File::create(archive).unwrap();
        if runtime_is_windows() {
            let mut writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            for (name, body) in &files {
                writer
                    .start_file(format!("embeddable/{name}"), options)
                    .unwrap();
                writer.write_all(body).unwrap();
            }
            writer.finish().unwrap();
        } else {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
            let mut builder = tar::Builder::new(encoder);
            for (name, body) in &files {
                let mut header = tar::Header::new_ustar();
                header.set_size(body.len() as u64);
                header.set_mode(0o755);
                header.set_entry_type(tar::EntryType::Regular);
                header.set_path(format!("embeddable/{name}")).unwrap();
                header.set_cksum();
                builder.append(&header, body.as_slice()).unwrap();
            }
            builder.into_inner().unwrap().finish().unwrap();
        }
    }

    #[test]
    fn restore_packaged_backend_versions_replaces_only_a_pin_file_that_drifted() {
        let dir = scratch_dir("restore-packaged-pins");
        let root = dir.join("lemonade");
        let runtime_dir = runtime_dir_in(&root);
        fs::create_dir_all(runtime_dir.join("resources")).unwrap();
        let pins = runtime_dir.join("resources/backend_versions.json");
        let backend = runtime_dir.join("backends").join("rocm");
        fs::create_dir_all(&backend).unwrap();
        fs::write(backend.join("llama-server"), b"installed backend").unwrap();

        let packaged = r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}"#;
        let source = dir.join("source-archive");
        write_embeddable_with_pins(&source, packaged);
        let sha256 = format!("{:x}", Sha256::digest(fs::read(&source).unwrap()));
        let manifest = LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "11.5.1".to_owned(),
            runtime_dir: runtime_dir.clone(),
            lemond: lemond_path_in(&runtime_dir),
            lemonade: lemonade_path_in(&runtime_dir),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        };
        cache_archive(&root, &source);
        let restore = |manifest: &LemonadeInstallManifest| {
            restore_packaged_backend_versions_with(
                manifest,
                "resources/backend_versions.json",
                &sha256,
                || {},
            )
        };

        // The state an earlier alignment left behind: the active ROCm version
        // and a newer llama.cpp build pinned in place of the packaged ones.
        fs::write(
            &pins,
            r#"{"therock":{"version":"7.14.1"},"llamacpp":{"rocm-stable":"b11783"}}"#,
        )
        .unwrap();
        let backup = restore(&manifest)
            .unwrap()
            .expect("a drifted pin file must be restored");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&pins).unwrap()).unwrap(),
            serde_json::from_str::<serde_json::Value>(packaged).unwrap()
        );
        assert_no_pin_temp_files(&runtime_dir.join("resources"));
        assert_eq!(
            fs::read(backend.join("llama-server")).unwrap(),
            b"installed backend",
            "restoring the pins must not touch backends already installed in the tree"
        );
        assert!(
            fs::read_dir(root.join("extract")).unwrap().next().is_none(),
            "the scratch extraction must be cleaned up"
        );

        // Re-serialized but equal content is not drift: nothing to restore.
        fs::write(
            &pins,
            serde_json::to_vec_pretty(
                &serde_json::from_str::<serde_json::Value>(packaged).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(restore(&manifest).unwrap().is_none());

        // The backup puts the drifted bytes back exactly, so a restore whose
        // retry fails does not discard the pin that was there.
        backup.put_back().unwrap();
        assert_eq!(
            fs::read_to_string(&pins).unwrap(),
            r#"{"therock":{"version":"7.14.1"},"llamacpp":{"rocm-stable":"b11783"}}"#
        );
        assert_no_pin_temp_files(&runtime_dir.join("resources"));
        fs::remove_dir_all(&dir).ok();
    }

    /// Puts `source` where `prepare_embeddable_with` leaves the verified
    /// archive of the `11.5.1` test manifests: `<root>/downloads/<name>`.
    fn cache_archive(root: &Path, source: &Path) {
        let downloads = root.join("downloads");
        fs::create_dir_all(&downloads).unwrap();
        fs::copy(source, downloads.join(embeddable_archive_name("11.5.1"))).unwrap();
    }

    fn assert_no_pin_temp_files(dir: &Path) {
        let leftovers: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "the atomic write must not leave a temp file behind: {leftovers:?}"
        );
    }

    /// Runs the restore against a fake embeddable archive that ships
    /// `{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}`,
    /// with `current` as the pin file already on disk. Returns what the restore
    /// returned and the pin file's content afterwards.
    fn restore_over(tag: &str, current: &str) -> (Option<PinFileBackup>, String) {
        restore_over_packaged(
            tag,
            r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}"#,
            current,
        )
    }

    /// [`restore_over`] with the pin file the archive ships given too.
    fn restore_over_packaged(
        tag: &str,
        packaged: &str,
        current: &str,
    ) -> (Option<PinFileBackup>, String) {
        let dir = scratch_dir(tag);
        let root = dir.join("lemonade");
        let runtime_dir = runtime_dir_in(&root);
        fs::create_dir_all(runtime_dir.join("resources")).unwrap();
        let pins = runtime_dir.join("resources/backend_versions.json");
        fs::write(&pins, current).unwrap();
        let source = dir.join("source-archive");
        write_embeddable_with_pins(&source, packaged);
        let sha256 = format!("{:x}", Sha256::digest(fs::read(&source).unwrap()));
        let manifest = LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "11.5.1".to_owned(),
            runtime_dir: runtime_dir.clone(),
            lemond: lemond_path_in(&runtime_dir),
            lemonade: lemonade_path_in(&runtime_dir),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        };
        cache_archive(&root, &source);
        let backup = restore_packaged_backend_versions_with(
            &manifest,
            "resources/backend_versions.json",
            &sha256,
            || {},
        )
        .unwrap();
        let after = fs::read_to_string(&pins).unwrap();
        // The backup, if any, still has to be able to put the file back.
        if let Some(backup) = &backup {
            assert!(backup.previous.is_some());
        }
        (backup, after)
    }

    #[test]
    fn restore_keeps_entries_a_user_edited_and_only_resets_the_alignment_pins() {
        let current = r#"{"therock":{"version":"7.14.1"},"llamacpp":{"rocm-stable":"b11783","vulkan":"b1111"},"custom":{"x":1}}"#;
        let (backup, after) = restore_over("restore-keeps-user-edits", current);

        let value: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(value["therock"]["version"], "7.13.0");
        assert_eq!(value["llamacpp"]["rocm-stable"], "b9752");
        assert_eq!(value["llamacpp"]["vulkan"], "b1111", "{after}");
        assert_eq!(value["custom"]["x"], 1, "{after}");
        assert!(backup.is_some());
    }

    #[test]
    fn restore_is_a_no_op_when_only_other_entries_differ_from_the_packaged_file() {
        let current = r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752","vulkan":"b1111"}}"#;
        let (backup, after) = restore_over("restore-noop-other-keys", current);

        assert!(backup.is_none(), "the alignment pins already match");
        assert_eq!(after, current, "the file must not be rewritten");
    }

    #[test]
    fn restore_removes_a_pin_the_packaged_file_does_not_have() {
        // A packaged file without `llamacpp.rocm-stable` must not leave the
        // kept alignment pin behind; the user's other entries still survive.
        let packaged = r#"{"therock":{"version":"7.13.0"},"llamacpp":{"vulkan":"b9747"}}"#;
        let current = r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b11783","vulkan":"b1111"}}"#;
        let (backup, after) = restore_over_packaged("restore-removes-pin", packaged, current);

        assert!(backup.is_some(), "the kept rocm-stable pin is drift");
        let value: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert!(value["llamacpp"].get("rocm-stable").is_none(), "{after}");
        assert_eq!(value["llamacpp"]["vulkan"], "b1111", "{after}");
        assert_eq!(value["therock"]["version"], "7.13.0");
    }

    #[test]
    fn restore_happens_when_only_the_rocm_stable_pin_drifted() {
        let current = r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b11783"}}"#;
        let (backup, after) = restore_over("restore-only-rocm-stable", current);

        assert!(backup.is_some(), "a drifted llama.cpp tag alone is drift");
        let value: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(value["llamacpp"]["rocm-stable"], "b9752");
        assert_eq!(value["therock"]["version"], "7.13.0");
    }

    #[test]
    fn restore_replaces_a_pin_file_that_is_not_json_with_the_packaged_one() {
        let (backup, after) = restore_over("restore-not-json", "{ not json");

        assert!(backup.is_some());
        let value: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(value["therock"]["version"], "7.13.0");
    }

    /// A runtime tree whose pin file holds a stale kept pin, and the manifest
    /// for it. The archive cache is left empty.
    fn stale_pin_tree(tag: &str) -> (PathBuf, PathBuf, PathBuf, LemonadeInstallManifest) {
        let dir = scratch_dir(tag);
        let root = dir.join("lemonade");
        let runtime_dir = runtime_dir_in(&root);
        fs::create_dir_all(runtime_dir.join("resources")).unwrap();
        let pins = runtime_dir.join("resources/backend_versions.json");
        fs::write(
            &pins,
            r#"{"therock":{"version":"7.14.1"},"llamacpp":{"rocm-stable":"b11783"}}"#,
        )
        .unwrap();
        let manifest = LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "11.5.1".to_owned(),
            runtime_dir: runtime_dir.clone(),
            lemond: lemond_path_in(&runtime_dir),
            lemonade: lemonade_path_in(&runtime_dir),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        };
        (dir, root, pins, manifest)
    }

    #[test]
    fn restore_without_a_cached_archive_downloads_nothing_and_leaves_the_pins() {
        // The restore runs after an install already failed (often because the
        // network is down), so it must not start a download of its own.
        let (dir, root, pins, manifest) = stale_pin_tree("restore-no-cache");
        let before = fs::read(&pins).unwrap();

        let error = restore_packaged_backend_versions_with(
            &manifest,
            "resources/backend_versions.json",
            "0000",
            || {},
        )
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(
            error.contains("is not cached") && error.contains("not downloading"),
            "{error}"
        );
        assert_eq!(
            fs::read(&pins).unwrap(),
            before,
            "the pin file is untouched"
        );
        assert!(
            !root.join("downloads").exists() && !root.join("extract").exists(),
            "a missing cache must not create a download or scratch directory"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_does_not_replace_a_cached_archive_that_fails_its_digest() {
        let (dir, root, pins, manifest) = stale_pin_tree("restore-bad-cache");
        let before = fs::read(&pins).unwrap();
        let source = dir.join("source-archive");
        write_embeddable_with_pins(
            &source,
            r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}"#,
        );
        cache_archive(&root, &source);

        let error = restore_packaged_backend_versions_with(
            &manifest,
            "resources/backend_versions.json",
            "not-the-digest-of-the-cached-archive",
            || {},
        )
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(
            error.contains("does not match its digest") && error.contains("not downloading"),
            "{error}"
        );
        assert_eq!(
            fs::read(&pins).unwrap(),
            before,
            "the pin file is untouched"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_reads_the_pin_file_after_the_archive_work_not_before() {
        let (dir, root, pins, manifest) = stale_pin_tree("restore-reads-late");
        let source = dir.join("source-archive");
        write_embeddable_with_pins(
            &source,
            r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}"#,
        );
        let sha256 = format!("{:x}", Sha256::digest(fs::read(&source).unwrap()));
        cache_archive(&root, &source);
        // Another process writes the pin file after this one started but
        // before it has finished with the archive.
        let concurrent = r#"{"therock":{"version":"7.14.2"},"llamacpp":{"rocm-stable":"b12000"},"custom":{"x":1}}"#;

        let backup = restore_packaged_backend_versions_with(
            &manifest,
            "resources/backend_versions.json",
            &sha256,
            || fs::write(&pins, concurrent).unwrap(),
        )
        .unwrap()
        .expect("the concurrent pins differ from the packaged ones");

        assert_eq!(
            backup.previous.as_deref(),
            Some(concurrent.as_bytes()),
            "the backup holds what was on disk when the restore wrote, not an earlier read"
        );
        let value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&pins).unwrap()).unwrap();
        assert_eq!(value["therock"]["version"], "7.13.0");
        assert_eq!(
            value["custom"]["x"], 1,
            "the concurrent writer's other entries survive"
        );
        backup.put_back().unwrap();
        assert_eq!(fs::read_to_string(&pins).unwrap(), concurrent);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_does_not_treat_an_unreadable_pin_file_as_absent() {
        // A directory where the file should be is unreadable on every platform
        // (and, like a sharing violation, is not `NotFound`). The restore must
        // fail without producing a backup that would delete the path later.
        let (dir, root, pins, manifest) = stale_pin_tree("restore-unreadable");
        fs::remove_file(&pins).unwrap();
        fs::create_dir(&pins).unwrap();
        fs::write(pins.join("keep"), b"kept").unwrap();
        let source = dir.join("source-archive");
        write_embeddable_with_pins(
            &source,
            r#"{"therock":{"version":"7.13.0"},"llamacpp":{"rocm-stable":"b9752"}}"#,
        );
        let sha256 = format!("{:x}", Sha256::digest(fs::read(&source).unwrap()));
        cache_archive(&root, &source);

        let error = restore_packaged_backend_versions_with(
            &manifest,
            "resources/backend_versions.json",
            &sha256,
            || {},
        )
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(error.contains("could not read"), "{error}");
        assert!(
            pins.is_dir() && pins.join("keep").is_file(),
            "the unreadable path is left exactly as it was"
        );
        assert_no_pin_temp_files(pins.parent().unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_backup_of_a_missing_pin_file_removes_the_file_again() {
        let dir = scratch_dir("pin-backup-missing");
        let pins = dir.join("backend_versions.json");
        fs::write(&pins, b"restored").unwrap();

        PinFileBackup::new(pins.clone(), None).put_back().unwrap();
        assert!(!pins.exists());
        // Already gone is not an error.
        PinFileBackup::new(pins, None).put_back().unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lemonade_root_uses_requested_engine_root() {
        let paths = AppPaths {
            config_dir: PathBuf::from("C:/Users/test/.rocm"),
            data_dir: PathBuf::from("C:/Users/test/.rocm"),
            cache_dir: PathBuf::from("C:/Users/test/.rocm/cache"),
        };
        let engine_root = PathBuf::from("D:/path/to/therock_venvs/engines");

        assert_eq!(
            lemonade_root(&paths, Some(&engine_root)),
            normalize_runtime_path_for_host(&engine_root).join(ENGINE_NAME)
        );
    }
}

/// Property test for the whole install path (`prepare_embeddable_with`), judged
/// on the filesystem by `archive_props` rather than by re-deriving the
/// extractor's own path checks. The extractor's own tar and zip property tests
/// are in `archive_extract::archive_properties`.
#[cfg(all(test, unix))]
mod archive_properties {
    use proptest::prelude::*;
    use sha2::{Digest as _, Sha256};

    use super::{
        AppPaths, installed_runtime_version, lemonade_path_in, lemonade_root, lemond_path_in,
        prepare_embeddable_with, runtime_dir_in,
    };
    use crate::archive_props::{
        Entry, Expect, Kind, Modes, benign_entries, entries, note_mode_scope, run_case,
        run_property, tar_gz_bytes,
    };

    /// What an install promises: the extractor it runs rejects every link kind
    /// and masks the modes the archive carries, so the installed tree is held to
    /// the whole contract, not just "no escape".
    const FULL: Expect = Expect {
        modes: Modes::Safe,
        no_outward_links: true,
    };

    /// Hostile entries — or benign names carrying hostile modes, which the
    /// extractor accepts — sometimes alongside a well-formed embeddable root,
    /// so the search-and-copy half of the install is reached too. Fully
    /// hostile archives alone almost never get past extraction (the extractor
    /// fails closed on the first bad entry), which would leave the copy step
    /// untested.
    fn install_entries() -> impl Strategy<Value = Vec<Entry>> {
        let modes = prop::sample::select(vec![0o644, 0o755, 0o4755, 0o2775, 0o777, 0o666]);
        let benign = (benign_entries(), modes).prop_map(|(mut entries, mode)| {
            for entry in &mut entries {
                entry.mode = mode;
            }
            entries
        });
        let body = prop_oneof![entries(), benign];
        (body, any::<bool>(), any::<bool>()).prop_map(|(mut entries, with_root, first)| {
            if with_root {
                let root = [
                    Entry {
                        name: b"e/lemond".to_vec(),
                        kind: Kind::File(b"lemond".to_vec()),
                        mode: 0o755,
                    },
                    Entry {
                        name: b"e/lemonade".to_vec(),
                        kind: Kind::File(b"lemonade".to_vec()),
                        mode: 0o755,
                    },
                ];
                if first {
                    entries.splice(0..0, root);
                } else {
                    entries.extend(root);
                }
            }
            entries
        })
    }

    /// The whole `prepare_embeddable` path — digest check, extract, root
    /// search, copy into the runtime tree, version marker — over hostile
    /// archives. Beyond "nothing outside the engine root changes", a failed
    /// install must never leave the version marker that tells the next run the
    /// runtime is complete.
    #[test]
    fn install_never_escapes_and_never_marks_a_failed_install_complete() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Reach of the install half: a property over installs that all fail
        // at extraction would say nothing about the copy step.
        let (cases, installed) = (AtomicUsize::new(0), AtomicUsize::new(0));
        note_mode_scope("lemonade-install", FULL.modes);
        run_property(
            "lemonade-install",
            256,
            install_entries(),
            None,
            |layout, entries| {
                cases.fetch_add(1, Ordering::Relaxed);
                let bytes = tar_gz_bytes(entries);
                let digest = format!("{:x}", Sha256::digest(&bytes));
                let paths = AppPaths {
                    config_dir: layout.dest.join("config"),
                    data_dir: layout.dest.join("data"),
                    cache_dir: layout.dest.join("cache"),
                };
                let mut install_result = None;
                let outcome = run_case(
                    layout,
                    "staged.tar.gz",
                    &bytes,
                    &[&layout.dest],
                    &layout.dest,
                    |layout, archive| {
                        let result = prepare_embeddable_with(
                            &paths,
                            Some(&layout.dest),
                            false,
                            "9.9.9",
                            &digest,
                            |_, destination| {
                                std::fs::copy(archive, destination)?;
                                Ok(())
                            },
                        );
                        install_result = Some(result.is_ok());
                        if result.is_ok() {
                            installed.fetch_add(1, Ordering::Relaxed);
                        }
                        result.map(|_| ())
                    },
                );
                outcome.judge(FULL)?;
                let runtime_dir = runtime_dir_in(&lemonade_root(&paths, Some(&layout.dest)));
                let marker = installed_runtime_version(&runtime_dir);
                if install_result == Some(false) && marker.is_some() {
                    return Err(format!(
                        "failed install left version marker {marker:?}: {outcome:#?}"
                    ));
                }
                if marker.is_some()
                    && !(lemond_path_in(&runtime_dir).is_file()
                        && lemonade_path_in(&runtime_dir).is_file())
                {
                    return Err(format!("marker without binaries: {outcome:#?}"));
                }
                Ok(())
            },
        )
        .unwrap();
        let (cases, installed) = (cases.into_inner(), installed.into_inner());
        eprintln!("[lemonade-install] {installed}/{cases} hostile archives installed successfully");
        assert!(
            installed * 100 / cases.max(1) >= 5,
            "install reach too low: {installed}/{cases} reached a completed install"
        );
    }
}
