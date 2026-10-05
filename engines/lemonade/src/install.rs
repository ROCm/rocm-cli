// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, download_file_to_path, normalize_runtime_path_for_host, runtime_is_windows,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

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

/// Archive paths currently being validated, published, or extracted by this
/// process, with a condvar to wake waiters. Keyed per archive so an install of
/// one archive does not serialize behind an unrelated one — the download window
/// is minutes long, and the cross-process guarantee comes from the per-archive
/// file lock, not from this set.
static ARCHIVE_CACHE_BUSY: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
static ARCHIVE_CACHE_BUSY_SIGNAL: std::sync::Condvar = std::sync::Condvar::new();

/// How deep the extracted tree is searched for the embeddable root before the
/// search gives up. Official archives place the root at the top level, so this
/// only bounds the work spent looking; exceeding it is not an error.
const MAX_EMBEDDABLE_SEARCH_DEPTH: usize = 4;

/// Recursion backstop for copying the embeddable tree. Containment is enforced
/// per entry, so this exists only to fail loudly instead of overflowing the
/// stack on a pathological tree; it is far above any plausible archive layout.
const MAX_COPY_RECURSION_DEPTH: usize = 64;

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

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for verification", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        bail!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

fn publish_verified_archive(
    temporary: tempfile::NamedTempFile,
    archive: &Path,
    expected_sha256: &str,
) -> Result<()> {
    match temporary.persist_noclobber(archive) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            let publication_error = error.error;
            verify_sha256(archive, expected_sha256).with_context(|| {
                format!(
                    "concurrent archive publication at {} failed verification after publish conflict ({publication_error})",
                    archive.display()
                )
            })
        }
        Err(error) => Err(error.error)
            .with_context(|| format!("failed to publish verified archive {}", archive.display())),
    }
}

fn ensure_cached_archive<F>(
    url: &str,
    archive: &Path,
    expected_sha256: &str,
    download: F,
) -> Result<ArchiveCacheGuard>
where
    F: FnMut(&str, &Path) -> Result<()>,
{
    ensure_cached_archive_after_invalid(url, archive, expected_sha256, download, || {})
}

#[derive(Debug)]
struct ArchiveCacheLock {
    file: fs::File,
}

impl Drop for ArchiveCacheLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// In-process exclusion for one archive path.
#[derive(Debug)]
struct ArchiveCacheProcessLock {
    key: PathBuf,
}

impl Drop for ArchiveCacheProcessLock {
    fn drop(&mut self) {
        let mut busy = ARCHIVE_CACHE_BUSY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        busy.remove(&self.key);
        drop(busy);
        ARCHIVE_CACHE_BUSY_SIGNAL.notify_all();
    }
}

fn lock_archive_cache_in_process(archive: &Path) -> ArchiveCacheProcessLock {
    let key = archive.to_path_buf();
    let mut busy = ARCHIVE_CACHE_BUSY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if busy.contains(&key) {
        // The holder may be inside a download window minutes long; without this
        // the command sits silent and looks hung.
        eprintln!(
            "Waiting for another install in this process to finish with {}...",
            archive.display()
        );
    }
    while busy.contains(&key) {
        busy = ARCHIVE_CACHE_BUSY_SIGNAL
            .wait(busy)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    busy.insert(key.clone());
    ArchiveCacheProcessLock { key }
}

/// Exclusive access to one cached archive, held by the caller for as long as
/// the verified bytes are relied upon.
///
/// Verification and use have to happen under the same lock: releasing it at the
/// end of verification would leave a window in which a co-resident writer could
/// swap the archive before it is read again for extraction.
#[derive(Debug)]
struct ArchiveCacheGuard {
    _process: ArchiveCacheProcessLock,
    _file: ArchiveCacheLock,
}

fn archive_lock_path(archive: &Path) -> Result<PathBuf> {
    let parent = archive
        .parent()
        .with_context(|| format!("archive path has no parent: {}", archive.display()))?;
    let file_name = archive
        .file_name()
        .with_context(|| format!("archive path has no file name: {}", archive.display()))?;
    let mut lock_name = OsString::from(".");
    lock_name.push(file_name);
    lock_name.push(".lock");
    Ok(parent.join(lock_name))
}

#[cfg(test)]
fn mark_archive_lock_test_stage(stage: &str) {
    const ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    if let (Ok(role), Some(dir)) = (std::env::var(ROLE), std::env::var_os(DIR)) {
        fs::write(PathBuf::from(dir).join(format!("{role}-{stage}")), b"")
            .expect("write archive lock test marker");
    }
}

#[cfg(test)]
fn start_archive_lock_test_heartbeat() -> Option<std::thread::JoinHandle<()>> {
    const ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    if std::env::var(ROLE).as_deref() != Ok("recoverer") {
        return None;
    }
    let dir = std::env::var_os(DIR)?;
    Some(std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        fs::write(PathBuf::from(dir).join("recoverer-heartbeat"), b"")
            .expect("write archive lock test heartbeat");
    }))
}

fn lock_archive_cache(archive: &Path) -> Result<ArchiveCacheLock> {
    let lock_path = archive_lock_path(archive)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open archive cache lock {}", lock_path.display()))?;
    #[cfg(test)]
    mark_archive_lock_test_stage("before-lock");
    #[cfg(test)]
    let heartbeat = start_archive_lock_test_heartbeat();
    // `lock` blocks with no timeout, and the holder may be minutes into a
    // download, so announce the wait instead of leaving the command silent.
    // Probing first keeps the common uncontended case quiet.
    if file.try_lock().is_err() {
        eprintln!(
            "Waiting for another process to finish with {}...",
            archive.display()
        );
        file.lock().with_context(|| {
            format!(
                "failed to acquire archive cache lock {}",
                lock_path.display()
            )
        })?;
    }
    #[cfg(test)]
    {
        mark_archive_lock_test_stage("after-lock");
        if let Some(heartbeat) = heartbeat {
            heartbeat
                .join()
                .expect("archive lock test heartbeat panicked");
        }
    }
    Ok(ArchiveCacheLock { file })
}

fn ensure_cached_archive_after_invalid<F, I>(
    url: &str,
    archive: &Path,
    expected_sha256: &str,
    mut download: F,
    after_invalid: I,
) -> Result<ArchiveCacheGuard>
where
    F: FnMut(&str, &Path) -> Result<()>,
    I: FnOnce(),
{
    let parent = archive
        .parent()
        .with_context(|| format!("archive path has no parent: {}", archive.display()))?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create archive cache directory {}",
            parent.display()
        )
    })?;
    let process_guard = lock_archive_cache_in_process(archive);
    let archive_guard = lock_archive_cache(archive)?;
    let guard = ArchiveCacheGuard {
        _process: process_guard,
        _file: archive_guard,
    };

    match fs::symlink_metadata(archive) {
        Ok(metadata) if metadata.file_type().is_file() => {
            if verify_sha256(archive, expected_sha256).is_ok() {
                eprintln!("Using verified cached {}.", archive.display());
                return Ok(guard);
            }
            after_invalid();
            match fs::remove_file(archive) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to remove invalid {}", archive.display())
                    });
                }
            }
        }
        Ok(_) => bail!(
            "archive cache path is not a regular file: {}",
            archive.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let temporary = tempfile::Builder::new()
        .prefix(".lemonade-download-")
        .tempfile_in(parent)
        .with_context(|| {
            format!(
                // Owner-only permissions are a Unix guarantee: `tempfile`
                // creates with mode 0600 there, while on Windows the file
                // inherits the parent directory's DACL.
                "failed to create temporary download beside {}",
                archive.display()
            )
        })?;
    let temporary_path = temporary.path().to_path_buf();
    eprintln!("Downloading {}...", archive.display());
    download(url, &temporary_path)?;
    verify_sha256(&temporary_path, expected_sha256)?;
    publish_verified_archive(temporary, archive, expected_sha256)?;
    Ok(guard)
}

/// Resolves an archive entry name to a path inside `destination`.
///
/// Entry names come from the archive, so they are untrusted: this rejects
/// absolute paths, drive-relative and UNC names, `..` traversal, and anything
/// else that is not a plain sequence of normal components. The returned path is
/// built component-by-component, so it cannot leave `destination`.
fn archive_entry_destination(destination: &Path, name: &Path) -> Result<PathBuf> {
    let mut resolved = destination.to_path_buf();
    let mut components = 0usize;
    for component in name.components() {
        match component {
            std::path::Component::Normal(part) => {
                // Reject separators smuggled through a single component (a
                // Windows-style `a\..\b` read on Unix is one component).
                let text = part.to_string_lossy();
                if text.contains('/') || text.contains('\\') || text == ".." {
                    bail!("archive entry has an unsafe name: {}", name.display());
                }
                resolved.push(part);
                components += 1;
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                bail!(
                    "archive entry escapes the extraction root: {}",
                    name.display()
                );
            }
        }
    }
    if components == 0 {
        bail!("archive entry has an empty name");
    }
    Ok(resolved)
}

/// Creates the parent directories for an extracted entry, inside `destination`.
fn create_entry_parent(destination: &Path, path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if !parent.starts_with(destination) {
        bail!(
            "archive entry escapes the extraction root: {}",
            path.display()
        );
    }
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    Ok(())
}

/// Writes one extracted regular file, refusing to follow an existing entry.
///
/// `create_new` matters: without it a directory entry extracted earlier — or a
/// symlink planted by a co-resident process between entries — would be written
/// through rather than rejected.
fn write_entry_file(path: &Path, reader: &mut dyn Read) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    std::io::copy(reader, &mut file)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_entry_mode(path: &Path, mode: Option<u32>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let Some(mode) = mode else {
        return Ok(());
    };
    // Keep the executable bit the runtime needs; drop setuid/setgid/sticky and
    // any group/other write bit the archive happened to carry.
    let mode = mode & 0o755;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .with_context(|| format!("failed to set permissions on {}", path.display()))
}

#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "mirrors the Unix signature; Windows has no mode bits to apply"
)]
const fn set_entry_mode(_path: &Path, _mode: Option<u32>) -> Result<()> {
    Ok(())
}

/// Extracts a gzip-compressed tar archive in-process.
fn extract_tar_gz(archive: &Path, destination: &Path) -> Result<()> {
    let file =
        fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::io::BufReader::new(file)));
    for entry in tar
        .entries()
        .with_context(|| format!("failed to read {}", archive.display()))?
    {
        let mut entry = entry.with_context(|| format!("failed to read {}", archive.display()))?;
        let name = entry
            .path()
            .with_context(|| {
                format!(
                    "archive entry has an undecodable name in {}",
                    archive.display()
                )
            })?
            .into_owned();
        let kind = entry.header().entry_type();
        // A pax *global* header carries archive-wide metadata, not a member: its
        // name (`pax_global_header` by convention) is not a path to create. The
        // `tar` crate consumes long-name, long-link and pax *local* headers
        // itself, but hands this one to the caller, so it has to be skipped
        // explicitly — `git archive` and `tar --format=pax` both emit it, and
        // treating it as a member would reject an otherwise valid archive.
        if kind == tar::EntryType::XGlobalHeader {
            continue;
        }
        let path = archive_entry_destination(destination, &name)?;
        match kind {
            tar::EntryType::Directory => {
                fs::create_dir_all(&path)
                    .with_context(|| format!("failed to create {}", path.display()))?;
            }
            tar::EntryType::Regular | tar::EntryType::Continuous => {
                create_entry_parent(destination, &path)?;
                let mode = entry.header().mode().ok();
                write_entry_file(&path, &mut entry)?;
                set_entry_mode(&path, mode)?;
            }
            // A sparse entry would expand correctly — the crate resolves the
            // sparse map — but a few-byte header can declare a multi-gigabyte
            // logical size, so accepting one turns a digest-pinned archive into
            // an unbounded write. No Lemonade artifact stores files sparsely, so
            // this stays rejected until one needs it.
            tar::EntryType::GNUSparse => bail!(
                "archive contains a sparse entry at {}, which is not supported",
                name.display()
            ),
            // Links, devices, fifos and sockets are the entry kinds that let an
            // archive reach outside its own tree; the official artifacts carry
            // none of them. See the link policy on `extract_archive` for what
            // to do if a future release ships one.
            other => bail!(
                "archive contains an unsupported entry ({other:?}) at {}",
                name.display()
            ),
        }
    }
    Ok(())
}

/// Extracts a zip archive in-process.
fn extract_zip(archive: &Path, destination: &Path) -> Result<()> {
    let file =
        fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("failed to read {}", archive.display()))?;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .with_context(|| format!("failed to read entry {index} of {}", archive.display()))?;
        // `enclosed_name` returns None for absolute or `..`-bearing names; the
        // raw name is still used for diagnostics and for the trailing-slash
        // directory convention.
        let raw = entry.name().to_owned();
        // See the link policy on `extract_archive`.
        if entry.is_symlink() {
            bail!("archive contains symlink at {raw}");
        }
        let name = entry
            .enclosed_name()
            .with_context(|| format!("archive entry escapes the extraction root: {raw}"))?;
        let path = archive_entry_destination(destination, &name)?;
        if entry.is_dir() {
            fs::create_dir_all(&path)
                .with_context(|| format!("failed to create {}", path.display()))?;
            continue;
        }
        // No third entry kind to reject here: `zip` defines `is_file` as
        // "neither a directory nor a symlink", and both are handled above.
        create_entry_parent(destination, &path)?;
        let mode = entry.unix_mode();
        write_entry_file(&path, &mut entry)?;
        set_entry_mode(&path, mode)?;
    }
    Ok(())
}

/// Extracts the embeddable archive into `destination`.
///
/// Extraction is done in-process rather than by shelling out to `tar`: the
/// external binary was resolved from `PATH` (so the extractor unpacking
/// untrusted bytes was itself unpinned), and its traversal behaviour varied
/// between GNU tar, the bsdtar shipped as Windows `tar.exe`, and busybox tar.
///
/// # Link policy
///
/// Extraction writes directories and regular files and nothing else: a link
/// entry is rejected rather than resolved, here and again when the extracted
/// tree is searched and copied out. That is deliberately conservative. The
/// v10.10.0 artifacts contain no links, and refusing them removes the whole
/// class of "entry points outside the tree" bugs instead of trying to reason
/// about each one.
///
/// The cost is that it fails closed on an authentic archive: if a future
/// Lemonade release ships, say, `libfoo.so -> libfoo.so.1`, installs stop with
/// "archive contains symlink at ...". That is a signal to widen the policy
/// deliberately — resolve the target and accept it only if it stays inside the
/// extraction root — not to drop the check.
fn extract_archive(archive: &Path, destination: &Path) -> Result<()> {
    let is_zip = archive
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"));
    if is_zip {
        extract_zip(archive, destination)
    } else {
        extract_tar_gz(archive, destination)
    }
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
    // See the link policy on `extract_archive`.
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

fn destination_child(destination: &Path, name: &std::ffi::OsStr) -> Result<PathBuf> {
    let component = Path::new(name);
    if component.components().count() != 1
        || !matches!(
            component.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        bail!(
            "archive entry is not one destination component: {}",
            Path::new(name).display()
        );
    }
    Ok(destination.join(component))
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    let source = fs::canonicalize(source)
        .with_context(|| format!("failed to canonicalize {}", source.display()))?;
    if !fs::symlink_metadata(&source)?.file_type().is_dir() {
        bail!("copy source is not a directory: {}", source.display());
    }
    fs::create_dir(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let destination = fs::canonicalize(destination)
        .with_context(|| format!("failed to canonicalize {}", destination.display()))?;
    copy_tree_entries(&source, &source, &destination, &destination, 0)
}

fn copy_tree_entries(
    source_root: &Path,
    source: &Path,
    destination_root: &Path,
    destination: &Path,
    depth: usize,
) -> Result<()> {
    if depth > MAX_COPY_RECURSION_DEPTH {
        bail!(
            "archive tree is deeper than {MAX_COPY_RECURSION_DEPTH} levels at {}",
            source.display()
        );
    }
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let file_type = entry.file_type()?;
        // See the link policy on `extract_archive`.
        if file_type.is_symlink() {
            bail!("refusing to copy symlink {}", source_path.display());
        }
        if !file_type.is_dir() && !file_type.is_file() {
            bail!(
                "refusing to copy non-regular entry {}",
                source_path.display()
            );
        }
        let canonical_source = fs::canonicalize(&source_path)?;
        if !canonical_source.starts_with(source_root) {
            bail!(
                "copy source escapes extraction root: {}",
                source_path.display()
            );
        }
        let destination_path = destination_child(destination, &entry.file_name())?;
        if !destination_path.starts_with(destination_root) {
            bail!(
                "copy destination escapes runtime root: {}",
                destination_path.display()
            );
        }
        if file_type.is_dir() {
            fs::create_dir(&destination_path)
                .with_context(|| format!("failed to create {}", destination_path.display()))?;
            let canonical_destination = fs::canonicalize(&destination_path)?;
            if !canonical_destination.starts_with(destination_root) {
                bail!(
                    "copy destination escapes runtime root: {}",
                    destination_path.display()
                );
            }
            copy_tree_entries(
                source_root,
                &canonical_source,
                destination_root,
                &canonical_destination,
                depth + 1,
            )?;
        } else {
            fs::copy(&canonical_source, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    canonical_source.display(),
                    destination_path.display()
                )
            })?;
            if !cfg!(windows) {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let metadata = fs::symlink_metadata(&canonical_source)?;
                    fs::set_permissions(
                        &destination_path,
                        fs::Permissions::from_mode(metadata.permissions().mode()),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command as ProcessCommand, Stdio};
    use std::time::Duration;

    /// A fresh scratch directory under the crate's `target/`. The base is
    /// `CARGO_MANIFEST_DIR`, a compile-time constant, so the path never derives from a
    /// runtime environment read.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("lemonade-fs-test-{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

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
    fn sha256_verification_accepts_matching_content_and_rejects_mismatch() {
        let dir = scratch_dir("archive-digest");
        let archive = dir.join("archive");
        fs::write(&archive, b"verified archive").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"verified archive"));

        verify_sha256(&archive, &expected).unwrap();
        let error = verify_sha256(&archive, &"0".repeat(64)).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repeated_valid_archive_publication_is_idempotent() {
        let dir = scratch_dir("valid-publication-race");
        let archive = dir.join("archive");
        let content = b"verified archive";
        let expected = format!("{:x}", Sha256::digest(content));

        for _ in 0..2 {
            let mut temporary = tempfile::NamedTempFile::new_in(&dir).unwrap();
            temporary.write_all(content).unwrap();
            verify_sha256(temporary.path(), &expected).unwrap();
            publish_verified_archive(temporary, &archive, &expected).unwrap();
        }

        assert_eq!(fs::read(&archive).unwrap(), content);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_publication_conflict_with_foreign_bytes_fails_closed() {
        let dir = scratch_dir("invalid-publication-race");
        let archive = dir.join("archive");
        let content = b"verified archive";
        let expected = format!("{:x}", Sha256::digest(content));
        let mut temporary = tempfile::NamedTempFile::new_in(&dir).unwrap();
        temporary.write_all(content).unwrap();
        verify_sha256(temporary.path(), &expected).unwrap();

        fs::write(&archive, b"unverified concurrent archive").unwrap();
        let error = publish_verified_archive(temporary, &archive, &expected).unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("concurrent archive publication"));
        assert!(error.contains("failed verification after publish conflict"));
        assert!(error.contains("SHA-256 mismatch"));
        assert_eq!(
            fs::read(&archive).unwrap(),
            b"unverified concurrent archive"
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poisoned_cached_archive_is_removed_and_redownloaded() {
        let dir = scratch_dir("poisoned-cache");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));
        let mut downloads = 0;

        ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
            downloads += 1;
            fs::write(destination, b"fresh archive")?;
            Ok(())
        })
        .unwrap();

        assert_eq!(downloads, 1);
        assert_eq!(fs::read(&archive).unwrap(), b"fresh archive");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_with_wrong_content_is_rejected_and_leaves_no_archive() {
        let dir = scratch_dir("bad-content-download");
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        // A download that succeeds at the transport level but returns the wrong
        // bytes must not be published, even though nothing errored.
        let error = ensure_cached_archive("test://archive", &archive, &expected, |_, path| {
            fs::write(path, b"attacker archive")?;
            Ok(())
        })
        .unwrap_err();

        let error = format!("{error:#}");
        assert!(error.contains("SHA-256 mismatch"), "{error}");
        assert!(!archive.exists(), "unverified bytes were published");
        // Only the lock file may remain; the temporary download must be gone.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name != ".archive.lock")
            .collect();
        assert!(leftovers.is_empty(), "leftovers: {leftovers:?}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_guard_outlives_verification_and_is_scoped_to_one_archive() {
        let dir = scratch_dir("archive-guard-scope");
        let archive = dir.join("archive");
        let other = dir.join("other-archive");
        let content = b"fresh archive";
        let expected = format!("{:x}", Sha256::digest(content));

        // The guard returned by verification is what callers hold across
        // extraction; while it is alive the same archive must not be lockable.
        let guard = ensure_cached_archive("test://archive", &archive, &expected, |_, path| {
            fs::write(path, content)?;
            Ok(())
        })
        .unwrap();

        let blocked = std::thread::spawn(move || {
            let _second = lock_archive_cache_in_process(&archive);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !blocked.is_finished(),
            "a second in-process lock on the held archive was granted"
        );

        // A different archive must not queue behind it.
        let unrelated = std::thread::spawn(move || {
            let _other = lock_archive_cache_in_process(&other);
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !unrelated.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "an unrelated archive serialized behind the held one"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        unrelated.join().unwrap();

        drop(guard);
        blocked.join().unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn poisoned_process_mutex_is_recovered_before_digest_validation() {
        let _ = std::thread::spawn(|| {
            let _guard = ARCHIVE_CACHE_BUSY.lock().unwrap();
            panic!("poison archive cache process mutex");
        })
        .join();
        let dir = scratch_dir("poisoned-process-mutex");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
            fs::write(destination, b"fresh archive")?;
            Ok(())
        })
        .unwrap();

        verify_sha256(&archive, &expected).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    const ARCHIVE_LOCK_TEST_ROLE: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_ROLE";
    const ARCHIVE_LOCK_TEST_DIR: &str = "ROCM_LEMONADE_ARCHIVE_LOCK_TEST_DIR";
    const ARCHIVE_LOCK_TEST_TIMEOUT: Duration = Duration::from_secs(10);

    struct ChildGuard(Option<std::process::Child>);

    impl ChildGuard {
        fn spawn(test_exe: &Path, dir: &Path, role: &str) -> Self {
            let child = ProcessCommand::new(test_exe)
                .args([
                    "--exact",
                    "install::tests::archive_cache_subprocess_helper",
                    "--nocapture",
                ])
                .env(ARCHIVE_LOCK_TEST_ROLE, role)
                .env(ARCHIVE_LOCK_TEST_DIR, dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            Self(Some(child))
        }

        fn is_running(&mut self) -> bool {
            self.0.as_mut().unwrap().try_wait().unwrap().is_none()
        }

        fn wait_until_exit(&mut self, timeout: Duration) -> std::process::ExitStatus {
            let deadline = std::time::Instant::now() + timeout;
            loop {
                if let Some(status) = self.0.as_mut().unwrap().try_wait().unwrap() {
                    self.0.take();
                    return status;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "timed out waiting for child process {}",
                    self.0.as_ref().unwrap().id()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                if child.try_wait().ok().flatten().is_none() {
                    let _ = child.kill();
                }
                let _ = child.wait();
            }
        }
    }

    #[test]
    fn archive_cache_subprocess_helper() {
        let Ok(role) = std::env::var(ARCHIVE_LOCK_TEST_ROLE) else {
            return;
        };
        let dir = PathBuf::from(std::env::var_os(ARCHIVE_LOCK_TEST_DIR).unwrap());
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));

        ensure_cached_archive_after_invalid(
            "test://archive",
            &archive,
            &expected,
            |_, destination| {
                fs::write(destination, b"fresh archive")?;
                Ok(())
            },
            || {
                if role == "validator" {
                    fs::write(dir.join("validator-paused"), b"").unwrap();
                    let deadline = std::time::Instant::now() + ARCHIVE_LOCK_TEST_TIMEOUT;
                    while !dir.join("release-validator").is_file() {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "timed out waiting to release validator"
                        );
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            },
        )
        .unwrap();
        verify_sha256(&archive, &expected).unwrap();
    }

    fn wait_for_test_file(path: &Path, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while !path.is_file() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn interprocess_lock_serializes_poisoned_validation_and_recovery() {
        let dir = scratch_dir("interprocess-archive-lock");
        let archive = dir.join("archive");
        fs::write(&archive, b"poisoned").unwrap();
        let test_exe = std::env::current_exe().unwrap();

        let mut validator = ChildGuard::spawn(&test_exe, &dir, "validator");
        wait_for_test_file(&dir.join("validator-paused"), ARCHIVE_LOCK_TEST_TIMEOUT);
        let mut recoverer = ChildGuard::spawn(&test_exe, &dir, "recoverer");
        wait_for_test_file(
            &dir.join("recoverer-before-lock"),
            ARCHIVE_LOCK_TEST_TIMEOUT,
        );
        wait_for_test_file(&dir.join("recoverer-heartbeat"), ARCHIVE_LOCK_TEST_TIMEOUT);
        assert!(
            !dir.join("recoverer-after-lock").is_file(),
            "recoverer acquired the inter-process lock before release"
        );
        assert!(
            recoverer.is_running(),
            "recoverer exited while waiting for the inter-process lock"
        );

        fs::write(dir.join("release-validator"), b"").unwrap();
        assert!(
            validator
                .wait_until_exit(ARCHIVE_LOCK_TEST_TIMEOUT)
                .success(),
            "validator subprocess failed"
        );
        assert!(
            recoverer
                .wait_until_exit(ARCHIVE_LOCK_TEST_TIMEOUT)
                .success(),
            "recoverer subprocess failed"
        );
        assert!(dir.join("recoverer-after-lock").is_file());
        let expected = format!("{:x}", Sha256::digest(b"fresh archive"));
        verify_sha256(&archive, &expected).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn failed_download_removes_private_partial_file() {
        let dir = scratch_dir("partial-download");
        let archive = dir.join("archive");
        let expected = format!("{:x}", Sha256::digest(b"complete"));

        let error =
            ensure_cached_archive("test://archive", &archive, &expected, |_, destination| {
                fs::write(destination, b"partial")?;
                bail!("download interrupted")
            })
            .unwrap_err();

        assert!(error.to_string().contains("download interrupted"));
        assert!(!archive.exists());
        assert_eq!(
            fs::read_dir(&dir)
                .unwrap()
                .filter(
                    |entry| entry.as_ref().unwrap().path() != archive_lock_path(&archive).unwrap()
                )
                .count(),
            0
        );
        fs::remove_dir_all(&dir).ok();
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
    fn destination_child_requires_one_normal_component() {
        let destination = Path::new("runtime");
        assert_eq!(
            destination_child(destination, std::ffi::OsStr::new("bin")).unwrap(),
            destination.join("bin")
        );
        assert!(destination_child(destination, std::ffi::OsStr::new("..")).is_err());
        assert!(destination_child(destination, std::ffi::OsStr::new(".")).is_err());
        assert!(destination_child(destination, std::ffi::OsStr::new("a/b")).is_err());
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

    /// Writes a `.tar.gz` whose single entry carries `entry_name` verbatim.
    ///
    /// The name is stamped into the raw header bytes because `tar::Builder`
    /// refuses to serialize `..` or absolute paths — the archives this guards
    /// against are hostile ones, which no well-behaved writer would produce.
    fn write_raw_named_tar_gz(path: &Path, entry_name: &str, body: &[u8]) {
        let file = fs::File::create(path).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        let name = header.as_old_mut();
        let bytes = entry_name.as_bytes();
        assert!(bytes.len() < name.name.len());
        name.name[..bytes.len()].copy_from_slice(bytes);
        header.set_cksum();
        builder.append(&header, body).unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    const EXTRACT_PATH_TEST_DIR: &str = "ROCM_LEMONADE_EXTRACT_PATH_TEST_DIR";

    /// Child half of [`extract_archive_does_not_execute_a_path_supplied_tar`].
    ///
    /// Runs in its own process so the hostile `PATH` can be set with
    /// `Command::env` — mutating `PATH` in-process would need `unsafe`, which
    /// the workspace denies.
    #[test]
    fn extract_path_subprocess_helper() {
        let Some(dir) = std::env::var_os(EXTRACT_PATH_TEST_DIR) else {
            return;
        };
        let dir = PathBuf::from(dir);
        let _ = extract_archive(&dir.join("archive.tar.gz"), &dir.join("extract"));
    }

    #[cfg(unix)]
    #[test]
    fn extract_archive_does_not_execute_a_path_supplied_tar() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("path-supplied-tar");
        let archive = dir.join("archive.tar.gz");
        fs::create_dir_all(dir.join("extract")).unwrap();
        write_raw_named_tar_gz(&archive, "payload.txt", b"real");

        let fake_bin = dir.join("bin");
        fs::create_dir_all(&fake_bin).unwrap();
        let marker = dir.join("impostor-ran");
        let fake_tar = fake_bin.join("tar");
        fs::write(
            &fake_tar,
            format!("#!/bin/sh\ntouch '{}'\nexit 0\n", marker.display()),
        )
        .unwrap();
        fs::set_permissions(&fake_tar, fs::Permissions::from_mode(0o755)).unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let status = ProcessCommand::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "install::tests::extract_path_subprocess_helper",
                "--nocapture",
            ])
            .env(EXTRACT_PATH_TEST_DIR, &dir)
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "helper subprocess failed");

        assert!(
            !marker.exists(),
            "extraction executed a PATH-supplied tar instead of a pinned in-process extractor"
        );
        fs::remove_dir_all(&dir).ok();
    }

    fn extract_hostile_tar_entry(tag: &str, entry_name: &str) -> String {
        let dir = scratch_dir(tag);
        let archive = dir.join("hostile.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();
        write_raw_named_tar_gz(&archive, entry_name, b"pwned");

        let error = extract_archive(&archive, &destination).unwrap_err();
        assert_eq!(
            fs::read_dir(&destination).unwrap().count(),
            0,
            "hostile entry was written into the extraction root"
        );
        fs::remove_dir_all(&dir).ok();
        format!("{error:#}")
    }

    #[test]
    fn extract_archive_rejects_parent_traversal_entry() {
        let error = extract_hostile_tar_entry("traversal-parent", "../escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_nested_parent_traversal_entry() {
        let error = extract_hostile_tar_entry("traversal-nested", "inner/../../escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_absolute_entry() {
        let error = extract_hostile_tar_entry("traversal-absolute", "/etc/escaped.txt");
        assert!(error.contains("escapes the extraction root"), "{error}");
    }

    #[test]
    fn extract_archive_rejects_symlink_entry() {
        let dir = scratch_dir("traversal-symlink");
        let archive = dir.join("hostile.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(0);
        header.set_mode(0o777);
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_link_name("/etc/passwd").unwrap();
        header.set_path("escape-link").unwrap();
        header.set_cksum();
        builder.append(&header, std::io::empty()).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
        assert!(error.contains("unsupported entry"), "{error}");
        assert!(!destination.join("escape-link").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_archive_writes_a_well_formed_tree() {
        let dir = scratch_dir("extract-well-formed");
        let archive = dir.join("good.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_ustar();
        header.set_size(5);
        header.set_mode(0o755);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_path("embeddable/bin/lemond").unwrap();
        header.set_cksum();
        builder.append(&header, &b"hello"[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        extract_archive(&archive, &destination).unwrap();
        let extracted = destination.join("embeddable").join("bin").join("lemond");
        assert_eq!(fs::read(&extracted).unwrap(), b"hello");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&extracted).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111, "executable bit was not preserved");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_archive_rejects_hostile_zip_entry() {
        let dir = scratch_dir("traversal-zip");
        let archive = dir.join("hostile.zip");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();
        // `add_directory`/`start_file` normalize names, so the traversal name is
        // written through the raw entry API a hostile producer would use.
        writer
            .start_file("../escaped.txt", options)
            .or_else(|_| writer.start_file("placeholder.txt", options))
            .unwrap();
        writer.write_all(b"pwned").unwrap();
        writer.finish().unwrap();

        // If the zip writer refused to emit the traversal name there is nothing
        // hostile to assert on; the tar cases cover the shared validator.
        let mut zip = zip::ZipArchive::new(fs::File::open(&archive).unwrap()).unwrap();
        let hostile = zip.by_index(0).unwrap().name().to_owned();
        if hostile == "../escaped.txt" {
            let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
            assert!(error.contains("escapes the extraction root"), "{error}");
            assert!(!dir.join("escaped.txt").exists());
        } else {
            extract_archive(&archive, &destination).unwrap();
            assert!(destination.join("placeholder.txt").is_file());
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// A pax global header is archive-wide metadata, not a member, and the
    /// `tar` crate hands it to the caller rather than consuming it the way it
    /// consumes long-name and pax local headers. Treating it as a member
    /// rejects any archive written by `git archive` or `tar --format=pax`.
    #[test]
    fn extract_archive_skips_pax_global_header() {
        let dir = scratch_dir("pax-global-header");
        let archive = dir.join("pax.tar.gz");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);

        let record = b"52 comment=0000000000000000000000000000000000000000\n";
        let mut global = tar::Header::new_ustar();
        global.set_size(record.len() as u64);
        global.set_mode(0o644);
        global.set_entry_type(tar::EntryType::XGlobalHeader);
        global.set_path("pax_global_header").unwrap();
        global.set_cksum();
        builder.append(&global, &record[..]).unwrap();

        let mut header = tar::Header::new_ustar();
        header.set_size(5);
        header.set_mode(0o755);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_path("embeddable/bin/lemond").unwrap();
        header.set_cksum();
        builder.append(&header, &b"hello"[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        extract_archive(&archive, &destination).unwrap();
        assert_eq!(
            fs::read(destination.join("embeddable").join("bin").join("lemond")).unwrap(),
            b"hello"
        );
        assert!(
            !destination.join("pax_global_header").exists(),
            "the global header was written out as a member"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// Zip counterpart to `extract_archive_rejects_symlink_entry`: the zip path
    /// has its own link check, keyed on the entry's unix mode.
    #[test]
    fn extract_archive_rejects_zip_symlink_entry() {
        let dir = scratch_dir("zip-symlink");
        let archive = dir.join("hostile.zip");
        let destination = dir.join("extract");
        fs::create_dir_all(&destination).unwrap();

        let file = fs::File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .add_symlink(
                "escape-link",
                "/etc/passwd",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
        writer.finish().unwrap();

        let error = format!("{:#}", extract_archive(&archive, &destination).unwrap_err());
        assert!(error.contains("symlink"), "{error}");
        assert!(!destination.join("escape-link").exists());
        fs::remove_dir_all(&dir).ok();
    }
}

/// Shared archive generator and filesystem oracle; see the module docs.
#[cfg(all(test, unix))]
#[path = "../../../crates/rocm-core/src/archive_props.rs"]
mod archive_props;

/// Property tests for the in-process Lemonade extractor, judged on the
/// filesystem by `archive_props` rather than by re-deriving the extractor's
/// own path checks.
#[cfg(all(test, unix))]
mod archive_properties {
    use std::io::Write as _;

    use proptest::prelude::*;
    use sha2::{Digest as _, Sha256};

    use super::archive_props::{
        Entry, Expect, Kind, benign_entries, entries, run_case, run_property, tar_gz_bytes,
    };
    use super::{
        AppPaths, extract_tar_gz, extract_zip, installed_runtime_version, lemonade_path_in,
        lemonade_root, lemond_path_in, prepare_embeddable_with, runtime_dir_in,
    };

    /// The in-process extractor rejects every link kind and masks modes, so it
    /// promises the whole contract, not just "no escape".
    const FULL: Expect = Expect {
        safe_modes: true,
        no_outward_links: true,
    };

    #[test]
    fn tar_extraction_keeps_every_entry_inside_the_root() {
        run_property(
            "lemonade-tar",
            512,
            entries(),
            Some(5),
            |layout, entries| {
                run_case(
                    layout,
                    "embeddable.tar.gz",
                    &tar_gz_bytes(entries),
                    &[&layout.dest],
                    &layout.dest,
                    |layout, archive| extract_tar_gz(archive, &layout.dest),
                )
                .judge(FULL)
            },
        )
        .unwrap();
    }

    /// Zip names are strings, so non-UTF-8 names are carried lossily; hard
    /// links do not exist in zip and are written as regular files.
    fn zip_bytes(entries: &[Entry]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for entry in entries {
            let name = String::from_utf8_lossy(&entry.name).into_owned();
            let options = zip::write::SimpleFileOptions::default().unix_permissions(entry.mode);
            // The writer refuses a few names (and duplicates); a refused entry
            // is simply left out of the archive.
            match &entry.kind {
                Kind::File(data) | Kind::Hardlink(data) => {
                    if writer.start_file(name, options).is_ok() {
                        writer.write_all(data).unwrap();
                    }
                }
                Kind::Dir => {
                    let _ = writer.add_directory(name, options);
                }
                Kind::Symlink(target) => {
                    let _ = writer.add_symlink(
                        name,
                        String::from_utf8_lossy(target).into_owned(),
                        options,
                    );
                }
            }
        }
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn zip_extraction_keeps_every_entry_inside_the_root() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Reach is measured on what the zip writer actually stored, since it
        // may drop or normalise some generated names.
        let (cases, traversal, symlink) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        run_property("lemonade-zip", 512, entries(), None, |layout, entries| {
            let bytes = zip_bytes(entries);
            let mut stored = zip::ZipArchive::new(std::io::Cursor::new(&bytes)).unwrap();
            cases.fetch_add(1, Ordering::Relaxed);
            let (mut has_traversal, mut has_symlink) = (false, false);
            for index in 0..stored.len() {
                let entry = stored.by_index_raw(index).unwrap();
                let name = entry.name();
                has_traversal |=
                    name.starts_with('/') || name.split(['/', '\\']).any(|part| part == "..");
                has_symlink |= entry.is_symlink();
            }
            if has_traversal {
                traversal.fetch_add(1, Ordering::Relaxed);
            }
            if has_symlink {
                symlink.fetch_add(1, Ordering::Relaxed);
            }
            run_case(
                layout,
                "embeddable.zip",
                &bytes,
                &[&layout.dest],
                &layout.dest,
                |layout, archive| extract_zip(archive, &layout.dest),
            )
            .judge(FULL)
        })
        .unwrap();
        let cases = cases.into_inner().max(1);
        let (traversal, symlink) = (traversal.into_inner(), symlink.into_inner());
        eprintln!(
            "[lemonade-zip] stored traversal/absolute names in {}%, symlinks in {}% of {cases} archives",
            traversal * 100 / cases,
            symlink * 100 / cases
        );
        assert!(traversal * 100 / cases >= 5 && symlink * 100 / cases >= 5);
    }

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
