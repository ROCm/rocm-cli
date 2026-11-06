// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a Lemonade install costs on disk, and where its trees live.
//!
//! A fresh install is a multi-gigabyte download and a tree several times that
//! size: on a Windows gfx1151 host, Lemonade's `llamacpp:rocm` backend fetched
//! 4.7 GB (Lemonade's own `Total:` line) and the finished tree measured 24.3 GiB,
//! almost all of it the ROCm runtime Lemonade bundles with the backend. Before
//! this module the only thing a user saw was a one-line "Preparing..." and then,
//! from `rocm serve`, up to ten silent minutes. [`plan_install_preflight`] decides
//! whether an install is about to write a new tree, and [`InstallPreflight`]
//! renders what will be installed, where, how big it is, and refuses up front
//! when the drive cannot hold even the download.
//!
//! The second half is the read-only view `apps/rocm` needs to report Lemonade
//! trees (`rocm storage`) and to forget an install record whose tree a runtime
//! uninstall just deleted. The record and the tree layout belong to this crate,
//! so the knowledge of both stays here rather than being re-parsed by the CLI.

use std::fs;
use std::path::{Path, PathBuf};

use rocm_core::{
    AppPaths, format_bytes, insufficient_space_message, runtime_path_is_same_or_inside,
};

use crate::install::{
    installed_runtime_version, lemonade_root, lemond_path_in, manifest_path, needs_extraction,
    runtime_dir_in,
};

/// Download size of a fresh install: the embeddable archive plus the llama.cpp
/// ROCm backend Lemonade fetches for it. Lemonade printed `Total: 4.7 GB` (decimal)
/// for the backend on a Windows gfx1151 host; the embeddable archive adds ~5 MB.
/// The real figure varies with the GPU family and Lemonade's pins, which is why
/// it is only ever printed as "about".
pub(crate) const FRESH_INSTALL_DOWNLOAD_BYTES: u64 = 4_700_000_000;

/// Disk space a fresh install ends up using: 24.3 GiB measured for the same
/// install, rounded up. Used only for a warning, never to refuse, because it is
/// an estimate (the `rocm_core::disk_space` convention: exact requirements may
/// refuse, estimates only warn).
pub(crate) const FRESH_INSTALL_DISK_BYTES: u64 = 25 * 1024 * 1024 * 1024;

/// What an install is about to do, decided before anything is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InstallPreflight {
    /// The folder the install writes into (`<root>/runtime`, its downloads...).
    pub(crate) target_root: PathBuf,
    /// The Lemonade tree the install record currently points at, when that is a
    /// different folder: the install is not going to reuse it.
    pub(crate) other_install: Option<PathBuf>,
    /// Free bytes on the drive holding `target_root`, when known.
    pub(crate) available: Option<u64>,
}

/// Outcome of comparing the preflight's free space with the install's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpaceVerdict {
    /// Enough room, or free space could not be determined (never block on that).
    Fits,
    /// Room for the download but likely not for the finished tree.
    Tight(String),
    /// Not even room for the download: the install cannot finish.
    Refuse(String),
}

/// Whether installing into `env_root` will write a new Lemonade tree, and if so
/// what the user needs to be told. `None` when the tree there is already current
/// and has a llama.cpp backend, so the install only re-checks it.
pub(crate) fn plan_install_preflight(
    paths: &AppPaths,
    env_root: Option<&Path>,
    reinstall: bool,
    available_space: impl FnOnce(&Path) -> Option<u64>,
) -> Option<InstallPreflight> {
    let target_root = lemonade_root(paths, env_root);
    let runtime_dir = runtime_dir_in(&target_root);
    let extraction = needs_extraction(
        reinstall,
        installed_runtime_version(&runtime_dir).as_deref(),
        rocm_deps::LEMONADE_VERSION,
        lemond_path_in(&runtime_dir).is_file(),
    );
    let backend_missing = !runtime_dir.join("bin").join("llamacpp").is_dir();
    if !extraction && !backend_missing {
        return None;
    }
    let other_install = recorded_runtime_dir(paths)
        .filter(|recorded| !runtime_path_is_same_or_inside(recorded, &target_root));
    let available = available_space(&target_root);
    Some(InstallPreflight {
        target_root,
        other_install,
        available,
    })
}

impl InstallPreflight {
    /// The lines printed before the install starts. Every claim is derived from
    /// the fields the install itself acts on: the target is where it writes, and
    /// `other_install` is set only when the record points somewhere else.
    pub(crate) fn notice_lines(&self) -> Vec<String> {
        let free = self
            .available
            .map(|bytes| format!(" ({} free on that drive)", format_bytes(bytes)))
            .unwrap_or_default();
        let mut lines = vec![
            format!(
                "Installing Lemonade {} and its llama.cpp ROCm backend into {}",
                rocm_deps::LEMONADE_VERSION,
                self.target_root.display()
            ),
            format!(
                "  This downloads about {} and needs about {} of disk space{free}; it can take several minutes.",
                format_bytes(FRESH_INSTALL_DOWNLOAD_BYTES),
                format_bytes(FRESH_INSTALL_DISK_BYTES),
            ),
        ];
        if let Some(other) = self.other_install.as_ref() {
            lines.push(format!(
                "  Lemonade is already installed at {}; that copy is not reused for this location. `rocm storage` lists both.",
                other.display()
            ));
        }
        lines
    }

    /// Whether the drive can hold the install. Refuses only below the download
    /// size: the finished tree is several times larger than what is downloaded
    /// for any GPU family, so a drive that cannot hold the download cannot hold
    /// the install, however far the estimate is off.
    pub(crate) fn space_verdict(&self) -> SpaceVerdict {
        const OPERATION: &str = "install Lemonade and its llama.cpp ROCm backend";
        let Some(available) = self.available else {
            return SpaceVerdict::Fits;
        };
        let download = rocm_core::with_margin(FRESH_INSTALL_DOWNLOAD_BYTES);
        if available < download {
            return SpaceVerdict::Refuse(format!(
                "{} Nothing was downloaded.",
                insufficient_space_message(OPERATION, &self.target_root, download, available)
            ));
        }
        let disk = rocm_core::with_margin(FRESH_INSTALL_DISK_BYTES);
        if available < disk {
            return SpaceVerdict::Tight(format!(
                "Warning: free disk space may be insufficient to {OPERATION}: about {} needed, {} available on the drive holding {}. The install will continue, but may fail partway through.",
                format_bytes(disk),
                format_bytes(available),
                self.target_root.display()
            ));
        }
        SpaceVerdict::Fits
    }
}

/// Free space on the drive that will hold `path`, with the e2e-only override that
/// lets a black-box scenario reach the refusal without filling a real disk.
pub(crate) fn available_space_for_install(path: &Path) -> Option<u64> {
    #[cfg(feature = "e2e-test-hooks")]
    if let Some(bytes) = std::env::var_os(FREE_SPACE_TEST_ENV)
        .and_then(|value| value.to_str().and_then(|value| value.parse::<u64>().ok()))
    {
        return Some(bytes);
    }
    rocm_core::available_space_for_path(path)
}

/// Test-only override of the free space [`available_space_for_install`] reports.
#[cfg(feature = "e2e-test-hooks")]
const FREE_SPACE_TEST_ENV: &str = "ROCM_E2E_LEMONADE_FREE_SPACE_BYTES";

/// The folder a Lemonade install for `env_root` lives in: `<env_root>/lemonade`,
/// or `<data>/engines/lemonade` when there is none.
pub fn install_tree_root(paths: &AppPaths, env_root: Option<&Path>) -> PathBuf {
    lemonade_root(paths, env_root)
}

/// The parts of a Lemonade tree that make up the install itself.
///
/// That is the extracted runtime with its backends, the cached archive, and
/// leftover extraction scratch. Deliberately not the whole folder: under
/// `<data>/engines/lemonade` the same folder also holds the install record, logs
/// and server state.
pub fn install_tree_dirs(tree_root: &Path) -> [PathBuf; 3] {
    [
        runtime_dir_in(tree_root),
        tree_root.join("downloads"),
        tree_root.join("extract"),
    ]
}

/// Whether `tree_root` holds an extracted Lemonade runtime.
pub fn tree_has_install(tree_root: &Path) -> bool {
    lemond_path_in(&runtime_dir_in(tree_root)).is_file()
}

/// What Lemonade's install record says, as far as the CLI can tell.
///
/// Three states rather than an `Option`, because they call for different
/// reporting: a record that cannot be read is not the same as no record, and a
/// record whose folder no longer holds an install means nothing is in use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallRecord {
    /// No record: Lemonade has never been installed by ROCm CLI, or its record
    /// was cleared.
    NotRecorded,
    /// A record file exists but could not be read or parsed.
    Unreadable(String),
    /// The record names `runtime_dir`; `holds_install` is whether the server
    /// binary is still there.
    Recorded {
        runtime_dir: PathBuf,
        holds_install: bool,
    },
}

/// Read Lemonade's install record.
pub fn install_record(paths: &AppPaths) -> InstallRecord {
    let record = manifest_path(paths);
    match fs::read(&record) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => InstallRecord::NotRecorded,
        Err(error) => InstallRecord::Unreadable(format!("{}: {error}", record.display())),
        Ok(bytes) => {
            match serde_json::from_slice::<crate::install::LemonadeInstallManifest>(&bytes) {
                Ok(manifest) => InstallRecord::Recorded {
                    holds_install: lemond_path_in(&manifest.runtime_dir).is_file(),
                    runtime_dir: manifest.runtime_dir,
                },
                Err(error) => InstallRecord::Unreadable(format!("{}: {error}", record.display())),
            }
        }
    }
}

/// The runtime folder the install record points at, if there is a readable one.
pub fn recorded_runtime_dir(paths: &AppPaths) -> Option<PathBuf> {
    match install_record(paths) {
        InstallRecord::Recorded { runtime_dir, .. } => Some(runtime_dir),
        InstallRecord::NotRecorded | InstallRecord::Unreadable(_) => None,
    }
}

/// What [`forget_install_record_inside`] did, so a caller can report it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordCleanup {
    /// The record does not name anything inside the removed folder, or there is
    /// no record. Nothing to report.
    NotAffected,
    /// The record named this runtime folder and has been removed.
    Cleared(PathBuf),
    /// The record names a folder that was removed, but deleting the record
    /// failed: Lemonade still points at files that are gone.
    Failed {
        runtime_dir: PathBuf,
        record: PathBuf,
        error: String,
    },
    /// The record could not be read, so it could not be compared with the
    /// removed folder.
    Unreadable(String),
}

/// Forget the install record when the tree it points at was inside `removed_root`,
/// which the caller has just deleted.
///
/// Leaving the record would point Lemonade at files that no longer exist. Without
/// it Lemonade reports itself as not installed, which is the true state. Every
/// outcome that leaves the user with something to know is returned rather than
/// printed, so the uninstall report can carry it.
pub fn forget_install_record_inside(paths: &AppPaths, removed_root: &Path) -> RecordCleanup {
    forget_install_record_inside_with(paths, removed_root, |record| fs::remove_file(record))
}

/// [`forget_install_record_inside`] with the deletion injectable, so the failure
/// outcome can be tested without a file the OS refuses to delete.
fn forget_install_record_inside_with(
    paths: &AppPaths,
    removed_root: &Path,
    remove_record: impl FnOnce(&Path) -> std::io::Result<()>,
) -> RecordCleanup {
    let runtime_dir = match install_record(paths) {
        InstallRecord::NotRecorded => return RecordCleanup::NotAffected,
        InstallRecord::Unreadable(error) => return RecordCleanup::Unreadable(error),
        InstallRecord::Recorded { runtime_dir, .. } => runtime_dir,
    };
    if !runtime_path_is_same_or_inside(&runtime_dir, removed_root) {
        return RecordCleanup::NotAffected;
    }
    let record = manifest_path(paths);
    match remove_record(&record) {
        Ok(()) => RecordCleanup::Cleared(runtime_dir),
        Err(error) => RecordCleanup::Failed {
            runtime_dir,
            record,
            error: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::{LemonadeInstallManifest, write_manifest};

    fn paths_in(root: &Path) -> AppPaths {
        AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }

    /// Plant an extracted tree the way `prepare_embeddable` leaves one: the
    /// server binary, the version marker, and a llama.cpp backend folder.
    fn plant_current_tree(tree_root: &Path) {
        let runtime_dir = runtime_dir_in(tree_root);
        fs::create_dir_all(runtime_dir.join("bin").join("llamacpp").join("rocm-stable")).unwrap();
        fs::write(lemond_path_in(&runtime_dir), b"").unwrap();
        crate::install::record_runtime_version(&runtime_dir, rocm_deps::LEMONADE_VERSION).unwrap();
    }

    fn record_install_at(paths: &AppPaths, runtime_dir: &Path) {
        write_manifest(
            paths,
            &LemonadeInstallManifest {
                env_id: "lemonade-embeddable-test".to_owned(),
                version: rocm_deps::LEMONADE_VERSION.to_owned(),
                runtime_dir: runtime_dir.to_path_buf(),
                lemond: runtime_dir.join("lemond"),
                lemonade: runtime_dir.join("lemonade"),
                backend_recipe: "llamacpp".to_owned(),
                backend_name: "rocm".to_owned(),
                installed_at_unix_ms: 0,
            },
        )
        .unwrap();
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The release-test sequence: Lemonade installed into the data folder, then
    /// an SDK runtime activated, so the next install targets the runtime's own
    /// folder. The notice must name the new folder and the copy it is not reusing.
    #[test]
    fn a_new_location_is_announced_with_the_install_it_does_not_reuse() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let base = install_tree_root(&paths, None);
        plant_current_tree(&base);
        record_install_at(&paths, &runtime_dir_in(&base));
        let runtime_engines = temp.path().join("runtimes").join("key").join("engines");

        let preflight =
            plan_install_preflight(&paths, Some(&runtime_engines), false, |_| Some(100 * GIB))
                .expect("a folder with no Lemonade in it needs a full install");

        assert_eq!(preflight.target_root, runtime_engines.join("lemonade"));
        assert_eq!(preflight.other_install, Some(runtime_dir_in(&base)));
        let notice = preflight.notice_lines().join("\n");
        assert!(
            notice.contains(&preflight.target_root.display().to_string()),
            "{notice}"
        );
        assert!(notice.contains("This downloads about 4.4 GiB"), "{notice}");
        assert!(notice.contains("needs about 25.0 GiB"), "{notice}");
        assert!(notice.contains("100.0 GiB free on that drive"), "{notice}");
        assert!(
            notice.contains(&format!(
                "Lemonade is already installed at {}; that copy is not reused",
                runtime_dir_in(&base).display()
            )),
            "{notice}"
        );
        assert_eq!(preflight.space_verdict(), SpaceVerdict::Fits);
    }

    /// A current tree with a backend is not a new install: nothing to announce,
    /// so a plain `rocm serve` on a ready host prints nothing extra.
    #[test]
    fn a_current_tree_is_not_announced() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let base = install_tree_root(&paths, None);
        plant_current_tree(&base);

        assert_eq!(plan_install_preflight(&paths, None, false, |_| None), None);
        // `--reinstall` wipes and re-extracts the same tree, so it is announced,
        // and the record pointing into that same tree is not "another" install.
        record_install_at(&paths, &runtime_dir_in(&base));
        let reinstall = plan_install_preflight(&paths, None, true, |_| None).unwrap();
        assert_eq!(reinstall.other_install, None);
        assert!(
            !reinstall
                .notice_lines()
                .join("\n")
                .contains("already installed"),
            "the tree being reinstalled must not be reported as a second copy"
        );
    }

    #[test]
    fn a_tree_without_a_backend_is_announced() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let base = install_tree_root(&paths, None);
        plant_current_tree(&base);
        fs::remove_dir_all(runtime_dir_in(&base).join("bin")).unwrap();

        assert!(plan_install_preflight(&paths, None, false, |_| None).is_some());
    }

    #[test]
    fn space_below_the_download_refuses_and_below_the_tree_warns() {
        let preflight = |available| InstallPreflight {
            target_root: PathBuf::from("/data/engines/lemonade"),
            other_install: None,
            available,
        };
        let SpaceVerdict::Refuse(message) = preflight(Some(2 * GIB)).space_verdict() else {
            panic!("2 GiB cannot hold a 4.4 GiB download");
        };
        assert!(
            message.contains("not enough free disk space to install Lemonade"),
            "{message}"
        );
        assert!(message.contains("Nothing was downloaded."), "{message}");
        assert!(matches!(
            preflight(Some(10 * GIB)).space_verdict(),
            SpaceVerdict::Tight(_)
        ));
        assert_eq!(
            preflight(Some(40 * GIB)).space_verdict(),
            SpaceVerdict::Fits
        );
        assert_eq!(preflight(None).space_verdict(), SpaceVerdict::Fits);
    }

    /// A runtime uninstall deletes the runtime's folder, Lemonade tree included.
    /// The record pointing into it is forgotten; one pointing elsewhere is kept.
    #[test]
    fn the_record_is_forgotten_only_when_its_tree_was_removed() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let runtime_root = temp.path().join("runtimes").join("key");
        let per_runtime = install_tree_root(&paths, Some(&runtime_root.join("engines")));
        record_install_at(&paths, &runtime_dir_in(&per_runtime));

        let other_runtime = temp.path().join("runtimes").join("other");
        assert_eq!(
            forget_install_record_inside(&paths, &other_runtime),
            RecordCleanup::NotAffected
        );
        assert!(recorded_runtime_dir(&paths).is_some());

        assert_eq!(
            forget_install_record_inside(&paths, &runtime_root),
            RecordCleanup::Cleared(runtime_dir_in(&per_runtime))
        );
        assert_eq!(recorded_runtime_dir(&paths), None);
        assert!(!manifest_path(&paths).exists());
        // Nothing recorded: nothing to forget.
        assert_eq!(
            forget_install_record_inside(&paths, &runtime_root),
            RecordCleanup::NotAffected
        );
    }

    /// The folder is already gone when the record is cleared, so a failed
    /// deletion leaves Lemonade pointing at nothing. That has to come back to
    /// the caller with the record's path, not vanish into a log line.
    #[test]
    fn a_record_that_cannot_be_deleted_is_reported_not_swallowed() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let runtime_root = temp.path().join("runtimes").join("key");
        let per_runtime = install_tree_root(&paths, Some(&runtime_root.join("engines")));
        record_install_at(&paths, &runtime_dir_in(&per_runtime));

        let outcome = forget_install_record_inside_with(&paths, &runtime_root, |_| {
            Err(std::io::Error::other("sharing violation"))
        });

        assert_eq!(
            outcome,
            RecordCleanup::Failed {
                runtime_dir: runtime_dir_in(&per_runtime),
                record: manifest_path(&paths),
                error: "sharing violation".to_owned(),
            }
        );
        assert!(manifest_path(&paths).exists(), "the record was not deleted");
    }

    /// An unparseable record is not "no record": it may well name the removed
    /// folder, and the caller must be told it could not be checked.
    #[test]
    fn an_unreadable_record_is_distinct_from_no_record() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        assert_eq!(install_record(&paths), InstallRecord::NotRecorded);

        let record = manifest_path(&paths);
        fs::create_dir_all(record.parent().unwrap()).unwrap();
        fs::write(&record, b"{ not json").unwrap();

        let InstallRecord::Unreadable(error) = install_record(&paths) else {
            panic!("a corrupt record must not read as NotRecorded");
        };
        assert!(error.contains("runtime.json"), "{error}");
        assert_eq!(recorded_runtime_dir(&paths), None);
        assert!(matches!(
            forget_install_record_inside(&paths, temp.path()),
            RecordCleanup::Unreadable(_)
        ));
        assert!(record.exists(), "an unreadable record must be left alone");
    }

    /// A record whose runtime folder lost its server binary names an install
    /// that is not there; `holds_install` is what lets the storage report say so.
    #[test]
    fn a_record_reports_whether_its_folder_still_holds_an_install() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let tree = install_tree_root(&paths, None);
        record_install_at(&paths, &runtime_dir_in(&tree));
        assert_eq!(
            install_record(&paths),
            InstallRecord::Recorded {
                runtime_dir: runtime_dir_in(&tree),
                holds_install: false
            }
        );
        plant_current_tree(&tree);
        assert_eq!(
            install_record(&paths),
            InstallRecord::Recorded {
                runtime_dir: runtime_dir_in(&tree),
                holds_install: true
            }
        );
    }
}
