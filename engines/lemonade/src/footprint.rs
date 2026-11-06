// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a Lemonade install costs on disk.
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
//! Everything here is decided before the install writes a byte, so the notice
//! and the refusal come first.

use std::path::{Path, PathBuf};

use rocm_core::{
    AppPaths, format_bytes, insufficient_space_message, runtime_path_is_same_or_inside,
};

use crate::install::{
    installed_runtime_version, lemonade_root, lemond_path_in, needs_extraction, read_manifest,
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

/// The runtime folder Lemonade's install record points at, if there is a
/// readable record.
fn recorded_runtime_dir(paths: &AppPaths) -> Option<PathBuf> {
    read_manifest(paths)
        .ok()
        .map(|manifest| manifest.runtime_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::{LemonadeInstallManifest, write_manifest};
    use std::fs;

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
        let base = lemonade_root(&paths, None);
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
        let base = lemonade_root(&paths, None);
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
        let base = lemonade_root(&paths, None);
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
}
