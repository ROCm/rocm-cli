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

/// Download size of an install that fetches the llama.cpp ROCm backend: the
/// embeddable archive plus the backend. Lemonade printed `Total: 4.7 GB`
/// (decimal) for the backend on a Windows gfx1151 host; the embeddable archive
/// adds ~5 MB.
///
/// An upper bound, never a promise. The backend is chosen after this preflight
/// (Lemonade reports per-GPU support, and `backend_alignment.rs` falls back to
/// `vulkan` where its ROCm build is unsupported), and the figure varies with the
/// GPU family and Lemonade's pins. So it is printed as "up to about" and only
/// ever warns; see [`MIN_INSTALL_BYTES`] for what can refuse.
pub(crate) const FRESH_INSTALL_DOWNLOAD_BYTES: u64 = 4_700_000_000;

/// Disk space an install that fetches the ROCm backend ends up using: 24.3 GiB
/// measured for the same install, rounded up. An estimate and an upper bound, so
/// it only ever warns (the `rocm_core::disk_space` convention: exact
/// requirements may refuse, estimates only warn).
pub(crate) const FRESH_INSTALL_DISK_BYTES: u64 = 25 * 1024 * 1024 * 1024;

/// What every Lemonade install needs whichever backend is chosen: the embeddable
/// archive (4.8 MB) plus the tree it extracts (10.6 MiB outside `bin`, measured on
/// the same host), rounded up. The one figure that is certain, so the one thing
/// a refusal may rest on: a drive without this much room cannot hold Lemonade at
/// all, however small a backend it picks.
pub(crate) const MIN_INSTALL_BYTES: u64 = 16 * 1024 * 1024;

/// What an install is about to do, decided before anything is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InstallPreflight {
    /// The folder the install writes into (`<root>/runtime`, its downloads...).
    pub(crate) target_root: PathBuf,
    /// The Lemonade install the record points at, when it really holds one and
    /// is a different folder: the install is not going to reuse it. Never set
    /// for a record whose folder no longer holds an install, because nothing is
    /// installed there to be "not reused".
    pub(crate) other_install: Option<PathBuf>,
    /// Bytes the install deletes before it extracts: the existing runtime
    /// folder when this install replaces it (`--reinstall`, a new Lemonade
    /// version, or a missing server binary), whose space is free by the time
    /// the download starts. Zero when the tree is kept.
    pub(crate) replaced_bytes: u64,
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
    let other_install = recorded_install_dir(paths)
        .filter(|recorded| !runtime_path_is_same_or_inside(recorded, &target_root));
    // `prepare_embeddable` removes the whole runtime folder, backends included,
    // before it extracts whenever `needs_extraction` says so, so that is the
    // space the install frees first.
    let replaced_bytes = if extraction && runtime_dir.exists() {
        directory_bytes(&runtime_dir)
    } else {
        0
    };
    let available = available_space(&target_root);
    Some(InstallPreflight {
        target_root,
        other_install,
        replaced_bytes,
        available,
    })
}

/// Total size of the files under `dir`, best effort: symlinks are not followed
/// and anything unreadable counts as zero, because this only sizes the space an
/// install is about to free.
fn directory_bytes(dir: &Path) -> u64 {
    let mut total = 0_u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() && !metadata.is_symlink() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
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
                "Installing Lemonade {} into {}",
                rocm_deps::LEMONADE_VERSION,
                self.target_root.display()
            ),
            format!(
                "  With its llama.cpp ROCm backend this is a download of up to about {} and up to about {} of disk space{free}; a smaller backend, if Lemonade picks one for this GPU, needs less. It can take several minutes.",
                download_text(),
                disk_text(),
            ),
        ];
        if self.replaced_bytes > 0 {
            lines.push(format!(
                "  The existing Lemonade install here ({}) is replaced first, and that space is counted as free.",
                format_bytes(self.replaced_bytes)
            ));
        }
        if let Some(other) = self.other_install.as_ref() {
            lines.push(format!(
                "  Lemonade is already installed at {}; that copy is not reused for this location.",
                other.display()
            ));
        }
        lines
    }

    /// Whether the drive can hold the install.
    ///
    /// Refuses only below [`MIN_INSTALL_BYTES`], the one figure that holds for
    /// every backend. The ROCm backend download and the finished tree are upper
    /// bounds, because the backend is not chosen until after the extraction this
    /// runs before, so a host with less room than they need is warned and
    /// allowed to try: refusing it could turn away a host whose smaller backend
    /// would fit.
    pub(crate) fn space_verdict(&self) -> SpaceVerdict {
        const OPERATION: &str = "install Lemonade";
        let Some(free_now) = self.available else {
            return SpaceVerdict::Fits;
        };
        // The install deletes the existing runtime folder before it downloads,
        // so its space counts. The warnings say so, so the figure shown is the
        // one compared.
        let available = free_now.saturating_add(self.replaced_bytes);
        let room = if self.replaced_bytes > 0 {
            format!(
                "{} will be free on the drive holding {} once the existing install there is replaced",
                format_bytes(available),
                self.target_root.display()
            )
        } else {
            format!(
                "only {} is free on the drive holding {}",
                format_bytes(available),
                self.target_root.display()
            )
        };
        let minimum = rocm_core::with_margin(MIN_INSTALL_BYTES);
        if available < minimum {
            return SpaceVerdict::Refuse(format!(
                "{} Nothing was downloaded.",
                insufficient_space_message(OPERATION, &self.target_root, minimum, available)
            ));
        }
        // The margin is working room for the comparison only. What is printed is
        // the same figure the notice printed, so one run never shows two numbers
        // for the same download.
        if available < rocm_core::with_margin(FRESH_INSTALL_DOWNLOAD_BYTES) {
            return SpaceVerdict::Tight(format!(
                "Warning: {room}, and Lemonade's llama.cpp ROCm backend alone is a download of up to about {} (plus room to unpack it). The install will continue: if Lemonade picks a smaller backend for this GPU it may fit, otherwise it will fail partway through.",
                download_text()
            ));
        }
        if available < rocm_core::with_margin(FRESH_INSTALL_DISK_BYTES) {
            return SpaceVerdict::Tight(format!(
                "Warning: free disk space may be insufficient to install Lemonade's llama.cpp ROCm backend: up to about {} needed (plus room to work in), and {room}. The install will continue, but may fail partway through.",
                disk_text()
            ));
        }
        SpaceVerdict::Fits
    }
}

/// The ROCm backend's download as printed everywhere: by the notice and by both
/// warnings, so they cannot disagree.
fn download_text() -> String {
    format_bytes(FRESH_INSTALL_DOWNLOAD_BYTES)
}

/// The finished install's disk use as printed everywhere; see [`download_text`].
fn disk_text() -> String {
    format_bytes(FRESH_INSTALL_DISK_BYTES)
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

/// The runtime folder Lemonade's record points at, only when it still holds an
/// install.
fn recorded_install_dir(paths: &AppPaths) -> Option<PathBuf> {
    read_manifest(paths)
        .ok()
        .map(|manifest| manifest.runtime_dir)
        .filter(|runtime_dir| lemond_path_in(runtime_dir).is_file())
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
        assert!(
            notice.contains(
                "a download of up to about 4.4 GiB and up to about 25.0 GiB of disk space"
            ),
            "{notice}"
        );
        assert!(
            notice.contains("a smaller backend, if Lemonade picks one for this GPU, needs less"),
            "{notice}"
        );
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

    /// A record whose folder lost its install (a runtime uninstalled by a build
    /// that left the record behind) names nothing that is installed, so the
    /// notice must not claim a copy "is already installed" or "not reused".
    #[test]
    fn a_record_naming_a_folder_with_no_install_is_not_announced_as_a_copy() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let gone = temp.path().join("runtimes").join("removed").join("engines");
        record_install_at(&paths, &runtime_dir_in(&lemonade_root(&paths, Some(&gone))));

        let preflight = plan_install_preflight(&paths, None, false, |_| Some(100 * GIB))
            .expect("an empty data folder needs a full install");

        assert_eq!(preflight.other_install, None);
        let notice = preflight.notice_lines().join("\n");
        assert!(!notice.contains("already installed"), "{notice}");
        assert!(!notice.contains("not reused"), "{notice}");
    }

    fn preflight_with(available: Option<u64>, replaced_bytes: u64) -> InstallPreflight {
        InstallPreflight {
            target_root: PathBuf::from("/data/engines/lemonade"),
            other_install: None,
            replaced_bytes,
            available,
        }
    }

    /// Only the one figure that holds for every backend can refuse: the backend
    /// is chosen after this runs, and a host Lemonade gives a smaller backend
    /// must not be turned away for a download it does not need.
    #[test]
    fn only_space_below_the_smallest_install_refuses_and_less_than_the_rocm_backend_warns() {
        const MIB: u64 = 1024 * 1024;
        let preflight = |available| preflight_with(available, 0);
        let SpaceVerdict::Refuse(message) = preflight(Some(2 * MIB)).space_verdict() else {
            panic!("2 MiB cannot hold the Lemonade archive and its extracted files");
        };
        assert!(
            message.contains("not enough free disk space to install Lemonade:"),
            "{message}"
        );
        assert!(message.contains("Nothing was downloaded."), "{message}");

        // Room for Lemonade itself but not for the ROCm backend download: warned
        // and allowed, because a smaller backend may still fit.
        for available in [200 * MIB, 2 * GIB] {
            let SpaceVerdict::Tight(warning) = preflight(Some(available)).space_verdict() else {
                panic!("{available} bytes is below the ROCm backend download: warn, not refuse");
            };
            assert!(
                warning.contains("if Lemonade picks a smaller backend for this GPU it may fit"),
                "{warning}"
            );
        }
        // Room for the download but not the finished tree.
        let SpaceVerdict::Tight(warning) = preflight(Some(10 * GIB)).space_verdict() else {
            panic!("10 GiB is below the finished install: warn");
        };
        assert!(!warning.contains("smaller backend"), "{warning}");
        assert_eq!(
            preflight(Some(40 * GIB)).space_verdict(),
            SpaceVerdict::Fits
        );
        assert_eq!(preflight(None).space_verdict(), SpaceVerdict::Fits);
    }

    /// The notice and both warnings describe the same download and the same
    /// disk use, so they must print the same figures. The margin is for the
    /// comparison only: with it printed, the notice said 4.4 GiB and the warning
    /// about 4.8 GiB for the same backend.
    #[test]
    fn the_notice_and_the_warnings_print_the_same_download_and_disk_figures() {
        let notice = preflight_with(Some(100 * GIB), 0).notice_lines().join("\n");
        let SpaceVerdict::Tight(download_warning) =
            preflight_with(Some(2 * GIB), 0).space_verdict()
        else {
            panic!("2 GiB is below the ROCm backend download");
        };
        let SpaceVerdict::Tight(disk_warning) = preflight_with(Some(10 * GIB), 0).space_verdict()
        else {
            panic!("10 GiB is below the finished install");
        };

        // Derived from the constants, not retyped: if the notice or a warning
        // ever prints a different number, one of these fails.
        let download = format_bytes(FRESH_INSTALL_DOWNLOAD_BYTES);
        let disk = format_bytes(FRESH_INSTALL_DISK_BYTES);
        assert!(
            notice.contains(&format!("a download of up to about {download}")),
            "{notice}"
        );
        assert!(
            notice.contains(&format!("up to about {disk} of disk space")),
            "{notice}"
        );
        assert!(
            download_warning.contains(&format!("a download of up to about {download}")),
            "{download_warning}"
        );
        assert!(
            disk_warning.contains(&format!("up to about {disk} needed")),
            "{disk_warning}"
        );
        // And the margin-inflated values are not what is shown.
        let inflated_download = format_bytes(rocm_core::with_margin(FRESH_INSTALL_DOWNLOAD_BYTES));
        let inflated_disk = format_bytes(rocm_core::with_margin(FRESH_INSTALL_DISK_BYTES));
        assert_ne!(inflated_download, download);
        assert_ne!(inflated_disk, disk);
        assert!(
            !download_warning.contains(&inflated_download),
            "{download_warning}"
        );
        assert!(!disk_warning.contains(&inflated_disk), "{disk_warning}");
        // The figures themselves, so a constant change is a visible edit here too.
        assert_eq!((download.as_str(), disk.as_str()), ("4.4 GiB", "25.0 GiB"));
    }

    /// An install that replaces the existing runtime folder deletes it before
    /// downloading (`prepare_embeddable` removes the whole folder, backends
    /// included), so its space is free by then. The preflight sizes that folder,
    /// counts it, and says so; an install that keeps the tree frees nothing.
    #[test]
    fn a_reinstall_counts_the_space_of_the_install_it_replaces() {
        let temp = tempfile::tempdir().unwrap();
        let paths = paths_in(temp.path());
        let tree = lemonade_root(&paths, None);
        plant_current_tree(&tree);
        let payload = runtime_dir_in(&tree)
            .join("bin")
            .join("llamacpp")
            .join("rocm-stable");
        fs::write(payload.join("payload.bin"), vec![0_u8; 123_456]).unwrap();

        let reinstall = plan_install_preflight(&paths, None, true, |_| Some(2 * GIB)).unwrap();
        assert!(
            reinstall.replaced_bytes >= 123_456,
            "{}",
            reinstall.replaced_bytes
        );
        let notice = reinstall.notice_lines().join("\n");
        assert!(
            notice.contains(&format!(
                "The existing Lemonade install here ({}) is replaced first, and that space is counted as free.",
                format_bytes(reinstall.replaced_bytes)
            )),
            "{notice}"
        );

        // A current tree whose backend is missing is topped up, not wiped.
        fs::remove_dir_all(runtime_dir_in(&tree).join("bin")).unwrap();
        let kept = plan_install_preflight(&paths, None, false, |_| Some(2 * GIB)).unwrap();
        assert_eq!(kept.replaced_bytes, 0);
        assert!(!kept.notice_lines().join("\n").contains("replaced first"));

        // The verdict counts it: 2 GiB free warns for the ROCm backend, but the
        // same drive with a 24 GiB install about to be deleted does not.
        assert!(matches!(
            preflight_with(Some(2 * GIB), 0).space_verdict(),
            SpaceVerdict::Tight(_)
        ));
        assert_eq!(
            preflight_with(Some(2 * GIB), 30 * GIB).space_verdict(),
            SpaceVerdict::Fits
        );
        // When it still does not fit, the warning shows the figure compared.
        let SpaceVerdict::Tight(warning) = preflight_with(Some(GIB), 2 * GIB).space_verdict()
        else {
            panic!("3 GiB is below the ROCm backend download");
        };
        assert!(
            warning.contains("3.0 GiB will be free on the drive holding /data/engines/lemonade once the existing install there is replaced"),
            "{warning}"
        );
    }
}
