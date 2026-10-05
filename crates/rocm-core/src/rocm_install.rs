// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::examine::extract_rocm_version;
use crate::runtime_is_windows;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRocmSummary {
    pub status: String,
    pub paths: Vec<PathBuf>,
    pub detail: Option<String>,
    /// Version of the best-ranked detected install, when one could be read.
    ///
    /// The resolver establishes this already; without carrying it here the human
    /// report named a path but never a version, so a machine with ROCm 7.14
    /// installed could not tell you which ROCm it had.
    ///
    /// `None` when no install was found, or when one was found whose layout
    /// declares no version anywhere.
    ///
    /// Optional and defaulted so the daemon's serialised snapshot
    /// (`apps/rocmd/src/lib.rs`) written before this field existed still loads.
    #[serde(default)]
    pub version: Option<String>,
}

/// Directories that hold conventional unmanaged ROCm install roots on Linux.
///
/// Each is searched for both the unversioned `rocm` root and the versioned
/// `rocm-X.Y[.Z]` siblings that a side-by-side install produces.
const LINUX_ROCM_SEARCH_DIRS: &[&str] = &["/opt", "/usr/local"];

/// Directories the Windows HIP SDK installer writes into.
///
/// Each is an install root in its own right and also the parent of versioned
/// installs — see [`RocmLayout::Children`].
const WINDOWS_ROCM_SEARCH_DIRS: &[&str] = &[r"C:\Program Files\AMD\ROCm", r"C:\Program Files\ROCm"];

/// How versioned installs are arranged under a search directory.
///
/// The two platforms disagree, so the resolver has to be told which shape it is
/// walking rather than assuming one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RocmLayout {
    /// `<dir>/rocm` alongside `rocm-X.Y[.Z]` siblings, e.g. `/opt/rocm` and
    /// `/opt/rocm-6.4.1`.
    Siblings,
    /// `<dir>` itself, with versions as bare `X.Y[.Z]` children, e.g.
    /// `C:\Program Files\AMD\ROCm` and `C:\Program Files\AMD\ROCm\6.4`.
    Children,
}

/// An unmanaged ("legacy") ROCm install root found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RocmInstall {
    pub(crate) path: PathBuf,
    pub(crate) version: Option<String>,
}

/// The search roots and layout this host's installs use.
fn host_rocm_search() -> (Vec<PathBuf>, RocmLayout) {
    let (dirs, layout) = if runtime_is_windows() {
        (WINDOWS_ROCM_SEARCH_DIRS, RocmLayout::Children)
    } else {
        (LINUX_ROCM_SEARCH_DIRS, RocmLayout::Siblings)
    };
    (dirs.iter().map(PathBuf::from).collect(), layout)
}

/// Every unmanaged ROCm install on this host, best candidate first.
///
/// Supplies the platform's search roots, layout, and `$ROCM_PATH` to
/// [`discover_rocm_installs_in_layout`].
pub(crate) fn discover_rocm_installs() -> Vec<RocmInstall> {
    let env_override = std::env::var_os("ROCM_PATH").map(PathBuf::from);
    let (search_dirs, layout) = host_rocm_search();
    discover_rocm_installs_in_layout(&search_dirs, env_override.as_deref(), layout)
}

/// [`discover_rocm_installs`] with the host's `$ROCM_PATH` and search roots
/// supplied by the caller instead of read from the process.
///
/// Reading `$ROCM_PATH` inside [`discover_rocm_installs`] is what forced its
/// callers' tests to mutate the process environment, and two of them racing on
/// that global is what made the Windows lane flake — that lane runs `cargo test`
/// (threads in one process) where `Test (affected crates)` runs `cargo nextest`
/// (a process per test), so only Windows observed it. Not every Linux lane uses
/// nextest, so that is where it fired rather than where it could. Passing the
/// override in keeps those tests hermetic. Same seam as [`discover_rocm_installs_in`], but
/// keeps the host's layout so the caller under test is the real one.
#[cfg(test)]
pub(crate) fn discover_rocm_installs_on_host_in(
    search_dirs: &[PathBuf],
    env_override: Option<&Path>,
) -> Vec<RocmInstall> {
    discover_rocm_installs_in_layout(search_dirs, env_override, host_rocm_search().1)
}

/// [`discover_rocm_installs_in_layout`] for the Linux sibling layout.
///
/// Production callers go through [`discover_rocm_installs`], which picks the
/// layout for the host; this spares the sibling-layout tests from restating it.
#[cfg(test)]
pub(crate) fn discover_rocm_installs_in(
    search_dirs: &[PathBuf],
    env_override: Option<&Path>,
) -> Vec<RocmInstall> {
    discover_rocm_installs_in_layout(search_dirs, env_override, RocmLayout::Siblings)
}

/// Rank the ROCm installs reachable from `search_dirs`, best candidate first.
///
/// Precedence, highest first:
///
/// 1. `env_override` (`$ROCM_PATH`) — an install the user named explicitly wins
///    over anything found by convention.
/// 2. The conventional active root, which is what lands on `PATH` and in the
///    linker cache on a normal install: `<dir>/rocm` under [`RocmLayout::Siblings`],
///    or `<dir>` itself under [`RocmLayout::Children`].
/// 3. Versioned installs, newest first.
///
/// Versions are ordered by numeric component, not lexically, so `6.10` beats
/// `6.2` — on both layouts. Candidates are deduplicated by canonical path, so
/// the common `/opt/rocm -> /opt/rocm-6.4.1` symlink yields one install rather
/// than two.
pub(crate) fn discover_rocm_installs_in_layout(
    search_dirs: &[PathBuf],
    env_override: Option<&Path>,
    layout: RocmLayout,
) -> Vec<RocmInstall> {
    let mut found: Vec<RocmInstall> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();

    if let Some(path) = env_override {
        push_rocm_candidate(path.to_path_buf(), &mut found, &mut seen);
    }

    for dir in search_dirs {
        let root = match layout {
            RocmLayout::Siblings => dir.join("rocm"),
            RocmLayout::Children => dir.clone(),
        };
        push_rocm_candidate(root, &mut found, &mut seen);
    }

    for dir in search_dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        let version_of = |path: &Path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| versioned_dir_version(name, layout))
        };
        let mut versioned: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| version_of(path).is_some())
            .collect();
        // Newest first, by numeric version component.
        versioned.sort_by(|a, b| {
            let key = |path: &Path| {
                version_of(path)
                    .map(|version| rocm_version_sort_key(&version))
                    .unwrap_or_default()
            };
            key(b).cmp(&key(a))
        });
        for candidate in versioned {
            push_rocm_candidate(candidate, &mut found, &mut seen);
        }
    }

    found
}

/// Record `candidate` as an install when it looks like one and is not already
/// recorded under another name (a symlink to a root already seen).
fn push_rocm_candidate(candidate: PathBuf, found: &mut Vec<RocmInstall>, seen: &mut Vec<PathBuf>) {
    if !legacy_rocm_candidate_exists(&candidate) {
        return;
    }
    let key = fs::canonicalize(&candidate).unwrap_or_else(|_| candidate.clone());
    if seen.contains(&key) {
        return;
    }
    seen.push(key);
    let version = rocm_install_version(&candidate);
    found.push(RocmInstall {
        path: candidate,
        version,
    });
}

/// The ROCm version an install root reports, if it reports one.
///
/// Prefers the `.info/version*` files the packaged installs ship, and falls
/// back to the version embedded in the resolved directory name — either a
/// `rocm-X.Y[.Z]` root (or the symlink pointing at one) or the bare `X.Y` the
/// Windows installer uses.
pub(crate) fn rocm_install_version(root: &Path) -> Option<String> {
    for name in ["version", "version-utils", "version-libs"] {
        let file = root.join(".info").join(name);
        if let Ok(text) = fs::read_to_string(&file) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    // Only the final component: an ancestor directory may itself contain
    // "rocm-" (a checkout, a home directory) and must not be mistaken for the
    // install's version. Canonicalizing first resolves `/opt/rocm` to the
    // `rocm-X.Y.Z` it points at.
    let resolved = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let name = resolved.file_name().and_then(|name| name.to_str())?;
    extract_rocm_version(name).or_else(|| bare_version(name))
}

/// The version a directory *name* advertises, for the given layout, or `None`
/// when the name is not a versioned install directory at all.
///
/// The two layouts name their versioned directories differently: Linux uses
/// `rocm-6.4.1` siblings, while the Windows installer uses a bare `6.4` under
/// its ROCm root. Matching the wrong shape is not harmless — accepting a bare
/// number on Linux would sweep in unrelated `/opt` and `/usr/local` entries.
fn versioned_dir_version(name: &str, layout: RocmLayout) -> Option<String> {
    match layout {
        RocmLayout::Siblings => extract_rocm_version(name),
        RocmLayout::Children => bare_version(name),
    }
}

/// `X.Y[.Z]` when `name` is exactly a dotted numeric version, else `None`.
///
/// Requires at least one dot so a lone number cannot pass, and every component
/// to be numeric so a directory such as `6.4-beta` or `docs` is rejected.
fn bare_version(name: &str) -> Option<String> {
    let mut parts = name.split('.');
    let mut count = 0;
    for part in &mut parts {
        if part.is_empty() || !part.chars().all(|ch| ch.is_ascii_digit()) {
            return None;
        }
        count += 1;
    }
    (count >= 2).then(|| name.to_owned())
}

/// Sort key that orders ROCm versions numerically: `6.10` outranks `6.2`.
fn rocm_version_sort_key(version: &str) -> Vec<u32> {
    version
        .split('.')
        .map(|part| part.parse::<u32>().unwrap_or(0))
        .collect()
}

pub fn detect_legacy_rocm_summary() -> LegacyRocmSummary {
    // One resolver on both platforms, so the human report, the JSON probe and
    // the fix-6 runner cannot disagree about which installs exist or which one
    // is active. `discover_rocm_installs` picks the search roots and layout for
    // the host.
    let installs = discover_rocm_installs();
    // The resolver ranks best-first, so the leading install's version is the one
    // that describes this machine. Keeping it costs nothing here and is the
    // difference between naming a path and naming the ROCm the user has.
    let version = installs.first().and_then(|install| install.version.clone());
    let paths: Vec<PathBuf> = installs.into_iter().map(|install| install.path).collect();

    let status = if paths.is_empty() {
        "not_detected"
    } else {
        "detected_unmanaged"
    };
    let detail = if paths.is_empty() {
        None
    } else {
        Some("legacy ROCm installs are reported for compatibility only; rocm-cli manages TheRock runtimes separately".to_owned())
    };

    LegacyRocmSummary {
        status: status.to_owned(),
        paths,
        detail,
        version,
    }
}

/// The best unmanaged ("legacy") ROCm install on this host, if any.
///
/// [`detect_legacy_rocm_summary`] resolves the same thing but as part of the
/// full [`ExamineSummary`] probe; this is the standalone version+path for a
/// caller like `rocm version` that wants just the SDK identity as a fallback
/// once it's already established no managed TheRock runtime is active.
pub fn detect_legacy_rocm_sdk() -> Option<(String, PathBuf)> {
    let install = discover_rocm_installs().into_iter().next()?;
    let version = install.version?;
    Some((version, install.path))
}

#[allow(clippy::case_sensitive_file_extension_comparisons)] // ROCm installs the runtime DLL as lowercase `amdhip64.dll`
fn legacy_rocm_candidate_exists(candidate: &Path) -> bool {
    if !candidate.exists() {
        return false;
    }
    if [
        candidate.join("bin").join("rocminfo"),
        candidate.join("bin").join("rocminfo.exe"),
        candidate.join("bin").join("hipcc"),
        candidate.join("bin").join("hipcc.bat"),
        candidate.join("lib").join("libamdhip64.so"),
        candidate.join("lib").join("libhsa-runtime64.so"),
        candidate.join(".info").join("version"),
    ]
    .iter()
    .any(|marker| marker.exists())
    {
        return true;
    }

    fs::read_dir(candidate.join("bin"))
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("amdhip64") && name.ends_with(".dll"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use anyhow::Result;

    #[test]
    fn linux_legacy_rocm_detection_ignores_rocdxg_only_directory() -> Result<()> {
        let (root, _) = temp_app_paths("rocdxg-only");
        let rocm = root.join("rocm");
        fs::create_dir_all(rocm.join("lib"))?;
        fs::write(rocm.join("lib").join("librocdxg.so"), "")?;

        assert!(!legacy_rocm_candidate_exists(&rocm));

        fs::create_dir_all(rocm.join("bin"))?;
        fs::write(rocm.join("bin").join("rocminfo"), "")?;
        assert!(legacy_rocm_candidate_exists(&rocm));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// Plant a directory that `legacy_rocm_candidate_exists` accepts as a real
    /// install, optionally shipping the `.info/version` a packaged install has.
    fn fake_rocm_install(root: &Path, info_version: Option<&str>) -> Result<()> {
        fs::create_dir_all(root.join("bin"))?;
        fs::write(root.join("bin").join("rocminfo"), "")?;
        if let Some(version) = info_version {
            fs::create_dir_all(root.join(".info"))?;
            fs::write(root.join(".info").join("version"), version)?;
        }
        Ok(())
    }

    #[test]
    fn discovers_versioned_rocm_install_without_a_default_root() -> Result<()> {
        // A box whose only ROCm lives at /opt/rocm-6.4.1: before the shared
        // resolver this reported nothing at all, which silently disabled the
        // structural scoring in fix-3, fix-6 and fix-8.
        let (root, _) = temp_app_paths("rocm-versioned-only");
        let opt = root.join("opt");
        fake_rocm_install(&opt.join("rocm-6.4.1"), None)?;

        let found = discover_rocm_installs_in(std::slice::from_ref(&opt), None);

        assert_eq!(found.len(), 1, "expected exactly one install: {found:?}");
        assert_eq!(found[0].path, opt.join("rocm-6.4.1"));
        assert_eq!(
            found[0].version.as_deref(),
            Some("6.4.1"),
            "version must fall back to the directory name"
        );

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn newest_rocm_install_is_chosen_numerically_not_lexically() -> Result<()> {
        // 6.10 outranks 6.2. A lexicographic sort -- the bug still live in the
        // Windows scans -- would pick 6.2 here.
        let (root, _) = temp_app_paths("rocm-newest");
        let opt = root.join("opt");
        fake_rocm_install(&opt.join("rocm-6.2.0"), None)?;
        fake_rocm_install(&opt.join("rocm-6.10.0"), None)?;

        let found = discover_rocm_installs_in(std::slice::from_ref(&opt), None);

        assert_eq!(
            found.len(),
            2,
            "both installs should be reported: {found:?}"
        );
        assert_eq!(
            found[0].path,
            opt.join("rocm-6.10.0"),
            "6.10 must outrank 6.2"
        );

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn explicitly_selected_rocm_install_outranks_the_default_root() -> Result<()> {
        // $ROCM_PATH is the user naming an install; it must beat the
        // conventional root, which is the precedence examine had inverted.
        let (root, _) = temp_app_paths("rocm-env-override");
        let opt = root.join("opt");
        fake_rocm_install(&opt.join("rocm"), None)?;
        let chosen = root.join("elsewhere").join("rocm-6.4.1");
        fake_rocm_install(&chosen, None)?;

        let found = discover_rocm_installs_in(&[opt], Some(&chosen));

        assert_eq!(found[0].path, chosen);

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn conventional_rocm_root_outranks_versioned_siblings() -> Result<()> {
        // Guards the claim that versioned support is purely additive: on a
        // conventional box the answer must not change.
        let (root, _) = temp_app_paths("rocm-conventional");
        let opt = root.join("opt");
        fake_rocm_install(&opt.join("rocm"), None)?;
        fake_rocm_install(&opt.join("rocm-6.2.0"), None)?;

        let found = discover_rocm_installs_in(std::slice::from_ref(&opt), None);

        assert_eq!(found[0].path, opt.join("rocm"));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_default_root_symlinked_to_a_versioned_one_is_reported_once() -> Result<()> {
        // The common packaged layout: /opt/rocm -> /opt/rocm-6.4.1. One
        // install, reported once, carrying the version from the target.
        let (root, _) = temp_app_paths("rocm-symlink");
        let opt = root.join("opt");
        let versioned = opt.join("rocm-6.4.1");
        fake_rocm_install(&versioned, None)?;
        std::os::unix::fs::symlink(&versioned, opt.join("rocm"))?;

        let found = discover_rocm_installs_in(std::slice::from_ref(&opt), None);

        assert_eq!(
            found.len(),
            1,
            "symlink and target are one install: {found:?}"
        );
        assert_eq!(found[0].path, opt.join("rocm"));
        assert_eq!(found[0].version.as_deref(), Some("6.4.1"));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn a_host_without_rocm_reports_no_installs() -> Result<()> {
        let (root, _) = temp_app_paths("rocm-absent");
        let opt = root.join("opt");
        fs::create_dir_all(opt.join("rocm-6.4.1"))?; // no marker files
        fs::create_dir_all(&opt)?;

        assert!(discover_rocm_installs_in(&[opt], None).is_empty());

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn discovers_windows_versioned_installs_under_the_rocm_root() -> Result<()> {
        // The Windows installer nests versions under its ROCm directory as bare
        // `X.Y` children, rather than the `rocm-X.Y` siblings Linux uses. Before
        // the layout split, only Linux was searched, so a Windows box with no
        // ROCM_PATH reported no install at all.
        let (root, _) = temp_app_paths("rocm-windows-versioned");
        let base = root.join("ROCm");
        fake_rocm_install(&base.join("6.2"), None)?;
        fake_rocm_install(&base.join("6.10"), None)?;

        let found = discover_rocm_installs_in_layout(
            std::slice::from_ref(&base),
            None,
            RocmLayout::Children,
        );

        let paths: Vec<_> = found.iter().map(|install| install.path.clone()).collect();
        let versions: Vec<_> = found
            .iter()
            .map(|install| install.version.clone())
            .collect();
        fs::remove_dir_all(root).ok();

        assert_eq!(
            paths,
            vec![base.join("6.10"), base.join("6.2")],
            "newest first, and 6.10 outranks 6.2 numerically rather than lexically"
        );
        assert_eq!(
            versions,
            vec![Some("6.10".to_owned()), Some("6.2".to_owned())],
            "the bare directory name is the version on this layout"
        );
        Ok(())
    }

    #[test]
    fn windows_layout_treats_the_rocm_root_itself_as_an_install() -> Result<()> {
        // A single-version HIP SDK puts the install directly in the ROCm root
        // with no version subdirectory, and it must outrank any versioned child.
        let (root, _) = temp_app_paths("rocm-windows-root");
        let base = root.join("ROCm");
        fake_rocm_install(&base, None)?;
        fake_rocm_install(&base.join("6.4"), None)?;

        let found = discover_rocm_installs_in_layout(
            std::slice::from_ref(&base),
            None,
            RocmLayout::Children,
        );
        let first = found.first().map(|install| install.path.clone());
        fs::remove_dir_all(root).ok();

        assert_eq!(first, Some(base), "the conventional root wins");
        Ok(())
    }

    #[test]
    fn a_bare_version_directory_is_not_an_install_on_the_linux_layout() -> Result<()> {
        // Guard the layouts against each other: accepting bare numbers under the
        // sibling layout would sweep in unrelated /opt and /usr/local entries.
        let (root, _) = temp_app_paths("rocm-bare-version-linux");
        let opt = root.join("opt");
        fake_rocm_install(&opt.join("6.4"), None)?;

        let found = discover_rocm_installs_in(std::slice::from_ref(&opt), None);
        fs::remove_dir_all(root).ok();

        assert!(found.is_empty(), "expected no installs, got {found:?}");
        Ok(())
    }

    #[test]
    fn bare_version_accepts_only_dotted_numbers() {
        assert_eq!(bare_version("6.4"), Some("6.4".to_owned()));
        assert_eq!(bare_version("6.4.1"), Some("6.4.1".to_owned()));
        // A lone number is a directory name, not a version.
        assert_eq!(bare_version("6"), None);
        // Anything non-numeric in any component disqualifies it.
        assert_eq!(bare_version("6.4-beta"), None);
        assert_eq!(bare_version("docs"), None);
        assert_eq!(bare_version("6."), None);
        assert_eq!(bare_version(".4"), None);
        assert_eq!(bare_version(""), None);
    }

    #[test]
    fn packaged_version_file_wins_over_the_directory_name() -> Result<()> {
        // A repackaged tree can sit in a differently-named directory; the
        // shipped .info/version is authoritative.
        let (root, _) = temp_app_paths("rocm-info-version");
        let install = root.join("opt").join("rocm-6.2.0");
        fake_rocm_install(&install, Some("6.4.1\n"))?;

        assert_eq!(rocm_install_version(&install).as_deref(), Some("6.4.1"));

        fs::remove_dir_all(root).ok();
        Ok(())
    }
}
