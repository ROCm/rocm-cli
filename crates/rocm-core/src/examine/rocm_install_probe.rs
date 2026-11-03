// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The ROCm install and package-repository probe: how ROCm was installed, and
//! which live repository files configure it.

use super::Examination;
use crate::probe_exec::{MEDIUM, run, which};
use std::path::Path;

/// AMD ROCm/AMDGPU repo files: `amdgpu.list` / `amdgpu.repo` are written by
/// rocm-cli's own repo-native install flow; `rocm.list` / `radeon.list` /
/// `rocm.repo` are names from AMD's documented manual/legacy setup steps that
/// rocm-cli itself never creates. Either kind marks a package-manager-repo-
/// managed ROCm.
const REPO_NATIVE_INSTALL_MARKERS: &[&str] = &[
    "/etc/apt/sources.list.d/amdgpu.list",
    "/etc/apt/sources.list.d/rocm.list",
    "/etc/apt/sources.list.d/radeon.list",
    "/etc/yum.repos.d/amdgpu.repo",
    "/etc/yum.repos.d/rocm.repo",
    "/etc/zypp/repos.d/amdgpu.repo",
    "/etc/zypp/repos.d/rocm.repo",
];

/// Pure marker-path -> install-method mapping, split out from
/// `probe_rocm_install` so it's testable without touching the real
/// filesystem. `existing` is the subset of `REPO_NATIVE_INSTALL_MARKERS`
/// found present on the host; returns `None` when none are present.
fn repo_native_method_from_markers(existing: &[&str]) -> Option<(&'static str, Vec<String>)> {
    let seen: Vec<String> = REPO_NATIVE_INSTALL_MARKERS
        .iter()
        .filter(|m| existing.contains(m))
        .map(|m| (*m).to_owned())
        .collect();
    if seen.is_empty() {
        None
    } else {
        Some(("repo-native", seen))
    }
}

/// Whether a lowercased repo-directory entry name is a live ROCm/AMDGPU repo
/// file, as opposed to one fix-7/fix-12 already told the user to quarantine
/// (`mv foo foo.bak`). Without this, `amdgpu.list.bak` sitting next to the
/// reinstalled `amdgpu.list` after following that advice counts as a second
/// stale repo and re-triggers the same fix.
// `name` is already lowercased by the caller, so a plain suffix match is fine.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn is_live_rocm_repo_file_name(name: &str) -> bool {
    // Allowlist the known repo-file extensions (apt's one-line-style `.list`,
    // apt's deb822-style `.sources`, dnf/zypper's `.repo`) rather than
    // denylist known quarantine suffixes: quarantining appends to the name
    // (`amdgpu.list` -> `amdgpu.list.bak`), so this also excludes any
    // quarantine convention this file doesn't know about, and stops an
    // unrelated file that merely mentions "rocm"/"amdgpu" in its name (a
    // note, a log) from being swept up as a live repo file.
    (name.ends_with(".list") || name.ends_with(".repo") || name.ends_with(".sources"))
        && (name.contains("rocm") || name.contains("amdgpu") || name.contains("radeon"))
}

/// Full paths of live (non-quarantined) ROCm/AMDGPU repo files directly inside
/// `dir`, split out from `probe_rocm_install` so the `.bak`/`.save`/`.disabled`
/// skip actually gets exercised against a real directory listing, not just the
/// pure `is_live_rocm_repo_file_name` name check in isolation. Silently empty
/// when `dir` does not exist or is not a directory.
fn live_rocm_repo_paths_in_dir(dir: &str) -> Vec<String> {
    // Read through `host_path` so a simulated host root re-roots the scan, but
    // report the logical `dir/name`, which is what a user would `mv`.
    let Ok(entries) = std::fs::read_dir(crate::host_path(dir)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| {
            is_live_rocm_repo_file_name(&entry.file_name().to_string_lossy().to_lowercase())
        })
        .map(|entry| {
            Path::new(dir)
                .join(entry.file_name())
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

pub(super) fn probe_rocm_install(e: &mut Examination) {
    // Shared with the human `rocm examine` report and the fix-6 PATH runner so
    // all three agree on which install is the active one. Handles versioned
    // roots (`/opt/rocm-6.4.1`) as well as the conventional `/opt/rocm`.
    let install = crate::discover_rocm_installs().into_iter().next();
    let rocm_dir = install
        .as_ref()
        .map(|install| install.path.to_string_lossy().into_owned())
        .unwrap_or_default();
    e.rocm_path = rocm_dir.clone();
    e.rocm_version = install
        .and_then(|install| install.version)
        .unwrap_or_default();

    let existing_markers: Vec<&str> = REPO_NATIVE_INSTALL_MARKERS
        .iter()
        .copied()
        .filter(|m| Path::new(m).exists())
        .collect();
    if let Some((method, seen)) = repo_native_method_from_markers(&existing_markers) {
        e.rocm_install_method = method.to_owned();
        e.rocm_repos_seen.extend(seen);
    }

    if e.rocm_install_method.is_empty() {
        if which("dpkg") {
            let (rc, out, _) = run("dpkg", &["-l", "rocm-hip-runtime"], MEDIUM);
            if rc == 0 && out.contains("rocm-hip-runtime") {
                e.rocm_install_method = "apt".to_owned();
            }
        }
        if e.rocm_install_method.is_empty() && which("rpm") {
            let (rc, out, _) = run("rpm", &["-q", "rocm-hip-runtime"], MEDIUM);
            if rc == 0 && out.contains("rocm-hip-runtime") {
                e.rocm_install_method = "dnf".to_owned();
            }
        }
    }
    if e.rocm_install_method.is_empty() {
        e.rocm_install_method = if rocm_dir.is_empty() {
            "none".to_owned()
        } else {
            "runfile-or-tarball".to_owned()
        };
    }

    merge_live_repo_paths(e, &REPO_SCAN_DIRS);
}

/// The repo directories `probe_rocm_install` scans for live (non-quarantined)
/// ROCm/AMDGPU repo files, named so a test can assert each family's directory
/// is actually present rather than only being provable by reading real system
/// paths (SLES repo detection via `/etc/zypp/repos.d` is new in this PR).
const REPO_SCAN_DIRS: [&str; 3] = [
    "/etc/apt/sources.list.d",
    "/etc/yum.repos.d",
    "/etc/zypp/repos.d",
];

/// Split out from `probe_rocm_install` so the directory loop itself -- not
/// just `is_live_rocm_repo_file_name` and `live_rocm_repo_paths_in_dir` in
/// isolation -- is driven by a test against real (temp) directories standing
/// in for each family's repo path.
fn merge_live_repo_paths(e: &mut Examination, dirs: &[&str]) {
    for dir in dirs {
        for full in live_rocm_repo_paths_in_dir(dir) {
            if !e.rocm_repos_seen.contains(&full) {
                e.rocm_repos_seen.push(full);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_live_rocm_repo_file_name_skips_quarantined_files() {
        assert!(is_live_rocm_repo_file_name("amdgpu.list"));
        assert!(is_live_rocm_repo_file_name("rocm.repo"));
        assert!(!is_live_rocm_repo_file_name("amdgpu.list.bak"));
        assert!(!is_live_rocm_repo_file_name("amdgpu.list.save"));
        assert!(!is_live_rocm_repo_file_name("amdgpu.list.disabled"));
        assert!(!is_live_rocm_repo_file_name("unrelated.list"));
        // Allowlisted by extension, not just denylisted by quarantine suffix:
        // a stray file that merely mentions "rocm" isn't a repo file.
        assert!(!is_live_rocm_repo_file_name("rocm-notes.txt"));
    }

    #[test]
    fn is_live_rocm_repo_file_name_recognizes_deb822_sources_files() {
        // Newer apt defaults to the deb822 `.sources` format (as opposed to
        // the one-line-style `.list`); fix-7's own `ls | grep` advice has no
        // extension filter and already matches these, so this filter must
        // not be the one place that silently misses them.
        assert!(is_live_rocm_repo_file_name("amdgpu.sources"));
        assert!(!is_live_rocm_repo_file_name("amdgpu.sources.bak"));
    }

    #[test]
    fn live_rocm_repo_paths_in_dir_skips_a_real_bak_file_on_disk() {
        // `is_live_rocm_repo_file_name` above only proves the name filter is
        // correct in isolation; this drives an actual `.bak` file (the kind
        // fix-7/fix-12 quarantine advice produces) through the directory scan
        // `probe_rocm_install` runs, closing the gap where only the live file
        // being listed first would still pass if the `.bak` skip were deleted.
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(dir.path().join("amdgpu.list"), "deb ...\n").expect("write live repo");
        std::fs::write(dir.path().join("amdgpu.list.bak"), "deb ...\n")
            .expect("write quarantined repo");
        std::fs::write(dir.path().join("unrelated.list"), "deb ...\n").expect("write unrelated");

        let mut seen = live_rocm_repo_paths_in_dir(&dir.path().to_string_lossy());
        seen.sort();

        assert_eq!(seen.len(), 1, "seen: {seen:?}");
        assert!(seen[0].ends_with("amdgpu.list"));
    }

    #[test]
    fn repo_scan_dirs_covers_all_three_package_families() {
        // Mutation-proofs the literal dir list `probe_rocm_install` scans:
        // deleting the `/etc/zypp/repos.d` entry (SLES repo detection, new in
        // this PR) left every existing test green because nothing previously
        // asserted this list's contents directly.
        assert!(REPO_SCAN_DIRS.contains(&"/etc/apt/sources.list.d"));
        assert!(REPO_SCAN_DIRS.contains(&"/etc/yum.repos.d"));
        assert!(REPO_SCAN_DIRS.contains(&"/etc/zypp/repos.d"));
    }

    #[test]
    fn merge_live_repo_paths_scans_every_family_dir_including_zypp() {
        // Drives `probe_rocm_install`'s actual scan loop (not just its two
        // helpers in isolation) across temp directories standing in for all
        // three package families, including the zypp one a mutation run
        // showed was silently unscanned by every pre-existing test.
        let apt_dir = tempfile::tempdir().expect("create apt temp dir");
        let yum_dir = tempfile::tempdir().expect("create yum temp dir");
        let zypp_dir = tempfile::tempdir().expect("create zypp temp dir");
        std::fs::write(apt_dir.path().join("amdgpu.list"), "deb ...\n").expect("write apt repo");
        std::fs::write(yum_dir.path().join("amdgpu.repo"), "[amdgpu]\n").expect("write yum repo");
        std::fs::write(zypp_dir.path().join("amdgpu.repo"), "[amdgpu]\n").expect("write zypp repo");

        let mut e = Examination::default();
        let dirs = [
            apt_dir.path().to_string_lossy().into_owned(),
            yum_dir.path().to_string_lossy().into_owned(),
            zypp_dir.path().to_string_lossy().into_owned(),
        ];
        let dir_refs: Vec<&str> = dirs.iter().map(String::as_str).collect();
        merge_live_repo_paths(&mut e, &dir_refs);

        assert_eq!(e.rocm_repos_seen.len(), 3, "seen: {:?}", e.rocm_repos_seen);
        assert!(
            e.rocm_repos_seen
                .iter()
                .any(|p| p.starts_with(dirs[0].as_str()))
        );
        assert!(
            e.rocm_repos_seen
                .iter()
                .any(|p| p.starts_with(dirs[1].as_str()))
        );
        assert!(
            e.rocm_repos_seen
                .iter()
                .any(|p| p.starts_with(dirs[2].as_str()))
        );
    }

    /// `probe_rocm_install` must scan every family's repo directory: narrowing
    /// its call to fewer of `REPO_SCAN_DIRS` leaves the constant and
    /// `merge_live_repo_paths` tests green, so this drives the probe itself
    /// against a simulated host root with a repo file in each directory.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn probe_rocm_install_scans_every_repo_family_directory() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = tempfile::tempdir().expect("create host root");
        for (dir, file) in [
            ("etc/apt/sources.list.d", "amdgpu.list"),
            ("etc/yum.repos.d", "amdgpu.repo"),
            ("etc/zypp/repos.d", "amdgpu.repo"),
        ] {
            std::fs::create_dir_all(root.path().join(dir)).expect("create repo dir");
            std::fs::write(root.path().join(dir).join(file), "x\n").expect("write repo file");
        }
        let mut e = Examination::default();
        {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, root.path());
            probe_rocm_install(&mut e);
        }
        for want in [
            "/etc/apt/sources.list.d/amdgpu.list",
            "/etc/yum.repos.d/amdgpu.repo",
            "/etc/zypp/repos.d/amdgpu.repo",
        ] {
            assert!(
                e.rocm_repos_seen.iter().any(|p| p == want),
                "missing {want}: {:?}",
                e.rocm_repos_seen
            );
        }
    }

    #[test]
    fn merge_live_repo_paths_does_not_duplicate_an_already_seen_path() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(dir.path().join("amdgpu.repo"), "[amdgpu]\n").expect("write repo");
        let dir_str = dir.path().to_string_lossy().into_owned();

        let mut e = Examination::default();
        // `join`, not a hardcoded `/`: the scan records `entry.path()`, which
        // uses the platform separator.
        e.rocm_repos_seen.push(
            dir.path()
                .join("amdgpu.repo")
                .to_string_lossy()
                .into_owned(),
        );
        merge_live_repo_paths(&mut e, &[dir_str.as_str()]);

        assert_eq!(
            e.rocm_repos_seen.len(),
            1,
            "an already-recorded path must not be duplicated: {:?}",
            e.rocm_repos_seen
        );
    }

    #[test]
    fn no_markers_present_yields_no_repo_native_method() {
        assert!(repo_native_method_from_markers(&[]).is_none());
    }

    #[test]
    fn every_declared_marker_individually_maps_to_repo_native() {
        for marker in REPO_NATIVE_INSTALL_MARKERS {
            let (method, seen) =
                repo_native_method_from_markers(&[marker]).expect("marker should match");
            assert_eq!(method, "repo-native");
            assert_eq!(seen, vec![(*marker).to_owned()]);
        }
    }

    #[test]
    fn multiple_markers_all_recorded_in_declaration_order() {
        // Passed in reverse of REPO_NATIVE_INSTALL_MARKERS' declared order to
        // confirm the output order tracks the constant, not the input.
        let existing = [
            "/etc/zypp/repos.d/amdgpu.repo",
            "/etc/apt/sources.list.d/amdgpu.list",
        ];
        let (method, seen) = repo_native_method_from_markers(&existing).expect("some");
        assert_eq!(method, "repo-native");
        assert_eq!(
            seen,
            vec![
                "/etc/apt/sources.list.d/amdgpu.list".to_owned(),
                "/etc/zypp/repos.d/amdgpu.repo".to_owned(),
            ]
        );
    }

    #[test]
    fn unrelated_paths_do_not_match_any_marker() {
        assert!(
            repo_native_method_from_markers(&["/etc/apt/sources.list.d/some-other.list"]).is_none()
        );
    }
}
