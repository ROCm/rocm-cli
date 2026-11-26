// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Offering the managed ROCm SDK to Lemonade as its ROCm runtime.
//!
//! Lemonade's `llamacpp:rocm` backend installer downloads a ROCm runtime of its
//! own (TheRock wheels or a tarball, several gigabytes) unless it finds a ROCm
//! root it accepts. It looks in `ROCM_PATH` first and takes a root whose HIP
//! runtime library it can see and whose version file, when there is one, names
//! the same `major.minor` ROCm version as the one its backend is pinned to
//! (`BackendUtils::resolve_rocm_root`, `validate_rocm_root` and
//! `has_matching_system_rocm_runtime` in Lemonade 2026.44.1's `src/cpp/server`).
//! rocm-cli already points `ROCM_PATH` at the managed SDK
//! ([`crate::process::lemonade_process_environment_vars`]) and sets the ROCm
//! version Lemonade's backend is pinned to to the active SDK's
//! ([`crate::backend_alignment`], for a release version), so what stands in the
//! way is the first test only:
//!
//! - On Windows the SDK's `bin` holds `amdhip64_7.dll`, which the probe accepts,
//!   so the SDK root is offered as it is.
//! - On Linux the probe wants an unversioned `libamdhip64.so` under `lib` or
//!   `lib64`, and the wheel layout of the SDK has only `libamdhip64.so.7`. A
//!   `rocm[devel]` install adds the unversioned name, but that pulls in the
//!   whole toolchain. So rocm-cli stages a small root of its own whose `lib`
//!   holds that one name as a link to the SDK's library, and whose `.info/version`
//!   copies the SDK's. Nothing else is read from it: the libraries llama.cpp
//!   loads come from the SDK through the loader paths rocm-cli sets.
//!
//! The version file is what keeps this honest. A root with no version file is
//! accepted whatever its version, so a root is only staged when the SDK's own
//! version can be read and copied; an SDK whose `major.minor` differs from the
//! backend's pin is still not reused, and Lemonade downloads its own runtime as
//! it did before.

use anyhow::Result;
#[cfg(unix)]
use anyhow::{Context, bail};
#[cfg(unix)]
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// The folder, under the data folder's Lemonade tree
/// (`<data>/engines/lemonade`, which `crate::install::lemonade_root` gives for no
/// environment), that holds the staged roots. An install inside a managed
/// runtime's own folder is elsewhere, so the staged root is not inside it.
#[cfg(unix)]
const STAGED_ROOTS_DIR: &str = "sdk-root";

/// The HIP runtime Lemonade's Linux probe looks for, by exactly this name.
#[cfg(unix)]
const LINUX_HIP_RUNTIME: &str = "libamdhip64.so";

/// The SDK Lemonade was offered as its ROCm runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfferedSdk {
    /// The SDK's ROCm version as its own version file records it. `None` for
    /// a root Lemonade accepts without one.
    pub(crate) version: Option<String>,
}

/// What `ROCM_PATH` should be for Lemonade, and whether Lemonade will take it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdkOffer {
    /// The directory to export as `ROCM_PATH`: the SDK's own root, or the root
    /// staged for it.
    pub(crate) rocm_root: PathBuf,
    /// `Some` when `rocm_root` is one Lemonade's probe accepts, so it can skip
    /// its own ROCm download for a matching version. `None` leaves Lemonade
    /// exactly where it was before: it downloads its own runtime.
    pub(crate) offered: Option<OfferedSdk>,
}

/// Decide what to export as `ROCM_PATH` for the managed SDK at `sdk_root`,
/// staging a root under `stage_parent` when the SDK's own layout is not one
/// Lemonade accepts.
///
/// `library_entries` are the SDK's library directories, searched for the HIP
/// runtime after the root's own `lib` and `lib64`. `windows` selects which
/// platform's probe to mirror; production passes
/// [`rocm_core::runtime_is_windows`].
///
/// Errors only when staging a root fails on disk; the caller falls back to
/// exporting the SDK root unchanged, which is what it did before this existed.
pub(crate) fn offer_sdk(
    stage_parent: &Path,
    sdk_root: &Path,
    library_entries: &[PathBuf],
    windows: bool,
) -> Result<SdkOffer> {
    let version = read_rocm_version(sdk_root);
    if lemonade_accepts_rocm_root(sdk_root, windows) {
        return Ok(SdkOffer {
            rocm_root: sdk_root.to_path_buf(),
            offered: Some(OfferedSdk { version }),
        });
    }
    let unavailable = || SdkOffer {
        rocm_root: sdk_root.to_path_buf(),
        offered: None,
    };
    // A staged root with no version file would be taken for any version of
    // ROCm, so without the SDK's own version there is nothing safe to stage.
    let Some(version) = version else {
        return Ok(unavailable());
    };
    #[cfg(unix)]
    {
        if windows {
            return Ok(unavailable());
        }
        let Some(hip_runtime) = find_hip_runtime(sdk_root, library_entries) else {
            return Ok(unavailable());
        };
        let staged = stage_parent
            .join(STAGED_ROOTS_DIR)
            .join(staged_root_name(sdk_root));
        stage_root(&staged, &hip_runtime, &version)?;
        Ok(SdkOffer {
            rocm_root: staged,
            offered: Some(OfferedSdk {
                version: Some(version),
            }),
        })
    }
    #[cfg(not(unix))]
    {
        // Only the Linux probe needs a root of this kind; Windows offers the
        // SDK root itself above, or leaves Lemonade to its own download.
        let _ = (stage_parent, library_entries, version);
        Ok(unavailable())
    }
}

/// Whether Lemonade's `validate_rocm_root` takes `root`: a Linux root needs an
/// unversioned `libamdhip64.so` in `lib` or `lib64`; a Windows root needs
/// `amdhip64.dll` or `amdhip64_<digits>.dll` as a regular file in `bin` or
/// `lib`. Mirrors Lemonade 2026.44.1, which does not accept other names, such as
/// `amdhip64_backup.dll`.
pub(crate) fn lemonade_accepts_rocm_root(root: &Path, windows: bool) -> bool {
    if windows {
        return ["bin", "lib"].iter().any(|subdir| {
            fs::read_dir(root.join(subdir)).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    entry.file_type().is_ok_and(|kind| kind.is_file())
                        && is_windows_hip_runtime(&entry.file_name().to_string_lossy())
                })
            })
        });
    }
    ["lib", "lib64"]
        .iter()
        .any(|subdir| root.join(subdir).join("libamdhip64.so").exists())
}

/// `amdhip64.dll`, or `amdhip64_` followed by one or more digits and `.dll`.
fn is_windows_hip_runtime(name: &str) -> bool {
    if name == "amdhip64.dll" {
        return true;
    }
    name.strip_prefix("amdhip64_")
        .and_then(|rest| rest.strip_suffix(".dll"))
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

/// The ROCm version a root records, read the way Lemonade reads it
/// (`BackendUtils::read_rocm_version_from_root`): the first line, trimmed, of the
/// first of `.info/version`, `share/rocm/version` and `version` that has one.
pub(crate) fn read_rocm_version(root: &Path) -> Option<String> {
    [
        root.join(".info").join("version"),
        root.join("share").join("rocm").join("version"),
        root.join("version"),
    ]
    .iter()
    .find_map(|path| {
        let text = fs::read_to_string(path).ok()?;
        let line = text.lines().next()?.trim();
        (!line.is_empty()).then(|| line.to_owned())
    })
}

/// The SDK's HIP runtime. The root's own `lib` and `lib64` are searched before the
/// SDK's other library directories, and the first directory with a match wins:
/// within it, `libamdhip64.so` if it is there, else the highest
/// `libamdhip64.so.<N>`.
#[cfg(unix)]
fn find_hip_runtime(sdk_root: &Path, library_entries: &[PathBuf]) -> Option<PathBuf> {
    let dirs = [sdk_root.join("lib"), sdk_root.join("lib64")]
        .into_iter()
        .chain(library_entries.iter().cloned());
    for dir in dirs {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        let mut best: Option<(Vec<u64>, PathBuf)> = None;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(rank) = hip_runtime_rank(&name) else {
                continue;
            };
            // `is_file` follows the link a versioned name usually is.
            if !entry.path().is_file() {
                continue;
            }
            if best.as_ref().is_none_or(|(best_rank, _)| rank > *best_rank) {
                best = Some((rank, entry.path()));
            }
        }
        if let Some((_, path)) = best {
            return Some(path);
        }
    }
    None
}

/// How much a file name is the HIP runtime: the exact name outranks every
/// versioned one, and a higher version outranks a lower one. `None` for any
/// other name.
#[cfg(unix)]
fn hip_runtime_rank(name: &str) -> Option<Vec<u64>> {
    if name == LINUX_HIP_RUNTIME {
        return Some(vec![u64::MAX]);
    }
    let version = name
        .strip_prefix(LINUX_HIP_RUNTIME)?
        .strip_prefix('.')?
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!version.is_empty()).then_some(version)
}

/// The folder name of the root staged for the SDK at `sdk_root`: a digest of
/// the path, so two SDKs never share a root and one SDK always finds its own.
#[cfg(unix)]
fn staged_root_name(sdk_root: &Path) -> String {
    let digest = Sha256::digest(sdk_root.to_string_lossy().as_bytes());
    digest[..8].iter().fold(String::new(), |mut name, byte| {
        let _ = write!(name, "{byte:02x}");
        name
    })
}

/// Make `staged` hold `lib/libamdhip64.so` linked to `hip_runtime` and
/// `.info/version` holding `version`, rewriting only what is not already so.
///
/// Every file is written under a temporary name and renamed into place, so a
/// `lemond` reading the root while another rocm-cli process stages the same one
/// sees the old content or the new, never a partial file.
#[cfg(unix)]
fn stage_root(staged: &Path, hip_runtime: &Path, version: &str) -> Result<()> {
    use std::os::unix::fs::symlink;

    let lib = staged.join("lib");
    let info = staged.join(".info");
    for dir in [&lib, &info] {
        fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    }
    // The version first: a root with the link and no version file is taken for
    // any ROCm version, so a failure between the two must leave the root
    // without the link, which the probe rejects.
    let version_file = info.join("version");
    let wanted = format!("{version}\n");
    if fs::read_to_string(&version_file).ok().as_deref() != Some(wanted.as_str()) {
        let temporary = info.join(format!(".version.{}", std::process::id()));
        fs::write(&temporary, &wanted)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        fs::rename(&temporary, &version_file)
            .with_context(|| format!("failed to replace {}", version_file.display()))?;
    }
    let link = lib.join(LINUX_HIP_RUNTIME);
    if fs::read_link(&link).ok().as_deref() != Some(hip_runtime) {
        let temporary = lib.join(format!(".{LINUX_HIP_RUNTIME}.{}", std::process::id()));
        let _ = fs::remove_file(&temporary);
        symlink(hip_runtime, &temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        fs::rename(&temporary, &link)
            .with_context(|| format!("failed to replace {}", link.display()))?;
    }
    if !link.exists() {
        bail!(
            "the staged ROCm root's {} does not resolve to a file",
            link.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A staging that fails writing the version leaves no link behind: a root
    /// with the link and no version file would be taken for any ROCm version.
    #[cfg(unix)]
    #[test]
    fn a_failed_version_write_leaves_no_link_to_be_taken_for_any_version() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("staged");
        let hip = dir.path().join("libamdhip64.so.7");
        fs::write(&hip, b"hip").unwrap();
        // The version file's place is a folder, so replacing it fails.
        fs::create_dir_all(staged.join(".info").join("version")).unwrap();

        assert!(stage_root(&staged, &hip, "10.1.0").is_err());
        assert!(
            fs::symlink_metadata(staged.join("lib").join("libamdhip64.so")).is_err(),
            "the link was written before the version"
        );
        assert!(!lemonade_accepts_rocm_root(&staged, false));
    }

    /// A Linux wheel-layout SDK core: only the versioned HIP runtime, and a
    /// version file.
    #[cfg(unix)]
    fn linux_wheel_sdk(dir: &Path, version: Option<&str>) -> PathBuf {
        let root = dir.join("_rocm_sdk_core");
        fs::create_dir_all(root.join("lib")).unwrap();
        fs::write(root.join("lib").join("libamdhip64.so.7"), b"hip").unwrap();
        if let Some(version) = version {
            fs::create_dir_all(root.join(".info")).unwrap();
            fs::write(root.join(".info").join("version"), format!("{version}\n")).unwrap();
        }
        root
    }

    fn windows_wheel_sdk(dir: &Path, hip_dll: &str) -> PathBuf {
        let root = dir.join("_rocm_sdk_core");
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::write(root.join("bin").join(hip_dll), b"hip").unwrap();
        fs::create_dir_all(root.join(".info")).unwrap();
        fs::write(root.join(".info").join("version"), "10.1.0\n").unwrap();
        root
    }

    /// Lemonade's Linux probe wants the unversioned name, in `lib` or `lib64`,
    /// and the wheel layout does not have it: the reason a root is staged.
    #[cfg(unix)]
    #[test]
    fn the_linux_probe_rejects_the_wheel_layout_and_accepts_the_unversioned_name() {
        let dir = tempfile::tempdir().unwrap();
        let wheel = linux_wheel_sdk(dir.path(), Some("10.1.0"));
        assert!(!lemonade_accepts_rocm_root(&wheel, false));

        for subdir in ["lib", "lib64"] {
            let root = dir.path().join(format!("root-{subdir}"));
            fs::create_dir_all(root.join(subdir)).unwrap();
            fs::write(root.join(subdir).join("libamdhip64.so"), b"hip").unwrap();
            assert!(lemonade_accepts_rocm_root(&root, false), "{subdir}");
        }
        assert!(!lemonade_accepts_rocm_root(
            &dir.path().join("absent"),
            false
        ));
    }

    /// The Windows probe accepts `amdhip64.dll` and `amdhip64_<digits>.dll` as a
    /// regular file in `bin` or `lib`, and nothing else; `windows` is a
    /// parameter so this runs on every host.
    #[test]
    fn the_windows_probe_accepts_only_the_hip_dll_names_lemonade_accepts() {
        let dir = tempfile::tempdir().unwrap();
        for (index, (name, accepted)) in [
            ("amdhip64.dll", true),
            ("amdhip64_7.dll", true),
            ("amdhip64_10.dll", true),
            ("amdhip64_backup.dll", false),
            ("amdhip64_.dll", false),
            ("amd_comgr_3.dll", false),
            ("amdhip64_7.dll.bak", false),
        ]
        .into_iter()
        .enumerate()
        {
            let root = windows_wheel_sdk(&dir.path().join(index.to_string()), name);
            assert_eq!(lemonade_accepts_rocm_root(&root, true), accepted, "{name}");
        }
        // `lib` is searched as well as `bin`.
        let root = dir.path().join("lib-only");
        fs::create_dir_all(root.join("lib")).unwrap();
        fs::write(root.join("lib").join("amdhip64_7.dll"), b"hip").unwrap();
        assert!(lemonade_accepts_rocm_root(&root, true));
        // A directory with the right name is not a regular file.
        let root = dir.path().join("dir-named-like-the-dll");
        fs::create_dir_all(root.join("bin").join("amdhip64_7.dll")).unwrap();
        assert!(!lemonade_accepts_rocm_root(&root, true));
    }

    /// Lemonade reads the first of three files, the first line of it, trimmed,
    /// and goes on to the next file when a file has no usable line.
    #[test]
    fn the_version_is_read_the_way_lemonade_reads_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        assert_eq!(read_rocm_version(root), None);

        fs::write(root.join("version"), "  7.1.0 \nsecond line\n").unwrap();
        assert_eq!(read_rocm_version(root).as_deref(), Some("7.1.0"));

        fs::create_dir_all(root.join("share").join("rocm")).unwrap();
        fs::write(root.join("share").join("rocm").join("version"), "7.2.0\n").unwrap();
        assert_eq!(read_rocm_version(root).as_deref(), Some("7.2.0"));

        fs::create_dir_all(root.join(".info")).unwrap();
        fs::write(root.join(".info").join("version"), "\n").unwrap();
        assert_eq!(
            read_rocm_version(root).as_deref(),
            Some("7.2.0"),
            "an empty first file does not hide the next"
        );
        fs::write(root.join(".info").join("version"), "10.1.0\n").unwrap();
        assert_eq!(read_rocm_version(root).as_deref(), Some("10.1.0"));
    }

    /// On Windows the SDK root is already one Lemonade accepts, so it is
    /// offered as it is and nothing is staged.
    #[test]
    fn a_windows_sdk_root_is_offered_as_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let root = windows_wheel_sdk(dir.path(), "amdhip64_7.dll");
        let stage_parent = dir.path().join("lemonade");
        let offer = offer_sdk(&stage_parent, &root, &[], true).unwrap();
        assert_eq!(
            offer,
            SdkOffer {
                rocm_root: root,
                offered: Some(OfferedSdk {
                    version: Some("10.1.0".to_owned())
                }),
            }
        );
        assert!(!stage_parent.exists(), "nothing is staged for Windows");
    }

    /// The Linux wheel layout is staged: the root handed to Lemonade is one its
    /// probe accepts, links to the SDK's HIP runtime, and carries the SDK's
    /// version, so the version check Lemonade runs is on the SDK's own.
    #[cfg(unix)]
    #[test]
    fn a_linux_wheel_layout_is_staged_into_a_root_lemonade_accepts() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = linux_wheel_sdk(dir.path(), Some("10.1.0"));
        let stage_parent = dir.path().join("lemonade");
        let offer = offer_sdk(&stage_parent, &sdk, &[], false).unwrap();

        assert_ne!(offer.rocm_root, sdk);
        assert!(offer.rocm_root.starts_with(stage_parent.join("sdk-root")));
        assert_eq!(
            offer.offered,
            Some(OfferedSdk {
                version: Some("10.1.0".to_owned())
            })
        );
        assert!(lemonade_accepts_rocm_root(&offer.rocm_root, false));
        assert_eq!(
            fs::read_link(offer.rocm_root.join("lib").join("libamdhip64.so")).unwrap(),
            sdk.join("lib").join("libamdhip64.so.7")
        );
        assert_eq!(
            read_rocm_version(&offer.rocm_root).as_deref(),
            Some("10.1.0")
        );
        // The SDK itself is not touched.
        assert!(!sdk.join("lib").join("libamdhip64.so").exists());
    }

    /// Staging again changes nothing, and an SDK upgraded in place (same path,
    /// new version and library) is restaged rather than left stale.
    #[cfg(unix)]
    #[test]
    fn staging_is_repeatable_and_follows_the_sdk() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = linux_wheel_sdk(dir.path(), Some("10.1.0"));
        let stage_parent = dir.path().join("lemonade");
        let first = offer_sdk(&stage_parent, &sdk, &[], false).unwrap();
        let again = offer_sdk(&stage_parent, &sdk, &[], false).unwrap();
        assert_eq!(first, again);

        fs::write(sdk.join(".info").join("version"), "10.2.0\n").unwrap();
        fs::write(sdk.join("lib").join("libamdhip64.so.8"), b"hip").unwrap();
        let upgraded = offer_sdk(&stage_parent, &sdk, &[], false).unwrap();
        assert_eq!(upgraded.rocm_root, first.rocm_root, "same SDK, same root");
        assert_eq!(
            read_rocm_version(&upgraded.rocm_root).as_deref(),
            Some("10.2.0")
        );
        assert_eq!(
            fs::read_link(upgraded.rocm_root.join("lib").join("libamdhip64.so")).unwrap(),
            sdk.join("lib").join("libamdhip64.so.8")
        );
        // No temporary files are left behind.
        for dir in [".info", "lib"] {
            let names: Vec<_> = fs::read_dir(upgraded.rocm_root.join(dir))
                .unwrap()
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect();
            assert!(
                names.iter().all(|name| !name.starts_with('.')),
                "{dir}: {names:?}"
            );
        }
    }

    /// Two SDKs get two roots, so choosing a different SDK never repoints the
    /// root a running server was given.
    #[cfg(unix)]
    #[test]
    fn each_sdk_gets_its_own_staged_root() {
        let dir = tempfile::tempdir().unwrap();
        let one = linux_wheel_sdk(&dir.path().join("one"), Some("10.1.0"));
        let two = linux_wheel_sdk(&dir.path().join("two"), Some("10.2.0"));
        let stage_parent = dir.path().join("lemonade");
        let first = offer_sdk(&stage_parent, &one, &[], false).unwrap();
        let second = offer_sdk(&stage_parent, &two, &[], false).unwrap();
        assert_ne!(first.rocm_root, second.rocm_root);
        assert_eq!(
            read_rocm_version(&first.rocm_root).as_deref(),
            Some("10.1.0")
        );
    }

    /// An SDK whose version cannot be read is not offered: a root with no
    /// version file is taken for any ROCm version, so staging one would send a
    /// backend built for one version to run on another.
    #[cfg(unix)]
    #[test]
    fn an_sdk_with_no_version_is_not_staged() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = linux_wheel_sdk(dir.path(), None);
        let stage_parent = dir.path().join("lemonade");
        let offer = offer_sdk(&stage_parent, &sdk, &[], false).unwrap();
        assert_eq!(
            offer,
            SdkOffer {
                rocm_root: sdk,
                offered: None
            }
        );
        assert!(!stage_parent.exists());
    }

    /// With no HIP runtime to link to there is nothing to stage, and the SDK
    /// root is passed on unchanged: Lemonade downloads its own runtime, as it
    /// did before.
    #[cfg(unix)]
    #[test]
    fn an_sdk_with_no_hip_runtime_is_not_staged() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = dir.path().join("_rocm_sdk_core");
        fs::create_dir_all(sdk.join(".info")).unwrap();
        fs::write(sdk.join(".info").join("version"), "10.1.0\n").unwrap();
        let offer = offer_sdk(&dir.path().join("lemonade"), &sdk, &[], false).unwrap();
        assert_eq!(offer.rocm_root, sdk);
        assert_eq!(offer.offered, None);
    }

    /// The HIP runtime may live in one of the SDK's other library directories
    /// (the SDK splits its libraries over several packages); the exact name
    /// wins over a versioned one, and a higher version over a lower.
    #[cfg(unix)]
    #[test]
    fn the_hip_runtime_is_found_by_name_then_version() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = dir.path().join("sdk");
        fs::create_dir_all(sdk.join("lib")).unwrap();
        let other = dir.path().join("other-lib");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("libamdhip64.so.9"), b"hip").unwrap();
        assert_eq!(
            find_hip_runtime(&sdk, std::slice::from_ref(&other)),
            Some(other.join("libamdhip64.so.9"))
        );

        for name in ["libamdhip64.so.7", "libamdhip64.so.10", "libamdhip64.so.x"] {
            fs::write(sdk.join("lib").join(name), b"hip").unwrap();
        }
        assert_eq!(
            find_hip_runtime(&sdk, std::slice::from_ref(&other)),
            Some(sdk.join("lib").join("libamdhip64.so.10")),
            "the root's own lib comes first, highest version, non-numeric name ignored"
        );
        fs::write(sdk.join("lib").join("libamdhip64.so"), b"hip").unwrap();
        assert_eq!(
            find_hip_runtime(&sdk, &[]),
            Some(sdk.join("lib").join("libamdhip64.so"))
        );
        // Another library whose name merely starts the same way is not it.
        assert_eq!(
            hip_runtime_rank("libamdhip64.so.7.1.2"),
            Some(vec![7, 1, 2])
        );
        assert_eq!(hip_runtime_rank("libamdhip64.sox"), None);
        assert_eq!(hip_runtime_rank("libamdhip64.so."), None);
        assert_eq!(hip_runtime_rank("libamdhip640.so.7"), None);
    }

    /// A dangling link is not a runtime: the probe and the loader both fail on
    /// it, so it is not staged.
    #[cfg(unix)]
    #[test]
    fn a_dangling_hip_runtime_link_is_not_staged() {
        let dir = tempfile::tempdir().unwrap();
        let sdk = dir.path().join("_rocm_sdk_core");
        fs::create_dir_all(sdk.join("lib")).unwrap();
        fs::create_dir_all(sdk.join(".info")).unwrap();
        fs::write(sdk.join(".info").join("version"), "10.1.0\n").unwrap();
        std::os::unix::fs::symlink(
            "libamdhip64.so.7.0",
            sdk.join("lib").join("libamdhip64.so.7"),
        )
        .unwrap();
        let offer = offer_sdk(&dir.path().join("lemonade"), &sdk, &[], false).unwrap();
        assert_eq!(offer.offered, None);
    }
}
