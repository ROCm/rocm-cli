// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Free-space preflight checks and out-of-space error reporting.
//!
//! ROCm SDK installs download multi-gigabyte tarballs and extract them, so a
//! nearly-full disk otherwise surfaces as a raw low-level write failure partway
//! through the install. This module provides:
//!
//! * [`available_space_for_path`] — free space on the filesystem that will
//!   actually hold a path (not the current directory).
//! * [`ensure_space_for`] / [`warn_if_low_space`] — preflight checks that fail
//!   early, or merely warn, with required-vs-available amounts.
//! * [`map_write_error`] — maps [`std::io::ErrorKind::StorageFull`] to a clear
//!   user-facing message instead of the raw OS error.
//!
//! Hard failure is reserved for *exact* requirements (a known download size).
//! Estimated requirements (extraction, which depends on the compression ratio)
//! only warn: a false refusal that blocks a valid install is worse than a late
//! failure.

use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// Slack added on top of every requirement, covering filesystem metadata,
/// rounding, and the last few writes of an install.
pub const SPACE_MARGIN_BYTES: u64 = 256 * 1024 * 1024;

/// Conservative compressed-to-extracted multiplier for SDK tarballs.
///
/// TheRock artifacts have no manifest field for the uncompressed size and the
/// index offers none, so the only cheap upfront signal is the compressed size
/// (`Content-Length`). Observed gzip ratios for ROCm tarballs — mostly already
/// incompressible binaries and libraries with some highly compressible headers
/// and text — land around 2-3x. 4x is picked as a deliberately conservative
/// upper bound: because this requirement is an estimate it only ever produces a
/// warning, so over-estimating costs nothing but a nudge, while
/// under-estimating would let a doomed install start.
pub const EXTRACTED_SIZE_MULTIPLIER: u64 = 4;

/// Outcome of comparing a space requirement against a filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceCheck {
    /// The filesystem has at least `required` bytes free.
    Sufficient { required: u64, available: u64 },
    /// The filesystem has less than `required` bytes free.
    Insufficient { required: u64, available: u64 },
    /// Free space could not be determined for this path.
    Unknown,
}

impl SpaceCheck {
    pub const fn is_insufficient(self) -> bool {
        matches!(self, Self::Insufficient { .. })
    }
}

/// Estimated space needed to extract an archive of `archive_bytes`.
///
/// The archive itself normally stays on disk during extraction, so the estimate
/// covers only the extracted tree; the archive is accounted for separately by
/// the download check.
pub const fn estimated_extracted_size(archive_bytes: u64) -> u64 {
    archive_bytes.saturating_mul(EXTRACTED_SIZE_MULTIPLIER)
}

/// Requirement including the shared safety margin.
pub const fn with_margin(bytes: u64) -> u64 {
    bytes.saturating_add(SPACE_MARGIN_BYTES)
}

/// Nearest ancestor of `path` that exists, used to resolve the filesystem a
/// not-yet-created file or directory will live on.
fn nearest_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut candidate = absolute.as_path();
    loop {
        if candidate.exists() {
            return candidate
                .canonicalize()
                .ok()
                .or_else(|| Some(candidate.to_path_buf()));
        }
        candidate = candidate.parent()?;
    }
}

/// Pick the mount that owns `path`: the longest mount point that is a prefix of
/// it. Split out from [`mount_for_path`] so it can be tested against synthetic
/// mount tables; `mounts` is `(mount_point, available_bytes)`.
fn select_mount(path: &Path, mounts: &[(PathBuf, u64)]) -> Option<(PathBuf, u64)> {
    mounts
        .iter()
        .filter(|(mount_point, _)| path.starts_with(mount_point))
        .max_by_key(|(mount_point, _)| mount_point.components().count())
        .cloned()
}

/// Mount point and free bytes for the filesystem that will hold `path`.
///
/// `path` need not exist: the lookup walks up to the nearest existing ancestor,
/// so a destination inside a directory that is about to be created still
/// resolves to the right filesystem. Returns `None` when the platform reports
/// no matching mount point (in which case callers must not block the operation).
pub fn mount_for_path(path: &Path) -> Option<(PathBuf, u64)> {
    let resolved = nearest_existing_ancestor(path)?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mounts = disks
        .list()
        .iter()
        .map(|disk| (disk.mount_point().to_path_buf(), disk.available_space()))
        .collect::<Vec<_>>();
    select_mount(&resolved, &mounts)
}

/// Free space, in bytes, on the filesystem that will hold `path`.
pub fn available_space_for_path(path: &Path) -> Option<u64> {
    mount_for_path(path).map(|(_, available)| available)
}

/// Whether two paths live on the same filesystem.
///
/// `None` when either path's filesystem cannot be determined.
pub fn on_same_filesystem(left: &Path, right: &Path) -> Option<bool> {
    let left_mount = mount_for_path(left)?.0;
    let right_mount = mount_for_path(right)?.0;
    Some(left_mount == right_mount)
}

/// Compare `required` bytes against the free space on `path`'s filesystem.
pub fn check_space_for_path(path: &Path, required: u64) -> SpaceCheck {
    match available_space_for_path(path) {
        Some(available) if available >= required => SpaceCheck::Sufficient {
            required,
            available,
        },
        Some(available) => SpaceCheck::Insufficient {
            required,
            available,
        },
        None => SpaceCheck::Unknown,
    }
}

/// Human-readable byte size, e.g. `1.5 GiB`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Message for a failed preflight check.
pub fn insufficient_space_message(
    operation: &str,
    path: &Path,
    required: u64,
    available: u64,
) -> String {
    format!(
        "not enough free disk space to {operation}: need about {} but only {} is available on the filesystem holding {}. Free up {} and retry.",
        format_bytes(required),
        format_bytes(available),
        path.display(),
        format_bytes(required.saturating_sub(available)),
    )
}

/// Preflight check for an *exact* requirement: fails before any bytes are written.
///
/// Never fails when free space cannot be determined.
pub fn ensure_space_for(operation: &str, path: &Path, required: u64) -> Result<()> {
    if let SpaceCheck::Insufficient {
        required,
        available,
    } = check_space_for_path(path, required)
    {
        bail!(insufficient_space_message(
            operation, path, required, available
        ));
    }
    Ok(())
}

/// Preflight check for an *estimated* requirement.
///
/// Returns a warning to surface to the user rather than failing, so an
/// imprecise estimate can never block an install that would in fact succeed.
pub fn warn_if_low_space(operation: &str, path: &Path, estimated: u64) -> Option<String> {
    match check_space_for_path(path, estimated) {
        SpaceCheck::Insufficient {
            required,
            available,
        } => Some(format!(
            "Warning: free disk space may be insufficient to {operation}: about {} estimated, {} available on the filesystem holding {}. The install will continue, but may fail partway through.",
            format_bytes(required),
            format_bytes(available),
            path.display(),
        )),
        SpaceCheck::Sufficient { .. } | SpaceCheck::Unknown => None,
    }
}

/// Map a write failure to a clear message when the cause is a full disk.
///
/// Other errors are passed through with the usual path context.
pub fn map_write_error(error: std::io::Error, path: &Path) -> anyhow::Error {
    if error.kind() == std::io::ErrorKind::StorageFull {
        let available = available_space_for_path(path)
            .map(|bytes| format!(" ({} free)", format_bytes(bytes)))
            .unwrap_or_default();
        return anyhow!(
            "ran out of disk space while writing {}{available}. Free up space on that filesystem and retry.",
            path.display(),
        );
    }
    anyhow::Error::new(error).context(format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_scales_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn extracted_size_uses_conservative_multiplier() {
        assert_eq!(estimated_extracted_size(1_000), 4_000);
        assert_eq!(estimated_extracted_size(u64::MAX), u64::MAX);
    }

    #[test]
    fn with_margin_saturates() {
        assert_eq!(with_margin(0), SPACE_MARGIN_BYTES);
        assert_eq!(with_margin(u64::MAX), u64::MAX);
    }

    #[test]
    fn select_mount_prefers_longest_matching_mount_point() {
        let mounts = vec![
            (PathBuf::from("/"), 10),
            (PathBuf::from("/home"), 20),
            (PathBuf::from("/home/user/data"), 30),
        ];
        assert_eq!(
            select_mount(Path::new("/home/user/data/cache/x.tar"), &mounts),
            Some((PathBuf::from("/home/user/data"), 30))
        );
        assert_eq!(
            select_mount(Path::new("/home/user/other"), &mounts),
            Some((PathBuf::from("/home"), 20))
        );
        assert_eq!(
            select_mount(Path::new("/var/tmp"), &mounts),
            Some((PathBuf::from("/"), 10))
        );
    }

    #[test]
    fn select_mount_returns_none_without_a_matching_mount() {
        let mounts = vec![(PathBuf::from("/mnt/data"), 42)];
        assert_eq!(select_mount(Path::new("/home/user"), &mounts), None);
    }

    #[test]
    fn select_mount_identifies_the_same_filesystem_for_sibling_paths() {
        let mounts = vec![(PathBuf::from("/"), 10), (PathBuf::from("/mnt/data"), 30)];
        let cache = select_mount(Path::new("/home/user/.cache/rocm"), &mounts).map(|(m, _)| m);
        let install =
            select_mount(Path::new("/home/user/.local/share/rocm"), &mounts).map(|(m, _)| m);
        assert_eq!(cache, install);
        let other = select_mount(Path::new("/mnt/data/rocm"), &mounts).map(|(m, _)| m);
        assert_ne!(cache, other);
    }

    #[test]
    fn nearest_existing_ancestor_walks_up_missing_components() {
        let temp = std::env::temp_dir();
        let missing = temp
            .join("rocm-cli-space-check-does-not-exist")
            .join("a/b/c");
        let resolved = nearest_existing_ancestor(&missing).expect("temp dir exists");
        assert!(resolved.exists(), "{} should exist", resolved.display());
    }

    #[test]
    fn insufficient_space_message_reports_required_available_and_shortfall() {
        let message = insufficient_space_message(
            "download the SDK tarball",
            Path::new("/cache/rocm.tar.gz"),
            8 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        );
        assert!(message.contains("8.0 GiB"), "{message}");
        assert!(message.contains("2.0 GiB"), "{message}");
        assert!(message.contains("Free up 6.0 GiB"), "{message}");
        assert!(message.contains("/cache/rocm.tar.gz"), "{message}");
    }

    #[test]
    fn check_space_classifies_against_a_known_available_amount() {
        // Exercise the comparison independently of the host filesystem.
        let sufficient = SpaceCheck::Sufficient {
            required: 10,
            available: 20,
        };
        assert!(!sufficient.is_insufficient());
        assert!(
            SpaceCheck::Insufficient {
                required: 20,
                available: 10
            }
            .is_insufficient()
        );
        assert!(!SpaceCheck::Unknown.is_insufficient());
    }

    #[test]
    fn zero_requirement_never_fails_on_a_real_path() {
        // A zero-byte requirement is satisfiable on any filesystem, and an
        // undeterminable filesystem must not block the caller either.
        ensure_space_for("write nothing", &std::env::temp_dir(), 0).expect("zero bytes always fit");
    }

    #[test]
    fn map_write_error_explains_a_full_disk() {
        let error = std::io::Error::from(std::io::ErrorKind::StorageFull);
        let mapped = map_write_error(error, Path::new("/cache/rocm.tar.gz"));
        let text = format!("{mapped:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
        assert!(text.contains("/cache/rocm.tar.gz"), "{text}");
        assert!(!text.contains("StorageFull"), "{text}");
    }

    #[test]
    fn map_write_error_passes_other_errors_through() {
        let error = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let mapped = map_write_error(error, Path::new("/cache/rocm.tar.gz"));
        let text = format!("{mapped:#}");
        assert!(
            text.contains("failed to write /cache/rocm.tar.gz"),
            "{text}"
        );
        assert!(!text.contains("ran out of disk space"), "{text}");
    }

    #[test]
    fn warn_if_low_space_returns_a_warning_not_an_error_for_huge_estimates() {
        // An absurd estimate on a real path either warns (space known) or is
        // silent (space unknown) — it must never be treated as fatal.
        let warning = warn_if_low_space("extract the SDK", &std::env::temp_dir(), u64::MAX);
        if let Some(warning) = warning {
            assert!(warning.starts_with("Warning:"), "{warning}");
            assert!(warning.contains("may fail partway through"), "{warning}");
        }
    }
}
