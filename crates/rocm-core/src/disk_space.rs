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
//!   user-facing message instead of the raw OS error, and
//!   [`subprocess_full_disk_error`] does the same for a helper process such as
//!   `tar`, whose out-of-space failure arrives as stderr text.
//!
//! Free space is only reported when the filesystem holding the path can be
//! identified with confidence: the platform's mount list is incomplete, so a
//! device-ID cross-check rejects a mount that merely looks like an ancestor.
//! An unidentifiable filesystem reports [`SpaceCheck::Unknown`], which never
//! blocks an operation — reporting another filesystem's free space would
//! refuse a valid install with a confident wrong number.
//!
//! Hard failure is reserved for *exact* requirements (a known download size).
//! Estimated requirements (extraction, which depends on the compression ratio)
//! only warn: a false refusal that blocks a valid install is worse than a late
//! failure.

use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// Smallest slack added on top of a requirement, covering filesystem metadata,
/// rounding, and the last few writes of an install.
///
/// The margin is proportional to the payload (see [`with_margin`]) with this as
/// a floor, so a small download — the `uv` binary is tens of megabytes — is not
/// refused on a machine that has ample room for it. A flat multi-hundred-megabyte
/// margin would turn a 20 MiB download into a 276 MiB requirement.
pub const SPACE_MARGIN_MIN_BYTES: u64 = 32 * 1024 * 1024;

/// Proportional part of the margin: one twentieth (5%) of the payload.
///
/// At SDK-tarball scale (~5 GiB) this lands near 256 MiB; at `uv` scale it
/// stays under the floor and [`SPACE_MARGIN_MIN_BYTES`] applies instead.
pub const SPACE_MARGIN_DIVISOR: u64 = 20;

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

/// Requirement including a safety margin proportional to the payload.
///
/// The margin is `max(bytes / SPACE_MARGIN_DIVISOR, SPACE_MARGIN_MIN_BYTES)`, so
/// it scales with what is actually being written instead of imposing a large
/// fixed cost on small downloads.
pub const fn with_margin(bytes: u64) -> u64 {
    let proportional = bytes / SPACE_MARGIN_DIVISOR;
    let margin = if proportional > SPACE_MARGIN_MIN_BYTES {
        proportional
    } else {
        SPACE_MARGIN_MIN_BYTES
    };
    bytes.saturating_add(margin)
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

/// Strip a Windows verbatim path prefix, leaving an ordinary path.
///
/// `Path::canonicalize` returns verbatim paths on Windows (`\\?\C:\Users\...`),
/// whose first component parses as `Prefix::VerbatimDisk`. Mount points reported
/// by the platform use `Prefix::Disk` (`C:\`), and `Path::starts_with` compares
/// prefixes by variant, so the two never match and every Windows path would
/// otherwise resolve to no mount at all. UNC verbatim paths (`\\?\UNC\server\share`)
/// map back to `\\server\share`.
///
/// Pure string handling so it is exercised on every platform, not only Windows.
pub(crate) fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    path.to_path_buf()
}

/// `Path::starts_with`, but tolerant of the platform's path-comparison rules.
///
/// Windows filesystems are case-insensitive, so a volume mounted at `C:\Data`
/// must still match a path spelled `C:\data\...`. Lowercasing preserves
/// component boundaries, so this stays component-wise — the `/data` vs
/// `/database` trap does not reappear.
fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    #[cfg(windows)]
    {
        let path = PathBuf::from(path.to_string_lossy().to_lowercase());
        let prefix = PathBuf::from(prefix.to_string_lossy().to_lowercase());
        path.starts_with(prefix)
    }
    #[cfg(not(windows))]
    {
        path.starts_with(prefix)
    }
}

/// Pick the mount that owns `path`: the longest mount point that is a prefix of
/// it. Split out from [`mount_for_path`] so it can be tested against synthetic
/// mount tables; `mounts` is `(mount_point, available_bytes)`.
///
/// Prefix matching alone is not proof of ownership — see
/// [`mount_owns_path`], which [`mount_for_path`] applies on top of this.
fn select_mount(path: &Path, mounts: &[(PathBuf, u64)]) -> Option<(PathBuf, u64)> {
    let path = strip_verbatim_prefix(path);
    mounts
        .iter()
        .filter(|(mount_point, _)| path_starts_with(&path, &strip_verbatim_prefix(mount_point)))
        .max_by_key(|(mount_point, _)| mount_point.components().count())
        .cloned()
}

/// Whether `mount_point` really is the filesystem holding `path`.
///
/// Longest-prefix selection is only correct if every mount is listed. It is not:
/// `sysinfo` omits tmpfs by default (`linux-tmpfs`) and skips NFS/CIFS unless
/// `linux-netdevs` is enabled, because `statvfs` on a hard-mounted network share
/// can hang. When the real mount is missing, the prefix filter does not fail —
/// it falls through to the nearest listed ancestor, in practice `/`, and reports
/// a completely unrelated filesystem's free space. That turns a valid install
/// into a hard refusal quoting a confident wrong number.
///
/// Comparing device IDs catches exactly that case: on a mismatch the caller
/// reports the space as unknown, which the design already treats as "never
/// block". Unix-only; other platforms have no cheap equivalent and keep the
/// prefix result.
#[cfg(unix)]
fn mount_owns_path(path: &Path, mount_point: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(path_meta) = std::fs::metadata(path) else {
        return false;
    };
    let Ok(mount_meta) = std::fs::metadata(mount_point) else {
        return false;
    };
    path_meta.dev() == mount_meta.dev()
}

#[cfg(not(unix))]
fn mount_owns_path(_path: &Path, _mount_point: &Path) -> bool {
    true
}

/// Mount point and free bytes for the filesystem that will hold `path`.
///
/// `path` need not exist: the lookup walks up to the nearest existing ancestor,
/// so a destination inside a directory that is about to be created still
/// resolves to the right filesystem. Returns `None` when the platform reports
/// no matching mount point (in which case callers must not block the operation).
pub fn mount_for_path(path: &Path) -> Option<(PathBuf, u64)> {
    let resolved = nearest_existing_ancestor(path)?;
    let (mount_point, available) = select_mount(&resolved, &listed_mounts())?;
    // Prefix matching cannot tell "this mount owns the path" from "the real
    // mount is missing from the list"; the device check can.
    mount_owns_path(&resolved, &mount_point).then_some((mount_point, available))
}

/// Mount points and free bytes as the platform reports them.
///
/// Refreshes storage figures only: the default sweep also reads
/// `/proc/diskstats` and `/sys/block/*/queue/rotational`, neither of which this
/// module uses.
fn listed_mounts() -> Vec<(PathBuf, u64)> {
    let disks = sysinfo::Disks::new_with_refreshed_list_specifics(
        sysinfo::DiskRefreshKind::nothing().with_storage(),
    );
    disks
        .list()
        .iter()
        .map(|disk| (disk.mount_point().to_path_buf(), disk.available_space()))
        .collect()
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

/// Compare `required` bytes against an already-resolved free-space figure.
///
/// `available` is `None` when the filesystem could not be determined. Separated
/// from the lookup so the policy — including the paths that decide to refuse an
/// install — is testable without depending on the host's real filesystems.
const fn classify_space(required: u64, available: Option<u64>) -> SpaceCheck {
    match available {
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

/// Compare `required` bytes against the free space on `path`'s filesystem.
pub fn check_space_for_path(path: &Path, required: u64) -> SpaceCheck {
    classify_space(required, available_space_for_path(path))
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
    ensure_space(operation, path, required, available_space_for_path(path))
}

/// [`ensure_space_for`] against a caller-supplied free-space figure.
fn ensure_space(operation: &str, path: &Path, required: u64, available: Option<u64>) -> Result<()> {
    if let SpaceCheck::Insufficient {
        required,
        available,
    } = classify_space(required, available)
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
    low_space_warning(operation, path, estimated, available_space_for_path(path))
}

/// [`warn_if_low_space`] against a caller-supplied free-space figure.
fn low_space_warning(
    operation: &str,
    path: &Path,
    estimated: u64,
    available: Option<u64>,
) -> Option<String> {
    match classify_space(estimated, available) {
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

/// Whether a subprocess's diagnostic output reports a full disk.
///
/// Extraction shells out to `tar`, so the failure arrives as a non-zero exit
/// status and a line of stderr rather than an [`std::io::Error`] that
/// [`map_write_error`] could classify. Matching the message is the only signal
/// available. `tar` and the GNU C library it goes through emit the `ENOSPC`
/// text localized, so this also matches the errno name that appears in
/// non-English locales' `tar` diagnostics.
pub fn output_reports_full_disk(text: &str) -> bool {
    let text = text.to_lowercase();
    text.contains("no space left on device")
        || text.contains("enospc")
        || text.contains("disk quota exceeded")
}

/// Rewrite a subprocess failure as a full-disk message when that is the cause.
///
/// `text` is the command's diagnostic output; `path` is the location being
/// written. Returns `None` when the failure is something else, leaving the
/// caller's own error reporting in place.
pub fn subprocess_full_disk_error(text: &str, path: &Path) -> Option<anyhow::Error> {
    if !output_reports_full_disk(text) {
        return None;
    }
    let available = available_space_for_path(path)
        .map(|bytes| format!(" ({} free)", format_bytes(bytes)))
        .unwrap_or_default();
    Some(anyhow!(
        "ran out of disk space while writing to {}{available}. Free up space on that filesystem and retry.",
        path.display(),
    ))
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
        assert_eq!(with_margin(0), SPACE_MARGIN_MIN_BYTES);
        assert_eq!(with_margin(u64::MAX), u64::MAX);
    }

    #[test]
    fn margin_stays_proportional_so_small_downloads_are_not_refused() {
        // A ~20 MiB helper download must not inherit an SDK-sized margin: the
        // requirement has to stay well inside a 200 MiB filesystem.
        let uv = 20 * 1024 * 1024;
        assert!(
            with_margin(uv) < 200 * 1024 * 1024,
            "{}",
            format_bytes(with_margin(uv))
        );
        // At SDK scale the proportional part takes over from the floor.
        let sdk = 5 * 1024 * 1024 * 1024;
        assert_eq!(with_margin(sdk), sdk + sdk / SPACE_MARGIN_DIVISOR);
        assert!(with_margin(sdk) - sdk > SPACE_MARGIN_MIN_BYTES);
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
    fn unknown_space_never_blocks_a_nonzero_requirement() {
        // The whole fail-open design rests on this: when the filesystem cannot
        // be identified, a large requirement must still pass. A zero-byte
        // requirement would satisfy `available >= required` trivially and prove
        // nothing, so this uses the largest requirement there is.
        ensure_space(
            "download the SDK tarball",
            Path::new("/cache/rocm.tar.gz"),
            u64::MAX,
            None,
        )
        .expect("unknown free space must never block");
        assert_eq!(
            low_space_warning("extract the SDK", Path::new("/install"), u64::MAX, None),
            None
        );
    }

    #[test]
    fn insufficient_space_refuses_and_sufficient_space_allows() {
        // The refusal path itself, driven by a synthetic figure rather than
        // whatever the host filesystem happens to have free.
        let error = ensure_space("download it", Path::new("/cache/x"), 100, Some(10))
            .expect_err("a real shortfall must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("not enough free disk space"), "{text}");
        assert!(text.contains("Free up 90 B"), "{text}");
        ensure_space("download it", Path::new("/cache/x"), 100, Some(100))
            .expect("exactly enough space must pass");
    }

    #[test]
    fn low_space_warning_fires_only_on_a_real_shortfall() {
        let warning = low_space_warning("extract the SDK", Path::new("/install"), 100, Some(10))
            .expect("a shortfall must warn");
        assert!(warning.starts_with("Warning:"), "{warning}");
        assert!(warning.contains("may fail partway through"), "{warning}");
        assert_eq!(
            low_space_warning("extract the SDK", Path::new("/install"), 100, Some(1_000)),
            None
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_mount_that_does_not_own_the_path_is_not_used_for_its_free_space() {
        // Regression guard for the dangerous case: sysinfo omits tmpfs, NFS and
        // CIFS mounts, so a path on one of them prefix-matches `/` and would
        // otherwise be reported with the root filesystem's free space. On this
        // host /dev/shm is tmpfs and is absent from the listed mounts.
        // `/proc` is a distinct filesystem on every Linux system and is never
        // reported by sysinfo, so it stands in for the tmpfs/NFS/CIFS mounts
        // that are invisible for the same reason — without depending on which
        // optional sysinfo features happen to be enabled.
        let unlisted = Path::new("/proc");
        let listed = listed_mounts();
        assert!(
            !listed.iter().any(|(mount, _)| mount == unlisted),
            "precondition: /proc must be absent from the reported mounts"
        );
        assert!(
            select_mount(unlisted, &listed).is_some(),
            "prefix matching alone still resolves a mount — that is the trap"
        );
        assert!(
            !mount_owns_path(unlisted, Path::new("/")),
            "/proc is not on the root filesystem"
        );
        assert_eq!(
            mount_for_path(&unlisted.join("rocm-sdk.tar.gz")),
            None,
            "a path on an unlisted filesystem must report unknown, not another mount's space"
        );
        // Failing open is the point: unknown space must not refuse the install.
        ensure_space_for(
            "download the SDK tarball",
            &unlisted.join("rocm-sdk.tar.gz"),
            u64::MAX,
        )
        .expect("an unresolvable filesystem must never block");
    }

    #[test]
    fn verbatim_windows_prefixes_are_stripped_before_matching() {
        // `canonicalize` yields `\\?\C:\...` on Windows, whose Prefix variant
        // never equals a mount point's `C:\`. Exercised as pure string handling
        // so the comparison is covered on every platform.
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\C:\Users\rocm")),
            PathBuf::from(r"C:\Users\rocm")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"\\?\UNC\server\share\rocm")),
            PathBuf::from(r"\\server\share\rocm")
        );
        // A path with no verbatim prefix is untouched, including on Unix.
        assert_eq!(
            strip_verbatim_prefix(Path::new("/home/user")),
            PathBuf::from("/home/user")
        );
        assert_eq!(
            strip_verbatim_prefix(Path::new(r"C:\Users")),
            PathBuf::from(r"C:\Users")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_mount_selection_matches_a_canonicalized_path() {
        // The end-to-end shape of #2: a canonicalized (verbatim) path against a
        // mount table spelled the way the platform reports it, plus the
        // case-insensitivity Windows volumes require.
        let mounts = vec![(PathBuf::from(r"C:\"), 42), (PathBuf::from(r"D:\Data"), 7)];
        assert_eq!(
            select_mount(Path::new(r"\\?\C:\Users\rocm\cache"), &mounts),
            Some((PathBuf::from(r"C:\"), 42))
        );
        assert_eq!(
            select_mount(Path::new(r"\\?\D:\data\rocm"), &mounts),
            Some((PathBuf::from(r"D:\Data"), 7))
        );
    }

    #[test]
    fn subprocess_enospc_output_is_recognized() {
        // `tar` failures arrive as stderr text, not an io::Error.
        assert!(output_reports_full_disk(
            "tar: /install/lib/libfoo.so: Cannot write: No space left on device"
        ));
        assert!(output_reports_full_disk("write error: ENOSPC"));
        assert!(output_reports_full_disk("tar: Disk quota exceeded"));
        assert!(!output_reports_full_disk(
            "tar: /cache/x.tar.gz: Cannot open: Permission denied"
        ));
    }

    #[test]
    fn subprocess_full_disk_error_replaces_only_enospc_failures() {
        let mapped = subprocess_full_disk_error(
            "tar: Cannot write: No space left on device",
            Path::new("/install/rocm"),
        )
        .expect("an ENOSPC subprocess failure must be recognized");
        let text = format!("{mapped:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
        assert!(text.contains("/install/rocm"), "{text}");
        assert!(!text.contains("No space left on device"), "{text}");
        assert!(
            subprocess_full_disk_error("tar: Permission denied", Path::new("/install/rocm"))
                .is_none()
        );
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
        // silent (space unknown) — it must never be treated as fatal. The
        // per-branch assertions live in `low_space_warning_fires_only_on_a_real_shortfall`,
        // which drives both outcomes deterministically; this one guards the
        // signature: it returns rather than erroring.
        let temp = std::env::temp_dir();
        let warning = warn_if_low_space("extract the SDK", &temp, u64::MAX);
        assert_eq!(
            warning.is_some(),
            available_space_for_path(&temp).is_some(),
            "a resolvable filesystem must warn on a u64::MAX estimate, and an \
             unresolvable one must stay silent"
        );
    }
}
