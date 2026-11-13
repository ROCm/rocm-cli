// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The record that an attempt to align Lemonade's ROCm backend failed.
//!
//! An alignment that fails costs two backend downloads and a forced reinstall of
//! the backend Lemonade ships. Its revert leaves the pin different from the active
//! ROCm SDK, so without a record every later install would repeat all of it. The
//! record lives inside the Lemonade runtime tree it describes, so a
//! `--reinstall` (or a new Lemonade version), which deletes and re-extracts that
//! tree, removes it as well.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// File name of the record, inside the Lemonade runtime directory.
pub(crate) const ALIGNMENT_FAILURE_RECORD: &str = ".rocm-cli-alignment-failed.json";

/// How long a recorded failure stops the same alignment from being tried again.
pub(crate) const ALIGNMENT_RETRY_AFTER: Duration = Duration::from_hours(24);

/// The command that tries the alignment again before the record expires. It
/// deletes the runtime tree, record included, so the next alignment runs.
pub(crate) const RETRY_ALIGNMENT_COMMAND: &str = "rocm engines install lemonade --reinstall";

/// What one failed alignment attempt tried, and when it failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AlignmentFailure {
    /// The active ROCm SDK version the backend could not be aligned to.
    pub(crate) target_rocm_version: String,
    /// The ROCm version `backend_versions.json` pinned before the attempt, which
    /// the revert put back.
    pub(crate) pinned_rocm_version: String,
    /// The llama.cpp build tag pinned before the attempt (tier 1 tried it).
    pub(crate) pinned_llamacpp_tag: Option<String>,
    /// The newest llama.cpp build tag tier 2 looked up, when the lookup worked.
    pub(crate) latest_llamacpp_tag: Option<String>,
    /// The Lemonade version whose runtime tree the attempt ran in.
    pub(crate) lemonade_version: String,
    /// When the attempt failed, in seconds since the Unix epoch.
    pub(crate) failed_at_unix_secs: u64,
}

/// What the runtime directory holds in place of a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoredAlignmentFailure {
    Absent,
    /// A record file exists but cannot be read or parsed; the text says why.
    Unreadable(String),
    Recorded(AlignmentFailure),
}

pub(crate) fn alignment_failure_path(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(ALIGNMENT_FAILURE_RECORD)
}

pub(crate) fn read_alignment_failure(runtime_dir: &Path) -> StoredAlignmentFailure {
    let path = alignment_failure_path(runtime_dir);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return StoredAlignmentFailure::Absent;
        }
        Err(error) => {
            return StoredAlignmentFailure::Unreadable(format!(
                "could not read {}: {error}",
                path.display()
            ));
        }
    };
    match serde_json::from_slice(&bytes) {
        Ok(record) => StoredAlignmentFailure::Recorded(record),
        Err(error) => StoredAlignmentFailure::Unreadable(format!(
            "could not parse {}: {error}",
            path.display()
        )),
    }
}

pub(crate) fn write_alignment_failure(runtime_dir: &Path, record: &AlignmentFailure) -> Result<()> {
    let path = alignment_failure_path(runtime_dir);
    rocm_core::atomic_write::write_file_atomically(&path, &serde_json::to_vec_pretty(record)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Remove the record. A missing record is not an error.
pub(crate) fn clear_alignment_failure(runtime_dir: &Path) -> Result<()> {
    let path = alignment_failure_path(runtime_dir);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

/// Seconds since the Unix epoch; zero for a clock set before it.
pub(crate) fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

impl AlignmentFailure {
    /// Whether the alignment should be skipped now: the same attempt (target,
    /// pin and Lemonade version) failed less than [`ALIGNMENT_RETRY_AFTER`] ago.
    ///
    /// A failure time later than `now` (the clock moved back) does not skip:
    /// trying again costs an install, and a record whose age cannot be known
    /// must not stop the alignment indefinitely.
    pub(crate) fn skips_alignment(
        &self,
        target_rocm_version: &str,
        pinned_rocm_version: &str,
        lemonade_version: &str,
        now: SystemTime,
    ) -> bool {
        self.target_rocm_version == target_rocm_version
            && self.pinned_rocm_version == pinned_rocm_version
            && self.lemonade_version == lemonade_version
            && self.age(now).is_some_and(|age| age < ALIGNMENT_RETRY_AFTER)
    }

    fn age(&self, now: SystemTime) -> Option<Duration> {
        let failed_at = UNIX_EPOCH.checked_add(Duration::from_secs(self.failed_at_unix_secs))?;
        now.duration_since(failed_at).ok()
    }

    /// The line printed when this record skipped the alignment and the pinned
    /// backend was installed instead. Built from the record and the same `now`
    /// [`Self::skips_alignment`] was asked with, so the times it names are the
    /// ones the decision used.
    pub(crate) fn skip_notice(&self, now: SystemTime) -> String {
        let age = self.age(now).unwrap_or_default();
        let remaining = ALIGNMENT_RETRY_AFTER.saturating_sub(age);
        format!(
            "Lemonade's ROCm backend could not be aligned to ROCm {target} {ago}; using the \
             backend pinned for ROCm {pinned} instead. Run `{RETRY_ALIGNMENT_COMMAND}` to try \
             again now, or the alignment is tried again {when}.",
            target = self.target_rocm_version,
            ago = describe_ago(age),
            pinned = self.pinned_rocm_version,
            when = describe_in(remaining),
        )
    }
}

const HOUR: u64 = 60 * 60;

/// "less than an hour ago", "1 hour ago", "5 hours ago": whole hours, rounded down.
fn describe_ago(age: Duration) -> String {
    match age.as_secs() / HOUR {
        0 => "less than an hour ago".to_owned(),
        1 => "1 hour ago".to_owned(),
        hours => format!("{hours} hours ago"),
    }
}

/// "in less than an hour", "in 1 hour", "in 5 hours": whole hours, rounded up
/// so it never names a time before the record expires.
fn describe_in(remaining: Duration) -> String {
    if remaining.as_secs() < HOUR {
        return "in less than an hour".to_owned();
    }
    match remaining.as_secs().div_ceil(HOUR) {
        1 => "in 1 hour".to_owned(),
        hours => format!("in {hours} hours"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;

    fn record(failed_at_unix_secs: u64) -> AlignmentFailure {
        AlignmentFailure {
            target_rocm_version: "7.99.0".to_owned(),
            pinned_rocm_version: "7.13.0".to_owned(),
            pinned_llamacpp_tag: Some("b9752".to_owned()),
            latest_llamacpp_tag: Some("b10952".to_owned()),
            lemonade_version: "11.5.1".to_owned(),
            failed_at_unix_secs,
        }
    }

    fn at(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    const FAILED_AT: u64 = 1_800_000_000;

    #[test]
    fn a_record_round_trips_and_clearing_it_twice_is_fine() {
        let dir = scratch_dir("alignment-record-round-trip");
        assert_eq!(read_alignment_failure(&dir), StoredAlignmentFailure::Absent);

        write_alignment_failure(&dir, &record(FAILED_AT)).unwrap();
        assert_eq!(
            read_alignment_failure(&dir),
            StoredAlignmentFailure::Recorded(record(FAILED_AT))
        );

        clear_alignment_failure(&dir).unwrap();
        assert!(!alignment_failure_path(&dir).exists());
        clear_alignment_failure(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_record_reads_as_unreadable_not_absent() {
        let dir = scratch_dir("alignment-record-corrupt");
        fs::write(alignment_failure_path(&dir), b"{ not json").unwrap();

        let StoredAlignmentFailure::Unreadable(reason) = read_alignment_failure(&dir) else {
            panic!("a corrupt record must read as unreadable");
        };
        assert!(reason.contains(ALIGNMENT_FAILURE_RECORD), "{reason}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_record_skips_only_the_same_attempt_within_the_retry_window() {
        let failure = record(FAILED_AT);
        let skips =
            |target, pinned, lemonade, now| failure.skips_alignment(target, pinned, lemonade, now);
        let window = ALIGNMENT_RETRY_AFTER.as_secs();

        assert!(skips("7.99.0", "7.13.0", "11.5.1", at(FAILED_AT)));
        assert!(skips(
            "7.99.0",
            "7.13.0",
            "11.5.1",
            at(FAILED_AT + window - 1)
        ));
        assert!(
            !skips("7.99.0", "7.13.0", "11.5.1", at(FAILED_AT + window)),
            "a record exactly 24 hours old has expired"
        );
        assert!(
            !skips("7.98.0", "7.13.0", "11.5.1", at(FAILED_AT)),
            "other target"
        );
        assert!(
            !skips("7.99.0", "7.12.0", "11.5.1", at(FAILED_AT)),
            "other pin"
        );
        assert!(
            !skips("7.99.0", "7.13.0", "11.6.0", at(FAILED_AT)),
            "other Lemonade"
        );
        assert!(
            !skips("7.99.0", "7.13.0", "11.5.1", at(FAILED_AT - 1)),
            "a failure time in the future does not skip"
        );
    }

    #[test]
    fn the_skip_notice_names_the_target_the_pin_the_command_and_the_times() {
        let failure = record(FAILED_AT);

        let fresh = failure.skip_notice(at(FAILED_AT + 60));
        assert!(
            fresh.contains("aligned to ROCm 7.99.0 less than an hour ago"),
            "{fresh}"
        );
        assert!(fresh.contains("pinned for ROCm 7.13.0"), "{fresh}");
        assert!(
            fresh.contains("`rocm engines install lemonade --reinstall`"),
            "{fresh}"
        );
        assert!(fresh.contains("tried again in 24 hours"), "{fresh}");

        let later = failure.skip_notice(at(FAILED_AT + 5 * HOUR + 1));
        assert!(later.contains("5 hours ago"), "{later}");
        assert!(later.contains("tried again in 19 hours"), "{later}");

        let one = failure.skip_notice(at(FAILED_AT + HOUR));
        assert!(one.contains("1 hour ago"), "{one}");
        assert!(one.contains("in 23 hours"), "{one}");

        let last = failure.skip_notice(at(FAILED_AT + 23 * HOUR + 1));
        assert!(last.contains("in less than an hour"), "{last}");
        let penultimate = failure.skip_notice(at(FAILED_AT + 23 * HOUR));
        assert!(penultimate.contains("in 1 hour"), "{penultimate}");
    }
}
