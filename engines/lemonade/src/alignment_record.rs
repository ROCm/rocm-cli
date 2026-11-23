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

use crate::REINSTALL_LEMONADE_COMMAND;

/// File name of the record, inside the Lemonade runtime directory.
pub(crate) const ALIGNMENT_FAILURE_RECORD: &str = ".rocm-cli-alignment-failed.json";

/// How long a recorded failure stops the same alignment from being tried again.
pub(crate) const ALIGNMENT_RETRY_AFTER: Duration = Duration::from_hours(24);

/// What one failed alignment attempt tried, and when it failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AlignmentFailure {
    /// The active ROCm SDK version the backend could not be aligned to.
    pub(crate) target_rocm_version: String,
    /// The ROCm version `backend_versions.json` pinned before the attempt, which
    /// the revert put back.
    pub(crate) pinned_rocm_version: String,
    /// The llama.cpp build tag pinned before the attempt (tier 1 tried it).
    /// Informational, for a person reading the file: the skip decision
    /// ([`Self::skips_alignment`]) does not read it.
    pub(crate) pinned_llamacpp_tag: Option<String>,
    /// The llama.cpp build tag tier 2 looked up (the newest one that ships an
    /// archive for the target ROCm version), when the lookup worked.
    /// Informational, for a person reading the file: the skip decision does not
    /// read it.
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
    /// The record is not keyed on the rocm-cli version: this engine cannot see
    /// the per-release version today, so a rocm-cli upgrade does not clear the
    /// record. It expires after [`ALIGNMENT_RETRY_AFTER`], and
    /// [`REINSTALL_LEMONADE_COMMAND`] clears it sooner.
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

    /// The age of a record [`Self::skips_alignment`] accepted for this `now`. Its
    /// callers print only after that, and a skip needs a known age, so the
    /// default is never taken.
    fn age_of_skipping_record(&self, now: SystemTime) -> Duration {
        let age = self.age(now);
        debug_assert!(
            age.is_some(),
            "a record that skipped the alignment has a known age"
        );
        age.unwrap_or_default()
    }

    /// The line printed when this record skipped the alignment and the pinned
    /// backend was installed instead. Built from the record and the same `now`
    /// [`Self::skips_alignment`] was asked with, so the times it names are the
    /// ones the decision used. Nothing retries on its own: only an install that
    /// runs once the record has expired attempts the alignment, so the line names
    /// the earliest such install rather than a time something happens. That time
    /// comes from [`ALIGNMENT_RETRY_AFTER`], the same window the decision uses.
    pub(crate) fn skip_notice(&self, now: SystemTime) -> String {
        let age = self.age_of_skipping_record(now);
        let remaining = ALIGNMENT_RETRY_AFTER.saturating_sub(age);
        format!(
            "Lemonade's ROCm backend could not be aligned to ROCm {target} {ago}; using the \
             backend pinned for ROCm {pinned} instead. Run `{REINSTALL_LEMONADE_COMMAND}` to try \
             again now; otherwise an install run {when} or later tries it again.",
            target = self.target_rocm_version,
            ago = describe_ago(age),
            pinned = self.pinned_rocm_version,
            when = describe_from_now(remaining),
        )
    }

    /// The context added to the error when the record skipped the alignment and
    /// preparing the pinned backend then failed. That preparation can fail before
    /// anything is installed (no free port, `lemond` not starting or not
    /// answering, the backend listing, no supported GPU backend) and Lemonade may
    /// pick a backend other than ROCm, so the line claims neither an install nor
    /// a backend kind. It says what [`REINSTALL_LEMONADE_COMMAND`] does -- re-extract
    /// the runtime tree without the record, so it tries the alignment again and
    /// downloads the backend again -- and not that it fixes the error beneath,
    /// which may be transient, or that it avoids the pinned backend, which a
    /// failing alignment reverts to.
    pub(crate) fn skipped_install_failure(&self, now: SystemTime) -> String {
        let age = self.age_of_skipping_record(now);
        format!(
            "preparing Lemonade's llama.cpp backend as pinned for ROCm {pinned} failed; the \
             alignment to ROCm {target} was skipped because it failed {ago}. \
             `{REINSTALL_LEMONADE_COMMAND}` re-extracts the runtime tree without that record, \
             so it tries the alignment again (downloading the backend again)",
            pinned = self.pinned_rocm_version,
            target = self.target_rocm_version,
            ago = describe_ago(age),
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

/// "1 hour from now", "5 hours from now": whole hours, rounded up from the full
/// duration (a part of a second counts) so it never names a time before the
/// record expires. Nothing left, which [`AlignmentFailure::skip_notice`] never
/// passes since a record with no time left skips nothing, still reads as
/// "1 hour from now" rather than "0 hours".
fn describe_from_now(remaining: Duration) -> String {
    const HOUR_NANOS: u128 = HOUR as u128 * 1_000_000_000;
    match remaining.as_nanos().div_ceil(HOUR_NANOS).max(1) {
        1 => "1 hour from now".to_owned(),
        hours => format!("{hours} hours from now"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;

    /// `ALIGNMENT_FAILURE_RECORD` in the black-box e2e steps (used by
    /// runtime-33) hard-codes this name to find the record on disk. Read from that
    /// file, so a rename on either side fails here, on every lane, rather than
    /// only when the nightly scenario runs. The include is test-only, and no
    /// workspace crate is packaged for publishing (`publish = false`).
    #[test]
    fn the_record_file_name_matches_the_one_the_e2e_steps_hard_code() {
        let steps = include_str!("../../../tests/e2e-cucumber/tests/e2e/lemonade_install_steps.rs");
        let definition =
            format!("const ALIGNMENT_FAILURE_RECORD: &str = \"{ALIGNMENT_FAILURE_RECORD}\";");
        assert!(
            steps.contains(&definition),
            "lemonade_install_steps.rs no longer defines `{definition}`"
        );
    }

    /// The e2e steps that check the CLI names the reinstall command hard-code it,
    /// since they test the binary from outside. Read from those files, so a
    /// change to the command on either side fails here on every lane, rather
    /// than only when the gated scenarios run (runtime-33 and the serve retry
    /// scenario).
    #[test]
    fn the_reinstall_command_matches_the_one_the_e2e_steps_hard_code() {
        let steps = [
            (
                "lemonade_install_steps.rs",
                include_str!("../../../tests/e2e-cucumber/tests/e2e/lemonade_install_steps.rs"),
            ),
            (
                "serving_steps.rs",
                include_str!("../../../tests/e2e-cucumber/tests/e2e/serving_steps.rs"),
            ),
        ];
        for (name, source) in steps {
            assert!(
                source.contains(REINSTALL_LEMONADE_COMMAND),
                "{name} no longer checks for `{REINSTALL_LEMONADE_COMMAND}`"
            );
        }
    }

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
        // Pins the advice only. That `--reinstall` removes the record is
        // `reinstall_removes_the_record_of_a_failed_alignment` in `install.rs`, and
        // that the alignment then runs is `with_no_record_the_alignment_runs_as_before`
        // in `backend_alignment.rs`.
        assert!(
            fresh.contains(&format!("`{REINSTALL_LEMONADE_COMMAND}`")),
            "{fresh}"
        );
        // The time named comes from the window the decision uses, so the line
        // cannot name a different one than `skips_alignment` applies.
        let window_hours = ALIGNMENT_RETRY_AFTER.as_secs() / HOUR;
        assert!(
            fresh.ends_with(&format!(
                "to try again now; otherwise an install run {window_hours} hours from now or \
                 later tries it again."
            )),
            "{fresh}"
        );
        // The docs (docs/engine-plugins.md, docs/manual-testing.md) name 24 hours.
        assert_eq!(window_hours, 24, "update the docs that name the window");

        let later = failure.skip_notice(at(FAILED_AT + 5 * HOUR + 1));
        assert!(later.contains("5 hours ago"), "{later}");
        assert!(
            later.contains("an install run 19 hours from now or later"),
            "{later}"
        );

        let one = failure.skip_notice(at(FAILED_AT + HOUR));
        assert!(one.contains("1 hour ago"), "{one}");
        assert!(one.contains("run 23 hours from now"), "{one}");

        let last = failure.skip_notice(at(FAILED_AT + 23 * HOUR + 1));
        assert!(last.contains("run 1 hour from now or later"), "{last}");
        let penultimate = failure.skip_notice(at(FAILED_AT + 23 * HOUR));
        assert!(
            penultimate.contains("run 1 hour from now or later"),
            "{penultimate}"
        );
        // The last second of the window still names a time after it.
        let final_second = failure.skip_notice(at(FAILED_AT + 24 * HOUR - 1));
        assert!(
            final_second.contains("run 1 hour from now or later"),
            "{final_second}"
        );
    }

    /// `age` keeps the clock's sub-second part (the failure time is whole
    /// seconds), so the time left is rarely whole seconds and a part of a second
    /// must round up rather than be dropped.
    #[test]
    fn the_time_left_rounds_up_from_its_sub_second_part() {
        let failure = record(FAILED_AT);
        let window = ALIGNMENT_RETRY_AFTER.as_secs();

        // 3,600.7 s left: "1 hour" would name a time 0.7 s before expiry.
        let over_an_hour =
            failure.skip_notice(at(FAILED_AT + window - HOUR - 1) + Duration::from_millis(300));
        assert!(
            over_an_hour.contains("run 2 hours from now or later"),
            "{over_an_hour}"
        );

        // 0.3 s left.
        let last_moment =
            failure.skip_notice(at(FAILED_AT + window - 1) + Duration::from_millis(700));
        assert!(
            last_moment.contains("run 1 hour from now or later"),
            "{last_moment}"
        );

        // No time left at all reads as the shortest wait, never "0 hours".
        assert_eq!(describe_from_now(Duration::ZERO), "1 hour from now");
        assert_eq!(
            describe_from_now(Duration::from_nanos(1)),
            "1 hour from now"
        );
    }
}
