// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `PATH` a scenario hands the `rocm` child when it plants a machine that
//! holds a `rocminfo` of its own.
//!
//! The scenarios about whether `rocminfo` is installed must not be decided by a
//! `rocminfo` the runner happens to have, so the runner's own `PATH` entries
//! that hold one are left out. The entries are written by the harness process
//! but read by the child, and the two can run from different directories (the
//! scenario's `command_dir`), so a relative entry such as `.` means two
//! different directories. [`planted_path_entries`] settles that before it
//! filters: each inherited relative entry is made absolute against the
//! harness's own working directory, and is kept or dropped by what that
//! directory holds. The fixture's own entries are never touched, because a
//! scenario writes them to be read from the child's directory (an empty entry
//! and `.` are the point of two of them).

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The program the runner's entries are checked for.
pub const PLANTED_TOOL: &str = "rocminfo";

/// The entries of the planted `PATH`, in order: `fixture` exactly as written,
/// then `fake_bin`, then each non-empty entry of `inherited` that holds no
/// [`PLANTED_TOOL`].
///
/// An inherited relative entry is resolved against `harness_cwd` first, checked
/// there with `holds_tool`, and kept in its resolved (absolute) form, so it still
/// names the directory it named when the harness read it after the child has
/// moved. An inherited empty entry is dropped: it would be read as the child's
/// working directory, which the scenario decides and the runner must not.
pub fn planted_path_entries(
    fixture: &[PathBuf],
    fake_bin: &Path,
    inherited: &OsStr,
    harness_cwd: &Path,
    holds_tool: impl Fn(&Path) -> bool,
) -> Vec<PathBuf> {
    let kept = std::env::split_paths(inherited)
        .filter(|entry| !entry.as_os_str().is_empty())
        // `join` leaves an absolute entry as it is and roots a relative one.
        .map(|entry| harness_cwd.join(entry))
        .filter(|resolved| !holds_tool(resolved));
    fixture
        .iter()
        .cloned()
        .chain(std::iter::once(fake_bin.to_owned()))
        .chain(kept)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An absolute directory on any host, so the rows are about the resolution
    /// rule and not about one platform's path syntax.
    fn abs(name: &str) -> PathBuf {
        std::env::temp_dir().join("planted-path-rows").join(name)
    }

    fn joined(entries: &[PathBuf]) -> std::ffi::OsString {
        std::env::join_paths(entries).expect("no entry holds a separator")
    }

    /// The planted entries for a host whose `rocminfo` is in exactly `holding`,
    /// with the harness running from `abs("harness")`.
    fn planted(fixture: &[PathBuf], inherited: &[PathBuf], holding: &[PathBuf]) -> Vec<PathBuf> {
        planted_path_entries(
            fixture,
            &abs("fake-bin"),
            &joined(inherited),
            &abs("harness"),
            |dir| holding.iter().any(|held| held == dir),
        )
    }

    #[test]
    fn an_inherited_dot_without_a_rocminfo_survives_as_the_harness_directory() {
        // The child runs from elsewhere, where a bare `.` would name the planted
        // directory and expose its stand-in, so it must not survive as `.`.
        assert_eq!(
            planted(&[], &[PathBuf::from(".")], &[]),
            vec![abs("fake-bin"), abs("harness")]
        );
    }

    #[test]
    fn an_inherited_dot_with_a_rocminfo_in_the_harness_directory_is_dropped() {
        assert_eq!(
            planted(&[], &[PathBuf::from(".")], &[abs("harness")]),
            vec![abs("fake-bin")]
        );
    }

    #[test]
    fn an_inherited_absolute_entry_is_checked_and_kept_as_it_is() {
        assert_eq!(
            planted(&[], &[abs("usr-bin"), abs("rocm-bin")], &[abs("rocm-bin")]),
            vec![abs("fake-bin"), abs("usr-bin")]
        );
    }

    #[test]
    fn an_inherited_relative_name_is_resolved_against_the_harness_directory() {
        assert_eq!(
            planted(
                &[],
                &[PathBuf::from("bin"), PathBuf::from("tools")],
                &[abs("harness").join("tools")]
            ),
            vec![abs("fake-bin"), abs("harness").join("bin")]
        );
    }

    #[test]
    fn inherited_empty_entries_are_dropped_wherever_they_stand() {
        assert_eq!(
            planted(&[], &[PathBuf::new(), abs("usr-bin"), PathBuf::new()], &[]),
            vec![abs("fake-bin"), abs("usr-bin")]
        );
    }

    #[test]
    fn fixture_entries_come_first_and_are_left_exactly_as_written() {
        // The empty entry, `.` and a name are what the scenarios are about; the
        // harness directory holding a `rocminfo` makes no difference to them.
        let fixture = [PathBuf::new(), PathBuf::from("."), PathBuf::from("odd")];
        assert_eq!(
            planted(
                &fixture,
                &[PathBuf::from(".")],
                &[abs("harness"), PathBuf::from("."), PathBuf::from("odd")]
            ),
            vec![
                PathBuf::new(),
                PathBuf::from("."),
                PathBuf::from("odd"),
                abs("fake-bin")
            ]
        );
    }
}
