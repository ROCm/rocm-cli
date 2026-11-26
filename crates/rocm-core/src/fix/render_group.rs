// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The groups `fix-4-render-group` adds the user to, decided in one place.
//!
//! `rocm diagnose` decides them from an examination (the group that owns
//! `/dev/kfd` and [`Examination::groups_on_host`]), and `rocm fix` decides them
//! from the same two facts read live (`run_render_group` in `fix/runners.rs`).
//! Both hand those facts to [`render_group_targets`] and build the `usermod`
//! arguments with [`usermod_args`], so the line the diagnosis prints and the
//! command the fix runs cannot name different groups.
//!
//! [`Examination::groups_on_host`]: crate::examine::Examination::groups_on_host

/// The group the fix falls back to when the group that owns `/dev/kfd` cannot
/// be used: the group the kernel's udev rules give `/dev/kfd` on current
/// distributions.
pub(crate) const DEFAULT_DEVICE_GROUP: &str = "render";

/// The group added beside the device's group when it exists: older
/// distributions give `/dev/kfd` and the render nodes to `video`.
pub(crate) const COMPANION_GROUP: &str = "video";

/// The `stat` format the examination and the fix read a device node with:
/// mode, owning user and owning group, `|`-separated. One format for both, so a
/// stand-in `stat` in a test answers both the same way.
pub(crate) const DEVICE_STAT_FORMAT: &str = "%A|%U|%G";

/// The mode, owning user and owning group in one line of `stat -c
/// DEVICE_STAT_FORMAT` output, or `None` when the line has another shape.
pub(crate) fn parse_device_stat(out: &str) -> Option<(String, String, String)> {
    match out.trim().split('|').collect::<Vec<_>>()[..] {
        [mode, user, group] => Some((mode.to_owned(), user.to_owned(), group.to_owned())),
        _ => None,
    }
}

/// Whether `name` can be a group name that `usermod -G` accepts as one.
///
/// Refuses what `stat -c %G` prints when it has no name for the group: the
/// literal `UNKNOWN` (in any case), and an empty answer. Refuses an all-digit
/// value, which `usermod` would read as a group id rather than a name, and
/// anything outside the portable name characters, which includes a leading
/// `-` that `getent` and `usermod` would take for an option.
pub(crate) fn is_group_name(name: &str) -> bool {
    !name.is_empty()
        && !name.eq_ignore_ascii_case("UNKNOWN")
        && !name.starts_with('-')
        && !name.bytes().all(|b| b.is_ascii_digit())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

/// The groups whose existence decides the plan, in lookup order and without
/// repeats: the group that owns `/dev/kfd` when [`is_group_name`] accepts it,
/// then [`DEFAULT_DEVICE_GROUP`] and [`COMPANION_GROUP`].
pub(crate) fn lookup_candidates(owner_group: &str) -> Vec<&str> {
    let mut candidates = Vec::with_capacity(3);
    for group in [owner_group, DEFAULT_DEVICE_GROUP, COMPANION_GROUP] {
        if is_group_name(group) && !candidates.contains(&group) {
            candidates.push(group);
        }
    }
    candidates
}

/// The arguments `getent` is run with to learn which of
/// [`lookup_candidates`] exist: `group` and the candidates.
pub(crate) fn getent_args(owner_group: &str) -> Vec<&str> {
    std::iter::once("group")
        .chain(lookup_candidates(owner_group))
        .collect()
}

/// Whether `getent` exited with an answer: `0` when every key was found, `2`
/// when one or more was not (`getent(1)`). Any other code, or a `getent` that
/// did not run, leaves the group database unread.
pub(crate) const fn getent_answered(code: i32) -> bool {
    matches!(code, 0 | 2)
}

/// The group names in `getent group` output: the first `:`-separated field of
/// each line.
pub(crate) fn parse_getent_group(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| line.split(':').next())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The groups to add the user to, or none when no group can be named.
///
/// The first is the group that owns `/dev/kfd` when it is a group name
/// ([`is_group_name`]) that exists on the host, otherwise
/// [`DEFAULT_DEVICE_GROUP`] when that exists. With neither there is nothing to
/// add, and the result is empty: a `usermod` naming a group the host does not
/// have fails, and the device's group is unknown. [`COMPANION_GROUP`] follows
/// when it exists and is not already first.
///
/// `group_exists` answers from the examination for `rocm diagnose` and from a
/// live `getent group` for `rocm fix`.
pub(crate) fn render_group_targets(
    owner_group: &str,
    group_exists: impl Fn(&str) -> bool,
) -> Vec<String> {
    let usable = |group: &str| is_group_name(group) && group_exists(group);
    let first = if usable(owner_group) {
        owner_group
    } else if usable(DEFAULT_DEVICE_GROUP) {
        DEFAULT_DEVICE_GROUP
    } else {
        return Vec::new();
    };
    let mut targets = vec![first.to_owned()];
    if first != COMPANION_GROUP && usable(COMPANION_GROUP) {
        targets.push(COMPANION_GROUP.to_owned());
    }
    targets
}

/// The `usermod` arguments that add `user` to `groups`, a comma-joined
/// [`render_group_targets`] result. The diagnosis passes `"$USER"`, the fix the
/// user's name.
pub(crate) const fn usermod_args<'a>(groups: &'a str, user: &'a str) -> [&'a str; 4] {
    ["-a", "-G", groups, user]
}

/// Who is asking why there is no group to add, because the sentence names
/// what read `/dev/kfd`'s owner and the host's groups.
#[derive(Clone, Copy)]
pub(crate) enum PlanReader {
    /// `rocm diagnose`, reading an `Examination`; `database_read` is whether
    /// the host's group database answered at all.
    Examination { database_read: bool },
    /// `rocm fix`, which reads both live and stops before planning when
    /// `getent` does not answer, so the database was read.
    Live,
}

/// Why there is no group to add, as one sentence.
pub(crate) fn no_target_reason(owner_group: &str, reader: PlanReader) -> String {
    if matches!(
        reader,
        PlanReader::Examination {
            database_read: false
        }
    ) {
        return "the examination does not say which groups the host has (`getent group` did \
                not answer, or the CLI that examined it predates the check), and no group is \
                named before it is checked"
            .to_owned();
    }
    let owner = if owner_group.is_empty() {
        match reader {
            PlanReader::Live => {
                "no owning group of /dev/kfd could be read (the node is missing or `stat` \
                 failed on it)"
                    .to_owned()
            }
            PlanReader::Examination { .. } => {
                "/dev/kfd has no owning group this examination could read".to_owned()
            }
        }
    } else if is_group_name(owner_group) {
        format!(
            "/dev/kfd is owned by '{owner_group}', which the host's group database does not name"
        )
    } else {
        format!("/dev/kfd is owned by '{owner_group}', which is not a group name")
    };
    format!("{owner}, and the host has no '{DEFAULT_DEVICE_GROUP}' group")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(owner: &str, existing: &[&str]) -> Vec<String> {
        render_group_targets(owner, |g| existing.contains(&g))
    }

    /// The wording names what read the host: `rocm diagnose` an examination,
    /// `rocm fix` the live host. The live wording is held end to end, with the
    /// exit code and nothing run, by
    /// `render_group_with_no_owning_group_says_so_in_live_wording`.
    #[test]
    fn the_no_group_reason_names_who_read_the_host() {
        let examined = |database_read| PlanReader::Examination { database_read };
        assert_eq!(
            no_target_reason("", examined(true)),
            concat!(
                "/dev/kfd has no owning group this examination could read, ",
                "and the host has no 'render' group"
            )
        );
        assert!(no_target_reason("", examined(false)).starts_with("the examination does not say"));
        let live = no_target_reason("", PlanReader::Live);
        assert!(!live.contains("examination"), "{live}");
        assert!(
            live.starts_with("no owning group of /dev/kfd could be read"),
            "{live}"
        );
    }

    #[test]
    fn the_owning_group_comes_first_and_video_is_not_repeated() {
        let all = ["render", "video", "kfd"];
        assert_eq!(targets("render", &all), ["render", "video"]);
        assert_eq!(targets("kfd", &all), ["kfd", "video"]);
        assert_eq!(targets("video", &all), ["video"]);
    }

    #[test]
    fn a_group_with_no_name_falls_back_to_render() {
        for owner in ["", "UNKNOWN", "unknown", "110", "-rf", "a b"] {
            assert_eq!(
                targets(owner, &["render", "video", "110", "-rf", "a b"]),
                ["render", "video"],
                "{owner:?}"
            );
        }
    }

    #[test]
    fn a_group_the_host_does_not_have_is_never_named() {
        assert_eq!(targets("kfd", &["render", "video"]), ["render", "video"]);
        assert_eq!(targets("render", &["render"]), ["render"]);
        assert!(targets("kfd", &["video"]).is_empty());
        assert!(targets("UNKNOWN", &[]).is_empty());
    }

    #[test]
    fn getent_output_names_its_groups() {
        assert_eq!(
            parse_getent_group("render:x:110:\nvideo:x:44:alice,bob\n"),
            ["render", "video"]
        );
        assert!(parse_getent_group("").is_empty());
        assert!(getent_answered(0) && getent_answered(2));
        assert!(!getent_answered(1) && !getent_answered(3));
    }

    #[test]
    fn only_group_names_are_looked_up() {
        assert_eq!(getent_args("kfd"), ["group", "kfd", "render", "video"]);
        assert_eq!(getent_args("render"), ["group", "render", "video"]);
        assert_eq!(getent_args("UNKNOWN"), ["group", "render", "video"]);
        assert_eq!(getent_args("-x"), ["group", "render", "video"]);
    }

    #[test]
    fn a_stat_line_reads_as_mode_user_and_group() {
        assert_eq!(
            parse_device_stat("crw-rw----|root|render\n"),
            Some((
                "crw-rw----".to_owned(),
                "root".to_owned(),
                "render".to_owned()
            ))
        );
        assert_eq!(parse_device_stat("render"), None);
    }
}
