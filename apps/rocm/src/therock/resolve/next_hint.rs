// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What a failed ROCm 10 resolution tells the user to do next.
//!
//! Only a failure a different `--version` could clear carries a version
//! suggestion. Such a failure is raised as a [`VersionUnsatisfied`]: the
//! requested release is not published, the release it resolved to has no
//! torchaudio for this Python and platform, or `uv` found no jointly
//! installable package set for it. Every other failure gets no version advice:
//! an undetermined device target, a `uv` that could not be fetched or run, or a
//! listing that could not be read. A different version leaves each of those
//! exactly where it was.
//!
//! [`next_layout_unsatisfied_hint`] writes the suggestion. A pinned install
//! (`rocm install sdk --version …`) is told which `--version`/`--family` to
//! re-run with. `rocm update`, which re-resolves an installed runtime with no
//! pin and has no `--family` or `--version` of its own, is given a whole
//! `rocm install sdk` command that keeps the runtime's format and toolchain
//! extras ([`NextLayoutRecovery`]).
//!
//! A leaf child of `therock/resolve.rs` that builds on `therock/source.rs` and
//! `therock/version.rs`. `wheel.rs` and `tarball.rs` raise the marker and call
//! the hint.

use super::super::source::{RuntimeVersionSelector, THEROCK_NEXT_MIN_MAJOR, next_layout_requested};
use super::super::version::compare_version_strings;
use std::fmt;

/// A next-layout resolution failure that a different `--version` could clear.
///
/// `attempted` is the release the request resolved to before it failed, when it
/// resolved to one. The hint leaves that release out, because suggesting it
/// again cannot help. The resolver that picked it records it here. A hint that
/// re-derived it would have to repeat that resolver's ordering (by version for a
/// wheel listing, by upload time for an unpinned tarball pick) and could get it
/// wrong.
///
/// Its `Display` is the failure's own message, so raising this in place of a
/// plain error leaves what the user reads unchanged.
#[derive(Debug)]
pub(in crate::therock) struct VersionUnsatisfied {
    attempted: Option<String>,
    message: String,
}

impl VersionUnsatisfied {
    pub(in crate::therock) fn new(attempted: Option<&str>, message: String) -> Self {
        Self {
            attempted: attempted.map(str::to_owned),
            message,
        }
    }

    pub(in crate::therock) fn attempted(&self) -> Option<&str> {
        self.attempted.as_deref()
    }

    /// The marker anywhere in `error`'s chain, whatever context was added on top.
    pub(in crate::therock) fn find(error: &anyhow::Error) -> Option<&Self> {
        error.chain().find_map(|cause| cause.downcast_ref::<Self>())
    }
}

impl fmt::Display for VersionUnsatisfied {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VersionUnsatisfied {}

/// The artifact kind a failed resolution was for, so a suggested install
/// command installs the same kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::therock) enum RecoveryFormat {
    Wheel,
    Tarball,
}

/// What a suggested command has to carry over from the request that failed.
#[derive(Clone, Copy, Debug)]
pub(in crate::therock) struct NextLayoutRecovery<'a> {
    pub(in crate::therock) format: RecoveryFormat,
    /// The exact arch the request resolved with, so the command repeats it.
    pub(in crate::therock) exact_arch: Option<&'a str>,
    /// Whether the runtime has the `devel` toolchain extras. Only an update
    /// carries it into the command; a pinned install re-runs its own command,
    /// flags included.
    pub(in crate::therock) include_devel: bool,
}

/// How many published releases a next-layout failure lists by name.
const NEXT_LAYOUT_HINT_MAX_LISTED: usize = 5;

/// Recovery guidance when a ROCm >= 10 request cannot be satisfied on the next
/// layout, for a failure raised as a [`VersionUnsatisfied`].
///
/// It never offers `--channel nightly`, because that channel refuses these pins.
///
/// The suggested `--version` is one the next layout publishes and is never the
/// release that just failed. `published` is that layout's version listing
/// (`None` when the listing could not be read), narrowed to one GPU family when
/// `family_scope` names it. `attempted` is the release the failure recorded,
/// and the hint leaves it out. It also leaves out any release
/// [`next_layout_requested`] would not route back to this layout, so a re-run
/// searches the same layout. When nothing else qualifies, the hint says so and
/// names no command.
///
/// The command depends on the caller:
/// - A pinned install (`version_selector` is `Some`) is told which
///   `--version`/`--family` pair to re-run with. Its other flags are its own.
/// - `rocm update` (`version_selector` is `None`, because an update re-resolves
///   with no pin) has no such flags. It is given the whole `rocm install sdk`
///   command, with the runtime's format and toolchain extras, so following it
///   installs the same kind of runtime.
pub(in crate::therock) fn next_layout_unsatisfied_hint(
    version_selector: Option<&RuntimeVersionSelector>,
    published: Option<&[String]>,
    attempted: Option<&str>,
    recovery: NextLayoutRecovery<'_>,
    family_scope: Option<&str>,
) -> String {
    let layout = format!("the ROCm {THEROCK_NEXT_MIN_MAJOR}+ next layout");
    let Some(published) = published else {
        return format!(
            "Could not read the releases {layout} publishes, so no other version can be suggested."
        );
    };
    let requested = version_selector.map_or_else(
        || "the newest release".to_owned(),
        RuntimeVersionSelector::describe,
    );
    let scope = family_scope.map_or_else(String::new, |family| format!(" for {family}"));
    let lead = match recovery.format {
        RecoveryFormat::Wheel => {
            format!("No complete compatible package stack was found for {requested} on {layout}.")
        }
        RecoveryFormat::Tarball => {
            format!("No archive{scope} was found for {requested} on {layout}.")
        }
    };
    let mut alternatives = published
        .iter()
        .filter(|version| attempted != Some(version.as_str()))
        .filter(|version| {
            next_layout_requested(&RuntimeVersionSelector::Version((*version).clone()))
        })
        .map(String::as_str)
        .collect::<Vec<_>>();
    alternatives.sort_by(|left, right| compare_version_strings(right, left));
    alternatives.dedup();
    let other = if attempted.is_some() { " other" } else { "" };
    let Some(newest) = alternatives.first() else {
        return format!(
            "{lead} It publishes no{other} stable release{scope}, so no `--version` pin can install from it yet."
        );
    };
    let mut listed = alternatives
        .iter()
        .take(NEXT_LAYOUT_HINT_MAX_LISTED)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if alternatives.len() > NEXT_LAYOUT_HINT_MAX_LISTED {
        let older = alternatives.len() - NEXT_LAYOUT_HINT_MAX_LISTED;
        listed.push_str(", and ");
        listed.push_str(&older.to_string());
        listed.push_str(" older");
    }
    let command = recovery_command(version_selector.is_some(), newest, recovery);
    let label = if attempted.is_some() {
        "Other stable releases it publishes"
    } else {
        "Stable releases it publishes"
    };
    format!("{lead} {label}{scope}: {listed}. {command}")
}

/// The sentence naming what to run. `pinned` is whether the failed request
/// carried a `--version`.
fn recovery_command(pinned: bool, version: &str, recovery: NextLayoutRecovery<'_>) -> String {
    // A ROCm 10 pin needs an exact arch, so a placeholder stands in only where
    // none is known, which a pinned install never reaches: it is refused
    // before resolution without one.
    let family = recovery.exact_arch.unwrap_or("<gfx>");
    let placeholder_note = if recovery.exact_arch.is_none() {
        ", where `<gfx>` is the raw `gfx` code `rocm examine` reports"
    } else {
        ""
    };
    if pinned {
        return format!(
            "Re-run with one of them, for example `--version {version} --family {family}`{placeholder_note}."
        );
    }
    let format = match recovery.format {
        RecoveryFormat::Wheel => "wheel",
        RecoveryFormat::Tarball => "tarball",
    };
    let devel = if recovery.include_devel {
        " --devel"
    } else {
        ""
    };
    format!(
        "Install one of them, for example `rocm install sdk --channel release --format {format} --version {version} --family {family}{devel}`{placeholder_note}."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions(list: &[&str]) -> Vec<String> {
        list.iter().map(|version| (*version).to_owned()).collect()
    }

    const WHEEL_AT_GFX1200: NextLayoutRecovery<'static> = NextLayoutRecovery {
        format: RecoveryFormat::Wheel,
        exact_arch: Some("gfx1200"),
        include_devel: false,
    };

    /// Every state the hint can be reached in. The rules under test: it never
    /// suggests the release the failure recorded, never names a command when
    /// the index offers nothing else, and a suggested install command
    /// installs the same kind of runtime that failed.
    #[test]
    fn next_layout_unsatisfied_hint_only_suggests_releases_that_were_not_tried() {
        let line = RuntimeVersionSelector::version("10.1").unwrap();
        let exact = RuntimeVersionSelector::version("10.1.0").unwrap();

        // A line pin that resolved to 10.1.2 and could not install it: the
        // other 10.1 release is still worth trying, 10.1.2 is not.
        let hint = next_layout_unsatisfied_hint(
            Some(&line),
            Some(&versions(&["10.0.0", "10.1.0", "10.1.2"])),
            Some("10.1.2"),
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("Other stable releases it publishes: 10.1.0, 10.0.0."),
            "{hint}"
        );
        // A pinned request re-runs its own command with a different
        // `--version`, so it is told to re-run, not handed a whole install
        // command. The unpinned `rocm update` case below gets the full
        // command instead.
        assert!(
            hint.contains(
                "Re-run with one of them, for example `--version 10.1.0 --family gfx1200`."
            ),
            "{hint}"
        );
        assert!(!hint.contains("Install one of them"), "{hint}");
        assert!(!hint.contains("rocm install sdk"), "{hint}");
        assert!(!hint.contains("10.1.2"), "{hint}");

        // The release the failure recorded is what is left out, not whatever
        // version order would pick: an unpinned tarball pick goes by upload
        // time, so it can have tried an older release than the newest.
        let hint = next_layout_unsatisfied_hint(
            None,
            Some(&versions(&["10.0.0", "10.1.0", "10.1.2"])),
            Some("10.1.0"),
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("Other stable releases it publishes: 10.1.2, 10.0.0."),
            "{hint}"
        );

        // An exact pin that is published but failed is not suggested again.
        let hint = next_layout_unsatisfied_hint(
            Some(&exact),
            Some(&versions(&["10.0.0", "10.1.0"])),
            Some("10.1.0"),
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("`--version 10.0.0 --family gfx1200`"),
            "{hint}"
        );
        assert!(!hint.contains("--version 10.1.0"), "{hint}");

        // An exact pin that is the only release: nothing else to suggest, so
        // no command is named.
        let hint = next_layout_unsatisfied_hint(
            Some(&exact),
            Some(&versions(&["10.1.0"])),
            Some("10.1.0"),
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("It publishes no other stable release"),
            "{hint}"
        );
        assert!(!hint.contains("--version 10"), "{hint}");

        // A pin nothing matched: no release was attempted, and none of the
        // published ones is "other".
        let hint = next_layout_unsatisfied_hint(
            Some(&line),
            Some(&versions(&["10.0.0"])),
            None,
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("Stable releases it publishes: 10.0.0. "),
            "{hint}"
        );

        // An index with no stable release at all; prereleases are not offered.
        let hint = next_layout_unsatisfied_hint(
            Some(&line),
            Some(&versions(&["10.1.0a20260822"])),
            None,
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(hint.contains("It publishes no stable release"), "{hint}");
        assert!(!hint.contains("10.1.0a20260822"), "{hint}");
        assert!(!hint.contains("--version 10"), "{hint}");

        // The listing itself could not be read: say that, not "no stack was
        // found", and suggest nothing.
        let hint = next_layout_unsatisfied_hint(Some(&line), None, None, WHEEL_AT_GFX1200, None);
        assert!(
            hint.starts_with("Could not read the releases the ROCm 10+ next layout publishes"),
            "{hint}"
        );
        assert!(
            !hint.contains("No complete compatible package stack"),
            "{hint}"
        );
        assert!(!hint.contains("--version 10"), "{hint}");

        // Only releases a re-run would route back to this layout are offered.
        // A four-component or sub-10 publish would re-resolve against the
        // canonical index, so suggesting it could not clear the failure.
        let hint = next_layout_unsatisfied_hint(
            Some(&line),
            Some(&versions(&["9.9.0", "10.0.0", "10.0.1.12345"])),
            None,
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("Stable releases it publishes: 10.0.0. "),
            "{hint}"
        );
        assert!(!hint.contains("10.0.1.12345"), "{hint}");
        assert!(!hint.contains("9.9.0"), "{hint}");

        // A long listing is capped, newest first.
        let hint = next_layout_unsatisfied_hint(
            Some(&line),
            Some(&versions(&[
                "10.0.0", "10.0.1", "10.0.2", "10.0.3", "10.0.4", "10.0.5", "10.0.6",
            ])),
            None,
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            hint.contains("10.0.6, 10.0.5, 10.0.4, 10.0.3, 10.0.2, and 2 older."),
            "{hint}"
        );
    }

    /// `rocm update` reaches the next layout with no pin. It has no `--family`
    /// or `--version` of its own, so the hint names a whole `rocm install sdk`
    /// command, and that command installs what the failed runtime was: the same
    /// format and the same toolchain extras.
    #[test]
    fn an_update_is_given_an_install_command_that_keeps_its_format_and_extras() {
        let published = versions(&["10.0.0", "10.1.0"]);

        let wheel_with_toolchain = next_layout_unsatisfied_hint(
            None,
            Some(&published),
            Some("10.1.0"),
            NextLayoutRecovery {
                format: RecoveryFormat::Wheel,
                exact_arch: Some("gfx1200"),
                include_devel: true,
            },
            None,
        );
        assert!(
            wheel_with_toolchain.contains(
                "Install one of them, for example `rocm install sdk --channel release --format wheel --version 10.0.0 --family gfx1200 --devel`."
            ),
            "{wheel_with_toolchain}"
        );
        assert!(
            wheel_with_toolchain.contains("for the newest release"),
            "{wheel_with_toolchain}"
        );
        assert!(
            !wheel_with_toolchain.contains("Re-run with one of them"),
            "{wheel_with_toolchain}"
        );
        assert!(
            !wheel_with_toolchain.contains("nightly"),
            "{wheel_with_toolchain}"
        );

        let plain_wheel = next_layout_unsatisfied_hint(
            None,
            Some(&published),
            Some("10.1.0"),
            WHEEL_AT_GFX1200,
            None,
        );
        assert!(
            plain_wheel.contains(
                "`rocm install sdk --channel release --format wheel --version 10.0.0 --family gfx1200`."
            ),
            "{plain_wheel}"
        );

        // A tarball runtime's update names a tarball install, and the catalog
        // it read was the family's, so the hint is about archives, not a
        // package stack. A tarball runtime records only its grouped family, so
        // no exact arch is known: the command says where to get one, and does
        // not claim the update itself takes `--family`.
        let tarball = next_layout_unsatisfied_hint(
            None,
            Some(&published),
            None,
            NextLayoutRecovery {
                format: RecoveryFormat::Tarball,
                exact_arch: None,
                include_devel: false,
            },
            Some("gfx120X-all"),
        );
        assert!(
            tarball.starts_with(
                "No archive for gfx120X-all was found for the newest release on the ROCm 10+ next layout."
            ),
            "{tarball}"
        );
        assert!(
            tarball.contains(
                "`rocm install sdk --channel release --format tarball --version 10.1.0 --family <gfx>`, where `<gfx>` is the raw `gfx` code `rocm examine` reports."
            ),
            "{tarball}"
        );
        assert!(!tarball.contains("package stack"), "{tarball}");
        assert!(!tarball.contains("re-run with `--family`"), "{tarball}");
    }

    /// The marker is found however many context layers sit on top of it, and a
    /// failure without one is not mistaken for it.
    #[test]
    fn the_marker_survives_added_context_and_keeps_its_message() {
        let marked = anyhow::Error::new(VersionUnsatisfied::new(
            Some("10.1.0"),
            "no torchaudio built for ROCm 10.1.0".to_owned(),
        ))
        .context("failed to resolve")
        .context("outer");
        let found = VersionUnsatisfied::find(&marked).expect("marker is in the chain");
        assert_eq!(found.attempted.as_deref(), Some("10.1.0"));
        assert!(
            format!("{marked:#}").contains("no torchaudio built for ROCm 10.1.0"),
            "{marked:#}"
        );

        let plain = anyhow::anyhow!("uv could not be downloaded").context("failed to resolve");
        assert!(VersionUnsatisfied::find(&plain).is_none());
    }
}
