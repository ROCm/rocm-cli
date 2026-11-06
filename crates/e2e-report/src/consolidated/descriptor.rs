// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Artifact-name identity: parses a report's artifact/directory name (e.g.
//! `e2e-gpu-strix-windows-nightly-report`) into the platform, OS, legacy
//! known-bugs flag and channel it stands for, and owns the platform-label rule
//! ([`full_platform_label`]) every report surface derives its lane name from.

/// The platform/OS a report belongs to, parsed from its artifact name.
///
/// Splitting these into separate fields (rather than one mashed
/// "Gpu Strix Ubuntu" label) is what lets the matrix show distinct
/// Platform / OS columns.
pub(super) struct Descriptor {
    pub(super) platform: String,
    pub(super) os: String,
    /// True for a legacy `known-bugs` artifact (xfail-inverted). With the
    /// one-job-per-platform model these no longer exist, but the flag is retained
    /// as a stable secondary sort key so old artifacts still order predictably.
    pub(super) known_bugs: bool,
    /// The channel suffix stripped from the artifact name, if any. Only a
    /// fallback: [`PlatformReport::effective_channel`](super::PlatformReport::effective_channel) prefers the manifest's
    /// `versions.channel` and uses this solely for the errored-run case where
    /// no `platform.json` was ever written.
    pub(super) channel_from_name: Option<&'static str>,
}

/// Parse an artifact/dir name like `e2e-gpu-strix-windows-report`
/// into its Platform / OS. Unknown shapes fall back to a titlecased
/// platform on Linux so a new artifact still renders sensibly.
pub(super) fn parse_descriptor(name: &str) -> Descriptor {
    // Strip prefix, then suffix, each relative to the prior result (not `name`),
    // so `e2e-report` correctly reduces to the empty core, not back to itself.
    let core = name.strip_prefix("e2e-").unwrap_or(name);
    let core = core.strip_suffix("-report").unwrap_or(core);

    // The nightly workflow always appends the channel as the final segment
    // before `-report` (e.g. `e2e-gpu-strix-windows-nightly-report`). Captured
    // as `channel_from_name` — a fallback for the errored-run case where no
    // `platform.json` was written, so the manifest has no channel to read.
    let (core, channel_from_name) = match core.strip_suffix("-release") {
        Some(rest) => (rest, Some("release")),
        None => match core.strip_suffix("-nightly") {
            Some(rest) => (rest, Some("nightly")),
            None => (core, None),
        },
    };

    // Legacy `-known-bugs` suffix (retained only as a stable secondary sort key).
    let (core, known_bugs) = match core.strip_suffix("known-bugs") {
        Some(rest) => (rest.trim_end_matches('-'), true),
        None => (core, false),
    };

    // The self-hosted sdk_version matrix (ROCMAI-430) runs non-`current` legs
    // under a distinct artifact name (`e2e-gpu-n-1-report`) so the two legs'
    // uploads don't collide; strip that suffix BEFORE matching the platform
    // table below, then fold it back into the platform label so the two legs
    // still render as distinct rows/columns instead of two identical-looking
    // "MI300X Linux" entries.
    let (core, release_label) = match core.strip_suffix("-n-1") {
        Some(rest) => (rest, Some("n-1")),
        None => match core.strip_suffix("-n-2") {
            Some(rest) => (rest, Some("n-2")),
            None => (core, None),
        },
    };

    let (platform, os) = match core {
        // The bare mock expect-pass artifact is `e2e-report` → core "report" or "".
        "" | "report" => ("Mock", "Linux"),
        "gpu" => ("MI300X", "Linux"),
        "gpu-rad3" => ("R9700", "Linux"),
        "gpu-mi350p" => ("MI350P", "Linux"),
        "gpu-w7900" => ("W7900", "Linux"),
        "gpu-strix-ubuntu" => ("Strix Halo", "Ubuntu"),
        "gpu-strix-windows" => ("Strix Halo", "Windows"),
        // Same silicon again, third host boundary: an Ubuntu distro under WSL2 on
        // the Windows box. It is neither the native Ubuntu nor the Windows lane —
        // GPU access goes through the WSL passthrough — so it needs its own OS
        // value, not a reuse of "Ubuntu", or the grid would show two rows that
        // claim to be the same host and disagree.
        "gpu-strix-wsl" => ("Strix Halo", "WSL2"),
        // `e2e-unknown-report`: a report whose platform.json sidecar was missing
        // or unrecognized (e.g. a GPU run that errored before writing it). The OS
        // is genuinely unknown here — a Windows GPU run that erupted early must NOT
        // be reported as Linux — so render Unknown / Unknown rather than defaulting
        // OS to Linux the way `fallback_descriptor` does for a titlecased platform.
        "unknown" => ("Unknown", "Unknown"),
        other => return fallback_descriptor(other, known_bugs, channel_from_name, release_label),
    };

    Descriptor {
        platform: with_paren_suffix(platform, release_label),
        os: os.to_string(),
        known_bugs,
        channel_from_name,
    }
}

fn fallback_descriptor(
    core: &str,
    known_bugs: bool,
    channel_from_name: Option<&'static str>,
    release_label: Option<&str>,
) -> Descriptor {
    let platform = core
        .split('-')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or_else(String::new, |f| {
                f.to_uppercase().collect::<String>() + c.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ");
    let platform = if platform.is_empty() {
        "Unknown".to_string()
    } else {
        platform
    };
    Descriptor {
        platform: with_paren_suffix(&platform, release_label),
        os: "Linux".to_string(),
        known_bugs,
        channel_from_name,
    }
}

/// Append a "(suffix)" to `base` when `suffix` is known. The one place the
/// parenthesised-suffix shape lives: [`parse_descriptor`] uses it for the
/// sdk_version matrix leg (`n-1`/`n-2`) so two legs of the same platform don't
/// render as identical rows/columns (`None`, the `current` leg, is left alone
/// because its artifact name is unchanged from before the matrix existed), and
/// the labels below use it for the channel.
pub(super) fn with_paren_suffix(base: &str, suffix: Option<&str>) -> String {
    match suffix {
        Some(s) => format!("{base} ({s})"),
        None => base.to_string(),
    }
}

/// The one full platform label: platform AND OS, plus the channel when known
/// (e.g. `Strix Halo Windows (nightly)`).
///
/// Every surface that names a whole platform lane builds it here:
/// `PlatformReport::label` (built by [`PlatformReport::load`](super::PlatformReport::load); per-platform detail
/// sections, XPASS notes and the command-coverage column key),
/// [`GridColumn::label`](super::GridColumn::label) (grid headings and the
/// needs-attention list) and the excluded-platform caption
/// ([`Grid::excluded`](super::Grid::excluded)). The callers differ only in where
/// `channel` comes from (the manifest preferred over the artifact name for the
/// report label, the manifest alone for the grid heading, the artifact name
/// alone when there is no manifest), never in the shape of the label.
///
/// Deliberately not the OS-less [`platform_cell`](super::platform_cell), which is
/// only safe where the OS has its own table column: two real lanes of the same
/// silicon family (e.g. "Strix Halo" Ubuntu vs. Windows, see
/// `parse_descriptor`) must stay distinguishable everywhere else, or an excluded
/// Windows lane would read as the identical string to an included Ubuntu column
/// of the same platform — an outright contradiction in the rendered report — and
/// two such excluded lanes would silently collapse into one `BTreeSet` entry,
/// undercounting how many artifacts are actually missing.
pub(super) fn full_platform_label(platform: &str, os: &str, channel: Option<&str>) -> String {
    with_paren_suffix(&format!("{platform} {os}"), channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_descriptor_maps_known_artifacts() {
        for (name, platform, os) in [
            ("e2e-report", "Mock", "Linux"),
            ("e2e-gpu-report", "MI300X", "Linux"),
            ("e2e-gpu-rad3-report", "R9700", "Linux"),
            ("e2e-gpu-mi350p-report", "MI350P", "Linux"),
            ("e2e-gpu-w7900-report", "W7900", "Linux"),
            ("e2e-gpu-strix-ubuntu-report", "Strix Halo", "Ubuntu"),
            ("e2e-gpu-strix-windows-report", "Strix Halo", "Windows"),
            // Must not fall through to `fallback_descriptor`, which would render
            // "Gpu Strix Wsl" on Linux — a WSL2 host reported as native Linux.
            ("e2e-gpu-strix-wsl-report", "Strix Halo", "WSL2"),
        ] {
            let d = parse_descriptor(name);
            assert_eq!(
                (d.platform.as_str(), d.os.as_str()),
                (platform, os),
                "{name}"
            );
        }
    }

    #[test]
    fn parse_descriptor_strips_nightly_workflow_channel_suffix() {
        // Real artifact names from `.github/workflows/nightly.yml`, which always
        // inserts the channel as the final segment before `-report`. A synthetic
        // label like "mi300x-release" wouldn't reproduce the bug this guards
        // against: nightly-channel artifacts falling through to
        // `fallback_descriptor` and being mislabeled "Linux".
        for (name, platform, os) in [
            ("e2e-gpu-release-report", "MI300X", "Linux"),
            ("e2e-gpu-nightly-report", "MI300X", "Linux"),
            ("e2e-gpu-rad3-release-report", "R9700", "Linux"),
            ("e2e-gpu-rad3-nightly-report", "R9700", "Linux"),
            ("e2e-gpu-mi350p-nightly-report", "MI350P", "Linux"),
            (
                "e2e-gpu-strix-ubuntu-nightly-report",
                "Strix Halo",
                "Ubuntu",
            ),
            (
                "e2e-gpu-strix-windows-nightly-report",
                "Strix Halo",
                "Windows",
            ),
            ("e2e-gpu-strix-wsl-nightly-report", "Strix Halo", "WSL2"),
        ] {
            let d = parse_descriptor(name);
            assert_eq!(
                (d.platform.as_str(), d.os.as_str()),
                (platform, os),
                "{name}"
            );
        }
    }

    #[test]
    fn parse_descriptor_distinguishes_sdk_version_matrix_legs() {
        // The sdk_version matrix (ROCMAI-430) uploads non-`current` legs under a
        // distinct artifact name so they don't collide with each other or with
        // the `current` leg's unchanged name. Platform/OS must still resolve
        // correctly, with the leg folded into the platform label so the rows
        // don't render as two identical "MI300X Linux" entries.
        for (name, platform, os) in [
            ("e2e-gpu-n-1-report", "MI300X (n-1)", "Linux"),
            ("e2e-gpu-n-2-report", "MI300X (n-2)", "Linux"),
            ("e2e-gpu-strix-wsl-n-1-report", "Strix Halo (n-1)", "WSL2"),
        ] {
            let d = parse_descriptor(name);
            assert_eq!(
                (d.platform.as_str(), d.os.as_str()),
                (platform, os),
                "{name}"
            );
        }
    }

    #[test]
    fn full_platform_label_is_platform_os_and_optional_channel() {
        assert_eq!(
            full_platform_label("Strix Halo", "Windows", Some("nightly")),
            "Strix Halo Windows (nightly)"
        );
        assert_eq!(
            full_platform_label("Strix Halo", "Ubuntu", None),
            "Strix Halo Ubuntu"
        );
        // The label a parsed artifact yields keeps the sdk_version leg inside the
        // platform part, before the OS.
        let d = parse_descriptor("e2e-gpu-strix-wsl-n-1-report");
        assert_eq!(
            full_platform_label(&d.platform, &d.os, d.channel_from_name),
            "Strix Halo (n-1) WSL2"
        );
    }

    #[test]
    fn parse_descriptor_unknown_is_not_falsely_linux() {
        // A report whose platform.json was missing (e.g. a GPU run that errored
        // before writing the sidecar) is labeled `e2e-unknown-report`. Its OS is
        // genuinely unknown — a Windows GPU run must NOT be reported as Linux — so
        // both platform AND os render "Unknown", never a default "Linux".
        let d = parse_descriptor("e2e-unknown-report");
        assert_eq!(d.platform, "Unknown");
        assert_eq!(d.os, "Unknown");
    }
}
