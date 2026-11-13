// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Artifact-name identity: parses a report's artifact/directory name (e.g.
//! `e2e-gpu-strix-windows-nightly-report`) into the platform, OS, legacy
//! known-bugs flag and channel it stands for, and owns the platform-label rule
//! ([`full_platform_label`]) every report surface derives its lane name from.
//! The artifact-name table (`KNOWN_PLATFORMS`) is also what maps a
//! `platform.json` `platform_slug` back to an artifact name
//! ([`artifact_name_for_platform_slug`], used by `xtask` for a report downloaded
//! without its per-artifact directory), so the two directions read one table.

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
    let core = artifact_core(name);

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

    // `e2e--report` reduces to the empty core, which is the bare mock artifact
    // just like `e2e-report` (core "report") — see the `Mock` row below.
    let core = if core.is_empty() { "report" } else { core };
    let Some(known) = KNOWN_PLATFORMS
        .iter()
        .find(|k| artifact_core(k.artifact) == core)
    else {
        return fallback_descriptor(core, known_bugs, channel_from_name, release_label);
    };

    Descriptor {
        platform: with_paren_suffix(known.platform, release_label),
        os: known.os.to_string(),
        known_bugs,
        channel_from_name,
    }
}

/// The identity-bearing core of an artifact name: `e2e-` prefix and `-report`
/// suffix stripped, each relative to the prior result (not `name`), so
/// `e2e-report` correctly reduces to `report`, not back to itself.
fn artifact_core(name: &str) -> &str {
    let core = name.strip_prefix("e2e-").unwrap_or(name);
    core.strip_suffix("-report").unwrap_or(core)
}

/// One row of the artifact-name table: the report artifact a lane uploads, the
/// platform and OS it stands for, and the `platform.json` `platform_slug`s that
/// identify the same lane when only the sidecar survives.
struct KnownPlatform {
    /// The un-suffixed artifact name (no channel, no sdk_version leg).
    artifact: &'static str,
    platform: &'static str,
    os: &'static str,
    /// The manifest `platform_slug` values that map back to `artifact`. Empty for
    /// a row no sidecar can name (the unknown sentinel).
    slugs: &'static [&'static str],
}

/// The sentinel artifact name for a report whose platform could not be
/// recovered: a missing or unrecognized `platform.json`, e.g. a GPU run that
/// errored before writing it. Never uploaded by a workflow — it only ever comes
/// out of [`artifact_name_for_platform_slug`]. The OS is genuinely unknown here —
/// a Windows GPU run that erupted early must NOT be reported as Linux — so it
/// renders Unknown / Unknown rather than defaulting OS to Linux the way
/// `fallback_descriptor` does for a titlecased platform.
const UNKNOWN_PLATFORM_ARTIFACT: &str = "e2e-unknown-report";

/// Every artifact name [`parse_descriptor`] resolves to a real platform and OS,
/// and the sidecar slugs that map back to it ([`artifact_name_for_platform_slug`]).
/// The one table both directions read, so a lane added or renamed here cannot
/// leave the artifact-name side and the slug side disagreeing.
const KNOWN_PLATFORMS: &[KnownPlatform] = &[
    // The bare mock expect-pass artifact `e2e-report`.
    KnownPlatform {
        artifact: "e2e-report",
        platform: "Mock",
        os: "Linux",
        slugs: &["mock"],
    },
    // The GitHub-hosted Windows lane: no GPU, native Windows.
    KnownPlatform {
        artifact: "e2e-windows-report",
        platform: "Mock",
        os: "Windows",
        slugs: &["mock-windows"],
    },
    KnownPlatform {
        artifact: "e2e-gpu-report",
        platform: "MI300X",
        os: "Linux",
        slugs: &["mi300x"],
    },
    // The gfx target IS the slug for this lane: `platform_hardware_slug` has no
    // product-name mapping for it.
    KnownPlatform {
        artifact: "e2e-gpu-rad3-report",
        platform: "R9700",
        os: "Linux",
        slugs: &["gfx1201"],
    },
    KnownPlatform {
        artifact: "e2e-gpu-mi350p-report",
        platform: "MI350P",
        os: "Linux",
        slugs: &["mi350p"],
    },
    // Like rad3 above, the raw gfx target is the slug for the W7900 lane.
    KnownPlatform {
        artifact: "e2e-gpu-w7900-report",
        platform: "W7900",
        os: "Linux",
        slugs: &["gfx1100"],
    },
    KnownPlatform {
        artifact: "e2e-gpu-strix-ubuntu-report",
        platform: "Strix Halo",
        os: "Ubuntu",
        slugs: &["strix-halo-linux"],
    },
    KnownPlatform {
        artifact: "e2e-gpu-strix-windows-report",
        platform: "Strix Halo",
        os: "Windows",
        slugs: &["strix-halo-windows"],
    },
    // Same silicon again, third host boundary: an Ubuntu distro under WSL2 on
    // the Windows box. It is neither the native Ubuntu nor the Windows lane —
    // GPU access goes through the WSL passthrough — so it needs its own OS
    // value, not a reuse of "Ubuntu", or the grid would show two rows that
    // claim to be the same host and disagree. The WSL workflow is statically
    // pinned to the Strix WSL runner, so a bare `wsl` slug (the probe could not
    // recover a gfx target) names the same lane as `strix-halo-wsl`.
    KnownPlatform {
        artifact: "e2e-gpu-strix-wsl-report",
        platform: "Strix Halo",
        os: "WSL2",
        slugs: &["wsl", "strix-halo-wsl"],
    },
    KnownPlatform {
        artifact: UNKNOWN_PLATFORM_ARTIFACT,
        platform: "Unknown",
        os: "Unknown",
        slugs: &[],
    },
];

/// The artifact name a `platform.json` `platform_slug` stands for.
///
/// Lets a report downloaded without its per-artifact directory (and so without a
/// name) render identically to one that kept it. `None`, or a slug no lane
/// claims, gets [`UNKNOWN_PLATFORM_ARTIFACT`], never a guessed real platform: a
/// hardware failure must not be misattributed to Mock/Linux.
pub fn artifact_name_for_platform_slug(slug: Option<&str>) -> &'static str {
    slug.and_then(|slug| KNOWN_PLATFORMS.iter().find(|k| k.slugs.contains(&slug)))
        .map_or(UNKNOWN_PLATFORM_ARTIFACT, |k| k.artifact)
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
            ("e2e-windows-report", "Mock", "Windows"),
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
    fn every_known_platform_round_trips_between_slug_and_artifact_name() {
        // The two directions read one table, so the thing that can still drift is
        // a row whose artifact name `parse_descriptor` no longer resolves back to
        // that row (a typo'd or re-shaped name falls through to the titlecased
        // Linux guess), a slug claimed by two rows (the first would silently win),
        // or a slug that stops reaching its artifact.
        let mut seen_slugs = std::collections::BTreeSet::new();
        let mut seen_artifacts = std::collections::BTreeSet::new();
        for k in KNOWN_PLATFORMS {
            assert!(
                k.artifact.starts_with("e2e-") && k.artifact.ends_with("-report"),
                "{} is not an `e2e-*-report` artifact name",
                k.artifact
            );
            assert!(
                seen_artifacts.insert(k.artifact),
                "duplicate {}",
                k.artifact
            );
            // Artifact name -> platform/OS, bare and with a channel suffix (the
            // bare mock artifact never carries one: its core is just `report`).
            let core = artifact_core(k.artifact);
            let mut names = vec![k.artifact.to_string()];
            if core != "report" {
                names.push(format!("e2e-{core}-nightly-report"));
            }
            for name in names {
                let d = parse_descriptor(&name);
                assert_eq!(
                    (d.platform.as_str(), d.os.as_str()),
                    (k.platform, k.os),
                    "{name} must resolve to its own row, not the fallback guess"
                );
            }
            // Slug -> artifact name, for every slug the row claims.
            for slug in k.slugs {
                assert!(seen_slugs.insert(*slug), "slug {slug} claimed twice");
                assert_eq!(
                    artifact_name_for_platform_slug(Some(slug)),
                    k.artifact,
                    "slug {slug}"
                );
            }
        }
        assert!(!seen_slugs.is_empty());
    }

    #[test]
    fn platform_slugs_map_to_the_artifact_names_workflows_upload() {
        // Pinned independently of the table so editing a row's artifact name or
        // slug is a deliberate, visible change: these are the `platform_slug`
        // values the harness writes and the artifact names the workflows upload.
        for (slug, artifact) in [
            ("mock", "e2e-report"),
            ("mock-windows", "e2e-windows-report"),
            ("mi300x", "e2e-gpu-report"),
            ("gfx1201", "e2e-gpu-rad3-report"),
            ("mi350p", "e2e-gpu-mi350p-report"),
            ("gfx1100", "e2e-gpu-w7900-report"),
            ("strix-halo-linux", "e2e-gpu-strix-ubuntu-report"),
            ("strix-halo-windows", "e2e-gpu-strix-windows-report"),
            ("strix-halo-wsl", "e2e-gpu-strix-wsl-report"),
            ("wsl", "e2e-gpu-strix-wsl-report"),
        ] {
            assert_eq!(artifact_name_for_platform_slug(Some(slug)), artifact);
        }
    }

    #[test]
    fn missing_or_unrecognized_slug_is_the_unknown_artifact_never_mock() {
        for slug in [None, Some(""), Some("some-future-gpu"), Some("Mock")] {
            let name = artifact_name_for_platform_slug(slug);
            assert_eq!(name, "e2e-unknown-report", "{slug:?}");
            let d = parse_descriptor(name);
            assert_eq!((d.platform.as_str(), d.os.as_str()), ("Unknown", "Unknown"));
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
