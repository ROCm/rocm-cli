// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Resolving the SDK tarball a request installs: reading the catalog's
//! scrapeable index (`parse_tarball_index_html`, with `validate_tarball_file_name`
//! refusing an entry that is not a plain file name), choosing the newest archive
//! this request can install (`select_tarball_candidate`, `tarball_family_token`),
//! and tying that to the resolved family and source (`resolve_tarball_artifact`).
//!
//! Builds on `fetch`, `family`, `source` and `version`. The one item it reads
//! from `therock.rs` is the platform token it matches archive names against,
//! `platform_tarball_token`, which stays there for good with the other keys and
//! roots helpers.

use super::super::family::{family_resolution_hint, resolve_family};
use super::super::fetch::download_text_cached;
use super::super::platform_tarball_token;
use super::super::source::{
    FamilyResolution, RuntimeVersionSelector, SourceLayout, TheRockChannel, resolve_source,
    select_source_layout, selector_is_release_line,
};
use super::super::version::{compare_version_strings, is_stable_runtime_version};
use super::wheel::next_layout_unsatisfied_hint;
use anyhow::{Context, Result, bail};
use rocm_core::AppPaths;
use serde::Deserialize;
use std::cmp::Ordering;
use std::path::{Component, Path};

#[derive(Debug, Clone)]
pub(in crate::therock) struct TarballArtifact {
    pub(in crate::therock) family: String,
    pub(in crate::therock) family_source: String,
    /// The catalog this archive was listed in, and which layout that catalog
    /// is, so provenance reports the stream the artifact actually came from
    /// rather than re-deriving it from the channel.
    pub(in crate::therock) catalog_url: String,
    pub(in crate::therock) layout: SourceLayout,
    pub(in crate::therock) file_name: String,
    pub(in crate::therock) version: String,
    pub(in crate::therock) url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TarballIndexFile {
    name: String,
    mtime: f64,
}

pub(in crate::therock) fn resolve_tarball_artifact(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
) -> Result<TarballArtifact> {
    resolve_tarball_artifact_with_timeout(
        paths,
        channel,
        family_override,
        version_selector,
        layout_override,
        None,
    )
}

pub(in crate::therock) fn resolve_tarball_artifact_with_timeout(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
    download_timeout_secs: Option<u64>,
) -> Result<TarballArtifact> {
    let family_resolution = resolve_family(paths, family_override)?;
    let layout = match layout_override {
        Some(layout) => layout,
        None => select_source_layout(channel, &family_resolution, version_selector)?,
    };
    let source = resolve_source(channel, layout);
    let catalog_cache_key = match layout {
        SourceLayout::Canonical => format!("tarball-index-{}", channel.as_str()),
        SourceLayout::Next => format!("next-tarball-index-{}", channel.as_str()),
    };
    let html = download_text_cached(
        paths,
        &catalog_cache_key,
        &source.tarball_catalog,
        download_timeout_secs,
    )?
    .text;
    let files = parse_tarball_index_html(&html).with_context(|| {
        format!(
            "unknown canonical TheRock tarball catalog layout at {}",
            source.tarball_catalog
        )
    })?;
    let (file, version) = select_tarball_candidate(
        &files,
        channel,
        layout,
        &family_resolution.family,
        version_selector,
    )
    .with_context(|| {
        let published =
            tarball_family_candidates(&files, channel, layout, &family_resolution.family)
                .into_iter()
                .map(|(_, version)| version)
                .collect::<Vec<_>>();
        tarball_resolution_failure_context(
            channel,
            layout,
            &family_resolution,
            version_selector,
            &published,
        )
    })?;
    Ok(TarballArtifact {
        family: family_resolution.family,
        family_source: family_resolution.source,
        url: format!(
            "{}/{}",
            source.tarball_catalog.trim_end_matches('/'),
            file.name
        ),
        catalog_url: source.tarball_catalog,
        layout,
        file_name: file.name,
        version,
    })
}

/// The filename token a catalog spells `family` with.
///
/// ROCm 10's catalog renamed exactly one family's archive token; every other
/// family keeps its canonical spelling. Confined here rather than pushed into
/// [`normalize_therock_family`] so a filename quirk cannot leak into the family
/// a manifest records or the extras a wheel install requests.
///
/// [`normalize_therock_family`]: rocm_core::normalize_therock_family
fn tarball_family_token(layout: SourceLayout, family: &str) -> &str {
    if matches!(layout, SourceLayout::Next) && family == "gfx103X-dgpu" {
        "gfx103X-all"
    } else {
        family
    }
}

/// Every archive in `files` this channel can install for `family`, before any
/// version pin narrows them, paired with the version its name carries.
///
/// [`select_tarball_candidate`] picks from these. A next-layout failure lists
/// their versions, so the releases it suggests are ones the pick could take.
fn tarball_family_candidates(
    files: &[TarballIndexFile],
    channel: TheRockChannel,
    layout: SourceLayout,
    family: &str,
) -> Vec<(TarballIndexFile, String)> {
    let prefix = format!(
        "therock-dist-{}-{}-",
        platform_tarball_token(),
        tarball_family_token(layout, family)
    );
    files
        .iter()
        .filter_map(|file| {
            let version = file
                .name
                .strip_prefix(&prefix)?
                .strip_suffix(".tar.gz")?
                .to_owned();
            if matches!(channel, TheRockChannel::Release) && !is_stable_runtime_version(&version) {
                return None;
            }
            Some((file.clone(), version))
        })
        .collect()
}

/// The newest archive in `files` that this request can actually install.
///
/// The release channel's stable-version filter does double duty on the next
/// catalog: that catalog publishes non-release siblings beside the real dist
/// archive (`...-tests-10.0.0.tar.gz`) whose leftover suffix is not a version at
/// all, and whose mtime is *later* than the archive they shadow, so an unfiltered
/// "newest wins" would pick the wrong file. Requiring a parseable stable version
/// excludes them without inventing a second grammar for the canonical catalogs,
/// which publish no such siblings and have never been held to one.
fn select_tarball_candidate(
    files: &[TarballIndexFile],
    channel: TheRockChannel,
    layout: SourceLayout,
    family: &str,
    version_selector: Option<&RuntimeVersionSelector>,
) -> Option<(TarballIndexFile, String)> {
    let mut candidates = tarball_family_candidates(files, channel, layout, family);
    if let Some(selector) = version_selector {
        candidates.retain(|(_, version)| selector.matches_version(version));
    }
    // An unpinned catalog pick is "newest upload". A major.minor line pin is
    // "newest stable release in that line", even when an older patch was
    // uploaded later. Exact pins match one archive, so the order does not
    // change which file they select.
    let prefer_newest_line_release = version_selector.is_some_and(selector_is_release_line);
    candidates.sort_by(|left, right| {
        let by_mtime = left
            .0
            .mtime
            .partial_cmp(&right.0.mtime)
            .unwrap_or(Ordering::Equal);
        let by_version = compare_version_strings(&left.1, &right.1);
        if prefer_newest_line_release {
            by_version.then(by_mtime)
        } else {
            by_mtime.then(by_version)
        }
    });
    candidates.pop()
}

fn tarball_resolution_failure_context(
    channel: TheRockChannel,
    layout: SourceLayout,
    family_resolution: &FamilyResolution,
    version_selector: Option<&RuntimeVersionSelector>,
    published: &[String],
) -> String {
    let (stream, hint) = match layout {
        SourceLayout::Next => (
            "next-layout",
            next_layout_unsatisfied_hint(
                version_selector,
                Some(published),
                family_resolution.raw_arch.as_deref(),
                Some(&family_resolution.family),
            ),
        ),
        SourceLayout::Canonical => (
            "canonical",
            family_resolution_hint(
                &family_resolution.source,
                &family_resolution.family,
                channel,
                "tarball",
            ),
        ),
    };
    format!(
        "{stream} TheRock {} tarball stream is incomplete for the resolved GPU family\n\n{hint}",
        channel.as_str()
    )
}

fn parse_tarball_index_html(html: &str) -> Result<Vec<TarballIndexFile>> {
    let start = html
        .find("const files = ")
        .context("tarball index did not contain the embedded file list")?;
    let json_start = start + "const files = ".len();
    let rest = &html[json_start..];
    let end = rest
        .find("];")
        .context("tarball index did not contain the end of the embedded file list")?;
    let json = format!("{}]", &rest[..end]);
    let files: Vec<TarballIndexFile> =
        serde_json::from_str(&json).context("failed to parse TheRock tarball index file list")?;
    for file in &files {
        validate_tarball_file_name(&file.name)?;
    }
    Ok(files)
}

fn validate_tarball_file_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    let stem = name
        .split_once('.')
        .map_or(name, |(stem, _)| stem)
        .trim_end_matches([' ', '.']);
    let windows_reserved = matches!(
        stem.to_ascii_uppercase().as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    );
    if name.contains(['/', '\\', ':'])
        || name.ends_with([' ', '.'])
        || windows_reserved
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || name.chars().any(char::is_control)
    {
        bail!("tarball catalog contains unsafe file name `{name}`");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::family_for_test;
    use super::*;
    use rocm_core::normalize_therock_family;

    #[test]
    fn tarball_catalog_rejects_unsafe_file_names_at_the_parser_boundary() {
        for name in [
            "../escape.tar.gz",
            "folder/escape.tar.gz",
            r"folder\escape.tar.gz",
            "/absolute.tar.gz",
            "control\nname.tar.gz",
            "archive.tar.gz:payload",
            "CON.tar.gz",
            "archive.tar.gz.",
            "archive.tar.gz ",
        ] {
            let html =
                format!(r#"<script>const files = [{{"name":"{name}","mtime":1.0}}];</script>"#);
            assert!(
                parse_tarball_index_html(&html).is_err(),
                "unsafe catalog name was accepted: {name:?}"
            );
        }
        let html = r#"<script>const files = [{"name":"therock-dist-linux-gfx120X-all-10.0.0.tar.gz","mtime":1.0}];</script>"#;
        assert_eq!(
            parse_tarball_index_html(html)
                .expect("safe catalog name must parse")
                .len(),
            1
        );
    }

    /// The next catalog's gfx103X token differs from the canonical token, and
    /// its later `-tests-` sibling must not win selection.
    #[test]
    fn next_gfx103x_tarball_selection_uses_alias_and_skips_tests_sibling() {
        let platform = platform_tarball_token();
        let next_prefix = format!("therock-dist-{platform}-gfx103X-all-");
        let files = vec![
            TarballIndexFile {
                name: format!("{next_prefix}10.0.0.tar.gz"),
                mtime: 1_787_612_008.0,
            },
            TarballIndexFile {
                name: format!("{next_prefix}tests-10.0.0.tar.gz"),
                mtime: 1_787_612_032.0,
            },
        ];

        assert!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Canonical,
                "gfx103X-dgpu",
                None,
            )
            .is_none(),
            "the canonical layout must not recognize the next-only family alias"
        );

        let (file, version) = select_tarball_candidate(
            &files,
            TheRockChannel::Release,
            SourceLayout::Next,
            "gfx103X-dgpu",
            None,
        )
        .expect("the next alias must select the real dist archive");

        assert_eq!(file.name, format!("{next_prefix}10.0.0.tar.gz"));
        assert_eq!(version, "10.0.0");
    }

    /// The rename `tarball_family_token` applies for the next catalog must not
    /// drift from what `normalize_therock_family` maps back to `gfx103X-dgpu`;
    /// a family manifests under is worthless if the catalog's own alias for it
    /// no longer round-trips.
    ///
    /// The first assertion is the one that actually pins the rename:
    /// `normalize_therock_family` already maps *both* `gfx103X-dgpu` and its
    /// `gfx103X-all` alias back to `gfx103X-dgpu` (that is the whole point of
    /// the alias), so asserting only the round-trip's final value passes
    /// identically whether or not `tarball_family_token` renames anything —
    /// it does not distinguish the rename from its absence.
    #[test]
    fn next_tarball_family_token_round_trips_through_normalize_therock_family() {
        let token = tarball_family_token(SourceLayout::Next, "gfx103X-dgpu");
        assert_eq!(token, "gfx103X-all");
        assert_eq!(
            normalize_therock_family(token),
            Some("gfx103X-dgpu".to_owned())
        );
    }

    /// A pinned tarball install must install what it named, not whatever is
    /// newest in the catalog.
    #[test]
    fn pinned_tarball_selection_installs_the_named_version() {
        let platform = platform_tarball_token();
        let prefix = format!("therock-dist-{platform}-gfx120X-all-");
        let files = vec![
            TarballIndexFile {
                name: format!("{prefix}10.0.0.tar.gz"),
                mtime: 1.0,
            },
            TarballIndexFile {
                name: format!("{prefix}10.1.0.tar.gz"),
                mtime: 2.0,
            },
        ];
        let pin = RuntimeVersionSelector::version("10.0.0").unwrap();

        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Next,
                "gfx120X-all",
                Some(&pin)
            )
            .map(|(_, version)| version),
            Some("10.0.0".to_owned())
        );
    }

    /// The next catalog spells exactly one family differently; the canonical
    /// catalog and every other family are untouched, and the family a manifest
    /// records never changes.
    #[test]
    fn next_tarball_token_renames_only_the_gfx103x_family() {
        assert_eq!(
            tarball_family_token(SourceLayout::Next, "gfx103X-dgpu"),
            "gfx103X-all"
        );
        assert_eq!(
            tarball_family_token(SourceLayout::Canonical, "gfx103X-dgpu"),
            "gfx103X-dgpu"
        );
        assert_eq!(
            tarball_family_token(SourceLayout::Next, "gfx120X-all"),
            "gfx120X-all"
        );
    }

    #[test]
    fn tarball_selection_never_crosses_channels() {
        let platform = platform_tarball_token();
        let files = vec![
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-7.14.0.tar.gz"),
                mtime: 1.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.1.0a20260822.tar.gz"),
                mtime: 2.0,
            },
        ];

        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Canonical,
                "gfx120X-all",
                None
            )
            .map(|(_, version)| version),
            Some("7.14.0".to_owned())
        );
        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Nightly,
                SourceLayout::Canonical,
                "gfx120X-all",
                None
            )
            .map(|(_, version)| version),
            Some("10.1.0a20260822".to_owned())
        );
    }

    #[test]
    fn release_line_pin_prefers_the_newest_stable_tarball_in_the_line() {
        let platform = platform_tarball_token();
        let files = vec![
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.1.0.tar.gz"),
                mtime: 5.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.1.2.tar.gz"),
                mtime: 1.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.1.0a20260822.tar.gz"),
                mtime: 9.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-10.0.0.tar.gz"),
                mtime: 8.0,
            },
            TarballIndexFile {
                name: format!("therock-dist-{platform}-gfx120X-all-tests-10.1.3.tar.gz"),
                mtime: 9.0,
            },
        ];
        let line = RuntimeVersionSelector::version("10.1").unwrap();
        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Next,
                "gfx120X-all",
                Some(&line),
            )
            .map(|(_, version)| version),
            Some("10.1.2".to_owned())
        );

        let exact = RuntimeVersionSelector::version("10.1.0").unwrap();
        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Next,
                "gfx120X-all",
                Some(&exact),
            )
            .map(|(_, version)| version),
            Some("10.1.0".to_owned())
        );

        // Unpinned, the same catalog keeps "newest upload wins": 10.0.0 was
        // uploaded after both 10.1 releases, so it is the pick even though
        // its version is lower. Only a line pin orders by version first.
        assert_eq!(
            select_tarball_candidate(
                &files,
                TheRockChannel::Release,
                SourceLayout::Next,
                "gfx120X-all",
                None,
            )
            .map(|(_, version)| version),
            Some("10.0.0".to_owned())
        );
    }

    #[test]
    fn tarball_failure_on_next_layout_suggests_a_published_release() {
        let family = family_for_test("gfx120X-all", Some("gfx1200"));
        let line = RuntimeVersionSelector::version("10.1").unwrap();
        let next = tarball_resolution_failure_context(
            TheRockChannel::Release,
            SourceLayout::Next,
            &family,
            Some(&line),
            &["10.0.0".to_owned()],
        );
        assert!(
            next.starts_with("next-layout TheRock release tarball"),
            "{next}"
        );
        assert!(
            next.contains("`--version 10.0.0 --family gfx1200`"),
            "{next}"
        );
        assert!(!next.contains("nightly"), "{next}");

        let canonical = tarball_resolution_failure_context(
            TheRockChannel::Release,
            SourceLayout::Canonical,
            &family,
            None,
            &[],
        );
        assert!(
            canonical.starts_with("canonical TheRock release tarball"),
            "{canonical}"
        );
        assert!(canonical.contains("--channel nightly"), "{canonical}");
    }
}
