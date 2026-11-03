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
    RuntimeVersionSelector, SourceLayout, TheRockChannel, resolve_source, select_source_layout,
};
use super::super::version::{compare_version_strings, is_stable_runtime_version};
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
        format!(
            "canonical TheRock {} tarball stream is incomplete for the resolved GPU family\n\n{}",
            channel.as_str(),
            family_resolution_hint(
                &family_resolution.source,
                &family_resolution.family,
                channel,
                "tarball",
            )
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
    let prefix = format!(
        "therock-dist-{}-{}-",
        platform_tarball_token(),
        tarball_family_token(layout, family)
    );
    let mut candidates = files
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
            if version_selector.is_some_and(|selector| !selector.matches_version(&version)) {
                return None;
            }
            Some((file.clone(), version))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.0
            .mtime
            .partial_cmp(&right.0.mtime)
            .unwrap_or(Ordering::Equal)
            .then_with(|| compare_version_strings(&left.1, &right.1))
    });
    candidates.pop()
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
}
