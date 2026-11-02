// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reading what a wheel index publishes: the package links of the aggregate
//! root and the exact device targets they name (`parse_aggregate_package_links`,
//! `parse_aggregate_device_targets`, `validate_aggregate_index_layout`), the
//! device payload a host gets out of them (`AggregateDeviceTarget`,
//! `ResolvedAggregateWheelSource`), each package's simple-index listing
//! (`load_simple_index_versions`, `parse_simple_index_versions`, with the
//! `wheel_*_tag_matches` helpers filtering by the interpreter's wheel tags),
//! and the PEP 440 order of the versions found there (`PackageReleaseKey`,
//! `package_release_key`).
//!
//! Builds on `fetch`, `python`, `source` and `version`; `slugify`, which keys
//! the cached listing, stays in `therock.rs` with the other keys helpers.

use super::super::fetch::download_text_cached;
use super::super::python::WheelCompatibility;
use super::super::slugify;
use super::super::source::SourceLayout;
use super::super::version::compare_version_strings;
use anyhow::{Result, bail};
use rocm_core::{AppPaths, extract_first_gfx_token, normalize_therock_family};

/// The package names a canonical aggregate index links to, lowercased.
///
/// The simple index publishes one `<a href="<name>/">` per package. Matching on
/// parsed names rather than on raw substrings keeps `rocm` from being satisfied
/// by `rocm-sdk-core` and `torch` from being satisfied by
/// `amd-torch-device-gfx1100`.
fn parse_aggregate_package_links(html: &str) -> Vec<String> {
    let mut names = Vec::new();
    for tail in html.split("href=\"").skip(1) {
        let Some(href) = tail.split('"').next() else {
            continue;
        };
        let name = href.trim().trim_matches('/').to_ascii_lowercase();
        if name.is_empty() || name.contains('/') {
            continue;
        }
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// The exact GFX targets the canonical source publishes a device payload for.
///
/// This is the authoritative list rather than a table in this file: the
/// aggregate `rocm` distribution declares one `device-<target>` extra per
/// `rocm-sdk-device-<target>` package the index links, and `rocm_sdk` refuses
/// any target outside that set. Reading it from the stream means a target the
/// source adds or withdraws needs no CLI change, and a target it never
/// published cannot be requested by accident.
pub(in crate::therock) fn parse_aggregate_device_targets(html: &str) -> Vec<String> {
    let mut targets = parse_aggregate_package_links(html)
        .iter()
        .filter_map(|name| name.strip_prefix("rocm-sdk-device-"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    targets.sort();
    targets
}

pub(in crate::therock) fn validate_aggregate_index_layout(html: &str) -> Result<()> {
    let names = parse_aggregate_package_links(html);
    let missing = ["rocm", "torch", "torchvision", "torchaudio"]
        .into_iter()
        .filter(|package| !names.iter().any(|name| name == package))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "unknown canonical TheRock aggregate index layout: expected package links for rocm, torch, torchvision, and torchaudio, but {} not published",
            missing.join(", ")
        );
    }
    if !names
        .iter()
        .any(|name| name.starts_with("rocm-sdk-device-"))
    {
        bail!(
            "unknown canonical TheRock aggregate index layout: no `rocm-sdk-device-*` payload packages are published, so no GPU backend could be selected"
        );
    }
    Ok(())
}

/// Which device payload the canonical aggregate source must supply for this host.
///
/// The aggregate `rocm` distribution ships no GPU backend unless a `device-*`
/// extra asks for one, so this choice decides whether the installed runtime can
/// launch a kernel at all. Exactly one exact target is ever requested. The
/// blanket `device-all` extra pulls every published payload — measured at 24
/// wheels and 4451 MiB on an MI300X that needs one of them — and then leaves
/// `rocm_sdk` to guess a target family out of that pile; a family-bucket alias
/// such as `device-gfx120X-all` is not an extra the source declares at all.
///
/// When no exact target can be pinned the answer is [`Undetermined`] rather
/// than a fallback: a preview still renders and says so, and a real install
/// refuses instead of producing a runtime with no kernels.
///
/// [`Undetermined`]: AggregateDeviceTarget::Undetermined
#[derive(Debug, Clone, Eq, PartialEq)]
pub(in crate::therock) enum AggregateDeviceTarget {
    /// The exact detected GFX target, published by the canonical source.
    Exact(String),
    /// No exact target could be pinned, and why.
    Undetermined(String),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(in crate::therock) struct ResolvedAggregateWheelSource {
    pub(in crate::therock) index_url: String,
    pub(in crate::therock) layout: SourceLayout,
    pub(in crate::therock) device_target: AggregateDeviceTarget,
    pub(in crate::therock) published_device_targets: Vec<String>,
}

/// Stands in for a real target in a preview, so a plan that cannot be installed
/// yet reads as incomplete rather than as installable.
const UNDETERMINED_DEVICE_TARGET: &str = "<undetermined>";

impl AggregateDeviceTarget {
    pub(in crate::therock) fn resolve(
        detected: Option<&str>,
        family: &str,
        published: &[String],
    ) -> Self {
        let Some(detected) = detected else {
            return Self::Undetermined("no AMD GPU target was detected on this host".to_owned());
        };
        // KFD reports feature suffixes (`gfx90a:sramecc+:xnack-`); published
        // targets never carry them.
        let Some(target) = extract_first_gfx_token(detected) else {
            return Self::Undetermined(format!(
                "detected GPU target `{detected}` is not a recognizable GFX target"
            ));
        };
        match normalize_therock_family(&target) {
            Some(detected_family) if detected_family == family => {}
            Some(detected_family) => {
                return Self::Undetermined(format!(
                    "detected GPU target `{target}` belongs to family `{detected_family}`, not the resolved target family `{family}`"
                ));
            }
            None => {
                return Self::Undetermined(format!(
                    "detected GPU target `{target}` belongs to no recognized package family"
                ));
            }
        }
        if !published.iter().any(|candidate| candidate == &target) {
            return Self::Undetermined(format!(
                "the canonical source publishes no `device-{target}` payload for detected GPU target `{target}` (published targets: {})",
                published.join(", ")
            ));
        }
        Self::Exact(target)
    }

    pub(in crate::therock) fn as_str(&self) -> &str {
        match self {
            Self::Exact(target) => target,
            Self::Undetermined(_) => UNDETERMINED_DEVICE_TARGET,
        }
    }

    pub(in crate::therock) fn reason(&self) -> Option<&str> {
        match self {
            Self::Exact(_) => None,
            Self::Undetermined(reason) => Some(reason),
        }
    }
}

/// A PEP 440 version, less its `+local` label, as a value whose derived order is
/// PEP 440's: field order is comparison order, and each enum's variants are
/// declared lowest first.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::therock) struct PackageReleaseKey {
    epoch: u64,
    /// Trailing zeros dropped, so `2.11` and `2.11.0` compare equal.
    release: Vec<u64>,
    pre: PreRelease,
    post: Option<u64>,
    dev: DevRelease,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PreRelease {
    /// `X.devN` with no pre- or post-release sorts below every `X` pre-release.
    DevOnly,
    /// `a` = 0, `b` = 1, `rc` = 2.
    Pre(u8, u64),
    Final,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DevRelease {
    Dev(u64),
    Absent,
}

impl PackageReleaseKey {
    /// `uv` only picks a pre-release (including any `.devN`) when no final
    /// release is available; a post-release counts as final.
    pub(in crate::therock) fn is_final(&self) -> bool {
        self.pre == PreRelease::Final && self.dev == DevRelease::Absent
    }
}

/// Parses the normalized PEP 440 form a package index publishes
/// (`[N!]N(.N)*[{a|b|rc}N][.postN][.devN][+local]`), already percent-decoded
/// by [`parse_simple_index_versions`]. Anything else is `None`, and is skipped
/// rather than guessed at.
pub(in crate::therock) fn package_release_key(version: &str) -> Option<PackageReleaseKey> {
    fn number(rest: &mut &str) -> Option<u64> {
        let len = rest.chars().take_while(char::is_ascii_digit).count();
        let (digits, tail) = rest.split_at(len);
        *rest = tail;
        digits.parse().ok()
    }

    let public = version.split('+').next().unwrap_or_default();
    let (epoch, mut rest) = match public.split_once('!') {
        Some((epoch, rest)) => (epoch.parse().ok()?, rest),
        None => (0, public),
    };

    let mut release = vec![number(&mut rest)?];
    while let Some(tail) = rest.strip_prefix('.')
        && tail.starts_with(|ch: char| ch.is_ascii_digit())
    {
        rest = tail;
        release.push(number(&mut rest)?);
    }
    while release.len() > 1 && release.last() == Some(&0) {
        release.pop();
    }

    let mut pre = None;
    for (tag, kind) in [("rc", 2), ("a", 0), ("b", 1)] {
        if let Some(tail) = rest.strip_prefix(tag) {
            rest = tail;
            pre = Some((kind, number(&mut rest)?));
            break;
        }
    }
    let post = match rest.strip_prefix(".post") {
        Some(tail) => {
            rest = tail;
            Some(number(&mut rest)?)
        }
        None => None,
    };
    let dev = match rest.strip_prefix(".dev") {
        Some(tail) => {
            rest = tail;
            Some(number(&mut rest)?)
        }
        None => None,
    };
    if !rest.is_empty() {
        return None;
    }

    Some(PackageReleaseKey {
        epoch,
        release,
        pre: match (pre, post, dev) {
            (Some((kind, number)), ..) => PreRelease::Pre(kind, number),
            (None, None, Some(_)) => PreRelease::DevOnly,
            (None, ..) => PreRelease::Final,
        },
        post,
        dev: dev.map_or(DevRelease::Absent, DevRelease::Dev),
    })
}

pub(in crate::therock) fn package_rocm_suffix(version: &str) -> Option<String> {
    let decoded = decode_simple_index_version(version);
    let lower = decoded.to_ascii_lowercase();
    let marker = "+rocm";
    let start = lower.rfind(marker)? + marker.len();
    decoded.get(start..).map(str::to_owned)
}

fn decode_simple_index_version(version: &str) -> String {
    version.replace("%2B", "+").replace("%2b", "+")
}

pub(in crate::therock) fn load_simple_index_versions(
    paths: &AppPaths,
    index_url: &str,
    package_name: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
    download_timeout_secs: Option<u64>,
) -> Result<Vec<String>> {
    let url = format!("{}/{package_name}/", index_url.trim_end_matches('/'));
    let html = download_text_cached(
        paths,
        &format!("simple-index-{}-{}", slugify(index_url), package_name),
        &url,
        download_timeout_secs,
    )?
    .text;
    Ok(parse_simple_index_versions(
        &html,
        package_name,
        wheel_compatibility,
    ))
}

fn parse_simple_index_versions(
    html: &str,
    package_name: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
) -> Vec<String> {
    let marker = format!("{package_name}-");
    let mut versions = Vec::new();
    for line in html.lines() {
        let mut rest = line;
        while let Some(start) = rest.find(&marker) {
            let version_start = start + marker.len();
            let Some(candidate) = rest.get(version_start..) else {
                break;
            };
            if let Some((version, consumed)) =
                parse_simple_index_version_candidate(candidate, wheel_compatibility)
            {
                if let Some(version) = version {
                    versions.push(version);
                }
                rest = candidate.get(consumed..).unwrap_or_default();
            } else {
                break;
            }
        }
    }
    versions.sort_by(|left, right| compare_version_strings(left, right));
    versions.dedup();
    versions
}

fn parse_simple_index_version_candidate(
    candidate: &str,
    wheel_compatibility: Option<&WheelCompatibility>,
) -> Option<(Option<String>, usize)> {
    let tar_pos = candidate.find(".tar.gz");
    let wheel_pos = candidate.find(".whl");
    match (tar_pos, wheel_pos) {
        (Some(tar_pos), Some(wheel_pos)) if tar_pos < wheel_pos => {
            let version = decode_simple_index_version(candidate.get(..tar_pos)?);
            Some((Some(version), tar_pos + ".tar.gz".len()))
        }
        (Some(tar_pos), None) => {
            let version = decode_simple_index_version(candidate.get(..tar_pos)?);
            Some((Some(version), tar_pos + ".tar.gz".len()))
        }
        (_, Some(wheel_pos)) => {
            let wheel_stem = candidate.get(..wheel_pos)?;
            if let Some(wheel_compatibility) = wheel_compatibility
                && !wheel_stem_matches_compatibility(wheel_stem, wheel_compatibility)
            {
                return Some((None, wheel_pos + ".whl".len()));
            }
            let version = wheel_stem.split('-').next()?;
            Some((
                Some(decode_simple_index_version(version)),
                wheel_pos + ".whl".len(),
            ))
        }
        (None, None) => None,
    }
}

fn wheel_stem_matches_compatibility(wheel_stem: &str, compatibility: &WheelCompatibility) -> bool {
    let mut parts = wheel_stem.rsplitn(4, '-');
    let Some(platform_tag) = parts.next() else {
        return false;
    };
    let Some(abi_tag) = parts.next() else {
        return false;
    };
    let Some(python_tag) = parts.next() else {
        return false;
    };
    if parts.next().is_none() {
        return false;
    }

    wheel_python_tag_matches(python_tag, &compatibility.python_tag)
        && wheel_abi_tag_matches(abi_tag, &compatibility.python_tag)
        && wheel_platform_tag_matches(platform_tag, &compatibility.platform_tags)
}

fn wheel_python_tag_matches(wheel_tag: &str, python_tag: &str) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| tag == python_tag || tag == "py3")
}

fn wheel_abi_tag_matches(wheel_tag: &str, python_tag: &str) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| tag == python_tag || tag == "abi3" || tag == "none")
}

fn wheel_platform_tag_matches(wheel_tag: &str, platform_tags: &[String]) -> bool {
    wheel_tag
        .split('.')
        .any(|tag| platform_tags.iter().any(|platform| platform == tag))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative slice of what the canonical aggregate index publishes.
    /// Notably it has no `gfx943`: MI300 steppings other than `gfx942` have no
    /// payload of their own, which is exactly the case a remap used to hide.
    const PUBLISHED_DEVICE_TARGETS_HTML: &str = r#"<!DOCTYPE html><html><body>
<a href="rocm/">rocm</a><br/>
<a href="torch/">torch</a><br/>
<a href="torchvision/">torchvision</a><br/>
<a href="torchaudio/">torchaudio</a><br/>
<a href="rocm-sdk-core/">rocm-sdk-core</a><br/>
<a href="amd-torch-device-gfx942/">amd-torch-device-gfx942</a><br/>
<a href="rocm-sdk-device-gfx90a/">rocm-sdk-device-gfx90a</a><br/>
<a href="rocm-sdk-device-gfx942/">rocm-sdk-device-gfx942</a><br/>
<a href="rocm-sdk-device-gfx1151/">rocm-sdk-device-gfx1151</a><br/>
<a href="rocm-sdk-device-gfx1201/">rocm-sdk-device-gfx1201</a><br/>
</body></html>"#;

    fn published_device_targets() -> Vec<String> {
        parse_aggregate_device_targets(PUBLISHED_DEVICE_TARGETS_HTML)
    }

    #[test]
    fn published_device_targets_come_from_the_sdk_payload_packages_only() {
        // `amd-torch-device-*` and `rocm-sdk-core` share the page; only the
        // `rocm-sdk-device-*` names name a `device-<target>` extra of `rocm`.
        assert_eq!(
            published_device_targets(),
            vec![
                "gfx1151".to_owned(),
                "gfx1201".to_owned(),
                "gfx90a".to_owned(),
                "gfx942".to_owned(),
            ]
        );
    }

    #[test]
    fn device_extra_is_the_exact_detected_target_the_source_publishes() {
        let target = AggregateDeviceTarget::resolve(
            Some("gfx1201"),
            "gfx120X-all",
            &published_device_targets(),
        );

        assert_eq!(target, AggregateDeviceTarget::Exact("gfx1201".to_owned()));
        assert_eq!(target.as_str(), "gfx1201");
    }

    #[test]
    fn device_extra_drops_the_kfd_feature_suffix() {
        assert_eq!(
            AggregateDeviceTarget::resolve(
                Some("gfx90a:sramecc+:xnack-"),
                "gfx90a",
                &published_device_targets(),
            ),
            AggregateDeviceTarget::Exact("gfx90a".to_owned())
        );
    }

    #[test]
    fn unpublished_detected_target_is_undetermined_rather_than_remapped() {
        // gfx943 normalizes to the same family as gfx942, so a family-level
        // answer would silently install gfx942 kernels on a chip the source
        // never published a payload for.
        let target = AggregateDeviceTarget::resolve(
            Some("gfx943"),
            "gfx94X-dcgpu",
            &published_device_targets(),
        );

        assert_eq!(target.as_str(), "<undetermined>");
        assert!(
            target
                .reason()
                .is_some_and(|reason| reason.contains("no `device-gfx943` payload"))
        );
    }

    #[test]
    fn no_detected_gpu_yields_an_undetermined_target_not_a_blanket_payload() {
        let target =
            AggregateDeviceTarget::resolve(None, "gfx110X-all", &published_device_targets());

        assert_eq!(target.as_str(), "<undetermined>");
        assert!(
            target
                .reason()
                .is_some_and(|reason| reason.contains("no AMD GPU target was detected"))
        );
    }

    #[test]
    fn a_detected_target_from_another_family_is_undetermined() {
        let target = AggregateDeviceTarget::resolve(
            Some("gfx1151"),
            "gfx120X-all",
            &published_device_targets(),
        );

        assert!(
            target
                .reason()
                .is_some_and(|reason| reason.contains("not the resolved target family"))
        );
    }

    #[test]
    fn aggregate_layout_without_device_payloads_is_rejected() {
        let error = validate_aggregate_index_layout(
            r#"<a href="rocm/">rocm</a><a href="torch/">torch</a><a href="torchvision/">torchvision</a><a href="torchaudio/">torchaudio</a>"#,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("no `rocm-sdk-device-*` payload packages"));
    }

    #[test]
    fn unknown_aggregate_layout_is_rejected_clearly() {
        let error =
            validate_aggregate_index_layout("<html><a href=\"gfx120X-all/\">legacy</a></html>")
                .unwrap_err()
                .to_string();
        assert!(error.contains("unknown canonical TheRock aggregate index layout"));
        // `rocm-sdk-core/` must not satisfy the `rocm` requirement.
        let error = validate_aggregate_index_layout(
            r#"<a href="rocm-sdk-core/">rocm-sdk-core</a><a href="torch/">torch</a>"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("rocm, torchvision, torchaudio not published"));
    }

    /// Each PEP 440 suffix number compares as a number, and the suffixes rank
    /// `.devN` < `aN` < `bN` < `rcN` < final < `.postN`, with a `.devN` on any
    /// of them ranking just below it. Pairs are `(older, newer)`.
    #[test]
    fn torchaudio_pin_orders_versions_as_pep_440_does() {
        for (older, newer) in [
            ("2.12.0a9", "2.12.0a10"),
            ("2.12.0rc2", "2.12.0rc10"),
            ("2.11.0.post9", "2.11.0.post10"),
            ("2.12.0.dev1", "2.12.0.dev2"),
            ("2.12.0.dev1", "2.12.0a1"),
            ("2.12.0a1", "2.12.0b1"),
            ("2.12.0b1", "2.12.0rc1"),
            ("2.12.0rc1.dev3", "2.12.0rc1"),
            ("2.11.0.post1.dev0", "2.11.0.post1"),
            ("2.11.0", "2.11.0.post1.dev0"),
            ("2.11", "2.11.0.1"),
            ("9.0.0", "1!2.0.0"),
        ] {
            assert!(
                package_release_key(&format!("{older}+rocm10.0.0"))
                    < package_release_key(&format!("{newer}+rocm10.0.0")),
                "{older} must order below {newer}"
            );
        }
        assert_eq!(
            package_release_key("2.11+rocm10.0.0"),
            package_release_key("2.11.0+rocm10.0.0"),
            "trailing zero release segments are insignificant"
        );
    }

    #[test]
    fn simple_index_parser_strips_wheel_tags_decodes_plus_and_filters_platform() {
        let compatibility = WheelCompatibility {
            python_tag: "cp312".to_owned(),
            platform_tags: vec!["win_amd64".to_owned(), "any".to_owned()],
        };
        let html = r#"
            <a href="torch-2.10.0%2Brocm7.13.0a20260513-cp312-cp312-win_amd64.whl">torch-2.10.0%2Brocm7.13.0a20260513-cp312-cp312-win_amd64.whl</a>
            <a href="torch-2.11.0%2Brocm7.13.0a20260514-cp313-cp313-win_amd64.whl">torch-2.11.0%2Brocm7.13.0a20260514-cp313-cp313-win_amd64.whl</a>
            <a href="torch-2.12.0+rocm7.13.0a20260515-cp312-cp312-linux_x86_64.whl">torch-2.12.0+rocm7.13.0a20260515-cp312-cp312-linux_x86_64.whl</a>
        "#;

        assert_eq!(
            parse_simple_index_versions(html, "torch", Some(&compatibility)),
            vec!["2.10.0+rocm7.13.0a20260513".to_owned()]
        );
    }
}
