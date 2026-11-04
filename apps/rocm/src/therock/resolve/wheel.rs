// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Resolving the wheel stack a request installs: which `rocm`, `torch`,
//! `torchvision` and `torchaudio` versions are published together for this
//! interpreter and platform, and which device payload this host needs.
//!
//! `resolve_pip_runtime` (and `resolve_pip_runtime_with_timeout`, which takes
//! the metadata download timeout and an optional source layout) resolves the
//! family and the layout, reads the aggregate root, decides the device payload
//! (`AggregateDeviceTarget`) and returns a `PipRuntimeResolution`. Versions are
//! chosen in `resolve_pip_runtime_from_index`: on the canonical layout
//! `select_matching_pip_package_versions` takes the newest `rocm` whose
//! `torch`, `torchvision` and `torchaudio` listings agree; on the next layout
//! `torchaudio_pin` pins a torchaudio first (the hardware-verified release when
//! it is published) and `resolve_published_pip_package_versions` has
//! `uv pip compile` choose the rest. `wheel_runtime_composition` turns a
//! resolution into the exact package specs (`therock_pip_package_specs`) an
//! install gives `uv` and records in the manifest, and
//! `canonical_auto_select_needs_next_python` peeks at the channel's own `rocm`
//! listing so the interpreter can be picked before resolution.
//!
//! Reading what an index publishes sits beneath that: the package links of the
//! aggregate root and the exact device targets they name
//! (`parse_aggregate_device_targets`, `validate_aggregate_index_layout`), each
//! package's simple-index listing (`load_simple_index_versions`,
//! `parse_simple_index_versions`, with the `wheel_*_tag_matches` helpers
//! filtering by the interpreter's wheel tags), and the PEP 440 order of the
//! versions found there (`PackageReleaseKey`, `package_release_key`).
//!
//! Builds on `family`, `fetch`, `process`, `python`, `source` and `version`, and
//! on `runtime_registry` for `WheelRuntimeComposition`; `slugify`, which keys
//! the cached listing, stays in `therock.rs` with the other keys helpers.

use super::super::family::resolve_family;
use super::super::fetch::download_text_cached;
use super::super::process::wait_with_output_bounded;
use super::super::python::WheelCompatibility;
use super::super::slugify;
use super::super::source::{
    FamilyResolution, RuntimeVersionSelector, SourceLayout, THEROCK_NEXT_MIN_MAJOR, TheRockChannel,
    resolve_source, select_source_layout,
};
use super::super::version::{
    ParsedVersion, compare_version_strings, is_stable_runtime_version, parse_version,
};
use crate::runtime_registry::WheelRuntimeComposition;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, detect_host_gfx_target, ensure_uv_binary, extract_first_gfx_token,
    normalize_therock_family, runtime_is_windows, uv_command_env,
};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

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
fn parse_aggregate_device_targets(html: &str) -> Vec<String> {
    let mut targets = parse_aggregate_package_links(html)
        .iter()
        .filter_map(|name| name.strip_prefix("rocm-sdk-device-"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    targets.sort();
    targets
}

fn validate_aggregate_index_layout(html: &str) -> Result<()> {
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
struct ResolvedAggregateWheelSource {
    index_url: String,
    layout: SourceLayout,
    device_target: AggregateDeviceTarget,
    published_device_targets: Vec<String>,
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
struct PackageReleaseKey {
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
    fn is_final(&self) -> bool {
        self.pre == PreRelease::Final && self.dev == DevRelease::Absent
    }
}

/// Parses the normalized PEP 440 form a package index publishes
/// (`[N!]N(.N)*[{a|b|rc}N][.postN][.devN][+local]`), already percent-decoded
/// by [`parse_simple_index_versions`]. Anything else is `None`, and is skipped
/// rather than guessed at.
fn package_release_key(version: &str) -> Option<PackageReleaseKey> {
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

fn package_rocm_suffix(version: &str) -> Option<String> {
    let decoded = decode_simple_index_version(version);
    let lower = decoded.to_ascii_lowercase();
    let marker = "+rocm";
    let start = lower.rfind(marker)? + marker.len();
    decoded.get(start..).map(str::to_owned)
}

fn decode_simple_index_version(version: &str) -> String {
    version.replace("%2B", "+").replace("%2b", "+")
}

fn load_simple_index_versions(
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

#[derive(Debug, Clone)]
pub(in crate::therock) struct PipRuntimeResolution {
    pub(in crate::therock) family: String,
    pub(in crate::therock) family_source: String,
    pub(in crate::therock) index_url: String,
    /// The layout `index_url` belongs to, carried forward so the composition
    /// this resolution produces records the stream that produced it.
    pub(in crate::therock) layout: SourceLayout,
    pub(in crate::therock) latest_version: String,
    /// Newest `rocm` version offered by the repository for this channel,
    /// regardless of whether it has a matching PyTorch wheel stack. When this is
    /// newer than `latest_version`, the repo's newest release could not be
    /// installed (no wheels) and we warn about it. `None` when a specific version
    /// was requested (the "latest" concept does not apply).
    pub(in crate::therock) newest_repo_version: Option<String>,
    /// The exact requirement lines handed to `uv pip compile` to choose this
    /// version set. This is the value that was sent, not a re-derivation of it,
    /// so the preview can show what resolution actually asked for and an
    /// acceptance test can hold it to the same extras the install plan names.
    /// `None` on the canonical layout, which picks versions by scraping each
    /// package's simple index rather than by resolving a requirement set.
    pub(in crate::therock) version_resolution_specs: Option<Vec<String>>,
    /// How the torchaudio in `version_resolution_specs` was chosen, kept so the
    /// plan can say whether it is the hardware-verified build. `None` on the
    /// canonical layout, which pins no torchaudio before resolving.
    pub(in crate::therock) torchaudio_pin: Option<TorchaudioPin>,
    pub(in crate::therock) package_versions: TheRockPipPackageVersions,
    /// The device payload the resolved source must supply for this host,
    /// decided against the targets that source actually publishes.
    pub(in crate::therock) device_target: AggregateDeviceTarget,
    /// Exact device payloads advertised by the resolved aggregate source.
    pub(in crate::therock) published_device_targets: Vec<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(in crate::therock) struct TheRockPipPackageVersions {
    rocm: String,
    pub(in crate::therock) torch: String,
    torchvision: String,
    torchaudio: String,
    pub(in crate::therock) compatibility_key: String,
}

/// The `rocm` wheel extras a given install asks for, excluding the device payload.
///
/// `devel` adds the compiler, headers, and static libraries — roughly doubling
/// the download — and is only needed to *build* GPU code. Running models needs
/// `libraries` alone, so the toolchain is installed only when asked for.
///
/// Two callers, and they are the two that must agree:
///
/// - [`therock_pip_package_specs`], which produces the specs `uv` installs and
///   the specs recorded in the manifest, and
/// - [`published_pip_requirements`], the requirement set `uv pip compile` is
///   asked to resolve versions against on the ROCm 10 (`next`) layout.
///
/// Version resolution and install composition are separate code paths —
/// resolution runs first and decides which versions exist, composition runs
/// second and decides what is installed — so a flag threaded correctly through
/// one says nothing about the other. Both go through this helper for that
/// reason. Everything else that names the toolchain (the install progress line,
/// `runtimes list`, `InstalledRuntimeManifest::includes_devel`) reads it back
/// out of the composed specs rather than re-deciding it.
const fn therock_sdk_extras(include_devel: bool) -> &'static str {
    if include_devel {
        "libraries,devel"
    } else {
        "libraries"
    }
}

/// The requirement lines handed to `uv pip compile` to pick a mutually
/// installable version set on the ROCm 10 (`next`) layout.
///
/// torch and torchvision are unversioned on purpose: choosing those versions is
/// what the resolve is for, and their published metadata depends on the exact
/// `rocm` build, so the `rocm` pin carries them to the matching `+rocm<build>`.
/// torchaudio's metadata names no `rocm` build, so left unversioned it resolves
/// to the newest build on the index — which is a different ROCm build for every
/// pin but the newest — and the caller pins it to `torchaudio_version`, already
/// chosen from this build's published torchaudio wheels.
///
/// The `rocm` extras must match the ones [`therock_pip_package_specs`] will
/// install — resolving against `devel` for someone who did not ask for it
/// constrains the chosen versions by a toolchain they declined, and can fail the
/// whole install with a toolchain-resolution error on a default
/// `rocm install sdk`.
fn published_pip_requirements(
    rocm_version: &str,
    torchaudio_version: &str,
    device_target: &str,
    include_devel: bool,
) -> Vec<String> {
    let device_extra = format!("device-{device_target}");
    vec![
        format!(
            "rocm[{},{device_extra}]=={rocm_version}",
            therock_sdk_extras(include_devel)
        ),
        format!("torch[{device_extra}]"),
        format!("torchvision[{device_extra}]"),
        format!("torchaudio=={torchaudio_version}"),
    ]
}

/// Package specs for a wheel SDK install.
///
/// The `device-<target>` extra is separate from [`therock_sdk_extras`] and
/// always present: it selects which GPU payload the wheels carry, not whether
/// the toolchain comes with them.
fn therock_pip_package_specs(
    package_versions: &TheRockPipPackageVersions,
    device_target: &str,
    include_devel: bool,
) -> Vec<String> {
    let device_extra = format!("device-{device_target}");
    let rocm_extras = format!("{},{device_extra}", therock_sdk_extras(include_devel));
    vec![
        format!("rocm[{rocm_extras}]=={}", package_versions.rocm),
        format!("torch[{device_extra}]=={}", package_versions.torch),
        format!(
            "torchvision[{device_extra}]=={}",
            package_versions.torchvision
        ),
        format!("torchaudio=={}", package_versions.torchaudio),
    ]
}

/// The exact install intent for `resolution` and its source-validated device target.
///
/// Kept beside [`therock_pip_package_specs`] so the specs that identify a
/// runtime are, by construction, the specs that get installed.
pub(in crate::therock) fn wheel_runtime_composition(
    resolution: &PipRuntimeResolution,
    device_target: &AggregateDeviceTarget,
    include_devel: bool,
) -> WheelRuntimeComposition {
    WheelRuntimeComposition {
        source_layout_generation: resolution.layout.generation().to_owned(),
        package_specs: therock_pip_package_specs(
            &resolution.package_versions,
            device_target.as_str(),
            include_devel,
        ),
        rocm_sdk_target: matches!(device_target, AggregateDeviceTarget::Exact(_))
            .then(|| device_target.as_str().to_owned()),
    }
}

pub(in crate::therock) fn resolve_pip_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
    include_devel: bool,
) -> Result<PipRuntimeResolution> {
    resolve_pip_runtime_with_timeout(
        paths,
        channel,
        family_override,
        wheel_compatibility,
        version_selector,
        layout_override,
        include_devel,
        None,
    )
}

/// `layout_override` is how an update keeps a runtime on the stream it was
/// installed from; a fresh install passes `None` and lets the request decide.
#[allow(clippy::too_many_arguments)]
pub(in crate::therock) fn resolve_pip_runtime_with_timeout(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_override: Option<&str>,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
    include_devel: bool,
    download_timeout_secs: Option<u64>,
) -> Result<PipRuntimeResolution> {
    let family_resolution = resolve_family(paths, family_override)?;
    let layout = match layout_override {
        Some(layout) => layout,
        None => select_source_layout(channel, &family_resolution, version_selector)?,
    };
    let source = resolve_source(channel, layout);
    let root_url = format!("{}/", source.wheel_index.trim_end_matches('/'));
    // Distinct cache keys per layout: the two streams answer the same request
    // with different package sets, so one must never serve the other's listing.
    let root_cache_key = match layout {
        SourceLayout::Canonical => format!("canonical-wheel-root-{}", channel.as_str()),
        SourceLayout::Next => format!("next-wheel-root-{}", channel.as_str()),
    };
    let root_html =
        download_text_cached(paths, &root_cache_key, &root_url, download_timeout_secs)?.text;
    validate_aggregate_index_layout(&root_html).with_context(|| {
        format!(
            "failed to resolve TheRock {} wheel runtime from canonical source {}",
            channel.as_str(),
            source.wheel_index
        )
    })?;
    let published_device_targets = parse_aggregate_device_targets(&root_html);
    // The canonical stream validates against what this host reports, exactly as
    // before. The next layout is reachable only through an explicit pin that
    // already carried a validated exact arch, and that arch — not a second probe
    // of whatever card happens to be plugged in — is what the pin asked to
    // install for.
    let detected_target = match layout {
        SourceLayout::Canonical => detect_host_gfx_target(),
        SourceLayout::Next => family_resolution.raw_arch.clone(),
    };
    let source = ResolvedAggregateWheelSource {
        index_url: source.wheel_index,
        layout,
        device_target: AggregateDeviceTarget::resolve(
            detected_target.as_deref(),
            &family_resolution.family,
            &published_device_targets,
        ),
        published_device_targets,
    };
    resolve_pip_runtime_from_index(
        paths,
        channel,
        &family_resolution,
        &source,
        wheel_compatibility,
        version_selector,
        include_devel,
        download_timeout_secs,
    )
    .with_context(|| {
        format!(
            "failed to resolve TheRock {} wheel runtime from canonical source {}\n\n{}",
            channel.as_str(),
            source.index_url,
            canonical_wheel_resolution_hint(channel)
        )
    })
}

/// `include_devel` reaches this far because version resolution, not only install
/// composition, has to ask for the extras the caller actually requested — see
/// [`published_pip_requirements`].
#[allow(clippy::too_many_arguments)]
fn resolve_pip_runtime_from_index(
    paths: &AppPaths,
    channel: TheRockChannel,
    family_resolution: &FamilyResolution,
    source: &ResolvedAggregateWheelSource,
    wheel_compatibility: &WheelCompatibility,
    version_selector: Option<&RuntimeVersionSelector>,
    include_devel: bool,
    download_timeout_secs: Option<u64>,
) -> Result<PipRuntimeResolution> {
    let index_url = source.index_url.as_str();
    let mut version_resolution_specs = None;
    let mut torchaudio_selection = None;
    let rocm_versions =
        load_simple_index_versions(paths, index_url, "rocm", None, download_timeout_secs)?;
    if matches!(channel, TheRockChannel::Release)
        && version_selector.is_none()
        && !rocm_versions
            .iter()
            .any(|version| is_stable_runtime_version(version))
    {
        bail!(
            "release channel only installs stable TheRock wheel versions, but no stable `rocm` package versions were found in {index_url}; try `rocm install sdk --channel release --format tarball` for stable release artifacts, or use `--channel nightly --format wheel` for preview builds"
        );
    }
    let package_versions = if matches!(source.layout, SourceLayout::Next) {
        let device_target = match &source.device_target {
            AggregateDeviceTarget::Exact(target) => target.as_str(),
            AggregateDeviceTarget::Undetermined(reason) => {
                bail!(
                    "cannot resolve ROCm X package metadata without an exact device target: {reason}"
                )
            }
        };
        let rocm_version = select_rocm_version(channel, &rocm_versions, version_selector)
            .with_context(|| {
                let requested = version_selector.map_or_else(
                    || "latest compatible version".to_owned(),
                    RuntimeVersionSelector::describe,
                );
                format!("no TheRock rocm package was found for {requested} in {index_url}")
            })?;
        // Chosen here, not by `uv`: see `published_pip_requirements`.
        let torchaudio_versions = load_simple_index_versions(
            paths,
            index_url,
            "torchaudio",
            Some(wheel_compatibility),
            download_timeout_secs,
        )?;
        let pin = torchaudio_pin(
            &torchaudio_versions,
            &rocm_version,
            rocm_engine_vllm::verified_torchaudio_release(&rocm_version),
            index_url,
        )?;
        // The canonical branch below budgets one `download_timeout_secs` per
        // package (rocm, then torch, torchvision, torchaudio: 4 sequential HTTP
        // fetches). `uv pip compile` still fetches metadata for all four
        // packages in one subprocess call — the torchaudio listing above only
        // chose its version — so it needs a comparable aggregate budget, not
        // the single-fetch one: reusing the bare per-fetch value here would make
        // a startup check that budgets 2s per fetch reliably time out a call
        // doing 4 fetches' worth of work. That listing is one more fetch on top,
        // with its own single-fetch budget.
        let requirements =
            published_pip_requirements(&rocm_version, &pin.version, device_target, include_devel);
        let versions = resolve_published_pip_package_versions(
            paths,
            index_url,
            &requirements,
            wheel_compatibility,
            download_timeout_secs.map(|secs| secs.saturating_mul(4)),
        )?;
        version_resolution_specs = Some(requirements);
        torchaudio_selection = Some(pin);
        versions
    } else {
        let torch_versions = load_simple_index_versions(
            paths,
            index_url,
            "torch",
            Some(wheel_compatibility),
            download_timeout_secs,
        )?;
        let torchvision_versions = load_simple_index_versions(
            paths,
            index_url,
            "torchvision",
            Some(wheel_compatibility),
            download_timeout_secs,
        )?;
        let torchaudio_versions = load_simple_index_versions(
            paths,
            index_url,
            "torchaudio",
            Some(wheel_compatibility),
            download_timeout_secs,
        )?;
        select_matching_pip_package_versions(
            channel,
            &rocm_versions,
            &torch_versions,
            &torchvision_versions,
            &torchaudio_versions,
            version_selector,
        )
        .with_context(|| {
            let requested = version_selector.map_or_else(
                || "latest compatible version".to_owned(),
                RuntimeVersionSelector::describe,
            );
            format!(
                "no mutually compatible TheRock rocm, torch, torchvision, and torchaudio versions were found for {requested} in {index_url}"
            )
        })?
    };
    let latest_version = package_versions.rocm.clone();
    // The repo's newest version for this channel, ignoring wheel availability.
    // Only meaningful when we auto-selected "latest" (no explicit request), so a
    // caller can warn when that newest version has no installable wheels.
    let newest_repo_version = if version_selector.is_none() {
        channel_rocm_candidates(&rocm_versions, channel)
            .into_iter()
            .last()
    } else {
        None
    };
    Ok(PipRuntimeResolution {
        family: family_resolution.family.clone(),
        family_source: family_resolution.source.clone(),
        index_url: index_url.to_owned(),
        layout: source.layout,
        latest_version,
        newest_repo_version,
        version_resolution_specs,
        torchaudio_pin: torchaudio_selection,
        package_versions,
        device_target: source.device_target.clone(),
        published_device_targets: source.published_device_targets.clone(),
    })
}

pub(in crate::therock) fn select_rocm_version(
    channel: TheRockChannel,
    rocm_versions: &[String],
    version_selector: Option<&RuntimeVersionSelector>,
) -> Option<String> {
    let mut candidates = if version_selector.is_some() {
        rocm_versions.to_vec()
    } else {
        channel_rocm_candidates(rocm_versions, channel)
    };
    if let Some(selector) = version_selector {
        candidates.retain(|version| selector.matches_version(version));
    }
    candidates.sort_by(|left, right| compare_version_strings(left, right));
    candidates.pop()
}

fn uv_python_version(compatibility: &WheelCompatibility) -> Result<String> {
    let digits = compatibility
        .python_tag
        .strip_prefix("cp")
        .context("managed Python reported an unsupported wheel tag")?;
    if digits.len() < 2 || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        bail!(
            "managed Python reported unsupported wheel tag `{}`",
            compatibility.python_tag
        );
    }
    Ok(format!("{}.{}", &digits[..1], &digits[1..]))
}

fn uv_python_platform(compatibility: &WheelCompatibility) -> Result<&'static str> {
    if compatibility
        .platform_tags
        .iter()
        .any(|tag| tag == "win_amd64")
    {
        Ok("x86_64-pc-windows-msvc")
    } else if compatibility
        .platform_tags
        .iter()
        .any(|tag| tag == "linux_x86_64")
    {
        Ok("x86_64-unknown-linux-gnu")
    } else if compatibility
        .platform_tags
        .iter()
        .any(|tag| tag == "linux_aarch64")
    {
        Ok("aarch64-unknown-linux-gnu")
    } else {
        bail!(
            "managed Python reported unsupported platform wheel tags: {}",
            compatibility.platform_tags.join(",")
        )
    }
}

fn parse_uv_compiled_package_versions(output: &str) -> Result<TheRockPipPackageVersions> {
    let mut versions = std::collections::HashMap::new();
    for line in output.lines().map(str::trim) {
        let Some((name, version)) = line.split_once("==") else {
            continue;
        };
        versions.insert(name.to_ascii_lowercase(), version.to_owned());
    }
    let required = |name: &str| {
        versions
            .get(name)
            .cloned()
            .with_context(|| format!("uv metadata resolution did not pin `{name}`"))
    };
    let rocm = required("rocm")?;
    let torch = required("torch")?;
    let torchvision = required("torchvision")?;
    let torchaudio = required("torchaudio")?;
    for (name, version) in [
        ("torch", torch.as_str()),
        ("torchvision", torchvision.as_str()),
        ("torchaudio", torchaudio.as_str()),
    ] {
        if package_rocm_suffix(version).as_deref() != Some(rocm.as_str()) {
            bail!(
                "published package metadata selected {name} {version}, which does not share ROCm build {rocm}"
            );
        }
    }
    Ok(TheRockPipPackageVersions {
        compatibility_key: rocm.clone(),
        rocm,
        torch,
        torchvision,
        torchaudio,
    })
}

/// `requirements` is passed in rather than composed here so that the lines sent
/// to `uv` and the lines the resolution reports back (and the preview prints)
/// are one value, not two that can disagree.
fn resolve_published_pip_package_versions(
    paths: &AppPaths,
    index_url: &str,
    requirements: &[String],
    compatibility: &WheelCompatibility,
    download_timeout_secs: Option<u64>,
) -> Result<TheRockPipPackageVersions> {
    let uv =
        ensure_uv_binary(paths).context("failed to acquire uv for ROCm X metadata resolution")?;
    let python_version = uv_python_version(compatibility)?;
    let python_platform = uv_python_platform(compatibility)?;
    let requirements = format!("{}\n", requirements.join("\n"));
    let mut child = Command::new(&uv)
        .args([
            "pip",
            "compile",
            "-",
            "--index-url",
            index_url,
            "--python-version",
            &python_version,
            "--python-platform",
            python_platform,
            "--no-header",
            "--no-annotate",
        ])
        .envs(uv_command_env(paths))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to launch {} for ROCm X metadata resolution",
                uv.display()
            )
        })?;
    // ponytail: written before `wait_with_output_bounded` spawns its stdout/
    // stderr reader threads, so a child that fills its stdout pipe before this
    // write returns would deadlock outside the timeout below. `requirements`
    // is a handful of short lines today; move the write onto its own thread
    // (or start the readers first) if it ever grows enough to matter.
    child
        .stdin
        .take()
        .context("uv metadata resolver stdin was unavailable")?
        .write_all(requirements.as_bytes())
        .context("failed to send ROCm X requirements to uv")?;
    let timeout = download_timeout_secs.map(Duration::from_secs);
    let output = wait_with_output_bounded(child, timeout)
        .context("failed to wait for ROCm X metadata resolution")?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        bail!("published ROCm X package metadata is not jointly installable: {detail}");
    }
    parse_uv_compiled_package_versions(&String::from_utf8_lossy(&output.stdout))
}

fn select_matching_pip_package_versions(
    channel: TheRockChannel,
    rocm_versions: &[String],
    torch_versions: &[String],
    torchvision_versions: &[String],
    torchaudio_versions: &[String],
    version_selector: Option<&RuntimeVersionSelector>,
) -> Option<TheRockPipPackageVersions> {
    let mut rocm_candidates = if version_selector.is_some() {
        rocm_versions.to_vec()
    } else {
        channel_rocm_candidates(rocm_versions, channel)
    };
    if let Some(selector) = version_selector {
        rocm_candidates.retain(|version| selector.matches_version(version));
    }
    rocm_candidates.sort_by(|left, right| compare_version_strings(left, right));

    for rocm_version in rocm_candidates.into_iter().rev() {
        let mut torch_candidates = package_versions_matching_rocm(torch_versions, &rocm_version);
        torch_candidates.sort_by(|left, right| compare_version_strings(left, right));

        for torch_version in torch_candidates.into_iter().rev() {
            let Some(torch_base) = parse_package_base_version(&torch_version) else {
                continue;
            };
            let torchaudio_version =
                select_latest_stack_package(torchaudio_versions, &rocm_version, |base| {
                    pytorch_audio_matches_torch(&torch_base, base)
                });
            let torchvision_version =
                select_latest_stack_package(torchvision_versions, &rocm_version, |base| {
                    pytorch_vision_matches_torch(&torch_base, base)
                });
            if let (Some(torchaudio), Some(torchvision)) = (torchaudio_version, torchvision_version)
            {
                return Some(TheRockPipPackageVersions {
                    compatibility_key: rocm_version.clone(),
                    rocm: rocm_version,
                    torch: torch_version,
                    torchvision,
                    torchaudio,
                });
            }
        }
    }

    None
}

fn channel_rocm_candidates(versions: &[String], channel: TheRockChannel) -> Vec<String> {
    let mut all = versions.to_vec();
    all.sort_by(|left, right| compare_version_strings(left, right));
    if matches!(channel, TheRockChannel::Release) {
        return all
            .iter()
            .filter(|version| is_stable_runtime_version(version))
            .cloned()
            .collect::<Vec<_>>();
    }
    all
}

fn select_latest_stack_package(
    versions: &[String],
    rocm_version: &str,
    matches_stack: impl Fn(&ParsedVersion) -> bool,
) -> Option<String> {
    let mut candidates = package_versions_matching_rocm(versions, rocm_version)
        .into_iter()
        .filter(|version| {
            parse_package_base_version(version).is_some_and(|base| matches_stack(&base))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| compare_version_strings(left, right));
    candidates.pop()
}

/// The torchaudio build pinned for a ROCm 10 install, and why it was chosen.
/// The plan line and install-log note are both [`TorchaudioPin::describe`], so
/// what is printed is derived from the decision rather than restated beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::therock) struct TorchaudioPin {
    /// The published version pinned, `+rocm…` label included.
    version: String,
    /// The ROCm build `version` was chosen for.
    rocm_version: String,
    basis: TorchaudioPinBasis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TorchaudioPinBasis {
    /// `version` is the hardware-verified release for this ROCm line.
    Verified,
    /// A verified release is recorded for this ROCm line, but the index
    /// publishes no build of it labelled for this exact ROCm build (`+rocm`
    /// label) for this Python and platform, so `version` is the newest build
    /// instead.
    VerifiedNotPublished {
        release: &'static str,
        /// A build of `release` this Python and platform can install, labelled
        /// for another ROCm build, when one is published. The verified release
        /// is matched by ROCm line, so a `10.1.1` request can find it published
        /// only as `+rocm10.1.0`; the reason must then name the ROCm build, not
        /// the Python or platform.
        other_build: Option<String>,
    },
    /// No verified release is recorded for this ROCm line, so `version` is the
    /// newest build.
    NoVerifiedRelease,
}

impl TorchaudioPin {
    const fn is_verified(&self) -> bool {
        matches!(self.basis, TorchaudioPinBasis::Verified)
    }

    pub(in crate::therock) fn describe(&self) -> String {
        let Self {
            version,
            rocm_version,
            basis,
        } = self;
        match basis {
            TorchaudioPinBasis::Verified => format!(
                "torchaudio {version}, the build verified on hardware for ROCm {rocm_version}"
            ),
            TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: None,
            } => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because the verified release, {release}, is not published for this Python and platform"
            ),
            TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: Some(other_build),
            } => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because the verified release, {release}, is published for this Python and platform only for another ROCm build ({other_build}), not for ROCm {rocm_version}"
            ),
            TorchaudioPinBasis::NoVerifiedRelease => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because no torchaudio release is recorded as verified for this ROCm line"
            ),
        }
    }
}

/// The install-log line for a torchaudio pin that is not hardware-verified, or
/// `None` for one that is.
///
/// An unverified torchaudio is the one stack choice ROCm CLI makes on its own
/// judgement, so a real install says so in the install log, not only in the
/// plan's `torchaudio_selection:` line. A verified pin needs no such warning.
/// The text is [`TorchaudioPin::describe`], so it cannot disagree with the pin.
pub(in crate::therock) fn unverified_torchaudio_note(pin: &TorchaudioPin) -> Option<String> {
    (!pin.is_verified()).then(|| format!("Note: {}", pin.describe()))
}

/// The torchaudio build to pin for `rocm_version`, from `published` (this
/// Python and platform's builds on the index).
///
/// `verified_release` comes first: it is the release confirmed on hardware for
/// this ROCm line ([`rocm_engine_vllm::verified_torchaudio_release`], read from
/// the vLLM ROCm 10.x discover-build table so the SDK and its vLLM install pin
/// the same build). Recency alone is not a safe choice among builds sharing a
/// `+rocm<version>` label: `whl-next` publishes `2.11.0`, `2.11.0.2` and
/// `2.11.0.3` all as `+rocm10.1.0`, and `2.11.0` is a CPU-only rebuild with no
/// HIP kernels, which nothing in the version or label distinguishes. Only when
/// no verified release is recorded, or it is not published here, does this fall
/// back to [`latest_build_for_rocm`], and the returned basis says so.
///
/// Refuses when no build for `rocm_version` is published at all, or when none of
/// the published builds' versions can be read — the second is not the first,
/// and reporting it as "none published" would send the user to the wrong place.
fn torchaudio_pin(
    published: &[String],
    rocm_version: &str,
    verified_release: Option<&'static str>,
    index_url: &str,
) -> Result<TorchaudioPin> {
    let built_for_rocm = package_versions_matching_rocm(published, rocm_version);
    if built_for_rocm.is_empty() {
        bail!(
            "no torchaudio built for ROCm {rocm_version} is published for this Python and platform in {index_url}"
        );
    }
    let pin = |version, basis| TorchaudioPin {
        version,
        rocm_version: rocm_version.to_owned(),
        basis,
    };
    let basis = match verified_release {
        Some(release) => match published_build_of(&built_for_rocm, release) {
            Some(version) => return Ok(pin(version, TorchaudioPinBasis::Verified)),
            // Not among `built_for_rocm`, so a labelled build of it in the
            // whole listing is one for another ROCm build.
            None => TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: published_build_of(
                    &published
                        .iter()
                        .filter(|version| package_rocm_suffix(version).is_some())
                        .cloned()
                        .collect::<Vec<_>>(),
                    release,
                ),
            },
        },
        None => TorchaudioPinBasis::NoVerifiedRelease,
    };
    let newest = latest_build_for_rocm(published, rocm_version).with_context(|| {
        format!(
            "torchaudio builds for ROCm {rocm_version} are published in {index_url}, but none has a PEP 440 version ROCm CLI can order: {}",
            built_for_rocm.join(", ")
        )
    })?;
    Ok(pin(newest, basis))
}

/// The build in `versions` whose public version is `release` under PEP 440
/// (local label ignored, trailing zeros insignificant), as `uv` would match
/// `torchaudio=={release}`. Not a prefix match: `2.11.0` must not find
/// `2.11.0.2`.
fn published_build_of(versions: &[String], release: &str) -> Option<String> {
    let wanted = package_release_key(release)?;
    versions
        .iter()
        .find(|version| package_release_key(version).as_ref() == Some(&wanted))
        .cloned()
}

/// The newest of `versions` built for exactly `rocm_version`, chosen the way
/// `uv` would choose among them: in PEP 440 order, with final releases ahead of
/// any pre-release.
///
/// The fallback for [`torchaudio_pin`] only, never the first choice: newest is
/// not a safe discriminator among builds sharing a `+rocm<version>` label (see
/// there for the CPU-only `2.11.0+rocm10.1.0` precedent, and the vLLM engine's
/// ROCm 10.x discover-build table, which pins `2.11.0.2` exactly for the same
/// reason). Nothing on the index pairs a torchaudio build with a torch build,
/// so this assumes the newest torchaudio of a ROCm build is the one built
/// against that build's torch — the same choice `uv` made unaided when the
/// pinned build was also the newest one on the index.
///
/// Not [`select_latest_stack_package`]: [`parse_version`] reads only
/// `major.minor.patch` and rejects the rest, and torchaudio publishes rebuilds
/// as extra release segments (`2.11.0.2`, `2.11.0.3`), so that would drop every
/// rebuild and settle on the original `2.11.0` build.
fn latest_build_for_rocm(versions: &[String], rocm_version: &str) -> Option<String> {
    package_versions_matching_rocm(versions, rocm_version)
        .into_iter()
        .filter_map(|version| package_release_key(&version).map(|key| (key, version)))
        .max_by(|(left, _), (right, _)| (left.is_final(), left).cmp(&(right.is_final(), right)))
        .map(|(_, version)| version)
}

fn package_versions_matching_rocm(versions: &[String], rocm_version: &str) -> Vec<String> {
    versions
        .iter()
        .filter(|version| package_rocm_suffix(version).as_deref() == Some(rocm_version))
        .cloned()
        .collect()
}

fn pytorch_audio_matches_torch(torch_base: &ParsedVersion, audio_base: &ParsedVersion) -> bool {
    audio_base.major == torch_base.major
        && audio_base.minor == torch_base.minor
        && audio_base.stage == torch_base.stage
}

fn pytorch_vision_matches_torch(torch_base: &ParsedVersion, vision_base: &ParsedVersion) -> bool {
    let Some(expected_minor) = torch_base.minor.checked_add(15) else {
        return false;
    };
    vision_base.major == 0
        && vision_base.minor == expected_minor
        && vision_base.stage == torch_base.stage
}

fn parse_package_base_version(version: &str) -> Option<ParsedVersion> {
    parse_version(version.split('+').next().unwrap_or(version))
}

#[cfg(test)]
fn select_latest_version(versions: &[String], channel: TheRockChannel) -> Option<String> {
    let mut stable = Vec::new();
    let mut all = versions.to_vec();
    all.sort_by(|left, right| compare_version_strings(left, right));
    for version in versions {
        if is_stable_runtime_version(version) {
            stable.push(version.clone());
        }
    }
    stable.sort_by(|left, right| compare_version_strings(left, right));
    match channel {
        TheRockChannel::Release => stable.pop(),
        TheRockChannel::Nightly => all.pop(),
    }
}

/// Whether a `Canonical`-layout install (auto-select, or a nightly
/// build-date/prerelease pin) will actually resolve a ROCm build at major >=
/// [`THEROCK_NEXT_MIN_MAJOR`], regardless of [`VersionStage`](super::super::version::VersionStage).
///
/// `requested_source_layout` only answers from the shape of the request: an
/// explicit *stable* pin at ROCm >= `THEROCK_NEXT_MIN_MAJOR` routes to `Next`
/// and gets `cp314` that way, but the canonical/nightly feed can resolve a
/// ROCm >= `THEROCK_NEXT_MIN_MAJOR` build (stable or prerelease) on its own,
/// with no `Next` routing involved — and the `cp314` requirement is a property
/// of the build itself, not of which index served it. Fetching the channel's
/// own interpreter-agnostic `rocm` listing ahead of interpreter selection is
/// the only way to know; `resolve_pip_runtime` re-fetches the same URL for the
/// real resolution afterwards and shares this call's cache entry, so this
/// costs no extra network round trip.
pub(in crate::therock) fn canonical_auto_select_needs_next_python(
    paths: &AppPaths,
    channel: TheRockChannel,
    version_selector: Option<&RuntimeVersionSelector>,
) -> Result<bool> {
    let source = resolve_source(channel, SourceLayout::Canonical);
    let rocm_versions = load_simple_index_versions(paths, &source.wheel_index, "rocm", None, None)?;
    Ok(highest_candidate_needs_next_python(
        &rocm_versions,
        channel,
        version_selector,
    ))
}

/// Pure decision core of [`canonical_auto_select_needs_next_python`], split out
/// so it can be tested without a network fetch.
fn highest_candidate_needs_next_python(
    rocm_versions: &[String],
    channel: TheRockChannel,
    version_selector: Option<&RuntimeVersionSelector>,
) -> bool {
    let mut candidates = channel_rocm_candidates(rocm_versions, channel);
    if let Some(selector) = version_selector {
        candidates.retain(|version| selector.matches_version(version));
    }
    candidates
        .last()
        .and_then(|version| parse_version(version))
        .is_some_and(|parsed| parsed.major >= THEROCK_NEXT_MIN_MAJOR)
}

/// Recovery guidance appended to family/index resolution failures so a clean
/// first run can recover without the user having to guess a `--family`.
///
/// `source` is the [`FamilyResolution::source`] that produced `family`:
/// auto-detected (`host`, `managed-runtime`) versus user-supplied (`manifest`
/// from `--family`, `env` from `ROCM_CLI_THEROCK_FAMILY`). The wording differs
/// so an auto-detected miss points the user at `--family`, while a user-supplied
/// miss confirms the family they already named. Both point at the other channel
/// and, where valid for the platform, the other install format.
fn canonical_wheel_resolution_hint(channel: TheRockChannel) -> String {
    let other_channel = match channel {
        TheRockChannel::Release => "nightly",
        TheRockChannel::Nightly => "release",
    };
    let mut hint = format!(
        "No complete compatible package stack was found in the canonical {} aggregate stream. Try `--channel {other_channel}`",
        channel.as_str()
    );
    if !runtime_is_windows() {
        hint.push_str(" or `--format tarball`");
    }
    hint.push('.');
    hint
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::test_runtime_manifest;
    use super::*;
    use crate::runtime_registry::{InstalledRuntimeManifest, wheel_composition_includes_devel};

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

    #[test]
    fn release_channel_prefers_stable_versions() {
        let versions = vec![
            "7.11.0".to_owned(),
            "7.12.0".to_owned(),
            "7.13.0a20260326".to_owned(),
        ];
        assert_eq!(
            select_latest_version(&versions, TheRockChannel::Release),
            Some("7.12.0".to_owned())
        );
    }

    #[test]
    fn release_channel_rejects_prerelease_only_versions() {
        let versions = vec!["7.13.0a20260326".to_owned(), "7.14.0rc1".to_owned()];
        assert_eq!(
            select_latest_version(&versions, TheRockChannel::Release),
            None
        );
    }

    #[test]
    fn nightly_accepts_future_prerelease_major_without_cli_changes() {
        let selected = select_matching_pip_package_versions(
            TheRockChannel::Nightly,
            &["10.1.0a20260822".to_owned()],
            &["2.12.0+rocm10.1.0a20260822".to_owned()],
            &["0.27.0+rocm10.1.0a20260822".to_owned()],
            &["2.12.0+rocm10.1.0a20260822".to_owned()],
            None,
        )
        .expect("future nightly major should resolve");

        assert_eq!(selected.rocm, "10.1.0a20260822");
    }

    #[test]
    fn auto_select_landing_on_next_major_needs_next_python() {
        // Same feed as `nightly_accepts_future_prerelease_major_without_cli_changes`:
        // auto-select reaches ROCm 10.1.0a20260822, which needs cp314, not cp312,
        // even though it's a prerelease served from the ordinary canonical/nightly
        // index rather than `Next`'s separately hosted one.
        assert!(highest_candidate_needs_next_python(
            &["10.1.0a20260822".to_owned()],
            TheRockChannel::Nightly,
            None,
        ));

        assert!(!highest_candidate_needs_next_python(
            &["7.14.0".to_owned()],
            TheRockChannel::Nightly,
            None,
        ));
    }

    #[test]
    fn pip_runtime_installs_pinned_target_complete_stack_from_aggregate_index() {
        let package_versions = TheRockPipPackageVersions {
            rocm: "7.13.0a20260513".to_owned(),
            torch: "2.10.0+rocm7.13.0a20260513".to_owned(),
            torchvision: "0.25.0+rocm7.13.0a20260513".to_owned(),
            torchaudio: "2.10.0+rocm7.13.0a20260513".to_owned(),
            compatibility_key: "7.13.0a20260513".to_owned(),
        };
        let package_specs = therock_pip_package_specs(&package_versions, "gfx942", true);

        assert_eq!(
            package_specs,
            vec![
                "rocm[libraries,devel,device-gfx942]==7.13.0a20260513".to_owned(),
                "torch[device-gfx942]==2.10.0+rocm7.13.0a20260513".to_owned(),
                "torchvision[device-gfx942]==0.25.0+rocm7.13.0a20260513".to_owned(),
                "torchaudio==2.10.0+rocm7.13.0a20260513".to_owned(),
            ]
        );
    }

    #[test]
    fn uv_metadata_plan_accepts_non_arithmetic_audio_version() -> Result<()> {
        let plan = parse_uv_compiled_package_versions(
            "rocm==10.0.0\ntorch==2.13.0+rocm10.0.0\ntorchvision==0.28.0+rocm10.0.0\ntorchaudio==2.11.0.2+rocm10.0.0\n",
        )?;

        assert_eq!(plan.rocm, "10.0.0");
        assert_eq!(plan.torch, "2.13.0+rocm10.0.0");
        assert_eq!(plan.torchvision, "0.28.0+rocm10.0.0");
        assert_eq!(plan.torchaudio, "2.11.0.2+rocm10.0.0");
        Ok(())
    }

    /// The live ROCm 10 index publishes torchaudio rebuilds as a fourth release
    /// segment beside the original build, for more than one ROCm build at once.
    #[test]
    fn torchaudio_pin_is_the_newest_rebuild_of_the_requested_rocm_build() {
        let published = [
            "2.11.0+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
            "2.11.0+rocm10.1.0",
            "2.11.0.2+rocm10.1.0",
            "2.11.0.3+rocm10.1.0",
        ]
        .map(str::to_owned);

        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0.2+rocm10.0.0")
        );
        assert_eq!(
            latest_build_for_rocm(&published, "10.1.0").as_deref(),
            Some("2.11.0.3+rocm10.1.0")
        );
        assert_eq!(latest_build_for_rocm(&published, "10.2.0"), None);
    }

    /// Mirrors uv's default: a pre-release is only chosen when no final
    /// release exists for that build, and a post-release counts as final.
    #[test]
    fn torchaudio_pin_prefers_final_releases_like_uv() {
        let published = [
            "2.12.0a0+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
            "2.11.0+rocm10.0.0",
        ]
        .map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0.2+rocm10.0.0")
        );

        let only_pre = ["2.12.0a0+rocm10.0.0", "2.12.0a1+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&only_pre, "10.0.0").as_deref(),
            Some("2.12.0a1+rocm10.0.0")
        );

        let post = ["2.11.0+rocm10.0.0", "2.11.0.post1+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&post, "10.0.0").as_deref(),
            Some("2.11.0.post1+rocm10.0.0")
        );
    }

    /// The live `whl-next` shape for `+rocm10.1.0`: a CPU-only `2.11.0`, the
    /// hardware-verified `2.11.0.2`, and a newer `2.11.0.3` sharing the label.
    fn rocm10_1_torchaudio_builds() -> Vec<String> {
        [
            "2.11.0+rocm10.1.0",
            "2.11.0.2+rocm10.1.0",
            "2.11.0.3+rocm10.1.0",
            "2.11.0.2+rocm10.0.0",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    /// A verified release that is published wins over a newer build with the
    /// same `+rocm` label, and the printed line says it is the verified one.
    #[test]
    fn torchaudio_pin_takes_the_published_verified_release_over_a_newer_build() -> Result<()> {
        let pin = torchaudio_pin(
            &rocm10_1_torchaudio_builds(),
            "10.1.0",
            Some("2.11.0.2"),
            "https://index.test",
        )?;
        assert_eq!(pin.version, "2.11.0.2+rocm10.1.0");
        assert_eq!(pin.basis, TorchaudioPinBasis::Verified);
        assert!(pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains("torchaudio 2.11.0.2+rocm10.1.0, the build verified on hardware")
                && !line.contains("not hardware-verified"),
            "{line}"
        );
        // The requirement handed to `uv` names the same build the line does.
        assert_eq!(
            published_pip_requirements("10.1.0", &pin.version, "gfx1200", false)[3],
            "torchaudio==2.11.0.2+rocm10.1.0"
        );
        Ok(())
    }

    /// The verified release is matched as a whole PEP 440 version, never a
    /// prefix: a verified `2.11.0` must not be satisfied by `2.11.0.2`.
    #[test]
    fn torchaudio_pin_matches_the_verified_release_exactly() -> Result<()> {
        let published = ["2.11.0.2+rocm10.1.0", "2.11.0.3+rocm10.1.0"].map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.0", Some("2.11.0"), "https://index.test")?;
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0",
                other_build: None,
            }
        );
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        Ok(())
    }

    /// A verified release published only for another ROCm build falls back to
    /// the newest build for the requested one, and the printed reason names the
    /// ROCm build as the cause, not the Python or platform: the verified release
    /// is matched by ROCm line, so it can be published here under another label
    /// of the same line.
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_the_verified_release_is_published_for_another_rocm_build()
    -> Result<()> {
        let published = [
            "2.11.0+rocm10.1.1",
            "2.11.0.3+rocm10.1.1",
            "2.11.0.2+rocm10.1.0",
        ]
        .map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.1", Some("2.11.0.2"), "https://index.test")?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.1");
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0.2",
                other_build: Some("2.11.0.2+rocm10.1.0".to_owned()),
            }
        );
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.1, the newest build published for ROCm 10.1.1"
            ) && line.contains("not hardware-verified")
                && line.contains(
                    "the verified release, 2.11.0.2, is published for this Python and platform only for another ROCm build (2.11.0.2+rocm10.1.0), not for ROCm 10.1.1"
                )
                && !line.contains("is not published for this Python and platform"),
            "{line}"
        );
        Ok(())
    }

    /// A verified release with no build at all for this Python and platform
    /// falls back to the newest build, and the printed line says the release is
    /// not published here. An unlabelled build is not a ROCm build, so it does
    /// not count as one "for another ROCm build".
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_the_verified_release_is_unpublished() -> Result<()>
    {
        let published = ["2.11.0+rocm10.1.0", "2.11.0.3+rocm10.1.0", "2.11.0.2"].map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.0", Some("2.11.0.2"), "https://index.test")?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0.2",
                other_build: None,
            }
        );
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.0, the newest build published for ROCm 10.1.0"
            ) && line.contains("not hardware-verified")
                && line.contains(
                    "the verified release, 2.11.0.2, is not published for this Python and platform"
                ),
            "{line}"
        );
        Ok(())
    }

    /// A real install notes an unverified torchaudio in its install log, for
    /// both reasons a pin can be unverified, and stays quiet for a verified
    /// one. The note is the pin's own `describe()` line, so it states the
    /// decision that was actually taken.
    #[test]
    fn unverified_torchaudio_note_is_logged_only_for_an_unverified_pin() -> Result<()> {
        let builds = rocm10_1_torchaudio_builds();
        let verified = torchaudio_pin(&builds, "10.1.0", Some("2.11.0.2"), "https://index.test")?;
        let not_published =
            torchaudio_pin(&builds, "10.1.0", Some("2.11.0.9"), "https://index.test")?;
        let no_verified_release = torchaudio_pin(&builds, "10.1.0", None, "https://index.test")?;
        assert_eq!(verified.basis, TorchaudioPinBasis::Verified);
        assert!(matches!(
            not_published.basis,
            TorchaudioPinBasis::VerifiedNotPublished { .. }
        ));
        assert_eq!(
            no_verified_release.basis,
            TorchaudioPinBasis::NoVerifiedRelease
        );

        assert_eq!(unverified_torchaudio_note(&verified), None);
        for pin in [&not_published, &no_verified_release] {
            let note = unverified_torchaudio_note(pin);
            assert_eq!(note, Some(format!("Note: {}", pin.describe())));
            assert!(
                note.as_deref()
                    .is_some_and(|note| note.contains("torchaudio 2.11.0.3+rocm10.1.0")
                        && note.contains("not hardware-verified")),
                "{note:?}"
            );
        }
        Ok(())
    }

    /// With no verified release recorded for the line, the newest build is
    /// pinned, and the printed line says it is not hardware-verified.
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_no_release_is_verified() -> Result<()> {
        let pin = torchaudio_pin(
            &rocm10_1_torchaudio_builds(),
            "10.1.0",
            None,
            "https://index.test",
        )?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        assert_eq!(pin.basis, TorchaudioPinBasis::NoVerifiedRelease);
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.0, the newest build published for ROCm 10.1.0"
            ) && line.contains("not hardware-verified")
                && line.contains("no torchaudio release is recorded as verified"),
            "{line}"
        );
        Ok(())
    }

    /// The verified release the SDK reads is the vLLM engine's: 10.1 has one,
    /// 10.0 has none yet. Pins the wiring, not the table (which the engine's
    /// own tests own), so a change to it is seen here as a deliberate choice.
    #[test]
    fn torchaudio_verified_release_comes_from_the_vllm_discover_table() {
        assert_eq!(
            rocm_engine_vllm::verified_torchaudio_release("10.1.0"),
            Some("2.11.0.2")
        );
        assert_eq!(
            rocm_engine_vllm::verified_torchaudio_release("10.0.0"),
            None
        );
    }

    /// A version the parser cannot read is left out, never guessed at: a newer
    /// unreadable build must not win the pin.
    #[test]
    fn torchaudio_pin_skips_versions_it_cannot_read() -> Result<()> {
        let published = [
            "2.11.0+rocm10.0.0",
            "2.12.0-local-hack+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
        ]
        .map(str::to_owned);
        assert_eq!(
            torchaudio_pin(&published, "10.0.0", None, "https://index.test")?.version,
            "2.11.0.2+rocm10.0.0"
        );
        Ok(())
    }

    /// The two refusals name different problems, so each says which one it is.
    #[test]
    fn torchaudio_pin_refusal_says_whether_the_build_is_missing_or_unreadable() {
        let other_build = ["2.11.0.3+rocm10.1.0"].map(str::to_owned);
        let missing = torchaudio_pin(
            &other_build,
            "10.0.0",
            Some("2.11.0.2"),
            "https://index.test",
        )
        .expect_err("no torchaudio for the pinned build must refuse")
        .to_string();
        assert!(
            missing.contains("no torchaudio built for ROCm 10.0.0 is published"),
            "{missing}"
        );

        let unreadable = ["2.12.0-local-hack+rocm10.0.0"].map(str::to_owned);
        let unreadable = torchaudio_pin(&unreadable, "10.0.0", None, "https://index.test")
            .expect_err("an unreadable-only build must refuse")
            .to_string();
        assert!(
            unreadable.contains("none has a PEP 440 version")
                && unreadable.contains("2.12.0-local-hack+rocm10.0.0")
                && !unreadable.contains("no torchaudio built"),
            "{unreadable}"
        );
    }

    /// A post-release that is itself a dev build is still a pre-release to
    /// `uv`, so a final release wins over it.
    #[test]
    fn torchaudio_pin_treats_a_dev_post_release_as_a_pre_release() {
        let published = ["2.11.0+rocm10.0.0", "2.11.0.post1.dev0+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0+rocm10.0.0")
        );
    }

    #[test]
    fn uv_metadata_plan_rejects_mixed_rocm_builds() {
        let error = parse_uv_compiled_package_versions(
            "rocm==10.0.0\ntorch==2.13.0+rocm10.0.0\ntorchvision==0.28.0+rocm10.1.0\ntorchaudio==2.11.0.2+rocm10.0.0\n",
        )
        .expect_err("a mixed ROCm build must fail closed")
        .to_string();

        assert!(error.contains("torchvision"), "{error}");
        assert!(
            error.contains("does not share ROCm build 10.0.0"),
            "{error}"
        );
    }

    #[test]
    fn uv_metadata_resolver_maps_supported_python_platforms() -> Result<()> {
        let windows = WheelCompatibility {
            python_tag: "cp312".to_owned(),
            platform_tags: vec!["win_amd64".to_owned(), "any".to_owned()],
        };
        assert_eq!(uv_python_version(&windows)?, "3.12");
        assert_eq!(uv_python_platform(&windows)?, "x86_64-pc-windows-msvc");

        let linux = WheelCompatibility {
            python_tag: "cp312".to_owned(),
            platform_tags: vec!["linux_x86_64".to_owned(), "any".to_owned()],
        };
        assert_eq!(uv_python_platform(&linux)?, "x86_64-unknown-linux-gnu");
        Ok(())
    }

    /// The default install skips the compiler toolchain, which is roughly half
    /// the download and is only needed to build GPU code. The torch stack is
    /// unaffected.
    #[test]
    fn pip_runtime_omits_devel_extra_by_default() {
        let package_versions = TheRockPipPackageVersions {
            rocm: "7.13.0a20260513".to_owned(),
            torch: "2.10.0+rocm7.13.0a20260513".to_owned(),
            torchvision: "0.25.0+rocm7.13.0a20260513".to_owned(),
            torchaudio: "2.10.0+rocm7.13.0a20260513".to_owned(),
            compatibility_key: "7.13.0a20260513".to_owned(),
        };

        let package_specs = therock_pip_package_specs(&package_versions, "gfx942", false);

        assert_eq!(
            package_specs,
            vec![
                "rocm[libraries,device-gfx942]==7.13.0a20260513".to_owned(),
                "torch[device-gfx942]==2.10.0+rocm7.13.0a20260513".to_owned(),
                "torchvision[device-gfx942]==0.25.0+rocm7.13.0a20260513".to_owned(),
                "torchaudio==2.10.0+rocm7.13.0a20260513".to_owned(),
            ]
        );
        assert!(
            !package_specs[0].contains("devel"),
            "default install must not request the toolchain: {package_specs:?}"
        );
    }

    fn devel_test_resolution() -> PipRuntimeResolution {
        PipRuntimeResolution {
            family: "gfx94X-dcgpu".to_owned(),
            family_source: "detected".to_owned(),
            index_url: "https://example.invalid/simple".to_owned(),
            layout: SourceLayout::Canonical,
            latest_version: "7.13.0".to_owned(),
            newest_repo_version: None,
            version_resolution_specs: None,
            torchaudio_pin: None,
            package_versions: TheRockPipPackageVersions {
                rocm: "7.13.0".to_owned(),
                torch: "2.11.0+rocm7.13.0".to_owned(),
                torchvision: "0.26.0+rocm7.13.0".to_owned(),
                torchaudio: "2.11.0+rocm7.13.0".to_owned(),
                compatibility_key: "7.13.0".to_owned(),
            },
            device_target: AggregateDeviceTarget::Exact("gfx942".to_owned()),
            published_device_targets: vec!["gfx942".to_owned()],
        }
    }

    /// The seam the `--devel` flag actually travels through.
    ///
    /// `wheel_runtime_composition` produces the specs handed to `uv` AND the
    /// specs recorded in the manifest, so hardcoding either polarity here is
    /// the single change that would silently restore the old behaviour. Both
    /// directions are pinned, and the device extra is asserted alongside so a
    /// fix to one axis cannot quietly drop the other.
    #[test]
    fn wheel_composition_requests_the_toolchain_only_when_asked() {
        let resolution = devel_test_resolution();
        let target = AggregateDeviceTarget::Exact("gfx942".to_owned());

        let runtime_only = wheel_runtime_composition(&resolution, &target, false);
        assert_eq!(
            runtime_only.package_specs[0], "rocm[libraries,device-gfx942]==7.13.0",
            "a default install must not request the toolchain: {:?}",
            runtime_only.package_specs
        );

        let with_devel = wheel_runtime_composition(&resolution, &target, true);
        assert_eq!(
            with_devel.package_specs[0], "rocm[libraries,devel,device-gfx942]==7.13.0",
            "--devel must reach the specs: {:?}",
            with_devel.package_specs
        );

        // The two axes are independent: opting into the toolchain must not
        // disturb the device payload, and vice versa.
        assert_eq!(
            runtime_only.package_specs[1..],
            with_devel.package_specs[1..],
            "devel must only affect the rocm spec"
        );
        assert_eq!(runtime_only.rocm_sdk_target, with_devel.rocm_sdk_target);
    }

    /// The other half of the same claim, on the *resolution* path.
    ///
    /// `uv pip compile` decides which versions exist before anything is
    /// composed, so asking it for `devel` on a default install constrains the
    /// version choice by a toolchain the user declined — and fails the install
    /// outright when no version can satisfy it. That is a separate code path
    /// from `wheel_runtime_composition` above, and the two must name the same
    /// extras.
    #[test]
    fn version_resolution_requests_the_toolchain_only_when_asked() {
        let runtime_only =
            published_pip_requirements("10.0.0", "2.11.0.2+rocm10.0.0", "gfx1200", false);
        assert_eq!(
            runtime_only[0], "rocm[libraries,device-gfx1200]==10.0.0",
            "a default install must not resolve against the toolchain: {runtime_only:?}"
        );

        let with_devel =
            published_pip_requirements("10.0.0", "2.11.0.2+rocm10.0.0", "gfx1200", true);
        assert_eq!(
            with_devel[0], "rocm[libraries,devel,device-gfx1200]==10.0.0",
            "--devel must reach the resolved requirements: {with_devel:?}"
        );

        // Only the `rocm` requirement carries the toolchain axis. torch and
        // torchvision are unversioned because choosing those versions is what
        // the resolve is for; torchaudio is pinned because its metadata does not
        // follow the `rocm` pin (see `published_pip_requirements`).
        assert_eq!(
            runtime_only[1..],
            with_devel[1..],
            "devel must only affect the rocm requirement"
        );
        assert_eq!(
            runtime_only[1..],
            [
                "torch[device-gfx1200]".to_owned(),
                "torchvision[device-gfx1200]".to_owned(),
                "torchaudio==2.11.0.2+rocm10.0.0".to_owned(),
            ]
        );
    }

    /// Resolution and composition are two code paths reading one flag, and the
    /// bug this pins is them disagreeing: the requirements handed to `uv` were
    /// hardcoded to `devel` while the plan printed and installed `libraries`.
    /// Asserting each in isolation cannot catch that; asserting they agree can.
    #[test]
    fn resolution_and_install_name_the_same_rocm_extras() {
        let resolution = devel_test_resolution();
        let target = AggregateDeviceTarget::Exact("gfx942".to_owned());

        for include_devel in [false, true] {
            let requirements = published_pip_requirements(
                &resolution.package_versions.rocm,
                &resolution.package_versions.torchaudio,
                "gfx942",
                include_devel,
            );
            let composition = wheel_runtime_composition(&resolution, &target, include_devel);
            assert_eq!(
                rocm_requirement_extras(&requirements[0]),
                rocm_requirement_extras(&composition.package_specs[0]),
                "resolution and install must request the same rocm extras for \
                 include_devel={include_devel}: {requirements:?} vs {:?}",
                composition.package_specs
            );
            // torchaudio is the one stack package resolution pins rather than
            // leaving to `uv`, so it has to be the very build install names.
            assert_eq!(
                requirements[3], composition.package_specs[3],
                "resolution and install must name the same torchaudio for \
                 include_devel={include_devel}"
            );
        }
    }

    /// The `...` of a `rocm[...]==version` requirement.
    fn rocm_requirement_extras(requirement: &str) -> &str {
        requirement
            .strip_prefix("rocm[")
            .and_then(|rest| rest.split_once(']'))
            .map_or_else(
                || panic!("not a rocm extras requirement: {requirement}"),
                |(extras, _)| extras,
            )
    }

    /// The manifest answer is derived from the specs that were installed, so a
    /// recorded composition and the `devel` field can never disagree.
    #[test]
    fn manifest_reads_devel_back_out_of_the_recorded_composition() {
        let resolution = devel_test_resolution();
        let target = AggregateDeviceTarget::Exact("gfx942".to_owned());

        for include_devel in [false, true] {
            let composition = wheel_runtime_composition(&resolution, &target, include_devel);
            assert_eq!(
                wheel_composition_includes_devel(Some(&composition)),
                Some(include_devel),
                "composition round-trip failed for include_devel={include_devel}"
            );

            // Even when the `devel` field contradicts the specs, the specs win:
            // they are what `uv` was given.
            let manifest = InstalledRuntimeManifest {
                wheel_composition: Some(composition),
                devel: !include_devel,
                ..test_runtime_manifest(
                    "release-wheel-gfx94X-dcgpu-7.13.0",
                    "therock-release:gfx94X-dcgpu",
                    1,
                )
            };
            assert_eq!(
                manifest.includes_devel(),
                include_devel,
                "the recorded specs must outrank a stale devel field"
            );
        }
    }

    #[test]
    fn pip_runtime_selects_latest_common_rocm_suffix_not_latest_rocm_package() {
        let rocm_versions = vec![
            "7.13.0".to_owned(),
            "7.13.1".to_owned(),
            "7.14.0".to_owned(),
        ];
        let torch_versions = vec![
            "2.9.1+rocm7.13.1".to_owned(),
            "2.10.0+rocm7.13.1".to_owned(),
        ];
        let torchvision_versions = vec![
            "0.24.0+rocm7.13.1".to_owned(),
            "0.25.0+rocm7.13.1".to_owned(),
        ];
        let torchaudio_versions = vec![
            "2.9.0+rocm7.13.1".to_owned(),
            "2.10.0+rocm7.13.1".to_owned(),
        ];

        let selected = select_matching_pip_package_versions(
            TheRockChannel::Release,
            &rocm_versions,
            &torch_versions,
            &torchvision_versions,
            &torchaudio_versions,
            None,
        )
        .expect("expected compatible package set");

        assert_eq!(selected.rocm, "7.13.1");
        assert_eq!(selected.torch, "2.10.0+rocm7.13.1");
        assert_eq!(selected.torchvision, "0.25.0+rocm7.13.1");
        assert_eq!(selected.torchaudio, "2.10.0+rocm7.13.1");
    }

    #[test]
    fn pip_runtime_rejects_date_only_rocm_suffix_matches() {
        let rocm_versions = vec!["7.14.0a20260602".to_owned()];
        let torch_versions = vec!["2.10.0+rocm7.13.0a20260602".to_owned()];
        let torchvision_versions = vec!["0.25.0+rocm7.13.0a20260602".to_owned()];
        let torchaudio_versions = vec!["2.10.0+rocm7.13.0a20260602".to_owned()];

        assert!(
            select_matching_pip_package_versions(
                TheRockChannel::Release,
                &rocm_versions,
                &torch_versions,
                &torchvision_versions,
                &torchaudio_versions,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn pip_runtime_selects_requested_build_date_stack() -> Result<()> {
        let rocm_versions = vec![
            "7.13.0a20260604".to_owned(),
            "7.13.0a20260605".to_owned(),
            "7.13.0a20260606".to_owned(),
        ];
        let torch_versions = vec![
            "2.10.0+rocm7.13.0a20260605".to_owned(),
            "2.10.0+rocm7.13.0a20260606".to_owned(),
        ];
        let torchvision_versions = vec![
            "0.25.0+rocm7.13.0a20260605".to_owned(),
            "0.25.0+rocm7.13.0a20260606".to_owned(),
        ];
        let torchaudio_versions = vec![
            "2.10.0+rocm7.13.0a20260605".to_owned(),
            "2.10.0+rocm7.13.0a20260606".to_owned(),
        ];
        let selector = RuntimeVersionSelector::build_date("06052026")?;

        let selected = select_matching_pip_package_versions(
            TheRockChannel::Release,
            &rocm_versions,
            &torch_versions,
            &torchvision_versions,
            &torchaudio_versions,
            Some(&selector),
        )
        .expect("expected requested build-date package set");

        assert_eq!(selected.rocm, "7.13.0a20260605");
        assert_eq!(selected.torch, "2.10.0+rocm7.13.0a20260605");
        assert_eq!(
            selector,
            RuntimeVersionSelector::BuildDate("2026-06-05".to_owned())
        );
        Ok(())
    }

    #[test]
    fn pip_runtime_rejects_requested_build_date_without_matching_stack() -> Result<()> {
        let rocm_versions = vec!["7.13.0a20260605".to_owned()];
        let torch_versions = vec!["2.10.0+rocm7.13.0a20260606".to_owned()];
        let torchvision_versions = vec!["0.25.0+rocm7.13.0a20260606".to_owned()];
        let torchaudio_versions = vec!["2.10.0+rocm7.13.0a20260606".to_owned()];
        let selector = RuntimeVersionSelector::build_date("2026-06-05")?;

        assert!(
            select_matching_pip_package_versions(
                TheRockChannel::Release,
                &rocm_versions,
                &torch_versions,
                &torchvision_versions,
                &torchaudio_versions,
                Some(&selector),
            )
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn aggregate_wheel_resolution_hint_does_not_recommend_family_override() {
        let hint = canonical_wheel_resolution_hint(TheRockChannel::Release);
        assert!(!hint.contains("--family"));
        assert!(hint.contains("--channel nightly"));
        if !runtime_is_windows() {
            assert!(hint.contains("--format tarball"));
        }
    }
}
