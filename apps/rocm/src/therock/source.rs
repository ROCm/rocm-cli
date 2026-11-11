// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Where a TheRock install reads from. The release/nightly channel, the
//! published hosting layout (canonical, or ROCm 10's separately hosted next
//! layout) and the artifact bases each pair resolves to, including the opt-in
//! rule that makes a base override inert unless asked for twice. Also the
//! version a user pins (`RuntimeVersionSelector`) and the resolved GPU family
//! (`FamilyResolution`) that travel with a request, because choosing the layout
//! reads both.
//!
//! Depends on `version` for the pin grammar; `family` builds on the types here.

use super::version::{
    VersionStage, normalize_requested_build_date, parse_version, runtime_version_build_date,
};
use anyhow::{Result, bail};
use rocm_core::detect_host_gpu_diagnostics;

const THEROCK_NIGHTLY_PIP_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/whl-multi-arch";
const THEROCK_RELEASE_PIP_INDEX_BASE: &str = "https://repo.amd.com/rocm/whl-multi-arch";
const THEROCK_RELEASE_TARBALL_BASE: &str = "https://repo.amd.com/rocm/tarball/";
const THEROCK_NIGHTLY_TARBALL_BASE: &str = "https://rocm.nightlies.amd.com/tarball/";
/// The next layout's stable aggregate pip index. Same simple-index shape the
/// canonical multi-arch stream publishes, on a separate host, with its own
/// package generation; it is not a mirror of, or a fallback for, the canonical
/// index.
const THEROCK_NEXT_PIP_INDEX_BASE: &str = "https://stable.repo.amd.com/rocm/whl-next";
/// The next layout's stable tarball catalog. Same scrapeable listing the
/// canonical catalog uses, but it publishes non-release sibling archives beside
/// the real dist archive (see `select_tarball_candidate` in `resolve/tarball.rs`).
const THEROCK_NEXT_TARBALL_BASE: &str = "https://stable.repo.amd.com/rocm/core/tarball/";
/// The next layout's nightly aggregate pip index: the same package shape as
/// [`THEROCK_NEXT_PIP_INDEX_BASE`] on the nightly host.
const THEROCK_NEXT_NIGHTLY_PIP_INDEX_BASE: &str = "https://nightly.repo.amd.com/rocm/whl-next";
/// The next layout's nightly tarball catalog, beside
/// [`THEROCK_NEXT_TARBALL_BASE`] on the nightly host.
const THEROCK_NEXT_NIGHTLY_TARBALL_BASE: &str = "https://nightly.repo.amd.com/rocm/core/tarball/";
/// The first nightly build date the next layout publishes. The canonical
/// nightly stream's last build is 2026-08-22 and the next layout's first is
/// 2026-08-23, so the two never overlap: an earlier build can only be fetched
/// from the canonical hosts and a later one only from the next layout's.
pub(super) const THEROCK_NEXT_NIGHTLY_FIRST_BUILD_DATE: &str = "2026-08-23";
pub(super) const THEROCK_SOURCE_LAYOUT_GENERATION: &str = "multi-arch-v2";
/// Recorded in a [`WheelRuntimeComposition`](crate::runtime_registry::WheelRuntimeComposition)
/// installed from the ROCm 10 layout, so a later update reads back the layout
/// that produced the runtime instead of guessing it from the version.
const THEROCK_NEXT_LAYOUT_GENERATION: &str = "next-v1";
/// The first stable ROCm major version published only in the next layout. A
/// stable release pin at or past it is a request the canonical stream provably
/// cannot serve.
pub(super) const THEROCK_NEXT_MIN_MAJOR: u32 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TheRockChannel {
    Release,
    Nightly,
}

/// Which published hosting layout a resolution reads from.
///
/// Orthogonal to [`TheRockChannel`]. `Next` is the layout AMD now publishes
/// every channel to (`stable.repo.amd.com` and `nightly.repo.amd.com`), and
/// what a request with no pin installs. `Canonical` is the earlier release and
/// nightly stream (`repo.amd.com` and `rocm.nightlies.amd.com`), which no longer
/// receives new builds but still serves every ROCm 7 release and the nightlies
/// up to 2026-08-22, so a pin of one of those reads it. Nothing degrades from
/// one layout into the other: [`layout_serving`] picks exactly one from the
/// request, and an install resolves only the URLs, device target and package
/// specs of that layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SourceLayout {
    Canonical,
    Next,
}

impl SourceLayout {
    pub(super) const fn generation(self) -> &'static str {
        match self {
            Self::Canonical => THEROCK_SOURCE_LAYOUT_GENERATION,
            Self::Next => THEROCK_NEXT_LAYOUT_GENERATION,
        }
    }

    /// The layout a recorded
    /// [`WheelRuntimeComposition`](crate::runtime_registry::WheelRuntimeComposition)
    /// was installed from.
    ///
    /// Manifests written before compositions existed carry no generation and
    /// therefore came from the canonical stream. A non-empty generation is an
    /// explicit provenance contract: reject values this build cannot reproduce
    /// instead of silently redirecting an update to the canonical source.
    pub(super) fn from_generation(generation: Option<&str>) -> Result<Self> {
        match generation {
            None | Some(THEROCK_SOURCE_LAYOUT_GENERATION) => Ok(Self::Canonical),
            Some(THEROCK_NEXT_LAYOUT_GENERATION) => Ok(Self::Next),
            Some(other) => bail!(
                "runtime records unsupported TheRock source layout generation `{other}`; update rocm-cli before updating this runtime"
            ),
        }
    }
}

/// The concrete bases one (channel, layout) pair resolves artifacts from.
///
/// Owns its strings because every base is overridable for fixture testing; see
/// [`env_override_base`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedSource {
    pub(super) wheel_index: String,
    pub(super) tarball_catalog: String,
    pub(super) layout: SourceLayout,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum RuntimeVersionSelector {
    Version(String),
    BuildDate(String),
}

impl RuntimeVersionSelector {
    pub(crate) fn version(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            bail!("TheRock version cannot be empty");
        }
        if trimmed
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
        {
            bail!("TheRock version must be a single version string");
        }
        Ok(Self::Version(trimmed.to_owned()))
    }

    pub(crate) fn build_date(value: impl AsRef<str>) -> Result<Self> {
        Ok(Self::BuildDate(normalize_requested_build_date(
            value.as_ref(),
        )?))
    }

    pub(super) fn describe(&self) -> String {
        match self {
            Self::Version(version) => format!("version {version}"),
            Self::BuildDate(date) => format!("build date {date}"),
        }
    }

    pub(super) fn matches_version(&self, version: &str) -> bool {
        match self {
            Self::Version(requested) => version == requested,
            Self::BuildDate(date) => runtime_version_build_date(version).as_deref() == Some(date),
        }
    }
}

impl TheRockChannel {
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "release" => Ok(Self::Release),
            "nightly" => Ok(Self::Nightly),
            other => bail!("unsupported TheRock channel: {other}"),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }
}

/// Whether a `ROCM_CLI_THEROCK_*_BASE` override is honoured at all.
///
/// Every artifact base is a trust boundary: a stray override left in a shell
/// profile would silently redirect a real install to an untrusted host. Reading
/// them requires a second, deliberate opt-in that no normal install sets, so the
/// overrides exist for fixture servers and operators who mean it, and are inert
/// otherwise. See `docs/release-trust.md`.
fn therock_base_override_allowed() -> bool {
    std::env::var("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE")
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// The base to use, given the opt-in flag and whatever the per-base variable
/// says. Pure so the trust rule is stated once and can be checked without
/// mutating process environment from a test.
fn select_base_override(allowed: bool, override_value: Option<&str>, default: &str) -> String {
    if !allowed {
        return default.to_owned();
    }
    override_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map_or_else(|| default.to_owned(), str::to_owned)
}

fn env_override_base(env_var: &str, default: &str) -> String {
    select_base_override(
        therock_base_override_allowed(),
        std::env::var(env_var).ok().as_deref(),
        default,
    )
}

/// The bases `channel` publishes under `layout`.
pub(super) fn resolve_source(channel: TheRockChannel, layout: SourceLayout) -> ResolvedSource {
    let (wheel_var, wheel_default, tarball_var, tarball_default) = match (layout, channel) {
        (SourceLayout::Canonical, TheRockChannel::Release) => (
            "ROCM_CLI_THEROCK_RELEASE_PIP_BASE",
            THEROCK_RELEASE_PIP_INDEX_BASE,
            "ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE",
            THEROCK_RELEASE_TARBALL_BASE,
        ),
        (SourceLayout::Canonical, TheRockChannel::Nightly) => (
            "ROCM_CLI_THEROCK_NIGHTLY_PIP_BASE",
            THEROCK_NIGHTLY_PIP_INDEX_BASE,
            "ROCM_CLI_THEROCK_NIGHTLY_TARBALL_BASE",
            THEROCK_NIGHTLY_TARBALL_BASE,
        ),
        (SourceLayout::Next, TheRockChannel::Release) => (
            "ROCM_CLI_THEROCK_NEXT_PIP_BASE",
            THEROCK_NEXT_PIP_INDEX_BASE,
            "ROCM_CLI_THEROCK_NEXT_TARBALL_BASE",
            THEROCK_NEXT_TARBALL_BASE,
        ),
        (SourceLayout::Next, TheRockChannel::Nightly) => (
            "ROCM_CLI_THEROCK_NEXT_NIGHTLY_PIP_BASE",
            THEROCK_NEXT_NIGHTLY_PIP_INDEX_BASE,
            "ROCM_CLI_THEROCK_NEXT_NIGHTLY_TARBALL_BASE",
            THEROCK_NEXT_NIGHTLY_TARBALL_BASE,
        ),
    };
    ResolvedSource {
        wheel_index: env_override_base(wheel_var, wheel_default),
        tarball_catalog: env_override_base(tarball_var, tarball_default),
        layout,
    }
}

/// The layout that publishes what `selector` asks for on `channel`.
///
/// No pin installs the channel's newest build, and every channel's newest
/// builds are in the next layout. A pin goes where that exact build lives:
///
/// - release: a stable version at or past [`THEROCK_NEXT_MIN_MAJOR`] is
///   next-only, an earlier one is canonical-only;
/// - nightly: a build date (given directly or carried by a prerelease version)
///   on or after [`THEROCK_NEXT_NIGHTLY_FIRST_BUILD_DATE`] is next-only, an
///   earlier one is canonical-only. A *stable* ROCm 10+ version on the nightly
///   channel names a release, so it routes to the next layout, where
///   [`select_source_layout`] refuses it.
///
/// A pin this CLI cannot read stays canonical: nothing can be said about where
/// it lives, and the canonical lookup fails on it as it always has.
pub(super) fn layout_serving(
    channel: TheRockChannel,
    selector: Option<&RuntimeVersionSelector>,
) -> SourceLayout {
    let Some(selector) = selector else {
        return SourceLayout::Next;
    };
    let by_build_date = |build_date: &str| {
        if build_date >= THEROCK_NEXT_NIGHTLY_FIRST_BUILD_DATE {
            SourceLayout::Next
        } else {
            SourceLayout::Canonical
        }
    };
    match (channel, selector) {
        (TheRockChannel::Nightly, RuntimeVersionSelector::BuildDate(date)) => by_build_date(date),
        (TheRockChannel::Release, RuntimeVersionSelector::BuildDate(_)) => SourceLayout::Canonical,
        (_, RuntimeVersionSelector::Version(version)) => {
            let Some(parsed) = parse_version(version) else {
                return SourceLayout::Canonical;
            };
            let stable_next =
                parsed.stage == VersionStage::Stable && parsed.major >= THEROCK_NEXT_MIN_MAJOR;
            match channel {
                TheRockChannel::Release if stable_next => SourceLayout::Next,
                TheRockChannel::Release => SourceLayout::Canonical,
                TheRockChannel::Nightly => match runtime_version_build_date(version) {
                    Some(date) => by_build_date(&date),
                    None if stable_next => SourceLayout::Next,
                    None => SourceLayout::Canonical,
                },
            }
        }
    }
}

/// The layout an install must read from, and the refusal when it cannot.
///
/// [`layout_serving`] decides the layout. The next layout additionally needs an
/// exact arch, because it picks its device payload by exact arch and a grouped
/// family names none, and it publishes no stable ROCm release on the nightly
/// channel.
pub(super) fn select_source_layout(
    channel: TheRockChannel,
    family_resolution: &FamilyResolution,
    version_selector: Option<&RuntimeVersionSelector>,
) -> Result<SourceLayout> {
    let layout = layout_serving(channel, version_selector);
    if layout == SourceLayout::Canonical {
        return Ok(layout);
    }
    if let Some(RuntimeVersionSelector::Version(version)) = version_selector
        && matches!(channel, TheRockChannel::Nightly)
        && let Some(parsed) = parse_version(version)
        && parsed.stage == VersionStage::Stable
    {
        bail!(
            "ROCm {} and newer stable releases are published only on the release channel; re-run `rocm install sdk --channel release --version {version}`",
            parsed.major
        );
    }
    if family_resolution.raw_arch.is_none() {
        bail!(
            "this install reads the ROCm {THEROCK_NEXT_MIN_MAJOR} package layout, which requires an exact GPU arch, but the resolved target family `{}` names a group of them.\n\
             That layout selects its device payload by exact arch, so re-run with the one this host has, for example `rocm install sdk --family gfx1200`.\n\
             An earlier build, which still publishes grouped families, is installed by pinning it: `--version 7.14.1` on the release channel, or `--build-date` for a nightly.\n\n{}",
            family_resolution.family,
            detect_host_gpu_diagnostics()
        );
    }
    Ok(SourceLayout::Next)
}

/// The pre-install gate: [`select_source_layout`] for a fresh request, and the
/// recorded layout untouched for an update, which names no pin and installs from
/// the layout its runtime was built from. Gating an update on the request it
/// does not make would route it to the newest layout and refuse a runtime of the
/// earlier stream that has only a family label, though its own resolution reads
/// the layout it recorded and never needed an exact arch.
pub(super) fn gate_install_layout(
    channel: TheRockChannel,
    family_resolution: &FamilyResolution,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
) -> Result<SourceLayout> {
    match layout_override {
        Some(layout) => Ok(layout),
        None => select_source_layout(channel, family_resolution, version_selector),
    }
}

#[derive(Debug, Clone)]
pub(super) struct FamilyResolution {
    pub(super) family: String,
    pub(super) source: String,
    /// The exact GFX arch behind `family`, when one is genuinely known —
    /// `gfx1200`, never the group `gfx120X-all`.
    ///
    /// A grouped family names no single arch, and the next layout selects its
    /// device payload by exact arch, so `None` here is what makes a ROCm 10 pin
    /// refuse instead of installing a runtime with the wrong kernels.
    pub(super) raw_arch: Option<String>,
}

/// The layout an install or update will read from, as far as is knowable before
/// an interpreter has been chosen. An update passes its manifest's recorded
/// layout; a fresh install passes its request and lets the pin decide.
pub(super) fn requested_source_layout(
    channel: TheRockChannel,
    layout_override: Option<SourceLayout>,
    version_selector: Option<&RuntimeVersionSelector>,
) -> SourceLayout {
    layout_override.unwrap_or_else(|| layout_serving(channel, version_selector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::family_for_test;

    /// Which layout serves a request, for every shape of request on both
    /// channels. A request with no pin reads the newest builds, which only the
    /// next layout publishes; a pin reads whichever layout holds that build, so
    /// an earlier ROCm 7 release or nightly stays installable.
    #[test]
    fn a_request_reads_the_layout_that_publishes_it() {
        use RuntimeVersionSelector::{BuildDate, Version};
        use SourceLayout::{Canonical, Next};
        let v = |value: &str| Version(value.to_owned());
        let d = |value: &str| BuildDate(value.to_owned());
        let rows: Vec<(TheRockChannel, Option<RuntimeVersionSelector>, SourceLayout)> = vec![
            // No pin: the newest build of either channel.
            (TheRockChannel::Release, None, Next),
            (TheRockChannel::Nightly, None, Next),
            // Release pins: stable 10+ is next-only, earlier stable is canonical-only.
            (TheRockChannel::Release, Some(v("10.1.0")), Next),
            (TheRockChannel::Release, Some(v("10.0.0")), Next),
            (TheRockChannel::Release, Some(v("7.14.1")), Canonical),
            (TheRockChannel::Release, Some(v("7.9.0")), Canonical),
            // Nightly pins: the last canonical build and the first next build
            // are adjacent days, so either side of that boundary is a row.
            (
                TheRockChannel::Nightly,
                Some(v("10.1.0a20260822")),
                Canonical,
            ),
            (TheRockChannel::Nightly, Some(v("10.1.0a20260823")), Next),
            (TheRockChannel::Nightly, Some(v("10.3.0a20261110")), Next),
            (
                TheRockChannel::Nightly,
                Some(v("7.14.0a20260612")),
                Canonical,
            ),
            (TheRockChannel::Nightly, Some(d("2026-08-22")), Canonical),
            (TheRockChannel::Nightly, Some(d("2026-08-23")), Next),
            (TheRockChannel::Nightly, Some(d("2026-11-10")), Next),
            // A stable ROCm 10+ version on the nightly channel names a
            // release; it reaches the next layout so the refusal can say so.
            (TheRockChannel::Nightly, Some(v("10.0.0")), Next),
            // A pin this CLI cannot read, or a build date on the release
            // channel (which has no dated builds), says nothing about where it
            // lives: it stays canonical and fails there as it always has.
            (TheRockChannel::Release, Some(v("main")), Canonical),
            (TheRockChannel::Nightly, Some(v("main")), Canonical),
            (TheRockChannel::Release, Some(d("2026-11-10")), Canonical),
        ];
        for (channel, selector, expected) in rows {
            assert_eq!(
                layout_serving(channel, selector.as_ref()),
                expected,
                "{channel:?} {selector:?}"
            );
        }
    }

    /// The routing rule on real values: the last build the legacy host publishes
    /// is 2026-08-22 and the first the new host publishes is 2026-08-23, so the
    /// boundary constant must sit exactly between them.
    #[test]
    fn the_nightly_boundary_sits_between_the_last_canonical_and_first_next_build() {
        assert_eq!(THEROCK_NEXT_NIGHTLY_FIRST_BUILD_DATE, "2026-08-23");
        let last_canonical = RuntimeVersionSelector::version("10.1.0a20260822").unwrap();
        let first_next = RuntimeVersionSelector::version("10.1.0a20260823").unwrap();
        assert_eq!(
            layout_serving(TheRockChannel::Nightly, Some(&last_canonical)),
            SourceLayout::Canonical
        );
        assert_eq!(
            layout_serving(TheRockChannel::Nightly, Some(&first_next)),
            SourceLayout::Next
        );
    }

    /// The selection a install uses is the routing rule plus its refusals, so a
    /// request that routes to the next layout with an exact arch selects it and
    /// an earlier pin needs no arch at all.
    #[test]
    fn selecting_a_layout_applies_the_routing_rule() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));
        let grouped = family_for_test("gfx120X-all", None);
        let old_pin = RuntimeVersionSelector::version("7.14.1").unwrap();
        let new_pin = RuntimeVersionSelector::version("10.1.0").unwrap();

        assert_eq!(
            select_source_layout(TheRockChannel::Release, &exact, None).unwrap(),
            SourceLayout::Next
        );
        assert_eq!(
            select_source_layout(TheRockChannel::Nightly, &exact, None).unwrap(),
            SourceLayout::Next
        );
        assert_eq!(
            select_source_layout(TheRockChannel::Release, &exact, Some(&new_pin)).unwrap(),
            SourceLayout::Next
        );
        // An earlier release needs no exact arch: the canonical layout serves
        // grouped families.
        assert_eq!(
            select_source_layout(TheRockChannel::Release, &grouped, Some(&old_pin)).unwrap(),
            SourceLayout::Canonical
        );
    }

    /// A grouped family names no single arch, and the next layout has no
    /// grouped device payload to fall back to, so the refusal has to name the
    /// flag and a value that would work.
    #[test]
    fn rocm_10_pin_without_an_exact_arch_refuses_actionably() {
        let grouped = family_for_test("gfx120X-all", None);
        let pin = RuntimeVersionSelector::version("10.0.0").unwrap();

        let error = select_source_layout(TheRockChannel::Release, &grouped, Some(&pin))
            .expect_err("a grouped family cannot select a device payload")
            .to_string();

        assert!(error.contains("requires an exact GPU arch"), "{error}");
        assert!(error.contains("--family gfx1200"), "{error}");
        // The way out for a grouped family is naming an earlier release, which
        // must itself be a pin the router sends to the canonical layout.
        assert!(error.contains("--version 7.14.1"), "{error}");
        let escape = RuntimeVersionSelector::version("7.14.1").unwrap();
        assert_eq!(
            select_source_layout(TheRockChannel::Release, &grouped, Some(&escape)).unwrap(),
            SourceLayout::Canonical
        );
    }

    /// With no pin, the newest build is only in the next layout, so a grouped
    /// family must be refused with the same actionable message rather than
    /// silently installing from the stale canonical stream.
    #[test]
    fn an_unpinned_install_with_a_grouped_family_refuses_instead_of_going_canonical() {
        let grouped = family_for_test("gfx120X-all", None);

        for channel in [TheRockChannel::Release, TheRockChannel::Nightly] {
            let error = select_source_layout(channel, &grouped, None)
                .expect_err("a grouped family cannot select a next-layout payload")
                .to_string();
            assert!(error.contains("requires an exact GPU arch"), "{error}");
        }
    }

    #[test]
    fn rocm_10_pin_on_nightly_refuses_instead_of_resolving() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));
        let pin = RuntimeVersionSelector::version("10.0.0").unwrap();

        let error = select_source_layout(TheRockChannel::Nightly, &exact, Some(&pin))
            .expect_err("the next layout has no nightly stream")
            .to_string();

        assert!(error.contains("--channel release"), "{error}");
    }

    /// A pinned nightly prerelease of a future major (e.g. a ROCm 10 alpha
    /// build) is routed by its build date, not its major: the legacy nightlies
    /// carry ROCm 10 alphas up to 2026-08-22. Gating on major alone would refuse
    /// this pin and then suggest a release-channel retry that the
    /// stable-version filter would also reject, leaving no working path.
    #[test]
    fn nightly_prerelease_pin_of_a_future_major_is_routed_by_build_date() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));
        let pin = RuntimeVersionSelector::version("10.1.0a20260822").unwrap();

        assert_eq!(
            select_source_layout(TheRockChannel::Nightly, &exact, Some(&pin)).unwrap(),
            SourceLayout::Canonical
        );
        let later = RuntimeVersionSelector::version("10.1.0a20260901").unwrap();
        assert_eq!(
            select_source_layout(TheRockChannel::Nightly, &exact, Some(&later)).unwrap(),
            SourceLayout::Next
        );
    }

    /// A ROCm major newer than any this CLI has seen must resolve without a CLI
    /// release: the layout is chosen by `major >= THEROCK_NEXT_MIN_MAJOR`, not by
    /// a list of known majors. A stable pin of a later major takes the next
    /// layout, and a pinned nightly prerelease of one is routed by its build date.
    #[test]
    fn a_future_rocm_major_selects_its_layout_without_a_cli_change() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));

        for stable in ["11.0.0", "12.3.4", "99.0.0"] {
            let pin = RuntimeVersionSelector::version(stable).unwrap();
            assert_eq!(
                select_source_layout(TheRockChannel::Release, &exact, Some(&pin)).unwrap(),
                SourceLayout::Next,
                "stable pin {stable}"
            );
        }

        // Built after the next layout began, so it can only be there; the major
        // plays no part.
        let alpha = RuntimeVersionSelector::version("11.0.0a20270101").unwrap();
        assert_eq!(
            select_source_layout(TheRockChannel::Nightly, &exact, Some(&alpha)).unwrap(),
            SourceLayout::Next
        );
    }

    /// An update names no pin, so on its own it would be routed to the newest
    /// layout and, with only a family label, refused; the layout it recorded is
    /// what it installs from, so that is what the gate lets through.
    #[test]
    fn an_update_is_gated_on_its_recorded_layout_not_on_the_request_it_does_not_make() {
        let grouped = family_for_test("gfx120X-all", None);

        for channel in [TheRockChannel::Release, TheRockChannel::Nightly] {
            assert_eq!(
                gate_install_layout(channel, &grouped, None, Some(SourceLayout::Canonical))
                    .unwrap(),
                SourceLayout::Canonical,
                "{channel:?}"
            );
            assert_eq!(
                gate_install_layout(channel, &grouped, None, Some(SourceLayout::Next)).unwrap(),
                SourceLayout::Next,
                "{channel:?}"
            );
            // A fresh request with no recorded layout is still gated.
            assert!(gate_install_layout(channel, &grouped, None, None).is_err());
        }
    }

    #[test]
    fn next_layout_resolves_its_own_bases() {
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let next = resolve_source(TheRockChannel::Release, SourceLayout::Next);

        assert_eq!(
            next.wheel_index,
            "https://stable.repo.amd.com/rocm/whl-next"
        );
        assert_eq!(
            next.tarball_catalog,
            "https://stable.repo.amd.com/rocm/core/tarball/"
        );
        assert_eq!(next.layout.generation(), "next-v1");

        let nightly = resolve_source(TheRockChannel::Nightly, SourceLayout::Next);
        assert_eq!(
            nightly.wheel_index,
            "https://nightly.repo.amd.com/rocm/whl-next"
        );
        assert_eq!(
            nightly.tarball_catalog,
            "https://nightly.repo.amd.com/rocm/core/tarball/"
        );
        assert_eq!(nightly.layout.generation(), "next-v1");
    }

    /// A base override is a redirect to an arbitrary host, so naming the
    /// variable must not be enough on its own.
    #[test]
    fn base_overrides_are_inert_without_the_explicit_opt_in() {
        let default = THEROCK_NEXT_PIP_INDEX_BASE;
        let fixture = "http://127.0.0.1:9/whl";

        // Naming the variable is not enough; without the opt-in it is ignored.
        assert_eq!(select_base_override(false, Some(fixture), default), default);
        assert_eq!(
            select_base_override(true, Some(fixture), default),
            fixture,
            "an opted-in override must be honoured"
        );
        // A variable that is present but blank is not a redirect.
        assert_eq!(select_base_override(true, Some("   "), default), default);
        assert_eq!(select_base_override(true, None, default), default);
    }

    /// The test above only proves the pure decision rule; it never reads
    /// process environment, so it can't catch `env_override_base` or
    /// `resolve_source` failing to wire that rule to the real variables. The
    /// only thing that does today is `therock-next-06`, an e2e scenario gated
    /// `@nightly` because it has to reach the live default index — so on the
    /// blocking lane this trust boundary otherwise has zero coverage of the
    /// actual env-reading path. Exercise it here instead, with no network.
    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn env_override_is_ignored_end_to_end_without_the_opt_in() {
        let _guard = PROCESS_ENV_TEST_LOCK.lock().unwrap();
        let old_override = std::env::var_os("ROCM_CLI_THEROCK_NEXT_PIP_BASE");
        let old_allow = std::env::var_os("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE");
        unsafe {
            std::env::set_var("ROCM_CLI_THEROCK_NEXT_PIP_BASE", "http://127.0.0.1:9/whl");
            std::env::remove_var("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE");
        }

        let next = resolve_source(TheRockChannel::Release, SourceLayout::Next);

        unsafe {
            match old_override {
                Some(value) => std::env::set_var("ROCM_CLI_THEROCK_NEXT_PIP_BASE", value),
                None => std::env::remove_var("ROCM_CLI_THEROCK_NEXT_PIP_BASE"),
            }
            match old_allow {
                Some(value) => std::env::set_var("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE", value),
                None => std::env::remove_var("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE"),
            }
        }

        assert_eq!(
            next.wheel_index, THEROCK_NEXT_PIP_INDEX_BASE,
            "an override named without the trust opt-in must not redirect resolution"
        );
    }

    /// The requirement is read before `resolve_pip_runtime` runs, so it has to
    /// be derivable from the request alone, and it has to agree with the layout
    /// resolution will then use.
    #[test]
    fn requested_source_layout_reads_the_request_not_the_resolution() {
        let next_pin = RuntimeVersionSelector::Version("10.1.0".to_owned());
        assert_eq!(
            requested_source_layout(TheRockChannel::Release, None, Some(&next_pin)),
            SourceLayout::Next
        );

        let old_pin = RuntimeVersionSelector::Version("7.2.3".to_owned());
        assert_eq!(
            requested_source_layout(TheRockChannel::Release, None, Some(&old_pin)),
            SourceLayout::Canonical
        );
        assert_eq!(
            requested_source_layout(TheRockChannel::Release, None, None),
            SourceLayout::Next
        );
        assert_eq!(
            requested_source_layout(TheRockChannel::Nightly, None, None),
            SourceLayout::Next
        );

        // An update supplies its manifest's layout, which wins over the (absent)
        // selector so the runtime keeps the interpreter its stream needs.
        assert_eq!(
            requested_source_layout(TheRockChannel::Nightly, Some(SourceLayout::Canonical), None),
            SourceLayout::Canonical
        );
        assert_eq!(
            requested_source_layout(TheRockChannel::Release, Some(SourceLayout::Next), None),
            SourceLayout::Next
        );
    }
}
