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
/// ROCm 10's aggregate pip index. Same simple-index shape the canonical
/// multi-arch stream publishes, on a separate host, with its own package
/// generation; it is not a mirror of, or a fallback for, the canonical index.
const THEROCK_NEXT_PIP_INDEX_BASE: &str = "https://stable.repo.amd.com/rocm/whl-next";
/// ROCm 10's tarball catalog. Same scrapeable listing the canonical catalog
/// uses, but it publishes non-release sibling archives beside the real dist
/// archive (see `select_tarball_candidate` in `resolve/tarball.rs`).
const THEROCK_NEXT_TARBALL_BASE: &str = "https://stable.repo.amd.com/rocm/core/tarball/";
pub(super) const THEROCK_SOURCE_LAYOUT_GENERATION: &str = "multi-arch-v2";
/// Recorded in a [`WheelRuntimeComposition`](crate::runtime_registry::WheelRuntimeComposition)
/// installed from the ROCm 10 layout, so a later update reads back the layout
/// that produced the runtime instead of guessing it from the version.
const THEROCK_NEXT_LAYOUT_GENERATION: &str = "next-v1";
/// The first ROCm major version published only in the next layout. A pin at or
/// past it is the one request the canonical stream provably cannot serve, and
/// therefore the only thing that selects [`SourceLayout::Next`].
pub(super) const THEROCK_NEXT_MIN_MAJOR: u32 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TheRockChannel {
    Release,
    Nightly,
}

/// Which published hosting layout a resolution reads from.
///
/// Orthogonal to [`TheRockChannel`]. `Canonical` is the release/nightly stream
/// the CLI has always installed and remains the answer for every request that
/// does not explicitly ask for something else. `Next` is ROCm 10's separately
/// hosted layout — an extension, never a fallback: nothing degrades into it and
/// nothing degrades out of it, so a canonical install resolves exactly the URLs,
/// device target, and package specs it resolved before the layout existed.
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
///
/// The next layout is hosted per-layout rather than per-channel: it has no
/// nightly stream, and [`select_source_layout`] refuses to reach it from one, so
/// there is nothing here to branch on.
pub(super) fn resolve_source(channel: TheRockChannel, layout: SourceLayout) -> ResolvedSource {
    match layout {
        SourceLayout::Canonical => match channel {
            TheRockChannel::Release => ResolvedSource {
                wheel_index: env_override_base(
                    "ROCM_CLI_THEROCK_RELEASE_PIP_BASE",
                    THEROCK_RELEASE_PIP_INDEX_BASE,
                ),
                tarball_catalog: env_override_base(
                    "ROCM_CLI_THEROCK_RELEASE_TARBALL_BASE",
                    THEROCK_RELEASE_TARBALL_BASE,
                ),
                layout,
            },
            TheRockChannel::Nightly => ResolvedSource {
                wheel_index: env_override_base(
                    "ROCM_CLI_THEROCK_NIGHTLY_PIP_BASE",
                    THEROCK_NIGHTLY_PIP_INDEX_BASE,
                ),
                tarball_catalog: env_override_base(
                    "ROCM_CLI_THEROCK_NIGHTLY_TARBALL_BASE",
                    THEROCK_NIGHTLY_TARBALL_BASE,
                ),
                layout,
            },
        },
        SourceLayout::Next => ResolvedSource {
            wheel_index: env_override_base(
                "ROCM_CLI_THEROCK_NEXT_PIP_BASE",
                THEROCK_NEXT_PIP_INDEX_BASE,
            ),
            tarball_catalog: env_override_base(
                "ROCM_CLI_THEROCK_NEXT_TARBALL_BASE",
                THEROCK_NEXT_TARBALL_BASE,
            ),
            layout,
        },
    }
}

/// Whether `selector` names a ROCm release that only the next layout publishes.
///
/// An exact *stable* pin is the only signal that qualifies. A build date names
/// a nightly build, an unparseable string names nothing this CLI can reason
/// about, and neither is evidence that the canonical stream cannot serve the
/// request. Nor is a pinned prerelease of a future major: the canonical
/// nightly stream already serves those (see
/// `nightly_accepts_future_prerelease_major_without_cli_changes`), so gating
/// on major alone would route a nightly alpha pin like `10.1.0a20260822` into
/// a refusal the release-channel retry it suggests could never satisfy.
pub(super) fn next_layout_requested(selector: &RuntimeVersionSelector) -> bool {
    let RuntimeVersionSelector::Version(version) = selector else {
        return false;
    };
    parse_version(version).is_some_and(|parsed| {
        parsed.stage == VersionStage::Stable && parsed.major >= THEROCK_NEXT_MIN_MAJOR
    })
}

/// The layout an install must read from, and the refusal when it cannot.
///
/// Only an explicit pin at ROCm >= [`THEROCK_NEXT_MIN_MAJOR`] selects `Next`,
/// because that is the one request the canonical stream provably cannot serve.
/// No selector, a build date, an unparseable pin, or an older pin all stay
/// canonical, so nothing that resolves today starts resolving somewhere else.
/// A pin that does reach `Next` still needs an exact arch, because that layout
/// picks its device payload by exact arch and a grouped family names none.
pub(super) fn select_source_layout(
    channel: TheRockChannel,
    family_resolution: &FamilyResolution,
    version_selector: Option<&RuntimeVersionSelector>,
) -> Result<SourceLayout> {
    let Some(selector) = version_selector.filter(|selector| next_layout_requested(selector)) else {
        return Ok(SourceLayout::Canonical);
    };
    let RuntimeVersionSelector::Version(version) = selector else {
        return Ok(SourceLayout::Canonical);
    };
    let major = parse_version(version).map_or(THEROCK_NEXT_MIN_MAJOR, |parsed| parsed.major);
    if !matches!(channel, TheRockChannel::Release) {
        bail!(
            "ROCm {major} and newer is published only on the release channel; re-run `rocm install sdk --channel release --version {version}`"
        );
    }
    if family_resolution.raw_arch.is_none() {
        bail!(
            "installing ROCm {major} requires an exact GPU arch, but the resolved target family `{}` names a group of them.\n\
             ROCm {major} selects its device payload by exact arch, so re-run with the one this host has, for example `rocm install sdk --version {version} --family gfx1200`.\n\n{}",
            family_resolution.family,
            detect_host_gpu_diagnostics()
        );
    }
    Ok(SourceLayout::Next)
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
    layout_override: Option<SourceLayout>,
    version_selector: Option<&RuntimeVersionSelector>,
) -> SourceLayout {
    match layout_override {
        Some(layout) => layout,
        None if version_selector.is_some_and(next_layout_requested) => SourceLayout::Next,
        None => SourceLayout::Canonical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PROCESS_ENV_TEST_LOCK;
    use crate::therock::test_support::family_for_test;

    /// Everything a user can ask for today must keep resolving the canonical
    /// stream. The next layout is an extension, so no existing request may start
    /// answering from somewhere else.
    #[test]
    fn only_a_rocm_10_pin_leaves_the_canonical_layout() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));
        let build_date = RuntimeVersionSelector::build_date("2026-06-05").unwrap();
        let older_pin = RuntimeVersionSelector::version("7.14.0").unwrap();
        let unparseable = RuntimeVersionSelector::version("main").unwrap();

        for selector in [
            None,
            Some(&build_date),
            Some(&older_pin),
            Some(&unparseable),
        ] {
            assert_eq!(
                select_source_layout(TheRockChannel::Release, &exact, selector).unwrap(),
                SourceLayout::Canonical,
                "selector {selector:?} must stay canonical"
            );
        }

        let pin = RuntimeVersionSelector::version("10.0.0").unwrap();
        assert_eq!(
            select_source_layout(TheRockChannel::Release, &exact, Some(&pin)).unwrap(),
            SourceLayout::Next
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
    /// build) is not evidence the canonical stream can't serve it — only a
    /// *stable* pin is. Gating on major alone would refuse this pin and then
    /// suggest a release-channel retry that the stable-version filter would
    /// also reject, leaving no working path.
    #[test]
    fn nightly_prerelease_pin_of_a_future_major_stays_canonical() {
        let exact = family_for_test("gfx120X-all", Some("gfx1200"));
        let pin = RuntimeVersionSelector::version("10.1.0a20260822").unwrap();

        assert_eq!(
            select_source_layout(TheRockChannel::Nightly, &exact, Some(&pin)).unwrap(),
            SourceLayout::Canonical
        );
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
    /// be derivable from the request alone. Only an explicit stable ROCm >= 10
    /// pin reaches the next layout; a build-date pin, an older pin, or no
    /// selector at all must stay canonical so nothing that installs today
    /// starts demanding a different interpreter.
    #[test]
    fn requested_source_layout_reads_the_request_not_the_resolution() {
        let next_pin = RuntimeVersionSelector::Version("10.1.0".to_owned());
        assert_eq!(
            requested_source_layout(None, Some(&next_pin)),
            SourceLayout::Next
        );

        let old_pin = RuntimeVersionSelector::Version("7.2.3".to_owned());
        assert_eq!(
            requested_source_layout(None, Some(&old_pin)),
            SourceLayout::Canonical
        );
        assert_eq!(requested_source_layout(None, None), SourceLayout::Canonical);

        // An update supplies its manifest's layout, which wins over the (absent)
        // selector so the runtime keeps the interpreter its stream needs.
        assert_eq!(
            requested_source_layout(Some(SourceLayout::Next), None),
            SourceLayout::Next
        );
    }
}
