// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Result, anyhow, bail};

use crate::discover_builds::{
    DiscoverVersion, VLLM_ROCM_DISCOVER_BUILD_TABLE, vllm_rocm_discover_build,
};

/// Known-good `(ROCm SDK version, vLLM version, ABI tag)` combinations for
/// `uv pip install vllm`, keyed by the ROCm SDK version recorded in the
/// runtime manifest (see [`crate::runtime::rocm_sdk_version_from_manifest`]).
///
/// Add a row only once wheels.vllm.ai actually publishes a build for that
/// ROCm SDK version — see `docs/vllm.md` for AMD's current ROCm X guidance
/// when this table has no matching row.
struct VllmRocmBuild {
    rocm_sdk_version: &'static str,
    vllm_version: &'static str,
    abi: &'static str,
}

const VLLM_ROCM_BUILD_TABLE: &[VllmRocmBuild] = &[VllmRocmBuild {
    rocm_sdk_version: "7.2.3",
    vllm_version: "0.26.0",
    abi: "rocm723",
}];

/// Prefix shared by every published vLLM ROCm wheel index.
///
/// Release indexes are `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`, which is
/// what lets [`vllm_rocm_build_from_index_url`] recover the build a custom
/// index serves and keep the requirement pinned to it.
const VLLM_ROCM_INDEX_PREFIX: &str = "https://wheels.vllm.ai/rocm";
/// The torchaudio release confirmed on hardware for `rocm_sdk_version`'s ROCm
/// `major.minor` line: the row's [`DiscoverVersion::Exact`] torchaudio, if it
/// has one. `None` for a line with no row, or whose row still matches a series,
/// because nothing has verified which build of that series works.
///
/// Read by `rocm install sdk` as well as by this crate's discovery, so the SDK
/// and its vLLM install pin the same torchaudio from this one table.
pub(crate) fn verified_torchaudio_release(rocm_sdk_version: &str) -> Option<&'static str> {
    match vllm_rocm_discover_build(rocm_sdk_version)?.torchaudio_version {
        DiscoverVersion::Exact(release) => Some(release),
        DiscoverVersion::Series(_) => None,
    }
}

/// Whether `recorded` (a runtime manifest's live `rocm_sdk.__version__` probe)
/// and `table_key` (a literal key in [`VLLM_ROCM_DISCOVER_BUILD_TABLE`]) share
/// a ROCm SDK major version, ignoring minor, patch, and any dev/pre-release
/// suffix. Major-only on purpose: it is broader than the `major.minor` row
/// lookup [`vllm_rocm_discover_build`] uses, so a 10.x line with no discover
/// row fails closed here instead of falling through to the default static
/// pin.
fn rocm_sdk_major_matches(recorded: &str, table_key: &str) -> bool {
    fn major(version: &str) -> Option<u64> {
        version.trim().split('.').next()?.parse().ok()
    }
    major(recorded).is_some() && major(recorded) == major(table_key)
}

/// Whether `recorded` (a runtime manifest's live `rocm_sdk.__version__` probe)
/// is the same release as `table_key` (a literal key in [`VLLM_ROCM_BUILD_TABLE`]),
/// ignoring any dev/pre-release suffix.
///
/// Real probes routinely carry a suffix the table keys never do (e.g.
/// `7.13.0a20260423`), and that suffix is not valid semver pre-release syntax
/// (no leading `-`), so comparing by exact string, or even by a semver parse,
/// would reject a matching release. Comparing by leading `major.minor.patch`
/// instead matches the release the table row actually covers.
fn rocm_sdk_version_matches(recorded: &str, table_key: &str) -> bool {
    fn release_triple(version: &str) -> Option<(u64, u64, u64)> {
        let mut parts = version.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch: String = parts
            .next()?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        Some((major, minor, patch.parse().ok()?))
    }
    release_triple(recorded).is_some_and(|version| Some(version) == release_triple(table_key))
}

/// How `install.rs`'s `install_vllm_with_uv` should install vLLM for a given target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VllmInstallRoute {
    /// Pin from [`VLLM_ROCM_BUILD_TABLE`] (or a caller-supplied index override).
    Static,
    /// Discover the current wheels for a [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] row.
    RocmDiscover,
}

/// Picks the install route for a vLLM install. A caller-supplied index
/// override always wins (it means the caller already knows exactly which
/// wheels to use), otherwise a known ROCm SDK version routes through
/// discovery, and everything else falls back to the static pin table.
pub(crate) fn vllm_install_route(
    index_override: Option<&str>,
    rocm_sdk_version: Option<&str>,
) -> VllmInstallRoute {
    if index_override.is_some() {
        return VllmInstallRoute::Static;
    }
    match rocm_sdk_version.and_then(vllm_rocm_discover_build) {
        Some(_) => VllmInstallRoute::RocmDiscover,
        None => VllmInstallRoute::Static,
    }
}

/// Wheel index and exact requirement for one `uv pip install vllm`.
///
/// The two are always derived from the same `<version>+<abi>` build, so the
/// install can never be pinned to something the index does not serve — and can
/// never be left unpinned, which would let the resolver fall through to PyPI's
/// non-ROCm build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VllmInstallTarget {
    pub(crate) index_url: String,
    pub(crate) requirement: String,
}

/// Index URL and requirement for the given runtime's recorded ROCm SDK version.
pub(crate) fn vllm_install_target(rocm_sdk_version: Option<&str>) -> Result<VllmInstallTarget> {
    resolve_vllm_install_target(
        std::env::var("ROCM_CLI_VLLM_ROCM_INDEX_URL").ok(),
        rocm_sdk_version,
    )
}

/// Resolve the wheel index and the exact requirement to install from it.
///
/// With `ROCM_CLI_VLLM_ROCM_INDEX_URL` set to a non-blank value, the build is
/// recovered from the URL when it has the published
/// `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>` shape — so pointing at another
/// release of the same index works *and* stays pinned, with no extra
/// configuration. A URL of any other shape cannot be pinned automatically, and
/// dropping the pin is not an option: the install passes `--extra-index-url`,
/// so PyPI stays in play and a bare `vllm` can silently resolve to the
/// non-ROCm build. Such a URL is therefore rejected rather than installed
/// unpinned.
///
/// Without an override, `rocm_sdk_version` (the ROCm SDK version recorded in
/// the target runtime's manifest) is looked up in [`VLLM_ROCM_BUILD_TABLE`].
/// A version with no row falls back to the table's first (default) row,
/// matching the single unconditional pin `rocm-cli` used before per-version
/// rows existed, rather than leaving an unrecognized version uninstallable,
/// *unless* the version's major release matches a
/// [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] row (see [`vllm_rocm_discover_build`]),
/// in which case guessing the default row's wheel would very likely install
/// an ABI-incompatible build, so that case fails closed instead. In practice
/// callers route such a version through discovery before ever reaching this
/// function (see [`vllm_install_route`]); this is a safety net for the case
/// where they don't.
fn resolve_vllm_install_target(
    index_override: Option<String>,
    rocm_sdk_version: Option<&str>,
) -> Result<VllmInstallTarget> {
    let index_override = index_override
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    if let Some(index_url) = index_override {
        let (version, abi) = vllm_rocm_build_from_index_url(&index_url).ok_or_else(|| {
            anyhow!(
                "cannot determine which vLLM build `{index_url}` serves: it is not a published \
                 release index of the form `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`. Installing \
                 an unpinned `vllm` instead would let the resolver fall back to PyPI's non-ROCm \
                 build, so point ROCM_CLI_VLLM_ROCM_INDEX_URL at a release index, or unset it to \
                 use the built-in one."
            )
        })?;

        return Ok(VllmInstallTarget {
            index_url,
            requirement: format!("vllm=={version}+{abi}"),
        });
    }

    let rocm_sdk_version = rocm_sdk_version.ok_or_else(|| {
        anyhow!(
            "cannot install vLLM: the target runtime's ROCm SDK version could not be determined \
             from its manifest, so no compatible vLLM build can be selected. Set \
             ROCM_CLI_VLLM_ROCM_INDEX_URL to a published release index to install anyway."
        )
    })?;

    let default_build = VLLM_ROCM_BUILD_TABLE
        .first()
        .ok_or_else(|| anyhow!("VLLM_ROCM_BUILD_TABLE has no default row"))?;
    let build = match VLLM_ROCM_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_version_matches(rocm_sdk_version, build.rocm_sdk_version))
    {
        Some(build) => build,
        None if VLLM_ROCM_DISCOVER_BUILD_TABLE
            .iter()
            .any(|build| rocm_sdk_major_matches(rocm_sdk_version, build.rocm_sdk_version)) =>
        {
            bail!(
                "cannot install vLLM: ROCm SDK version `{rocm_sdk_version}` has no static \
                 build pin, and its major release is only known to `rocm-cli` via live wheel \
                 discovery, not a static pin; guessing the default \
                 vllm=={}+{} build would very likely install an incompatible wheel. Set \
                 ROCM_CLI_VLLM_ROCM_INDEX_URL to a published release index to install anyway.",
                default_build.vllm_version,
                default_build.abi,
            );
        }
        None => default_build,
    };

    Ok(VllmInstallTarget {
        index_url: format!(
            "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
            build.vllm_version, build.abi
        ),
        requirement: format!("vllm=={}+{}", build.vllm_version, build.abi),
    })
}

/// Recover the `<version>+<abi>` build a published release index serves.
///
/// Returns `None` for anything that is not
/// `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`, including the rolling top-level
/// index, which is latest-only and therefore not a pin.
fn vllm_rocm_build_from_index_url(index_url: &str) -> Option<(String, String)> {
    let valid = |segment: &str| {
        !segment.is_empty()
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    };
    let rest = index_url
        .trim_end_matches('/')
        .strip_prefix(VLLM_ROCM_INDEX_PREFIX)?
        .strip_prefix('/')?;
    let (version, abi) = rest.split_once('/')?;
    (valid(version) && valid(abi)).then(|| (version.to_owned(), abi.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_recording_sdk_version(version: Option<&str>) -> crate::runtime::VllmRuntime {
        crate::runtime::VllmRuntime {
            runtime_id: "wheel-gfx1151".to_owned(),
            command: std::path::PathBuf::from("/rocm/runtimes/wheel/gfx1151/bin/vllm"),
            source: "managed_runtime_manifest:wheel-gfx1151".to_owned(),
            rocm_sdk_version: version.map(ToOwned::to_owned),
            ..crate::runtime::VllmRuntime::for_test()
        }
    }

    /// What every reader of the discover-build data decides for one recorded
    /// ROCm SDK version: the row the install lookup finds, the install route,
    /// whether launch-time env setup treats the runtime as a discover build, and
    /// how the static-pin fallback answers (`Some(true)` resolves a pin,
    /// `Some(false)` fails closed, `None` when no version is recorded).
    #[derive(Debug, PartialEq, Eq)]
    struct DiscoverReaders {
        row: Option<&'static str>,
        route: VllmInstallRoute,
        launch_treats_as_discover: bool,
        static_pin_resolves: Option<bool>,
    }

    fn discover_readers(version: Option<&str>) -> DiscoverReaders {
        DiscoverReaders {
            row: version
                .and_then(vllm_rocm_discover_build)
                .map(|build| build.rocm_sdk_version),
            route: vllm_install_route(None, version),
            launch_treats_as_discover: crate::runtime::runtime_has_discover_build(
                &runtime_recording_sdk_version(version),
            ),
            static_pin_resolves: version.map(|v| install_target(None, Some(v)).is_ok()),
        }
    }

    /// Install routing and launch-time env setup are two readers of the same
    /// discover-build data. For every table row, spelled the way a real probe
    /// spells it (exact, with a patch, with a dev suffix), both must agree that
    /// the runtime is a discover build, and the static-pin fallback must fail
    /// closed for it. For a version outside the table, both must agree it is not.
    #[test]
    fn every_discover_row_routes_installs_and_launch_env_the_same_way() {
        for build in VLLM_ROCM_DISCOVER_BUILD_TABLE {
            let key = build.rocm_sdk_version;
            let (major_minor, _) = key
                .rsplit_once('.')
                .expect("table key is major.minor.patch");
            for spelling in [
                key.to_owned(),
                format!("{major_minor}.7"),
                format!("{key}a20260822"),
                format!("{key}.dev0+abc"),
            ] {
                assert_eq!(
                    discover_readers(Some(&spelling)),
                    DiscoverReaders {
                        row: Some(key),
                        route: VllmInstallRoute::RocmDiscover,
                        launch_treats_as_discover: true,
                        static_pin_resolves: Some(false),
                    },
                    "readers disagree for {spelling}, a spelling of table row {key}"
                );
            }
        }

        // Outside the table: no row, static route, no discovery env. A 10.x line
        // with no row still fails closed instead of taking the 7.2.3 default
        // pin; other lines (and unparsable text) resolve that default.
        assert_eq!(
            discover_readers(Some("10.2.0")),
            DiscoverReaders {
                row: None,
                route: VllmInstallRoute::Static,
                launch_treats_as_discover: false,
                static_pin_resolves: Some(false),
            }
        );
        for outside in ["7.2.3", "999.0.0", "not-a-version"] {
            assert_eq!(
                discover_readers(Some(outside)),
                DiscoverReaders {
                    row: None,
                    route: VllmInstallRoute::Static,
                    launch_treats_as_discover: false,
                    static_pin_resolves: Some(true),
                },
                "readers disagree for {outside}, which is outside the table"
            );
        }
        assert_eq!(
            discover_readers(None),
            DiscoverReaders {
                row: None,
                route: VllmInstallRoute::Static,
                launch_treats_as_discover: false,
                static_pin_resolves: None,
            }
        );
        // `static_pin_resolves: None` above means no version was recorded; the
        // fallback itself fails for that case, with its own message.
        assert!(install_target(None, None).is_err());
    }

    fn install_target(
        index: Option<&str>,
        rocm_sdk_version: Option<&str>,
    ) -> Result<VllmInstallTarget> {
        resolve_vllm_install_target(index.map(ToOwned::to_owned), rocm_sdk_version)
    }
    #[test]
    fn vllm_install_target_resolves_a_known_rocm_sdk_version() {
        let build = VLLM_ROCM_BUILD_TABLE
            .first()
            .expect("build table has at least one row for this test to check");
        let target = install_target(None, Some(build.rocm_sdk_version))
            .expect("a table row resolves to a pinned target");
        assert_eq!(
            target.index_url,
            format!(
                "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
                build.vllm_version, build.abi
            )
        );
        assert_eq!(
            target.requirement,
            format!("vllm=={}+{}", build.vllm_version, build.abi)
        );

        let blank = install_target(Some("  "), Some(build.rocm_sdk_version))
            .expect("a blank override is ignored");
        assert_eq!(blank, target);
    }
    #[test]
    fn vllm_install_target_falls_back_to_the_default_row_for_an_unknown_rocm_sdk_version() {
        let default_build = VLLM_ROCM_BUILD_TABLE
            .first()
            .expect("build table has at least one row for this test to check");

        for unknown_version in ["999.0.0", "7.13.0a20260326", "not-a-version"] {
            let target = install_target(None, Some(unknown_version))
                .expect("an unrecognized version still resolves to the default pin");
            assert_eq!(
                target.requirement,
                format!("vllm=={}+{}", default_build.vllm_version, default_build.abi)
            );
        }
    }
    #[test]
    fn vllm_install_target_refuses_to_guess_a_static_pin_for_an_unmatched_discover_major() {
        // `10.1.0a20260822` shares a major with the `VLLM_ROCM_DISCOVER_BUILD_TABLE`
        // row but does not match any `VLLM_ROCM_BUILD_TABLE` row, so guessing the
        // ROCm 7.2.3 default here would install an incompatible wheel. Callers
        // route this version through discovery before reaching this function (see
        // `vllm_install_route`); this checks the fallback itself fails closed.
        let error = install_target(None, Some("10.1.0a20260822"))
            .expect_err("an unmatched version in a known discovery major must not guess")
            .to_string();
        assert!(error.contains("10.1.0a20260822"), "{error}");
        assert!(error.contains("ROCM_CLI_VLLM_ROCM_INDEX_URL"), "{error}");
    }
    #[test]
    fn vllm_install_target_fails_when_no_rocm_sdk_version_is_known() {
        let error = install_target(None, None)
            .expect_err("with no override and no version, nothing can be pinned")
            .to_string();
        assert!(error.contains("could not be determined"), "{error}");
    }
    #[test]
    fn vllm_install_target_stays_pinned_for_a_same_shape_index_override() {
        for (index, expected_url) in [
            (
                " https://wheels.vllm.ai/rocm/0.27.0/rocm730 ",
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730",
            ),
            (
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730/",
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730/",
            ),
        ] {
            let target = install_target(Some(index), None).expect("published index shape resolves");
            assert_eq!(target.index_url, expected_url);
            assert_eq!(target.requirement, "vllm==0.27.0+rocm730");
        }
    }
    #[test]
    fn vllm_install_target_fails_rather_than_unpinning_an_unknown_index() {
        for index in [
            "https://example.test/rocm/wheels",
            // Rolling latest-only index: no version/ABI to pin to.
            "https://wheels.vllm.ai/rocm/",
            "https://wheels.vllm.ai/rocm/0.27.0",
            "https://wheels.vllm.ai/rocm//rocm730",
            "https://wheels.vllm.ai/rocm/0.27.0/rocm 730",
        ] {
            let error = install_target(Some(index), None)
                .expect_err("an unpinnable index must fail")
                .to_string();
            assert!(
                error.contains(index.trim()),
                "error for {index} should quote the index URL: {error}"
            );
            assert!(
                error.contains(VLLM_ROCM_INDEX_PREFIX),
                "error for {index} should show the expected index shape: {error}"
            );
        }
    }
    #[test]
    fn every_build_table_row_has_a_self_consistent_index_and_requirement() {
        for build in VLLM_ROCM_BUILD_TABLE {
            let index_url = format!(
                "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
                build.vllm_version, build.abi
            );
            assert_eq!(
                vllm_rocm_build_from_index_url(&index_url),
                Some((build.vllm_version.to_owned(), build.abi.to_owned())),
                "row for ROCm SDK {} must parse back to its own build",
                build.rocm_sdk_version
            );
        }
    }
    #[test]
    fn rocm_sdk_version_matches_ignores_dev_suffixes_and_rejects_other_releases() {
        assert!(rocm_sdk_version_matches("10.0.0", "10.0.0"));
        assert!(rocm_sdk_version_matches("7.13.0a20260423", "7.13.0"));
        assert!(rocm_sdk_version_matches("7.2.3.dev0+abc", "7.2.3"));
        assert!(!rocm_sdk_version_matches("7.14.1", "7.2.3"));
        assert!(!rocm_sdk_version_matches("garbage", "7.2.3"));
    }
    #[test]
    fn rocm_sdk_major_matches_ignores_minor_patch_and_dev_suffixes() {
        assert!(rocm_sdk_major_matches("10.0.0", "10.0.0"));
        assert!(rocm_sdk_major_matches("10.1.0a20260822", "10.0.0"));
        assert!(rocm_sdk_major_matches("10.99.7.dev0+abc", "10.0.0"));
        assert!(!rocm_sdk_major_matches("7.13.0", "10.0.0"));
        assert!(!rocm_sdk_major_matches("garbage", "10.0.0"));
    }
    #[test]
    fn vllm_install_route_prefers_an_index_override_even_for_a_discover_version() {
        assert_eq!(
            vllm_install_route(Some("https://example.test/rocm"), Some("10.0.0")),
            VllmInstallRoute::Static
        );
    }
    #[test]
    fn vllm_install_route_discovers_for_a_known_discover_version() {
        assert_eq!(
            vllm_install_route(None, Some("10.0.0")),
            VllmInstallRoute::RocmDiscover
        );
        // A same-major, different-minor/patch nightly must still route
        // through discovery rather than falling back to the static table.
        assert_eq!(
            vllm_install_route(None, Some("10.1.0a20260822")),
            VllmInstallRoute::RocmDiscover
        );
    }
    #[test]
    fn vllm_install_route_falls_back_to_static_for_unknown_or_missing_versions() {
        assert_eq!(
            vllm_install_route(None, Some("7.2.3")),
            VllmInstallRoute::Static
        );
        assert_eq!(vllm_install_route(None, None), VllmInstallRoute::Static);
    }
    /// Only an exact torchaudio pin counts as verified: the 10.1 row's
    /// `2.11.0.2` is, the 10.0 row's `2.11` series is not, and a line with no
    /// row has nothing to offer. `rocm install sdk` reads this same answer.
    #[test]
    fn verified_torchaudio_release_is_only_an_exact_row_pin() {
        assert_eq!(verified_torchaudio_release("10.1.0"), Some("2.11.0.2"));
        assert_eq!(
            verified_torchaudio_release("10.1.0a20260928"),
            Some("2.11.0.2")
        );
        assert_eq!(verified_torchaudio_release("10.0.0"), None);
        assert_eq!(verified_torchaudio_release("10.2.0"), None);
        assert_eq!(verified_torchaudio_release("7.2.3"), None);
    }
    /// An unknown 10.x line fails closed rather than silently resolving the
    /// static table's 7.2.3 pin, which is the whole point of keeping
    /// [`rocm_sdk_major_matches`] alongside the narrower row lookup.
    #[test]
    fn unknown_rocm10_line_still_fails_closed() {
        let err = install_target(None, Some("10.2.0")).expect_err("10.2.0 has no vLLM build");
        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("7.2.3"),
            "should not fall back to the static pin: {rendered}"
        );
    }
}
