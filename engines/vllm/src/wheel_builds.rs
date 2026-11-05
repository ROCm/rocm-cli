// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Result, anyhow, bail};

use crate::runtime::{ROCM_SDK_10_0, ROCM_SDK_10_1, rocm_sdk_series_matches};

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
/// How a [`VllmRocmDiscoverBuild`] row constrains the release discovery may
/// resolve for one package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoverVersion {
    /// `{pkg}=={series}.*`: the newest release in the series.
    Series(&'static str),
    /// `{pkg}=={release}`: exactly this release. Under PEP 440 a specifier
    /// without a local version still matches any `+rocm…` local version of
    /// it, but not a longer release such as `{release}.2`. Needed where AMD
    /// publishes rebuilds as extra release segments (torchaudio
    /// `2.11.0`/`2.11.0.2`/`2.11.0.3`, each paired with a different torch), so
    /// a series match would pick the rebuild for the wrong torch.
    Exact(&'static str),
}

impl DiscoverVersion {
    /// The requirement handed to `uv` for `pkg`.
    pub(crate) fn requirement(self, pkg: &str) -> String {
        match self {
            Self::Series(series) => format!("{pkg}=={series}.*"),
            Self::Exact(release) => format!("{pkg}=={release}"),
        }
    }
}

/// A ROCm SDK version whose vLLM/flash-attn/amd-aiter wheels aren't published
/// under a fixed filename (AMD rotates the dev-tag suffix constantly), so the
/// exact wheel must be discovered from the index at install time instead of
/// pinned in [`VLLM_ROCM_BUILD_TABLE`].
pub(crate) struct VllmRocmDiscoverBuild {
    pub(crate) rocm_sdk_version: &'static str,
    /// PEP 425 interpreter tag this row's wheels are published for. AMD builds
    /// ROCm 10.x `vllm`, `flash-attn` and `amd-aiter` for exactly one CPython
    /// version, so a venv on any other interpreter resolves nothing at all.
    pub(crate) python_tag: &'static str,
    /// Index publishing this row's rotating-dev-tag vLLM/flash-attn/amd-aiter
    /// wheels. Per-row because ROCm 10.1 frameworks are staged on a different
    /// host than 10.0's.
    pub(crate) vllm_index_url: &'static str,
    /// Index publishing this row's torch build.
    pub(crate) torch_index_url: &'static str,
    pub(crate) vllm_version_prefix: &'static str,
    pub(crate) flash_attn_version_prefix: &'static str,
    pub(crate) amd_aiter_version_prefix: &'static str,
    pub(crate) torch_version_prefix: &'static str,
    /// torchvision's version prefix is always the ROCm "Next" torch minor
    /// plus 15 (torch 2.12 pairs with torchvision 0.27); torchaudio has no
    /// such rule and is hardcoded per row to whatever AMD has actually
    /// published for it. Both are discovered fresh alongside torch rather
    /// than trusted from a pre-install snapshot, so the triplet installed
    /// here is always coherent with the torch version this row actually pins
    /// (see `install_vllm_rocm10_discover`).
    pub(crate) torchvision_version_prefix: &'static str,
    /// [`DiscoverVersion::Exact`] wherever the row's index publishes more than
    /// one rebuild of a torchaudio series, which `whl-next` does (`2.11.0`,
    /// `2.11.0.2`, `2.11.0.3`, each paired with a different torch): a series
    /// match takes the newest, i.e. the rebuild for some *other* row's torch.
    /// [`DiscoverVersion::Series`] is only safe on an index that serves one
    /// build per series.
    ///
    /// An [`DiscoverVersion::Exact`] torchaudio is also the hardware-verified
    /// build for the row's ROCm line, and `rocm install sdk` pins it too, via
    /// [`verified_torchaudio_release`]: set one only for a build confirmed to
    /// load against the row's torch on real hardware, never as a guess.
    pub(crate) torchaudio_version: DiscoverVersion,
}

const VLLM_ROCM_DISCOVER_BUILD_TABLE: &[VllmRocmDiscoverBuild] = &[
    VllmRocmDiscoverBuild {
        rocm_sdk_version: ROCM_SDK_10_0,
        python_tag: "cp314",
        vllm_index_url: "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/",
        torch_index_url: "https://stable.repo.amd.com/rocm/whl-next/",
        vllm_version_prefix: "0.27",
        flash_attn_version_prefix: "2.8",
        amd_aiter_version_prefix: "0.1",
        torch_version_prefix: "2.12",
        torchvision_version_prefix: "0.27",
        // Still a series match, unlike the 10.1 row below, and unverified
        // against `whl-next` now that both rows read from it: that index is
        // flat across ROCm lines, so a series match here can resolve a
        // `+rocm10.1.0` wheel, which `ensure_rocm_local_version_matches`
        // rejects. Left as-is rather than pinned blind -- which torchaudio
        // rebuild pairs with 10.0's torch 2.12 needs a live 10.0 box to
        // establish, and this row has no coverage on the self-hosted lanes.
        // `rocm install sdk` reads this field too: see `torchaudio_version`.
        torchaudio_version: DiscoverVersion::Series("2.11"),
    },
    VllmRocmDiscoverBuild {
        rocm_sdk_version: ROCM_SDK_10_1,
        python_tag: "cp314",
        vllm_index_url: "https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm/",
        // Torch stack from `whl-next`, not staging: it is the index
        // `rocm install sdk` resolves the 10.1 SDK's own `+rocm10.1.0` torch
        // stack from, so vLLM realigns onto the same builds rather than
        // staging's `rc` ones.
        torch_index_url: "https://stable.repo.amd.com/rocm/whl-next/",
        vllm_version_prefix: "0.29",
        flash_attn_version_prefix: "2.8",
        amd_aiter_version_prefix: "0.1",
        // 2.13, not 2.12: AMD's own `rocm/vllm:rocm10.1.0_..._vllm-0.29.0`
        // image (whose Dockerfile build args pin `TORCH_VERSION=2.13.0`,
        // `TORCHVISION_VERSION=0.28.0`) builds this exact vllm release
        // against torch 2.13, not 2.12. `vllm` 0.29's own `_rocm_C.abi3.so`
        // needs a `c10::NotImplementedError` constructor symbol torch
        // 2.12.0+rocm10.1.0 does not export; 2.13 is published on this same
        // `whl-next` index and matches what vllm was actually built against.
        torch_version_prefix: "2.13",
        torchvision_version_prefix: "0.28",
        // Exact, not `2.11.*`: `whl-next` publishes three `+rocm10.1.0`
        // torchaudio builds, `2.11.0`, `2.11.0.2` and `2.11.0.3`. `2.11.0` is
        // a CPU-only rebuild (no `libamdhip64`/`libc10_hip`/`libtorch_hip`
        // in its `.so`s) that predates torch's stable-ABI split and does not
        // reference the `torch_exception_get_what` / `torch_new_stable_ivalue`
        // symbols torch 2.13 exports. `2.11.0.2` and `2.11.0.3` are both HIP
        // builds against that same stable ABI (confirmed on a live Strix Halo
        // box: torch 2.13.0+rocm10.1.0 exports both symbols as `T`, and
        // torchaudio 2.11.0.2 imports them as `U`); a series match would
        // resolve the newest, `2.11.0.3`, so pin the lower of the two
        // confirmed-working builds instead. `rocm install sdk` pins this same
        // build for the 10.1 line: see `torchaudio_version`.
        torchaudio_version: DiscoverVersion::Exact("2.11.0.2"),
    },
];

/// Looks up the discovery build recipe for a ROCm SDK version, if any.
///
/// Matched on `major.minor`, ignoring patch and any dev/pre-release suffix. A
/// discover row is a live resolver recipe rather than one pinned filename, so
/// it does span a release line, but only a `major.minor` one: 10.0 and 10.1
/// publish genuinely different wheels (vLLM `0.27` from
/// `rocm.frameworks.amd.com` versus `0.29` from the staging host), so a
/// major-only row would serve one SDK's wheels to the other. Patch is still
/// ignored, because AMD does rotate the patch and dev tag within a line
/// (`whl-multi-arch/torch/` carries `+rocm7.13.0`, `+rocm7.14.0` and
/// `+rocm7.14.1` as coexisting builds) and a row keyed to an exact string
/// would silently stop matching. An unknown line matches nothing and fails
/// closed in [`resolve_vllm_install_target`] rather than falling back to a
/// stale static pin.
pub(crate) fn vllm_rocm_discover_build(
    rocm_sdk_version: &str,
) -> Option<&'static VllmRocmDiscoverBuild> {
    VLLM_ROCM_DISCOVER_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_series_matches(rocm_sdk_version, build.rocm_sdk_version))
}

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
    use crate::runtime::ROCM_DISCOVER_BUILD_VERSIONS;

    #[test]
    fn vllm_rocm_discover_build_table_matches_the_discover_version_list() {
        // `ROCM_DISCOVER_BUILD_VERSIONS` (runtime.rs) and
        // `VLLM_ROCM_DISCOVER_BUILD_TABLE` (here) are two separate pieces of data that
        // must name the same ROCm SDK versions — the list drives launch-time env setup
        // (`apply_therock_env`), the table drives install routing, and a row present in
        // only one of them means a runtime gets installed one way and launched another.
        // See the doc comment on `ROCM_DISCOVER_BUILD_VERSIONS` for why nothing but this
        // test enforces the agreement.
        for build in VLLM_ROCM_DISCOVER_BUILD_TABLE {
            assert!(
                ROCM_DISCOVER_BUILD_VERSIONS.contains(&build.rocm_sdk_version),
                "{} is a VLLM_ROCM_DISCOVER_BUILD_TABLE row but missing from \
                 ROCM_DISCOVER_BUILD_VERSIONS; launch-time env setup would silently skip it",
                build.rocm_sdk_version
            );
        }
        for version in ROCM_DISCOVER_BUILD_VERSIONS {
            assert!(
                VLLM_ROCM_DISCOVER_BUILD_TABLE
                    .iter()
                    .any(|build| build.rocm_sdk_version == *version),
                "{version} is in ROCM_DISCOVER_BUILD_VERSIONS but has no \
                 VLLM_ROCM_DISCOVER_BUILD_TABLE row; install routing has nothing to install \
                 for a runtime launch-time env setup now treats as a discover build"
            );
        }
    }

    /// The ROCm SDK versions the discover-build table serves, in table order.
    ///
    /// Pinned on purpose: adding, removing or reordering a row changes which
    /// runtimes are installed by discovery and which get the discovery launch
    /// environment, so it has to show up as a reviewed edit of this list
    /// rather than slip through a test that only compares the data with itself.
    #[test]
    fn discover_build_table_pins_the_versions_it_serves_in_order() {
        let versions: Vec<&str> = VLLM_ROCM_DISCOVER_BUILD_TABLE
            .iter()
            .map(|build| build.rocm_sdk_version)
            .collect();
        assert_eq!(
            versions,
            ["10.0.0", "10.1.0"],
            "the discover-build table's versions or order changed: install routing, the \
             launch environment and the fail-closed static fallback all follow it"
        );
    }

    fn runtime_recording_sdk_version(version: Option<&str>) -> crate::runtime::VllmRuntime {
        crate::runtime::VllmRuntime {
            runtime_id: "wheel-gfx1151".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: std::path::PathBuf::from("/rocm/runtimes/wheel/gfx1151/bin/vllm"),
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:wheel-gfx1151".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: version.map(ToOwned::to_owned),
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
    fn vllm_rocm_discover_build_looks_up_known_and_unknown_versions() {
        assert!(vllm_rocm_discover_build("10.0.0").is_some());
        // AMD tags preview wheels with the real target release (verified via
        // whl-multi-arch/torch/'s coexisting +rocm7.13.0/7.14.0/7.14.1
        // builds), so a pre-release suffix on a known major.minor must still
        // match its row (here the dedicated 10.1.0 one), not just an exact
        // version string.
        assert!(vllm_rocm_discover_build("10.1.0a20260822").is_some());
        assert!(vllm_rocm_discover_build("999.0.0").is_none());
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
    /// A series renders with the `.*` that lets `uv` take the newest release
    /// in it; an exact release renders bare, so it cannot.
    #[test]
    fn discover_version_renders_series_and_exact_requirements() {
        assert_eq!(
            DiscoverVersion::Series("2.11").requirement("torchaudio"),
            "torchaudio==2.11.*"
        );
        // No trailing `.*`: that would also admit the `2.11.0.2`/`2.11.0.3`
        // rebuilds an exact pin exists to exclude.
        assert_eq!(
            DiscoverVersion::Exact("2.11.0").requirement("torchaudio"),
            "torchaudio==2.11.0"
        );
    }
    /// The 10.1 row must reach the staging host for vLLM, which is only
    /// published there, but takes its torch stack from `whl-next` just like
    /// 10.0, the same index `rocm install sdk` resolves its own
    /// `+rocm10.1.0` torch stack from (though the SDK's resolve and this
    /// row's discovery are independent: this row pins its own version
    /// prefixes rather than matching whatever the SDK happened to install).
    /// Its torchaudio is pinned exactly: a `2.11.*` match resolves the
    /// newest rebuild, `2.11.0.3`; `2.11.0.2` is the lower of the two builds
    /// confirmed (on a live Strix Halo box) to load against torch 2.13
    /// (therock-next-09). 10.0 keeps the production frameworks index for
    /// vLLM. The live half of this is therock-next-09.
    #[test]
    fn discover_rows_select_their_own_indexes() {
        let ten_zero = vllm_rocm_discover_build("10.0.0").expect("10.0.0 has a discover row");
        assert_eq!(
            ten_zero.vllm_index_url,
            "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/"
        );
        assert_eq!(
            ten_zero.torch_index_url,
            "https://stable.repo.amd.com/rocm/whl-next/"
        );
        assert_eq!(ten_zero.vllm_version_prefix, "0.27");

        // A real probe carries a patch and dev suffix the table key never does.
        let ten_one = vllm_rocm_discover_build("10.1.0a20260928").expect("10.1 has a discover row");
        assert_eq!(
            ten_one.vllm_index_url,
            "https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm/"
        );
        assert_eq!(
            ten_one.torch_index_url,
            "https://stable.repo.amd.com/rocm/whl-next/"
        );
        assert_eq!(ten_one.vllm_version_prefix, "0.29");
        // 2.13/0.28, not 2.12/0.27: the pair AMD's own
        // `rocm/vllm:...vllm-0.29.0` image builds this vllm release against.
        assert_eq!(ten_one.torch_version_prefix, "2.13");
        assert_eq!(ten_one.torchvision_version_prefix, "0.28");
        // `whl-next` carries `2.11.0`, `2.11.0.2` and `2.11.0.3` for
        // `+rocm10.1.0`; `2.11.0` is CPU-only, and a series match would
        // resolve the newest, `2.11.0.3`, not the confirmed-working
        // `2.11.0.2`, so this must stay exact and never a series.
        assert_eq!(
            ten_one.torchaudio_version,
            DiscoverVersion::Exact("2.11.0.2")
        );

        // An unpublished line must not borrow another line's wheels.
        assert!(vllm_rocm_discover_build("10.2.0").is_none());
        assert!(vllm_rocm_discover_build("7.2.3").is_none());
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
