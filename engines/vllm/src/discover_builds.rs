// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// Whether `recorded` and `table_key` name the same ROCm `major.minor` line,
/// ignoring patch and any dev/pre-release suffix. See
/// [`vllm_rocm_discover_build`] for why the line, not the exact release, is
/// what a discover row covers. Lives here, beside the table, so that
/// `runtime.rs` and `wheel_builds.rs` can both match against it without either
/// importing the other's discover-build code.
pub(crate) fn rocm_sdk_series_matches(recorded: &str, table_key: &str) -> bool {
    fn series(version: &str) -> Option<(u64, u64)> {
        let mut parts = version.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor: String = parts
            .next()?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        Some((major, minor.parse().ok()?))
    }
    series(recorded).is_some_and(|version| Some(version) == series(table_key))
}

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
/// pinned in `wheel_builds::VLLM_ROCM_BUILD_TABLE`.
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
    /// [`verified_torchaudio_release`](crate::wheel_builds::verified_torchaudio_release): set one only for a build confirmed to
    /// load against the row's torch on real hardware, never as a guess.
    pub(crate) torchaudio_version: DiscoverVersion,
}

/// Every ROCm SDK `major.minor` line the engine installs by live wheel
/// discovery, one row each, and the only list of those versions.
///
/// Three readers follow it with nothing else to keep in step: install routing
/// ([`vllm_rocm_discover_build`], through `wheel_builds::vllm_install_route`),
/// the static-pin fallback that fails closed for a 10.x line with no row
/// (`wheel_builds::vllm_install_target`), and launch-time environment setup
/// (`runtime::runtime_has_discover_build`). Adding, removing or reordering a
/// row therefore changes all three together;
/// `discover_build_table_pins_the_versions_it_serves_in_order` makes that a
/// reviewed edit.
pub(crate) const VLLM_ROCM_DISCOVER_BUILD_TABLE: &[VllmRocmDiscoverBuild] = &[
    VllmRocmDiscoverBuild {
        rocm_sdk_version: "10.0.0",
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
        rocm_sdk_version: "10.1.0",
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
/// closed in [`crate::wheel_builds::vllm_install_target`] rather than falling back to a
/// stale static pin.
pub(crate) fn vllm_rocm_discover_build(
    rocm_sdk_version: &str,
) -> Option<&'static VllmRocmDiscoverBuild> {
    VLLM_ROCM_DISCOVER_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_series_matches(rocm_sdk_version, build.rocm_sdk_version))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
