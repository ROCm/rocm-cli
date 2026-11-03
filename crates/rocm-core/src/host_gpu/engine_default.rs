// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which serve engine a host defaults to: the platform fallback, and the vLLM
//! preference for the TheRock families that serve through it.

use super::HostGpuSummary;
use crate::gfx_target::normalize_therock_family;

pub const fn default_engine_for_platform() -> &'static str {
    "lemonade"
}

/// The engine this host serves on by default, absent an explicit choice.
///
/// [`default_engine_for_platform`] alone answers "what does this OS fall back
/// to", which is not the same question: on Instinct data-center parts serving
/// goes through vLLM, and reporting the platform constant there contradicts what
/// `serve` actually selects. Use this wherever the CLI *tells the user* what the
/// default engine is; `default_engine_for_platform` remains correct as the
/// last-resort fallback once GPU and recipe preferences have been exhausted.
///
/// A value the user configured still outranks this — callers that have a
/// configured engine must prefer it, mirroring `select_serve_engine`.
#[must_use]
pub fn default_engine_for_host(summary: &HostGpuSummary) -> &'static str {
    preferred_serve_engine_for_host_gpu_summary(summary).unwrap_or_else(default_engine_for_platform)
}

const VLLM_PREFERRED_THEROCK_FAMILIES: &[&str] = &["gfx906", "gfx908", "gfx90a"];

pub fn preferred_serve_engine_for_host_gpu_summary(
    summary: &HostGpuSummary,
) -> Option<&'static str> {
    // The vLLM engine adapter bails out on native Windows builds, so never prefer it
    // there. WSL builds as a Linux target and therefore remains eligible.
    if cfg!(windows) {
        return None;
    }
    preferred_serve_engine_for_therock_family(
        summary
            .therock_family
            .as_deref()
            .or(summary.gfx_target.as_deref()),
    )
}

fn preferred_serve_engine_for_therock_family(family: Option<&str>) -> Option<&'static str> {
    let family = family?.trim();
    if family.is_empty() {
        return None;
    }

    let family = normalize_therock_family(family)
        .as_deref()
        .unwrap_or(family)
        .to_ascii_lowercase();
    if family.ends_with("-dcgpu")
        || VLLM_PREFERRED_THEROCK_FAMILIES
            .iter()
            .any(|candidate| *candidate == family)
    {
        Some("vllm")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_engine_is_always_usable_on_windows() {
        if cfg!(windows) {
            assert_eq!(default_engine_for_platform(), "lemonade");
        }
    }

    #[test]
    fn instinct_dcgpu_family_prefers_vllm() {
        // On Instinct data-center GPUs (TheRock `*-dcgpu` families, e.g. the
        // MI300X's gfx94X-dcgpu) the default serving engine is vLLM. This is the
        // GPU-family preference the serve engine selection honors before falling
        // back to a recipe's own preferred engine. vLLM is Linux-only, so the
        // preference does not apply on native Windows.
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        let preferred = preferred_serve_engine_for_host_gpu_summary(&summary);
        if cfg!(windows) {
            assert_eq!(preferred, None, "vLLM is not preferred on native Windows");
        } else {
            assert_eq!(preferred, Some("vllm"));
        }
    }

    #[test]
    fn host_default_engine_is_vllm_on_instinct() {
        // What `rocm examine` and `rocm engines list` report on an MI300X. The
        // platform constant said "lemonade" here while serve picked vLLM, so the
        // reported default contradicted the actual behaviour.
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        if cfg!(windows) {
            assert_eq!(default_engine_for_host(&summary), "lemonade");
        } else {
            assert_eq!(default_engine_for_host(&summary), "vllm");
        }
    }

    #[test]
    fn host_default_engine_covers_every_vllm_preferred_family() {
        // Guards the whole preferred set, not just the dcgpu branch, so adding a
        // family to VLLM_PREFERRED_THEROCK_FAMILIES cannot leave the reported
        // default behind.
        for family in VLLM_PREFERRED_THEROCK_FAMILIES {
            let summary = HostGpuSummary {
                name: None,
                gfx_target: Some((*family).to_owned()),
                therock_family: Some((*family).to_owned()),
            };
            let expected = if cfg!(windows) { "lemonade" } else { "vllm" };
            assert_eq!(
                default_engine_for_host(&summary),
                expected,
                "unexpected default for {family}"
            );
        }
    }

    #[test]
    fn host_default_engine_is_lemonade_without_a_vllm_preference() {
        // Strix Halo (gfx1151), a consumer family, and a machine whose GPU has not
        // been identified at all must all keep the platform default.
        for summary in [
            HostGpuSummary {
                name: Some("AMD Radeon 8060S".to_owned()),
                gfx_target: Some("gfx1151".to_owned()),
                therock_family: Some("gfx1151".to_owned()),
            },
            HostGpuSummary {
                name: Some("AMD Radeon".to_owned()),
                gfx_target: Some("gfx1100".to_owned()),
                therock_family: Some("gfx110X-all".to_owned()),
            },
            HostGpuSummary::default(),
        ] {
            assert_eq!(default_engine_for_host(&summary), "lemonade");
        }
    }

    #[test]
    fn host_default_engine_never_reports_vllm_on_native_windows() {
        // The vLLM adapter bails on native Windows, so no GPU may talk the
        // reported default into vLLM there — including an Instinct part.
        if !cfg!(windows) {
            return;
        }
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        assert_eq!(default_engine_for_host(&summary), "lemonade");
    }

    #[test]
    fn consumer_gpu_family_has_no_vllm_preference() {
        // A non-dcgpu consumer family (e.g. gfx110X-all) has no GPU-level vLLM
        // preference, so serve selection falls through to the recipe/platform
        // default rather than forcing vLLM.
        let summary = HostGpuSummary {
            name: Some("AMD Radeon".to_owned()),
            gfx_target: Some("gfx1100".to_owned()),
            therock_family: Some("gfx110X-all".to_owned()),
        };
        assert_eq!(preferred_serve_engine_for_host_gpu_summary(&summary), None);
    }

    #[test]
    fn preferred_serve_engine_uses_vllm_for_supported_therock_families() {
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx90a")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx950")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx999-dcgpu")),
            Some("vllm")
        );
        assert_eq!(preferred_serve_engine_for_therock_family(None), None);
    }

    #[test]
    fn preferred_serve_engine_host_summary_respects_platform_and_fields() {
        // `gfx_target` is consulted as a fallback when `therock_family` is absent.
        let summary = HostGpuSummary {
            gfx_target: Some("gfx950".to_owned()),
            ..HostGpuSummary::default()
        };
        // The vLLM adapter is unsupported on native Windows, so the preference is
        // gated off there while remaining active on Linux/WSL builds.
        let expected = if cfg!(windows) { None } else { Some("vllm") };
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&summary),
            expected
        );

        // No GPU information never resolves to a vLLM preference on any platform.
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&HostGpuSummary::default()),
            None
        );
    }
}
