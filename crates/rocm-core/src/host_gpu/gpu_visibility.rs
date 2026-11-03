// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which AMD GPUs this process may use: the usable-index probe and the
//! `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` mask arithmetic behind it.

#[cfg(target_os = "linux")]
use super::linux::{linux_drm_amdgpu_card_count, linux_kfd_gpu_node_count};
#[cfg(target_os = "linux")]
use super::wsl::{detect_wsl_summary, is_wsl_host};

/// AMD GPU device ordinals usable for a GPU-required launch on this host, after
/// applying the runtime visibility mask (`HIP_VISIBLE_DEVICES`, then
/// `ROCR_VISIBLE_DEVICES`).
///
/// `Some(indices)` is an authoritative answer: an empty vector means no GPU is
/// usable, so a GPU-required launch must fail rather than fall back to CPU or
/// assume device 0. `None` means availability could not be probed on this
/// platform (e.g. the KFD device exists but its topology is unreadable, or a
/// non-Linux target) and callers should not block a launch on this basis.
#[must_use]
pub fn usable_amd_gpu_indices() -> Option<Vec<u32>> {
    probe_usable_amd_gpu_indices()
}

/// Whether at least one AMD GPU is usable for a GPU-required launch.
///
/// `false` only when the probe authoritatively reports zero usable devices; an
/// unprobeable platform reports `true` so it does not block launches (see
/// [`usable_amd_gpu_indices`]).
#[must_use]
pub fn has_usable_amd_gpu() -> bool {
    usable_amd_gpu_indices().is_none_or(|indices| !indices.is_empty())
}

#[cfg(target_os = "linux")]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    // WSL2 reaches the GPU through /dev/dxg and the Windows host driver. It has
    // no KFD topology and no amdgpu DRM card, and `linux_kfd_gpu_node_count`
    // reads "topology unreadable AND no /dev/kfd" as an authoritative zero
    // rather than as unknown -- which is exactly the WSL2 shape. So a machine
    // whose `examine` said `wsl_rocdxg_ready` was refused a GPU-required launch
    // for having none.
    //
    // Answer from the plumbing that platform actually uses, so `serve` and
    // `examine` cannot contradict each other in either direction: ready means a
    // device, not-ready means none, and both match what the report prints.
    if is_wsl_host() {
        let ready = detect_wsl_summary().is_some_and(|wsl| wsl.rocdxg_ready());
        // One device: WSL2 exposes no per-device topology to enumerate, and the
        // visibility mask still applies on top so an explicit
        // HIP_VISIBLE_DEVICES="" is honoured.
        return usable_amd_gpu_indices_from(usize::from(ready), visibility_mask_from_env());
    }
    let present =
        combine_amd_gpu_counts(linux_kfd_gpu_node_count(), linux_drm_amdgpu_card_count())?;
    usable_amd_gpu_indices_from(present, visibility_mask_from_env())
}

#[cfg(not(target_os = "linux"))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    None
}

/// Combine the KFD-topology and DRM-card AMD GPU counts into one "GPUs present"
/// figure. KFD counts *compute* nodes and is authoritative for HIP ordinals, so
/// it wins whenever it reports at least one GPU. DRM is used only as the
/// zero-KFD fallback, so that a host KFD does not account for is not wrongly
/// told it has no GPU and blocked from serving. The fallback was added when the
/// KFD count read a node file that no kernel exposes and so came back zero
/// everywhere; how often a correctly-read KFD still reports zero has not been
/// surveyed, so treat this as a hedge rather than a description of known
/// hardware.
///
/// DRM must not *raise* a nonzero KFD count: a display/render-only AMD DRM card
/// with no KFD compute node (e.g. KFD=1, DRM=2) would otherwise invent a usable
/// HIP ordinal that passes `--gpu` validation but fails later inside HIP.
///
/// `None` only when NEITHER surface could be read (availability truly unknown).
#[cfg(any(target_os = "linux", test))]
fn combine_amd_gpu_counts(kfd: Option<usize>, drm: Option<usize>) -> Option<usize> {
    match kfd {
        // KFD is compute-authoritative: prefer it whenever it sees a GPU.
        Some(k) if k > 0 => Some(k),
        // Zero KFD compute nodes: let DRM answer rather than block serving.
        Some(_) => Some(drm.unwrap_or(0)),
        // KFD unreadable: use DRM if it could be read, else availability unknown.
        None => drm,
    }
}

/// The GPU visibility mask read from the environment: both variables, kept
/// separately because they apply at different layers and in a fixed order.
/// `ROCR_VISIBLE_DEVICES` masks at the ROCr level and HIP then re-indexes the
/// survivors as `0..N`; `HIP_VISIBLE_DEVICES` selects *within* that re-indexed
/// set. Collapsing the two into one "winning" value loses the composition and
/// makes HIP tokens look like physical ordinals. See
/// [`usable_amd_gpu_indices_from`].
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone)]
struct GpuVisibilityMask {
    /// `ROCR_VISIBLE_DEVICES`, when set. Its tokens are physical ordinals.
    rocr: Option<String>,
    /// `HIP_VISIBLE_DEVICES`, when set. Its tokens are ordinals in the space ROCr
    /// leaves behind — the same as physical ordinals only when `rocr` is unset.
    hip: Option<String>,
}

/// The active GPU visibility mask: whichever of `ROCR_VISIBLE_DEVICES` and
/// `HIP_VISIBLE_DEVICES` are set, and both when both are. `None` when neither is
/// set; an explicitly empty value is carried through as an empty string so
/// callers can distinguish "unset" (all visible) from "set to nothing" (all
/// masked out).
///
/// Linux-only: its sole caller is the Linux probe. (Not `+ test` — no test
/// references it directly, so compiling it into a non-Linux test build would be
/// dead code, which the workspace lints deny.)
#[cfg(target_os = "linux")]
fn visibility_mask_from_env() -> Option<GpuVisibilityMask> {
    let read = |key: &str| std::env::var_os(key).map(|value| value.to_string_lossy().into_owned());
    let mask = GpuVisibilityMask {
        rocr: read("ROCR_VISIBLE_DEVICES"),
        hip: read("HIP_VISIBLE_DEVICES"),
    };
    if mask.rocr.is_none() && mask.hip.is_none() {
        return None;
    }
    Some(mask)
}

/// The tokens of `mask` naming a device in `0..count`, de-duplicated and in mask
/// order. An empty mask yields an empty set (every device hidden). `None` when
/// the mask cannot be interpreted by ordinal alone — a UUID token, or an ordinal
/// outside `0..count` — so callers must not mistake it for "no GPU".
#[cfg(any(target_os = "linux", test))]
fn mask_tokens_within(count: usize, mask: &str) -> Option<Vec<u32>> {
    if mask.is_empty() {
        return Some(Vec::new());
    }
    let mut visible = Vec::new();
    for token in mask.split(',') {
        let index = token.trim().parse::<u32>().ok()?;
        if (index as usize) >= count {
            return None;
        }
        if !visible.contains(&index) {
            visible.push(index);
        }
    }
    Some(visible)
}

/// Apply the visibility `mask` to the present device ordinals (`0..present`).
/// A `None` `mask` means no mask is set (every present device is visible); a
/// `None` *return* means the mask could not be interpreted authoritatively, so
/// callers must not mistake it for "no GPU".
///
/// The returned ordinals are always in HIP space — the space rocm-cli pins its
/// selection through `HIP_VISIBLE_DEVICES`. The two variables are therefore
/// composed in the order the runtime applies them, not treated as alternatives:
///
/// 1. `ROCR_VISIBLE_DEVICES` hides physical devices *below* HIP, which then
///    re-indexes the survivors as `0..N`. Returning the raw physical tokens would
///    make `--gpu` validation reject the ordinals that actually bind and accept
///    ones that do not.
/// 2. `HIP_VISIBLE_DEVICES` then selects within that `0..N` space. Its tokens are
///    already HIP ordinals and are kept as-is — but they must be range-checked
///    against `N`, not against `present`: under an active ROCR mask a HIP token
///    can sit below the physical count and still name no device HIP can see
///    (EAI-7194). Checking it against `present` accepted a `--gpu` ordinal that
///    cannot bind, and steered `--gpu auto` onto it — the exact failure this
///    composition exists to prevent, reached through the other variable.
#[cfg(any(target_os = "linux", test))]
fn usable_amd_gpu_indices_from(
    present: usize,
    mask: Option<GpuVisibilityMask>,
) -> Option<Vec<u32>> {
    let Some(mask) = mask else {
        return Some((0..present as u32).collect());
    };
    // How many devices HIP can see at all: the ROCr survivors, or every present
    // device when no ROCR mask is set.
    let hip_space = match mask.rocr.as_deref() {
        None => present,
        Some(rocr) => {
            let survivors = mask_tokens_within(present, rocr)?;
            if survivors.is_empty() {
                // ROCr hid every device, so there is nothing for HIP to select
                // from whatever HIP_VISIBLE_DEVICES names. Authoritatively empty.
                return Some(Vec::new());
            }
            survivors.len()
        }
    };
    let Some(hip) = mask.hip.as_deref() else {
        // No HIP mask: every device HIP can see is selectable, numbered 0..N.
        return Some((0..hip_space as u32).collect());
    };
    mask_tokens_within(hip_space, hip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ready_wsl_host_offers_a_device_and_an_unready_one_does_not() {
        // The EAI-7944 shape. `serve` used to count KFD nodes and DRM cards,
        // neither of which exists on WSL2, and read the result as an
        // authoritative zero -- refusing to launch on a machine whose own
        // `examine` reported the GPU ready.
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(true), None),
            Some(vec![0])
        );
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(false), None),
            Some(vec![])
        );
        // An explicit empty mask still wins, so a user can opt out on WSL as
        // anywhere — HIP_VISIBLE_DEVICES="" hides the device.
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(true), hip_mask("")),
            Some(vec![])
        );
    }

    #[test]
    fn combine_amd_gpu_counts_prefers_compute_authoritative_kfd() {
        // KFD is compute-authoritative: a nonzero KFD count wins, and DRM must not
        // raise it. A display/render-only AMD DRM card (KFD=1, DRM=2) must NOT
        // invent a second usable HIP ordinal, or an explicit `--gpu 1` would pass
        // validation and then fail inside HIP.
        assert_eq!(combine_amd_gpu_counts(Some(1), Some(2)), Some(1));
        // Same principle with more DRM cards: still bounded by the KFD count.
        assert_eq!(combine_amd_gpu_counts(Some(2), Some(8)), Some(2));
        // KFD larger than DRM (e.g. multi-partition compute nodes): KFD still wins.
        assert_eq!(combine_amd_gpu_counts(Some(3), Some(1)), Some(3));
        // Strix Halo shape: KFD reports 0 GPU nodes, DRM sees the iGPU → 1 present.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(1)), Some(1));
        // Discrete GPUs: both surfaces agree.
        assert_eq!(combine_amd_gpu_counts(Some(8), Some(8)), Some(8));
        // One surface unreadable → use the other.
        assert_eq!(combine_amd_gpu_counts(None, Some(1)), Some(1));
        assert_eq!(combine_amd_gpu_counts(Some(2), None), Some(2));
        // Neither readable → unknown (caller must not treat as zero).
        assert_eq!(combine_amd_gpu_counts(None, None), None);
        // Both agree on zero → authoritative no-GPU.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(0)), Some(0));
    }

    #[test]
    fn usable_gpu_indices_unset_mask_returns_all_present_devices() {
        assert_eq!(usable_amd_gpu_indices_from(0, None), Some(Vec::new()));
        assert_eq!(usable_amd_gpu_indices_from(1, None), Some(vec![0]));
        assert_eq!(usable_amd_gpu_indices_from(3, None), Some(vec![0, 1, 2]));
    }

    /// Only `HIP_VISIBLE_DEVICES` set: its ordinals are already in HIP space and
    /// are used as-is.
    fn hip_mask(value: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: None,
            hip: Some(value.to_owned()),
        })
    }

    /// Only `ROCR_VISIBLE_DEVICES` set: its physical ordinals are re-indexed into
    /// HIP space (survivors become `0..N`).
    fn rocr_mask(value: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: Some(value.to_owned()),
            hip: None,
        })
    }

    /// Both variables set. ROCr applies first and HIP re-indexes the survivors,
    /// so `hip`'s tokens are ordinals *within* `rocr`'s survivor list.
    fn rocr_then_hip_mask(rocr: &str, hip: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: Some(rocr.to_owned()),
            hip: Some(hip.to_owned()),
        })
    }

    #[test]
    fn usable_gpu_indices_empty_mask_hides_every_device() {
        // The masked-device path: GPUs are present but fully masked out.
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("")),
            Some(Vec::new())
        );
        // An empty ROCR mask hides every device too.
        assert_eq!(
            usable_amd_gpu_indices_from(2, rocr_mask("")),
            Some(Vec::new())
        );
    }

    #[test]
    fn usable_gpu_indices_honors_valid_ordinal_masks() {
        assert_eq!(
            usable_amd_gpu_indices_from(4, hip_mask("2,0")),
            Some(vec![2, 0])
        );
        // Duplicates are collapsed.
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("1,1")),
            Some(vec![1])
        );
    }

    #[test]
    fn usable_gpu_indices_treats_unsupported_masks_as_unprobeable() {
        assert_eq!(usable_amd_gpu_indices_from(2, hip_mask("0,5")), None);
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("GPU-deadbeef")),
            None
        );
    }

    #[test]
    fn a_hip_mask_keeps_its_ordinals_but_a_rocr_mask_is_reindexed_to_hip_space() {
        // A HIP_VISIBLE_DEVICES mask is already in the HIP-ordinal space rocm-cli
        // exports through, so its tokens are used as-is.
        assert_eq!(
            usable_amd_gpu_indices_from(4, hip_mask("2,3")),
            Some(vec![2, 3])
        );
        // A ROCR_VISIBLE_DEVICES mask hides physical devices below HIP, which then
        // re-indexes the survivors as 0..N. On a 4-GPU host, ROCR=2,3 leaves two
        // devices that HIP sees as ordinals 0 and 1 — the values that actually
        // bind when exported via HIP_VISIBLE_DEVICES — not the physical 2 and 3.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_mask("2,3")),
            Some(vec![0, 1])
        );
        // A single-device ROCR mask re-indexes to just ordinal 0.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_mask("3")),
            Some(vec![0])
        );
        // An out-of-range token is still "cannot interpret", regardless of source.
        assert_eq!(usable_amd_gpu_indices_from(2, rocr_mask("5")), None);
    }

    #[test]
    fn a_hip_mask_under_a_rocr_mask_is_bounded_by_the_rocr_survivors() {
        // EAI-7194, second half: both variables set. ROCr applies first and HIP
        // re-indexes the survivors as 0..N, so HIP tokens are ordinals within that
        // reduced set — NOT physical ordinals. Range-checking them against the
        // physical `present` accepted ordinals that cannot bind.
        //
        // 4 GPUs present, ROCR=2,3 leaves two devices HIP numbers 0 and 1.
        // HIP=3 names nothing HIP can see, even though 3 < 4 physically: the probe
        // cannot resolve the visible set and must say "unknown", not confidently
        // hand back [3] for `--gpu 3` to be accepted against and `--gpu auto` to be
        // steered onto.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "3")),
            None
        );
        // The same shape one ordinal lower is a real device: HIP ordinal 1 is the
        // second ROCr survivor (physical 3), and it stays selectable.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "1")),
            Some(vec![1])
        );
        // Selecting every survivor keeps both re-indexed ordinals.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "1,0")),
            Some(vec![1, 0])
        );
        // A one-device ROCr set leaves only HIP ordinal 0; ordinal 1 is unknown,
        // not physical ordinal 1.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("3", "0")),
            Some(vec![0])
        );
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("3", "1")),
            None
        );
        // Either variable hiding everything is authoritative: an empty ROCR mask
        // leaves HIP nothing to select, so a HIP mask cannot resurrect a device.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("", "0")),
            Some(Vec::new())
        );
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "")),
            Some(Vec::new())
        );
        // An uninterpretable ROCR mask is still "unknown" whatever HIP says.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("GPU-deadbeef", "0")),
            None
        );
        assert_eq!(
            usable_amd_gpu_indices_from(2, rocr_then_hip_mask("5", "0")),
            None
        );
    }
}
