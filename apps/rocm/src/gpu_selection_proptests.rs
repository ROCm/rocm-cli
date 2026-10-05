// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for `--gpu` selection: auto-selection and explicit-index
//! validation. The oracle is stated from the contract in the doc comments
//! (never a masked-out device, never a busy device while an idle one exists, at
//! most one ordinal), not from the implementation's pass structure.

use super::*;
use proptest::prelude::*;

/// Case count: `default`, unless `PROPTEST_CASES` (proptest's own knob, which
/// an explicit count would otherwise override) asks for a longer run.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn visible_strategy() -> impl Strategy<Value = Option<Vec<u32>>> {
    prop_oneof![
        Just(None),
        // Masks are subsets of 0..8, in arbitrary order, de-duplicated the way
        // `mask_tokens_within` returns them.
        prop::collection::vec(0u32..8, 0..6).prop_map(|mut v| {
            let mut seen = Vec::new();
            v.retain(|i| {
                let fresh = !seen.contains(i);
                seen.push(*i);
                fresh
            });
            Some(v)
        }),
    ]
}

fn vram_strategy() -> impl Strategy<Value = Option<Vec<GpuVramUsage>>> {
    prop_oneof![
        Just(None),
        prop::collection::vec((0u32..8, 0u64..200_000, 0u64..200_000), 0..8).prop_map(|rows| {
            Some(
                rows.into_iter()
                    .map(|(index, used_mb, total_mb)| GpuVramUsage {
                        index,
                        used_mb,
                        total_mb,
                    })
                    .collect(),
            )
        }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(4096)))]

    #[test]
    fn auto_selection_honours_mask_range_and_busy_set(
        detected in prop::option::of(0usize..8),
        visible in visible_strategy(),
        busy in prop::collection::vec(0u32..8, 0..8),
        vram in vram_strategy(),
    ) {
        let picked = select_auto_gpu_index(detected, visible.as_deref(), &busy, vram.as_deref());
        prop_assert!(picked.len() <= 1);
        let in_range = |i: u32| (i as usize) < detected.unwrap_or(0);
        let allowed = |i: u32| in_range(i) && visible.as_ref().is_none_or(|v| v.contains(&i));
        if let Some(&index) = picked.first() {
            prop_assert!(allowed(index), "picked {index}, outside mask/range");
            let idle_exists = (0..8).any(|i| allowed(i) && !busy.contains(&i));
            prop_assert!(
                !idle_exists || !busy.contains(&index),
                "picked busy GPU {index} although an unpinned visible GPU exists"
            );
        } else {
            prop_assert!(!(0..8).any(allowed), "returned no pin although a visible GPU exists");
        }
    }

    #[test]
    fn explicit_index_is_accepted_only_inside_the_visible_set(
        index in 0u32..10,
        // `detect_gpu_count` maps an empty amd-smi list to `None`, never
        // `Some(0)`; `validate_pinned_gpu_index` relies on that (`count - 1`).
        detected in prop::option::of(1usize..8),
        visible in visible_strategy(),
        mask_active in any::<bool>(),
    ) {
        let result = validate_pinned_gpu_index(index, detected, visible.as_deref(), mask_active);
        match (&visible, &result) {
            (Some(v), Ok(pinned)) => {
                prop_assert_eq!(pinned, &vec![index]);
                prop_assert!(v.contains(&index));
            }
            (Some(v), Err(_)) => prop_assert!(!v.contains(&index)),
            (None, Ok(pinned)) => {
                prop_assert_eq!(pinned, &vec![index]);
                prop_assert!(detected.is_none_or(|c| (index as usize) < c));
            }
            (None, Err(_)) => prop_assert!(detected.is_some_and(|c| (index as usize) >= c)),
        }
    }
}
