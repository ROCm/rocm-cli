// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for `--gpu` selection: auto-selection and explicit-index
//! validation. The oracle is stated from the contract in the doc comments, not
//! from the implementation's pass structure.
//!
//! `--gpu auto` (`select_auto_gpu_index`) may choose an ordinal only when it is
//! present and visible:
//!
//! * Present, without VRAM telemetry: inside the dense range `0..n` of a
//!   non-zero amd-smi device count `n`.
//! * Present, with VRAM telemetry: reported by a telemetry row. The rows'
//!   ordinals can be non-contiguous under a visibility mask, so they are never
//!   inferred from how many rows there are. A non-zero amd-smi count `n` also
//!   makes all of `0..n` present when every row falls inside it: a short row
//!   set there is a device the telemetry parser dropped, not a mask. A row at or
//!   above `n` shows the ordinal space is sparse, so the rows alone stay
//!   authoritative.
//! * Visible: in the visibility-resolved set, when that set is known.
//!
//! Among those, a GPU pinned by a running service is never chosen while an
//! unpinned one exists; when every one is pinned, one is still returned (there
//! is no CPU fallback). With neither a non-zero amd-smi count nor any telemetry
//! row there is nothing to rank, so no pin is returned rather than assuming
//! device 0.

use super::*;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Case count: `default`, unless `PROPTEST_CASES` (proptest's own knob, which
/// an explicit count would otherwise override) asks for a longer run.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Ordinals are drawn from `0..ORDINALS`: room for sparse sets, and for a row to
/// sit at or above a smaller amd-smi count.
const ORDINALS: u32 = 8;

/// A set of distinct ordinals in arbitrary order. Neither masks nor amd-smi
/// rows are ordered by contract.
fn ordinal_set() -> impl Strategy<Value = Vec<u32>> {
    prop::sample::subsequence((0..ORDINALS).collect::<Vec<_>>(), 0..=ORDINALS as usize)
        .prop_shuffle()
}

fn visible_strategy() -> impl Strategy<Value = Option<Vec<u32>>> {
    prop_oneof![Just(None), ordinal_set().prop_map(Some)]
}

/// The amd-smi `list` count. `Some(0)` (amd-smi ran and saw nothing) is its
/// own arm so the zero-count paths are reached often.
fn detected_strategy() -> impl Strategy<Value = Option<usize>> {
    prop_oneof![
        Just(None),
        Just(Some(0)),
        (1..=ORDINALS as usize).prop_map(Some),
    ]
}

/// One telemetry row: idle, partly used, fully used, or of unknown capacity.
fn row_strategy(index: u32) -> impl Strategy<Value = GpuVramUsage> {
    prop_oneof![
        (1u64..200_000).prop_map(|total| (0, total)),
        (1u64..200_000).prop_flat_map(|total| (0..=total, Just(total))),
        (1u64..200_000).prop_map(|total| (total, total)),
        (0u64..200_000).prop_map(|used| (used, 0)),
    ]
    .prop_map(move |(used_mb, total_mb)| GpuVramUsage {
        index,
        used_mb,
        total_mb,
    })
}

fn rows_for(indices: Vec<u32>) -> impl Strategy<Value = Vec<GpuVramUsage>> {
    indices
        .into_iter()
        .map(row_strategy)
        .collect::<Vec<_>>()
        .prop_shuffle()
}

/// Telemetry: absent, present with no rows, rows for an arbitrary (usually
/// sparse) ordinal set, or rows for a dense `0..k` (the unmasked shape, which
/// the arbitrary arm would otherwise make rare).
fn vram_strategy() -> impl Strategy<Value = Option<Vec<GpuVramUsage>>> {
    prop_oneof![
        Just(None),
        Just(Some(Vec::new())),
        ordinal_set().prop_flat_map(rows_for).prop_map(Some),
        (1..=ORDINALS)
            .prop_flat_map(|k| rows_for((0..k).collect()))
            .prop_map(Some),
    ]
}

type AutoInputs = (
    Option<usize>,
    Option<Vec<u32>>,
    Vec<u32>,
    Option<Vec<GpuVramUsage>>,
);

/// `(detected, visible, busy, vram)` for `select_auto_gpu_index`.
fn auto_inputs() -> impl Strategy<Value = AutoInputs> {
    (
        detected_strategy(),
        visible_strategy(),
        ordinal_set(),
        vram_strategy(),
    )
}

/// The ordinals `--gpu auto` may choose, as the module docs state the contract.
/// Deliberately not phrased through `effective_gpu_count` or the selection's own
/// candidate construction.
fn choosable(
    detected: Option<usize>,
    visible: Option<&[u32]>,
    vram: Option<&[GpuVramUsage]>,
) -> Vec<u32> {
    let listed = |i: u32| detected.is_some_and(|n| (i as usize) < n);
    let present = |i: u32| match vram {
        None => listed(i),
        Some(rows) => {
            rows.iter().any(|row| row.index == i)
                || (listed(i) && rows.iter().all(|row| listed(row.index)))
        }
    };
    let visible = |i: u32| visible.is_none_or(|v| v.contains(&i));
    (0..ORDINALS)
        .filter(|&i| present(i) && visible(i))
        .collect()
}

/// How often a sample of `auto_inputs` lands in each shape the contract
/// distinguishes.
#[derive(Debug, Default)]
struct AutoReach {
    samples: u32,
    no_count_no_telemetry: u32,
    no_count_with_rows: u32,
    zero_count_with_rows: u32,
    zero_count_without_rows: u32,
    rows_not_a_dense_prefix: u32,
    mask_hides_some_rows: u32,
    listed_range_adds_unreported: u32,
    row_beyond_listed_count: u32,
    busy_and_free_choosable: u32,
    every_choosable_busy: u32,
    known_devices_all_hidden: u32,
    something_choosable: u32,
}

impl AutoReach {
    fn record(&mut self, (detected, visible, busy, vram): &AutoInputs) {
        fn bump(counter: &mut u32, hit: bool) {
            *counter += u32::from(hit);
        }
        self.samples += 1;
        let rows: Vec<u32> = vram.iter().flatten().map(|row| row.index).collect();
        let has_rows = !rows.is_empty();
        let listed = detected.filter(|&n| n > 0);
        let choosable = choosable(*detected, visible.as_deref(), vram.as_deref());
        let busy_choosable = choosable.iter().filter(|i| busy.contains(i)).count();

        bump(
            &mut self.no_count_no_telemetry,
            detected.is_none() && vram.is_none(),
        );
        bump(&mut self.no_count_with_rows, detected.is_none() && has_rows);
        bump(
            &mut self.zero_count_with_rows,
            *detected == Some(0) && has_rows,
        );
        bump(
            &mut self.zero_count_without_rows,
            *detected == Some(0) && !has_rows,
        );
        bump(
            &mut self.rows_not_a_dense_prefix,
            rows.iter().any(|&i| i > 0 && !rows.contains(&(i - 1))),
        );
        bump(
            &mut self.mask_hides_some_rows,
            visible.as_ref().is_some_and(|v| {
                rows.iter().any(|i| !v.contains(i)) && rows.iter().any(|i| v.contains(i))
            }),
        );
        if let Some(n) = listed {
            bump(
                &mut self.listed_range_adds_unreported,
                has_rows
                    && rows.iter().all(|&i| (i as usize) < n)
                    && (0..n as u32).any(|i| !rows.contains(&i)),
            );
            bump(
                &mut self.row_beyond_listed_count,
                rows.iter().any(|&i| (i as usize) >= n),
            );
        }
        bump(
            &mut self.busy_and_free_choosable,
            busy_choosable > 0 && busy_choosable < choosable.len(),
        );
        bump(
            &mut self.every_choosable_busy,
            !choosable.is_empty() && busy_choosable == choosable.len(),
        );
        bump(
            &mut self.known_devices_all_hidden,
            (listed.is_some() || has_rows) && choosable.is_empty(),
        );
        bump(&mut self.something_choosable, !choosable.is_empty());
    }
}

/// The property below is only as strong as the shapes its generator reaches.
/// This counts them over a fixed-seed sample of the same strategy and fails if
/// a generator change starves one.
#[test]
fn auto_selection_inputs_reach_every_contract_shape() {
    let mut runner = TestRunner::deterministic();
    let strategy = auto_inputs();
    let mut reach = AutoReach::default();
    for _ in 0..4096 {
        let inputs = strategy
            .new_tree(&mut runner)
            .expect("strategy generates a value")
            .current();
        reach.record(&inputs);
    }
    eprintln!("auto-selection reach: {reach:#?}");
    // 2% of the sample: well under every measured share, well above zero.
    let floor = reach.samples / 50;
    for (shape, hits) in [
        ("no count, no telemetry", reach.no_count_no_telemetry),
        ("no count, telemetry rows", reach.no_count_with_rows),
        ("zero count, telemetry rows", reach.zero_count_with_rows),
        ("zero count, no rows", reach.zero_count_without_rows),
        (
            "rows not a dense 0..k prefix",
            reach.rows_not_a_dense_prefix,
        ),
        ("mask hides some rows", reach.mask_hides_some_rows),
        (
            "listed range adds unreported ordinals",
            reach.listed_range_adds_unreported,
        ),
        (
            "row at or beyond listed count",
            reach.row_beyond_listed_count,
        ),
        (
            "busy and free both choosable",
            reach.busy_and_free_choosable,
        ),
        ("every choosable GPU busy", reach.every_choosable_busy),
        ("known devices all hidden", reach.known_devices_all_hidden),
        ("something choosable", reach.something_choosable),
    ] {
        assert!(
            hits >= floor,
            "{shape}: {hits} of {} samples, below the floor of {floor}",
            reach.samples
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(4096)))]

    #[test]
    fn auto_selection_honours_mask_range_and_busy_set(
        (detected, visible, busy, vram) in auto_inputs(),
    ) {
        let picked = select_auto_gpu_index(detected, visible.as_deref(), &busy, vram.as_deref());
        let choosable = choosable(detected, visible.as_deref(), vram.as_deref());

        let any_row = vram.as_ref().is_some_and(|rows| !rows.is_empty());
        if detected.unwrap_or(0) == 0 && !any_row {
            prop_assert!(picked.is_empty(), "nothing to rank, yet picked {picked:?}");
        }
        match picked.as_slice() {
            [] => prop_assert!(
                choosable.is_empty(),
                "returned no pin although {choosable:?} may be chosen"
            ),
            [index] => {
                prop_assert!(
                    choosable.contains(index),
                    "picked {index}, outside the choosable set {choosable:?}"
                );
                let free_exists = choosable.iter().any(|i| !busy.contains(i));
                prop_assert!(
                    !free_exists || !busy.contains(index),
                    "picked busy GPU {index} although an unpinned choosable GPU exists"
                );
            }
            more => prop_assert!(false, "auto-selection returned several ordinals: {more:?}"),
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
