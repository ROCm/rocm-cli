// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure efficiency derivations (tokens-per-watt). No I/O, no rendering.
//!
//! This is the join that turns vLLM throughput + amd-smi power into a
//! per-instance capacity number. See `../wiki/concepts/metric-registry.md`.

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::metrics::{GpuMetrics, Instance};

/// Normalize a GPU identifier to a bare index for joining.
///
/// Docker discovery records `Instance.gpu_ids` as bare indices (`"0"`, `"2"`,
/// from `HIP_VISIBLE_DEVICES`), while amd-smi records `GpuMetrics.device_id`
/// as `"gpu-0"`. Stripping the `gpu-` prefix lets the two match; any other
/// shape passes through unchanged so unexpected ids still compare by equality.
pub fn normalize_gpu_id(id: &str) -> &str {
    id.strip_prefix("gpu-").unwrap_or(id)
}

/// Whether a GPU's `device_id` is one of `gpu_ids`, after normalization.
pub(crate) fn device_in(device_id: &str, gpu_ids: &[String]) -> bool {
    let dev = normalize_gpu_id(device_id);
    gpu_ids.iter().any(|id| normalize_gpu_id(id) == dev)
}

/// Whether any GPU in `gpus` belongs to `gpu_ids`.
///
/// Distinguishes a failed id join (no overlap) from a successful join where
/// power happens to be zero — used by the runner to warn when ids don't line
/// up on real hardware.
pub fn gpu_ids_overlap(gpu_ids: &[String], gpus: &[GpuMetrics]) -> bool {
    gpus.iter().any(|g| device_in(&g.device_id, gpu_ids))
}

/// Board power (W) attributed to each instance, index-aligned with `instances`.
///
/// A GPU's power is split among the instances that occupy it, so the instance
/// figures never add up to more than the board reports. Two instances sharing
/// one GPU each get a share, not the whole board.
///
/// The split is proportional to each occupant's VRAM from
/// `per_container_vram_mb` (the per-process `amd-smi` attribution). A
/// multi-GPU instance's VRAM is spread evenly over its GPUs. If any occupant of
/// a GPU has no per-container reading the split on that GPU is equal, rather
/// than guessing a skew from partial data. A sole occupant gets the whole GPU.
///
/// Power on a GPU no instance occupies (idle, or a process outside the managed
/// set) is attributed to nobody; it still counts toward node power. An entry is
/// `None` when none of its GPUs matched a GPU reporting power (id join found
/// nothing, power telemetry unavailable), so the UI shows `—` instead of a
/// fabricated `0 W`.
#[must_use]
pub fn attribute_power_w<S: BuildHasher>(
    instances: &[Instance],
    gpus: &[GpuMetrics],
    per_container_vram_mb: &HashMap<String, u64, S>,
) -> Vec<Option<f64>> {
    let mut watts = vec![0.0_f64; instances.len()];
    let mut matched = vec![false; instances.len()];
    for gpu in gpus {
        let power = f64::from(gpu.power_w);
        if !power.is_finite() || power <= 0.0 {
            continue;
        }
        let occupants: Vec<usize> = instances
            .iter()
            .enumerate()
            .filter(|(_, i)| device_in(&gpu.device_id, &i.gpu_ids))
            .map(|(k, _)| k)
            .collect();
        if occupants.is_empty() {
            continue;
        }
        let weights: Option<Vec<f64>> = occupants
            .iter()
            .map(|&k| {
                let inst = &instances[k];
                per_container_vram_mb
                    .get(&inst.container_id)
                    .filter(|&&mb| mb > 0)
                    .map(|&mb| mb as f64 / inst.gpu_ids.len().max(1) as f64)
            })
            .collect();
        let weight_sum = weights.as_ref().map(|w| w.iter().sum::<f64>());
        for (pos, &k) in occupants.iter().enumerate() {
            let share = match (&weights, weight_sum) {
                (Some(w), Some(sum)) if sum > 0.0 => w[pos] / sum,
                _ => 1.0 / occupants.len() as f64,
            };
            watts[k] = power.mul_add(share, watts[k]);
            matched[k] = true;
        }
    }
    watts
        .into_iter()
        .zip(matched)
        .map(|(w, m)| m.then_some(w))
        .collect()
}

/// Tokens-per-watt for one serving instance: generation throughput (tok/s)
/// divided by the power (W) attributed to it by [`attribute_power_w`].
///
/// Returns `None` when throughput or attributed power is unknown, or power is
/// not positive — never a divide-by-zero or a negative number. A zero
/// throughput with live power is a real `Some(0.0)`, not `None`.
#[must_use]
pub fn tokens_per_watt(gen_tps: Option<f64>, power_w: Option<f64>) -> Option<f64> {
    let (tps, power) = (gen_tps?, power_w?);
    (power > 0.0).then(|| tps / power)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(device_id: &str, power_w: f32) -> GpuMetrics {
        GpuMetrics {
            device_id: device_id.into(),
            power_w,
            ..GpuMetrics::default()
        }
    }

    #[test]
    fn normalize_strips_gpu_prefix_only() {
        assert_eq!(normalize_gpu_id("gpu-0"), "0");
        assert_eq!(normalize_gpu_id("gpu-7"), "7");
        assert_eq!(normalize_gpu_id("0"), "0");
        assert_eq!(normalize_gpu_id("3"), "3");
    }

    fn inst(id: &str, gpu_ids: &[&str]) -> Instance {
        Instance {
            container_id: id.into(),
            gpu_ids: gpu_ids.iter().map(|s| (*s).to_string()).collect(),
            ..Instance::default()
        }
    }

    fn vram(entries: &[(&str, u64)]) -> HashMap<String, u64> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect()
    }

    #[test]
    fn sole_occupant_gets_whole_board_via_bare_index_join() {
        // The crux: instance "0" must match amd-smi "gpu-0".
        let p = attribute_power_w(&[inst("a", &["0"])], &[gpu("gpu-0", 250.0)], &vram(&[]));
        assert_eq!(p, vec![Some(250.0)]);
    }

    #[test]
    fn tensor_parallel_instance_sums_its_gpus_and_ignores_others() {
        let gpus = [
            gpu("gpu-0", 200.0),
            gpu("gpu-1", 300.0),
            gpu("gpu-2", 999.0),
        ];
        let p = attribute_power_w(&[inst("a", &["0", "1"])], &gpus, &vram(&[]));
        assert_eq!(p, vec![Some(500.0)]);
    }

    #[test]
    fn shared_gpu_splits_by_vram_and_conserves_board_power() {
        let gpus = [gpu("gpu-0", 400.0)];
        let insts = [inst("a", &["0"]), inst("b", &["0"])];
        let p = attribute_power_w(&insts, &gpus, &vram(&[("a", 30_000), ("b", 10_000)]));
        assert_eq!(p, vec![Some(300.0), Some(100.0)]);
        let total: f64 = p.iter().flatten().sum();
        assert!((total - 400.0).abs() < 1e-9, "must not exceed board power");
    }

    #[test]
    fn shared_gpu_splits_equally_without_complete_vram() {
        let gpus = [gpu("gpu-0", 400.0)];
        let insts = [inst("a", &["0"]), inst("b", &["0"])];
        // No readings, or one occupant missing: equal split, no guessed skew.
        for v in [
            vram(&[]),
            vram(&[("a", 30_000)]),
            vram(&[("a", 30_000), ("b", 0)]),
        ] {
            assert_eq!(
                attribute_power_w(&insts, &gpus, &v),
                vec![Some(200.0), Some(200.0)]
            );
        }
    }

    #[test]
    fn multi_gpu_instance_vram_is_spread_across_its_gpus() {
        // `a` spans gpu 0+1 (40 GB total -> 20 GB each); `b` is alone on gpu 1
        // with 20 GB. gpu-0 is all `a`; gpu-1 is split 50/50.
        let gpus = [gpu("gpu-0", 100.0), gpu("gpu-1", 100.0)];
        let insts = [inst("a", &["0", "1"]), inst("b", &["1"])];
        let p = attribute_power_w(&insts, &gpus, &vram(&[("a", 40_000), ("b", 20_000)]));
        assert_eq!(p, vec![Some(150.0), Some(50.0)]);
    }

    #[test]
    fn unoccupied_gpu_power_is_attributed_to_nobody() {
        let gpus = [gpu("gpu-0", 100.0), gpu("gpu-1", 500.0)];
        let p = attribute_power_w(&[inst("a", &["0"])], &gpus, &vram(&[]));
        assert_eq!(p, vec![Some(100.0)]);
    }

    #[test]
    fn unmatched_empty_or_unpowered_yield_none() {
        let v = vram(&[]);
        let a0 = [inst("a", &["0"])];
        assert_eq!(
            attribute_power_w(&a0, &[gpu("gpu-5", 250.0)], &v),
            vec![None]
        );
        assert_eq!(
            attribute_power_w(&[inst("a", &[])], &[gpu("gpu-0", 250.0)], &v),
            vec![None]
        );
        assert_eq!(attribute_power_w(&a0, &[], &v), vec![None]);
        // Zero / NaN power means "no telemetry" (e.g. Windows without ADLX).
        assert_eq!(attribute_power_w(&a0, &[gpu("gpu-0", 0.0)], &v), vec![None]);
        assert_eq!(
            attribute_power_w(&a0, &[gpu("gpu-0", f32::NAN)], &v),
            vec![None]
        );
    }

    #[test]
    fn tokens_per_watt_divides_and_guards() {
        assert_eq!(tokens_per_watt(Some(500.0), Some(250.0)), Some(2.0));
        assert_eq!(tokens_per_watt(None, Some(250.0)), None);
        assert_eq!(tokens_per_watt(Some(500.0), None), None);
        assert_eq!(tokens_per_watt(Some(500.0), Some(0.0)), None);
        // Idle instance, powered GPU: 0 tok/W is a real reading, not "-".
        assert_eq!(tokens_per_watt(Some(0.0), Some(250.0)), Some(0.0));
    }

    #[test]
    fn overlap_detects_match_and_mismatch() {
        let gpus = [gpu("gpu-0", 250.0), gpu("gpu-1", 250.0)];
        assert!(gpu_ids_overlap(&["1".into()], &gpus));
        assert!(!gpu_ids_overlap(&["5".into()], &gpus));
        assert!(!gpu_ids_overlap(&[], &gpus));
        assert!(!gpu_ids_overlap(&["0".into()], &[]));
    }
}
