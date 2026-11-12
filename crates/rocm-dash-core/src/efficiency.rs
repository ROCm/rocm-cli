// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure efficiency derivations (tokens-per-watt). No I/O, no rendering.
//!
//! This is the join that turns vLLM throughput + amd-smi power into a
//! per-instance capacity number.

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::metrics::{GpuMetrics, Instance, ObservationFreshness, Snapshot};

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
/// An occupant that is still starting and has no reading while another occupant
/// does is left out of the split. (A status of `Unknown` means no readiness
/// information, not "not serving", so it stays in.) It has not
/// allocated GPU memory, so the evidence says it is drawing next to nothing, and
/// an equal share would show the serving neighbour about half its real power and
/// roughly double its real tokens per watt until the newcomer loads.
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
        let reading = |k: usize| {
            per_container_vram_mb
                .get(&instances[k].container_id)
                .is_some_and(|&mb| mb > 0)
        };
        let occupants: Vec<usize> = if occupants.iter().any(|&k| reading(k)) {
            occupants
                .into_iter()
                .filter(|&k| reading(k) || !instances[k].status.is_starting())
                .collect()
        } else {
            occupants
        };
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

/// Whole-node efficiency: aggregate generation throughput over aggregate power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeEfficiency {
    /// Σ `gen_tps` ÷ Σ board `power_w`, in tokens per watt.
    pub tokens_per_watt: f64,
    /// Freshness of the throughput behind the figure (instances at zero tok/s
    /// contribute nothing and are not counted): `Held` when any of it is
    /// carried from a prior scrape window, `Fresh` when all of it is known to
    /// be current, and `None` when some of it has no freshness metadata (a
    /// legacy recording) and none is known to be held. Unknown is never
    /// reported as fresh. The figure is still reported when held, so a
    /// one-scrape hiccup does not blank it, but a consumer must mark it: a
    /// held value expires within the observation validity window.
    pub freshness: Option<ObservationFreshness>,
}

impl NodeEfficiency {
    /// Whether any throughput behind the figure is held.
    #[must_use]
    pub fn held(&self) -> bool {
        self.freshness == Some(ObservationFreshness::Held)
    }
}

/// Aggregate generation throughput across instances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeThroughput {
    /// Σ `gen_tps` over the instances with a finite reading, in tokens/s. A real
    /// zero counts: an idle node is zero throughput, not no data.
    pub tokens_per_sec: f64,
    /// Freshness of the throughput behind the sum, as for
    /// [`NodeEfficiency::freshness`]: instances at zero tok/s add nothing to the
    /// sum and are not counted.
    pub freshness: Option<ObservationFreshness>,
}

impl NodeThroughput {
    /// Whether any throughput behind the sum is held.
    #[must_use]
    pub fn held(&self) -> bool {
        self.freshness == Some(ObservationFreshness::Held)
    }
}

/// Node throughput over `instances`, or `None` when no instance has a finite
/// `gen_tps` (a single NaN/Inf must not poison the sum).
///
/// This is the one place the node's throughput and its freshness are worked
/// out, so every surface that shows the aggregate (the efficiency hero, the
/// throughput hero, the Hardware header, the assistant) agrees on both. An
/// instance at zero tok/s contributes nothing to the sum, so its freshness says
/// nothing about the sum's either: a held idle instance must not mark a figure
/// that comes from fresh ones as held.
#[must_use]
pub fn node_throughput<'a>(
    instances: impl IntoIterator<Item = &'a Instance>,
) -> Option<NodeThroughput> {
    let mut tokens_per_sec = 0.0_f64;
    let mut any = false;
    let mut held = false;
    let mut unknown = false;
    for inst in instances {
        let Some(v) = inst.gen_tps.filter(|v| v.is_finite()) else {
            continue;
        };
        tokens_per_sec += v;
        any = true;
        if v <= 0.0 {
            continue;
        }
        match inst.gen_tps_observation.as_ref().map(|m| m.freshness) {
            Some(ObservationFreshness::Held) => held = true,
            Some(ObservationFreshness::Fresh) => {}
            None => unknown = true,
        }
    }
    let freshness = if held {
        Some(ObservationFreshness::Held)
    } else if unknown {
        None
    } else {
        Some(ObservationFreshness::Fresh)
    };
    any.then_some(NodeThroughput {
        tokens_per_sec,
        freshness,
    })
}

/// Total board power (W) of the GPUs that report a usable reading.
///
/// A reading that is not finite or not positive is skipped, as
/// [`attribute_power_w`] does: zero means "no telemetry" (a platform without a
/// power source), and a faulty sensor reporting infinity, NaN or a negative
/// value must not turn a quotient into a plausible-looking 0 or deflate the
/// total. Every surface that shows the node's power or divides by it uses this,
/// so they agree on what counts.
#[must_use]
pub fn node_power_w(gpus: &[GpuMetrics]) -> f64 {
    gpus.iter()
        .map(|g| f64::from(g.power_w))
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum()
}

/// Node-level efficiency for a snapshot, or `None` when it cannot be stated.
///
/// Throughput is [`node_throughput`]; power is [`node_power_w`], so one faulty
/// sensor cannot turn the quotient into a plausible-looking 0 (infinity) or
/// deflate the total (negative). It is `None` with no traffic
/// (`Σ gen_tps == 0`), no usable power telemetry, or a non-finite quotient.
///
/// Power here is the whole board, including GPUs no instance occupies, because
/// the question is the node's efficiency, not an instance's: per-instance
/// figures use attributed power ([`attribute_power_w`]) instead.
///
/// A node whose only throughput is held at zero tok/s reports a fresh 0 tok/s
/// from [`node_throughput`] (nothing contributes, so nothing is held), and no
/// efficiency figure from here, since there is no throughput to divide.
#[must_use]
pub fn node_efficiency(snap: &Snapshot) -> Option<NodeEfficiency> {
    let throughput = node_throughput(&snap.instances)?;
    let power = node_power_w(&snap.gpus);
    if throughput.tokens_per_sec > 0.0 && power > 0.0 {
        let tokens_per_watt = throughput.tokens_per_sec / power;
        tokens_per_watt.is_finite().then_some(NodeEfficiency {
            tokens_per_watt,
            freshness: throughput.freshness,
        })
    } else {
        None
    }
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
    use crate::metrics::InstanceStatus;

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

    fn snap_with(instances: Vec<Instance>, gpus: Vec<GpuMetrics>) -> Snapshot {
        Snapshot {
            gpus,
            instances,
            ..Snapshot::default()
        }
    }

    fn serving(
        id: &str,
        gen_tps: Option<f64>,
        freshness: Option<ObservationFreshness>,
    ) -> Instance {
        use crate::metrics::ObservationMetadata;
        Instance {
            container_id: id.into(),
            gen_tps,
            gen_tps_observation: freshness.map(|freshness| ObservationMetadata {
                observed_at: chrono::Utc::now(),
                freshness,
            }),
            ..Instance::default()
        }
    }

    #[test]
    fn node_efficiency_divides_total_throughput_by_total_board_power() {
        let snap = snap_with(
            vec![
                serving("a", Some(300.0), None),
                serving("b", Some(200.0), None),
            ],
            vec![gpu("gpu-0", 400.0), gpu("gpu-1", 600.0)],
        );
        // (300 + 200) / (400 + 600) = 0.5 tok/W
        let n = node_efficiency(&snap).expect("some");
        assert!((n.tokens_per_watt - 0.5).abs() < 1e-9, "got {n:?}");
        assert!(!n.held());
    }

    #[test]
    fn node_efficiency_counts_power_of_gpus_no_instance_occupies() {
        // The node pays for an idle GPU too, so it stays in the denominator.
        let snap = snap_with(
            vec![serving("a", Some(100.0), None)],
            vec![gpu("gpu-0", 100.0), gpu("gpu-1", 300.0)],
        );
        let n = node_efficiency(&snap).expect("some");
        assert!((n.tokens_per_watt - 0.25).abs() < 1e-9, "got {n:?}");
    }

    #[test]
    fn node_efficiency_is_none_without_power_or_traffic() {
        let no_power = snap_with(
            vec![serving("a", Some(100.0), None)],
            vec![gpu("gpu-0", 0.0)],
        );
        assert_eq!(node_efficiency(&no_power), None);
        let no_traffic = snap_with(vec![serving("a", None, None)], vec![gpu("gpu-0", 500.0)]);
        assert_eq!(node_efficiency(&no_traffic), None);
        let zero_traffic = snap_with(
            vec![serving("a", Some(0.0), None)],
            vec![gpu("gpu-0", 500.0)],
        );
        assert_eq!(node_efficiency(&zero_traffic), None);
    }

    #[test]
    fn a_held_idle_instance_does_not_mark_a_figure_that_comes_from_fresh_ones() {
        let gpus = vec![gpu("gpu-0", 500.0)];
        let snap = snap_with(
            vec![
                serving("busy", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("idle", Some(0.0), Some(ObservationFreshness::Held)),
            ],
            gpus.clone(),
        );
        let n = node_efficiency(&snap).unwrap();
        assert!(
            !n.held(),
            "a zero-throughput held instance must not flag the node"
        );
        assert_eq!(n.freshness, Some(ObservationFreshness::Fresh));
        // ...and one with unknown freshness at zero does not make it unknown.
        let snap = snap_with(
            vec![
                serving("busy", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("idle", Some(0.0), None),
            ],
            gpus,
        );
        assert_eq!(
            node_efficiency(&snap).unwrap().freshness,
            Some(ObservationFreshness::Fresh)
        );
    }

    #[test]
    fn node_throughput_and_node_efficiency_agree_on_freshness() {
        // The two panels that show these must never disagree: a held instance
        // at zero beside a fresh one is fresh for both.
        let snap = snap_with(
            vec![
                serving("busy", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("idle", Some(0.0), Some(ObservationFreshness::Held)),
            ],
            vec![gpu("gpu-0", 500.0)],
        );
        let t = node_throughput(&snap.instances).unwrap();
        assert!((t.tokens_per_sec - 250.0).abs() < 1e-9);
        assert!(!t.held());
        assert_eq!(t.freshness, node_efficiency(&snap).unwrap().freshness);
        // A held instance that does contribute marks both.
        let held = snap_with(
            vec![
                serving("busy", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("slow", Some(5.0), Some(ObservationFreshness::Held)),
            ],
            vec![gpu("gpu-0", 500.0)],
        );
        assert!(node_throughput(&held.instances).unwrap().held());
        assert!(node_efficiency(&held).unwrap().held());
    }

    #[test]
    fn node_throughput_distinguishes_idle_from_no_data() {
        // A real zero is a figure; nothing finite is no figure.
        let idle = [serving("a", Some(0.0), Some(ObservationFreshness::Fresh))];
        assert!(node_throughput(&idle).unwrap().tokens_per_sec.abs() < f64::EPSILON);
        let none = [serving("a", None, None), serving("b", Some(f64::NAN), None)];
        assert_eq!(node_throughput(&none), None);
        assert_eq!(node_throughput(&[]), None);
    }

    #[test]
    fn node_efficiency_skips_a_faulty_power_sensor_like_attribute_power_does() {
        let insts = vec![serving("a", Some(250.0), Some(ObservationFreshness::Fresh))];
        // An infinite reading must not turn the quotient into 0.
        let inf = snap_with(
            insts.clone(),
            vec![gpu("gpu-0", 500.0), gpu("gpu-1", f32::INFINITY)],
        );
        let n = node_efficiency(&inf).unwrap();
        assert!((n.tokens_per_watt - 0.5).abs() < 1e-9, "got {n:?}");
        // A negative reading must not deflate the total.
        let neg = snap_with(
            insts.clone(),
            vec![gpu("gpu-0", 500.0), gpu("gpu-1", -300.0)],
        );
        let n = node_efficiency(&neg).unwrap();
        assert!((n.tokens_per_watt - 0.5).abs() < 1e-9, "got {n:?}");
        // Only faulty readings: no usable power, so no figure. A lone infinity is
        // the case only the filter decides: unfiltered, tps / inf is a finite 0.
        let only_bad = snap_with(
            insts.clone(),
            vec![gpu("gpu-0", f32::NAN), gpu("gpu-1", -5.0)],
        );
        assert_eq!(node_efficiency(&only_bad), None);
        let lone_inf = snap_with(insts, vec![gpu("gpu-0", f32::INFINITY)]);
        assert_eq!(node_efficiency(&lone_inf), None);
    }

    #[test]
    fn node_power_sums_only_usable_readings() {
        let gpus = [
            gpu("gpu-0", 300.0),
            gpu("gpu-1", f32::INFINITY),
            gpu("gpu-2", f32::NAN),
            gpu("gpu-3", -40.0),
            gpu("gpu-4", 0.0),
            gpu("gpu-5", 200.0),
        ];
        assert!((node_power_w(&gpus) - 500.0).abs() < 1e-9);
        assert!(node_power_w(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn node_efficiency_ignores_non_finite_gen_tps() {
        // One NaN sample must not poison the sum for the whole node.
        let snap = snap_with(
            vec![
                serving("finite", Some(250.0), Some(ObservationFreshness::Held)),
                serving("poisoned", Some(f64::NAN), Some(ObservationFreshness::Held)),
            ],
            vec![gpu("gpu-0", 500.0)],
        );
        let n = node_efficiency(&snap).expect("finite gen_tps must still produce a result");
        assert!((n.tokens_per_watt - 0.5).abs() < 1e-9, "got {n:?}");
    }

    #[test]
    fn node_efficiency_is_held_when_any_contributing_throughput_is_held() {
        let gpus = vec![gpu("gpu-0", 500.0)];
        let fresh = snap_with(
            vec![serving("a", Some(250.0), Some(ObservationFreshness::Fresh))],
            gpus.clone(),
        );
        let n = node_efficiency(&fresh).unwrap();
        assert!(!n.held());
        assert_eq!(n.freshness, Some(ObservationFreshness::Fresh));
        // Unknown freshness (legacy snapshot) is neither held nor claimed
        // fresh: nothing measured it.
        let unknown = snap_with(vec![serving("a", Some(250.0), None)], gpus.clone());
        let n = node_efficiency(&unknown).unwrap();
        assert!(!n.held());
        assert_eq!(n.freshness, None);
        // One unknown among fresh ones keeps the whole figure unknown.
        let partly = snap_with(
            vec![
                serving("a", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("b", Some(10.0), None),
            ],
            gpus.clone(),
        );
        assert_eq!(node_efficiency(&partly).unwrap().freshness, None);
        let mixed = snap_with(
            vec![
                serving("a", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("b", Some(100.0), Some(ObservationFreshness::Held)),
            ],
            gpus.clone(),
        );
        assert!(node_efficiency(&mixed).unwrap().held());
        // A held instance whose throughput is non-finite does not contribute,
        // so it cannot mark the node held.
        let excluded = snap_with(
            vec![
                serving("a", Some(250.0), Some(ObservationFreshness::Fresh)),
                serving("b", Some(f64::INFINITY), Some(ObservationFreshness::Held)),
            ],
            gpus,
        );
        assert!(!node_efficiency(&excluded).unwrap().held());
        assert_eq!(
            node_efficiency(&excluded).unwrap().freshness,
            Some(ObservationFreshness::Fresh)
        );
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
    fn a_starting_occupant_with_no_reading_does_not_dilute_a_serving_neighbour() {
        let gpus = [gpu("gpu-0", 400.0)];
        let mut serving = inst("a", &["0"]);
        serving.status = InstanceStatus::Ready;
        let mut loading = inst("b", &["0"]);
        loading.status = InstanceStatus::Starting { phase: None };
        let insts = [serving, loading];
        // `a` has a VRAM reading; `b` is still loading and has none.
        let p = attribute_power_w(&insts, &gpus, &vram(&[("a", 30_000)]));
        assert_eq!(p, vec![Some(400.0), None], "the loader gets no share yet");
        // Once it has allocated memory it takes part in the split.
        let p = attribute_power_w(&insts, &gpus, &vram(&[("a", 30_000), ("b", 10_000)]));
        assert_eq!(p, vec![Some(300.0), Some(100.0)]);
        // With no readings at all nothing is known, so the split stays equal.
        let p = attribute_power_w(&insts, &gpus, &vram(&[]));
        assert_eq!(p, vec![Some(200.0), Some(200.0)]);
    }

    #[test]
    fn a_starting_occupant_is_left_out_but_an_unmeasured_serving_one_still_forces_equal() {
        let gpus = [gpu("gpu-0", 400.0)];
        let mut measured = inst("a", &["0"]);
        measured.status = InstanceStatus::Ready;
        let mut unmeasured = inst("b", &["0"]);
        unmeasured.status = InstanceStatus::Ready;
        let mut loading = inst("c", &["0"]);
        loading.status = InstanceStatus::Starting { phase: None };
        let insts = [measured, unmeasured, loading];
        // Only `a` has a reading. `c` is starting with none, so it is dropped;
        // `b` is serving but unmeasured (e.g. a missed /proc join), so leaving
        // it out would credit `a` with the whole board: the rest stay equal.
        let p = attribute_power_w(&insts, &gpus, &vram(&[("a", 30_000)]));
        assert_eq!(p, vec![Some(200.0), Some(200.0), None]);
        // Without the starting occupant in the picture the result is the same
        // for the other two, so the starting one is what changed the split.
        let p2 = attribute_power_w(&insts[..2], &gpus, &vram(&[("a", 30_000)]));
        assert_eq!(p2, vec![Some(200.0), Some(200.0)]);
        // An `Unknown` status is not "starting": it stays in the split.
        let mut unknown = inst("d", &["0"]);
        unknown.status = InstanceStatus::Unknown;
        let p3 = attribute_power_w(&[insts[0].clone(), unknown], &gpus, &vram(&[("a", 30_000)]));
        assert_eq!(p3, vec![Some(200.0), Some(200.0)]);
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
