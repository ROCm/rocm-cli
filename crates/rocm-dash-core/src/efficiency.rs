//! Pure efficiency derivations (tokens-per-watt). No I/O, no rendering.
//!
//! This is the join that turns vLLM throughput + amd-smi power into a
//! per-instance capacity number. See `../wiki/concepts/metric-registry.md`.

use crate::metrics::GpuMetrics;

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

/// Tokens-per-watt for one serving instance: generation throughput (tok/s)
/// divided by the summed power (W) of the GPUs it occupies.
///
/// Returns `None` when throughput is unknown (`gen_tps` is `None`) or the
/// matched GPUs report no power (e.g. amd-smi unavailable, or the id join
/// found nothing) — never a divide-by-zero or a negative number. A zero
/// throughput with live GPUs is a real `Some(0.0)`, not `None`.
pub fn tokens_per_watt(
    gen_tps: Option<f64>,
    gpu_ids: &[String],
    gpus: &[GpuMetrics],
) -> Option<f64> {
    let tps = gen_tps?;
    let total_w: f64 = gpus
        .iter()
        .filter(|g| device_in(&g.device_id, gpu_ids))
        .map(|g| f64::from(g.power_w))
        .sum();
    if total_w > 0.0 {
        Some(tps / total_w)
    } else {
        None
    }
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

    #[test]
    fn joins_bare_index_to_prefixed_device_id() {
        // The crux: instance "0" must match amd-smi "gpu-0".
        let gpus = [gpu("gpu-0", 250.0)];
        let r = tokens_per_watt(Some(500.0), &["0".into()], &gpus);
        assert_eq!(r, Some(2.0));
    }

    #[test]
    fn sums_power_across_tensor_parallel_gpus() {
        let gpus = [
            gpu("gpu-0", 200.0),
            gpu("gpu-1", 300.0),
            gpu("gpu-2", 999.0),
        ];
        // Instance occupies gpus 0 and 1 → 500 W; gpu-2 excluded.
        let r = tokens_per_watt(Some(1000.0), &["0".into(), "1".into()], &gpus);
        assert_eq!(r, Some(2.0));
    }

    #[test]
    fn none_throughput_yields_none() {
        let gpus = [gpu("gpu-0", 250.0)];
        assert_eq!(tokens_per_watt(None, &["0".into()], &gpus), None);
    }

    #[test]
    fn zero_total_power_yields_none() {
        let gpus = [gpu("gpu-0", 0.0)];
        assert_eq!(tokens_per_watt(Some(500.0), &["0".into()], &gpus), None);
    }

    #[test]
    fn unmatched_ids_yield_none() {
        let gpus = [gpu("gpu-5", 250.0)];
        assert_eq!(tokens_per_watt(Some(500.0), &["0".into()], &gpus), None);
    }

    #[test]
    fn empty_gpu_ids_yield_none() {
        let gpus = [gpu("gpu-0", 250.0)];
        assert_eq!(tokens_per_watt(Some(500.0), &[], &gpus), None);
    }

    #[test]
    fn no_gpu_telemetry_yields_none() {
        assert_eq!(tokens_per_watt(Some(500.0), &["0".into()], &[]), None);
    }

    #[test]
    fn zero_throughput_with_live_gpus_is_some_zero() {
        // Idle instance, GPUs powered: 0 tok/W is a real reading, not "-".
        let gpus = [gpu("gpu-0", 250.0)];
        assert_eq!(tokens_per_watt(Some(0.0), &["0".into()], &gpus), Some(0.0));
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
