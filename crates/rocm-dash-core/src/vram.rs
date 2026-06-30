// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure VRAM attribution helpers. No I/O, no rendering.
//!
//! The `/proc` read and the amd-smi subprocess that feed these live in the
//! collectors/daemon; here we only do the math that turns per-process VRAM +
//! device totals into a per-instance `(used, total)` pair.
//!
//! The attribution has two paths, in priority order:
//!  1. **Per-process** — sum the VRAM of the GPU processes that resolve (via an
//!     injected PID→container resolver) to the instance's container id.
//!  2. **Device-summed fallback** — when no per-process entry exists for the
//!     instance, attribute the device-summed *used* over the instance's GPUs.
//!     `total` is always device-summed over the instance's GPUs.
//!
//! Graceful degradation: an instance that matches nothing (empty `gpu_ids`,
//! no container entry — e.g. Lemonade) resolves to `(0, 0)`, never a panic and
//! never a confidently-wrong number. See `../wiki/concepts/metric-registry.md`.

use std::collections::HashMap;

use crate::efficiency::device_in;
use crate::metrics::GpuMetrics;
use crate::traits::GpuProcess;

/// Device-summed `(used_mb, total_mb)` over the GPUs in `gpu_ids`.
///
/// Reuses the `efficiency` join (`"0"` ↔ `"gpu-0"` normalization), so it does
/// not duplicate the id-matching logic. Empty `gpu_ids` or no matching GPU →
/// `(0, 0)`.
pub fn device_vram(gpu_ids: &[String], gpus: &[GpuMetrics]) -> (u64, u64) {
    gpus.iter()
        .filter(|g| device_in(&g.device_id, gpu_ids))
        .fold((0u64, 0u64), |(used, total), g| {
            (used + g.vram_used_mb, total + g.vram_total_mb)
        })
}

/// Aggregate per-process VRAM into a per-container `used_mb` map.
///
/// `resolve` maps a process host PID to its container id (the runner injects a
/// `/proc/<pid>/cgroup` reader; tests inject a fixture closure). Processes
/// whose PID resolves to `None` are skipped. Multiple processes of the same
/// container accumulate; different containers stay separate. This keeps all
/// `/proc` I/O out of core while leaving the aggregation fully unit-testable.
pub fn aggregate_process_vram<F: Fn(u32) -> Option<String>>(
    procs: &[GpuProcess],
    resolve: F,
) -> HashMap<String, u64> {
    let mut per_container: HashMap<String, u64> = HashMap::new();
    for p in procs {
        if let Some(container_id) = resolve(p.pid) {
            *per_container.entry(container_id).or_insert(0) += p.vram_used_mb;
        }
    }
    per_container
}

/// Resolve one instance's `(used_mb, total_mb)`.
///
/// `total` is always the device-summed total over `gpu_ids`. `used` is the
/// per-container value from `per_container_used` when present (the per-process
/// path) and otherwise the device-summed used over `gpu_ids` (the fallback).
pub fn resolve_instance_vram<S: std::hash::BuildHasher>(
    container_id: &str,
    gpu_ids: &[String],
    gpus: &[GpuMetrics],
    per_container_used: &HashMap<String, u64, S>,
) -> (u64, u64) {
    let (device_used, total) = device_vram(gpu_ids, gpus);
    let used = match per_container_used.get(container_id) {
        Some(&v) => v,
        None => device_used,
    };
    (used, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(device_id: &str, used: u64, total: u64) -> GpuMetrics {
        GpuMetrics {
            device_id: device_id.into(),
            vram_used_mb: used,
            vram_total_mb: total,
            ..GpuMetrics::default()
        }
    }

    fn proc(pid: u32, device_id: &str, used: u64) -> GpuProcess {
        GpuProcess {
            pid,
            device_id: device_id.into(),
            vram_used_mb: used,
        }
    }

    #[test]
    fn device_vram_sums_used_and_total_with_normalization() {
        // Instance "0","1" must match amd-smi "gpu-0","gpu-1".
        let gpus = [
            gpu("gpu-0", 1000, 8000),
            gpu("gpu-1", 2000, 8000),
            gpu("gpu-2", 9999, 8000), // excluded
        ];
        let (used, total) = device_vram(&["0".into(), "1".into()], &gpus);
        assert_eq!((used, total), (3000, 16000));
    }

    #[test]
    fn device_vram_empty_and_no_match_are_zero() {
        let gpus = [gpu("gpu-0", 1000, 8000)];
        assert_eq!(device_vram(&[], &gpus), (0, 0));
        assert_eq!(device_vram(&["5".into()], &gpus), (0, 0));
        assert_eq!(device_vram(&["0".into()], &[]), (0, 0));
    }

    #[test]
    fn aggregate_sums_per_container_and_skips_unresolved() {
        let procs = [
            proc(10, "gpu-0", 500),
            proc(11, "gpu-0", 700), // same container as pid 10
            proc(20, "gpu-1", 300), // different container
            proc(99, "gpu-2", 999), // resolves to None → skipped
        ];
        let resolve = |pid: u32| match pid {
            10 | 11 => Some("container-a".to_string()),
            20 => Some("container-b".to_string()),
            _ => None,
        };
        let map = aggregate_process_vram(&procs, resolve);
        assert_eq!(map.get("container-a"), Some(&1200));
        assert_eq!(map.get("container-b"), Some(&300));
        assert_eq!(map.len(), 2); // pid 99 skipped
    }

    #[test]
    fn aggregate_empty_when_nothing_resolves() {
        let procs = [proc(1, "gpu-0", 500)];
        let map = aggregate_process_vram(&procs, |_| None);
        assert!(map.is_empty());
    }

    #[test]
    fn resolve_uses_per_container_map_when_present() {
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let mut per = HashMap::new();
        per.insert("abc".to_string(), 4242);
        // used comes from the map (per-process), total from device sum.
        let (used, total) = resolve_instance_vram("abc", &["0".into()], &gpus, &per);
        assert_eq!((used, total), (4242, 8000));
    }

    #[test]
    fn resolve_falls_back_to_device_used_when_absent() {
        let gpus = [gpu("gpu-0", 1000, 8000), gpu("gpu-1", 2000, 8000)];
        let per = HashMap::new(); // container not present
        let (used, total) =
            resolve_instance_vram("missing", &["0".into(), "1".into()], &gpus, &per);
        assert_eq!((used, total), (3000, 16000));
    }

    #[test]
    fn resolve_lemonade_empty_gpu_ids_stays_zero() {
        // Empty gpu_ids + not in map → (0,0), no panic (Lemonade case).
        let gpus = [gpu("gpu-0", 1000, 8000)];
        let per = HashMap::new();
        let (used, total) = resolve_instance_vram("lemonade-synthetic", &[], &gpus, &per);
        assert_eq!((used, total), (0, 0));
    }
}
