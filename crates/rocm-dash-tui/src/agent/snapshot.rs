// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure JSON summarization helpers over the cached telemetry [`super::StateSnapshot`].
//!
//! Split out of `agent.rs` to keep the `AgentClient` seam focused. No `rig`
//! dependency at all — testable without Rig / async. The shared seam types
//! (`StateSnapshot`, `AgentError`, `AgentClient`) stay in `agent/mod.rs`.

use serde_json::{Value, json};

use rocm_dash_core::metrics::{GpuMetrics, ObservationFreshness};

use super::StateSnapshot;

fn gpus_of(snap: &StateSnapshot) -> &[GpuMetrics] {
    snap.latest.as_ref().map_or(&[], |s| s.gpus.as_slice())
}

fn gpu_json(g: &GpuMetrics) -> Value {
    json!({
        "device_id": g.device_id,
        "gpu_utilization_pct": g.gpu_utilization_pct,
        "temperature_c": g.temperature_c,
        "power_w": g.power_w,
        "vram_used_mb": g.vram_used_mb,
        "vram_total_mb": g.vram_total_mb,
    })
}

/// Per-GPU util/temp/power/VRAM from the latest snapshot. `gpu_index` selects
/// one GPU; `None` returns all.
pub fn gpu_status_json(snap: &StateSnapshot, gpu_index: Option<usize>) -> Value {
    let gpus = gpus_of(snap);
    match gpu_index {
        Some(i) => match gpus.get(i) {
            Some(g) => json!({ "gpu_index": i, "gpu": gpu_json(g) }),
            None => json!({ "error": format!("no GPU at index {i}"), "gpu_count": gpus.len() }),
        },
        None => json!({ "gpus": gpus.iter().map(gpu_json).collect::<Vec<_>>() }),
    }
}

/// Discovered serving instances with gen_tps, daemon-provided tok/W,
/// freshness metadata, and observed_at timestamp.
///
/// Fields added by EAI-7960:
/// - `gen_tps`: live generation throughput from the Instance (same as daemon)
/// - `tokens_per_watt`: daemon-provided efficiency (never recomputed here)
/// - `freshness`: `"fresh"` | `"held"` | `null` (null for legacy/unknown)
/// - `observed_at`: ISO-8601 UTC string when present, `null` when absent
pub fn list_instances_json(snap: &StateSnapshot) -> Value {
    let arr: Vec<Value> = snap
        .instances
        .iter()
        .map(|i| {
            let (freshness, observed_at) = match &i.gen_tps_observation {
                Some(m) => {
                    let f = match m.freshness {
                        ObservationFreshness::Fresh => "fresh",
                        ObservationFreshness::Held => "held",
                    };
                    (
                        serde_json::Value::String(f.into()),
                        serde_json::Value::String(m.observed_at.to_rfc3339()),
                    )
                }
                None => (serde_json::Value::Null, serde_json::Value::Null),
            };
            json!({
                "name": i.container_name,
                "model": i.model_name,
                "status": format!("{:?}", i.status),
                "kv_cache_usage_pct": i.kv_cache_usage_pct,
                "running_reqs": i.running_reqs,
                "waiting_reqs": i.waiting_reqs,
                "gen_tps": i.gen_tps,
                "tokens_per_watt": i.tokens_per_watt,
                "freshness": freshness,
                "observed_at": observed_at,
            })
        })
        .collect();
    json!({ "instances": arr, "instance_count": arr.len() })
}

/// Per-instance tokens-per-watt from the daemon-provided field.
///
/// Uses `Instance::tokens_per_watt` directly so this tool matches the screen
/// truth exactly (the daemon computes the same value the TUI displays). Also
/// includes gen_tps, freshness, and observed_at for completeness.
pub fn tokens_per_watt_json(snap: &StateSnapshot) -> Value {
    let arr: Vec<Value> = snap
        .instances
        .iter()
        .map(|i| {
            let (freshness, observed_at) = match &i.gen_tps_observation {
                Some(m) => {
                    let f = match m.freshness {
                        ObservationFreshness::Fresh => "fresh",
                        ObservationFreshness::Held => "held",
                    };
                    (
                        serde_json::Value::String(f.into()),
                        serde_json::Value::String(m.observed_at.to_rfc3339()),
                    )
                }
                None => (serde_json::Value::Null, serde_json::Value::Null),
            };
            json!({
                "name": i.container_name,
                "gen_tps": i.gen_tps,
                "tokens_per_watt": i.tokens_per_watt,
                "freshness": freshness,
                "observed_at": observed_at,
            })
        })
        .collect();
    json!({ "instances": arr })
}

/// Pass^N / Pass@N rollup over the cached bench rows, reusing the core rollup.
pub fn bench_summary_json(snap: &StateSnapshot) -> Value {
    let rollups = rocm_dash_core::bench_rollup::rollup_pass_n(snap.bench_rows.iter());
    let arr: Vec<Value> = rollups
        .iter()
        .map(|r| {
            json!({
                "cell": r.cell,
                "model": r.model,
                "n_trials": r.n_trials,
                "n_passed": r.n_passed,
                "pass_n_of_n": r.pass_n_of_n,
                "pass_at_n": r.pass_at_n,
            })
        })
        .collect();
    json!({ "groups": arr, "group_count": rollups.len() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::fixture_snapshot;
    use rocm_dash_core::metrics::Instance;

    fn fixture_with_observation(
        freshness: rocm_dash_core::metrics::ObservationFreshness,
    ) -> StateSnapshot {
        use chrono::Utc;
        use rocm_dash_core::metrics::{ObservationMetadata, Snapshot};
        let observed_at = Utc::now();
        let inst = Instance {
            container_name: "vllm-obs".into(),
            model_name: "llama3".into(),
            gpu_ids: vec!["0".into()],
            gen_tps: Some(300.0),
            tokens_per_watt: Some(1.5),
            gen_tps_observation: Some(ObservationMetadata {
                observed_at,
                freshness,
            }),
            ..Default::default()
        };
        StateSnapshot {
            latest: Some(Snapshot::default()),
            instances: vec![inst],
            bench_rows: vec![],
        }
    }

    #[test]
    fn gpu_status_json_returns_known_gpu_metrics() {
        let snap = fixture_snapshot();
        let v = gpu_status_json(&snap, Some(2));
        let g = &v["gpu"];
        assert_eq!(g["device_id"], "gpu-2");
        assert_eq!(g["gpu_utilization_pct"], 87.0);
        assert_eq!(g["temperature_c"], 71.0);
        assert_eq!(g["power_w"], 250.0);
        assert_eq!(g["vram_used_mb"], 90000);
        // All-GPU form lists every GPU.
        let all = gpu_status_json(&snap, None);
        assert_eq!(all["gpus"].as_array().unwrap().len(), 3);
        // Out-of-range index → graceful error object, not a panic.
        let oob = gpu_status_json(&snap, Some(9));
        assert!(oob["error"].is_string());
    }

    #[test]
    fn list_instances_json_reports_instance_fields() {
        let v = list_instances_json(&fixture_snapshot());
        assert_eq!(v["instance_count"], 1);
        let i = &v["instances"][0];
        assert_eq!(i["name"], "vllm-a");
        assert_eq!(i["model"], "deepseek-r1");
        assert_eq!(i["kv_cache_usage_pct"], 42.0);
        assert_eq!(i["running_reqs"], 3);
    }

    #[test]
    fn tokens_per_watt_json_matches_core_efficiency() {
        // gen_tps 500 / power 250 (gpu-2) = 2.0, matching the reducer.
        let v = tokens_per_watt_json(&fixture_snapshot());
        assert_eq!(v["instances"][0]["tokens_per_watt"], 2.0);
    }

    #[test]
    fn bench_summary_json_rolls_up_groups() {
        let v = bench_summary_json(&fixture_snapshot());
        assert_eq!(v["group_count"], 1);
        let g = &v["groups"][0];
        assert_eq!(g["cell"], "c1");
        assert_eq!(g["n_trials"], 1);
        assert_eq!(g["pass_at_n"], true);
    }

    // ── EAI-7960: agent tool freshness tests ────────────────────────────────
    // RED: list_instances_json / tokens_per_watt_json must include gen_tps,
    // tokens_per_watt (daemon-provided), freshness, and observed_at fields.

    #[test]
    fn list_instances_json_includes_gen_tps_and_daemon_tpw() {
        let snap = fixture_with_observation(rocm_dash_core::metrics::ObservationFreshness::Fresh);
        let v = list_instances_json(&snap);
        let i = &v["instances"][0];
        // Must include gen_tps from Instance.
        assert_eq!(i["gen_tps"], 300.0, "gen_tps must appear in list_instances");
        // Must use daemon-provided tokens_per_watt, not recomputed.
        assert_eq!(i["tokens_per_watt"], 1.5, "daemon tpw must appear");
        // Must include freshness for non-None metadata.
        assert!(i["freshness"].is_string(), "freshness must be a string");
        assert_eq!(i["freshness"], "fresh");
        // Must include observed_at.
        assert!(
            i["observed_at"].is_string(),
            "observed_at must be an ISO string"
        );
    }

    #[test]
    fn list_instances_json_held_freshness_field() {
        let snap = fixture_with_observation(rocm_dash_core::metrics::ObservationFreshness::Held);
        let v = list_instances_json(&snap);
        let i = &v["instances"][0];
        assert_eq!(i["freshness"], "held");
    }

    #[test]
    fn list_instances_json_legacy_meta_none_freshness_is_null() {
        // Legacy instance without metadata: freshness must not be fabricated.
        let inst = Instance {
            container_name: "vllm-legacy".into(),
            model_name: "m".into(),
            gen_tps: Some(100.0),
            tokens_per_watt: Some(0.5),
            gen_tps_observation: None,
            ..Default::default()
        };
        let snap = StateSnapshot {
            latest: Some(rocm_dash_core::metrics::Snapshot::default()),
            instances: vec![inst],
            bench_rows: vec![],
        };
        let v = list_instances_json(&snap);
        let i = &v["instances"][0];
        // freshness must be null (not "fresh" fabricated).
        assert!(
            i["freshness"].is_null(),
            "legacy metadata must yield null freshness, got: {:?}",
            i["freshness"]
        );
    }

    #[test]
    fn tokens_per_watt_json_uses_daemon_value_not_recomputed() {
        // Daemon tokens_per_watt=1.5; if re-computed via gen_tps/power it would differ.
        // The tool must report 1.5, not a recomputed value.
        let snap = fixture_with_observation(rocm_dash_core::metrics::ObservationFreshness::Fresh);
        let v = tokens_per_watt_json(&snap);
        let i = &v["instances"][0];
        assert_eq!(i["tokens_per_watt"], 1.5, "must use daemon-provided tpw");
        // freshness also present.
        assert_eq!(i["freshness"], "fresh");
    }
}
