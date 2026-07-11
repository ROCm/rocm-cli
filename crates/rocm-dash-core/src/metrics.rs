// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Live metric types. snake_case + units encoded in field names.
//! See `../wiki/concepts/metric-registry.md` and `../wiki/data/metric-field-index.md`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::partition::{ComputePartitionMode, MemoryPartitionMode};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Snapshot {
    pub timestamp: DateTime<Utc>,
    pub host: SystemMetrics,
    pub gpus: Vec<GpuMetrics>,
    pub gpu_system_info: Option<GpuSystemInfo>,
    pub instances: Vec<Instance>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemMetrics {
    pub cpu_overall_pct: f32,
    pub cpu_per_core_pct: Vec<f32>,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub swap_used_mb: u64,
    pub swap_total_mb: u64,
    pub disk_read_bps: u64,
    pub disk_write_bps: u64,
    pub net_rx_bps: u64,
    pub net_tx_bps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuMetrics {
    pub device_id: String,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub gpu_utilization_pct: f32,
    pub temperature_c: f32,
    pub power_w: f32,
    pub clock_mhz: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuSystemInfo {
    pub rocm_version: Option<String>,
    pub driver_version: Option<String>,
    pub gpu_model: String,
    pub physical_gpu_count: u32,
    pub logical_gpu_count: u32,
    pub partition_mode: ComputePartitionMode,
    pub memory_partition_mode: MemoryPartitionMode,
    pub compute_partition_mode: ComputePartitionMode,
    pub vram_per_logical_gpu_mb: u64,
    pub lemond_version: Option<String>,
    pub llama_server_build: Option<String>,
    pub ccr_version: Option<String>,
    pub llamacpp_backend: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InstanceStatus {
    /// The endpoint has passed its model-ready probe (HTTP `/v1/models` or
    /// equivalent) and is serving inference requests.
    Ready,
    Running,
    Starting,
    Stopped,
    Error,
    #[default]
    Unknown,
}

impl InstanceStatus {
    /// True for a status that means "actively serving" from the user's point
    /// of view. `Ready` is the precise state; `Running` is kept as a synonym
    /// for sources that have not yet been wired to the real readiness probe
    /// (e.g. Docker-discovered containers before their first successful scrape).
    #[must_use]
    pub const fn is_serving(self) -> bool {
        matches!(self, Self::Ready | Self::Running)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Instance {
    pub container_id: String,
    pub container_name: String,
    pub status: InstanceStatus,
    pub model_name: String,
    pub gpu_ids: Vec<String>,
    pub partition_info: Option<String>,
    pub quantization: Option<String>,
    pub tensor_parallel_size: u32,
    pub port: Option<u16>,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub kv_cache_usage_pct: Option<f32>,
    pub running_reqs: Option<u32>,
    pub waiting_reqs: Option<u32>,
    /// Live generation throughput (tokens/s), derived from the vLLM
    /// `generation_tokens_total` counter rate. `None` until two scrapes seen.
    #[serde(default)]
    pub gen_tps: Option<f64>,
    /// Efficiency: `gen_tps` ÷ summed power (W) of the GPUs this instance
    /// occupies. `None` when throughput or GPU power telemetry is unavailable.
    #[serde(default)]
    pub tokens_per_watt: Option<f64>,
    /// Live average time-to-first-token (ms), windowed from the vLLM TTFT
    /// histogram. `None` until two scrapes (or absent for non-vLLM engines);
    /// Observe shows `—`. Serde-default for NDJSON replay back-compat.
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Live average time-per-output-token (ms), windowed from the vLLM TPOT
    /// histogram. `None` until two scrapes (or absent); Observe shows `—`.
    #[serde(default)]
    pub tpot_ms: Option<f64>,
    pub launch_args: Vec<String>,
    pub env_vars: std::collections::BTreeMap<String, String>,
    pub log_file: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_deserializes_without_efficiency_fields() {
        // Replay back-compat: NDJSON sessions recorded before tokens_per_watt /
        // ttft_ms / tpot_ms existed must still load, defaulting them to None.
        let legacy = r#"{
            "container_id": "c1", "container_name": "vllm-a", "status": "running",
            "model_name": "deepseek-r1", "gpu_ids": ["0"], "partition_info": null,
            "quantization": null, "tensor_parallel_size": 1, "port": 8000,
            "vram_used_mb": 0, "vram_total_mb": 0, "kv_cache_usage_pct": 42.0,
            "running_reqs": 3, "waiting_reqs": 0, "launch_args": [], "env_vars": {},
            "log_file": null
        }"#;
        let inst: Instance = serde_json::from_str(legacy).expect("legacy instance must parse");
        assert_eq!(inst.gen_tps, None);
        assert_eq!(inst.tokens_per_watt, None);
        assert_eq!(inst.ttft_ms, None);
        assert_eq!(inst.tpot_ms, None);
        assert_eq!(inst.kv_cache_usage_pct, Some(42.0));
    }

    #[test]
    fn instance_round_trips_ttft_tpot_through_serde() {
        // The new live-latency fields survive a serialize → deserialize cycle.
        let inst = Instance {
            container_id: "c1".into(),
            ttft_ms: Some(180.5),
            tpot_ms: Some(22.0),
            ..Default::default()
        };
        let json = serde_json::to_string(&inst).expect("serialize");
        let back: Instance = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.ttft_ms, Some(180.5));
        assert_eq!(back.tpot_ms, Some(22.0));
    }

    #[test]
    fn instance_status_ready_round_trips_as_lowercase_json() {
        // The new `Ready` variant must serialize/deserialize the same way the
        // pre-existing variants do (bare lowercase string, no wrapper object).
        let json = serde_json::to_string(&InstanceStatus::Ready).expect("serialize");
        assert_eq!(json, "\"ready\"");
        let back: InstanceStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, InstanceStatus::Ready);
    }

    #[test]
    fn instance_status_is_serving_covers_ready_and_running_only() {
        assert!(InstanceStatus::Ready.is_serving());
        assert!(InstanceStatus::Running.is_serving());
        assert!(!InstanceStatus::Starting.is_serving());
        assert!(!InstanceStatus::Stopped.is_serving());
        assert!(!InstanceStatus::Error.is_serving());
        assert!(!InstanceStatus::Unknown.is_serving());
    }
}
