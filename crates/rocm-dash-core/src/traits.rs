// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Collector traits. Impls live in `rocm-dash-collectors`.
//! These are sync + `Send` so collector code can decide its own async/blocking strategy.

use thiserror::Error;

use crate::bench_schema::BenchmarkRow;
use crate::metrics::{GpuMetrics, GpuSystemInfo, Instance};

#[derive(Debug, Error)]
pub enum CollectorError {
    #[error("collector not available on this host: {0}")]
    Unsupported(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, CollectorError>;

/// Discrete GPU device summary, separate from per-tick `GpuMetrics`.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub device_id: String,
    pub model: String,
    pub vram_total_mb: u64,
}

/// Per-process GPU resource usage.
#[derive(Debug, Clone)]
pub struct GpuProcess {
    pub pid: u32,
    pub device_id: String,
    pub vram_used_mb: u64,
}

pub trait GpuCollector: Send + Sync {
    fn name(&self) -> &'static str;
    fn devices(&self) -> Result<Vec<GpuDevice>>;
    fn metrics(&self) -> Result<Vec<GpuMetrics>>;
    fn system_info(&self) -> Result<GpuSystemInfo>;
    fn processes(&self) -> Result<Vec<GpuProcess>>;
}

/// A discovered serving process or container.
#[derive(Debug, Clone, Default)]
pub struct DiscoveredService {
    pub container_id: String,
    pub container_name: String,
    pub model_name: String,
    pub gpu_ids: Vec<String>,
    pub port: Option<u16>,
    pub tensor_parallel_size: u32,
    pub dtype: Option<String>,
    /// Effective quantization for the instance. Prefers vLLM's explicit
    /// `--quantization` flag; falls back to `--dtype` when no explicit
    /// quantization was passed (the closest signal vLLM exposes). `dtype`
    /// retains the `--dtype`-only value independently.
    pub quantization: Option<String>,
    pub launch_args: Vec<String>,
    pub env_vars: std::collections::BTreeMap<String, String>,
    pub pid: u32,
    pub log_file: Option<String>,
}

pub trait ServiceDiscovery: Send + Sync {
    fn name(&self) -> &'static str;
    fn discover(&self) -> Result<Vec<DiscoveredService>>;
}

/// Per-tick liveness metrics for one instance (KV cache, request counts).
#[derive(Debug, Clone, Default)]
pub struct InstanceSample {
    pub kv_cache_usage_pct: Option<f32>,
    pub running_reqs: Option<u32>,
    pub waiting_reqs: Option<u32>,
    /// Cumulative vLLM `generation_tokens_total` counter at scrape time. The
    /// runner differences successive readings into `Instance.gen_tps`; this
    /// raw counter is not stored on `Instance`.
    pub gen_tokens_total: Option<f64>,
    /// Instantaneous generation throughput (tokens/sec) reported **directly** by
    /// an engine that exposes a rate rather than a cumulative counter (e.g.
    /// Lemonade `/api/v1/stats.tokens_per_second`). vLLM leaves this `None` and
    /// uses the `gen_tokens_total` delta path. Plain data — no async/HTTP/render.
    pub gen_tps: Option<f64>,
    /// Cumulative vLLM `time_to_first_token_seconds_sum` / `_count` histogram
    /// readings at scrape time. The runner windows successive readings into
    /// `Instance.ttft_ms` (Δsum/Δcount × 1000), mirroring the `gen_tps` path;
    /// these raw cumulative values are not stored on `Instance`. `None` for
    /// engines that don't expose the histogram.
    pub ttft_sum_s: Option<f64>,
    pub ttft_count: Option<f64>,
    /// Cumulative vLLM `time_per_output_token_seconds_sum` / `_count` histogram
    /// readings; windowed into `Instance.tpot_ms` the same way.
    pub tpot_sum_s: Option<f64>,
    pub tpot_count: Option<f64>,
}

pub trait InstanceMetrics: Send + Sync {
    fn name(&self) -> &'static str;
    /// Returns `None` when the service is reachable but reports no data,
    /// `Err` when unreachable.
    fn fetch(&self, service: &DiscoveredService) -> Result<InstanceSample>;
}

/// Tail benchmark results from disk (CSV + JSON sidecars).
pub trait BenchTailer: Send + Sync {
    fn name(&self) -> &'static str;
    /// Drain any rows that appeared since the last call.
    fn drain(&mut self) -> Result<Vec<BenchmarkRow>>;
}

/// Merge per-instance metadata + live sample into a finished `Instance`.
pub fn merge_instance(
    svc: &DiscoveredService,
    sample: &InstanceSample,
    vram_used_mb: u64,
    vram_total_mb: u64,
) -> Instance {
    Instance {
        container_id: svc.container_id.clone(),
        container_name: svc.container_name.clone(),
        status: crate::metrics::InstanceStatus::Running,
        model_name: svc.model_name.clone(),
        gpu_ids: svc.gpu_ids.clone(),
        partition_info: None,
        quantization: svc.quantization.clone(),
        tensor_parallel_size: svc.tensor_parallel_size,
        port: svc.port,
        vram_used_mb,
        vram_total_mb,
        kv_cache_usage_pct: sample.kv_cache_usage_pct,
        running_reqs: sample.running_reqs,
        waiting_reqs: sample.waiting_reqs,
        // Engines that report an instantaneous rate (Lemonade) carry it on the
        // sample; counter-based engines (vLLM) leave it `None` and the runner
        // fills `gen_tps` from the `gen_tokens_total` delta. `tokens_per_watt`
        // is derived at snapshot assembly.
        gen_tps: sample.gen_tps,
        tokens_per_watt: None,
        // Derived at the runner by windowing the sample's cumulative histogram
        // readings (mirrors gen_tps); left None at merge time.
        ttft_ms: None,
        tpot_ms: None,
        launch_args: svc.launch_args.clone(),
        env_vars: svc.env_vars.clone(),
        log_file: svc.log_file.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_instance_propagates_quantization() {
        let svc = DiscoveredService {
            quantization: Some("fp8".into()),
            ..Default::default()
        };
        let inst = merge_instance(&svc, &InstanceSample::default(), 0, 0);
        assert_eq!(inst.quantization.as_deref(), Some("fp8"));
    }

    #[test]
    fn merge_instance_quantization_none_when_svc_none() {
        let svc = DiscoveredService::default();
        let inst = merge_instance(&svc, &InstanceSample::default(), 0, 0);
        assert_eq!(inst.quantization, None);
    }
}
