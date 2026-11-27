// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! vLLM Prometheus `/metrics` scraper.
//!
//! Field paths and the kv-cache 0..1 → 0..100 scaling are vendored from
//! instinct-dash `VllmMetricsService.ts`.
//!
//! Sync `InstanceMetrics::fetch` returns `Unsupported` (the underlying client
//! is async); call `fetch_async` from the runner.

use std::time::Duration;

use reqwest::Client;
use rocm_dash_core::traits::{
    CollectorError, DiscoveredService, InstanceMetrics, InstanceSample, Result,
};
use tracing::warn;

const KEY_RUNNING: &str = "vllm:num_requests_running";
const KEY_WAITING: &str = "vllm:num_requests_waiting";
// vLLM renamed these series; the current name is read first and the old one is
// the fallback for older servers. Names checked against `vllm/v1/metrics/loggers.py`:
// - KV cache: `kv_cache_usage_perc` (since v0.9.2) replaced `gpu_cache_usage_perc`,
//   which is hidden by default in v0.11.x and gone in v0.12.0.
// - Per-token latency: `inter_token_latency_seconds` (since v0.10.2) replaced
//   `time_per_output_token_seconds`, which is hidden by default from v0.12.0 and
//   gone in v0.15.0. Both are per-token-interval histograms, so Δsum/Δcount keeps
//   its meaning. `request_time_per_output_token_seconds` is a per-request mean, a
//   different quantity, and is deliberately not read.
const KEY_KV_CACHE: &str = "vllm:kv_cache_usage_perc";
const KEY_KV_CACHE_LEGACY: &str = "vllm:gpu_cache_usage_perc";
const KEY_GEN_TOKENS: &str = "vllm:generation_tokens_total";
// TTFT / TPOT latency histograms. We read the cumulative `_sum` (seconds) and
// `_count` series; the runner windows successive readings into an average ms
// (Δsum/Δcount), mirroring the `generation_tokens_total` → `gen_tps` path.
const KEY_TTFT_SUM: &str = "vllm:time_to_first_token_seconds_sum";
const KEY_TTFT_COUNT: &str = "vllm:time_to_first_token_seconds_count";
const KEY_TPOT: &str = "vllm:inter_token_latency_seconds";
const KEY_TPOT_LEGACY: &str = "vllm:time_per_output_token_seconds";

#[derive(Debug, Clone)]
pub struct VllmPrometheusCollector {
    host: String,
    client: Client,
}

impl Default for VllmPrometheusCollector {
    fn default() -> Self {
        Self::new("127.0.0.1", Duration::from_secs(2))
    }
}

impl VllmPrometheusCollector {
    pub fn new(host: impl Into<String>, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            host: host.into(),
            client,
        }
    }

    /// Scrape `http://{host}:{port}/metrics` and parse the three vLLM keys we care about.
    pub async fn fetch_async(&self, svc: &DiscoveredService) -> Result<InstanceSample> {
        let port = svc
            .port
            .ok_or_else(|| CollectorError::Unsupported("instance has no port".into()))?;
        let url = format!("http://{}:{port}/metrics", self.host);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CollectorError::Transport(format!("GET {url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(CollectorError::Transport(format!(
                "GET {url}: status {}",
                resp.status()
            )));
        }
        let text = resp
            .text()
            .await
            .map_err(|e| CollectorError::Transport(format!("body {url}: {e}")))?;
        Ok(parse(&text))
    }
}

impl InstanceMetrics for VllmPrometheusCollector {
    fn name(&self) -> &'static str {
        "vllm-prometheus"
    }

    fn fetch(&self, _svc: &DiscoveredService) -> Result<InstanceSample> {
        Err(CollectorError::Unsupported(
            "VllmPrometheusCollector is async - call fetch_async() from a tokio runtime".into(),
        ))
    }
}

// --- pure helpers, fully unit-testable ---------------------------------------

// `pub(crate)` so the engine registry seam can dispatch to the vLLM parser
// unchanged; the parsing logic itself is untouched.
pub(crate) fn parse(text: &str) -> InstanceSample {
    let running = extract(text, KEY_RUNNING).map(|v| v.round() as u32);
    let waiting = extract(text, KEY_WAITING).map(|v| v.round() as u32);
    // vLLM reports kv-cache as 0..1; expose 0..100 to match the TUI's convention.
    let kv = extract(text, KEY_KV_CACHE)
        .or_else(|| extract(text, KEY_KV_CACHE_LEGACY))
        .map(|v| (v * 100.0) as f32);
    // Cumulative output-token counter; the runner differences it into a rate.
    let gen_tokens_total = extract(text, KEY_GEN_TOKENS);
    // Cumulative latency histograms; the runner windows them into avg ms.
    let ttft_sum_s = extract(text, KEY_TTFT_SUM);
    let ttft_count = extract(text, KEY_TTFT_COUNT);
    let (tpot_sum_s, tpot_count) = extract_histogram(text, KEY_TPOT, KEY_TPOT_LEGACY);
    if running.is_none() && waiting.is_none() && kv.is_none() && gen_tokens_total.is_none() {
        warn!("vllm metrics payload had none of the expected keys");
    }
    InstanceSample {
        kv_cache_usage_pct: kv,
        running_reqs: running,
        waiting_reqs: waiting,
        gen_tokens_total,
        // vLLM uses the cumulative-counter path; no direct rate.
        gen_tps: None,
        ttft_sum_s,
        ttft_count,
        tpot_sum_s,
        tpot_count,
    }
}

/// Read a histogram's `_sum` / `_count` pair from one metric family: `current`
/// if it exposes both, else `legacy`. The two series are never taken from
/// different families, since Δsum/Δcount across families would be meaningless.
fn extract_histogram(text: &str, current: &str, legacy: &str) -> (Option<f64>, Option<f64>) {
    for base in [current, legacy] {
        let sum = extract(text, &format!("{base}_sum"));
        let count = extract(text, &format!("{base}_count"));
        if sum.is_some() && count.is_some() {
            return (sum, count);
        }
    }
    (None, None)
}

/// Extract the first sample value for a Prometheus metric.
///
/// Matches lines of the form `metric_name{labels...} <value>` or
/// `metric_name <value>`, skipping `# HELP` / `# TYPE` headers. Returns the
/// first matching numeric value, or `None` if absent / unparseable.
fn extract(text: &str, metric: &str) -> Option<f64> {
    for line in text.lines() {
        let line = line.trim_start();
        if line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix(metric) else {
            continue;
        };
        // The next char must be a label-opener `{` or whitespace.
        // This prevents `vllm:num_requests_running` from matching e.g.
        // a hypothetical `vllm:num_requests_running_total`.
        let value_part = match rest.chars().next() {
            Some('{') => {
                let close = rest.find('}')?;
                rest[close + 1..].trim_start()
            }
            Some(c) if c.is_whitespace() => rest.trim_start(),
            None => return None,
            _ => continue,
        };
        let value_str = value_part.split_whitespace().next()?;
        return value_str.parse::<f64>().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Legacy (vLLM <= 0.11) series names, kept to prove the fallback.
    const SAMPLE: &str = "\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model=\"deepseek-r1\"} 12.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model=\"deepseek-r1\"} 3.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model=\"deepseek-r1\"} 0.4231
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model=\"deepseek-r1\"} 1234567.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model=\"deepseek-r1\"} 50.0
vllm:time_to_first_token_seconds_count{model=\"deepseek-r1\"} 200.0
vllm:time_to_first_token_seconds_bucket{model=\"deepseek-r1\",le=\"0.1\"} 5.0
# HELP vllm:time_per_output_token_seconds Histogram of time per output token.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{model=\"deepseek-r1\"} 30.0
vllm:time_per_output_token_seconds_count{model=\"deepseek-r1\"} 1500.0
";

    // Series and label shapes as vLLM >= 0.15 serves them (`vllm/v1/metrics/loggers.py`
    // at v0.15.0 and v0.26.0): the legacy KV-cache and per-output-token names are
    // gone, `vllm:request_time_per_output_token_seconds` is a different quantity
    // (a per-request mean) and must not be picked up as TPOT.
    const SAMPLE_CURRENT: &str = "\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine=\"0\",model_name=\"m\"} 4.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine=\"0\",model_name=\"m\"} 1.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.5
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine=\"0\",model_name=\"m\"} 8000.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{engine=\"0\",model_name=\"m\"} 10.0
vllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"m\"} 40.0
# HELP vllm:inter_token_latency_seconds Histogram of inter-token latency in seconds.
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{engine=\"0\",le=\"0.01\",model_name=\"m\"} 10.0
vllm:inter_token_latency_seconds_sum{engine=\"0\",model_name=\"m\"} 16.0
vllm:inter_token_latency_seconds_count{engine=\"0\",model_name=\"m\"} 800.0
# HELP vllm:request_time_per_output_token_seconds Histogram of time_per_output_token_seconds per request.
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_sum{engine=\"0\",model_name=\"m\"} 99.0
vllm:request_time_per_output_token_seconds_count{engine=\"0\",model_name=\"m\"} 7.0
";

    // vLLM 0.9.2 to 0.11.x serves both generations of the series: the deprecated
    // KV-cache gauge until 0.11 hides it, and the deprecated TPOT one until 0.12.
    // Distinct values per generation let the test tell which one was read.
    const SAMPLE_TRANSITION: &str = "\
vllm:num_requests_running{model_name=\"m\"} 2.0
vllm:kv_cache_usage_perc{model_name=\"m\"} 0.25
vllm:gpu_cache_usage_perc{model_name=\"m\"} 0.75
vllm:time_per_output_token_seconds_sum{model_name=\"m\"} 111.0
vllm:time_per_output_token_seconds_count{model_name=\"m\"} 222.0
vllm:inter_token_latency_seconds_sum{model_name=\"m\"} 3.0
vllm:inter_token_latency_seconds_count{model_name=\"m\"} 300.0
";

    #[test]
    fn parses_current_vllm_names() {
        let s = parse(SAMPLE_CURRENT);
        assert_eq!(s.running_reqs, Some(4));
        assert_eq!(s.kv_cache_usage_pct, Some(50.0));
        assert_eq!(s.tpot_sum_s, Some(16.0));
        assert_eq!(s.tpot_count, Some(800.0));
        assert_eq!(s.ttft_sum_s, Some(10.0));
        assert_eq!(s.ttft_count, Some(40.0));
    }

    #[test]
    fn current_names_win_when_both_generations_are_served() {
        let s = parse(SAMPLE_TRANSITION);
        assert_eq!(s.kv_cache_usage_pct, Some(25.0));
        assert_eq!(s.tpot_sum_s, Some(3.0));
        assert_eq!(s.tpot_count, Some(300.0));
    }

    #[test]
    fn tpot_pair_is_never_mixed_across_families() {
        // A current `_sum` with no current `_count` must not be paired with the
        // legacy `_count`: the pair falls back whole.
        let text = "\
vllm:inter_token_latency_seconds_sum 3.0
vllm:time_per_output_token_seconds_sum 30.0
vllm:time_per_output_token_seconds_count 1500.0
";
        let s = parse(text);
        assert_eq!(s.tpot_sum_s, Some(30.0));
        assert_eq!(s.tpot_count, Some(1500.0));
        // Neither family complete: nothing is reported rather than half a pair.
        let s = parse("vllm:inter_token_latency_seconds_sum 3.0\n");
        assert_eq!(s.tpot_sum_s, None);
        assert_eq!(s.tpot_count, None);
    }

    #[test]
    fn per_request_tpot_histogram_is_not_read_as_tpot() {
        let text = "\
vllm:request_time_per_output_token_seconds_sum 99.0
vllm:request_time_per_output_token_seconds_count 7.0
";
        let s = parse(text);
        assert_eq!(s.tpot_sum_s, None);
        assert_eq!(s.tpot_count, None);
    }

    #[test]
    fn kv_cache_current_name_scales_into_pct() {
        let s = parse("vllm:kv_cache_usage_perc 0.4231\n");
        let kv = s.kv_cache_usage_pct.unwrap();
        assert!((kv - 42.31).abs() < 0.01, "kv was {kv}");
    }

    #[test]
    fn parses_full_sample() {
        let s = parse(SAMPLE);
        assert_eq!(s.running_reqs, Some(12));
        assert_eq!(s.waiting_reqs, Some(3));
        let kv = s.kv_cache_usage_pct.unwrap();
        assert!((kv - 42.31).abs() < 0.01, "kv was {kv}");
        assert_eq!(s.gen_tokens_total, Some(1_234_567.0));
    }

    #[test]
    fn parses_ttft_tpot_histograms() {
        let s = parse(SAMPLE);
        // The `_sum`/`_count` series are read; `_bucket` lines are ignored
        // (distinct metric name, so the sum key never matches a bucket line).
        assert_eq!(s.ttft_sum_s, Some(50.0));
        assert_eq!(s.ttft_count, Some(200.0));
        assert_eq!(s.tpot_sum_s, Some(30.0));
        assert_eq!(s.tpot_count, Some(1500.0));
    }

    #[test]
    fn ttft_tpot_absent_yields_none() {
        // A payload with the core gauges but no latency histograms → all None.
        let text = "vllm:num_requests_running 7\nvllm:generation_tokens_total 9000\n";
        let s = parse(text);
        assert_eq!(s.ttft_sum_s, None);
        assert_eq!(s.ttft_count, None);
        assert_eq!(s.tpot_sum_s, None);
        assert_eq!(s.tpot_count, None);
    }

    #[test]
    fn parses_generation_counter_without_labels() {
        let s = parse("vllm:generation_tokens_total 9000\n");
        assert_eq!(s.gen_tokens_total, Some(9000.0));
    }

    #[test]
    fn parses_metric_without_labels() {
        let text = "vllm:num_requests_running 7\n";
        let s = parse(text);
        assert_eq!(s.running_reqs, Some(7));
    }

    #[test]
    fn missing_metrics_yield_none() {
        let s = parse("# nothing useful here\n");
        assert_eq!(s.running_reqs, None);
        assert_eq!(s.waiting_reqs, None);
        assert_eq!(s.kv_cache_usage_pct, None);
        assert_eq!(s.gen_tokens_total, None);
    }

    #[test]
    fn kv_cache_scales_zero_to_one_into_pct() {
        let s = parse("vllm:gpu_cache_usage_perc 1.0\n");
        assert_eq!(s.kv_cache_usage_pct, Some(100.0));
        let s = parse("vllm:gpu_cache_usage_perc 0.0\n");
        assert_eq!(s.kv_cache_usage_pct, Some(0.0));
    }

    #[test]
    fn extract_ignores_comment_lines() {
        let text = "# vllm:num_requests_running is a gauge\nvllm:num_requests_running 5\n";
        assert_eq!(extract(text, KEY_RUNNING), Some(5.0));
    }

    #[test]
    fn extract_handles_scientific_notation() {
        let s = parse("vllm:gpu_cache_usage_perc 5.0e-2\n");
        let kv = s.kv_cache_usage_pct.unwrap();
        assert!((kv - 5.0).abs() < 0.001, "kv was {kv}");
    }

    #[test]
    fn extract_does_not_match_prefix() {
        // Defensive: `vllm:num_requests_running` must not match a longer key.
        let text = "vllm:num_requests_running_total 99\nvllm:num_requests_running 7\n";
        assert_eq!(extract(text, KEY_RUNNING), Some(7.0));
    }

    #[test]
    fn sync_fetch_returns_unsupported() {
        let c = VllmPrometheusCollector::default();
        let svc = DiscoveredService {
            port: Some(8000),
            ..Default::default()
        };
        assert!(matches!(c.fetch(&svc), Err(CollectorError::Unsupported(_))));
    }

    #[tokio::test]
    async fn fetch_async_requires_port() {
        let c = VllmPrometheusCollector::default();
        let svc = DiscoveredService::default();
        let r = c.fetch_async(&svc).await;
        assert!(matches!(r, Err(CollectorError::Unsupported(_))));
    }
}
