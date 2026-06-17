//! vLLM Prometheus `/metrics` scraper.
//!
//! Field paths and the kv-cache 0..1 → 0..100 scaling are vendored from
//! instinct-dash `VllmMetricsService.ts`. See `../../wiki/entities/vllm.md`.
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
const KEY_KV_CACHE: &str = "vllm:gpu_cache_usage_perc";
const KEY_GEN_TOKENS: &str = "vllm:generation_tokens_total";

#[derive(Debug, Clone)]
pub struct VllmPrometheusCollector {
    host: String,
    client: Client,
}

impl Default for VllmPrometheusCollector {
    fn default() -> Self {
        Self::new("127.0.0.1", Duration::from_millis(2000))
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
            "VllmPrometheusCollector is async — call fetch_async() from a tokio runtime".into(),
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
    let kv = extract(text, KEY_KV_CACHE).map(|v| (v * 100.0) as f32);
    // Cumulative output-token counter; the runner differences it into a rate.
    let gen_tokens_total = extract(text, KEY_GEN_TOKENS);
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
    }
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
";

    #[test]
    fn parses_full_sample() {
        let s = parse(SAMPLE);
        assert_eq!(s.running_reqs, Some(12));
        assert_eq!(s.waiting_reqs, Some(3));
        let kv = s.kv_cache_usage_pct.unwrap();
        assert!((kv - 42.31).abs() < 0.01, "kv was {kv}");
        assert_eq!(s.gen_tokens_total, Some(1234567.0));
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
