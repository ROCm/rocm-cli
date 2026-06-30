// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Lemonade Server collector — pure parsers over the Lemonade REST API.
//!
//! Schema verified against the official docs (lemonade-server.ai/docs/api):
//! - `GET /api/v1/stats` → per-request performance metrics:
//!   `time_to_first_token`, `tokens_per_second`, `input_tokens`, `output_tokens`,
//!   `prompt_tokens`, `decode_token_times[]`. (No KV-cache / running / waiting —
//!   Lemonade does not expose those; those `InstanceSample` fields stay `None`.)
//! - `GET /api/v1/health` → `status` + `model_loaded` (most-recent model name).
//!
//! The parsers (`parse_stats`, `parse_health_model`) are **pure** (serde only)
//! and the deterministic, fixture-tested anchor. The async scrape
//! ([`LemonadeCollector`]) fetches those bodies over HTTP and degrades to
//! "not reachable" (never panics) so a host with no Lemonade endpoint is a no-op.

use std::time::Duration;

use reqwest::Client;
use rocm_dash_core::metrics::Instance;
use rocm_dash_core::traits::{
    CollectorError, DiscoveredService, InstanceSample, Result, merge_instance,
};
use serde::Deserialize;

/// Lemonade's default OpenAI-compatible port. Mirrors `rocm_dash_tui::skills`
/// (defined locally to avoid a cross-crate dep from collectors → tui).
pub const LEMONADE_PORT: u16 = 13305;
/// The runtime-stats path on the canonical `/api/v1` base.
pub const LEMONADE_STATS_PATH: &str = "/api/v1/stats";
/// The health/liveness path on the canonical `/api/v1` base.
pub const LEMONADE_HEALTH_PATH: &str = "/api/v1/health";

/// `/api/v1/stats` response — performance metrics from the most recent request.
/// Every field is optional: the endpoint only populates them after an inference.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LemonadeStats {
    #[serde(default)]
    pub time_to_first_token: Option<f64>,
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub decode_token_times: Option<Vec<f64>>,
}

/// `/api/v1/health` response — liveness + the most-recently-loaded model name.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LemonadeHealth {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub model_loaded: Option<String>,
}

/// PURE: parse a `/api/v1/stats` body into an [`InstanceSample`].
///
/// Lemonade reports an instantaneous `tokens_per_second` rate (mapped to `gen_tps`) rather than a
/// cumulative counter; it exposes no KV-cache / running / waiting metrics, so
/// those stay `None`. Malformed/empty JSON → an all-`None` default, never panics.
pub fn parse_stats(body: &str) -> InstanceSample {
    let stats = parse_stats_struct(body);
    InstanceSample {
        kv_cache_usage_pct: None,
        running_reqs: None,
        waiting_reqs: None,
        // Lemonade has no cumulative token counter; surface the rate directly.
        gen_tokens_total: None,
        gen_tps: stats.tokens_per_second,
        // TTFT/TPOT histograms are a vLLM-Prometheus concept; Lemonade leaves
        // them None (Observe shows `—`).
        ttft_sum_s: None,
        ttft_count: None,
        tpot_sum_s: None,
        tpot_count: None,
    }
}

/// PURE: parse the structured `/api/v1/stats` body. Returns the full Lemonade
/// metric set (for callers that want TTFT / token counts). Malformed → default.
pub fn parse_stats_struct(body: &str) -> LemonadeStats {
    object_or_default(body)
}

/// PURE: extract the loaded model name from a `/api/v1/health` body, if present
/// and non-empty. Malformed JSON → `None`, never panics.
pub fn parse_health_model(body: &str) -> Option<String> {
    let health: LemonadeHealth = object_or_default(body);
    health.model_loaded.filter(|s| !s.is_empty())
}

/// Deserialize a JSON **object** body into `T`, returning `T::default()` for any
/// non-object/invalid input. serde's derived `Deserialize` decodes a struct from
/// a positional JSON *array* too — guarding on object shape prevents a wrong-shape
/// body (e.g. `[1,2,3]`) from being misread into fields.
fn object_or_default<T: serde::de::DeserializeOwned + Default>(body: &str) -> T {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(value @ serde_json::Value::Object(_)) => {
            serde_json::from_value(value).unwrap_or_default()
        }
        _ => T::default(),
    }
}

/// A stable synthetic id for the local Lemonade endpoint (it is a server, not a
/// container, so it has no Docker id). Distinct from any vLLM/Docker container id.
pub fn lemonade_container_id(host: &str, port: u16) -> String {
    format!("lemonade-{host}-{port}")
}

/// PURE: build a `DiscoveredService` for a Lemonade endpoint. Lemonade is a
/// single local server (no tensor-parallel sharding metadata), so TP = 1.
pub fn lemonade_service(host: &str, port: u16, model_name: &str) -> DiscoveredService {
    DiscoveredService {
        container_id: lemonade_container_id(host, port),
        container_name: "lemonade".to_string(),
        model_name: if model_name.is_empty() {
            "lemonade".to_string()
        } else {
            model_name.to_string()
        },
        port: Some(port),
        tensor_parallel_size: 1,
        ..Default::default()
    }
}

/// PURE: build a finished `Instance` from a Lemonade `/health` model name + a `/stats` body.
///
/// This is the fixture→Instance anchor (no network). `gen_tps` flows from
/// `tokens_per_second`; KV/req fields stay `None` (Lemonade does not report them).
pub fn lemonade_instance(host: &str, port: u16, model_name: &str, stats_body: &str) -> Instance {
    let svc = lemonade_service(host, port, model_name);
    let sample = parse_stats(stats_body);
    let mut inst = merge_instance(&svc, &sample, 0, 0);
    inst.status = rocm_dash_core::metrics::InstanceStatus::Running;
    inst
}

/// Async scrape of a local Lemonade endpoint. Network/parse failure degrades to
/// `Err`/`None` ("not reachable") — never a panic. The pure parsers above do the
/// actual mapping; this only does I/O.
#[derive(Debug, Clone)]
pub struct LemonadeCollector {
    host: String,
    port: u16,
    client: Client,
}

impl LemonadeCollector {
    pub fn new(host: impl Into<String>, port: u16, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            host: host.into(),
            port,
            client,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}:{}{path}", self.host, self.port)
    }

    async fn get_text(&self, path: &str) -> Result<String> {
        let url = self.url(path);
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
        resp.text()
            .await
            .map_err(|e| CollectorError::Transport(format!("body {url}: {e}")))
    }

    /// Scrape `/api/v1/stats` → an `InstanceSample` (rate → `gen_tps`).
    pub async fn fetch_stats(&self) -> Result<InstanceSample> {
        Ok(parse_stats(&self.get_text(LEMONADE_STATS_PATH).await?))
    }

    /// Scrape `/api/v1/health` → the loaded model name (if any).
    pub async fn fetch_health_model(&self) -> Result<Option<String>> {
        Ok(parse_health_model(
            &self.get_text(LEMONADE_HEALTH_PATH).await?,
        ))
    }

    /// Probe the endpoint: if `/api/v1/health` answers, return a `DiscoveredService`
    /// for it (model name from health, falling back to "lemonade"); otherwise
    /// `None` (endpoint absent/unreachable) — a clean no-op, no panic.
    pub async fn discover(&self) -> Option<DiscoveredService> {
        match self.fetch_health_model().await {
            Ok(model) => Some(lemonade_service(
                &self.host,
                self.port,
                model.as_deref().unwrap_or(""),
            )),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative `/api/v1/stats` response (lemonade-server.ai/docs/api).
    const STATS_FIXTURE: &str = r#"{
        "time_to_first_token": 2.14,
        "tokens_per_second": 33.33,
        "input_tokens": 128,
        "output_tokens": 5,
        "prompt_tokens": 133,
        "decode_token_times": [0.03, 0.03, 0.03, 0.03, 0.03]
    }"#;

    // Representative `/api/v1/health` response.
    const HEALTH_FIXTURE: &str = r#"{
        "status": "ok",
        "version": "9.3.3",
        "model_loaded": "Llama-3.2-1B-Instruct-Hybrid",
        "all_models_loaded": []
    }"#;

    #[test]
    fn parse_stats_maps_tokens_per_second_to_gen_tps() {
        let sample = parse_stats(STATS_FIXTURE);
        assert_eq!(sample.gen_tps, Some(33.33));
        // Lemonade does not expose these — they must stay None (not zero).
        assert_eq!(sample.kv_cache_usage_pct, None);
        assert_eq!(sample.running_reqs, None);
        assert_eq!(sample.waiting_reqs, None);
        assert_eq!(sample.gen_tokens_total, None);
    }

    #[test]
    fn parse_stats_struct_captures_full_metric_set() {
        let stats = parse_stats_struct(STATS_FIXTURE);
        assert_eq!(stats.time_to_first_token, Some(2.14));
        assert_eq!(stats.tokens_per_second, Some(33.33));
        assert_eq!(stats.input_tokens, Some(128));
        assert_eq!(stats.output_tokens, Some(5));
        assert_eq!(stats.prompt_tokens, Some(133));
        assert_eq!(stats.decode_token_times.unwrap().len(), 5);
    }

    #[test]
    fn parse_health_model_extracts_loaded_model() {
        assert_eq!(
            parse_health_model(HEALTH_FIXTURE).as_deref(),
            Some("Llama-3.2-1B-Instruct-Hybrid")
        );
    }

    #[test]
    fn malformed_or_empty_input_is_graceful_not_panic() {
        // Garbage / empty / wrong-shape → all-None sample, no panic.
        for body in [
            "",
            "not json",
            "{}",
            "[1,2,3]",
            r#"{"tokens_per_second": "x"}"#,
        ] {
            let sample = parse_stats(body);
            assert_eq!(sample.gen_tps, None, "body {body:?}");
            assert!(sample.kv_cache_usage_pct.is_none());
        }
        assert_eq!(parse_health_model("not json"), None);
        assert_eq!(parse_health_model(r#"{"model_loaded": ""}"#), None);
    }

    #[test]
    fn fixture_stats_plus_health_becomes_an_instance() {
        // The fixture→Instance anchor: no network. Health gives the model name,
        // /stats gives the live rate → an Instances-tab row.
        let model = parse_health_model(HEALTH_FIXTURE).expect("model");
        let inst = lemonade_instance("127.0.0.1", LEMONADE_PORT, &model, STATS_FIXTURE);
        assert_eq!(inst.model_name, "Llama-3.2-1B-Instruct-Hybrid");
        assert_eq!(inst.container_name, "lemonade");
        assert_eq!(inst.container_id, "lemonade-127.0.0.1-13305");
        assert_eq!(inst.port, Some(13305));
        assert_eq!(inst.gen_tps, Some(33.33));
        assert_eq!(
            inst.status,
            rocm_dash_core::metrics::InstanceStatus::Running
        );
        // Lemonade exposes no KV/req metrics → those stay None on the Instance.
        assert_eq!(inst.kv_cache_usage_pct, None);
        assert_eq!(inst.running_reqs, None);
    }

    #[tokio::test]
    async fn discover_is_clean_noop_when_endpoint_absent() {
        // Probe a port with no server → connection refused → None, no panic.
        // (Port 0 is never a live listener; the GET fails fast.)
        let collector = LemonadeCollector::new("127.0.0.1", 1, Duration::from_millis(200));
        assert!(collector.discover().await.is_none());
        assert!(collector.fetch_stats().await.is_err());
    }

    /// Integration-gated: scrape a REAL local Lemonade server (start it first via
    /// `rocm skill run install-lemonade --apply`). Not run in CI.
    #[tokio::test]
    #[ignore = "requires a running Lemonade server on :13305"]
    async fn live_lemonade_scrape_produces_instance() {
        let collector =
            LemonadeCollector::new("127.0.0.1", LEMONADE_PORT, Duration::from_millis(1500));
        let svc = collector.discover().await.expect("lemonade reachable");
        assert_eq!(svc.port, Some(LEMONADE_PORT));
        // A stats scrape returns a sample (gen_tps may be None until an inference).
        let _sample = collector.fetch_stats().await.expect("stats scrape");
    }
}
