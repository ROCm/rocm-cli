// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Lemonade Server collector — pure parsers over the Lemonade REST API.
//!
//! Schema verified against the official docs (lemonade-server.ai/docs/api):
//! - `GET /api/v1/stats` → per-request performance metrics:
//!   `time_to_first_token`, `tokens_per_second`, `input_tokens`, `output_tokens`,
//!   `prompt_tokens`, `decode_token_times[]`. (No KV-cache / running / waiting —
//!   Lemonade does not expose those; those `InstanceSample` fields stay `None`.)
//! - `GET /api/v1/health` → `status` + `all_models_loaded`. Legacy servers
//!   report only `model_loaded` (most recent access, including load/inference).
//!
//! The parsers (`parse_stats`, `parse_health`) are **pure** (serde only)
//! and the deterministic, fixture-tested anchor. The async scrape
//! ([`LemonadeCollector`]) fetches those bodies over HTTP and returns transport
//! errors to the runner, which supplies endpoint context in snapshot warnings.
//! Discovery of a never-seen absent external endpoint remains a quiet no-op.

use std::time::Duration;

use reqwest::Client;
use rocm_dash_core::traits::{CollectorError, DiscoveredService, InstanceSample, Result};
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

/// Lemonade health response. `all_models_loaded`, when present, is the
/// authoritative loaded set. Older servers report only `model_loaded`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LemonadeHealth {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub model_loaded: Option<String>,
    #[serde(default)]
    pub all_models_loaded: Option<Vec<serde_json::Value>>,
}

impl LemonadeHealth {
    /// Sorted, deduplicated loaded model names, only from a healthy response.
    /// An empty router is healthy but does not constitute a serving model.
    pub fn loaded_models(&self) -> Vec<String> {
        if self.status.as_deref() != Some("ok") {
            return Vec::new();
        }
        let names: Vec<&str> = match &self.all_models_loaded {
            Some(models) => models
                .iter()
                .filter_map(|model| {
                    model.as_str().or_else(|| {
                        ["model_name", "id", "name"]
                            .into_iter()
                            .filter_map(|field| {
                                model.get(field).and_then(serde_json::Value::as_str)
                            })
                            .find(|name| !name.trim().is_empty())
                    })
                })
                .collect(),
            None => self.model_loaded.as_deref().into_iter().collect(),
        };
        names
            .into_iter()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Singleton attribution for server-global last-request stats. Health's
    /// `model_loaded` means last access (load OR inference), not last request,
    /// so it cannot disambiguate multiple loaded models. A conflicting explicit
    /// name also rejects the singleton fallback.
    pub fn stats_model(&self) -> Option<String> {
        let loaded = self.loaded_models();
        if loaded.len() != 1 {
            return None;
        }
        if let Some(last) = self
            .model_loaded
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            return loaded.into_iter().find(|name| name == last);
        }
        loaded.into_iter().next()
    }
}

/// Map a failed request to a [`CollectorError`]. A refused or unanswered
/// connection becomes `Io(ConnectionRefused)`, the typed marker for "nothing is
/// listening" that [`is_connect_error`] reads; every other failure stays a
/// `Transport` error. A timeout counts as unanswered because Windows does not
/// refuse a connection to a closed local port: the attempt hangs until the
/// request timeout.
fn send_error(url: &str, error: &reqwest::Error) -> CollectorError {
    let message = format!("GET {url}: {error}");
    if error.is_connect() || error.is_timeout() {
        CollectorError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            message,
        ))
    } else {
        CollectorError::Transport(message)
    }
}

/// True when `error` means no server accepted the connection, as opposed to a
/// server that answered with an HTTP status or a broken body.
pub fn is_connect_error(error: &CollectorError) -> bool {
    matches!(error, CollectorError::Io(io) if io.kind() == std::io::ErrorKind::ConnectionRefused)
}

/// PURE: parse a `/api/v1/stats` body into an [`InstanceSample`].
///
/// Lemonade reports the `tokens_per_second` rate of the most recent request (mapped to `gen_tps`;
/// it is not a live rate: the same value is reported until another request completes, so the
/// dashboard keeps showing the last request's rate until a new request replaces it) rather than a
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

/// PURE: parse a health body, rejecting invalid/wrong-shape input to default.
pub fn parse_health(body: &str) -> LemonadeHealth {
    let Ok(value @ serde_json::Value::Object(_)) = serde_json::from_str(body) else {
        return LemonadeHealth::default();
    };
    if value
        .get("all_models_loaded")
        .is_some_and(|models| !models.is_array())
    {
        return LemonadeHealth::default();
    }
    serde_json::from_value(value).unwrap_or_default()
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

/// A model identity within an external endpoint. Encoding the model's UTF-8
/// bytes avoids separator collisions and keeps identity stable across reorder.
pub fn lemonade_model_id(host: &str, port: u16, model_name: &str) -> String {
    use std::fmt::Write as _;
    let mut id = format!("{}-model-", lemonade_container_id(host, port));
    for byte in model_name.as_bytes() {
        let _ = write!(id, "{byte:02x}");
    }
    id
}

/// PURE: build metadata for an external loaded model.
///
/// Callers must establish healthy loaded-model evidence first; answering HTTP
/// alone is insufficient. `model_name` must be a trimmed, non-empty name, as
/// produced by [`LemonadeHealth::loaded_models`]. Lemonade exposes no TP metadata.
pub fn lemonade_service(host: &str, port: u16, model_name: &str) -> DiscoveredService {
    DiscoveredService {
        container_id: lemonade_model_id(host, port, model_name),
        container_name: "lemonade".to_string(),
        model_name: model_name.to_string(),
        port: Some(port),
        tensor_parallel_size: 1,
        status: rocm_dash_core::metrics::InstanceStatus::Ready,
        ..Default::default()
    }
}

/// PURE: map a health payload to one discovered service per loaded model.
/// Invalid, unhealthy or empty payloads yield no rows. The daemon runner and
/// the tests both use this single mapping.
pub fn services_from_health(
    host: &str,
    port: u16,
    health: &LemonadeHealth,
) -> Vec<DiscoveredService> {
    health
        .loaded_models()
        .iter()
        .map(|model| lemonade_service(host, port, model))
        .collect()
}

/// Async scrape of a local Lemonade endpoint. Network/parse failure degrades to
/// transport errors or an empty loaded set, never a panic. The pure parsers above do the
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
        let host = if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("http://{host}:{}{path}", self.port)
    }

    /// GET `path`, falling back to the legacy route on a 404. `Ok(None)` means
    /// neither route exists: the server does not speak the Lemonade protocol.
    async fn get_text_opt(&self, path: &str, auth_optional: bool) -> Result<Option<String>> {
        let mut url = self.url(path);
        let mut resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| send_error(&url, &e))?;
        // Router versions expose either /api/v1 or /v1. Only a missing route
        // warrants compatibility fallback, never an authentication or health
        // failure. This does not add a llama-server metrics protocol.
        if resp.status() == reqwest::StatusCode::NOT_FOUND
            && let Some(legacy_path) = path.strip_prefix("/api")
        {
            url = self.url(legacy_path);
            resp = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| send_error(&url, &e))?;
        }
        if resp.status() == reqwest::StatusCode::NOT_FOUND
            || (auth_optional
                && matches!(
                    resp.status(),
                    reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
                ))
        {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(CollectorError::Transport(format!(
                "GET {url}: status {}",
                resp.status()
            )));
        }
        resp.text()
            .await
            .map(Some)
            .map_err(|e| CollectorError::Transport(format!("body {url}: {e}")))
    }

    async fn get_text(&self, path: &str) -> Result<String> {
        self.get_text_opt(path, false).await?.ok_or_else(|| {
            CollectorError::Transport(format!("GET {}: status 404 Not Found", self.url(path)))
        })
    }

    /// Scrape `/api/v1/stats` → an `InstanceSample` (rate → `gen_tps`).
    pub async fn fetch_stats(&self) -> Result<InstanceSample> {
        Ok(parse_stats(&self.get_text(LEMONADE_STATS_PATH).await?))
    }

    /// Registry-target lookup: `Ok(None)` if the Lemonade health route is absent
    /// or requires credentials (401/403). The registry still supplies ownership
    /// and status; this does not establish health or permit a stats scrape.
    /// Transport failures and 5xx responses are still errors. External discovery
    /// uses [`Self::fetch_health`] instead, where auth rejection remains an error.
    pub async fn fetch_health_if_router(&self) -> Result<Option<LemonadeHealth>> {
        Ok(self
            .get_text_opt(LEMONADE_HEALTH_PATH, true)
            .await?
            .map(|body| parse_health(&body)))
    }

    /// Fetch the health payload. Invalid payloads carry no loaded models.
    pub async fn fetch_health(&self) -> Result<LemonadeHealth> {
        Ok(parse_health(&self.get_text(LEMONADE_HEALTH_PATH).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same path the daemon runner takes: health fetch, then the shared mapping.
    async fn discovered(collector: &LemonadeCollector) -> Vec<DiscoveredService> {
        match collector.fetch_health().await {
            Ok(health) => services_from_health(&collector.host, collector.port, &health),
            Err(_) => Vec::new(),
        }
    }

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
        "all_models_loaded": [{"model_name":"Llama-3.2-1B-Instruct-Hybrid"}]
    }"#;

    #[test]
    fn loaded_set_and_stats_attribution_are_conservative() {
        let health = parse_health(
            r#"{"status":"ok","all_models_loaded":[" Model-B ",{"id":"Model-A"},{"name":"Model-B"},{"model_name":"","id":"Model-C"},null,7,{}]}"#,
        );
        assert_eq!(health.loaded_models(), ["Model-A", "Model-B", "Model-C"]);
        assert_eq!(health.stats_model(), None);
        let health = parse_health(
            r#"{"status":"ok","model_loaded":"Model-B","all_models_loaded":["Model-A","Model-B"]}"#,
        );
        assert_eq!(
            health.stats_model(),
            None,
            "last access can be a load, not a request"
        );
        for body in [
            r#"{"status":"ok","all_models_loaded":["Model-A"]}"#,
            r#"{"status":"ok","model_loaded":"Model-A"}"#,
        ] {
            assert_eq!(parse_health(body).stats_model().as_deref(), Some("Model-A"));
        }
        assert_eq!(
            parse_health(
                r#"{"status":"ok","model_loaded":"Model-B","all_models_loaded":["Model-A"]}"#
            )
            .stats_model(),
            None
        );
        for body in [
            r#"{"status":"ok","all_models_loaded":{},"model_loaded":"Model-A"}"#,
            r#"{"status":"bad","model_loaded":"Model-A"}"#,
            r#"{"model_loaded":"Model-A"}"#,
        ] {
            assert!(parse_health(body).loaded_models().is_empty());
        }
    }

    #[tokio::test]
    async fn health_if_router_distinguishes_missing_route_from_failure() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        // A plain llama-server: neither Lemonade health route exists.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"status":"ok"}"#))
            .mount(&server)
            .await;
        let collector =
            LemonadeCollector::new("127.0.0.1", server.address().port(), Duration::from_secs(1));
        assert!(
            collector
                .fetch_health_if_router()
                .await
                .expect("missing routes are not an error")
                .is_none()
        );
        assert!(collector.fetch_health().await.is_err());
        // A router answering on the legacy route is still a router.
        server.reset().await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"status":"ok","model_loaded":"Model-A"}"#),
            )
            .mount(&server)
            .await;
        let health = collector.fetch_health_if_router().await.unwrap().unwrap();
        assert_eq!(health.loaded_models(), ["Model-A"]);
        // Only registry lookup tolerates auth rejection, without claiming health.
        for status in [401, 403] {
            server.reset().await;
            Mock::given(method("GET"))
                .and(path(LEMONADE_HEALTH_PATH))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
            assert!(collector.fetch_health_if_router().await.unwrap().is_none());
            assert!(collector.fetch_health().await.is_err());
        }
        // Unrelated server failures must still warn.
        for status in [500, 503] {
            server.reset().await;
            Mock::given(method("GET"))
                .and(path(LEMONADE_HEALTH_PATH))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
            assert!(
                collector.fetch_health_if_router().await.is_err(),
                "status {status} must stay an error"
            );
        }
    }

    #[tokio::test]
    async fn health_http_failure_never_falls_back_to_successful_route() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        for status in [401, 403, 500, 503] {
            server.reset().await;
            Mock::given(method("GET"))
                .and(path(LEMONADE_HEALTH_PATH))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/v1/health"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(r#"{"status":"ok","model_loaded":"Model-A"}"#),
                )
                .expect(0)
                .mount(&server)
                .await;
            let collector = LemonadeCollector::new(
                "127.0.0.1",
                server.address().port(),
                Duration::from_secs(1),
            );
            assert!(discovered(&collector).await.is_empty());
            server.verify().await;
        }
    }

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
    fn parse_health_extracts_loaded_model() {
        assert_eq!(
            parse_health(HEALTH_FIXTURE).stats_model().as_deref(),
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
        assert!(parse_health("not json").loaded_models().is_empty());
        assert!(
            parse_health(r#"{"model_loaded": ""}"#)
                .loaded_models()
                .is_empty()
        );
    }

    #[test]
    fn fixture_stats_plus_health_becomes_an_instance() {
        // The fixture→Instance anchor: no network. Health gives the model name,
        // /stats gives the live rate → an Instances-tab row.
        let model = parse_health(HEALTH_FIXTURE).stats_model().expect("model");
        let svc = lemonade_service("127.0.0.1", LEMONADE_PORT, &model);
        let inst = rocm_dash_core::traits::merge_instance(&svc, &parse_stats(STATS_FIXTURE), 0, 0);
        assert_eq!(inst.model_name, "Llama-3.2-1B-Instruct-Hybrid");
        assert_eq!(inst.container_name, "lemonade");
        assert_eq!(
            inst.container_id,
            lemonade_model_id("127.0.0.1", 13305, &model)
        );
        assert_eq!(inst.port, Some(13305));
        assert_eq!(inst.gen_tps, Some(33.33));
        assert_eq!(inst.status, rocm_dash_core::metrics::InstanceStatus::Ready);
        // Lemonade exposes no KV/req metrics → those stay None on the Instance.
        assert_eq!(inst.kv_cache_usage_pct, None);
        assert_eq!(inst.running_reqs, None);
    }

    #[tokio::test]
    async fn discovery_rejects_invalid_health_and_healthy_empty_router() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let collector =
            LemonadeCollector::new("127.0.0.1", server.address().port(), Duration::from_secs(1));
        for body in [
            "not json",
            "{}",
            "[]",
            "null",
            r#"{"status":"error","model_loaded":"Model-A"}"#,
            r#"{"status":"ok","all_models_loaded":[]}"#,
            r#"{"status":"ok","all_models_loaded":[],"model_loaded":"Old-Model"}"#,
            r#"{"status":"ok","model_loaded":"   "}"#,
        ] {
            server.reset().await;
            Mock::given(method("GET"))
                .and(path(LEMONADE_HEALTH_PATH))
                .respond_with(ResponseTemplate::new(200).set_body_string(body))
                .mount(&server)
                .await;
            assert!(
                discovered(&collector).await.is_empty(),
                "health {body:?} must not advertise a serving model"
            );
        }
    }

    #[tokio::test]
    async fn discovery_recognizes_modern_loaded_model_without_legacy_field() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LEMONADE_HEALTH_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"status":"ok","all_models_loaded":[{"model_name":"Model-A"}]}"#,
            ))
            .mount(&server)
            .await;
        let collector =
            LemonadeCollector::new("127.0.0.1", server.address().port(), Duration::from_secs(1));
        assert_eq!(discovered(&collector).await[0].model_name, "Model-A");
    }

    #[tokio::test]
    async fn discovery_accepts_v1_health_only_when_api_prefix_is_absent() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"status":"ok","model_loaded":"Model-A"}"#),
            )
            .mount(&server)
            .await;
        let collector =
            LemonadeCollector::new("127.0.0.1", server.address().port(), Duration::from_secs(1));
        assert_eq!(discovered(&collector).await[0].model_name, "Model-A");
    }

    #[tokio::test]
    async fn v1_fallback_status_failure_names_the_requested_v1_url() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LEMONADE_HEALTH_PATH))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let port = server.address().port();
        let collector = LemonadeCollector::new("127.0.0.1", port, Duration::from_secs(1));
        let error = collector.fetch_health().await.unwrap_err().to_string();
        let wanted = format!("http://127.0.0.1:{port}/v1/health");
        assert!(error.contains(&wanted), "{error}");
        assert!(!error.contains("/api/v1/health"), "{error}");
    }

    #[tokio::test]
    async fn v1_fallback_transport_failure_names_host_port_and_path() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        // 404 on the primary route, then the retry is cut off by the client
        // timeout so it fails at the transport level.
        Mock::given(method("GET"))
            .and(path(LEMONADE_HEALTH_PATH))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
            .mount(&server)
            .await;
        let port = server.address().port();
        let collector = LemonadeCollector::new("127.0.0.1", port, Duration::from_millis(200));
        let error = collector.fetch_health().await.unwrap_err().to_string();
        let wanted = format!("GET http://127.0.0.1:{port}/v1/health");
        assert!(error.contains(&wanted), "{error}");
    }

    #[tokio::test]
    async fn health_and_stats_return_transport_errors_when_endpoint_absent() {
        // Reserve then release a random port: an absent listener yields health
        // and stats transport errors, never a panic.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let collector = LemonadeCollector::new("127.0.0.1", port, Duration::from_millis(200));
        assert!(collector.fetch_health().await.is_err());
        assert!(collector.fetch_stats().await.is_err());
    }

    /// Windows does not refuse a connection to a closed local port: the attempt
    /// hangs until the request timeout. Both outcomes mean nothing answered, so
    /// both must read as "absent" and not as a broken server.
    #[tokio::test]
    async fn closed_and_unanswered_endpoints_both_read_as_nothing_listening() {
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let closed_port = closed.local_addr().unwrap().port();
        drop(closed);
        let refused = LemonadeCollector::new("127.0.0.1", closed_port, Duration::from_millis(200))
            .fetch_health()
            .await
            .unwrap_err();
        assert!(is_connect_error(&refused), "{refused}");

        // Accepts the connection at the kernel level but never answers.
        let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let silent_port = silent.local_addr().unwrap().port();
        let timed_out =
            LemonadeCollector::new("127.0.0.1", silent_port, Duration::from_millis(200))
                .fetch_health()
                .await
                .unwrap_err();
        assert!(is_connect_error(&timed_out), "{timed_out}");
        drop(silent);
    }

    /// An HTTP status error is a server that answered, so it is not "absent".
    #[tokio::test]
    async fn an_answering_server_with_an_error_status_is_not_nothing_listening() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let error =
            LemonadeCollector::new("127.0.0.1", server.address().port(), Duration::from_secs(1))
                .fetch_health()
                .await
                .unwrap_err();
        assert!(!is_connect_error(&error), "{error}");
    }

    /// Integration-gated: scrape a REAL local Lemonade server (start it first via
    /// `rocm skill run install-lemonade --apply`). Not run in CI.
    #[tokio::test]
    #[ignore = "requires a running Lemonade server on :13305"]
    async fn live_lemonade_scrape_produces_instance() {
        let collector =
            LemonadeCollector::new("127.0.0.1", LEMONADE_PORT, Duration::from_millis(1500));
        let svcs = discovered(&collector).await;
        let svc = svcs.first().expect("lemonade reachable");
        assert_eq!(svc.port, Some(LEMONADE_PORT));
        // A stats scrape returns a sample (gen_tps may be None until an inference).
        let _sample = collector.fetch_stats().await.expect("stats scrape");
    }
}
