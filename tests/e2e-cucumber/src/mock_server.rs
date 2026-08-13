// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use serde_json::{Value, json};

use crate::http_server::{self, ServerHandle};

/// How often [`MockServer::wait_for_chat_request`] re-checks the captured
/// request while waiting for the client to POST.
const CHAT_REQUEST_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Controls what each `/metrics` scrape returns in a [`ScriptedMetrics`]-backed mock.
///
/// Every variant is a deterministic primitive that tests can switch to at
/// runtime without restarting the server, covering all EAI-7960 contract states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsMode {
    /// Every scrape increments the cumulative counter by 20 tok/scrape; yields a
    /// positive `gen_tps` from the second scrape onward (busy/running instance).
    Growing,
    /// Counter frozen at the given cumulative value; `gen_tps` approaches 0.0
    /// after the rate window (idle instance still reporting a counter).
    Unchanged(u64),
    /// HTTP 200 with a valid Prometheus body that omits
    /// `vllm:generation_tokens_total` entirely. The runner's parser returns
    /// `sample.gen_tokens_total = None`; `gen_tps` stays `None` and the
    /// validity timer is NOT refreshed (missing ≠ failure, ≠ zero).
    Omitted,
    /// HTTP 200 with a structurally invalid (unparseable) body. All sample
    /// fields come back `None`; semantically equivalent to `Omitted` at the
    /// runner level but triggered by a parse failure rather than a missing key.
    Malformed,
    /// HTTP 503 transport failure. The runner's failure path clears
    /// `prev_gen_tokens` and sets `gen_tps = None` immediately (EAI-7960 bug).
    ///
    /// Under the EAI-7960 contract, `gen_tps` **must** remain held at its last
    /// observed value for `clamp(3 × instance_tick, 6 s, 30 s)` before clearing.
    Failure,
    /// Counter resets to the given value, which must be below the accumulated
    /// Growing total. `gen_tps_from_delta` returns `None` (cur < prev_val guard)
    /// and re-baselines to the new lower value for recovery.
    Reset(u64),
    /// Growing counter but `running_reqs = 0` — engine is alive but idle,
    /// no in-flight requests. Distinct from `Unchanged` (counter still grows)
    /// and `Omitted` (counter is present but running_reqs shows idle load).
    RunningIdle,
}

/// Scriptable state for a mock `/metrics` endpoint. Unlike [`MetricsCounter`],
/// the mode can be switched at runtime so a single test can drive the full
/// Growing → Failure → Growing lifecycle without restarting the server.
#[derive(Debug)]
struct ScriptedMetrics {
    /// Monotonically-advancing tick for Growing mode; mirrors [`MetricsCounter`].
    ticks: AtomicU64,
    /// Current mode; swapped by the test thread, read by the Axum handler thread.
    mode: Mutex<MetricsMode>,
    /// How many HTTP 503 failure responses have been served. Used by
    /// `MockServer::metrics_failure_count` so tests can poll deterministically
    /// (confirm the failure actually landed) instead of sleeping a fixed wall time.
    failure_count: AtomicU64,
}

impl ScriptedMetrics {
    const fn new() -> Self {
        Self {
            ticks: AtomicU64::new(0),
            mode: Mutex::new(MetricsMode::Growing),
            failure_count: AtomicU64::new(0),
        }
    }

    fn set_mode(&self, mode: MetricsMode) {
        *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = mode;
    }

    fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Advance one scrape. Returns `Some(body)` for 200-OK modes,
    /// `None` for `Failure` (the handler maps `None` → HTTP 503).
    fn scrape(&self) -> Option<String> {
        let mode = *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match mode {
            MetricsMode::Growing | MetricsMode::RunningIdle => {
                let tick = self.ticks.fetch_add(1, Ordering::Relaxed) + 1;
                let gen_tokens_total = tick * 20;
                let running = i32::from(mode != MetricsMode::RunningIdle);
                let ttft_sum_s = tick as f64 * 0.050;
                let tpot_sum_s = tick as f64 * 20.0 * 0.020;
                let tpot_count = gen_tokens_total;
                Some(format!(
                    "\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} {running}
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model=\"mock\"}} 0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{{model=\"mock\"}} 0.25
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {gen_tokens_total}
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}
# HELP vllm:time_per_output_token_seconds Histogram of time per output token.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {tpot_count}
"
                ))
            }
            MetricsMode::Unchanged(n) => {
                let tick = n.max(1);
                let ttft_sum_s = tick as f64 * 0.050;
                let tpot_sum_s = tick as f64 * 20.0 * 0.020;
                let tpot_count = n;
                Some(format!(
                    "\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} 1
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {n}
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}
# HELP vllm:time_per_output_token_seconds Histogram of time per output token.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {tpot_count}
"
                ))
            }
            MetricsMode::Omitted => {
                // Valid Prometheus body with no generation_tokens_total line.
                // runner: sample.gen_tokens_total = None → gen_tps stays None,
                // prev_gen_tokens is NOT cleared (success path, not failure path).
                Some(
                    "\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model=\"mock\"} 1
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model=\"mock\"} 0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model=\"mock\"} 0.25
"
                    .to_string(),
                )
            }
            MetricsMode::Malformed => {
                // HTTP 200 with structurally invalid body; parser returns all-None.
                Some(
                    "# malformed: not valid prometheus exposition format\n!!!invalid!!!\n"
                        .to_string(),
                )
            }
            MetricsMode::Failure => {
                self.failure_count.fetch_add(1, Ordering::Relaxed);
                None // handler maps None → HTTP 503
            }
            MetricsMode::Reset(n) => {
                // Counter explicitly lower than any Growing accumulation.
                let tick = n.max(1);
                let ttft_sum_s = tick as f64 * 0.050;
                let tpot_sum_s = tick as f64 * 20.0 * 0.020;
                Some(format!(
                    "\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {n}
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}
# HELP vllm:time_per_output_token_seconds Histogram of time per output token.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {tick}
"
                ))
            }
        }
    }
}

/// Unit tests for [`ScriptedMetrics`] body generation — verify the raw payload
/// each mode produces matches the contract before the runner even sees it.
#[cfg(test)]
mod scripted_metrics_tests {
    use super::{MetricsMode, ScriptedMetrics};
    use std::sync::atomic::Ordering;

    #[test]
    fn growing_increments_counter_and_running_reqs_is_1() {
        let s = ScriptedMetrics::new();
        let body = s.scrape().expect("Growing must return Some");
        assert!(
            body.contains("generation_tokens_total"),
            "counter line absent"
        );
        assert!(body.contains("} 20"), "first tick should be 20 tokens");
        assert!(
            body.contains("num_requests_running") && body.contains("} 1"),
            "running_reqs must be 1 for Growing"
        );
        assert_eq!(s.ticks.load(Ordering::Relaxed), 1, "tick counter advanced");
    }

    #[test]
    fn running_idle_counter_grows_but_running_reqs_is_0() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::RunningIdle);
        let body = s.scrape().expect("RunningIdle must return Some");
        assert!(
            body.contains("generation_tokens_total"),
            "counter line absent"
        );
        // running_reqs = 0 distinguishes idle from busy
        let running_line = body
            .lines()
            .find(|l| l.starts_with("vllm:num_requests_running"))
            .unwrap_or("");
        assert!(
            running_line.ends_with("} 0"),
            "running_reqs must be 0 for RunningIdle, got: {running_line}"
        );
    }

    #[test]
    fn unchanged_always_returns_same_counter() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::Unchanged(1_000));
        let b1 = s.scrape().expect("Unchanged must return Some");
        let b2 = s.scrape().expect("Unchanged must return Some on 2nd call");
        // Both should reference the same fixed value
        assert!(b1.contains("} 1000"), "first scrape counter mismatch");
        assert!(b2.contains("} 1000"), "second scrape counter mismatch");
        assert_eq!(
            s.ticks.load(Ordering::Relaxed),
            0,
            "ticks must not advance for Unchanged"
        );
    }

    #[test]
    fn omitted_body_has_no_generation_tokens_total_line() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::Omitted);
        let body = s.scrape().expect("Omitted returns HTTP 200");
        assert!(
            !body.contains("generation_tokens_total"),
            "Omitted body must not contain the counter key"
        );
        // But other gauge lines are present (valid subset of a Prometheus payload).
        assert!(
            body.contains("num_requests_running"),
            "gauge lines must be present"
        );
    }

    #[test]
    fn malformed_body_is_returned_with_some_not_none() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::Malformed);
        // Malformed is HTTP 200 — scrape() must return Some, not None.
        let body = s.scrape().expect("Malformed returns HTTP 200 (Some)");
        assert!(!body.is_empty(), "Malformed body must be non-empty");
        assert!(
            !body.contains("generation_tokens_total"),
            "Malformed body must not accidentally contain the counter key"
        );
    }

    #[test]
    fn failure_returns_none_and_increments_failure_count() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::Failure);
        assert!(s.scrape().is_none(), "Failure must return None → HTTP 503");
        assert_eq!(s.failure_count(), 1, "failure_count must increment");
        let _ = s.scrape();
        assert_eq!(s.failure_count(), 2, "failure_count accumulates");
    }

    #[test]
    fn reset_returns_specified_low_counter() {
        let s = ScriptedMetrics::new();
        s.set_mode(MetricsMode::Reset(3));
        let body = s.scrape().expect("Reset returns HTTP 200");
        assert!(
            body.contains("} 3"),
            "Reset body must carry the specified low counter value"
        );
    }

    #[test]
    fn mode_switch_growing_to_failure_to_growing() {
        let s = ScriptedMetrics::new();
        // Growing → positive counter
        assert!(s.scrape().is_some());
        // Failure → HTTP 503
        s.set_mode(MetricsMode::Failure);
        assert!(s.scrape().is_none());
        assert_eq!(s.failure_count(), 1);
        // Recovery (back to Growing) → resumes ticking
        s.set_mode(MetricsMode::Growing);
        let body = s.scrape().expect("recovery scrape must succeed");
        assert!(
            body.contains("generation_tokens_total"),
            "counter resumes after recovery"
        );
    }
}
#[derive(Clone)]
struct ServerState {
    model_name: String,
    /// `Some` only when this server was started via
    /// [`MockServer::start_with_metrics`]; drives the `/metrics` route. Kept as
    /// an `Option` (rather than always registering the route) so a plain
    /// [`MockServer::start`] — used by every scenario that doesn't care about
    /// dashboard metrics — gets a 404 for `/metrics`, matching a vLLM instance
    /// that scenario never asked to simulate.
    metrics: Option<Arc<MetricsCounter>>,
    /// `Some` only when started via [`MockServer::start_with_scripted_metrics`];
    /// drives the scriptable `/metrics` route. Mutually exclusive with `metrics`.
    scripted_metrics: Option<Arc<ScriptedMetrics>>,
    /// The most recently received `/v1/chat/completions` request body, shared
    /// with the `MockServer` handle so scenarios can assert on exactly what the
    /// CLI sent — not just on the (fixed) canned reply, which would silently
    /// mask a corrupted or missing prompt. `None` until a chat request arrives.
    last_chat_request: Arc<Mutex<Option<Value>>>,
    /// Every URI path a chat request has arrived on, in order.
    ///
    /// This server answers chat on BOTH `/v1/chat/completions` and the
    /// unversioned `/chat/completions`, so a scenario that only checks "did the
    /// request succeed" cannot tell the two apart — and a client that drops the
    /// `/v1` prefix would pass while 404ing against a real engine. Recording the
    /// path lets a scenario assert the versioned route was the one used.
    chat_paths: Arc<Mutex<Vec<String>>>,
}

/// Deterministic, monotonically-advancing state behind the mock `/metrics`
/// route, so successive scrapes exercise the daemon's rate/average windowing
/// (`gen_tps_from_delta`, `avg_ms_from_histogram` in
/// `rocm-dash-daemon::runner`) the same way a real vLLM would:
///   * `vllm:generation_tokens_total` strictly increases every scrape, so the
///     counter-delta windowing yields a positive, visible generation rate
///     from the second scrape onward (never zero/negative/`None`).
///   * the TTFT/TPOT histograms' `_sum`/`_count` pairs both advance by a fixed
///     amount per tick, so their ratio — and therefore the windowed average
///     latency — stays constant scrape over scrape instead of drifting.
struct MetricsCounter {
    ticks: AtomicU64,
}

impl MetricsCounter {
    const fn new() -> Self {
        Self {
            ticks: AtomicU64::new(0),
        }
    }

    /// Advance one scrape and render the resulting Prometheus exposition text.
    fn scrape(&self) -> String {
        // Start at 1 (not 0) so even the very first scrape already reports
        // non-zero cumulative counters, giving tests a realistic "already
        // serving" sample without a "first scrape is empty" special case.
        let tick = self.ticks.fetch_add(1, Ordering::Relaxed) + 1;

        // 20 generation tokens per scrape keeps gen_tps comfortably positive
        // even at the daemon's multi-second poll interval.
        let gen_tokens_total = tick * 20;
        // One request "completes" per scrape: a fixed 50ms TTFT and a fixed
        // 20ms/token TPOT over those 20 tokens. Sum and count both grow
        // linearly in `tick`, so Δsum/Δcount — the windowed average the
        // daemon reports — is the same constant on every pair of scrapes.
        let ttft_count = tick;
        let ttft_sum_s = tick as f64 * 0.050;
        let tpot_count = tick * 20;
        let tpot_sum_s = tick as f64 * 20.0 * 0.020;

        format!(
            "\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} 1
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model=\"mock\"}} 0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{{model=\"mock\"}} 0.25
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {gen_tokens_total}
# HELP vllm:time_to_first_token_seconds Histogram of time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {ttft_count}
# HELP vllm:time_per_output_token_seconds Histogram of time per output token.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {tpot_count}
"
        )
    }
}

#[derive(Debug)]
pub struct MockServer {
    server: ServerHandle,
    /// Shared with the running server's `ServerState`; see the field doc there.
    last_chat_request: Arc<Mutex<Option<Value>>>,
    /// Shared with the running server's `ServerState`; see the field doc there.
    chat_paths: Arc<Mutex<Vec<String>>>,
    /// `Some` only when started with [`MockServer::start_with_scripted_metrics`].
    scripted: Option<Arc<ScriptedMetrics>>,
}

impl MockServer {
    pub async fn start(model_name: &str) -> Self {
        Self::spawn(model_name, false).await
    }

    /// Like [`Self::start`], but also opts into a deterministic vLLM-flavoured
    /// `/metrics` route (see [`MetricsCounter`]) — for scenarios that exercise
    /// the dashboard's live generation-rate / TTFT / TPOT display against a
    /// served model. Plain [`Self::start`] registers no `/metrics` route at
    /// all, so it keeps returning a 404 there, same as before this method
    /// existed.
    pub async fn start_with_metrics(model_name: &str) -> Self {
        Self::spawn(model_name, true).await
    }

    /// Like [`Self::start_with_metrics`] but the `/metrics` endpoint starts in
    /// [`MetricsMode::Growing`] and can be switched to [`MetricsMode::Failure`]
    /// (and back) at runtime via [`Self::set_metrics_mode`] while the server is
    /// live. Use this for EAI-7960 scenarios that need to exercise the
    /// validity-window holding behaviour or the recovery path.
    pub async fn start_with_scripted_metrics(model_name: &str) -> Self {
        let last_chat_request = Arc::new(Mutex::new(None));
        let chat_paths = Arc::new(Mutex::new(Vec::new()));
        let scripted = Arc::new(ScriptedMetrics::new());
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: None,
            scripted_metrics: Some(Arc::clone(&scripted)),
            last_chat_request: Arc::clone(&last_chat_request),
            chat_paths: Arc::clone(&chat_paths),
        };
        let app = Router::new()
            .route("/v1/models", get(handle_models))
            .route("/models", get(handle_models))
            .route("/v1/chat/completions", post(handle_chat))
            .route("/chat/completions", post(handle_chat))
            .route("/metrics", get(handle_scripted_metrics))
            .with_state(state);
        Self {
            server: http_server::spawn(app).await,
            last_chat_request,
            scripted: Some(scripted),
            chat_paths,
        }
    }

    /// Switch the scripted `/metrics` endpoint mode while the server is running.
    ///
    /// # Panics
    /// Panics when the server was not started with
    /// [`Self::start_with_scripted_metrics`].
    pub fn set_metrics_mode(&self, mode: MetricsMode) {
        self.scripted
            .as_ref()
            .expect("set_metrics_mode requires start_with_scripted_metrics")
            .set_mode(mode);
    }

    /// How many HTTP 503 failure responses the scripted endpoint has served.
    /// Lets tests poll deterministically — confirm the failure landed — without
    /// relying on a fixed wall-time sleep.
    ///
    /// # Panics
    /// Panics when the server was not started with
    /// [`Self::start_with_scripted_metrics`].
    pub fn metrics_failure_count(&self) -> u64 {
        self.scripted
            .as_ref()
            .expect("metrics_failure_count requires start_with_scripted_metrics")
            .failure_count()
    }

    async fn spawn(model_name: &str, with_metrics: bool) -> Self {
        let last_chat_request = Arc::new(Mutex::new(None));
        let chat_paths = Arc::new(Mutex::new(Vec::new()));
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: with_metrics.then(|| Arc::new(MetricsCounter::new())),
            scripted_metrics: None,
            last_chat_request: Arc::clone(&last_chat_request),
            chat_paths: Arc::clone(&chat_paths),
        };

        let mut app = Router::new()
            .route("/v1/models", get(handle_models))
            .route("/models", get(handle_models))
            .route("/v1/chat/completions", post(handle_chat))
            .route("/chat/completions", post(handle_chat));
        if with_metrics {
            app = app.route("/metrics", get(handle_metrics));
        }
        let app = app.with_state(state);

        Self {
            server: http_server::spawn(app).await,
            scripted: None,
            last_chat_request,
            chat_paths,
        }
    }

    /// Every URI path a chat request has arrived on, in order.
    ///
    /// Use this to assert the client used the versioned `/v1/chat/completions`
    /// route: this server also answers the unversioned `/chat/completions`, so
    /// a bare "the request succeeded" assertion cannot tell them apart and would
    /// pass for a client that 404s against a real engine.
    #[must_use]
    pub fn chat_paths(&self) -> Vec<String> {
        self.chat_paths
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// A server that is reachable but rejects every request with `503`.
    ///
    /// Models an engine that is listening but cannot serve — the case a client
    /// must report rather than quietly recording zero measurements. Its
    /// `/v1/models` route rejects too, so callers must pass an explicit model.
    pub async fn start_rejecting() -> Self {
        async fn reject() -> axum::http::StatusCode {
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        }

        let app = Router::new()
            .route("/v1/models", get(reject))
            .route("/models", get(reject))
            .route("/v1/chat/completions", post(reject))
            .route("/chat/completions", post(reject));

        Self {
            server: http_server::spawn(app).await,
            last_chat_request: Arc::new(Mutex::new(None)),
            chat_paths: Arc::new(Mutex::new(Vec::new())),
            scripted: None,
        }
    }

    /// The OpenAI-compatible API root (`.../v1`) the CLI is pointed at.
    pub fn base_url(&self) -> String {
        self.server
            .url()
            .join("v1")
            .expect("v1 is a valid relative path")
            .to_string()
    }

    pub const fn port(&self) -> u16 {
        self.server.port()
    }

    /// The most recently received `/v1/chat/completions` request body, if any
    /// chat request has landed yet. Recovers a poisoned lock rather than
    /// propagating the panic: a torn-down request body is still the most
    /// recent one worth inspecting on assertion failure.
    pub fn last_chat_request(&self) -> Option<Value> {
        self.last_chat_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Poll for a chat request to arrive, so scenarios that assert on the exact
    /// request body don't race the TUI's async send. Returns the body once
    /// present, or `Err` with a diagnostic if none arrives within `timeout`.
    pub async fn wait_for_chat_request(&self, timeout: Duration) -> Result<Value, String> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(body) = self.last_chat_request() {
                return Ok(body);
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for a chat request"
                ));
            }
            tokio::time::sleep(CHAT_REQUEST_POLL_INTERVAL).await;
        }
    }

    /// Shut the server down explicitly. Equivalent to dropping it — the
    /// handle stops the server on drop — but says so at the call site.
    pub fn stop(self) {
        drop(self);
    }
}

/// Lifecycle fields of the on-disk managed-service record that vary across
/// callers.
///
/// The default matches a service that passed its readiness probe with no
/// process attached (the `rocm-demo-env` shape: `status: "ready"`, no
/// `startup_phase`, `supervisor_pid: 0`, `engine_pid: null`). Cucumber
/// scenarios that need a mid-startup record, or one the CLI's process-liveness
/// overlay keeps alive (pointed at this test process), override the relevant
/// fields via [`write_service_record_with`] instead of duplicating the whole
/// record shape.
#[derive(Debug, Clone, Copy)]
pub struct ServiceRecordOptions {
    pub status: &'static str,
    pub startup_phase: Option<&'static str>,
    pub supervisor_pid: u32,
    pub engine_pid: Option<u32>,
}

impl Default for ServiceRecordOptions {
    fn default() -> Self {
        Self {
            status: "ready",
            startup_phase: None,
            supervisor_pid: 0,
            engine_pid: None,
        }
    }
}

/// Write a managed-service record pointing the CLI at a mock server on `port`,
/// using [`ServiceRecordOptions::default`] (ready, no attached process).
///
/// Drops the JSON into `services_dir` (`<data>/services/`) exactly as `rocm serve
/// --managed` would. Shared by the cucumber `World` and the standalone
/// `rocm-demo-env` binary so the on-disk schema lives in one place. Black-box:
/// plain JSON matching the CLI's on-disk schema, not a typed import from the
/// rocm-cli crates.
pub fn write_service_record(services_dir: &Path, model: &str, port: u16) {
    write_service_record_with(services_dir, model, port, ServiceRecordOptions::default());
}

/// Like [`write_service_record`], but with caller-specified lifecycle fields.
///
/// See [`ServiceRecordOptions`] -- e.g. a "still loading" status/startup_phase,
/// or `supervisor_pid`/`engine_pid` pointed at a live process so the CLI's
/// liveness overlay doesn't mark the planted record dead.
pub fn write_service_record_with(
    services_dir: &Path,
    model: &str,
    port: u16,
    options: ServiceRecordOptions,
) {
    std::fs::create_dir_all(services_dir).expect("failed to create services dir");

    // The CLI only extracts host:port from `endpoint_url` and appends
    // `/v1/models` for its readiness probe, which the mock serves.
    let record = json!({
        "service_id": "e2e-mock",
        "engine": "vllm",
        "model_ref": model,
        "canonical_model_id": model,
        "host": "127.0.0.1",
        "port": port,
        "endpoint_url": http_server::loopback_url(port)
            .join("v1")
            .expect("v1 is a valid relative path")
            .to_string(),
        "mode": "managed",
        "status": options.status,
        "startup_phase": options.startup_phase,
        "supervisor_pid": options.supervisor_pid,
        "engine_pid": options.engine_pid,
        "runtime_id": null,
        "env_id": null,
        "device_policy": null,
        "gpu_indices": [],
        "engine_recipe_json": null,
        "restart_count": 0,
        "last_restart_unix_ms": null,
        "manifest_path": services_dir.join("e2e-mock.json"),
        "log_path": services_dir.join("e2e-mock.log"),
        "engine_state_path": services_dir.join("e2e-mock.state.json"),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        services_dir.join("e2e-mock.json"),
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
}

async fn handle_models(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_name, "object": "model"}]
    }))
}

async fn handle_metrics(State(state): State<ServerState>) -> String {
    state
        .metrics
        .as_ref()
        .expect("metrics route requires metrics state")
        .scrape()
}

async fn handle_scripted_metrics(State(state): State<ServerState>) -> (StatusCode, String) {
    match state
        .scripted_metrics
        .as_ref()
        .expect("scripted_metrics route requires scripted_metrics state")
        .scrape()
    {
        Some(body) => (StatusCode::OK, body),
        None => (StatusCode::SERVICE_UNAVAILABLE, String::new()),
    }
}

async fn handle_chat(
    State(state): State<ServerState>,
    uri: axum::http::Uri,
    Json(body): Json<Value>,
) -> Json<Value> {
    // Record which of the two chat routes the client actually used, so a
    // scenario can distinguish the versioned path from the unversioned one.
    state
        .chat_paths
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(uri.path().to_string());
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_string();

    // Record the request body so scenarios can assert on exactly what the CLI
    // sent (see `MockServer::last_chat_request` / `wait_for_chat_request`) —
    // the canned reply below never varies with the prompt, so without this a
    // corrupted or missing request would pass unnoticed.
    *state
        .last_chat_request
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(body);

    // Tests assert on the request contract, not the reply text, so the default
    // is a fixed stub. Demos (`rocm-demo-env`) set `ROCM_MOCK_CHAT_REPLY` to a
    // realistic answer so the recorded screencast reads naturally.
    let content = std::env::var("ROCM_MOCK_CHAT_REPLY")
        .unwrap_or_else(|_| "This is a mock response for testing.".to_string());

    Json(json!({
        "id": "mock-completion-1",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": model,
        "system_fingerprint": null,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }))
}
