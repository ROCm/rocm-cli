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
    /// Every `/v1/chat/completions` request body, in arrival order. Unlike
    /// `last_chat_request` this keeps the whole history, so a scenario can count
    /// how many completions the CLI asked for and read each prompt — the only
    /// way to prove a request was planned once rather than twice, when the two
    /// calls would otherwise leave nothing but the final body behind.
    completion_requests: Arc<Mutex<Vec<Value>>>,
    /// `Some` only when started via [`MockServer::start_with_chat_script`]; the
    /// replies to hand out, in request order. `None` keeps the fixed stub reply.
    chat_script: Option<Arc<Vec<Value>>>,
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
    /// Shared with the running server's `ServerState`; see the field doc there.
    completion_requests: Arc<Mutex<Vec<Value>>>,
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
        let completion_requests = Arc::new(Mutex::new(Vec::new()));
        let scripted = Arc::new(ScriptedMetrics::new());
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: None,
            scripted_metrics: Some(Arc::clone(&scripted)),
            last_chat_request: Arc::clone(&last_chat_request),
            chat_paths: Arc::clone(&chat_paths),
            completion_requests: Arc::clone(&completion_requests),
            chat_script: None,
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
            completion_requests,
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
        let completion_requests = Arc::new(Mutex::new(Vec::new()));
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: with_metrics.then(|| Arc::new(MetricsCounter::new())),
            scripted_metrics: None,
            last_chat_request: Arc::clone(&last_chat_request),
            chat_paths: Arc::clone(&chat_paths),
            completion_requests: Arc::clone(&completion_requests),
            chat_script: None,
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
            completion_requests,
        }
    }

    /// Like [`Self::start`], but chat completions are answered from `script`
    /// instead of the fixed stub: the Nth `/v1/chat/completions` request gets
    /// `script[N]` as its complete response body (build entries with
    /// [`chat_text_reply`] / [`chat_tool_call_reply`]).
    ///
    /// For scenarios whose claim is about the *sequence* of completions a
    /// command makes — "asked exactly once", "the plan shown is the one the
    /// provider gave" — and which therefore need each call to answer
    /// differently, so a repeated call is visible in the output rather than
    /// returning the same bytes. A request past the end of the script is answered
    /// with HTTP 500 (and still recorded), so asking more often than scripted
    /// fails loudly instead of recycling an earlier reply. `GET /v1/models` does
    /// not consume an entry.
    pub async fn start_with_chat_script(model_name: &str, script: Vec<Value>) -> Self {
        let last_chat_request = Arc::new(Mutex::new(None));
        let chat_paths = Arc::new(Mutex::new(Vec::new()));
        let completion_requests = Arc::new(Mutex::new(Vec::new()));
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: None,
            scripted_metrics: None,
            last_chat_request: Arc::clone(&last_chat_request),
            chat_paths: Arc::clone(&chat_paths),
            completion_requests: Arc::clone(&completion_requests),
            chat_script: Some(Arc::new(script)),
        };
        let app = Router::new()
            .route("/v1/models", get(handle_models))
            .route("/models", get(handle_models))
            .route("/v1/chat/completions", post(handle_chat))
            .route("/chat/completions", post(handle_chat))
            .with_state(state);
        Self {
            server: http_server::spawn(app).await,
            last_chat_request,
            chat_paths,
            completion_requests,
            scripted: None,
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

    /// Every chat completion request body received so far, oldest first.
    ///
    /// Includes requests answered with an error, so a count taken after a
    /// command finishes is the number of completions it asked for. Recovers a
    /// poisoned lock for the same reason [`Self::last_chat_request`] does.
    #[must_use]
    pub fn chat_requests(&self) -> Vec<Value> {
        self.completion_requests
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
            completion_requests: Arc::new(Mutex::new(Vec::new())),
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

    /// Poll for a chat request that satisfies `matches`, so a scenario asserting
    /// on the user's turn cannot be answered by an earlier unrelated one.
    ///
    /// [`Self::wait_for_chat_request`] returns whichever body was recorded
    /// first, and the local-endpoint detection probe (`Say ok.`, `max_tokens: 2`)
    /// is itself a chat request that lands before the user has typed anything.
    /// A scenario with no intervening screen wait therefore sees the probe and
    /// reports the real turn as missing. Only `last_chat_request` is retained,
    /// so this narrows to the newest body rather than searching a history.
    pub async fn wait_for_chat_request_where(
        &self,
        timeout: Duration,
        matches: impl Fn(&Value) -> bool,
    ) -> Result<Value, String> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(body) = self.last_chat_request()
                && matches(&body)
            {
                return Ok(body);
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "timed out after {timeout:?} waiting for a matching chat request; \
                     the most recent one was: {:?}",
                    self.last_chat_request()
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
    /// The supervisor's start-time token (`rocm_core::process_start_ticks`),
    /// the same identity check a real launch path writes alongside
    /// `supervisor_pid`. `None` (the default) keeps the historical
    /// `"supervisor_start_ticks": null` — a record with no token at all, which
    /// `identity_state`'s `(None, _) => Matches` arm treats as a best-effort
    /// match on PID alone. Set it when a scenario's stop must exercise the
    /// stronger start-time-verified identity check instead.
    pub supervisor_start_ticks: Option<u64>,
    /// The runtime the service recorded when it was launched, as a runtime
    /// *key* (`release-tarball-gfx942`) — the field is named `runtime_id` on
    /// disk, but every launch path resolves its selector to an exact key before
    /// writing it, and the CLI compares it against a key. `None` (the default)
    /// keeps the historical `"runtime_id": null`, i.e. a record that predates
    /// runtime pinning and therefore says nothing about which runtime it runs
    /// on. Set it when a scenario needs the CLI to see a server still on a
    /// *different* runtime than the one being activated.
    pub runtime_id: Option<&'static str>,
    /// The bind address recorded for the service. `None` (the default) keeps
    /// the loopback `127.0.0.1` every other scenario relies on. Set a public
    /// address such as `0.0.0.0` when a scenario needs the CLI to treat the
    /// service as publicly bound: restarting one with no endpoint key is
    /// refused before anything is stopped, which makes a failed restart
    /// observable without an engine.
    pub host: Option<&'static str>,
    /// Whether the record already carries the "inference was verified" latch
    /// (`inference_verified_at_unix_ms`) a real launch writes once a completion
    /// first succeeds. `false` (the default) keeps the historical record with
    /// no latch, so the CLI's readiness check sends its own `Say ok.` completion
    /// probe to the service. Set it when the mock's chat replies are scripted
    /// ([`MockServer::start_with_chat_script`]): with the latch the CLI skips
    /// that probe, so the first scripted reply goes to the request the scenario
    /// is about instead of being consumed by readiness.
    pub inference_verified: bool,
}

impl Default for ServiceRecordOptions {
    fn default() -> Self {
        Self {
            status: "ready",
            startup_phase: None,
            supervisor_pid: 0,
            engine_pid: None,
            supervisor_start_ticks: None,
            runtime_id: None,
            host: None,
            inference_verified: false,
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
/// `supervisor_pid`/`engine_pid` pointed at a live process so the CLI's
/// liveness overlay doesn't mark the planted record dead, a `runtime_id`
/// naming the runtime the service was launched against, or a non-loopback
/// `host`.
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
        "host": options.host.unwrap_or("127.0.0.1"),
        "port": port,
        "endpoint_url": http_server::loopback_url(port)
            .join("v1")
            .expect("v1 is a valid relative path")
            .to_string(),
        "mode": "managed",
        "status": options.status,
        "startup_phase": options.startup_phase,
        "supervisor_pid": options.supervisor_pid,
        "supervisor_start_ticks": options.supervisor_start_ticks,
        "engine_pid": options.engine_pid,
        "runtime_id": options.runtime_id,
        "inference_verified_at_unix_ms": options.inference_verified.then_some(1_u64),
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

/// A complete `chat.completion` response body whose assistant message is plain
/// text. The shape [`MockServer`] answers with by default, and what a
/// [`MockServer::start_with_chat_script`] entry usually is.
#[must_use]
pub fn chat_text_reply(model: &str, content: &str) -> Value {
    json!({
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
    })
}

/// A `chat.completion` response body in which the assistant calls tool `name`.
///
/// An OpenAI-style `tool_calls` entry with `arguments` serialized as a JSON
/// string, and no text content.
#[must_use]
pub fn chat_tool_call_reply(model: &str, name: &str, arguments: &Value) -> Value {
    json!({
        "id": "mock-completion-1",
        "object": "chat.completion",
        "created": 1_700_000_000_u64,
        "model": model,
        "system_fingerprint": null,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments.to_string()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    })
}

async fn handle_chat(
    State(state): State<ServerState>,
    uri: axum::http::Uri,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
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
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(body.clone());

    // The history doubles as the script cursor: the index this request takes is
    // the number of requests recorded before it, read under the same lock that
    // appends it, so two concurrent requests can never be handed one entry.
    let index = {
        let mut history = state
            .completion_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        history.push(body);
        history.len() - 1
    };

    if let Some(script) = &state.chat_script {
        return script.get(index).map_or_else(
            || {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": format!(
                            "mock chat script exhausted: request {} arrived but only {} \
                             replies were scripted",
                            index + 1,
                            script.len()
                        )
                    })),
                )
            },
            |reply| (StatusCode::OK, Json(reply.clone())),
        );
    }

    // Tests assert on the request contract, not the reply text, so the default
    // is a fixed stub. Demos (`rocm-demo-env`) set `ROCM_MOCK_CHAT_REPLY` to a
    // realistic answer so the recorded screencast reads naturally.
    let content = std::env::var("ROCM_MOCK_CHAT_REPLY")
        .unwrap_or_else(|_| "This is a mock response for testing.".to_string());

    (StatusCode::OK, Json(chat_text_reply(&model, &content)))
}

/// Tests for the scripted chat replies and the request history: the harness
/// pieces the planner scenarios stand on, so a script that skipped, repeated or
/// recycled an entry — or a history that lost a request — would make a
/// "asked exactly once" assertion meaningless without failing anything itself.
#[cfg(test)]
mod chat_script_tests {
    use super::{
        MockServer, ServiceRecordOptions, chat_text_reply, chat_tool_call_reply,
        write_service_record_with,
    };
    use serde_json::{Value, json};
    use std::time::Duration;

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client")
    }

    /// POST one chat completion whose single user message is `prompt`; returns
    /// the HTTP status and the parsed body.
    async fn post_chat(mock: &MockServer, prompt: &str) -> (u16, Value) {
        let response = client()
            .post(format!("{}/chat/completions", mock.base_url()))
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": prompt}]
            }))
            .send()
            .await
            .expect("chat request failed");
        let status = response.status().as_u16();
        (status, response.json().await.expect("non-JSON chat reply"))
    }

    fn reply_text(body: &Value) -> &str {
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("reply carried no text")
    }

    fn user_prompt(request: &Value) -> &str {
        request["messages"][0]["content"]
            .as_str()
            .expect("request carried no user prompt")
    }

    #[tokio::test]
    async fn scripted_replies_come_back_in_request_order() {
        let mock = MockServer::start_with_chat_script(
            "m",
            vec![
                chat_text_reply("m", "first"),
                chat_text_reply("m", "second"),
                chat_text_reply("m", "third"),
            ],
        )
        .await;

        for (prompt, expected) in [("a", "first"), ("b", "second"), ("c", "third")] {
            let (status, body) = post_chat(&mock, prompt).await;
            assert_eq!(status, 200, "scripted request must succeed");
            assert_eq!(
                reply_text(&body),
                expected,
                "request `{prompt}` must get the reply scripted for its position"
            );
        }
    }

    #[tokio::test]
    async fn every_request_is_recorded_in_order_and_last_request_still_tracks() {
        let mock = MockServer::start_with_chat_script(
            "m",
            vec![chat_text_reply("m", "one"), chat_text_reply("m", "two")],
        )
        .await;

        post_chat(&mock, "alpha").await;
        post_chat(&mock, "beta").await;

        let requests = mock.chat_requests();
        assert_eq!(requests.len(), 2, "both requests must be kept");
        assert_eq!(user_prompt(&requests[0]), "alpha");
        assert_eq!(user_prompt(&requests[1]), "beta");
        let last = mock.last_chat_request().expect("no last request recorded");
        assert_eq!(
            user_prompt(&last),
            "beta",
            "`last_chat_request` must keep returning the newest body"
        );
    }

    #[tokio::test]
    async fn listing_models_does_not_consume_a_script_entry() {
        let mock =
            MockServer::start_with_chat_script("m", vec![chat_text_reply("m", "only")]).await;

        for _ in 0..3 {
            let listing = client()
                .get(format!("{}/models", mock.base_url()))
                .send()
                .await
                .expect("models request failed");
            assert_eq!(listing.status().as_u16(), 200);
        }
        assert!(
            mock.chat_requests().is_empty(),
            "a model listing is not a chat completion and must not be recorded as one"
        );

        let (status, body) = post_chat(&mock, "x").await;
        assert_eq!(status, 200);
        assert_eq!(
            reply_text(&body),
            "only",
            "the first completion must still get the first scripted reply"
        );
    }

    #[tokio::test]
    async fn a_request_past_the_end_of_the_script_fails_and_is_still_recorded() {
        let mock =
            MockServer::start_with_chat_script("m", vec![chat_text_reply("m", "only")]).await;

        let (status, _) = post_chat(&mock, "one").await;
        assert_eq!(status, 200);
        let (status, body) = post_chat(&mock, "two").await;
        assert_eq!(status, 500, "an unscripted request must not get a reply");
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|error| error.contains("exhausted")),
            "the failure must say the script ran out, got {body}"
        );
        let (status, _) = post_chat(&mock, "three").await;
        assert_eq!(status, 500, "the script must not wrap around");

        let requests = mock.chat_requests();
        assert_eq!(
            requests.len(),
            3,
            "requests past the end must still be counted, so over-asking is visible"
        );
        assert_eq!(user_prompt(&requests[2]), "three");
    }

    #[tokio::test]
    async fn a_tool_call_reply_carries_the_call_in_the_openai_shape() {
        let mock = MockServer::start_with_chat_script(
            "m",
            vec![chat_tool_call_reply(
                "m",
                "natural_language_plan",
                &json!({"request": "serve a model"}),
            )],
        )
        .await;

        let (_, body) = post_chat(&mock, "x").await;
        let call = &body["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "natural_language_plan");
        let arguments: Value =
            serde_json::from_str(call["arguments"].as_str().expect("arguments not a string"))
                .expect("arguments not JSON");
        assert_eq!(arguments["request"], "serve a model");
    }

    #[tokio::test]
    async fn without_a_script_every_request_gets_the_same_stub_and_is_recorded() {
        let mock = MockServer::start("m").await;

        let (_, first) = post_chat(&mock, "a").await;
        let (_, second) = post_chat(&mock, "b").await;

        assert_eq!(reply_text(&first), reply_text(&second));
        assert_eq!(mock.chat_requests().len(), 2);
    }

    #[test]
    fn the_inference_latch_is_written_only_when_asked_for() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let latch = |options: ServiceRecordOptions| {
            write_service_record_with(dir.path(), "m", 1, options);
            let record: Value = serde_json::from_slice(
                &std::fs::read(dir.path().join("e2e-mock.json")).expect("record not written"),
            )
            .expect("record not JSON");
            record["inference_verified_at_unix_ms"].clone()
        };

        assert_eq!(
            latch(ServiceRecordOptions::default()),
            Value::Null,
            "the default record must stay unlatched so existing scenarios keep probing"
        );
        assert_eq!(
            latch(ServiceRecordOptions {
                inference_verified: true,
                ..ServiceRecordOptions::default()
            }),
            json!(1),
            "an inference-verified record must carry a non-null latch"
        );
    }
}
