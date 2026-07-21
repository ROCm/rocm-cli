// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::net::TcpListener;

/// How often [`MockServer::wait_for_chat_request`] re-checks the captured
/// request while waiting for the client to POST.
const CHAT_REQUEST_POLL_INTERVAL: Duration = Duration::from_millis(20);

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
    /// The most recently received `/v1/chat/completions` request body, shared
    /// with the `MockServer` handle so scenarios can assert on exactly what the
    /// CLI sent — not just on the (fixed) canned reply, which would silently
    /// mask a corrupted or missing prompt. `None` until a chat request arrives.
    last_chat_request: Arc<Mutex<Option<Value>>>,
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

pub struct MockServer {
    addr: SocketAddr,
    shutdown: tokio::sync::oneshot::Sender<()>,
    /// Shared with the running server's `ServerState`; see the field doc there.
    last_chat_request: Arc<Mutex<Option<Value>>>,
}

impl std::fmt::Debug for MockServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockServer")
            .field("addr", &self.addr)
            .finish_non_exhaustive()
    }
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

    async fn spawn(model_name: &str, with_metrics: bool) -> Self {
        let last_chat_request = Arc::new(Mutex::new(None));
        let state = ServerState {
            model_name: model_name.to_string(),
            metrics: with_metrics.then(|| Arc::new(MetricsCounter::new())),
            last_chat_request: Arc::clone(&last_chat_request),
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

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    rx.await.ok();
                })
                .await
                .ok();
        });

        Self {
            addr,
            shutdown: tx,
            last_chat_request,
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    pub const fn port(&self) -> u16 {
        self.addr.port()
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

    pub fn stop(self) {
        self.shutdown.send(()).ok();
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
        "endpoint_url": format!("http://127.0.0.1:{port}/v1"),
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

async fn handle_chat(State(state): State<ServerState>, Json(body): Json<Value>) -> Json<Value> {
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
