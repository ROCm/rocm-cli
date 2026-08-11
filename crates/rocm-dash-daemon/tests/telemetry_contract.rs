// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Deterministic lower-level contract scenarios for EAI-7960 telemetry correctness.
//!
//! These tests drive the real daemon runner through a scriptable local HTTP
//! server (no PTY, no TUI) and assert on raw [`Snapshot`] broadcast events.
//! They complement the principal black-box regression in
//! `tests/e2e-cucumber/features/dash.feature` (Scenario 8) by covering each
//! distinct state transition at the broadcast seam rather than the rendered screen.
//!
//! Coverage matrix:
//!
//! | #  | Scenario                                          | Contract           | Current     | State |
//! |----|---------------------------------------------------|--------------------|-------------|-------|
//! | 1  | Frozen counter → zero rate                        | `gen_tps ≈ 0.0`    | `≈ 0.0`     | GREEN |
//! | 2  | Counter reset → invalidation + re-baseline        | None then positive | same        | GREEN |
//! | 3  | Service removal → `InstanceGone` event            | fired next disc    | same        | GREEN |
//! | 4  | Single failure → gen_tps held for validity window | Some(…) held       | None immed. | RED   |
//! | 5  | Omitted counter → None (not zero, not panic)      | None, baseline kept| None, kept  | GREEN |
//! | 6  | Omitted → success path preserves baseline (≠ Fail)| immediate recovery | same        | GREEN |
//! | 7  | Malformed payload → None (not zero, not panic)    | None               | None        | GREEN |
//! | 8  | RunningIdle → gen_tps present + running_reqs = 0  | Some(+), req=0     | same        | GREEN |
//! | 9  | Expiry boundary: held inside window, gone after   | B1 held, B2 none   | B1 RED      | RED   |
//!
//! Scenarios 1–3, 5–8 verify already-correct or distinguishably-observable
//! behaviour.  Scenarios 4 and 9 reproduce the EAI-7960 root cause.
//!
//! Timing: `tick_override = 200 ms`, `discovery_tick = 200 ms`,
//! `instance_tick = 400 ms` (scrape every 2nd base tick).
//! Expiry window boundary: `clamp(3 × 400ms, 6 s, 30 s) = 6 s`.
//!
//! ## Frozen observation schema (proposed — do not implement in production during this task)
//!
//! The controller-reviewed additive schema for EAI-7960 observation metadata.
//! Production types (`rocm-dash-core`) are unchanged; this block is the
//! candidate contract for the implementer to reference.
//!
//! ```rust,ignore
//! // Existing value carrier — unchanged, remains the sole numeric source.
//! // Instance.gen_tps: Option<f64>
//!
//! // New additive field on Instance (or on InstanceMetrics if that is the
//! // structural container).  Absent for legacy snapshots.
//! #[serde(default, skip_serializing_if = "Option::is_none")]
//! pub gen_tps_observation: Option<ObservationMetadata>,
//!
//! pub struct ObservationMetadata {
//!     /// Wall-clock time of the last successful Prometheus scrape for this
//!     /// instance's gen_tps reading (UTC, RFC 3339 on the wire via serde).
//!     pub observed_at: DateTime<Utc>,
//!     /// Whether the value in `gen_tps` is freshly computed this tick or
//!     /// retained from the last successful observation.
//!     pub freshness: ObservationFreshness,
//! }
//!
//! // serde rename_all = "snake_case"
//! pub enum ObservationFreshness {
//!     Fresh,  // serialises as "fresh"
//!     Held,   // serialises as "held"
//! }
//! ```
//!
//! **Invariants:**
//! - `Instance.gen_tps` is the **only** numeric value carrier; no parallel
//!   `held_gen_tps`, `held_ttft_ms`, or `held_tpot_ms` fields.
//! - Tokens-per-watt derives from `gen_tps` and uses the same freshness
//!   metadata; it has no separate metadata source.
//! - Legacy snapshots: absent `gen_tps_observation` ⇒ `None / unknown`; never
//!   fabricated as `Fresh`.  Legacy `gen_tps` may render numerically without
//!   any freshness claim.
//!
//! **Timing / invalidation rules (fixed):**
//! - Internal counter rate window: `clamp(5 × instance_tick, 10 s, 30 s)`.
//! - Observation validity after last successful scrape:
//!   `clamp(3 × instance_tick, 6 s, 30 s)`.
//! - Successful scrape: `gen_tps` is recomputed (Fresh), validity clock resets.
//! - Missing / unparseable counter (HTTP 200, field absent): gen_tps holds,
//!   validity clock NOT reset, `prev_gen_tokens` preserved.
//! - Unchanged valid counter after full rate window: `gen_tps = Some(0.0)`
//!   (Fresh — idle is not unknown).
//! - Scrape failure (HTTP non-200): gen_tps holds (Held) until validity expires,
//!   then `None`.  `prev_gen_tokens` cleared on failure so recovery re-baselines.
//! - Counter reset (`cur < prev_val`): immediate `None`, new baseline set.
//! - Identity change (host:port reassigned): treated as reset.
//! - Service removal / non-serving terminal state: immediate `None`, no hold.
//! - Daemon marks each assembled snapshot `Fresh` only when newly observed this
//!   tick; otherwise `Held` until validity expires, then `gen_tps = None`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rocm_dash_core::protocol::Event;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio::time::timeout;

use rocm_dash_daemon::bench_ring::BenchRing;
use rocm_dash_daemon::runner::{self, RunnerOptions};
use rocm_dash_daemon::snapshot_ring::SnapshotRing;

// ── Timing constants ────────────────────────────────────────────────────────

const TICK: Duration = Duration::from_millis(200);
const INSTANCE_TICK: Duration = Duration::from_millis(400);
const SCENARIO_DEADLINE: Duration = Duration::from_secs(15);
const SVC_ID: &str = "contract-svc";

/// Observation validity window derived from the contract formula.
/// `clamp(3 × INSTANCE_TICK, 6 s, 30 s)` — with `INSTANCE_TICK = 400 ms`
/// this evaluates to `max(1.2 s, 6 s) = 6 s`, matching the spec lower bound.
fn observation_validity_window() -> Duration {
    (3 * INSTANCE_TICK).clamp(Duration::from_secs(6), Duration::from_secs(30))
}

/// Drain all snapshot events already buffered in the broadcast receiver so
/// that subsequent `wait_for_snapshot` calls return only snapshots assembled
/// **after** the next mode switch. Must be called immediately before setting
/// `MockMode::Failure` to avoid a stale-snapshot false pass.
fn drain_snapshots(rx: &mut broadcast::Receiver<Event>) {
    while rx.try_recv().is_ok() {}
}

/// Poll the mock's `failure_count` until it reaches `>= expected_min`, bounded
/// by `SCENARIO_DEADLINE`. Guarantees that at least `expected_min` HTTP 503
/// responses were actually served by the mock before the caller reads the next
/// broadcast snapshot — eliminating the sleep-based race in M4.
async fn await_failure_served(failure_count: &AtomicU64, expected_min: u64) {
    let deadline = tokio::time::Instant::now() + SCENARIO_DEADLINE;
    loop {
        if failure_count.load(Ordering::Relaxed) >= expected_min {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "mock failure was never served within {SCENARIO_DEADLINE:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ── Scriptable mock HTTP server ─────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum MockMode {
    /// Monotonically-increasing counter; internal tick counter advances.
    Growing,
    /// Counter frozen at the given cumulative value.
    Frozen(u64),
    /// Counter reset to a value *below* any reasonable Growing accumulation.
    Reset(u64),
    /// Endpoint returns HTTP 503.
    Failure,
    /// HTTP 200 with a valid body that omits `vllm:generation_tokens_total`.
    /// The runner's parser returns `sample.gen_tokens_total = None` (success
    /// path); `prev_gen_tokens` is preserved unlike `Failure`.
    Omitted,
    /// HTTP 200 with a structurally invalid body; all parsed fields come back
    /// `None` — same runner treatment as `Omitted` but triggered by parse error.
    Malformed,
    /// Growing counter but `running_reqs = 0` — idle engine, reqs shown as 0.
    RunningIdle,
}

fn prom_body(gen_tokens_total: u64, tick: u64) -> String {
    let ttft_sum_s = tick as f64 * 0.050;
    let tpot_sum_s = tick as f64 * 20.0 * 0.020;
    format!(
        "\
# HELP vllm:num_requests_running running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} 1
# HELP vllm:num_requests_waiting waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model=\"mock\"}} 0
# HELP vllm:gpu_cache_usage_perc kv.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{{model=\"mock\"}} 0.25
# HELP vllm:generation_tokens_total gen_tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {gen_tokens_total}
# HELP vllm:time_to_first_token_seconds ttft.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}
# HELP vllm:time_per_output_token_seconds tpot.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {gen_tokens_total}
"
    )
}

fn http_ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Like [`prom_body`] but omits the `vllm:generation_tokens_total` line.
/// The runner's parser returns `sample.gen_tokens_total = None` and takes the
/// success path (HTTP 200); `prev_gen_tokens` is **not** removed.
fn prom_body_omitted(tick: u64) -> String {
    let ttft_sum_s = tick as f64 * 0.050;
    let tpot_sum_s = tick as f64 * 20.0 * 0.020;
    format!(
        "\
# HELP vllm:num_requests_running running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} 1
# HELP vllm:num_requests_waiting waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model=\"mock\"}} 0
# HELP vllm:time_to_first_token_seconds ttft.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}
# HELP vllm:time_per_output_token_seconds tpot.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {tick}
"
    )
}

/// Invalid Prometheus exposition text; causes the parser to return all-None
/// sample fields (same runner treatment as `Omitted`, but triggered by a parse
/// error rather than a missing key).
const MALFORMED_BODY: &str = "# malformed: not valid prometheus exposition format\n!!!invalid!!!\n";

const HTTP_503: &str =
    "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

/// Spawn a persistent looping HTTP server on a random port.
/// Returns the bound port (ready before the future resolves).
async fn spawn_mock_server(
    mode: Arc<Mutex<MockMode>>,
    ticks: Arc<AtomicU64>,
    failure_count: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let mode = Arc::clone(&mode);
            let ticks = Arc::clone(&ticks);
            let failure_count = Arc::clone(&failure_count);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let current = *mode
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let response = match current {
                    MockMode::Growing => {
                        let tick = ticks.fetch_add(1, Ordering::Relaxed) + 1;
                        http_ok(&prom_body(tick * 20, tick))
                    }
                    MockMode::RunningIdle => {
                        // Counter grows identically to Growing but running_reqs=0.
                        let tick = ticks.fetch_add(1, Ordering::Relaxed) + 1;
                        let gen_tokens_total = tick * 20;
                        let ttft_sum_s = tick as f64 * 0.050;
                        let tpot_sum_s = tick as f64 * 20.0 * 0.020;
                        let body = format!(
                            "\
# HELP vllm:num_requests_running running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model=\"mock\"}} 0\n\
# HELP vllm:generation_tokens_total gen.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model=\"mock\"}} {gen_tokens_total}\n\
# HELP vllm:time_to_first_token_seconds ttft.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{{model=\"mock\"}} {ttft_sum_s}
vllm:time_to_first_token_seconds_count{{model=\"mock\"}} {tick}\n\
# HELP vllm:time_per_output_token_seconds tpot.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_sum{{model=\"mock\"}} {tpot_sum_s}
vllm:time_per_output_token_seconds_count{{model=\"mock\"}} {gen_tokens_total}\n"
                        );
                        http_ok(&body)
                    }
                    MockMode::Frozen(n) => http_ok(&prom_body(n, n.max(1))),
                    MockMode::Reset(n) => http_ok(&prom_body(n, 1)),
                    MockMode::Failure => {
                        failure_count.fetch_add(1, Ordering::Relaxed);
                        HTTP_503.to_string()
                    }
                    MockMode::Omitted => {
                        let tick = ticks.fetch_add(1, Ordering::Relaxed) + 1;
                        http_ok(&prom_body_omitted(tick))
                    }
                    MockMode::Malformed => http_ok(MALFORMED_BODY),
                };
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    port
}

fn write_service_record(services_dir: &std::path::Path, port: u16) {
    std::fs::create_dir_all(services_dir).unwrap();
    let json = format!(
        r#"{{"service_id":"{SVC_ID}","engine":"vllm","model_ref":"ContractModel/1B","canonical_model_id":"ContractModel/1B","host":"127.0.0.1","port":{port},"status":"ready","created_at_unix_ms":0}}"#
    );
    std::fs::write(services_dir.join(format!("{SVC_ID}.json")), json).unwrap();
}

fn remove_service_record(services_dir: &std::path::Path) {
    let _ = std::fs::remove_file(services_dir.join(format!("{SVC_ID}.json")));
}

fn fast_opts(services_dir: std::path::PathBuf) -> RunnerOptions {
    RunnerOptions {
        services_dir: Some(services_dir),
        discovery_tick: TICK,
        instance_tick: INSTANCE_TICK,
        gpu_tick: TICK,
        disable_vllm_metrics: false,
        enable_docker: false,
        enable_lemonade: false,
        ..RunnerOptions::default()
    }
}

async fn spawn_runner(
    tx: broadcast::Sender<Event>,
    opts: RunnerOptions,
) -> tokio::task::JoinHandle<()> {
    let ring = Arc::new(Mutex::new(SnapshotRing::new(16)));
    let bench_ring = Arc::new(Mutex::new(BenchRing::new(4)));
    tokio::spawn(runner::run_loop(
        Some(TICK),
        tx,
        ring,
        bench_ring,
        None,
        opts,
    ))
}

/// Drain broadcast events until `pred(&snap)` returns `Some(T)`, or deadline expires.
async fn wait_for_snapshot<T>(
    rx: &mut broadcast::Receiver<Event>,
    pred: impl Fn(&rocm_dash_core::metrics::Snapshot) -> Option<T>,
) -> Result<T, String> {
    let deadline = tokio::time::Instant::now() + SCENARIO_DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!("timed out after {SCENARIO_DEADLINE:?}"));
        }
        match timeout(remaining, rx.recv()).await {
            Ok(Ok(Event::Snapshot(snap))) => {
                if let Some(v) = pred(&snap) {
                    return Ok(v);
                }
            }
            Ok(Ok(_)) | Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(broadcast::error::RecvError::Closed)) => {
                return Err("broadcast channel closed".into());
            }
            Err(_) => return Err(format!("timed out after {SCENARIO_DEADLINE:?}")),
        }
    }
}

/// Extract the `contract-svc` instance's `gen_tps` from a snapshot.
fn svc_gen_tps(snap: &rocm_dash_core::metrics::Snapshot) -> Option<Option<f64>> {
    snap.instances
        .iter()
        .find(|i| i.container_id == SVC_ID)
        .map(|i| i.gen_tps)
}

// ── Scenario 1: Frozen counter → zero rate ─────────────────────────────────

/// Contract: a frozen cumulative counter produces `gen_tps = Some(0.0)` —
/// distinct from `None` ("no data").  An idle engine is not the same as an
/// unknown engine.
///
/// GREEN: `gen_tps_from_delta` computes `0 / dt = 0.0` (runner.rs:704).
#[tokio::test]
async fn zero_gen_tps_after_frozen_counter() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    // Frozen at 1 000 tokens; first scrape sets baseline, second computes delta=0.
    let mode = Arc::new(Mutex::new(MockMode::Frozen(1_000)));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    let gen_tps = wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v < 0.1)
    })
    .await
    .unwrap_or_else(|e| panic!("frozen counter never produced zero gen_tps: {e}"));

    stop.store(true, Ordering::Relaxed);
    assert!(
        (0.0..0.1).contains(&gen_tps),
        "expected ~0.0 tok/s, got {gen_tps}"
    );
}

// ── Scenario 2: Counter reset → invalidation + re-baseline ─────────────────

/// Contract: a counter that drops below its previous reading (`cur < prev_val`)
/// immediately yields `gen_tps = None` for that tick, and the new lower value
/// becomes the baseline so recovery to a positive rate is possible.
///
/// GREEN: `gen_tps_from_delta` has an explicit guard `cur < prev_val → None`
/// (runner.rs:701), and re-inserts the new value as the baseline.
#[tokio::test]
async fn counter_reset_invalidates_baseline_immediately() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Growing));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Step 1: establish positive baseline.
    wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await
    .unwrap_or_else(|e| panic!("positive gen_tps never established: {e}"));

    // Step 2: reset counter to 5 — below the ≥40 tokens any Growing scrape produces.
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Reset(5);

    // Step 3: expect gen_tps = None immediately after the reset scrape.
    wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| if g.is_none() { Some(()) } else { None })
    })
    .await
    .unwrap_or_else(|e| panic!("gen_tps never cleared after counter reset: {e}"));

    // Step 4: resume Growing — new baseline of 5 means counter > 5 = positive rate.
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Growing;

    // Step 5: gen_tps recovers.
    wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await
    .unwrap_or_else(|e| panic!("gen_tps never recovered after re-baseline: {e}"));

    stop.store(true, Ordering::Relaxed);
}

// ── Scenario 3: Service removal → InstanceGone event ───────────────────────

/// Contract: removing the managed-service JSON record fires an `InstanceGone`
/// event on the next discovery tick and the instance is absent from subsequent
/// Snapshots.
///
/// GREEN: runner.rs:362-377 computes `known_services.difference(&disc.seen)`
/// and fires `InstanceGone` for each absent id.
#[tokio::test]
async fn service_removal_fires_instance_gone() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Growing));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    let sdir = services_dir.clone();
    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Wait for the instance to appear.
    wait_for_snapshot(&mut rx, |snap| {
        snap.instances
            .iter()
            .find(|i| i.container_id == SVC_ID)
            .map(|_| ())
    })
    .await
    .unwrap_or_else(|e| panic!("instance never appeared: {e}"));

    // Remove the record.
    remove_service_record(&sdir);

    // Wait for the InstanceGone event.
    let deadline = tokio::time::Instant::now() + SCENARIO_DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "InstanceGone event never fired within {SCENARIO_DEADLINE:?}"
        );
        match timeout(remaining, rx.recv()).await {
            Ok(Ok(Event::InstanceGone { container_id })) if container_id == SVC_ID => break,
            Ok(Ok(_)) | Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(broadcast::error::RecvError::Closed)) => panic!("broadcast closed"),
            Err(_) => panic!("timed out waiting for InstanceGone"),
        }
    }

    // Confirm the instance is absent from subsequent Snapshots.
    wait_for_snapshot(&mut rx, |snap| {
        if snap.instances.iter().any(|i| i.container_id == SVC_ID) {
            None
        } else {
            Some(())
        }
    })
    .await
    .unwrap_or_else(|e| panic!("instance still present after InstanceGone: {e}"));

    stop.store(true, Ordering::Relaxed);
}

// ── Scenario 4: Single failure → gen_tps held for validity window (RED) ────

/// Contract: after a single failed `/metrics` scrape, `gen_tps` must remain
/// held at its last positive value for the validity window
/// `clamp(3 × instance_tick, 6 s, 30 s)` before clearing.
///
/// **Current behaviour — RED:** `runner.rs:462-477` clears `gen_tps` to `None`
/// and removes `prev_gen_tokens` on the very tick of the failure.  The first
/// Snapshot after the failure already shows `gen_tps = None`.
///
/// This reproduces the same root cause as cucumber Scenario 8 at the daemon
/// broadcast seam (no PTY / TUI layer).
#[tokio::test]
async fn gen_tps_held_for_validity_window_after_single_failure() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Growing));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Step 1: establish positive gen_tps.
    let baseline = wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await
    .unwrap_or_else(|e| panic!("positive gen_tps never established: {e}"));
    assert!(baseline > 0.0);

    // Step 2: drain buffered pre-failure snapshots, then inject failure.
    // Draining ensures wait_for_snapshot below only sees post-failure events.
    drain_snapshots(&mut rx);
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Failure;

    // Step 3: poll until the mock confirms ≥ 1 HTTP 503 was served — guarantees
    // the runner has processed the failure before we read the next snapshot.
    await_failure_served(&failure_count, 1).await;

    let post_failure_gen_tps = wait_for_snapshot(&mut rx, svc_gen_tps)
        .await
        .unwrap_or_else(|e| panic!("no snapshot arrived after failure: {e}"));

    stop.store(true, Ordering::Relaxed);

    // CONTRACT: the held value must still be present.
    // CURRENT: None — immediate clear.  This assertion FAILS (RED).
    assert!(
        post_failure_gen_tps.is_some(),
        "EAI-7960 REGRESSION (daemon broadcast seam): gen_tps was cleared to None \
         immediately after the first failed /metrics scrape.\n\
         Contract: hold for clamp(3 × {INSTANCE_TICK:?}, 6 s, 30 s).\n\
         Root cause: runner.rs:462-477 clears gen_tps on the same tick as the failure.\n\
         Baseline was {baseline:.2} tok/s; post-failure was None.\n\
         This test must FAIL (RED) until the EAI-7960 fix is applied."
    );
}

// ── Scenario 5: Omitted counter → None (not zero, not panic) ───────────────

/// Contract: when the `/metrics` endpoint returns HTTP 200 but the body omits
/// `vllm:generation_tokens_total`, `gen_tps` is `None` — not `Some(0.0)`.
/// `prev_gen_tokens` must NOT be removed (success path, not failure path).
///
/// GREEN: `gen_tps_from_delta` is only reached when `gen_tokens_total.is_some()`;
/// an absent counter falls straight to `None` without touching `prev_gen_tokens`.
#[tokio::test]
async fn omitted_counter_yields_none_not_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Omitted));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Wait for a snapshot where the service is visible and gen_tps is stable.
    // Two consecutive omitted scrapes are sufficient.
    let mut got_none = false;
    let mut got_zero = false;
    let deadline = tokio::time::Instant::now() + SCENARIO_DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        if let Ok(Ok(Event::Snapshot(snap))) = timeout(remaining, rx.recv()).await {
            if let Some(gen_tps_opt) = svc_gen_tps(&snap) {
                match gen_tps_opt {
                    None => got_none = true,
                    Some(v) if v < 0.01 => got_zero = true,
                    _ => {}
                }
                if got_none {
                    break; // contract satisfied
                }
            }
        }
    }

    stop.store(true, Ordering::Relaxed);

    assert!(
        got_none,
        "omitted counter never produced gen_tps = None; got_zero = {got_zero}"
    );
    assert!(
        !got_zero,
        "omitted counter produced gen_tps = Some(~0.0); \
         contract requires None (missing ≠ zero)"
    );
}

// ── Scenario 6: Omitted → success path preserves baseline (≠ Failure) ───────

/// Contract: after an omitted-counter scrape (HTTP 200, counter absent),
/// switching back to Growing gives an immediate positive `gen_tps` on the
/// very next scrape — because `prev_gen_tokens` was preserved on the success
/// path. This distinguishes `Omitted` from `Failure`, which removes the baseline.
///
/// GREEN: the success path in runner.rs only calls
/// `prev_gen_tokens.insert(id, (cur, now))` inside `and_then`, so an absent
/// counter leaves `prev_gen_tokens` untouched.
#[tokio::test]
async fn omitted_preserves_baseline_so_recovery_is_immediate() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Growing));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Step 1: establish positive baseline.
    wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await
    .unwrap_or_else(|e| panic!("positive gen_tps never established: {e}"));

    // Step 2: one omitted-counter scrape (success path, prev_gen_tokens preserved).
    // Drain buffered pre-step snapshots, record ticks baseline, then switch to Omitted.
    drain_snapshots(&mut rx);
    let omit_baseline = ticks.load(Ordering::Relaxed);
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Omitted;
    // Poll until the mock confirms ≥ 1 Omitted response was served (ticks advances in Omitted mode).
    loop {
        if ticks.load(Ordering::Relaxed) > omit_baseline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Step 3: recover immediately — Growing again with preserved prev_gen_tokens.
    // A single successful scrape with a counter > prev_val should yield positive rate.
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Growing;

    // Step 4: expect positive rate to come back within ONE instance_tick.
    // If baseline were erased (like Failure) this would take TWO ticks.
    let recovered = wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await;

    stop.store(true, Ordering::Relaxed);

    recovered.unwrap_or_else(|e| {
        panic!(
            "gen_tps did not recover after omitted→growing: {e}\n\
             If the baseline was erased (failure path taken), recovery requires \
             two ticks instead of one."
        )
    });
}

// ── Scenario 7: Malformed payload → None (not zero, not panic) ──────────────

/// Contract: an HTTP 200 response with a structurally invalid body produces
/// `gen_tps = None`, not `Some(0.0)`, and does not panic.
///
/// GREEN: the parser extracts only named metric lines; an unrecognised body
/// returns all-None sample fields, identical to Omitted at the runner level.
#[tokio::test]
async fn malformed_payload_yields_none_not_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Malformed));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    let mut saw_none = false;
    let mut saw_zero = false;
    let deadline = tokio::time::Instant::now() + SCENARIO_DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        if let Ok(Ok(Event::Snapshot(snap))) = timeout(remaining, rx.recv()).await {
            if let Some(gen_tps_opt) = svc_gen_tps(&snap) {
                match gen_tps_opt {
                    None => {
                        saw_none = true;
                        break;
                    }
                    Some(v) if v < 0.01 => saw_zero = true,
                    _ => {}
                }
            }
        }
    }

    stop.store(true, Ordering::Relaxed);

    assert!(
        saw_none,
        "malformed payload never produced gen_tps = None; saw_zero = {saw_zero}"
    );
    assert!(
        !saw_zero,
        "malformed payload produced gen_tps = Some(~0.0); \
         contract requires None (unparseable ≠ zero)"
    );
}

// ── Scenario 8: RunningIdle → gen_tps present + running_reqs = 0 ────────────

/// Contract: a vLLM instance whose counter grows but whose
/// `num_requests_running` is 0 (idle between bursts) shows a positive `gen_tps`
/// and `running_reqs = Some(0)`.  This is distinct from an `Omitted` payload
/// (counter present) and from a busy instance (running_reqs > 0).
///
/// GREEN: runner.rs reads both fields independently from the Prometheus scrape.
#[tokio::test]
async fn running_idle_yields_gen_tps_and_zero_running_reqs() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::RunningIdle));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Wait for a snapshot with positive gen_tps from this idle instance.
    let snap_result = wait_for_snapshot(&mut rx, |snap| {
        snap.instances
            .iter()
            .find(|i| i.container_id == SVC_ID)
            .and_then(|i| {
                // Both conditions must hold: counter-derived rate AND idle reqs.
                let has_gen_tps = i.gen_tps.map_or(false, |v| v > 0.0);
                let is_idle = i.running_reqs == Some(0);
                if has_gen_tps && is_idle {
                    Some((i.gen_tps.unwrap(), i.running_reqs.unwrap()))
                } else {
                    None
                }
            })
    })
    .await;

    stop.store(true, Ordering::Relaxed);

    let (gen_tps, running_reqs) = snap_result.unwrap_or_else(|e| {
        panic!(
            "RunningIdle never produced gen_tps > 0 with running_reqs = 0: {e}\n\
             Check that the mock body sets num_requests_running to 0 and \
             generation_tokens_total increments correctly."
        )
    });
    assert!(gen_tps > 0.0, "gen_tps must be positive: {gen_tps}");
    assert_eq!(running_reqs, 0, "running_reqs must be 0 for idle instance");
}

// ── Scenario 9: Expiry boundary — held inside window, gone after ─────────────

/// Pins both boundaries of the EAI-7960 validity-window contract at the daemon
/// broadcast seam.
///
/// **BOUNDARY 1 (held assertion — RED today):** immediately after the first
/// failed scrape, `gen_tps` must still be `Some(_)` (held at its last observed
/// value).  Current code clears it to `None` immediately — this assertion FAILS.
///
/// **BOUNDARY 2 (expired assertion — unreachable today):** after the full
/// validity window `clamp(3 × instance_tick, 6 s, 30 s)` = 6 s elapses, the
/// held value must be cleared and `gen_tps` must be `None`.
///
/// Both snapshots are captured before any assertion runs, so the test documents
/// both boundaries even though only boundary 1 is checked by the final
/// `assert!`. The boundary 2 assertion becomes GREEN once the fix is applied.
#[tokio::test]
async fn gen_tps_expiry_boundary_held_then_unavailable() {
    let tmp = tempfile::TempDir::new().unwrap();
    let services_dir = tmp.path().join("services");

    let mode = Arc::new(Mutex::new(MockMode::Growing));
    let ticks = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failure_count = Arc::new(AtomicU64::new(0));
    let port = spawn_mock_server(
        Arc::clone(&mode),
        Arc::clone(&ticks),
        Arc::clone(&failure_count),
        Arc::clone(&stop),
    )
    .await;

    write_service_record(&services_dir, port);
    let (tx, mut rx) = broadcast::channel::<Event>(64);
    let _runner = spawn_runner(tx, fast_opts(services_dir)).await;

    // Establish positive baseline.
    let baseline = wait_for_snapshot(&mut rx, |snap| {
        svc_gen_tps(snap).and_then(|g| g).filter(|&v| v > 0.0)
    })
    .await
    .unwrap_or_else(|e| panic!("positive gen_tps never established: {e}"));
    assert!(baseline > 0.0);

    // Inject sustained failure — keep the mode locked to Failure.
    // Drain buffered pre-failure snapshots so boundary-1 sees only post-failure events.
    drain_snapshots(&mut rx);
    *mode
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = MockMode::Failure;

    // ── BOUNDARY 1 ──────────────────────────────────────────────────────────
    // Poll until the mock confirms ≥ 1 HTTP 503 was served, then capture
    // the very first post-failure snapshot.  Contract: gen_tps must be Some(_).
    // Current: None (immediate clear). → RED assertion below.
    await_failure_served(&failure_count, 1).await;
    let boundary1_gen_tps = wait_for_snapshot(&mut rx, svc_gen_tps)
        .await
        .unwrap_or_else(|e| panic!("no snapshot for boundary-1 check: {e}"));

    // ── BOUNDARY 2 ──────────────────────────────────────────────────────────
    // Wait for the full validity window + two instance-tick buffer to elapse.
    // Contract: gen_tps must then be None (expired/unavailable).
    // NOTE: this sleep is necessary — the validity window is a real wall-clock
    // duration that cannot be shortened without touching production code.
    // After the sleep, drain the broadcast backlog: the channel has accumulated
    // held-gen_tps snapshots from the sleep period; we want the NEXT snapshot
    // (from after the drain) which must be from after the validity window.
    let validity_window = observation_validity_window(); // clamp(3×INSTANCE_TICK, 6s, 30s)
    tokio::time::sleep(validity_window + 2 * INSTANCE_TICK + Duration::from_millis(500)).await;
    // Discard held-period snapshots buffered during the sleep; the immediately
    // following wait_for_snapshot will read the first post-drain tick, which
    // is by construction past the validity deadline.
    drain_snapshots(&mut rx);
    let boundary2_gen_tps = wait_for_snapshot(&mut rx, svc_gen_tps)
        .await
        .unwrap_or_else(|e| panic!("no snapshot for boundary-2 check: {e}"));

    stop.store(true, Ordering::Relaxed);

    // Assert BOUNDARY 1 first; failure here prevents boundary-2 from running.
    assert!(
        boundary1_gen_tps.is_some(),
        "EAI-7960 BOUNDARY-1 FAILED (daemon broadcast seam): gen_tps cleared to None \
         immediately after first failure instead of being held.\n\
         Contract: hold for clamp(3 × {INSTANCE_TICK:?}, 6 s, 30 s).\n\
         Root cause: runner.rs clears gen_tps on the same tick as the failure.\n\
         Baseline was {baseline:.2} tok/s; post-failure was None.\n\
         This assertion must FAIL (RED) until the EAI-7960 fix is applied."
    );

    // Assert BOUNDARY 2 (reachable only after boundary-1 is fixed).
    assert!(
        boundary2_gen_tps.is_none(),
        "EAI-7960 BOUNDARY-2 FAILED: gen_tps is still Some({:?}) after the \
         validity window ({validity_window:?}) elapsed.\n\
         Contract: held value must expire to None after the window.",
        boundary2_gen_tps
    );
}
